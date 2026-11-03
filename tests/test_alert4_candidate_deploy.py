"""Fail-closed conversion boundaries for the physical-five alert4bg candidate."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts.host import deploy_alert4_candidate as d


def digest(value):
    return hashlib.sha256(value).hexdigest()


def samples(purpose, per_class):
    rows = []
    x = np.zeros((5 * per_class, 1, 64, 96), np.float32)
    for i in range(len(x)):
        x[i, 0, 0, 0] = i + (1000 if purpose == 'validation' else 0)
        rows.append(dict(recording_id=purpose + str(i), sampling_identity=purpose + str(i),
                         source_group=purpose + str(i), uploader_group=purpose + str(i),
                         parent_sha256=digest((purpose + str(i)).encode()),
                         pcm_sha256_16k=digest(('pcm' + purpose + str(i)).encode()),
                         purpose=purpose, target_index=i // per_class, derivation=None,
                         start_sample_16k=0, negative_subtype='ambient',
                         feature_sha256=digest(x[i].tobytes())))
    return x, rows


class CandidateDeployTests(unittest.TestCase):
    def test_five_logits_are_not_legacy_six_or_merged_probabilities(self):
        p = d.probabilities(np.log([[.27, .26, .4, .02, .05]]))
        np.testing.assert_allclose(p, [[.27, .26, .4, .02, .05]])
        for bad in (np.zeros((2, 6)), np.zeros((2, 8)), [[0, 0, 0, 0, np.nan]]):
            with self.assertRaises(ValueError):
                d.probabilities(bad)

    def test_calibration_rejects_validation_augments_duplicates_and_bad_hashes(self):
        x, rows = samples('train', 32)
        d.check_calibration(x, rows)
        for change in ({'purpose': 'validation'}, {'derivation': {'gain': 1}},
                       {'sampling_identity': 'train1'}, {'target_index': 4},
                       {'feature_sha256': '0' * 64}, {'parent_sha256': ''}):
            bad = copy.deepcopy(rows)
            bad[0].update(change)
            with self.assertRaises(ValueError):
                d.check_calibration(x, bad)
        for bad_x in (x[:159], x.astype(np.float64), x.reshape(160, 64, 96)):
            with self.assertRaises(ValueError):
                d.check_calibration(bad_x, rows)

    def test_validation_is_unaugmented_unique_source_and_hash_bound(self):
        x, rows = samples('validation', 2)
        y = np.repeat(np.arange(5), 2)
        d.check_validation(x, y, rows)
        for key, value in [('purpose', 'heldout'), ('derivation', {}),
                           ('recording_id', rows[1]['recording_id']), ('target_index', 3)]:
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                d.check_validation(x, y, bad)
        with self.assertRaises(ValueError):
            d.check_validation(x, y.astype(float), rows)

    def test_every_known_identity_is_disjoint_between_train_and_validation(self):
        _, train = samples('train', 2)
        _, val = samples('validation', 2)
        d.check_disjoint(train, val)
        for field in d.IDENTITY_FIELDS:
            bad = copy.deepcopy(val)
            bad[0][field] = train[0][field]
            with self.assertRaises(ValueError, msg=field):
                d.check_disjoint(train, bad)

    def test_provenance_cannot_relabel_glass_or_generic_dog_as_bark(self):
        _, rows = samples('train', 1)
        r = rows[3]
        source = dict(recording_id=r['recording_id'], source_group=r['source_group'],
                      uploader_group=r['uploader_group'], sampling_identity=r['sampling_identity'],
                      sha256=r['parent_sha256'], purpose='train', target_index=3,
                      target_id='dog_bark', negative_subtype='bark', source_id='fsd50k',
                      original_labels=['Bark', 'Dog'], class_contract=d.CONTRACT)
        r['negative_subtype'] = 'bark'
        d.check_source_binding([r], [source])
        for labels in (['Glass'], ['Dog'], ['Growling'], ['Doorbell']):
            bad = dict(source, original_labels=labels)
            with self.assertRaises(ValueError):
                d.check_source_binding([r], [bad])
        with self.assertRaises(ValueError):
            d.check_source_binding([dict(r, parent_sha256='0' * 64)], [source])

    def test_files_require_valid_pins_safe_relative_paths_and_complete_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'q').mkdir()
            (root / 'q' / 'a').write_bytes(b'ok')
            d.check_files(root, {'q/a': digest(b'ok')}, required={'q/a'})
            for pins in ({'q/a': 'bad'}, {'../a': digest(b'ok')},
                         {'q/../q/a': digest(b'ok')}, {'q\\a': digest(b'ok')},
                         {'C:/a': digest(b'ok')}, {}):
                with self.assertRaises(ValueError):
                    d.check_files(root, pins, required={'q/a'})
            (root / 'q' / 'a').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                d.check_files(root, {'q/a': digest(b'ok')})

    def test_weight_and_parity_checks_reject_wrong_output_shape_and_drift(self):
        weights = {k: np.zeros(v, np.float32) for k, v in d.SHAPES.items()}
        d.check_weights(weights)
        weights['fc8.weight'] = np.zeros((6, 2304), np.float32)
        with self.assertRaises(ValueError):
            d.check_weights(weights)
        reference = np.tile(np.arange(5, dtype=np.float32), (10, 1))
        self.assertEqual(d.parity(reference, reference)['top1_matches'], 10)
        for bad in (reference + .001, reference[:, :4], np.full_like(reference, np.nan)):
            with self.assertRaises(ValueError):
                d.parity(reference, bad)
        near_tie = np.zeros((1, 5), np.float32)
        changed_winner = near_tie.copy()
        changed_winner[0, 1] = 1e-6
        with self.assertRaises(ValueError):
            d.parity(near_tie, changed_winner)

    def test_fixed_graph_rejects_weight_only_or_wrong_output_count(self):
        fixed = '\n'.join('layer { name: "fix%d" type: "FixedNeuron" }' % i for i in range(6))
        fixed += '\nlayer { name: "conv1" type: "ConvolutionFixed" }'
        fixed += '\nlayer { name: "fc8" top: "fc8" type: "InnerProductFixed" inner_product_param { num_output: 5 } }'
        d.check_fixed_graph(fixed)
        for bad in (fixed.replace('FixedNeuron', 'ReLU'), fixed.replace('num_output: 5', 'num_output: 6')):
            with self.assertRaises(ValueError):
                d.check_fixed_graph(bad)

    def test_export_completion_rejects_legacy_contract_and_failed_folding(self):
        meta = dict(schema=2, contract='soundjie-alert4bg-v2',
                    status='verified_fp32_export_ready_for_separate_DNNDK_ARM_stage',
                    output_shape=[1, 5], output_tensor_name='fc8', checkpoint_reload_parity=True,
                    calibration_train_only=True, calibration_counts=[32] * 5,
                    folding=dict(max_absolute_error=0.00001, top1_identical=True),
                    files={name: '1' * 64 for name in d.EXPORT_FILES},
                    output_contract=dict(contract='soundjie-alert4bg-v2', physical_output_count=5,
                        logical_output_count=5, physical_labels=d.LABELS, display_labels=d.LABELS,
                        background_index=4, probability_merge_groups=[[i] for i in range(5)], output_tensor_name='fc8'))
        d.check_export(meta)
        for field, bad_value in [('contract', 'soundjie-siren5-v1'), ('output_shape', [1, 6]),
                                 ('calibration_counts', [32] * 6), ('checkpoint_reload_parity', False),
                                 ('status', 'training'), ('files', {}),
                                 ('folding', dict(max_absolute_error=.001, top1_identical=True)),
                                 ('folding', dict(max_absolute_error=0., top1_identical=False))]:
            with self.assertRaises(ValueError):
                d.check_export(dict(meta, **{field: bad_value}))

    def test_threshold_assessment_uses_fixed_logits_and_never_promotes_runner_up(self):
        from scripts.host.soundjie_alert4bg import apply_thresholds
        np.testing.assert_array_equal(apply_thresholds(np.array([[.1, .1, .4, .35, .05]]),
                                                      [.4, .4, .5, .3]), [4])
        _, rows = samples('validation', 2)
        y = np.repeat(np.arange(5), 2)
        p = np.full((10, 5), .025)
        p[np.arange(10), y] = .9
        p[8] = [.7, .1, .05, .05, .1]  # One background false siren.
        cfg = dict(grid_start=.4, grid_stop_exclusive=1., grid_step=.025,
                   per_class_background_fpr_limit=.02)
        result = d.assess_scores(y, rows, np.log(p), [.4] * 4, cfg)
        self.assertAlmostEqual(result['post_int8_thresholds'][0], .7)
        self.assertEqual(result['post_int8']['negative_false_alert_rate'], 0.)
        self.assertEqual(result['post_int8']['dog_recall'], 1.)
        self.assertFalse(result['independent_acceptance'])
        self.assertFalse(result['board_tested'])
        self.assertTrue(result['thresholds_fit_on_same_development_validation'])

    def test_create_only_writes_preserve_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'result.json'
            d.write(p, {'ready': True})
            with self.assertRaises(FileExistsError):
                d.write(p, {'ready': False})
            self.assertEqual(d.read(p), {'ready': True})


if __name__ == '__main__':
    unittest.main()
