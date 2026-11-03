"""Six-output conversion rejects legacy semantics and calibration leakage."""
import copy
import unittest
import numpy as np
from scripts.host import deploy_soundjie_alert5bg as d
from scripts.host.soundjie_alert5bg import output_contract


def samples(purpose, count=32):
    x=np.zeros((6*count,1,64,96),np.float32);rows=[]
    for i in range(len(x)):
        identity=purpose+str(i); x[i,0,0,0]=i
        rows.append(dict(purpose=purpose,target_index=i//count,derivation=None,
            recording_id=identity,source_group=identity,uploader_group=identity,sampling_identity=identity,
            parent_sha256=d.sha_bytes(identity.encode()),pcm_sha256_16k=d.sha_bytes(identity.encode()),
            feature_sha256=d.sha_bytes(x[i].tobytes()),negative_subtype='ordinary_speech' if i//count==4 else 'ambient'))
    return x,rows


class DeployAlert5bgTests(unittest.TestCase):
    def test_requires_physical_six_and_fc8_weights(self):
        d.probabilities(np.zeros((2,6)))
        with self.assertRaises(ValueError):d.probabilities(np.zeros((2,5)))
        weights={k:np.zeros(v,np.float32) for k,v in d.SHAPES.items()}; d.check_weights(weights)
        weights['fc8.weight']=np.zeros((5,2304),np.float32)
        with self.assertRaises(ValueError):d.check_weights(weights)

    def test_calibration_is_192_distinct_unaugmented_training_sources(self):
        x,rows=samples('train');d.check_calibration(x,rows)
        for change in ({'purpose':'validation'},{'derivation':{}},{'sampling_identity':rows[1]['sampling_identity']},{'target_index':5}):
            bad=copy.deepcopy(rows);bad[0].update(change)
            with self.assertRaises(ValueError):d.check_calibration(x,bad)

    def test_speech_is_positive_from_original_semantic_labels(self):
        _,rows=samples('train',1);r=rows[4]
        source=dict(r,sha256=r['parent_sha256'],source_id='librispeech_dev_clean',original_labels=['speech'],
                    target_id='speech',class_contract=d.CONTRACT,path='11-22-33.flac')
        d.check_source_binding([r],[source])
        for bad in (dict(source,target_index=5,target_id='background_unknown'),dict(source,original_labels=['crying'])):
            with self.assertRaises(ValueError):d.check_source_binding([r],[bad])

    def test_validation_and_calibration_are_disjoint(self):
        x,rows=samples('validation',2);d.check_validation(x,np.repeat(np.arange(6),2),rows)
        _,tr=samples('train');d.check_disjoint(tr,rows)
        for field in d.IDENTITY_FIELDS:
            bad=copy.deepcopy(rows);bad[0][field]=tr[0][field]
            with self.assertRaises(ValueError):d.check_disjoint(tr,bad)

    def test_folding_and_contract_completion_gate(self):
        meta=dict(schema=2,contract=d.CONTRACT,status='verified_fp32_export_ready_for_separate_DNNDK_ARM_stage',
            output_shape=[1,6],output_tensor_name='fc8',checkpoint_reload_parity=True,calibration_train_only=True,
            calibration_counts=[32]*6,folding=dict(max_absolute_error=1e-6,top1_identical=True),
            files={n:'1'*64 for n in d.EXPORT_FILES},output_contract=output_contract())
        d.check_export(meta)
        for change in ({'contract':'soundjie-alert6-v1'},{'output_shape':[1,5]}, {'calibration_counts':[32]*5},
                       {'folding':dict(max_absolute_error=.01,top1_identical=True)}):
            with self.assertRaises(ValueError):d.check_export(dict(meta,**change))

    def test_fixed_activation_graph_and_kernel_contract(self):
        graph='\n'.join('layer { name: "fix%d" type: "FixedNeuron" }'%i for i in range(6))
        graph+='\nlayer { name: "conv1" type: "ConvolutionFixed" }'
        graph+='\nlayer { name: "fc8" top: "fc8" type: "InnerProductFixed" inner_product_param { num_output: 6 } }'
        d.check_fixed_graph(graph)
        for bad in (graph.replace('FixedNeuron','ReLU'),graph.replace('num_output: 6','num_output: 5')):
            with self.assertRaises(ValueError):d.check_fixed_graph(bad)
        c=d.compile_contract();self.assertEqual(c['kernel'],'soundjie_alert5bg');self.assertEqual(c['output_count'],6)

    def test_fixed_thresholds_use_bg5_and_keep_speech(self):
        _,rows=samples('validation',2);y=np.repeat(np.arange(6),2);p=np.full((12,6),.02);p[np.arange(12),y]=.9
        p[10]=[.7,.06,.06,.06,.06,.06]
        cfg=dict(grid_start=.4,grid_stop_exclusive=1.,grid_step=.025,per_class_background_fpr_limit=.02)
        result=d.assess_scores(y,rows,np.log(p),[.4]*5,cfg)
        self.assertEqual(len(result['post_int8_thresholds']),5)
        self.assertEqual(result['post_int8']['per_class']['speech']['correct'],2)
        self.assertEqual(result['post_int8']['negative_false_alert_rate'],0.)
        self.assertEqual(len(result['per_alert_background_bound_met']),5)
        self.assertFalse(result['independent_acceptance'])

    def test_speech_is_foreground_but_not_an_alert(self):
        from scripts.host.assess_soundjie_alert5bg import operating_metrics
        y=np.array([5,5,4]);p=np.array([[0,0,0,0,1.,0],[1.,0,0,0,0,0],[0,0,0,0,1.,0]])
        rows=[dict(target_index=int(c),negative_subtype='ambient',sampling_identity=str(i),source_group=str(i),
                   speech_evidence='explicit_speech' if c==4 else None) for i,c in enumerate(y)]
        m=operating_metrics(y,p,rows,[.5]*5)
        self.assertEqual(m['background_false_foreground_count'],2)
        self.assertEqual(m['background_false_alert_count'],1)
        self.assertEqual(m['background_false_speech_count'],1)
        self.assertEqual(m['speech_subgroups']['explicit_speech']['recall'],1.)

    def test_float64_avoids_float32_one_saturation(self):
        from scripts.host.train_soundjie_alert6 import softmax
        from scripts.host.soundjie_alert5bg import apply_thresholds
        z=np.array([[20.,0,0,0,0,0]],np.float32)
        self.assertEqual(softmax(z)[0,0],1.)
        self.assertLess(d.probabilities(z)[0,0],1.)
        self.assertEqual(apply_thresholds(d.probabilities(z),[1.]*5)[0],5)


if __name__=='__main__':unittest.main()
