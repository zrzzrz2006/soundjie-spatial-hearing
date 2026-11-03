import csv
import io
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import zipfile

from scripts.host import acquire_soundjie_fsd50k as fsd
from scripts.host.build_soundjie_d4_manifest import partition_records


CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/soundjie_fsd50k_sources.json').read_text(encoding='utf-8'))


class FSD50KTests(unittest.TestCase):
    def test_csv_quoted_multilabel_and_mid_alignment(self):
        rows = fsd.parse_ground_truth('fname,labels,mids,split\n123,"Shatter,Glass","/m/07rn7sz,/m/039jq",train\n')
        self.assertEqual(rows[0]['labels'], ['Shatter', 'Glass'])
        with self.assertRaises(ValueError):
            fsd.parse_ground_truth('fname,labels,mids,split\n123,"Shatter,Glass",/m/039jq,train\n')
        with self.assertRaises(ValueError):
            fsd.parse_ground_truth('fname,labels,mids,split\n../123,Glass,/m/039jq,train\n')

    def test_precise_labels_and_conflicts(self):
        self.assertEqual(fsd.classify(['Shatter', 'Glass'], CONFIG)['target_index'], 3)
        self.assertIsNone(fsd.classify(['Glass'], CONFIG))
        self.assertIsNone(fsd.classify(['Alarm'], CONFIG))
        self.assertIsNone(fsd.classify(['Shatter', 'Glass', 'Screaming'], CONFIG))
        self.assertIsNone(fsd.classify(['Crying_and_sobbing', 'Human_voice', 'Speech'], CONFIG))
        self.assertIsNone(fsd.classify(['Vehicle_horn_and_car_horn_and_honking', 'Traffic_noise_and_roadway_noise'], CONFIG))
        self.assertEqual(fsd.classify(['Crying_and_sobbing', 'Screaming', 'Human_voice'], CONFIG)['target_index'], 5)

    def test_old_ids_every_purpose_and_hashes(self):
        rows = [dict(source_group='esc50:' + str(i), purpose=p, sha256=str(i)*64)
                for i, p in enumerate(('train', 'validation', 'demo_shoot', 'demo_calibration', 'd5_holdout'), 1)]
        ids, hashes = fsd.exclusions(rows)
        self.assertEqual(ids, {'1', '2', '3', '4', '5'})
        self.assertEqual(len(hashes), 5)

    def test_uploader_split_stable_casefold_and_order_independent(self):
        self.assertEqual(fsd.uploader_group(' Alice '), fsd.uploader_group('ALICE'))
        groups = [fsd.uploader_group('person' + str(i)) for i in range(100)]
        forward = {g: fsd.purpose_for(g, CONFIG) for g in groups}
        backward = {g: fsd.purpose_for(g, CONFIG) for g in reversed(groups)}
        self.assertEqual(forward, backward)
        self.assertEqual(set(forward.values()), {'train', 'validation'})

    def test_license_fail_closed(self):
        self.assertEqual(fsd.license_name('http://creativecommons.org/licenses/by/3.0/', CONFIG), 'CC-BY-3.0')
        self.assertIsNone(fsd.license_name('http://creativecommons.org/licenses/by-nc/3.0/', CONFIG))
        self.assertIsNone(fsd.license_name('https://example.org/licenses/by/3.0/', CONFIG))

    def test_selection_excludes_old_ids_and_caps_uploader_before_audio(self):
        rows = [dict(fname=str(i), labels=['Shatter', 'Glass'], mids=['/m/07rn7sz', '/m/039jq'], split='train') for i in range(1, 12)]
        meta = {r['fname']: dict(uploader='alice' if int(r['fname']) < 10 else 'bob', license='http://creativecommons.org/publicdomain/zero/1.0/') for r in rows}
        selected, rejected, _ = fsd.make_plan(rows, meta, {'1'}, CONFIG)
        reverse, _, _ = fsd.make_plan(list(reversed(rows)), meta, {'1'}, CONFIG)
        self.assertEqual(selected, reverse)
        self.assertNotIn('1', {r['freesound_id'] for r in selected})
        self.assertEqual(rejected['old_freesound_id_any_purpose'], 1)
        self.assertEqual(len([r for r in selected if r['author']=='alice']), 4)
        for author in ('alice', 'bob'):
            self.assertEqual(len({r['purpose'] for r in selected if r['author']==author}), 1)

    def test_known_historical_uploaders_removed_across_all_new_purposes(self):
        rows = [dict(freesound_id='2', source_group=fsd.uploader_group('alice'), purpose='train'),
                dict(freesound_id='3', source_group=fsd.uploader_group('ALICE'), purpose='validation'),
                dict(freesound_id='4', source_group=fsd.uploader_group('bob'), purpose='validation')]
        kept, audit = fsd.exclude_known_old_uploaders(rows, {'1', '9'}, {'1': {'uploader': 'Alice'}})
        self.assertEqual([r['freesound_id'] for r in kept], ['4'])
        self.assertEqual(audit['old_ids_without_uploader'], 1)

    def test_central_directory_crc_and_offsets(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('FSD50K.dev_audio/123.wav', b'audio bytes')
        data = stream.getvalue()
        eocd = fsd.parse_eocd(data[-65536:])
        entries = fsd.parse_central_directory(data[eocd['offset']:eocd['offset'] + eocd['size']])
        item = entries['123']
        self.assertEqual(item['uncompressed_size'], 11)
        fsd.verify_member(b'audio bytes', item)
        with self.assertRaises(ValueError):
            fsd.verify_member(b'wrong audio', item)
        with self.assertRaises(ValueError):
            fsd.parse_central_directory(b'bad archive')

    def test_ignored_or_wrong_http_range_rejected(self):
        with self.assertRaises(ValueError):
            fsd.validate_range(200, None, 10, 19, b'0123456789')
        with self.assertRaises(ValueError):
            fsd.validate_range(206, 'bytes 0-9/100', 10, 19, b'0123456789')
        fsd.validate_range(206, 'bytes 10-19/100', 10, 19, b'0123456789')

    def test_lowercase_http_header_names(self):
        with patch.object(fsd, 'get_bytes', return_value=(b'0123456789', 206, {'content-range': 'bytes 10-19/100'})):
            self.assertEqual(fsd.range_bytes('https://example.org/audio.zip', 10, 19), b'0123456789')

    def test_partition_contract_and_duplicate_audio_cross_split(self):
        base = dict(recording_id='fsd50k:1', source_group='freesound-uploader:alice',
                    purpose='train', domain='direct', sha256='a'*64, target_index=3,
                    target_id='glass_breaking', path='F:/audio/1.wav', license='CC0-1.0',
                    original_url='https://freesound.org/people/alice/sounds/1/')
        self.assertEqual(len(partition_records([base])['train']), 1)
        with self.assertRaises(ValueError):
            partition_records([base, dict(base, recording_id='fsd50k:2', purpose='validation')])
        with self.assertRaises(ValueError):
            partition_records([base, dict(base, recording_id='fsd50k:2', source_group='bob', purpose='validation')])


if __name__ == '__main__':
    unittest.main()
