"""Probe selection must not convert train/augmented or weak vocal rows into speech smoke."""
import copy
import unittest
import numpy as np
from scripts.host.prepare_soundjie_alert5bg_probes import select_indices


def fixture():
    rows=[dict(purpose='validation',derivation=None,target_index=c,recording_id=str(c),
               speech_evidence='explicit_speech' if c==4 else None) for c in range(6)]
    p=np.full((6,6),.02);p[np.arange(6),np.arange(6)]=.9
    return rows,p


class ProbeSelectionTests(unittest.TestCase):
    def test_speech_uses_explicit_labels_despite_higher_weak_vocal_score(self):
        rows,p=fixture();rows.append(dict(rows[4],recording_id='weak',speech_evidence='inherited_vocal_weak_label'))
        p=np.vstack([p,[0,0,0,0,1,0]])
        self.assertEqual(select_indices(rows,p,[.5]*5),list(range(6)))

    def test_training_or_augmented_validation_is_rejected(self):
        rows,p=fixture()
        for change in ({'purpose':'train'},{'derivation':{'gain_db':0}}):
            bad=copy.deepcopy(rows);bad[0].update(change)
            with self.assertRaises(ValueError):select_indices(bad,p,[.5]*5)

    def test_missing_correct_class_is_rejected(self):
        rows,p=fixture();p[0]=p[1]
        with self.assertRaises(ValueError):select_indices(rows,p,[.5]*5)


if __name__=='__main__':unittest.main()
