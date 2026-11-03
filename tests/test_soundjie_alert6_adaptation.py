import copy
import unittest
import numpy as np
import torch
from scripts.host.soundjie_alert6 import LABELS,make_model
from scripts.host.adapt_soundjie_alert6 import guard_replay_rows,domain_batches,fixed_bn_train,mixed_loss,select_epoch


class AdaptationTests(unittest.TestCase):
    def test_feature_lineage_pairs_original_id_with_original_hash(self):
        from scripts.host.adapt_soundjie_alert6 import replay_provenance
        parent,row=self.source();row.update(sha256='captured-hash',negative_subtype='alarm')
        entry=replay_provenance(row,12,-3)
        self.assertEqual(entry['parent_sha256'],parent['sha256'])
        self.assertEqual(entry['replay_sha256'],'captured-hash')

    def test_cache_member_cannot_invent_a_training_identity(self):
        from scripts.host.adapt_soundjie_alert6 import replay_provenance,guard_cache_membership
        parent,row=self.source();row.update(sha256='captured-hash',negative_subtype='alarm')
        entry=replay_provenance(row,0,0);guard_cache_membership([entry],[row])
        entry['parent_recording_id']='validation-parent'
        with self.assertRaises(ValueError):guard_cache_membership([entry],[row])

    def test_self_consistent_cache_hashes_cannot_replace_pinned_receipt(self):
        from scripts.host.adapt_soundjie_alert6 import verify_receipt_bindings
        cfg={'schema':1};audit={'cache_sha256':'edited','provenance_sha256':'edited-provenance'}
        receipt={'config':cfg,'cache_sha256':'pinned','provenance_sha256':'pinned-provenance','audit_sha256':'pinned-audit'}
        with self.assertRaises(ValueError):verify_receipt_bindings(receipt,cfg,audit,'edited','edited-provenance','edited-audit')

    def source(self):
        parent=dict(recording_id='p',purpose='train',source_id='fsd50k',source_group='author',
                    source_recording_id='freesound:p',sha256='hash',target_index=0,target_id=LABELS[0])
        row=dict(recording_id='replay:p',parent_recording_id='p',parent_sha256='hash',purpose='train',
                 source_group='author',source_recording_id='freesound:p',target_index=0,target_id=LABELS[0],
                 adaptation_allowed=True,reception={'passed':True},domain='acoustic_replay')
        return parent,row

    def test_parent_validation_rejected_despite_train_flag(self):
        p,r=self.source();p['purpose']='validation'
        with self.assertRaises(ValueError):guard_replay_rows([r],[p])

    def test_changed_parent_identity_or_unreceived_audio_rejected(self):
        p,r=self.source();guard_replay_rows([r],[p])
        for key,value in [('source_group','other'),('parent_sha256','other'),('source_recording_id','other'),
                          ('adaptation_allowed',False),('reception',{'passed':False})]:
            bad=dict(r);bad[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):guard_replay_rows([bad],[p])

    def test_validation_group_cannot_be_smuggled_in(self):
        p,r=self.source();v=dict(p,recording_id='v',purpose='validation')
        with self.assertRaises(ValueError):guard_replay_rows([r],[p,v])

    def test_two_domains_and_recording_families_are_balanced(self):
        y=np.repeat(np.arange(6),11);ids=np.tile(['many']*10+['one'],6)
        batches=domain_batches(y,ids,y,ids,per_class=512,seed=8)
        for a,b in batches:self.assertEqual((len(a),len(b)),(16,16))
        for indices in [np.concatenate([a for a,b in batches]),np.concatenate([b for a,b in batches])]:
            np.testing.assert_array_equal(np.bincount(y[indices]),np.full(6,512))
            self.assertLess(abs(np.mean(ids[indices]=='one')-.5),.04)

    def test_bn_statistics_frozen_but_student_weights_train(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        student=make_model();teacher=copy.deepcopy(student).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        before={k:v.clone() for k,v in student.state_dict().items()}
        fixed_bn_train(student)
        x=torch.randn(2,1,64,96);y=torch.tensor([0,5]);optimizer=torch.optim.SGD(student.parameters(),lr=.01)
        loss=mixed_loss(student,teacher,x,y,x*.5,y,temperature=2.,distill_weight=.3)
        loss.backward();optimizer.step()
        for key,value in student.state_dict().items():
            if 'running_' in key or 'num_batches_tracked' in key:torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(student.conv1.weight,before['conv1.weight']))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_lower_replay_loss_weight_preserves_exact_domain_weighting(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        student=make_model().eval();teacher=copy.deepcopy(student).eval()
        x=torch.randn(2,1,64,96);rx=x*.4;y=torch.tensor([0,5]);ry=torch.tensor([2,3])
        expected=.75*torch.nn.functional.cross_entropy(student(x),y)+.25*torch.nn.functional.cross_entropy(student(rx),ry)
        actual=mixed_loss(student,teacher,x,y,rx,ry,temperature=2.,distill_weight=0.,replay_fraction=.25)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        with self.assertRaises(ValueError):mixed_loss(student,teacher,x,y,rx,ry,2.,.3,replay_fraction=-1.)

    def test_epoch_zero_kept_if_recall_or_false_alert_guard_fails(self):
        base=dict(macro_recall=.7,negative_false_alert_rate=.1,per_class={k:{'recall':.7} for k in LABELS})
        bad=copy.deepcopy(base);bad['macro_recall']=.8;bad['negative_false_alert_rate']=.13
        recall_bad=copy.deepcopy(base);recall_bad['macro_recall']=.8;recall_bad['per_class'][LABELS[0]]['recall']=.59
        history=[dict(epoch=0,metrics=base),dict(epoch=1,metrics=bad),dict(epoch=2,metrics=recall_bad)]
        self.assertEqual(select_epoch(history,base)['epoch'],0)
        good=copy.deepcopy(base);good['macro_recall']=.72
        history.append(dict(epoch=3,metrics=good));self.assertEqual(select_epoch(history,base)['epoch'],3)


if __name__=='__main__':unittest.main()
