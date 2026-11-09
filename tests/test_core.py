"""Synthetic checks; never train on participant data."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import study as s
from train import model
from prepare_data import preprocess

class CoreTests(unittest.TestCase):
    def test_models_and_matched_initial_encoders(self):
        torch.set_num_threads(1)
        for base in ('r0','r1','r2'):
            linear=model(base+'_linear',3001);mlp=model(base+'_mlp',3001)
            for a,b in zip(linear.encoders.parameters(),mlp.encoders.parameters()): self.assertTrue(torch.equal(a,b))
            for head,net in [('linear',linear),('mlp',mlp)]:
                mode=base+'_'+head;self.assertEqual(sum(p.numel() for p in net.parameters()),s.PARAMETERS[mode]);self.assertEqual(s.nominal_macs(mode),s.MACS[mode])
                x=torch.zeros(2,s.model_spec(mode)['input_channels'],512);self.assertEqual(net(x).shape,(2,))
    def test_checkpoint_ties_and_ranges(self):
        h=[dict(epoch=i,inner_mae_bpm=10.) for i in range(1,41)];h[1]['inner_mae_bpm']=9.99995;h[11]['inner_mae_bpm']=9.;h[35]['inner_mae_bpm']=8.
        self.assertEqual(s.select_checkpoints(h)[12]['selected_epoch'],12);self.assertEqual(s.select_checkpoints(h)[40]['selected_epoch'],36)
        with self.assertRaises(ValueError):s.select_checkpoints(h[:-1])
        h[0]['outer_mae_bpm']=1.
        with self.assertRaises(ValueError):s.select_checkpoints(h)
    def test_float32_clipped_selection(self):
        a=np.array([20,100,240],np.float32);b=np.array([40,90,210],np.float32);self.assertEqual(s.selection_mae(a,b),10.)
        with self.assertRaises(ValueError):s.selection_mae(a.astype(np.float64),b)
    def test_preprocess_shape_mean_and_nonmutation(self):
        rng=np.random.default_rng(42);ppg=rng.normal(size=1024).astype(np.float32);acc=rng.normal(size=(1024,3)).astype(np.float32);p=ppg.copy();a=acc.copy()
        x=preprocess(ppg,acc,np.array([0,128,256]));self.assertEqual(x.shape,(3,4,512));self.assertEqual(x.dtype,np.float32)
        np.testing.assert_allclose(x.mean(axis=2),0,atol=1e-7);np.testing.assert_array_equal(p,ppg);np.testing.assert_array_equal(a,acc)
    def test_split_integrity(self):
        self.assertEqual(len(s.FOLDS),15)
        for fold,roles in s.FOLDS.items():
            self.assertEqual([len(roles[k]) for k in ('train','inner','outer')],[11,3,1]);self.assertEqual(sorted(sum(roles.values(),[])),list(range(1,16)))
if __name__=='__main__':unittest.main()
