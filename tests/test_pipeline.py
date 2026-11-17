"""Local smoke tests; real 2.63GB XML integration must run in user's WSL."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from settings import MAX_SLOTS
from ball_missing.mask_generator import generate_mask,_run_lengths
from inference.decoders import vit3,spell_proxy
from ad_tree.features import make_tree_features
from preprocessing import load_idsse


class PipelineSmoke(unittest.TestCase):
    def test_viterbi(self):
        n=100
        h=np.r_[np.full(50,.9),np.full(50,.05)]
        a=1-h; d=np.zeros(n)+.01
        result=vit3(h,a,d)
        self.assertEqual(int(result[0]),0)
        self.assertEqual(int(result[-1]),1)
        info=spell_proxy(result,result,np.ones(n,int))
        self.assertAlmostEqual(info['spell_f1_proxy'],1.0,places=5)

    def test_run_gaps_dont_join(self):
        mask=np.array([1,1,1,1,1],bool)
        self.assertEqual(_run_lengths(mask,np.array([10,15,20,35,40])),[3,2])

    def test_mask_quota(self):
        n=3000; frames=np.arange(n)*5
        observed=np.ones(n,bool)
        observed[100:110]=False
        gt=np.ones(n,bool)
        syn,report=generate_mask(gt,observed,frames,trials=4,tolerance=.02)
        self.assertFalse(np.any(syn[100:110]))
        self.assertAlmostEqual(report['final_alive_missing_rate'],.106,delta=.02)

    def test_tree_features_shape(self):
        rng=np.random.default_rng(4)
        n=150;xy=np.full((n,MAX_SLOTS,2),np.nan,np.float32)
        xy[:,:5,0]=rng.normal(50,10,(n,5))
        xy[:,:5,1]=rng.normal(35,5,(n,5))
        teams=np.zeros(MAX_SLOTS,np.int8);teams[:3]=-1;teams[3:5]=1
        per=np.r_[np.ones(75,int),np.ones(75,int)*2]
        f=make_tree_features(xy,teams,per)
        self.assertEqual(f.shape,(150,347))
        self.assertTrue(np.isfinite(f).all())

    def test_frame_join(self):
        # Mock Kloppy and XML extraction to test exact frame+period matching.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'base.npz'
            for k,prefix in load_idsse.PREFIX.items():
                (root/f'{prefix}DFL-MAT-J03WPY.xml').write_text('<x/>')
            xy=np.zeros((6,MAX_SLOTS,2),np.float32)
            teams=np.zeros(MAX_SLOTS,np.int8);teams[:2]=[-1,1]
            frames=np.array([10,15,20,100,105,110])
            per=np.array([1,1,1,2,2,2])
            import pandas as pd
            ball=pd.DataFrame({'frame':frames,'period':per,
                'ball_x':[50.]*6,'ball_y':[30.]*6,'gt_orig':[0,0,1,2,2,1],
                'gt_mid':[0,0,0,2,2,1],'gt_loose':[0,0,1,2,2,1]})
            with patch.object(load_idsse,'_player_array_from_kloppy',return_value=(xy,teams,frames,per)),\
                 patch.object(load_idsse,'_original_ball_and_labels',return_value=ball):
                load_idsse.build_one(root,'J03WPY',out)
            with np.load(out) as z:
                self.assertTrue(np.array_equal(z['frame'],frames))
                self.assertTrue(np.array_equal(z['gt_mid'],[0,0,0,2,2,1]))
                self.assertEqual(z['xy'].shape,(6,46,2))


if __name__=='__main__':unittest.main()
