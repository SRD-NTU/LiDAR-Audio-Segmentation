import sys,unittest,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
import numpy as np,torch
import acoustic_wm
acoustic_wm.__path__.append(str(FUSION/'code/acoustic_wm'))
sys.path.insert(0,str(FUSION))
from acoustic_wm.fusion import fuse_clouds,actions_from_poses
from recovery.union import geometric_overlap,assemble
from recovery.features import ConservativeRanker
from train_controls import StandardFusion,scores
class IntegrationTests(unittest.TestCase):
 def test_rigid_registration_and_membership(self):
  from scipy.spatial.transform import Rotation
  rng=np.random.default_rng(7);world=rng.normal(size=(50,3));left=np.eye(4);right=np.eye(4);right[:3,:3]=Rotation.from_euler('xyz',[.4,.2,.3]).as_matrix();right[:3,3]=[.7,-.8,.4]
  local=(world-right[:3,3])@right[:3,:3];action=actions_from_poses(left,[right])[0]
  fused=fuse_clouds(world,local,action,0);np.testing.assert_allclose(fused.xyz[:50],fused.xyz[50:],atol=5e-7)
  combined=fuse_clouds(np.array([[.011,0,0],[1.011,0,0]]),np.array([[.012,0,0],[2.011,0,0]]),np.r_[np.zeros(3),np.eye(3).ravel()],.02)
  np.testing.assert_array_equal(combined.view_membership,[3,1,2]);np.testing.assert_array_equal(combined.reference_to_fused,[0,1]);np.testing.assert_array_equal(combined.second_to_fused,[0,2])
 def test_overlap_matches_uncompressed(self):
  r=torch.tensor([[1,1,0],[0,1,1]],dtype=torch.bool);f=torch.tensor([[1,0,1,0],[0,1,1,1]],dtype=torch.bool);ri=torch.tensor([0,1,1,2,2,2]);fi=torch.tensor([0,1,2,2,3,3]);a=r[:,ri];b=f[:,fi]
  expected=(a[:,None]&b[None]).sum(-1)/(a[:,None]|b[None]).sum(-1);torch.testing.assert_close(geometric_overlap(r,ri,f,fi),expected)
 def test_zero_correction_retains_reference(self):
  x,base,_,_=assemble(torch.randn(3,245),torch.tensor([-3.,-1.,-2.]),torch.randn(4,245),torch.tensor([999.,2.,3.,4.]),torch.rand(3,4),torch.randn(3,4,3),torch.randn(4,4,3))
  model=ConservativeRanker(torch.zeros(297),torch.ones(297),'mlp',2);score,_=model(x[None],base[None],torch.ones(1,7,dtype=torch.bool));self.assertEqual(int(score.argmax()),1)
 def test_standard_fusion_receives_full_covariance_and_has_gradients(self):
  model=StandardFusion(98);base=torch.zeros(2,3);data={'base':base,'feature':torch.randn(2,6,3,98),'observation':torch.randn(2,6,4108),'valid':torch.ones(2,3,dtype=torch.bool),'query_valid':torch.ones(2,6,dtype=torch.bool)}
  score=scores(model,data,torch.arange(2),torch.tensor([2,4]),'standard');self.assertEqual(tuple(score.shape),(2,3));loss=torch.nn.functional.cross_entropy(score,torch.tensor([0,1]));loss.backward();self.assertTrue(torch.isfinite(model.candidate[-1].weight.grad).all())
 def test_bounded_ranker_can_change_ranking(self):
  model=ConservativeRanker(torch.zeros(1),torch.ones(1),'linear',2.)
  with torch.no_grad():model.network.weight[0,0]=10
  score,delta=model(torch.tensor([[[0.],[1.]]]),torch.tensor([[0.,-.1]]),torch.ones(1,2,dtype=torch.bool));self.assertEqual(int(score.argmax()),1);self.assertLessEqual(float(delta.abs().max()),2.)
if __name__=='__main__':unittest.main()
