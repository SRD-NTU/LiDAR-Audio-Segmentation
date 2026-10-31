"""Behavioral tests: acquisition boundary, differentiability, geometry and decoding."""
from pathlib import Path
import sys,unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import ROOT,output_path
from activeview_v3.models import *
from activeview_v3.inference import mask_overlap,plan
from activeview_v3.navigation import ReachableSpace,relative_pose
import numpy as np

def batch():
 torch.manual_seed(17);b,k,h,d=2,5,4,128
 return dict(base=torch.randn(b,k),valid=torch.ones(b,k,dtype=torch.bool),sites=torch.randn(b,k,h).log_softmax(-1),
  mu=torch.randn(b,k,h,d),var=torch.ones(b,k,h,d),target=torch.randn(b,d),audible=torch.ones(b,dtype=torch.bool),
  latent=torch.randn(b,k,128),residual=torch.randn(b,k,d),support=torch.randn(b,k,12),actions=torch.randn(b,12),
  distance=torch.ones(b),overlap=torch.eye(k)[None].repeat(b,1,1),iou=torch.rand(b,k),null_iou=torch.zeros(b),
  initial_silent=torch.zeros(b,dtype=torch.bool))

class Contracts(unittest.TestCase):
 def test_world_gradient_and_frozen_updater(self):
  for neural in [False,True]:
   x=batch();e=EvidenceAdapter(neural=neural).requires_grad_(False);w=ObservationWorld()
   loss,_=world_loss(w,e,x,scale=.25);loss.backward()
   self.assertTrue(torch.isfinite(loss));self.assertGreater(float(w.out.weight.grad.abs().sum()),0)
   self.assertTrue(all(p.grad is None for p in e.parameters()))
 def test_same_updater_real_and_imagined(self):
  x=batch()
  for neural in [False,True]:
   e=EvidenceAdapter(neural=neural)
   torch.testing.assert_close(e.scores(x),e.scores(x,x['target'][:,None])[:,0])
 def test_future_and_labels_cannot_change_plan(self):
  x=batch();e=EvidenceAdapter().eval();w=ObservationWorld().eval();d=DirectValue().eval()
  original=imagined_utility(w,e,x);direct=d(x,e)
  x.update(target=torch.full_like(x['target'],float('nan')),iou=torch.full_like(x['iou'],float('nan')),
           null_iou=torch.full_like(x['null_iou'],float('nan')),audible=~x['audible'])
  torch.testing.assert_close(original,imagined_utility(w,e,x));torch.testing.assert_close(direct,d(x,e))
 def test_silent_observation_is_no_evidence(self):
  x=batch();x['audible'][:]=False;e=EvidenceAdapter()
  torch.testing.assert_close(e.scores(x),e.prior(x))
 def test_zero_initialized_world_matches_physical_control(self):
  x=batch();e=EvidenceAdapter();w=ObservationWorld();control=ObservationWorld(correction=False)
  torch.testing.assert_close(imagined_utility(w,e,x),imagined_utility(control,e,x))
 def test_padding_is_not_a_source_hypothesis(self):
  x=batch();x['valid'][:,-1]=False;x['base'][:,-1]=1e4;e=EvidenceAdapter();w=ObservationWorld()
  u=imagined_utility(w,e,x);x['mu'][:,-1]=1e3;x['var'][:,-1]=100
  torch.testing.assert_close(u,imagined_utility(w,e,x))
 def test_original_point_multiplicity(self):
  row=dict(masks=np.array([[1,0],[1,1]]),inverse=np.array([0,0,0,1]))
  np.testing.assert_allclose(mask_overlap(row),[[1,.75],[.75,1]])
 def test_output_confinement(self):
  with self.assertRaises(ValueError):output_path(ROOT/'../forbidden.json')
 def test_rigid_transform(self):
  pose=relative_pose(np.array([1,2,3]),.8,np.array([2,3,4]),-.2)
  r=pose[3:].reshape(3,3);np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-6)
  self.assertAlmostEqual(float(np.linalg.det(r)),1.,places=6)
 def test_navigation_rejects_unreachable_and_budget(self):
  class Path:pass
  class Map:
   def snap_point(self,x):return x
   def find_path(self,p):
    p.geodesic_distance=float(np.linalg.norm(p.requested_end-p.requested_start));p.points=[p.requested_start,p.requested_end]
    return p.requested_end[0]>=0
  space=ReachableSpace(Map(),Path,[0,0,0])
  self.assertIsNone(space.project([-1,0],0));self.assertIsNone(space.project([4,0],0))
  poses=space.sample(np.random.default_rng(0),30)
  self.assertEqual(len(poses),30);self.assertTrue(all(.15<=p['path_distance_m']<=3 for p in poses))
  self.assertGreater(len({round(p['yaw'],5) for p in poses}),12)

if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
