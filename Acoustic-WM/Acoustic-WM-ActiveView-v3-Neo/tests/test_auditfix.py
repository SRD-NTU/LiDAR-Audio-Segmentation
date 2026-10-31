"""Independent behavior checks for the revised evidence and control paths."""
from pathlib import Path
import sys,unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from activeview_v3.embedding import AcousticEmbedding
from activeview_v3.models import EvidenceAdapter,ObservationWorld,DirectValue,world_loss
from activeview_v3.multiview import JointBelief,component_increment

def batch():
 torch.manual_seed(11)
 b,k,h,d=2,3,4,128
 raw=torch.randn(b,48,16)
 target=AcousticEmbedding('cpu').encode_packed(raw)
 return dict(base=torch.randn(b,k),valid=torch.ones(b,k,dtype=torch.bool),
  sites=torch.randn(b,k,h).log_softmax(-1),mu=torch.randn(b,k,h,d),var=torch.ones(b,k,h,d),
  raw_target=raw,target=target,band_reliability=torch.ones(b,48),audible=torch.ones(b,dtype=torch.bool),
  latent=torch.randn(b,k,128),residual=torch.randn(b,k,d),support=torch.randn(b,k,12),
  actions=torch.randn(b,12),distance=torch.ones(b),iou=torch.rand(b,k))

class AuditFixTests(unittest.TestCase):
 def test_all_active_recovers_original_likelihood(self):
  x=batch();model=EvidenceAdapter()
  current=model.scores(x)
  original=model.scores({k:v for k,v in x.items() if k not in ['raw_target','band_reliability']})
  torch.testing.assert_close(current,original,atol=1e-4,rtol=1e-5)
 def test_inactive_bands_cannot_change_real_evidence(self):
  x=batch();model=EvidenceAdapter()
  x['band_reliability'][:,1:]=0
  first=model.scores(x)
  x['raw_target'][:,1:]+=100
  torch.testing.assert_close(model.scores(x),first,atol=0,rtol=0)
  x['band_reliability'][:]=0
  torch.testing.assert_close(model.scores(x),model.prior(x),atol=0,rtol=0)
 def test_full_direct_reads_site_coordinates_and_trains(self):
  x=batch();evidence=EvidenceAdapter().requires_grad_(False)
  control=DirectValue(full_information=True)
  x['support'].requires_grad_(True)
  output=control(x,evidence)
  output.sum().backward()
  self.assertEqual(tuple(output.shape),(2,))
  self.assertGreater(float(x['support'].grad.abs().sum()),0)
  self.assertGreater(float(control.site_encoder[0].weight.grad.abs().sum()),0)
 def test_masked_world_loss_has_finite_gradient(self):
  x=batch();evidence=EvidenceAdapter().requires_grad_(False);world=ObservationWorld()
  x['band_reliability'][:,10:]=0
  loss,_=world_loss(world,evidence,x)
  loss.backward()
  self.assertTrue(bool(torch.isfinite(loss)))
  self.assertGreater(float(world.out.weight.grad.abs().sum()),0)
 def test_masked_joint_update_matches_two_view_updater(self):
  x=batch();evidence=EvidenceAdapter();x['band_reliability'][:,12:]=0
  for i in range(len(x['base'])):
   one={k:v[i:i+1] for k,v in x.items()}
   belief=JointBelief(evidence.prior(one)[0],one['sites'][0])
   belief.update(component_increment(evidence,one,one['target'],1.)[0],str(i))
   torch.testing.assert_close(belief.marginals()[0],evidence.scores(one).log_softmax(-1)[0],
                              atol=1e-5,rtol=1e-5)

if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
