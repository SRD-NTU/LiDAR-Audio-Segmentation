import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
import numpy as np,torch
from protocol import *
from controlled_model import ControlledActiveView
class ProtocolTests(unittest.TestCase):
 def test_rotations_and_budget(self):
  a=np.tile(np.r_[np.zeros(3),np.eye(3).ravel()],(3,1));a[:,0]=[1,3,4]
  _,v,c,d=valid_actions(a,[True,False,True],[9,8,7],3)
  np.testing.assert_array_equal(v,[True,False,False]);np.testing.assert_array_equal(c,[9,8,7])
  a[0,3]=-1
  with self.assertRaises(ValueError):valid_actions(a)
 def test_duplicate_invariance(self):
  torch.manual_seed(5);m=np.array([[1,1,0,0],[0,0,1,1],[1,0,0,0]],bool)
  base=torch.tensor([1.,.5,-.1]);sites=torch.randn(3,2).log_softmax(-1);q=torch.randn(4,3,2,48,4,4,dtype=torch.complex64);lv=torch.randn(4,3,2,48);w=torch.ones(48)/48
  g,d,_=mask_groups(m);u=grouped_information(base,sites,q,lv,w,g,d)
  ix=torch.tensor([2,0,1,0,0,2]);g,d,_=mask_groups(m[ix]);v=grouped_information(base[ix],sites[ix],q[:,ix],lv[:,ix],w,g,d)
  torch.testing.assert_close(u,v,atol=1e-5,rtol=1e-5)
 def test_no_transitive_group_merging(self):
  m=np.array([[1,1,0],[1,1,1],[0,1,1]],bool);g,_,_=mask_groups(m,.6)
  self.assertNotEqual(g[0],g[2])
 def test_probability_is_marginal(self):
  model=ControlledActiveView.__new__(ControlledActiveView)
  state=dict(masks=np.eye(2,dtype=bool),inverse=np.arange(2),base=torch.zeros(2),silent=False)
  result=model.result(state,torch.zeros(2));np.testing.assert_allclose(result['point_probability'],[.5,.5])
 def test_one_acquisition_after_plan(self):
  model=ControlledActiveView.__new__(ControlledActiveView);events=[];future={'value':1}
  def plan(*args,**kwargs):events.append('plan');return {'query':0}
  def acquire(query):events.append('acquire');return future['value'],1
  def complete(state,o,w):events.append('complete');return state['query']
  model.plan=plan;model.complete=complete
  a=model.run(None,None,None,None,acquire);future['value']=999
  b=model.run(None,None,None,None,acquire)
  self.assertEqual(a,b);self.assertEqual(events,['plan','acquire','complete']*2)
 def test_real_plan_ignores_future_payload(self):
  from acoustic_wm.models.future_grounding import FutureAcousticReasoner
  model=ControlledActiveView.__new__(ControlledActiveView);model.device='cpu'
  state=dict(base=torch.tensor([.2,.5]),log_site=torch.zeros(2,1),geometry=(None,None,None),hypotheses=torch.zeros(2,1,3),observed=torch.zeros(48,4,4,dtype=torch.complex64),reliability=torch.ones(48)/48,masks=np.eye(2,dtype=bool))
  class Future:
   information=staticmethod(FutureAcousticReasoner.information)
   def predict(self,*args):
    q=torch.zeros(2,2,1,48,4,4,dtype=torch.complex64);q[1,1]=1;return q,torch.zeros(2,2,1,48)
  model.future=Future();model.current=lambda *args:dict(state)
  a=np.tile(np.r_[np.zeros(3),np.eye(3).ravel()],(2,1));a[1,0]=1
  self.assertEqual(model.plan({},None,None,a)['query'],1)
 def test_empty_valid_menu_never_acquires(self):
  model=ControlledActiveView.__new__(ControlledActiveView);model.plan=lambda *a,**k:{'query':-1};model.complete=lambda p:{'budget_used':0}
  def acquire(q):raise AssertionError('Must not acquire')
  self.assertEqual(model.run(None,None,None,None,acquire)['budget_used'],0)
if __name__=='__main__':unittest.main()
