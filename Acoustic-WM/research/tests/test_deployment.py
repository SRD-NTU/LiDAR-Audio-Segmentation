import sys,unittest,importlib.util,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
import numpy as np,torch,acoustic_wm
acoustic_wm.__path__.append(str(FUSION/'code/acoustic_wm'));sys.path.insert(0,str(FUSION))
spec=importlib.util.spec_from_file_location('experiment_pipeline',FUSION/'recovery/pipeline.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from deployment_menu import load_menu
class DeploymentTests(unittest.TestCase):
 def test_actual_recovery_filters_actions_before_one_acquisition(self):
  events=[]
  class StopAcquisition(Exception):pass
  class Encoder:
   def encode_arrays(self,*args):events.append('reference_encoding');return {}
  class Model:
   def plan(self,row,obs,w,actions):events.append(('plan',actions[:,0].tolist()));return {'query':0}
  def acquire(index):events.append(('acquire',index));raise StopAcquisition()
  actions=np.tile(np.r_[np.zeros(3),np.eye(3).ravel()],(2,1));actions[:,0]=[4.,1.]
  pipe=module.RecoveryPipeline(Model(),Encoder(),None)
  with self.assertRaises(StopAcquisition):pipe.run(np.ones((3,3)),np.ones((4096,4)),actions,acquire,movement_cost=[1.,10.],max_translation_m=3.)
  self.assertEqual(events,['reference_encoding',('plan',[1.]),('acquire',1)])
 def test_actual_recovery_no_valid_action_returns_initial_result(self):
  class Encoder:
   def encode_arrays(self,*args):return {}
  class Model:
   def current(self,*args):return {'base':torch.zeros(1)}
   def result(self,*args):return dict(sound_mask=np.array([1,0,0],np.uint8),point_probability=np.array([1.,0.,0.]),posterior=np.ones(1),selected_object=0)
  def acquire(index):raise AssertionError('No acquisition allowed')
  action=np.r_[4.,0.,0.,np.eye(3).ravel()][None]
  result=module.RecoveryPipeline(Model(),Encoder(),None).run(np.ones((3,3)),np.ones((4096,4)),action,acquire,movement_cost=[8.])
  self.assertEqual(result['budget_used'],0);self.assertEqual(result['distance_m'],0);self.assertEqual(result['chosen_action'],-1)
 def test_menu_loader_accepts_all_invalid_without_opening_future(self):
  pose=np.eye(4).tolist();menu={'availability_basis':'geometry','reference':dict(lidar='missing.npz',audio='missing.wav',receiver_to_world=pose),'observations':[dict(id='x',lidar='unread_future.npz',audio='unread_future.wav',receiver_to_world=pose,valid=False,movement_cost=4.)]}
  with tempfile.TemporaryDirectory(dir=BASE/'runtime') as tmp:
   path=Path(tmp)/'menu.json';path.write_text(json.dumps(menu));_,_,valid,cost=load_menu(path)
  self.assertFalse(valid.any());self.assertEqual(cost[0],4.)
if __name__=='__main__':unittest.main()
