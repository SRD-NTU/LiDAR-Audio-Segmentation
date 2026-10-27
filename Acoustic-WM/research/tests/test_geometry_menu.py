from pathlib import Path
import sys,unittest,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geometry_menu import sample_geometry_menu
class MenuTests(unittest.TestCase):
 def test_menu_depends_only_on_geometry_and_seed(self):
  class Geometry:
   def seed(self,s):self.rng=np.random.default_rng(s)
   def get_random_navigable_point(self):return np.array([*self.rng.uniform(-2,2,1),0.,*self.rng.uniform(-2,2,1)])
  cfg={'receiver_count':12,'attempt_limit':1000,'radius_m':3.,'minimum_separation_m':.3}
  first=sample_geometry_menu(Geometry(),lambda p,d:True,123,cfg)
  # Different target annotations cannot be supplied to this API.
  second=sample_geometry_menu(Geometry(),lambda p,d:True,123,cfg)
  self.assertEqual(first,second);self.assertEqual(len(first),12)
  for p in first:self.assertLessEqual(np.linalg.norm(np.array(p['position'])-first[0]['position']),3.)
if __name__=='__main__':unittest.main()
