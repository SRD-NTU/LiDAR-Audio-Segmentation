"""Target-free menu generation: the function cannot receive source identity."""
import numpy as np

def sample_geometry_menu(pathfinder,clearance,seed,config):
 pathfinder.seed(seed);rng=np.random.default_rng(seed)
 for anchor_attempt in range(config.get('anchor_attempts',32)):
  anchor=None;poses=[]
  for attempt in range(config['attempt_limit']):
   floor=np.asarray(pathfinder.get_random_navigable_point(),dtype=float)
   if not np.isfinite(floor).all():continue
   pos=floor+np.array([0.,1.2,0.])
   if not clearance(pos,.15):continue
   if anchor is None:anchor=pos.copy()
   if np.linalg.norm(pos-anchor)>config['radius_m']:continue
   if any(np.linalg.norm(pos-np.asarray(p['position']))<config['minimum_separation_m'] for p in poses):continue
   poses.append({'position':pos.tolist(),'floor':floor.tolist(),'yaw':float(rng.uniform(-np.pi,np.pi))})
   if len(poses)==config['receiver_count']:return poses
 raise RuntimeError('No geometry-only common menu satisfies the frozen pose count and radius')
