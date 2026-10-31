"""Known-map navigation interface. No source location or target label is accepted."""
import numpy as np
def rotation(yaw):
 c,s=np.cos(yaw),np.sin(yaw)
 return np.array([[c,0,s],[0,1,0],[-s,0,c]])@np.array([[0,-1,0],[0,0,1],[-1,0,0]])
def relative_pose(origin,origin_yaw,position,yaw):
 r=rotation(origin_yaw)
 return np.r_[(np.asarray(position)-origin)@r,(r.T@rotation(yaw)).reshape(-1)].astype('float32')
class ReachableSpace:
 def __init__(self,pathfinder,shortest_path_factory,origin_floor,budget=3.,height=1.2,clearance=None):
  self.pathfinder=pathfinder;self.path_factory=shortest_path_factory
  self.origin=np.asarray(origin_floor,dtype=float);self.budget=budget;self.height=height;self.clearance=clearance
 def project(self,xz,yaw):
  proposal=self.origin.copy();proposal[[0,2]]=xz
  floor=np.asarray(self.pathfinder.snap_point(proposal),dtype=float)
  if not np.isfinite(floor).all() or np.linalg.norm(floor[[0,2]]-xz)>.2:return None
  path=self.path_factory();path.requested_start=self.origin;path.requested_end=floor
  if not self.pathfinder.find_path(path):return None
  distance=float(path.geodesic_distance)
  if not np.isfinite(distance) or distance>self.budget+1e-6 or distance<.15:return None
  position=floor+np.array([0,self.height,0])
  if self.clearance is not None and not self.clearance(position,.10):return None
  return dict(position=position.tolist(),floor=floor.tolist(),yaw=float(np.arctan2(np.sin(yaw),np.cos(yaw))),
              path_distance_m=distance,path=np.asarray(path.points).tolist())
 def sample(self,rng,count,centre=None,scale=None):
  result=[]
  for _ in range(max(1000,count*150)):
   if centre is None:
    angle=rng.uniform(-np.pi,np.pi);radius=self.budget*np.sqrt(rng.random())
    xz=self.origin[[0,2]]+radius*np.array([np.cos(angle),np.sin(angle)]);yaw=rng.uniform(-np.pi,np.pi)
   else:
    proposal=rng.normal(centre,scale);xz=proposal[:2];yaw=proposal[2]
   pose=self.project(xz,yaw)
   if pose is not None:result.append(pose)
   if len(result)==count:return result
  if not result:raise RuntimeError('No reachable receiver position within movement budget')
  return result
