from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.rendering import Renderer
from activeview_v3.route import RouteSpace
from activeview_v3.pose_validation import validate_committed_pose
import copy
import numpy as np
sample='test/00025-ixTj1aTMup2/304/00001'
commit=json.loads((ROOT/'data/multiview'/sample/'random-1/view-4.commit.json').read_text())
listener=json.loads((DATA/sample/'metadata.json').read_text())['listener']
r=Renderer(sample.split('/')[1])
try:
 anchor=r.space(listener);route=RouteSpace(anchor,commit['current_floor'],commit['visited_positions']);pose=commit['pose']
 floor=np.asarray(pose['floor']);pf=anchor.pathfinder
 report=dict(committed=pose,anchor_reprojection=anchor.project(floor[[0,2]],pose['yaw']),route_reprojection=route.project(floor[[0,2]],pose['yaw']),snap_3d=np.asarray(pf.snap_point(floor)).tolist(),clearance=r.mesh.clearance(np.asarray(pose['position']),.10),separation=float(np.linalg.norm(np.asarray(commit['visited_positions'])-pose['position'],axis=1).min()))
 for name,start in [('anchor',anchor.origin),('current',np.asarray(commit['current_floor']))]:
  path=anchor.path_factory();path.requested_start=start;path.requested_end=floor
  success=pf.find_path(path);report[name+'_exact_path']=dict(success=bool(success),distance=float(path.geodesic_distance),points=np.asarray(path.points).tolist())
 validated=validate_committed_pose(route,pose)
 assert validated==pose
 report['exact_pose_repair_passed']=True
 for field,delta in [('floor',.5),('position',.1),('path_distance_m',.1)]:
  invalid=copy.deepcopy(pose)
  if field=='path_distance_m':invalid[field]+=delta
  else:invalid[field][1]+=delta
  try:validate_committed_pose(route,invalid)
  except ValueError:pass
  else:raise AssertionError('Accepted invalid '+field)
 report['invalid_geometry_rejected']=True
 write_json('reports/committed-pose-diagnosis.json',report)
 print(json.dumps(report),flush=True)
finally:r.close()
