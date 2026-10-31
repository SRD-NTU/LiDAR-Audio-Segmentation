"""Separate simulator process. Replies expose geometry until a pose is committed."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.rendering import Renderer,dry_segment
from activeview_v3.route import RouteSpace
from activeview_v3.pose_validation import validate_committed_pose
from acoustic_wm.audio.predictive_features import covariance48_from_wave
import numpy as np,traceback,soundfile as sf

renderer=None;metadata=None;space=None
for line in sys.stdin:
 try:
  command=json.loads(line);kind=command['command']
  if kind=='geometry':
   scene=command['scene']
   if renderer is None or renderer.scene['scene_id']!=scene:
    if renderer is not None:renderer.close()
    renderer=Renderer(scene)
   metadata=None;space=renderer.space(command['listener']);reply=dict(ready=True)
  elif kind=='reference':
   sample=command['sample'];scene=sample.split('/')[1]
   if renderer is None or renderer.scene['scene_id']!=scene:
    if renderer is not None:renderer.close()
    renderer=Renderer(scene)
   metadata=json.loads((DATA/sample/'metadata.json').read_text());space=renderer.space(metadata['listener'])
   reply=dict(listener={k:metadata['listener'][k] for k in ['position_world','floor_position_world','yaw_rad']})
  elif kind=='route':
   if metadata is None:raise ValueError('Reference must be loaded before route geometry')
   anchor=renderer.space(metadata['listener'])
   space=RouteSpace(anchor,command['current_floor'],command['visited_positions'])
   reply=dict(ready=True)
  elif kind=='sample':
   reply=space.sample(np.random.default_rng(command['seed']),command['count'],command.get('centre'),command.get('scale'))
  elif kind=='acquire':
   if metadata is None:raise ValueError('Geometry-only server has no source event; provide a real recording through the completion API')
   requested=command['pose'];pose=validate_committed_pose(space,requested)
   wave,rir=renderer.acquire(metadata['source']['position_world'],dry_segment(metadata),pose)
   observed,weight=covariance48_from_wave(wave)
   path=output_path(command['output']);temporary=path.with_suffix('.partial.npz')
   np.savez_compressed(temporary,observed=observed,weight=weight);temporary.replace(path)
   if command.get('save_audio',False):
    sf.write(str(path.with_suffix('.wav')),wave,44100,subtype='FLOAT')
    np.savez_compressed(path.with_suffix('.rir.npz'),rir=rir)
   reply=dict(validation_revision='committed-3d-v1',exact_pose_fallback=pose is requested,output=str(path),rir_sha256=hashlib.sha256(rir.tobytes()).hexdigest(),
    nonzero_channels=(np.max(np.abs(rir),axis=1)>0).tolist(),pose=pose)
  elif kind=='close':
   if renderer is not None:renderer.close()
   print('V3RPC '+json.dumps(dict(ok=True,result=None)),flush=True);break
  else:raise ValueError('Unknown command')
  print('V3RPC '+json.dumps(dict(ok=True,result=reply)),flush=True)
 except Exception as exc:
  traceback.print_exc(file=sys.stderr);print('V3RPC '+json.dumps(dict(ok=False,error=str(exc))),flush=True)
