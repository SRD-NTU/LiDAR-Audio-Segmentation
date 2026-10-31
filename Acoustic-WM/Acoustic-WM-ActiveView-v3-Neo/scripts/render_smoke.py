from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.rendering import Renderer,dry_segment
from activeview_v3.navigation import relative_pose
from acoustic_wm.audio.predictive_features import covariance48_from_wave
import time,numpy as np,soundfile as sf
start=time.time();sample=samples('train')[0];metadata=json.loads((DATA/sample/'metadata.json').read_text())
renderer=Renderer(metadata['scene_id']);loaded=time.time()
try:
 poses=renderer.space(metadata['listener']).sample(np.random.default_rng(20261028),2)
 results=[]
 for i,pose in enumerate(poses):
  before=time.time();wave,rir=renderer.acquire(metadata['source']['position_world'],dry_segment(metadata),pose)
  observed,weight=covariance48_from_wave(wave)
  sf.write(output_path('examples/smoke-%d.wav'%i),wave,44100,subtype='FLOAT')
  np.savez_compressed(output_path('examples/smoke-%d.npz'%i),observed=observed,reliability=weight,rir=rir)
  results.append(dict(pose=pose,seconds=time.time()-before,nonzero_channels=(np.max(np.abs(rir),axis=1)>0).tolist(),
                      active_bands=int((weight>0).sum())))
 report=dict(sample=sample,scene_load_seconds=loaded-start,seconds=time.time()-start,observations=results)
 write_json('reports/render_smoke.json',report);print(json.dumps(report),flush=True)
finally:renderer.close()
