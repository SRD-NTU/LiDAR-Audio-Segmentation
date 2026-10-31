"""Render continuous geometry-only destinations; retain silent observations without retries."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.rendering import Renderer,dry_segment
from activeview_v3.navigation import relative_pose
from acoustic_wm.audio.predictive_features import covariance48_from_wave
import argparse,time,numpy as np

def run(args):
 rows=samples(args.split);scenes=sorted({r.split('/')[1] for r in rows})
 start=time.time();done=0
 for scene in getattr(args,'scenes',scenes[args.rank::args.workers]):
  todo=[r for r in rows if r.split('/')[1]==scene and not (ROOT/'data/rendered'/(r+'.json')).exists()]
  if not todo:continue
  renderer=Renderer(scene)
  try:
   for sample in todo:
    metadata=json.loads((DATA/sample/'metadata.json').read_text());listener=metadata['listener']
    # Source ID and source position cannot influence pose sampling or RNG seed.
    key=json.dumps(dict(scene=scene,floor=listener['floor_position_world'],yaw=listener['yaw_rad'],seed=20261028),sort_keys=True)
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    poses=renderer.space(listener).sample(np.random.default_rng(seed),args.poses)
    if len(poses)!=args.poses:raise RuntimeError('Insufficient valid continuous poses for '+sample)
    observations=[];weights=[];hashes=[];nonzero=[];actions=[];dry=dry_segment(metadata)
    for pose in poses:
     wave,rir=renderer.acquire(metadata['source']['position_world'],dry,pose)
     obs,w=covariance48_from_wave(wave);observations.append(obs);weights.append(w)
     hashes.append(hashlib.sha256(rir.tobytes()).hexdigest())
     nonzero.append((np.max(np.abs(rir),axis=1)>0).tolist())
     actions.append(relative_pose(np.asarray(listener['position_world']),listener['yaw_rad'],pose['position'],pose['yaw']))
    destination=output_path('data/rendered/'+sample+'.npz')
    temporary=destination.with_suffix('.partial.npz')
    np.savez_compressed(temporary,observed=observations,weight=weights,actions=actions,
                        distance=np.array([p['path_distance_m'] for p in poses],dtype='float32'))
    temporary.replace(destination)
    write_json('data/rendered/'+sample+'.json',dict(sample=sample,poses=poses,seed=seed,
     pose_rule='uniform disk proposals, geometry-only projection and path validation, no source inputs',
     budget_m=3.,sensor_height_m=1.2,rir_sha256=hashes,nonzero_channels=nonzero,
     original_metadata_sha256=sha(DATA/sample/'metadata.json'),archive_reference_sha256=sha(ARCHIVE/'cache/reference'/(sample+'.npz')),
     observations_sha256=sha(destination),zeros_retained=True,acoustic_retries=False))
    done+=1
    if done==1 or done%10==0:
     status=dict(split=args.split,worker=args.rank,completed_by_worker=done,sample=sample,seconds=round(time.time()-start,1))
     write_json('reports/render-%s-%d.json'%(args.split,args.rank),status);print(json.dumps(status),flush=True)
  finally:renderer.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],required=True)
 p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=4)
 p.add_argument('--poses',type=int,default=8);run(p.parse_args())
