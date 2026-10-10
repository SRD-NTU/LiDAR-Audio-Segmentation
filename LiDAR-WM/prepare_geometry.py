"""Cache XYZ-only pretrained features, never opening any annotation arrays."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from lidarwm.data import read_manifest
from lidarwm.sonata_geometry import SonataGeometry
from predict import project_xyz


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=os.environ.get('AUDIO_LIDAR_DATA_ROOT'))
    p.add_argument('--output',default='cache/sonata-xyz-v2')
    p.add_argument('--split',choices=('train','val'),required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--grid-size',type=float,default=.05)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Run this through Slurm on a GPU')
    model=SonataGeometry(grid_size=args.grid_size).cuda().eval()
    records=read_manifest(args.root,args.split)[args.shard::args.shards]
    if args.limit:records=records[:args.limit]
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    metadata=dict(encoder='facebook/sonata',revision='df99897',encoder_sha256=model.checkpoint_sha256,
        code_commit='18c09ff8d713494f78a8213792262b910977a65d',grid_size=args.grid_size,patch_size=128,
        feature_dim=256,projection_seed=443,adapter_version='deterministic-pooling-v2',feature_dtype='float16',
        inputs=['XYZ','XYZ-derived normals','constant zero color'],
        observed_rgb=False,annotation_inputs=False,test_features_generated=False,
        pretraining_overlap='HM3D pretraining overlap with benchmark scenes is unresolved; not a strict unseen-scene certificate.')
    manifest=destination/'encoder.json'
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:raise ValueError('Existing geometry cache uses a different encoder contract')
    if not manifest.exists():
        temporary=destination/f'encoder.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(metadata,indent=2));os.replace(temporary,manifest)
    start=time.monotonic();completed=0
    for record in records:
        path=destination/(record.relative_path+'.npz')
        if path.exists():continue
        with np.load(Path(args.root)/record.relative_path/'lidar.npz',allow_pickle=False) as archive:
            xyz=archive['xyz'].astype('float32') # deliberately no ray_id or labels
        lidar,valid,_,_=project_xyz(xyz)
        tensors=[torch.from_numpy(a)[None].cuda() for a in (lidar,valid)]
        output=model._encode(*tensors,sparse=True)[0]
        features=output['features'].half().cpu().numpy()
        if not np.isfinite(features).all():raise FloatingPointError(f'Invalid geometry feature: {record.relative_path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,features=features,inverse=output['inverse'].int().cpu().numpy(),
                pixels=output['pixels'].int().cpu().numpy(),xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest())
        os.replace(temporary,path);completed+=1
        if completed%25==0 or completed==1:
            print(json.dumps(dict(split=args.split,shard=args.shard,completed=completed,
                total=len(records),seconds=time.monotonic()-start,voxels=len(features),
                gpu_memory_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
    print(json.dumps(dict(event='complete',split=args.split,shard=args.shard,completed=completed,total=len(records))),flush=True)


if __name__=='__main__':main()
