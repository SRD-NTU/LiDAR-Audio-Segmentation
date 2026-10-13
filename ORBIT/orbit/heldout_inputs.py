"""Prepare observation-only TEST inputs without opening labels or scoring TEST.

Only the already-frozen TRAIN-fitted Utonia projection and spatial adapter are
used. The final method and checkpoint must still be chosen on validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from apex import data as D
from .spatial import BandSRP,BandReliability
from .utonia_features import UtoniaFeatures,HERE


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--projection',type=Path,required=True)
    p.add_argument('--spatial',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    projection=torch.load(args.projection,map_location=args.device,weights_only=True)
    geometry=UtoniaFeatures(HERE/'pretrained/utonia/utonia.pth',projection,args.device)
    cal=json.loads((D.DATA/'metadata/microphone_array.json').read_text())
    scorer=BandSRP(cal['positions_lidar_m'],device=args.device)
    ck=torch.load(args.spatial,map_location='cpu',weights_only=False)
    if not ck['adaptive']:raise ValueError('Expected adaptive band reliability')
    adapter=BandReliability(True).to(args.device).eval();adapter.load_state_dict(ck['model'])
    rels=D.rels('test');post=[];start=time.time()
    for i,rel in enumerate(rels):
        path=args.output/(rel+'.npz')
        if path.exists():raise FileExistsError(path)
        with np.load(D.DATA/rel/'lidar.npz',allow_pickle=False) as z:
            xyz=z['xyz'].astype(np.float32)
        features,pixels,inverse=geometry.encode(xyz)
        with np.load(D.GEOMETRY/(rel+'.npz'),allow_pickle=False) as z:
            if not np.array_equal(pixels,z['pixels']) or not np.array_equal(inverse,z['inverse']):
                raise AssertionError(f'Calibrated voxel mapping changed: {rel}')
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,features=features.half().cpu().numpy(),pixels=pixels,inverse=inverse,
                 xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest())
        wave,rate=sf.read(D.DATA/rel/'audio.wav',dtype='float32')
        if rate!=44100 or wave.ndim!=2 or wave.shape[1]!=4:raise ValueError(rel)
        scores,coherence=scorer(wave[None])
        logits=adapter(torch.tensor(scores,device=args.device),
                       torch.tensor(coherence,device=args.device))[0]
        posterior=(logits-logits.max()).exp().cpu().numpy().astype(np.float32)
        post.append(posterior)
        if (i+1)%100==0:print(f'heldout observations {i+1}/{len(rels)} {time.time()-start:.1f}s',flush=True)
    array=np.stack(post)
    np.save(args.output/'adaptive_test_posterior.npy',array)
    (args.output/'rels.json').write_text(json.dumps(rels))
    result=dict(split='test',n=len(rels),test_observation_inputs_read=True,
                test_annotations_read=False,test_performance_evaluated=False,
                geometry_projection_sha256=D.sha(args.projection),spatial_checkpoint_sha256=D.sha(args.spatial),
                calibration_sha256=D.sha(D.DATA/'metadata/microphone_array.json'),
                utonia_checkpoint_sha256=D.sha(HERE/'pretrained/utonia/utonia.pth'),
                source_sha256=D.sha(Path(__file__)),posterior_shape=list(array.shape),
                checked_voxel_mappings=len(rels),elapsed_seconds=time.time()-start)
    (args.output/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
