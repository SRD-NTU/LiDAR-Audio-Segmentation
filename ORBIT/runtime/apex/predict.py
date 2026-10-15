"""Deploy from XYZ and four-channel WAV. Never loads instance/semantic labels or RIRs."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from . import data as D
from .model import APEX
from .run import tensor_inputs


def range_grid(xyz):
    r = np.linalg.norm(xyz, axis=1)
    az = np.mod(np.arctan2(xyz[:,1],xyz[:,0]), 2*np.pi)
    el = np.arcsin(np.clip(xyz[:,2]/r.clip(1e-6), -1, 1))
    col = np.rint(az/(2*np.pi)*1800).astype(np.int64) % 1800
    row = np.rint((el+np.pi/8)/(np.pi/4)*63).astype(np.int64)
    if ((row < 0) | (row >= 64)).any():
        raise ValueError('Input is outside the calibrated 64-beam LiDAR field of view')
    pixels = row*1800+col
    if len(np.unique(pixels)) != len(pixels):
        raise ValueError('More than one return per calibrated ray; ambiguous projection')
    grid = np.zeros((3,64,1800),np.float32)
    valid = np.zeros((1,64,1800),np.float32)
    grid[:,row,col] = xyz.T; valid[:,row,col] = 1
    return grid/20, valid, pixels


@torch.inference_mode()
def raw_inputs(xyz, wave, rate, calibration, device='cuda'):
    if rate != 44100 or wave.ndim != 2 or wave.shape[1] != 4:
        raise ValueError('Expected a 44100 Hz waveform with four physical microphone channels')
    if not np.isfinite(xyz).all() or not np.isfinite(wave).all():
        raise ValueError('Nonfinite observations')
    sys.path.insert(0,str(D.ROOT/'ORBIT/runtime'))
    from lidarwm.sonata_geometry import SonataGeometry
    from lidarwm.foundation import FrozenClap
    grid, valid, pixels = range_grid(xyz)
    encoder = SonataGeometry(checkpoint=str(D.ROOT/'ORBIT/pretrained/sonata/sonata.pth')).to(device).eval()
    encoded = encoder._encode(torch.from_numpy(grid)[None].to(device),
                              torch.from_numpy(valid)[None].to(device), sparse=True)[0]
    features = encoded['features'].half().cpu().numpy()
    lookup = np.full(64*1800, -1, np.int64)
    lookup[encoded['pixels'].cpu().numpy()] = encoded['inverse'].cpu().numpy()
    inverse = lookup[pixels]
    if (inverse < 0).any():
        raise ValueError('Unmapped LiDAR returns')
    del encoder
    if device == 'cuda':
        torch.cuda.empty_cache()
    # Existing local CLAP wrapper resolves its downloaded weights relative to WM.
    old = Path.cwd()
    try:
        os.chdir(D.ROOT/'ORBIT')
        clap = FrozenClap(device)
        emb, _ = clap([wave],sample_rate=rate)
    finally:
        os.chdir(old)
    post = D.acoustic.srp_phat_3d(wave.astype(np.float64),rate,np.asarray(calibration['positions_lidar_m']))
    centre,count = D.geometry_inputs(xyz,features,inverse)
    return dict(features=features,xyz=centre,count=count,acoustic=D.acoustic_features(centre,post),
                audio=emb[0].cpu().numpy(),inverse=inverse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--lidar',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=D.DATA/'metadata/microphone_array.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    ck = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if D.sha(args.calibration) != ck['config']['train_provenance']['calibration_sha256']:
        raise ValueError('Microphone calibration differs from training')
    with np.load(args.lidar,allow_pickle=False) as z:
        xyz = z['xyz'].astype(np.float32)
    wave,rate = sf.read(args.audio,dtype='float32')
    x = raw_inputs(xyz,wave,rate,json.loads(args.calibration.read_text()),args.device)
    m = APEX(expand=ck['expand']).to(args.device).eval()
    m.load_state_dict(ck['model'])
    with torch.inference_mode():
        prob = m(*tensor_inputs(x,args.device))['logits'].sigmoid().cpu().numpy()[x['inverse']]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output, probability=prob, sound_mask=(prob>=ck['threshold']).astype(np.uint8),
                        threshold=ck['threshold'], checkpoint_sha256=D.sha(args.checkpoint))
    print(f'Wrote {len(prob)} point predictions to {args.output}',flush=True)


if __name__ == '__main__':
    main()
