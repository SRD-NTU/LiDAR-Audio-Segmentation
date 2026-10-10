"""Predict from XYZ and a four-channel WAV without loading annotations."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from lidarwm.data import _read_float_wav
from lidarwm.engine import build_model
from lidarwm.foundation import FrozenClap
from lidarwm.inference import predict_mask_logits

def project_xyz(xyz):
    radius=np.linalg.norm(xyz,axis=1)
    angle=np.mod(np.arctan2(xyz[:,1],xyz[:,0]),2*np.pi)
    elevation=np.arcsin(np.clip(xyz[:,2]/np.maximum(radius,1e-6),-1,1))
    col=np.rint(angle/(2*np.pi)*1800).astype('int64')%1800
    row=np.rint((elevation+np.pi/8)/(np.pi/4)*63).astype('int64')
    if np.any((row<0)|(row>=64)): raise ValueError('Point elevation outside calibrated 64-beam field of view')
    grid=np.zeros((3,64,1800),dtype='float32'); valid=np.zeros((1,64,1800),dtype='float32')
    grid[:,row,col]=xyz.T; valid[:,row,col]=1
    distance=np.linalg.norm(grid,axis=0,keepdims=True)
    lidar=np.concatenate((grid/20,distance/20,grid/np.maximum(distance,1e-5)),axis=0)
    return lidar,valid,row,col

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--lidar',required=True,help='NPZ with xyz, or NPY [N,3]')
    p.add_argument('--audio',required=True,help='Four-channel 44100 Hz WAV')
    p.add_argument('--output',required=True);args=p.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=checkpoint['config']
    if args.lidar.endswith('.npz'):
        with np.load(args.lidar,allow_pickle=False) as archive: xyz=archive['xyz'].astype('float32')
    else: xyz=np.load(args.lidar,allow_pickle=False).astype('float32')
    rate,wave=_read_float_wav(Path(args.audio))
    if rate!=44100: raise ValueError('Expected 44100 Hz audio')
    lidar,valid,row,col=project_xyz(xyz)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(cfg).to(device).eval();model.load_state_dict(checkpoint['model'])
    extra={}
    if cfg['model'].get('architecture') in ('revised','instance_field','suppressed','graph_world','object_world','foundation_world'):
        embedding,scores=FrozenClap(device)([wave],rate)
        extra=dict(audio_embedding=embedding,text_logits=scores)
    stride=cfg['data'].get('audio_stride',3)
    reduced=(resample_poly(wave,1,stride,axis=0) if cfg['data'].get('antialias') else wave[::stride]).T.copy()
    logits=predict_mask_logits(model,torch.from_numpy(lidar)[None].to(device),torch.from_numpy(reduced)[None].to(device),
      torch.from_numpy(valid)[None].to(device),extra,tta=cfg.get('evaluation',{}).get('tta',False))
    probability=logits.sigmoid()[0,0].cpu().numpy()[row,col]
    threshold=checkpoint['validation']['threshold']
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,xyz=xyz,probability=probability,mask=(probability>=threshold).astype('uint8'),threshold=threshold)
    print(f'Wrote {len(xyz)} predictions to {destination}')

if __name__=='__main__': main()
