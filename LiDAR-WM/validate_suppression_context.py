"""Test whether LS3D's source anchors need a context-aware point-mask decoder."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from lidarwm.engine import build_model
from lidarwm.suppression import choose_candidate
from train_mask_suppression import load_split


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/suppression-physics/best.pt')
    p.add_argument('--output',default='reports/suppression_context.json');args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=build_model(state['config']).cuda().eval()
    model.load_state_dict(state['model'])
    root=Path(state['provenance']['provenance']['world_checkpoint']).parent
    scenes=load_split(root,'val');data_root=Path(state['config']['data']['root'])
    grid=list(itertools.product((0.,.01,.03,.05,.1,.2,.3,1.),(.25,.4,.6,.8,1.2,1.6,2.),(0.,.5,1.)))
    sums=np.zeros((len(grid),4),dtype='float64');sample_iou=np.zeros((len(grid),len(scenes)))
    baseline_iou=[]
    for i,scene in enumerate(scenes):
        path=str(scene['path'])
        z=torch.from_numpy(scene['features']).cuda();c=torch.from_numpy(scene['positions']).cuda()
        q,d=model.refiner(((z-model.feature_mean)/model.feature_std).clamp(-6,6),c)
        q=q.sigmoid().cpu().numpy();d=d.cpu().numpy()
        with np.load(root/(path+'.npz')) as archive:
            bank=np.unpackbits(archive['masks'],axis=1,count=int(archive['points']),bitorder='little').astype(bool)
        with np.load(data_root/path/'lidar.npz') as archive:
            xyz=archive['xyz']/20;ray=archive['ray_id']
            pixel=(ray%64)*1800+ray//64
            target=archive['sound_mask'].astype(bool)
        bank=bank[:,pixel];baseline=bank[0]
        base_inter=(baseline&target).sum();base_union=(baseline|target).sum()
        baseline_iou.append((base_inter+1e-9)/(base_union+1e-9))
        distances={}
        for gi,(margin,radius,movement) in enumerate(grid):
            winner=choose_candidate(q,margin)
            if winner==0:prediction=baseline
            else:
                key=(winner,movement)
                if key not in distances:
                    center=scene['positions'][winner]+movement*d[winner]
                    distances[key]=np.linalg.norm(xyz-center,axis=-1)
                prediction=(baseline|bank[winner])&(distances[key]<=radius/20)
            intersection=(prediction&target).sum();union=(prediction|target).sum()
            sample_iou[gi,i]=(intersection+1e-9)/(union+1e-9)
            sums[gi]+=intersection,union,prediction.sum(),target.sum()
        if i%128==127:print(json.dumps(dict(done=i+1,best_iou=float(sample_iou[:,:i+1].mean(1).max()))),flush=True)
    means=sample_iou.mean(1);best=int(means.argmax());margin,radius,movement=grid[best]
    result=dict(checkpoint=args.checkpoint,baseline_mean_iou=float(np.mean(baseline_iou)),
      validation=dict(mean_iou=float(means[best]),micro_iou=float(sums[best,0]/sums[best,1]),
        precision=float(sums[best,0]/max(sums[best,2],1)),recall=float(sums[best,0]/max(sums[best,3],1))),
      decoder=dict(margin=margin,radius_metres=radius,movement_scale=movement),validation_samples=len(scenes),test_evaluated=False,
      grid=[dict(margin=m,radius_metres=r,movement_scale=d,mean_iou=float(value)) for (m,r,d),value in zip(grid,means)],
      samples=[dict(path=str(s['path']),baseline_iou=float(b),iou=float(value))
        for s,b,value in zip(scenes,baseline_iou,sample_iou[best])])
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('grid','samples')},indent=2),flush=True)


if __name__=='__main__':main()
