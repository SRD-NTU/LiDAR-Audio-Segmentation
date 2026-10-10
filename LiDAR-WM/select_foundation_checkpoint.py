"""Freeze exact, validation-selected weights; never claim geometry oracle IoU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from lidarwm.engine import atomic_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();candidates=[]
    for stage in ('selection','joint','end-to-end'):
        path=Path(args.prefix+'-'+stage)/'best.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('selection_metric','mean_iou')!='mean_iou':raise ValueError('Oracle geometry checkpoints are NOT deployment candidates')
        candidates.append((float(state['validation']['mean_iou']),stage,path,state))
    score,stage,path,state=max(candidates,key=lambda item:item[0])
    state['selection_provenance']=dict(selected_stage=stage,source=str(path),test_used=False,
        candidates=[dict(stage=s,validation_mean_iou=v) for v,s,_,_ in candidates],
        pretraining_scene_overlap='Unresolved HM3D overlap; do not claim strict unseen-scene foundation transfer.')
    # Final diagnostics MUST extract fresh features from XYZ, not rely on a cache.
    state['config']['data']['geometry_feature_root']=None
    state['config']['evaluation']['tta']=False
    destination=Path(args.output)
    if destination.exists():raise FileExistsError(f'Refusing to replace an existing frozen deployment: {destination}')
    atomic_checkpoint(state,destination)
    print(json.dumps(dict(stage=stage,mean_iou=score,output=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':main()
