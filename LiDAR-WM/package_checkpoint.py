"""Freeze an inference-only checkpoint with a validated ensemble threshold."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from lidarwm.engine import atomic_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--report',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    report=json.loads(Path(args.report).read_text())
    if state['epoch']!=report['epoch'] or report['checkpoint']!=args.checkpoint or report['split']!='val':
        raise ValueError('Report does not identify this checkpoint epoch and validation split')
    digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if report.get('checkpoint_sha256',digest)!=digest:raise ValueError('Checkpoint content changed after evaluation')
    if report['method']!='mean probabilities over four calibrated square-array rotations':
        raise ValueError('Unsupported inference method')
    destination=Path(args.output)
    if destination.exists():raise FileExistsError(f'Refusing to overwrite frozen checkpoint: {destination}')
    cfg=copy.deepcopy(state['config']);cfg['evaluation']['tta']=True
    cfg['training']['resume']=None;cfg['training'].pop('initialize_from',None)
    result=dict(model=state['model'],epoch=state['epoch'],config=cfg,validation=report['validation'],
      best_iou=report['validation']['mean_iou'],inference_only=True,
      provenance=dict(source_checkpoint=args.checkpoint,
        source_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),validation_report=report))
    atomic_checkpoint(result,destination);print(f'Frozen inference checkpoint: {destination}')


if __name__=='__main__':main()
