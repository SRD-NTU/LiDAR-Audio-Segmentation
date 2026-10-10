"""Validation selection for square-array observation ensembling, never test."""
import argparse
import json
import copy
import hashlib
import io
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lidarwm.data import AudioLidarDataset,collate_samples
from lidarwm.engine import build_model,to_device,atomic_checkpoint
from lidarwm.inference import predict_mask_logits
from lidarwm.metrics import BinarySegmentationMetrics


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='reports/tta.json')
    p.add_argument('--freeze-output',help='Save the exact evaluated weights and ensemble threshold for inference')
    args=p.parse_args()
    if args.freeze_output and Path(args.freeze_output).exists():raise FileExistsError(args.freeze_output)
    payload=Path(args.checkpoint).read_bytes()
    ckpt=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);cfg=ckpt['config']
    digest=hashlib.sha256(payload).hexdigest();del payload
    model=build_model(cfg).cuda().eval();model.load_state_dict(ckpt['model'])
    ds=AudioLidarDataset(cfg['data']['root'],'val',feature_root=cfg['data']['feature_root'],antialias=True)
    metrics=BinarySegmentationMetrics(tuple(cfg['evaluation']['thresholds']))
    loader=DataLoader(ds,batch_size=2,num_workers=4,collate_fn=collate_samples)
    for raw in loader:
        b=to_device(raw,torch.device('cuda'))
        extra={key:b[key] for key in ('audio_embedding','text_logits')}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=predict_mask_logits(model,b['lidar'],b['audio'],b['valid'],extra,tta=True)
        metrics.update(logits,b['mask'],b['valid'])
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=digest,epoch=ckpt['epoch'],split='val',samples=len(ds),
      method='mean probabilities over four calibrated square-array rotations',validation=metrics.compute(),
      baseline_validation=ckpt['validation'])
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if args.freeze_output:
        frozen_cfg=copy.deepcopy(cfg);frozen_cfg['evaluation']['tta']=True
        frozen_cfg['training']['resume']=None;frozen_cfg['training'].pop('initialize_from',None)
        atomic_checkpoint(dict(model=ckpt['model'],epoch=ckpt['epoch'],config=frozen_cfg,
          validation=result['validation'],best_iou=result['validation']['mean_iou'],inference_only=True,
          provenance=result),Path(args.freeze_output))


if __name__=='__main__':main()
