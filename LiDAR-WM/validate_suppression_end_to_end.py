"""Recompute geometry/audio -> world -> suppression -> masks on the full val split."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from lidarwm.data import AudioLidarDataset,collate_samples
from lidarwm.engine import build_model,forward_batch,to_device
from lidarwm.metrics import BinarySegmentationMetrics


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='reports/suppression_end_to_end.json');args=p.parse_args()
    payload=Path(args.checkpoint).read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    sha=hashlib.sha256(payload).hexdigest();del payload
    cfg=state['config'];model=build_model(cfg).cuda().eval();model.load_state_dict(state['model'])
    ds=AudioLidarDataset(cfg['data']['root'],'val',feature_root=cfg['data']['feature_root'],antialias=True)
    loader=DataLoader(ds,batch_size=2,num_workers=4,collate_fn=collate_samples)
    baseline=BinarySegmentationMetrics((cfg['model']['baseline_threshold'],))
    integrated=BinarySegmentationMetrics((.5,));rows=[];started=time.time()
    for raw in loader:
        b=to_device(raw,torch.device('cuda'))
        output=forward_batch(model,b)
        if not torch.isfinite(output['mask_logits']).all():raise FloatingPointError('Nonfinite integrated output')
        baseline.update(output['baseline_mask_logits'],b['mask'],b['valid'])
        integrated.update(output['mask_logits'],b['mask'],b['valid'])
        for i,path in enumerate(raw['path']):
            truth=b['mask'][i,0]>.5;valid=b['valid'][i,0]>.5
            old=(output['baseline_mask_logits'][i,0].sigmoid()>=cfg['model']['baseline_threshold'])&valid
            new=(output['mask_logits'][i,0]>0)&valid
            def iou(mask):return float(((mask&truth).sum()+1e-9)/((mask|truth).sum()+1e-9))
            rows.append(dict(path=path,baseline_iou=iou(old),iou=iou(new),changed=bool((old!=new).any())))
        if len(rows)%128==0:print(json.dumps(dict(done=len(rows),elapsed=time.time()-started,validation=integrated.compute())),flush=True)
    scenes=sorted({row['path'].split('/')[1] for row in rows})
    sums=np.asarray([sum(r['iou']-r['baseline_iou'] for r in rows if r['path'].split('/')[1]==s) for s in scenes])
    counts=np.asarray([sum(r['path'].split('/')[1]==s for r in rows) for s in scenes])
    selection=np.random.default_rng(1337).integers(0,len(scenes),(10000,len(scenes)))
    ci=np.quantile(sums[selection].sum(1)/counts[selection].sum(1),[.025,.975]).tolist()
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=sha,validation_samples=len(ds),
      baseline=baseline.compute(),validation=integrated.compute(),
      mean_iou_gain=integrated.compute()['mean_iou']-baseline.compute()['mean_iou'],
      scene_cluster_gain_95ci=ci,changed_samples=sum(r['changed'] for r in rows),
      inputs='XYZ and observed four-channel audio, with waveform-derived CLAP; no annotation inputs',
      test_evaluated=False,samples=rows)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)


if __name__=='__main__':main()
