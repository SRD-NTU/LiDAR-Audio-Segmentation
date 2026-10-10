"""Input-contract check and validation audio negative controls. No test reads."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from lidarwm.data import AudioLidarDataset,collate_samples
from lidarwm.engine import build_model,forward_batch,to_device
from lidarwm.metrics import BinarySegmentationMetrics
from predict import project_xyz


class PairedObservations(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        original=self.base[i];other=self.base[(i+len(self.base)//2)%len(self.base)]
        for key in ('audio','audio_embedding','text_logits'):
            original['other_'+key]=other[key]
        return original


def collate_paired(items):
    output=collate_samples(items)
    for key in ('audio','audio_embedding','text_logits'):
        output['other_'+key]=torch.stack([item['other_'+key] for item in items])
    return output


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',default='reports/input_controls.json');args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=checkpoint['config']
    model=build_model(cfg).cuda().eval();model.load_state_dict(checkpoint['model'])
    ds=AudioLidarDataset(cfg['data']['root'],'val',feature_root=cfg['data']['feature_root'],antialias=True)
    sample_path=Path(cfg['data']['root'])/ds.records[0].relative_path/'lidar.npz'
    # The deployable projector must reproduce training topology using XYZ only.
    with np.load(sample_path) as z:
        _,_,row,col=project_xyz(z['xyz'])
        projection_accuracy=float((col*64+row==z['ray_id']).mean())
    if projection_accuracy!=1:raise ValueError(f'Raw XYZ projection mismatch: {projection_accuracy}')
    threshold=checkpoint['validation']['threshold']
    names=('observed','unrelated_audio','channel_permuted')
    metrics={name:BinarySegmentationMetrics((threshold,)) for name in names}
    loader=DataLoader(PairedObservations(ds),batch_size=2,num_workers=4,collate_fn=collate_paired)
    for raw in loader:
        b=to_device(raw,torch.device('cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for name in names:
                inputs=dict(b)
                if name=='unrelated_audio':
                    for key in ('audio','audio_embedding','text_logits'):inputs[key]=b['other_'+key]
                elif name=='channel_permuted':inputs['audio']=b['audio'][:,[3,1,2,0]]
                output=forward_batch(model,inputs)
                if not torch.isfinite(output['mask_logits']).all():raise FloatingPointError('Nonfinite validation output')
                metrics[name].update(output['mask_logits'],b['mask'],b['valid'])
    result=dict(checkpoint=args.checkpoint,epoch=checkpoint['epoch'],split='val',samples=len(ds),
      fixed_validation_threshold=threshold,xyz_only_projection_accuracy=projection_accuracy,
      controls={name:metric.compute() for name,metric in metrics.items()})
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
