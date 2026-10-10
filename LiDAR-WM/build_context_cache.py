"""Train LS3D to select source anchors by the final context-decoded mask IoU."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from lidarwm.engine import atomic_checkpoint
from lidarwm.suppression import overlap_targets


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='cache/suppression-v1')
    p.add_argument('--output',default='cache/suppression-context')
    p.add_argument('--radius',type=float,default=1.6);args=p.parse_args()
    root=Path(args.cache);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'world.pt').exists():raise FileExistsError('Use a fresh context cache directory')
    world=torch.load(root/'world.pt',map_location='cpu',weights_only=False)
    atomic_checkpoint(world,output/'world.pt')
    provenance=json.loads((root/'provenance.json').read_text());provenance['world_checkpoint']=str(output/'world.pt')
    provenance['context_radius_metres']=args.radius
    data_root=Path(world['config']['data']['root']);summary={};started=time.time()
    for split in ('train','val'):
        rows=[];files=sorted((root/split).glob('*/*/*.npz'))
        for i,path in enumerate(files):
            relative=path.relative_to(root)
            with np.load(path) as z:values={key:z[key].copy() for key in z.files}
            with np.load(data_root/str(values['path'])/'lidar.npz') as z:
                xyz=z['xyz'].astype('float32')/20;ray=z['ray_id'].astype('int64')
                target=z['sound_mask'].astype(bool)
            pixel=(ray%64)*1800+ray//64
            count=int(values['points'])
            masks=np.unpackbits(values['masks'],axis=1,count=count,bitorder='little').astype(bool)
            base=torch.from_numpy(masks[0,pixel]).cuda()
            seed=torch.from_numpy(masks[:,pixel]).cuda()
            positions=torch.from_numpy(values['positions']).cuda()
            # Direct distance evaluation avoids TF32 cancellation for nearby points.
            distance=torch.cdist(positions,torch.from_numpy(xyz).cuda(),compute_mode='donot_use_mm_for_euclid_dist')
            decoded=((base[None]|seed)&(distance<=args.radius/20)).cpu().numpy()
            decoded[0]=masks[0,pixel]  # The exact world-mask fallback remains available.
            new=np.zeros_like(masks);new[:,pixel]=decoded
            packed=np.packbits(new,axis=1,bitorder='little')
            truth=np.zeros(count,dtype=bool);truth[pixel]=target
            iou,intersection,target_count=overlap_targets(packed,truth)
            values.update(masks=packed,iou=iou,intersection=intersection,areas=new.sum(1).astype('int32'),target=target_count)
            destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination,**values)
            rows.append((float(iou[0]),float(iou.max()),len(iou)))
            if i%128==127:print(json.dumps(dict(split=split,done=i+1,total=len(files),elapsed=time.time()-started,
              oracle_iou=float(np.mean([r[1] for r in rows])))),flush=True)
        summary[split]=dict(samples=len(rows),baseline_mean_iou=float(np.mean([r[0] for r in rows])),
          oracle_bank_mean_iou=float(np.mean([r[1] for r in rows])),mean_candidates=float(np.mean([r[2] for r in rows])))
    provenance['summary']=summary;(output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':main()
