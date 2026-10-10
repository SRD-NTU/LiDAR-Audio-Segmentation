"""Freeze the world model and construct train/val-only mask-candidate caches."""
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
from lidarwm.engine import build_model,to_device,atomic_checkpoint
from lidarwm.suppression import WorldFeatures,make_proposals,proposal_features,overlap_targets


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='cache/suppression');p.add_argument('--max-samples',type=int)
    p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    frozen=output/'world.pt'
    if frozen.exists():
        state=torch.load(frozen,map_location='cpu',weights_only=False)
    else:
        payload=Path(args.checkpoint).read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        state={key:state[key] for key in ('model','config','epoch','validation')}
        state['source_sha256']=hashlib.sha256(payload).hexdigest();state['source_checkpoint']=args.checkpoint
        atomic_checkpoint(state,frozen)
    cfg=state['config'];threshold=float(state['validation']['threshold'])
    feature_dim=None
    if cfg.get('evaluation',{}).get('tta',False):raise ValueError('Choose a single-pass world checkpoint for this first comparison')
    model=build_model(cfg).cuda().eval();model.load_state_dict(state['model']);extract=WorldFeatures(model)
    summary={};started=time.time()
    for split in ('train','val'):
        dataset=AudioLidarDataset(cfg['data']['root'],split,feature_root=cfg['data']['feature_root'],antialias=True,
          max_samples=args.max_samples)
        loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=4,collate_fn=collate_samples)
        rows=[]
        for raw in loader:
            paths=[output/(path+'.npz') for path in raw['path']]
            if all(path.exists() for path in paths):
                for path in paths:
                    with np.load(path) as z:
                        rows.append((float(z['iou'][0]),float(z['iou'].max()),len(z['iou'])))
                        feature_dim=z['features'].shape[-1]
                continue
            b=to_device(raw,torch.device('cuda'))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                o=extract(b['lidar'],b['audio'],b['valid'],audio_embedding=b['audio_embedding'],text_logits=b['text_logits'])
            predicted={key:value.float().cpu().numpy() for key,value in o.items() if key in
              ('mask_logits','geometry_tokens','render_logits','physical_render_logits','doa_logits')}
            predicted['semantic']=o['semantic_logits'].float().softmax(1).cpu().numpy()
            predicted['category']=o['category_logits'].float().softmax(-1).cpu().numpy()
            predicted['probability']=o['mask_logits'].float().sigmoid().cpu().numpy()
            for i,path in enumerate(paths):
                lidar=raw['lidar'][i].numpy();valid=raw['valid'][i,0].numpy()>.5
                prob=predicted['probability'][i,0]
                masks=make_proposals(lidar,valid,prob,threshold)
                features,positions,areas=proposal_features(lidar,valid,prob,masks,predicted['geometry_tokens'][i],
                  predicted['semantic'][i],predicted['category'][i],predicted['render_logits'][i],
                  predicted['physical_render_logits'][i],predicted['doa_logits'][i])
                feature_dim=features.shape[-1]
                # Annotation use begins ONLY here, after proposals and features are fixed.
                iou,intersection,target=overlap_targets(masks,raw['mask'][i,0].numpy()*valid)
                truth=raw['mask'][i,0].numpy().astype(bool)&valid
                xyz=lidar[:3].reshape(3,-1).T
                center=xyz[truth.reshape(-1)].mean(0) if truth.any() else np.zeros(3)
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,features=features,positions=positions,masks=masks,areas=areas,
                  iou=iou,intersection=intersection,target=target,center=center.astype('float32'),
                  points=valid.size,path=raw['path'][i])
                rows.append((float(iou[0]),float(iou.max()),len(iou)))
            if len(rows)%64<args.batch_size:
                print(json.dumps(dict(split=split,done=len(rows),total=len(dataset),elapsed=time.time()-started,
                  baseline_iou=float(np.mean([r[0] for r in rows])),oracle_bank_iou=float(np.mean([r[1] for r in rows])))),flush=True)
        summary[split]=dict(samples=len(rows),baseline_mean_iou=float(np.mean([r[0] for r in rows])),
          oracle_bank_mean_iou=float(np.mean([r[1] for r in rows])),mean_candidates=float(np.mean([r[2] for r in rows])))
    extract.close()
    provenance=dict(world_checkpoint=str(frozen),source_checkpoint=state['source_checkpoint'],
      source_sha256=state['source_sha256'],epoch=state['epoch'],threshold=threshold,summary=summary,
      upstream_commit='06273c07856d356aaf9fa5498a79153d4aef39f8',
      inputs='XYZ, four-channel audio, frozen world predictions only; annotations used only for targets',
      test_evaluated=False,feature_dim=feature_dim)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':main()
