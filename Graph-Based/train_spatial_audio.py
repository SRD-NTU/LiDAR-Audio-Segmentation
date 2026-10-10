"""Train phase reliability with source directions as TRAINING targets only.

Direction accuracy is an auxiliary diagnostic, never reported as point IoU.
No source metadata, category, instance ID or listener pose is an inference input.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from graph_audio.spatial_audio import SpatialReliability,direction_grid,grid_field
from prepare import ROOT,DATASET
from train_trees import manifests


def pool(names):
    folder = ROOT/'cache/spatial_audio'
    folder.mkdir(parents=True,exist_ok=True)
    ready = folder/'ready.json'
    shape = (len(names),len(direction_grid()),54)
    if ready.exists() and json.loads(ready.read_text())['names']==names:
        return np.memmap(folder/'fields.mmap',mode='r',dtype=np.float16,shape=shape),np.load(folder/'auxiliary.npz')
    fields = np.memmap(folder/'fields.mmap',mode='w+',dtype=np.float16,shape=shape)
    coherence, targets = [],[]
    for i,name in enumerate(names):
        tables=np.load(ROOT/'cache/phase_lut'/(name+'.npy')).astype(np.float32)
        fields[i]=grid_field(tables).astype(np.float16)
        coherence.append(np.sqrt((tables**2).mean(-1)).ravel())
        metadata=json.loads((DATASET/name/'metadata.json').read_text())
        target=np.asarray(metadata['relative_source']['position_lidar'],np.float32)
        target/=max(float(np.linalg.norm(target)),1e-8)
        target[2]=abs(target[2])
        targets.append(target)
        if (i+1)%128==0 or i+1==len(names):
            print(f'Spatial supervision pool {i+1}/{len(names)}',flush=True)
    fields.flush()
    np.savez(folder/'auxiliary.npz',coherence=np.asarray(coherence,np.float32),targets=np.asarray(targets,np.float32))
    ready.write_text(json.dumps(dict(names=names,shape=shape,metadata_used_only_for_loss=True)))
    return fields,np.load(folder/'auxiliary.npz')


@torch.inference_mode()
def evaluate(model,fields,coherence,targets,ids,uniform=False):
    grid=torch.from_numpy(direction_grid())
    angles,loss=[],[]
    model.eval()
    for start in range(0,len(ids),64):
        batch=ids[start:start+64]
        field=torch.from_numpy(fields[batch].astype(np.float32))
        logits=field.mean(-1)*10 if uniform else model.logits(field,coherence[batch])
        cosine=(grid[logits.argmax(-1)]*targets[batch]).sum(-1).clamp(-1,1)
        angles.extend(torch.rad2deg(torch.acos(cosine)).tolist())
        expected=(grid @ targets[batch].T).T
        soft=(-(1-expected)/(1-np.cos(np.deg2rad(12)))).softmax(-1)
        loss.extend((-(soft*logits.log_softmax(-1)).sum(-1)).tolist())
    return dict(samples=len(ids),mean_angular_error_deg=float(np.mean(angles)),median_angular_error_deg=float(np.median(angles)),
                fraction_under_10deg=float(np.mean(np.asarray(angles)<10)),fraction_under_20deg=float(np.mean(np.asarray(angles)<20)),
                soft_cross_entropy=float(np.mean(loss)),elevation_sign_unidentifiable=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=60)
    args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261007)
    rng=np.random.default_rng(20261007)
    train,val=manifests('train',None,False),manifests('val',None,False)
    names=train+val
    run=ROOT/'runs/spatial-audio-reliability'; run.mkdir(parents=True,exist_ok=True)
    (run/'config.json').write_text(json.dumps(dict(epochs=args.epochs,seed=20261007,
        training_samples=len(train),validation_samples=len(val),inputs=['Observed phase lookup','Observed coherence'],
        metadata_used_only_for_loss_and_evaluation=True),indent=2))
    (run/'train_manifest.json').write_text(json.dumps(train))
    (run/'val_manifest.json').write_text(json.dumps(val))
    sources={str(p.relative_to(ROOT)):p.read_text() for p in
             (ROOT/'train_spatial_audio.py',ROOT/'graph_audio/spatial_audio.py')}
    (run/'source_snapshot.json').write_text(json.dumps(sources))
    (run/'source_hashes.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}))
    fields,auxiliary=pool(names)
    coherence=torch.from_numpy(auxiliary['coherence'].copy())
    targets=torch.from_numpy(auxiliary['targets'].copy())
    grid=torch.from_numpy(direction_grid())
    tr=np.arange(len(train)); va=np.arange(len(train),len(names))
    model=SpatialReliability(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.03)
    baseline=evaluate(model,fields,coherence,targets,va,uniform=True)
    print('Uniform physics:',json.dumps(baseline),flush=True)
    best=np.inf
    for epoch in range(args.epochs):
        model.train(); losses=[]
        order=rng.permutation(tr)
        for start in range(0,len(order),64):
            ids=order[start:start+64]
            field=torch.from_numpy(fields[ids].astype(np.float32))
            expected=(grid @ targets[ids].T).T
            soft=(-(1-expected)/(1-np.cos(np.deg2rad(12)))).softmax(-1)
            logits=model.logits(field,coherence[ids])
            loss=-(soft*logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        validation=evaluate(model,fields,coherence,targets,va)
        print(f'Spatial epoch {epoch}: train CE {np.mean(losses):.4f}; validation {json.dumps(validation)}',flush=True)
        if validation['soft_cross_entropy']<best:
            best=validation['soft_cross_entropy']
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation=validation,
                            training_samples=len(train),input='Observed phase tables and their coherence only',
                            target='Absolute-elevation source direction; training metadata only'),run/'best.pt')
    selected=torch.load(run/'best.pt',weights_only=False)
    result=dict(model='spatial-audio-reliability',split='val',auxiliary_metric_not_iou=True,
                training_samples=len(train),baseline=baseline,validation=selected['validation'],epoch=selected['epoch'],
                checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                note='Coplanar microphones cannot resolve elevation sign. Geometry must resolve that ambiguity. No segmentation gain has been measured by this script.')
    (ROOT/'reports/spatial-audio-reliability.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
