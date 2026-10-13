"""Evaluate audio-estimated range as a candidate-object ranking cue.

TRAIN candidate weights use out-of-scene-fold audio distance estimates. VAL
uses a model fitted on all TRAIN scenes. No source positions enter inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from apex import data as D
from .range_probe import load,fit

WEIGHTS=(0.,.25,.5,1.,2.)


def score(row,distance,weight):
    # Candidate geometry indices 0:3 are the point-count-weighted center / 10.
    center=np.linalg.norm(row['features'][:,:3]*10,axis=1)
    error=np.maximum(np.abs(center-distance)-.5,0)
    return row['baseline']-weight*error


def value(rows,prediction,weight):
    return float(np.mean([r['ious'][score(r,prediction[r['rel']],weight).argmax()] for r in rows]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.model_output.exists():raise FileExistsError(args.output)
    tr,x,y,scene=load('train');va,v,z,vscene=load('val')
    if set(scene)&set(vscene):raise AssertionError('Scene overlap')
    mean=x.mean(0);std=x.std(0).clip(.05)
    x=np.c_[(x-mean)/std,np.ones(len(x))]
    v=np.c_[(v-mean)/std,np.ones(len(v))]
    folds=np.array_split(np.unique(scene),4)
    regularization_grid=[];oof_by_regularization={}
    for regularization in (10.,100.,1000.,10000.):
        oof=np.zeros(len(tr))
        for hold in folds:
            fit_idx=~np.isin(scene,hold);valid=~fit_idx
            coef=fit(x[fit_idx],np.log(y[fit_idx]),regularization)
            oof[valid]=np.exp(x[valid]@coef)
        oof_by_regularization[regularization]=oof
        regularization_grid.append(dict(regularization=regularization,
            train_scene_cv_mae_m=float(np.mean(np.abs(oof-y)))))
    choice=min(regularization_grid,key=lambda row:row['train_scene_cv_mae_m'])
    regularization=choice['regularization']
    train_distance=dict(zip(tr,oof_by_regularization[regularization]))
    coef=fit(x,np.log(y),regularization)
    val_distance=dict(zip(va,np.exp(v@coef)))
    bank_train=torch.load(args.bank/'train.pt',map_location='cpu',weights_only=False)
    bank_val=torch.load(args.bank/'val.pt',map_location='cpu',weights_only=False)
    if any(row['rel'] not in train_distance for row in bank_train):raise ValueError('Train order mismatch')
    if any(row['rel'] not in val_distance for row in bank_val):raise ValueError('Validation order mismatch')
    grid=[dict(weight=w,train_mean_iou=value(bank_train,train_distance,w)) for w in WEIGHTS]
    chosen=max(grid,key=lambda row:(row['train_mean_iou'],-row['weight']))
    weight=chosen['weight'];samples=[]
    for row in bank_val:
        j=int(score(row,val_distance[row['rel']],weight).argmax())
        k=int(row['baseline'].argmax())
        meta=json.loads((D.DATA/row['rel']/'metadata.json').read_text())
        source_distance=float(meta['relative_source']['distance_m'])
        samples.append(dict(rel=row['rel'],iou=float(row['ious'][j]),
                            baseline_iou=float(row['ious'][k]),
                            audio_estimated_distance_m=float(val_distance[row['rel']]),
                            source_distance_m=source_distance,
                            tier='easy' if source_distance<=3 else 'medium' if source_distance<=5 else 'hard'))
    result=dict(method='scene-cross-fitted audio range plus object center',split='val',test_used=False,
                n_train=len(bank_train),n_val=len(bank_val),range_regularization=choice,
                range_train_scene_cv=regularization_grid,
                range_validation_mae_m=float(np.mean([abs(s['audio_estimated_distance_m']-s['source_distance_m']) for s in samples])),
                baseline_range_mae_m=float(np.mean(np.abs(np.full(len(z),np.median(y))-z))),
                train_weight_grid=grid,selected_weight=weight,
                validation_mean_iou=float(np.mean([s['iou'] for s in samples])),
                baseline_mean_iou=float(np.mean([s['baseline_iou'] for s in samples])),
                by_difficulty={name:dict(n=sum(s['tier']==name for s in samples),
                    mean_iou=float(np.mean([s['iou'] for s in samples if s['tier']==name])))
                    for name in ('easy','medium','hard')},
                bank_manifest_sha256=D.sha(args.bank/'COMPLETE.json'),
                source_sha256=D.sha(Path(__file__)),samples=samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.model_output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.model_output,mean=mean,std=std,coef=coef,
                        regularization=np.float64(regularization),weight=np.float64(weight))
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','train_weight_grid','range_train_scene_cv')},indent=2))


if __name__=='__main__':main()
