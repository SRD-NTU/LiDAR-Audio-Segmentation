"""Test whether frozen waveform embeddings carry transferable source range."""
import argparse
import json
from pathlib import Path
import numpy as np
from apex import data as D


def load(split):
    rels,_,embedding,_=D.audio_bank(split)
    distance=np.array([float(json.loads((D.DATA/rel/'metadata.json').read_text())
                             ['relative_source']['distance_m']) for rel in rels])
    scenes=np.array([rel.split('/')[1] for rel in rels])
    return rels,embedding.astype(np.float64),distance,scenes


def fit(x,y,regularization):
    # Dual ridge is less efficient here than 512-d feature-space ridge, but
    # it keeps the fit independent of category/source-object identifiers.
    return np.linalg.solve(x.T@x+regularization*np.eye(x.shape[1]),x.T@y)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tr,x,y,scene=load('train');va,v,z,vscene=load('val')
    if set(scene)&set(vscene):raise AssertionError('Scene overlap')
    mean=x.mean(0);std=x.std(0).clip(.05)
    x=(x-mean)/std;v=(v-mean)/std
    x=np.c_[x,np.ones(len(x))];v=np.c_[v,np.ones(len(v))]
    target=np.log(y)
    unique=np.unique(scene);folds=np.array_split(unique,4)
    grid=[]
    for regularization in (10.,100.,1000.,10000.):
        errors=[]
        for hold in folds:
            fit_idx=~np.isin(scene,hold);valid=~fit_idx
            coef=fit(x[fit_idx],target[fit_idx],regularization)
            pred=np.exp(x[valid]@coef)
            errors.extend(np.abs(pred-y[valid]))
        grid.append(dict(regularization=regularization,scene_cv_mae_m=float(np.mean(errors))))
    choice=min(grid,key=lambda r:r['scene_cv_mae_m'])
    coef=fit(x,target,choice['regularization'])
    pred=np.exp(v@coef)
    baseline=np.full(len(z),float(np.median(y)))
    result=dict(split='val',test_used=False,train_samples=len(tr),val_samples=len(va),
                train_scenes=len(unique),val_scenes=len(np.unique(vscene)),
                method='frozen CLAP embedding ridge range probe',
                selected=choice,train_scene_cv=grid,
                validation_mae_m=float(np.mean(np.abs(pred-z))),
                constant_train_median_mae_m=float(np.mean(np.abs(baseline-z))),
                validation_within_1m=float(np.mean(np.abs(pred-z)<1)),
                source_sha256=D.sha(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
