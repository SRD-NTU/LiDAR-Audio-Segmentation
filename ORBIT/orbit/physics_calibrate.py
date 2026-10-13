"""Calibrate an interpretable acoustic and object-size candidate score on TRAIN."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from apex import data as D

ACOUSTIC=(-.5,0.,.5,1.,2.)
SIZE=(-1.,-.5,0.,.5,1.)
GEOMETRY_SUMMARY_DIM=12
LOG_MEAN_ACOUSTIC_INDEX=GEOMETRY_SUMMARY_DIM+16
LOG_POINT_COUNT_INDEX=10


def score(row,acoustic_weight,size_weight):
    f=row['features']
    # Ranking features start with 12 geometry values, then 20 acoustic values.
    # Acoustic slot 16 is log mean direction evidence /14; geometry slot 10
    # is log(1 + original point count) /10.
    return (row['baseline']+acoustic_weight*f[:,LOG_MEAN_ACOUSTIC_INDEX]*14
            +size_weight*f[:,LOG_POINT_COUNT_INDEX]*10)


def value(rows,a,b):
    return float(np.mean([r['ious'][score(r,a,b).argmax()] for r in rows]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tr=torch.load(args.bank/'train.pt',map_location='cpu',weights_only=False)
    va=torch.load(args.bank/'val.pt',map_location='cpu',weights_only=False)
    grid=[dict(acoustic=a,size=b,train_mean_iou=value(tr,a,b)) for a in ACOUSTIC for b in SIZE]
    # On tied TRAIN means choose the smallest correction to the frozen baseline.
    winner=max(grid,key=lambda r:(r['train_mean_iou'],-abs(r['acoustic'])-abs(r['size'])))
    a,b=winner['acoustic'],winner['size']
    samples=[]
    for r in va:
        j=int(score(r,a,b).argmax());k=int(r['baseline'].argmax())
        meta=json.loads((D.DATA/r['rel']/'metadata.json').read_text())
        dist=float(meta['relative_source']['distance_m'])
        samples.append(dict(rel=r['rel'],iou=float(r['ious'][j]),baseline_iou=float(r['ious'][k]),
                            oracle_iou=float(r['ious'].max()),
                            tier='easy' if dist<=3 else 'medium' if dist<=5 else 'hard'))
    result=dict(method='train calibrated acoustic and size candidate score',split='val',test_used=False,
                n_train=len(tr),n_val=len(va),train_grid=grid,selected=winner,
                validation_mean_iou=float(np.mean([r['iou'] for r in samples])),
                baseline_mean_iou=float(np.mean([r['baseline_iou'] for r in samples])),
                by_difficulty={name:dict(n=sum(r['tier']==name for r in samples),
                    mean_iou=float(np.mean([r['iou'] for r in samples if r['tier']==name])))
                    for name in ('easy','medium','hard')},
                source_sha256=D.sha(Path(__file__)),bank_manifest_sha256=D.sha(args.bank/'COMPLETE.json'),
                samples=samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','train_grid')},indent=2))


if __name__=='__main__':main()
