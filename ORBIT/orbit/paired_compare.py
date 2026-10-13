"""Paired scene-cluster bootstrap for full-validation foreground IoU results."""
import argparse
import json
from pathlib import Path
import numpy as np


def samples(path,key):
    data=json.loads(path.read_text())
    if data.get('split')=='val' and data.get('full_validation') is False:
        raise ValueError('Refusing partial validation input')
    rows=data.get('samples')
    if not rows and 'metrics' in data:
        raise ValueError('No per-observation records')
    value={r['rel']:float(r[key]) for r in rows}
    if len(value)!=768:raise ValueError('Expected exactly 768 unique validation records')
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-key',required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--candidate-key',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    ref=samples(args.reference,args.reference_key)
    cand=samples(args.candidate,args.candidate_key)
    if set(ref)!=set(cand):raise ValueError('Different validation observations')
    rels=sorted(ref)
    delta=np.array([cand[rel]-ref[rel] for rel in rels])
    scene=np.array([rel.split('/')[1] for rel in rels])
    groups={s:delta[scene==s] for s in sorted(set(scene))}
    names=list(groups)
    rng=np.random.default_rng(726)
    bootstrap=np.array([np.concatenate([groups[s] for s in rng.choice(names,len(names))]).mean()
                        for _ in range(10000)])
    result=dict(n=len(rels),scenes=len(groups),split='val',test_used=False,
                reference_mean_iou=float(np.mean(list(ref.values()))),
                candidate_mean_iou=float(np.mean(list(cand.values()))),
                gain_points=float(100*delta.mean()),
                scene_bootstrap_ci95_gain_points=(100*np.percentile(bootstrap,[2.5,97.5])).tolist(),
                by_scene_gain_points={s:float(100*g.mean()) for s,g in groups.items()},
                reference=str(args.reference),candidate=str(args.candidate),
                note='Descriptive interval: model/checkpoint choices may have used these validation scenes.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
