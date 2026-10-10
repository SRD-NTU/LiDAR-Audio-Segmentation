"""Validation-only calibration of observation-derived graph/phase corrections.

Keep the world model's fine point detail; a shared voxel residual can suppress
inconsistent surfaces without replacing point probabilities by voxel averages.
No semantic/instance annotations construct proposals or corrections.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.special import expit, logit
from prepare import ROOT, DATASET
from train_trees import manifests, refine,refine_weights
from graph_audio.metrics import THRESHOLDS, summarize


def threshold_scores(probability, target, thresholds=THRESHOLDS):
    """Exact confusion counts at every threshold in one histogram pass."""
    bins = np.searchsorted(thresholds, probability, side='right')
    total = np.bincount(bins, minlength=len(thresholds)+1)
    positive = np.bincount(bins, weights=target, minlength=len(thresholds)+1).astype(np.int64)
    tp = np.cumsum(positive[::-1])[::-1][1:]
    selected = np.cumsum(total[::-1])[::-1][1:]
    fp = selected - tp; fn = int(target.sum()) - tp
    return [dict(tp=int(t), fp=int(f), fn=int(n), iou=float(t/max(t+f+n,1)),
                 precision=float(t/max(t+f,1)), recall=float(t/max(t+n,1)),
                 f1=float(2*t/max(2*t+f+n,1))) for t,f,n in zip(tp,fp,fn)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-cache', action='store_true')
    parser.add_argument('--available', action='store_true', help='Explicit subset diagnostic')
    parser.add_argument('--limit',type=int,help='Explicit subset diagnostic, not a full result')
    parser.add_argument('--output',help='Optional report path for integration smoke tests')
    parser.add_argument('--spatial-head',type=Path,help='Optional learned phase-reliability checkpoint')
    args = parser.parse_args()
    spatial_model=None
    spatial_digest=None
    spatial_path=None
    if args.spatial_head:
        import torch
        from graph_audio.spatial_audio import SpatialReliability,scene_field
        torch.set_num_threads(1)
        spatial_path=str(args.spatial_head.resolve().relative_to(ROOT.resolve()))
        spatial_digest=hashlib.sha256(args.spatial_head.read_bytes()).hexdigest()
        spatial_model=SpatialReliability().eval()
        spatial_model.load_state_dict(torch.load(args.spatial_head,map_location='cpu',weights_only=False)['model'])
    names = manifests('val')
    root = ROOT/'cache/teacher'
    if args.wait_cache:
        while not all((root/(n+'.npz')).exists() for n in names):
            print('Waiting for complete validation teacher cache', flush=True)
            time.sleep(20)
    if args.available:
        names = [n for n in names if (root/(n+'.npz')).exists()]
    if args.limit:
        names=names[:args.limit]
    if not names:
        raise ValueError('No teacher observations available')
    with (DATASET/'metadata/samples.csv').open() as handle:
        metadata = {r['sample']:r for r in csv.DictReader(handle)}
    recipes = [(s,a) for s in (0.,.25,.5,.75) for a in (-.5,0.,.5,1.)]
    rows = {(s,a,float(t)):[] for s,a in recipes for t in THRESHOLDS}
    digest = None
    for i,name in enumerate(names,1):
        with np.load(root/(name+'.npz')) as teacher, np.load(ROOT/'cache/graphs'/(name+'.npz')) as graph:
            if digest is None:
                digest = str(teacher['teacher_sha256'])
            if digest != str(teacher['teacher_sha256']) or str(teacher['xyz_sha256']) != str(graph['xyz_sha256']):
                raise ValueError('Teacher/geometry provenance mismatch')
            base = np.clip(teacher['point_probability'],1e-5,1-1e-5)
            node = np.clip(teacher['probability'],1e-5,1-1e-5)
            if not np.isfinite(base).all() or not np.isfinite(node).all():
                raise FloatingPointError(name)
            inverse = graph['inverse']
            if spatial_model is None:
                field = graph['acoustic'].astype(np.float32).mean(1)
            else:
                tables=np.load(ROOT/'cache/phase_lut'/(name+'.npy')).astype(np.float32)
                field=scene_field(graph['acoustic'],tables,spatial_model)
            field = (field-field.mean())/max(float(field.std()),.1)
            weights = refine_weights(graph)
            with np.load(DATASET/name/'lidar.npz') as lidar:
                target = lidar['sound_mask'].astype(np.int64)
            for strength in (0.,.25,.5,.75):
                delta = logit(np.clip(refine(node,graph,strength,weights),1e-5,1-1e-5))-logit(node)
                for alpha in (-.5,0.,.5,1.):
                    probability = expit(logit(base)+(delta+alpha*field)[inverse])
                    for threshold,score in zip(THRESHOLDS,threshold_scores(probability,target)):
                        rows[strength,alpha,float(threshold)].append(dict(sample=name,
                            scene=metadata[name]['scene'],category=metadata[name]['category'],**score))
        if i%64 == 0 or i==len(names):
            print(f'Frozen world graph calibration {i}/{len(names)}',flush=True)
    best = max(rows,key=lambda k: np.mean([r['iou'] for r in rows[k]]))
    selected = rows[best]
    raw = summarize(rows[0.,0.,float(np.float32(.05))])
    categories = defaultdict(list)
    for row in selected:
        categories[row['category']].append(row)
    model_name='frozen-world-learned-spatial-calibration' if spatial_model is not None else 'frozen-world-graph-calibration'
    report = dict(split='val',full_split=len(names)==768,model=model_name,
        teacher_sha256=digest,graph_strength=best[0],phase_alpha=best[1],threshold=best[2],
        metrics=summarize(selected),raw_teacher_at_005=raw,
        calibration_trials=[dict(graph_strength=s,phase_alpha=a,threshold=t,metrics=summarize(v))
                            for (s,a,t),v in rows.items()],
        per_category={c:summarize(v) for c,v in categories.items()},samples_detail=selected,
        trainable_graph=False,test_used_for_selection=False,
        notes='Postprocessing baseline; thresholds and recipes selected on validation only.')
    report_name='teacher-learned-spatial-calibration' if spatial_model is not None else 'teacher-graph-calibration'
    if spatial_model is not None:
        report.update(spatial_head_checkpoint=spatial_path,spatial_head_sha256=spatial_digest,
                      new_spatial_head_trained=True,spatial_head_active=best[1]!=0)
        if best[1]==0:
            report['notes']+=' The best selected phase coefficient is zero: the learned spatial head does not contribute to this mask result.'
    destination = ROOT/'reports'/(report_name+('' if len(names)==768 else '-subset')+'.json')
    if args.output:
        destination=Path(args.output)
    destination.write_text(json.dumps(report,indent=2))
    if len(names)==768:
        run=ROOT/'runs'/model_name
        run.mkdir(parents=True,exist_ok=True)
        correction=dict(type='calibrated_world',teacher_checkpoint_sha256=digest,
            graph_strength=best[0],phase_alpha=best[1],threshold=best[2],training_samples=3264,
            training_provenance='Inherited frozen world trained on 3264 cases; no new neural weights',
            new_graph_trained=False,selection_split='val',validation_samples=768)
        if spatial_model is not None:
            correction.update(spatial_head_checkpoint=spatial_path,spatial_head_sha256=spatial_digest,
                              new_spatial_head_trained=True,
                              spatial_head_active=best[1]!=0,
                              training_provenance='Inherited frozen world; newly trained spatial audio reliability head; no new graph-world weights')
        (run/'correction.json').write_text(json.dumps(correction,indent=2))
        (run/'config.json').write_text(json.dumps(correction,indent=2))
        (run/'validation.json').write_text(json.dumps(report,indent=2))
        (run/'val_manifest.json').write_text(json.dumps(names))
    print(json.dumps({k:v for k,v in report.items() if k not in ('calibration_trials','samples_detail','per_category')},indent=2),flush=True)
    if len(names)==768:
        from summarize import main as refresh_reports
        refresh_reports()


if __name__ == '__main__':
    main()
