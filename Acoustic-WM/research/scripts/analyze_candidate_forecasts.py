"""Post hoc candidate and physical-condition diagnostics from saved forecasts.

No model is run. Ground-truth mask IoU selects a diagnostic candidate only after
all forecasts were saved. This script never treats geometric mask versions as
verified physical-instance identities.
"""
from pathlib import Path
import argparse
import json

import numpy as np

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE.parents[1]
METHODS=['repeat_current','analytic_direct','no_assimilation','no_transition','world']


def summary(rows,key):
    if not rows:return None
    scenes=sorted({r['scene'] for r in rows})
    totals=np.asarray([sum(r[key] for r in rows if r['scene']==s) for s in scenes],float)
    counts=np.asarray([sum(r['scene']==s for r in rows) for s in scenes],int)
    rng=np.random.default_rng(20261024)
    ix=rng.integers(0,len(scenes),(2000,len(scenes)))
    draws=totals[ix].sum(1)/counts[ix].sum(1)
    return dict(mean=float(totals.sum()/counts.sum()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                episodes=len(rows),scenes=len(scenes),resampling_unit='scene')


def condition(value,bounds,labels):
    if value is None:return 'unknown'
    for bound,label in zip(bounds,labels):
        if value<=bound:return label
    return labels[-1]


def run(archive,scope,split,output):
    experiment=archive if scope=='historical' else archive/'scopes/common'
    data=PROJECT/'dataset/v2-tetra4-4channel-full' if scope=='historical' else archive/'generated/common'
    samples=json.loads((experiment/'manifests'/f'{split}.json').read_text())
    rows=[];diagnostic=[]
    for sample in samples:
        cache=experiment/'cache/controlled'/(sample+'.npz')
        with np.load(cache,allow_pickle=False) as z:
            q=int(z['queries'][4])
            if q<0 or not bool(z['valid'][q]):continue
            quality=z['iou'].astype(float)
            oracle=int(z['oracle_candidate'])
            errors=z['candidate_mse'][:,q,:].astype(float)
            foreground=int(z['reference_foreground_points'])
            action_distance=float(np.linalg.norm(z['actions'][q,:3]))
            negative=(quality<=.1)&(np.arange(len(quality))!=oracle)
        metadata=json.loads((data/sample/'metadata.json').read_text())
        scene=sample.split('/')[1]
        row=dict(sample=sample,scene=scene,scope=scope,split=split,
                 menu_sha256=metadata.get('menu_sha256'),
                 dry_audio_sha256=metadata['audio'].get('dry_audio_sha256'),
                 dry_audio_id=metadata['audio'].get('dry_audio_id'),
                 source_distance_m=float(metadata['relative_source']['distance_m']),
                 source_points=foreground,
                 zero_response_channels=len(metadata['audio'].get('zero_response_channels',[])),
                 action_distance_m=action_distance,oracle_candidate_iou=float(quality[oracle]),
                 oracle_candidate_index=oracle,grouped_action_index=q,
                 world_error=float(errors[4,oracle]),analytic_error=float(errors[1,oracle]),
                 analytic_minus_world=float(errors[1,oracle]-errors[4,oracle]))
        row['distance_bin']=condition(row['source_distance_m'],[3.,5.,8.,float('inf')],
                                      ['<=3m','3-5m','5-8m','>8m'])
        row['action_bin']=condition(action_distance,[1.,2.,3.,float('inf')],
                                     ['<=1m','1-2m','2-3m','>3m'])
        row['visibility']='visible' if foreground else 'absent'
        row['response']='zero_channel' if row['zero_response_channels'] else 'all_nonzero'
        rows.append(row)
        # A meaningful correct/wrong candidate contrast needs a well-covered
        # target. The source-absent episodes in the shared menu are excluded.
        if foreground and quality[oracle]>=.5 and negative.any():
            candidate=dict(row)
            for method_index,method in enumerate(METHODS):
                e=errors[method_index]
                candidate[method+'_oracle_error']=float(e[oracle])
                candidate[method+'_wrong_mean_error']=float(e[negative].mean())
                candidate[method+'_wrong_minus_oracle']=float(e[negative].mean()-e[oracle])
                candidate[method+'_oracle_rank']=int(np.count_nonzero(e<e[oracle])+1)
                candidate[method+'_top_error_candidate_iou']=float(quality[int(e.argmin())])
            diagnostic.append(candidate)
    if not rows:raise ValueError('No saved valid-action forecasts found')
    groups={}
    for key in ['distance_bin','action_bin','visibility','response']:
        groups[key]={label:summary([r for r in rows if r[key]==label],'analytic_minus_world')
                     for label in sorted({r[key] for r in rows})}
    report=dict(scope=scope,split=split,source='saved pre-acquisition candidate forecasts',
                comparison='same grouped action and same oracle-selected reference candidate',
                metric='future-observation-reliability-weighted squared complex covariance Frobenius error; site posterior and valid-action averaging are already applied in candidate_mse',
                oracle_selection='ground truth used after forecasting to select best-IoU reference candidate; deployment does not know this candidate',
                summary=dict(analytic_minus_world=summary(rows,'analytic_minus_world'),
                             world_better_episodes=sum(r['analytic_minus_world']>0 for r in rows),
                             analytic_better_episodes=sum(r['analytic_minus_world']<0 for r in rows),
                             equal_episodes=sum(r['analytic_minus_world']==0 for r in rows)),
                physical_conditions=groups,
                candidate_contrast=dict(eligibility='reference target visible, oracle IoU >= 0.5, and at least one other candidate with IoU <= 0.1',
                                        episodes=len(diagnostic),
                                        methods={m:dict(wrong_minus_oracle=summary(diagnostic,m+'_wrong_minus_oracle'),
                                                        mean_oracle_rank=float(np.mean([r[m+'_oracle_rank'] for r in diagnostic])) if diagnostic else None)
                                                 for m in METHODS}),
                samples=rows,candidate_samples=diagnostic)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),episodes=len(rows),candidate_eligible=len(diagnostic),
                          analytic_minus_world=report['summary']['analytic_minus_world']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=PROJECT/'.importants/Experiments_20261024')
    parser.add_argument('--scope',choices=['historical','common'],default='common')
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--output',type=Path,default=HERE/'reports/candidate_forecast_analysis.json')
    args=parser.parse_args()
    run(args.archive,args.scope,args.split,args.output)
