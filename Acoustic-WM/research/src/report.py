"""Auditable summaries, paired scene bootstrap and scoped interpretation."""
from common import *
import numpy as np

def cluster_ci(rows,key,baseline=None,repeats=2000):
 scenes=sorted({r['scene'] for r in rows});totals=[];counts=[]
 for scene in scenes:
  values=[r[key]-(r[baseline] if baseline else 0) for r in rows if r['scene']==scene];totals.append(sum(values));counts.append(len(values))
 rng=np.random.default_rng(20261024);ix=rng.integers(0,len(scenes),(repeats,len(scenes)));boot=np.asarray(totals)[ix].sum(1)/np.asarray(counts)[ix].sum(1)
 return {'mean':float(sum(totals)/sum(counts)),'scene_macro_mean':float(np.mean(np.asarray(totals)/counts)),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':len(rows),'scenes':len(scenes),'resampling_unit':'scene'}

def controls_report():
 results={};paths=sorted((ROOT/'runs/controls').glob('*/test.json'))
 for path in paths:
  result=json.loads(path.read_text());methods={}
  for i,policy in enumerate(result['policies']):
   rows=[dict(scene=r['scene'],value=r['iou'][i],baseline=r['single_iou']) for r in result['samples']]
   methods[policy]={'absolute':cluster_ci(rows,'value'),'gain_over_single':cluster_ci(rows,'value','baseline')}
  comparisons={}
  for target,reference in [('grouped_active','random'),('grouped_active','fixed'),('grouped_active','legacy_active'),('legacy_active','random'),('grouped_active','analytic_active')]:
   ti=result['policies'].index(target);ri=result['policies'].index(reference)
   comparisons[target+' minus '+reference]=cluster_ci([dict(scene=r['scene'],value=r['iou'][ti],baseline=r['iou'][ri]) for r in result['samples']],'value','baseline')
  methods['paired_policy_differences']=comparisons
  results[path.parent.name]=methods
 matched={}
 for seed in [1337,2027,3407]:
  world=ROOT/'runs/controls'/f'world-{seed}/test.json'
  if not world.exists():continue
  left=json.loads(world.read_text());lookup={r['sample']:r for r in left['samples']}
  for control in ['standard','analytic_direct','repeat_current','no_assimilation','no_transition']:
   path=ROOT/'runs/controls'/f'{control}-{seed}/test.json'
   if not path.exists():continue
   right=json.loads(path.read_text());rows=[dict(scene=r['scene'],value=lookup[r['sample']]['iou'][4],baseline=r['iou'][4]) for r in right['samples']]
   matched[f'world minus {control}, seed {seed}, same grouped action']=cluster_ci(rows,'value','baseline')
 results['matched_update_differences']=matched
 write_json(ROOT/'reports/controlled_results.json',results)
 return results

def prediction_report():
 from scipy.stats import spearmanr
 rows=[];duplicate=[];candidate_counts=[]
 for split in ['val','test']:
  manifest=ROOT/'manifests'/f'{split}.json'
  if not manifest.exists():continue
  for sample in json.loads(manifest.read_text()):
   path=ROOT/'cache/controlled'/(sample+'.npz')
   if not path.exists():continue
   with np.load(path) as z:
    valid=z['valid'];ids=np.flatnonzero(valid)
    if not len(ids):continue
    errors=z['candidate_mse'][:,ids];var=z['predicted_variance'][:,ids];best=int(z['oracle_candidate']);quality=z['iou'];wrong=quality<max(.1,float(quality[best])-.2)
    chosen=int(z['queries'][4]);local_index=int(np.flatnonzero(ids==chosen)[0]);qmask=quality>=.5
    for m,name in enumerate(['repeat_current','analytic_direct','no_assimilation','no_transition','world']):
     e=errors[m,:,best];v=var[m,:,best]
     rows.append(dict(sample=sample,scene=sample.split('/')[1],split=split,method=name,oracle_candidate_iou=float(quality[best]),oracle_candidate_mse=float(e.mean()),selected_action_mse=float(e[local_index]),mean_predicted_variance=float(v.mean()),wrong_candidate_mse=float(errors[m][:,wrong].mean()) if wrong.any() else None))
    best_valid=lambda u:int(ids[np.argmax(u[ids])])
    duplicate.append(dict(sample=sample,scene=sample.split('/')[1],legacy_changed=best_valid(z['legacy_utilities'])!=best_valid(z['duplicate_legacy_utilities']),grouped_changed=best_valid(z['group_utilities'][1])!=best_valid(z['duplicate_group_utilities']),grouped_utility_max_difference=float(np.max(np.abs(z['group_utilities'][1]-z['duplicate_group_utilities']))),threshold_queries=[best_valid(u) for u in z['group_utilities']],candidates=len(z['base']),groups=z['group_counts'].tolist()))
 summary={}
 for split in ['val','test']:
  summary[split]={}
  for name in ['repeat_current','analytic_direct','no_assimilation','no_transition','world']:
   subset=[r for r in rows if r['split']==split and r['method']==name]
   if not subset:continue
   errors=np.array([r['oracle_candidate_mse'] for r in subset]);variance=np.array([r['mean_predicted_variance'] for r in subset]);order=np.argsort(variance);bins=[{'count':len(ix),'mean_error':float(errors[ix].mean()),'mean_predicted_variance':float(variance[ix].mean())} for ix in np.array_split(order,min(10,len(order))) if len(ix)]
   correlation=float(spearmanr(errors,variance).statistic) if np.std(variance)>0 and np.std(errors)>0 else None
   if correlation is not None and not np.isfinite(correlation):correlation=None
   summary[split][name]={'oracle_candidate_covariance_error':cluster_ci(subset,'oracle_candidate_mse'),'executed_grouped_action_error':cluster_ci(subset,'selected_action_mse'),'oracle_candidate_iou':float(np.mean([r['oracle_candidate_iou'] for r in subset])),'variance_error_spearman':correlation,'variance_bins':bins,'interpretation':'GT only selects the best reference candidate for diagnostics after prediction; these are not true-source prediction errors or calibrated uncertainty.'}
 write_json(ROOT/'reports/prediction.json',{'summary':summary,'samples':rows})
 write_json(ROOT/'reports/multiplicity.json',{'samples':duplicate,'legacy_duplicate_flip_rate':float(np.mean([r['legacy_changed'] for r in duplicate])) if duplicate else None,'grouped_duplicate_flip_rate':float(np.mean([r['grouped_changed'] for r in duplicate])) if duplicate else None})

def fusion_report():
 reports={}
 for folder in sorted((BASE/'runs/fusion').glob('evaluation-v2-*')):
  records=[json.loads(p.read_text()) for p in (folder/'test').rglob('*.json')]
  if not records:continue
  summary={}
  for scope in ['all_historical_queries','historical_queries_within_3m']:
   subset=records if scope.startswith('all') else [r for r in records if r['locally_valid_3m']]
   summary[scope]={}
   for name in ['reference_only_ranker','reference_evidence_ranker','reference_expansion_ranker','union_ranker','zero_correction']:
    summary[scope][name]={}
    for metric in ['reference','second','two_scan','second_only_voxel']:
     rows=[dict(scene=r['scene'],value=r['methods'][name][metric]['iou'],baseline=r['methods']['reference_expansion_ranker'][metric]['iou']) for r in subset if r['methods'][name][metric] is not None]
     if rows:summary[scope][name][metric]={'absolute':cluster_ci(rows,'value'),'delta_to_reference_expansion':cluster_ci(rows,'value','baseline')}
  reports[folder.name]=summary
 write_json(BASE/'reports/fusion_results.json',reports)

def status():
 expected={s:len(json.loads((ROOT/'manifests'/f'{s}.json').read_text())) if (ROOT/'manifests'/f'{s}.json').exists() else None for s in ['train','val','test']}
 actual={s:len(list((ROOT/'cache/controlled'/s).rglob('*.npz'))) for s in expected}
 report={'scope':SCOPE,'expected_samples':expected,'extracted_samples':actual,'control_runs_complete':len(list((ROOT/'runs/controls').glob('*/complete.json'))),'fusion_runs_complete':len(list((BASE/'runs/fusion').glob('*/complete.json'))),'common_scenes_complete':len(list((BASE/'generated/common/scenes').glob('*/COMPLETE.json'))),'errors':[str(p.relative_to(BASE)) for p in (ROOT/'reports/errors').glob('*.json')],'originals_unchanged':None}
 from protected_source import identity
 protected=json.loads((BASE/'manifests/protected_source.json').read_text())
 observed=identity(protected['root'])
 report['originals_unchanged']=observed['sha256']==protected['sha256']
 report['protected_source_sha256']=observed['sha256']
 report['changed_originals']=[] if report['originals_unchanged'] else [protected['root']]
 write_json(ROOT/'reports/status.json',report);return report
if __name__=='__main__':
 print(json.dumps(status(),indent=2));controls_report();prediction_report()
 if SCOPE=='historical':fusion_report()
