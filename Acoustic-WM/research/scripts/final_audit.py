"""Post hoc feasibility and simulator output audit for shared-menu test data."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import write_json
import numpy as np
COMMON=ROOT/'scopes/common'
manifest=json.loads((COMMON/'manifests/test.json').read_text())
controls={name:json.loads((COMMON/'runs/controls'/f'{name}-1337/test.json').read_text()) for name in ['world','analytic_direct','standard']}
by_sample={name:{r['sample']:r for r in data['samples']} for name,data in controls.items()}
rows=[]
for sample in manifest:
 with np.load(COMMON/'cache/controlled'/(sample+'.npz')) as z:
  foreground=int(z['reference_foreground_points']);oracle=float(np.max(z['iou']));valid=int(z['valid'].sum())
 meta=json.loads((ROOT/'generated/common'/sample/'metadata.json').read_text())
 zero=len(meta['audio'].get('zero_response_channels',[]))
 result={'sample':sample,'scene':sample.split('/')[1],'reference_foreground_points':foreground,'oracle_candidate_iou':oracle,'valid_actions':valid,'zero_response_channels':zero}
 for name,data in controls.items():
  r=by_sample[name][sample];result[name+'_single']=float(r['single_iou']);result[name+'_grouped']=float(r['iou'][data['policies'].index('grouped_active')]);result[name+'_random']=float(r['iou'][data['policies'].index('random')])
 rows.append(result)

def paired(subset,a,b):
 from collections import defaultdict
 groups=defaultdict(list)
 for r in subset:groups[r['scene']].append(r[a]-r[b])
 scenes=sorted(groups)
 if not scenes:return None
 totals=np.asarray([sum(groups[s]) for s in scenes]);counts=np.asarray([len(groups[s]) for s in scenes]);rng=np.random.default_rng(20261024);ix=rng.integers(0,len(scenes),(2000,len(scenes)))
 draws=totals[ix].sum(1)/counts[ix].sum(1)
 return {'mean':float(totals.sum()/counts.sum()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'samples':len(subset),'scenes':len(scenes)}

subsets={'all':rows,'reference_source_visible':[r for r in rows if r['reference_foreground_points']>0],'reference_source_absent':[r for r in rows if r['reference_foreground_points']==0]}
summary={}
for label,subset in subsets.items():
 summary[label]={'samples':len(subset),'zero_response_any':sum(r['zero_response_channels']>0 for r in subset),'zero_response_all_four':sum(r['zero_response_channels']==4 for r in subset),'no_valid_action':sum(r['valid_actions']==0 for r in subset),'oracle_iou_mean':float(np.mean([r['oracle_candidate_iou'] for r in subset])) if subset else None}
 for name in controls:
  summary[label][name+'_grouped_mean']=float(np.mean([r[name+'_grouped'] for r in subset])) if subset else None
  summary[label][name+'_grouped_minus_random']=paired(subset,name+'_grouped',name+'_random')
 summary[label]['world_minus_standard_same_grouped']=paired(subset,'world_grouped','standard_grouped')
 summary[label]['world_minus_analytic_same_grouped']=paired(subset,'world_grouped','analytic_direct_grouped')
write_json(ROOT/'reports/common_feasibility.json',{'status':'post hoc descriptive diagnostic; reference visibility uses withheld ground truth, not a deployment input or model-selection stratum','scope':'source-independent shared-menu test, historically exposed scenes','summary':summary,'samples':rows})
print(json.dumps(summary,indent=2),flush=True)
