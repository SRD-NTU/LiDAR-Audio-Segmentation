from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.evaluation import scene_summary
from activeview_v3.provenance import verify_manifest
import argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--split',default='test')
p.add_argument('--protocol',default='config/frozen_evaluation_auditfix1.json');args=p.parse_args()
protocol=json.loads((ROOT/args.protocol).read_text());rows=samples(args.split)
if 'provenance' not in protocol:raise RuntimeError('Use the archived reporter for the legacy protocol')
provenance=verify_manifest(protocol)
methods=protocol['methods'];results={m:[] for m in methods};missing=[]
for row in rows:
 for method in methods:
  path=ROOT/protocol['results_root']/row/(method+'.json')
  if not path.exists():missing.append(str(path));continue
  record=json.loads(path.read_text())
  if record['protocol_sha256']!=protocol['protocol_sha256'] or record['provenance']!=provenance:raise RuntimeError('Protocol mismatch')
  results[method].append(record)
if missing:raise RuntimeError('%d results missing; first: %s'%(len(missing),missing[0]))
arrays={m:np.array([r['iou'] for r in records]) for m,records in results.items()}
arrays['singleview']=np.array([r['singleview_iou'] for r in results['world']])
arrays['random_mean']=np.mean([arrays['random-%d'%i] for i in range(3)],axis=0)
summary={m:scene_summary(v,rows) for m,v in arrays.items()}
paired={m:scene_summary(arrays['world']-arrays[m],rows) for m in ['frozen','direct','random_mean','singleview']}
report=dict(split=args.split,metrics=summary,world_paired_gain=paired,protocol=protocol,
 max_path_m={m:max(r['pose']['path_distance_m'] for r in results[m]) for m in methods},
 all_singleview_checkpoints_unchanged=fingerprints()==protocol['singleview_checkpoints'],provenance=provenance)
write_json('reports/'+protocol['revision']+'/'+args.split+'_results.json',report)
np.savez(output_path('reports/'+protocol['revision']+'/'+args.split+'_results.npz'),rows=rows,**arrays)
print(json.dumps(dict(metrics=summary,world_paired_gain=paired),indent=2))
