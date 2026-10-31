"""Held-out predictive checks; never used to train or select a checkpoint."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.data import Dataset
from activeview_v3.models import EvidenceAdapter,ObservationWorld
from activeview_v3.navigation import rotation
import numpy as np,torch

p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val')
p.add_argument('--revision',default='auditfix1');p.add_argument('--limit',type=int,default=0)
args=p.parse_args();seed_all(1337)
folder=ROOT/'checkpoints'/args.revision
protocol_path=ROOT/'config'/('frozen_evaluation_'+args.revision+'.json')
world_path=ROOT/json.loads(protocol_path.read_text())['world'] if protocol_path.exists() else folder/'world-s1337/world.pt'
with torch.no_grad():
 ck=torch.load(folder/'evidence.pt',map_location='cuda',weights_only=False)
 evidence=EvidenceAdapter(**ck['config']).cuda().eval();evidence.load_state_dict(ck['model'])
 ck=torch.load(world_path,map_location='cuda',weights_only=False)
 world=ObservationWorld(**ck['config']).cuda().eval();world.load_state_dict(ck['model'])
 frozen=ObservationWorld(correction=False).cuda().eval()
data=Dataset(args.split);n=min(len(data),args.limit) if args.limit else len(data)
records=[]
with torch.no_grad():
 for i in range(n):
  row=data.rows[i];meta=json.loads((DATA/row/'metadata.json').read_text())
  listener=meta['listener'];true_source=np.asarray(meta['source']['position_world'])
  local=(true_source-np.asarray(listener['position_world']))@rotation(listener['yaw_rad'])
  k=int(data.arrays['iou'][i].argmax());site=data.arrays['support'][i,k].reshape(4,3)
  distances=np.linalg.norm(site-local,axis=-1);h=int(distances.argmin())
  batch=data.pairs([i]);active=batch['band_reliability'].gt(0).sum(-1)
  entry=dict(sample=row,oracle_iou=float(data.arrays['iou'][i,k]),
             nearest_site_distance_m=float(distances[h]),active_bands=active.cpu().tolist())
  for name,model in [('world',world),('frozen',frozen)]:
   mean,var=model(batch,evidence)
   ll=evidence.component_log_likelihood(batch,mean,var,batch['target'],measured=True)
   component=ll[:,k,h];mixture=torch.logsumexp(batch['sites'][:,k]+ll[:,k],-1)
   raw=batch['raw_target'].float().reshape(mean.shape[0],768)*evidence.packing_factor.repeat(48)
   reference=evidence.embedding_mean+(mean[:,k,h]*evidence.embedding_scale)@evidence.embedding_basis.T
   mask=batch['band_reliability'].gt(0).float().repeat_interleave(16,-1)
   residual=((raw-reference)*mask)@evidence.embedding_basis/evidence.embedding_scale
   retained=(mask@evidence.embedding_basis.square()).clamp_min(1e-8)
   z=residual/(var[:,k,h]*retained).sqrt()
   entry[name]=dict(component_log_score=float(component.mean()),mixture_log_score=float(mixture.mean()),
                    z2=float(z.square().mean()),coverage_1sigma=float(z.abs().le(1).float().mean()),
                    coverage_196sigma=float(z.abs().le(1.96).float().mean()))
  records.append(entry)
  if (i+1)%100==0:print('Diagnosed',i+1,'/',n,flush=True)

summary={}
for subset,chosen in [('all',records),('near_site_and_good_mask',
                       [r for r in records if r['nearest_site_distance_m']<=.5 and r['oracle_iou']>=.5])]:
 summary[subset]=dict(references=len(chosen))
 if chosen:
  for name in ['world','frozen']:
   summary[subset][name]={key:float(np.mean([r[name][key] for r in chosen])) for key in chosen[0][name]}
report=dict(split=args.split,revision=args.revision,checkpoint_sha256=sha(world_path),
            evidence_sha256=sha(folder/'evidence.pt'),
            interpretation='Site is the nearest of four hypotheses to the known source, not an exact true component. Masked Gaussian log scores omit common normalization terms; coverage is approximate because projected dimensions are correlated.',
            summary=summary,records=records)
write_json('reports/'+args.revision+'/world_diagnostics_'+args.split+'.json',report)
print(json.dumps(summary,indent=2),flush=True)
