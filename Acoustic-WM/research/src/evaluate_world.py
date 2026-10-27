"""Held-out supervised future prediction diagnostics, separate from planning.

True source position is provided equally to every prediction variant ONLY in
this diagnostic. Neither action selection nor segmentation reads it.
"""
from common import *
import numpy as np,torch
from extract import forecasts,VARIANTS
from acoustic_wm.model import ReleasedModel

def run():
 seed_all();model=ReleasedModel(ACTIVE,'cuda');result=[]
 for split in ['val','test']:
  for sample in json.loads((ROOT/'manifests'/f'{split}.json').read_text()):
   path=ROOT/'cache/controlled'/(sample+'.npz')
   with np.load(path) as z:
    state={'geometry':tuple(model.tensor(z[k]) for k in ['reference_world_points','reference_world_features','reference_world_valid']),'hypotheses':model.tensor(z['source_for_supervised_world_training'][None,None]),'observed':torch.view_as_complex(model.tensor(z['observed']).contiguous())}
    actions=z['actions'];valid=z['valid']
   with torch.inference_mode():q,lv=forecasts(model,state,actions)
   # Fetch future observation targets after all variant predictions exist.
   with np.load(path) as z:
    target=torch.view_as_complex(model.tensor(z['future_observed']).contiguous());weight=model.tensor(z['weight']);ids=np.flatnonzero(valid)
   if not len(ids):continue
   mse=((q[:,:,0,0]-target[None]).abs().square().sum((-1,-2))*weight[None]).sum(-1).cpu().numpy()
   variance=(lv[:,:,0,0].exp()*weight[None]).sum(-1).cpu().numpy()
   for i,name in enumerate(VARIANTS):result.append(dict(sample=sample,scene=sample.split('/')[1],split=split,method=name,mse=float(mse[i,ids].mean()),predicted_variance=float(variance[i,ids].mean())))
 from report import cluster_ci
 summary={}
 for split in ['val','test']:
  summary[split]={name:cluster_ci([r for r in result if r['split']==split and r['method']==name],'mse') for name in VARIANTS if any(r['split']==split for r in result)}
 write_json(ROOT/'reports/true_source_prediction.json',{'scope':'separate supervised prediction diagnostic; source position supplied equally to all variants, never used by deployed planner','summary':summary,'samples':result})
if __name__=='__main__':run()
