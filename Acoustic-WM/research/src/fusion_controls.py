"""Matched-capacity ranker experiments on immutable historical two-scan caches.

All controls share each reference's recorded pre-acquisition query. Results are
conditional on those historical menus, with explicit <=3 m and all-query strata.
"""
from common import *
import argparse,time
import numpy as np,torch
sys.path.insert(0,str(FUSION))
from recovery.features import ConservativeRanker,pad_rows,objective
OLD=ARCHIVE/'fusion_recovery_v2';PRIOR=ARCHIVE/'fusion_training_v1'

def rows(split,kind):
 result=[]
 for sample in json.loads((ROOT/'manifests'/f'{split}.json').read_text()):
  r=torch.load(OLD/'extended'/(sample+'.pt'),map_location='cpu',weights_only=False)
  n=r['reference_count'];r=dict(r)
  if kind.startswith('reference'):
   for key in ['x','base','iou','second_iou','two_scan_iou']:r[key]=r[key][:n]
  if kind=='reference_audio':
   r['x']=r['x'].clone();r['x'][:,245:]=0
  result.append(r)
 return result

def fit(seed,kind):
 seed_all(seed);folder=ROOT/'runs/fusion'/f'{kind}-{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 tr=rows('train',kind);va=rows('val',kind);joined=torch.cat([r['x'] for r in tr]).float();mean=joined.mean(0);std=joined.std(0,unbiased=False).clamp_min(.01);del joined
 train=pad_rows(tr,'cuda');validation=pad_rows(va,'cuda')
 model=ConservativeRanker(mean,std,'mlp',2.).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 def evaluate(data):
  model.eval();values=[]
  with torch.no_grad():
   for start in range(0,len(data[0]),32):
    x,b,q,v=[t[start:start+32] for t in data];scores,_=model(x,b,v);values.extend(q.gather(1,scores.argmax(-1)[:,None])[:,0].cpu().tolist())
  return float(np.mean(values))
 best=evaluate(validation);epoch_best=0;stale=0;history=[]
 def save(epoch):save_pt(folder/'best.pt',dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,epoch=epoch,seed=seed,kind=kind,configuration={'kind':'mlp','bound':2.}))
 save(0)
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(tr),device='cuda');losses=[]
  for start in range(0,len(order),16):
   ids=order[start:start+16];x,b,q,v=[t[ids] for t in train];opt.zero_grad(set_to_none=True);score,delta=model(x,b,v);loss=objective(score,b,q,v,delta)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fusion loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
  metric=evaluate(validation);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),val_iou=metric));write_json(folder/'history.json',history)
  print('FUSION',kind,seed,epoch,metric,flush=True)
  if metric>best+1e-7:best=metric;epoch_best=epoch;stale=0;save(epoch)
  else:stale+=1
  if stale>=6:break
 write_json(folder/'complete.json',dict(val_iou=best,epoch=epoch_best,epochs=epoch,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(folder/'best.pt'),selection_split='val',test_used=False))
 del train,validation,model;torch.cuda.empty_cache()


def mask_metrics(sample,chosen,expanded=True):
 # This is metrics-only: labels cannot enter the learned ranker.
 from scipy.spatial import cKDTree
 original=torch.load(PRIOR/'cache'/(sample+'.pt'),map_location='cpu',weights_only=False)
 ext=torch.load(OLD/'extended'/(sample+'.pt'),map_location='cpu',weights_only=False)
 with np.load(PRIOR/'geometry'/(sample+'.npz')) as z:
  fm=np.unpackbits(z['masks_packed'],axis=1,count=int(z['mask_width'])).astype(bool);rmap=z['reference_map'];smap=z['second_map'];xyz=z['reference_xyz'];sxyz=z['second_xyz']
 action=original['action'].numpy();aligned=sxyz@action[3:].reshape(3,3).T+action[:3]
 cells=np.floor(xyz/.02).astype(np.int64);scells=np.floor(aligned/.02).astype(np.int64)
 def structured(a):return np.ascontiguousarray(a).view(np.dtype((np.void,a.dtype.itemsize*3))).ravel()
 new=~np.isin(structured(scells),np.unique(structured(cells)))
 with np.load(DATA/sample/'lidar.npz') as z:truth=z['sound_mask'].astype(bool)
 with np.load(DATA/original['queried_sample']/'lidar.npz') as z:second=z['sound_mask'].astype(bool)
 n=ext['reference_count']
 def prediction(i,expand):
  if i>=n:return fm[i-n,rmap],fm[i-n,smap]
  with np.load(OLD/'geometry'/(sample+'.npz')) as z:
   m=np.unpackbits(z['reference_masks_packed'],axis=1,count=int(z['reference_width'])).astype(bool)
   ref=m[i,z['reference_inverse']];pred=m[i,z['second_nearest']].copy();covered=z['second_covered'];pred[~covered]=False
   if expand and z['allow_extension'][i]:pred[~covered]=fm[z['matched_fused'][i],smap[~covered]]
  return ref,pred
 def metric(p,t):
  inter=int((p&t).sum());union=int((p|t).sum());return {'iou':inter/union if union else 1.,'intersection':inter,'union':union,'points':len(t),'positive_points':int(t.sum())}
 results={}
 for name,i in chosen.items():
  ref,pred=prediction(int(i),name not in ['reference_only_ranker','reference_evidence_ranker'])
  r=metric(ref,truth);s=metric(pred,second);new_metric=metric(pred[new],second[new]) if new.any() else None
  both={'iou':(r['intersection']+s['intersection'])/(r['union']+s['union']) if r['union']+s['union'] else 1.,'intersection':r['intersection']+s['intersection'],'union':r['union']+s['union']}
  results[name]={'reference':r,'second':s,'two_scan':both,'second_only_voxel':new_metric,'choice':int(i)}
 return dict(sample=sample,scene=sample.split('/')[1],distance_m=float(np.linalg.norm(action[:3])),locally_valid_3m=bool(np.linalg.norm(action[:3])<=3+1e-6),queried_sample=original['queried_sample'],methods=results)

@torch.no_grad()
def evaluate(seed,split='test'):
 seed_all(seed);models={}
 for kind in ['reference_audio','reference','union']:
  folder=ROOT/'runs/fusion'/f'{kind}-{seed}';ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
  model=ConservativeRanker(ck['mean'],ck['std'],'mlp',2.).cuda().eval();model.load_state_dict(ck['model']);models[kind]=model
 samples=json.loads((ROOT/'manifests'/f'{split}.json').read_text())
 for sample in samples:
  path=ROOT/'runs/fusion'/f'evaluation-v2-{seed}'/(sample+'.json')
  if path.exists():continue
  r=torch.load(OLD/'extended'/(sample+'.pt'),map_location='cpu',weights_only=False);n=r['reference_count'];choices={}
  for kind,model in models.items():
   x=r['x'][:n].clone() if kind.startswith('reference') else r['x'];b=r['base'][:n] if kind.startswith('reference') else r['base']
   if kind=='reference_audio':x[:,245:]=0
   score,_=model(x[None].cuda(),b[None].cuda(),torch.ones(1,len(b),dtype=torch.bool,device='cuda'));choice=int(score.argmax())
   if kind=='reference_audio':choices['reference_only_ranker']=choice
   elif kind=='reference':choices['reference_evidence_ranker']=choice;choices['reference_expansion_ranker']=choice
   else:choices['union_ranker']=choice
  choices['zero_correction']=int(r['base'].argmax());report=mask_metrics(sample,choices);write_json(path,report)
 print('FUSION_EVALUATION',seed,split,len(samples),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1337);p.add_argument('--kind',choices=['reference_audio','reference','union','all'],default='all');a=p.parse_args()
 for kind in (['reference_audio','reference','union'] if a.kind=='all' else [a.kind]):fit(a.seed,kind)
 if a.kind=='all':
  for s in ['val','test']:evaluate(a.seed,s)
