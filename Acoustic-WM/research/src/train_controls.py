"""TRAIN-only matched updates; select epochs on VAL grouped-policy IoU."""
from common import *
import argparse,time
import numpy as np,torch
from torch import nn
from acoustic_wm.models.future_grounding import FutureEvidenceHead
from acoustic_wm.model import ReleasedModel
from extract import VARIANTS,POLICIES

class StandardFusion(nn.Module):
 """Strong learned fusion with full spatial covariance, semantics and pose.

 No future geometry and no learned acoustic forecasts are used. Analytic spatial
 residuals locate candidate supports. The network sees the same reference bank.
 """
 def __init__(self,dimension):
  super().__init__();self.observation=nn.Sequential(nn.Linear(2*48*4*4*2+1024+12,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,64),nn.GELU())
  self.candidate=nn.Sequential(nn.Linear(dimension+64,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
  nn.init.zeros_(self.candidate[-1].weight);nn.init.zeros_(self.candidate[-1].bias)
  self.register_buffer('mean',torch.zeros(dimension));self.register_buffer('std',torch.ones(dimension))
 def forward(self,base,features,observation,valid=None):
  context=self.observation(observation.float()).unsqueeze(-2).expand(*features.shape[:-1],64)
  x=((features.float()-self.mean)/self.std).clamp(-10,10)
  delta=self.candidate(torch.cat([x,context],-1)).squeeze(-1)
  score=base+delta
  return score.masked_fill(~valid,-1e4) if valid is not None else score

def observation_vector(row,queries):
 n=len(queries)
 return np.concatenate([np.broadcast_to(row['observed'].reshape(1,-1),(n,1536)),row['future_observed'][queries].reshape(n,-1),np.broadcast_to(row['current_audio'][None],(n,512)),row['future_audio'][queries],row['actions'][queries]],-1).astype(np.float32)

def load(split,method,device='cuda'):
 rows=json.loads((ROOT/'manifests'/f'{split}.json').read_text());records=[];width=0
 for sample in rows:
  with np.load(ROOT/'cache/controlled'/(sample+'.npz')) as z:
   queries=z['queries'];safe=np.maximum(queries,0);n=len(z['base']);width=max(n,width)
   item={k:z[k] for k in ['base','sites','iou']};item['queries']=queries;item['sample']=sample
   item['initial_silent']=not bool(z['reference_reliability'].any());item['silent_output']=item['initial_silent'] & ((z['weight'][safe].sum(-1)==0) | (queries<0));item['null_iou']=float(z['null_iou'])
   if method=='standard':
    item['feature']=z['standard'][safe];item['observation']=observation_vector(z,safe)
   else:
    item['energy']=z['energy'][VARIANTS.index(method),safe];item['weight']=z['weight'][safe]
   records.append(item)
 n=len(records);base=torch.full((n,width),-1e4,device=device);quality=torch.zeros(n,width,device=device);valid=torch.zeros(n,width,dtype=torch.bool,device=device);query_valid=torch.tensor(np.stack([r['queries']>=0 for r in records]),device=device)
 sites=torch.zeros(n,width,4,device=device)
 if method=='standard':
  dim=records[0]['feature'].shape[-1];feature=torch.zeros(n,6,width,dim,dtype=torch.float16,device=device);observation=torch.zeros(n,6,4108,device=device)
 else:
  energy=torch.zeros(n,6,width,4,48,dtype=torch.float16,device=device);weight=torch.zeros(n,6,48,device=device)
 for i,r in enumerate(records):
  k=len(r['base']);base[i,:k]=torch.as_tensor(r['base'],device=device);sites[i,:k]=torch.as_tensor(r['sites'],device=device);quality[i,:k]=torch.as_tensor(r['iou'],device=device);valid[i,:k]=True
  if method=='standard':feature[i,:,:k]=torch.as_tensor(r['feature'],device=device);observation[i]=torch.as_tensor(r['observation'],device=device)
  else:energy[i,:,:k]=torch.as_tensor(r['energy'],device=device);weight[i]=torch.as_tensor(r['weight'],device=device)
 result=dict(base=base,sites=sites,quality=quality,valid=valid,query_valid=query_valid,initial_silent=torch.tensor([r['initial_silent'] for r in records],device=device),silent_output=torch.tensor(np.stack([r['silent_output'] for r in records]),device=device),null_iou=torch.tensor([r['null_iou'] for r in records],device=device))
 result.update(dict(feature=feature,observation=observation) if method=='standard' else dict(energy=energy,weight=weight))
 return result,rows

def scores(model,data,ids,policy,method):
 base=data['base'][ids];valid=data['valid'][ids]
 if method=='standard':result=model(base,data['feature'][ids,policy],data['observation'][ids,policy],valid)
 else:result=model(base,data['sites'][ids],data['energy'][ids,policy],data['weight'][ids,policy],valid)
 return torch.where(data['query_valid'][ids,policy,None],result,base)

def loss_fn(score,quality,valid):
 target=quality.masked_fill(~valid,-1)
 target=(target*10).masked_fill(~valid,-1e4).softmax(-1)
 return -(target*score.log_softmax(-1)).sum(-1).mean()

@torch.no_grad()
def evaluate(model,data,method):
 model.eval();choices=[]
 for begin in range(0,len(data['base']),32):
  ids=torch.arange(begin,min(begin+32,len(data['base'])),device='cuda')
  choices.append(torch.stack([scores(model,data,ids,p,method).argmax(-1) for p in range(6)],1))
 choice=torch.cat(choices);quality=data['quality'].gather(1,choice);quality=torch.where(data['silent_output'],data['null_iou'][:,None],quality);choice=torch.where(data['silent_output'],torch.full_like(choice,-1),choice);return quality.mean(0).cpu().numpy(),choice.cpu().numpy(),quality.cpu().numpy()

def run(method,seed):
 seed_all(seed);folder=ROOT/'runs/controls'/f'{method}-{seed}';folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'complete.json').exists():
  train,tr_rows=load('train',method);val,va_rows=load('val',method)
  if method=='standard':
   model=StandardFusion(train['feature'].shape[-1]).cuda();x=train['feature'].float();valid=train['valid'][:,None,:,None].expand(-1,6,-1,1)
   count=valid.sum((0,1,2));mean=(x*valid).sum((0,1,2))/count;std=((x-mean).square()*valid).sum((0,1,2))/count
   model.mean.copy_(mean);model.std.copy_(std.sqrt().clamp_min(.01));del x,valid
  else:
   ck=torch.load(ACTIVE/'checkpoints/future_evidence.pt',map_location='cpu',weights_only=False)
   model=FutureEvidenceHead().cuda();model.load_state_dict(ck['model'])
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if method=='standard' else .01,weight_decay=.01 if method=='standard' else 0.)
  best=-1.;best_epoch=0;history=[]
  for epoch in range(1,41):
   model.train();order=torch.randperm(len(tr_rows),device='cuda');losses=[]
   for begin in range(0,len(order),16):
    ids=order[begin:begin+16];policy=torch.randint(0,6,(len(ids),),device='cuda');score=scores(model,train,ids,policy,method)
    loss=loss_fn(score,train['quality'][ids],train['valid'][ids]);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
   metric,_,_=evaluate(model,val,method);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),val_iou_by_policy=metric.tolist()));write_json(folder/'history.json',history)
   if metric[4]>best:
    best=float(metric[4]);best_epoch=epoch;save_pt(folder/'best.pt',{'model':{k:v.cpu() for k,v in model.state_dict().items()},'method':method,'seed':seed,'epoch':epoch,'dimension':train['feature'].shape[-1] if method=='standard' else None})
   print('CONTROL',method,seed,epoch,metric.tolist(),flush=True)
  write_json(folder/'complete.json',dict(val_iou=best,epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(folder/'best.pt'),selection='VAL grouped-active IoU',test_used=False))
  del train,val,model;torch.cuda.empty_cache()
 ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
 model=(StandardFusion(ck['dimension']) if method=='standard' else FutureEvidenceHead()).cuda();model.load_state_dict(ck['model'])
 for split in ['val','test']:
  if (folder/f'{split}.json').exists():continue
  data,rows=load(split,method);metric,choice,quality=evaluate(model,data,method)
  base_choice=data['base'].argmax(-1);base=torch.where(data['initial_silent'],data['null_iou'],data['quality'].gather(1,base_choice[:,None])[:,0]).cpu().tolist()
  write_json(folder/f'{split}.json',dict(method=method,seed=seed,policies=POLICIES,mean_iou=metric.tolist(),single_iou=float(np.mean(base)),samples=[dict(sample=r,scene=r.split('/')[1],choices=choice[i].tolist(),iou=quality[i].tolist(),single_iou=base[i]) for i,r in enumerate(rows)]))
  del data;torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=VARIANTS+['standard'],required=True);p.add_argument('--seed',type=int,default=1337);a=p.parse_args();run(a.method,a.seed)
