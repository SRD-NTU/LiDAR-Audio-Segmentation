"""Reproduce world-model training from separate observation and RIR supervision.

This entry point deliberately writes new weights; it cannot replace the frozen
V0 checkpoint. TRAIN optimization and VAL selection only. No TEST access.
"""
from common import *
import argparse
import numpy as np,torch
from acoustic_wm.world_model.predictive_world import PredictiveAcousticWorld
from acoustic_wm.audio.predictive_features import POOL48
from acoustic_wm.audio.spatial_features import NFFT,KEEP

def load(split):
 assert split in ['train','val']
 rows=json.loads((ROOT/'manifests'/f'{split}.json').read_text());values=[]
 for sample in rows:
  with np.load(ROOT/'cache/controlled'/(sample+'.npz')) as z:
   r={k:z[k] for k in ['reference_world_points','reference_world_features','reference_world_valid','source_for_supervised_world_training','observed','receiver','rotation']}
  rir=np.load(DATA/sample/'rir.npy');multiple=int(np.ceil(rir.shape[1]/NFFT));h=np.fft.rfft(rir,n=multiple*NFFT,axis=-1)[:,::multiple][:,KEEP].T
  q=h[:,:,None]*h[:,None,:].conj();q/=np.maximum((abs(h)**2).sum(-1)[:,None,None],1e-20);r['transfer']=np.einsum('bf,fmn->bmn',POOL48,q).astype(np.complex64);values.append(r)
 tensors={k:torch.as_tensor(np.stack([r[k] for r in values]),device='cuda') for k in values[0]};tensors['observed']=torch.view_as_complex(tensors['observed'].contiguous())
 groups={}
 for i,r in enumerate(rows):groups.setdefault(str(Path(r).parent),[]).append(i)
 partners=[np.array([j for j in groups[str(Path(r).parent)] if j!=i]) for i,r in enumerate(rows)]
 return tensors,partners

def run(seed,epochs):
 seed_all(seed);mic=json.loads((ACTIVE/'assets/sensor/metadata/microphone_array.json').read_text())['positions_lidar_m'];model=PredictiveAcousticWorld(mic).cuda();tr,tp=load('train');va,vp=load('val')
 optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00005)
 folder=ROOT/'runs/retrained_world'/str(seed);folder.mkdir(parents=True,exist_ok=True)
 def step(t,partners,ids,randomize):
  src=t['source_for_supervised_world_training'][ids,None];q,lv,state=model(t['reference_world_points'][ids],t['reference_world_features'][ids],t['reference_world_valid'][ids],src)
  error=(q[:,0]-t['transfer'][ids]).abs().square().sum((-1,-2));obs_error=(q[:,0]-t['observed'][ids]).abs().square().sum((-1,-2));loss=error.mean()+.02*(obs_error*(-lv[:,0]).exp()+lv[:,0]).mean()
  state=model.assimilate(state,q,t['observed'][ids,None]);left=ids;metrics=[error.mean()]
  for horizon in range(2):
   right=np.array([np.random.choice(partners[int(i)]) if randomize else partners[int(i)][horizon%len(partners[int(i)])] for i in ids]);rot=t['rotation'][left].transpose(-1,-2)@t['rotation'][right];translation=torch.einsum('bi,bij->bj',t['receiver'][right]-t['receiver'][left],t['rotation'][left]);action=torch.cat([translation,rot.flatten(1)],-1)[:,None]
   state,src,q,lv=model.imagine(state,src,action);err=(q[:,0]-t['transfer'][right]).abs().square().sum((-1,-2));oe=(q[:,0]-t['observed'][right]).abs().square().sum((-1,-2));loss=loss+.5*err.mean()+.01*(oe*(-lv[:,0]).exp()+lv[:,0]).mean();metrics.append(err.mean());left=right
  return loss,metrics
 best=float('inf');history=[]
 for epoch in range(1,epochs+1):
  model.train();order=np.random.permutation(len(tp));losses=[]
  for i in range(0,len(order),64):
   optimizer.zero_grad(set_to_none=True);loss,_=step(tr,tp,order[i:i+64],True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
  scheduler.step();model.eval();metric=[]
  with torch.no_grad():
   for i in range(0,len(vp),64):
    ids=np.arange(i,min(i+64,len(vp)));_,m=step(va,vp,ids,False);metric.append([float(x) for x in m]+[len(ids)])
  a=np.array(metric);mean=np.average(a[:,:3],weights=a[:,3],axis=0);selection=float(mean[0]+.5*mean[1:].sum());history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_covariance_error=mean.tolist(),selection=selection));write_json(folder/'history.json',history)
  if selection<best:best=selection;save_pt(folder/'best.pt',dict(model={k:v.cpu() for k,v in model.state_dict().items()},epoch=epoch,seed=seed,selection=selection,scope=SCOPE))
  save_pt(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,seed=seed))
  print('WORLD_TRAIN',seed,epoch,selection,flush=True)
 write_json(folder/'complete.json',dict(seed=seed,epochs=epochs,best_validation=best,checkpoint_sha256=sha(folder/'best.pt'),test_used=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1337);p.add_argument('--epochs',type=int,default=120);a=p.parse_args();run(a.seed,a.epochs)
