from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.data import Dataset,A
from activeview_v3.models import EvidenceAdapter,ObservationWorld,DirectValue,evidence_loss,world_loss
from activeview_v3.evaluation import evaluate_evidence,utilities,action_metrics
import argparse,time,torch,numpy as np

def checkpoint_dir(args):
 return ROOT/'checkpoints'/args.revision if args.revision else ROOT/'checkpoints'

def report_dir(args):
 return ROOT/'reports'/args.revision if args.revision else ROOT/'reports'

def load_evidence(args):
 ck=torch.load(checkpoint_dir(args)/'evidence.pt',map_location='cuda',weights_only=False)
 model=EvidenceAdapter(**ck['config']).cuda();model.load_state_dict(ck['model'])
 return model.eval().requires_grad_(False),ck

def train_evidence(train,val,args):
 folder=checkpoint_dir(args);path=folder/'evidence.pt'
 if (folder/'evidence_complete.json').exists():return
 global_best=-1.;global_epoch=0;variant_results={};history=[];scales=[0.,.1,.25,.5,1.,2.];start=time.time()
 for neural in [False,True]:
  name='context' if neural else 'calibration';seed_all(args.seed)
  model=EvidenceAdapter(neural=neural).cuda()
  if neural:
   initial=torch.load(path,map_location='cuda',weights_only=False)
   model.load_state_dict(initial['model'],strict=False)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.0005 if neural else .005,weight_decay=.03 if neural else .01)
  best=-1.;best_epoch=0
  for epoch in range(1,61):
   model.train();order=np.random.permutation(len(train));losses=[]
   for start_idx in range(0,len(order),32):
    ids=order[start_idx:start_idx+32];batch=train.batch(ids,np.random.randint(A,size=len(ids)))
    loss=evidence_loss(model,batch)
    if not torch.isfinite(loss):raise FloatingPointError('Evidence loss is not finite')
    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
    losses.append(float(loss.detach()))
   entry=dict(variant=name,epoch=epoch,loss=float(np.mean(losses)),seconds=time.time()-start)
   if epoch==1 or epoch%2==0:
    result=evaluate_evidence(model,val,scales=scales)
    scores={s:float(result['iou'][str(s)].mean()) for s in scales};scale=max(scores,key=scores.get)
    entry['validation_expected_iou_by_scale']=scores
    if scores[scale]>best:best=scores[scale];best_epoch=epoch
    if scores[scale]>global_best:
     global_best=scores[scale];global_epoch=epoch
     save_pt(path,dict(model=model.state_dict(),config=model.config,epoch=epoch,scale=scale,selection=global_best,
                       frozen_checkpoints=fingerprints(),embedding_sha256=sha(ROOT/'checkpoints/embedding.npz')))
   history.append(entry);write_json(folder/'evidence_history.json',history);print(json.dumps(dict(stage='evidence',**entry)),flush=True)
   if epoch-best_epoch>=14:break
  variant_results[name]=dict(best_epoch=best_epoch,validation_expected_iou=best)
  save_pt(folder/('evidence-%s-last.pt'%name),dict(model=model.state_dict(),config=model.config,epoch=epoch))
 write_json(folder/'evidence_complete.json',dict(best_epoch=global_epoch,validation_expected_iou=global_best,
   variants=variant_results,test_used=False,singleview_validation_iou=float(val.baseline_iou().mean())))

def train_world(train,val,args):
 name='world-s%d'%args.seed;folder=output_path(checkpoint_dir(args)/name/'placeholder').parent
 if (folder/'complete.json').exists():return
 evidence,eck=load_evidence(args);scale=eck['scale'];world=ObservationWorld().cuda()
 optimizer=torch.optim.AdamW(world.parameters(),lr=.0003,weight_decay=.03)
 result=evaluate_evidence(evidence,val,scales=[scale]);quality=result['iou'][str(scale)]
 best=-1.;best_epoch=0;history=[];start=time.time()
 # Epoch zero is an explicit frozen-observation control, recorded separately.
 frozen=ObservationWorld(correction=False).cuda().eval()
 initial=utilities(frozen,evidence,val,scale=scale)
 control=action_metrics(quality,initial,val.arrays['distance'])
 write_json(report_dir(args)/('frozen_world_validation_s%d.json'%args.seed),dict(**control,test_used=False))
 np.savez(output_path(report_dir(args)/('frozen_world_validation_s%d.npz'%args.seed)),utility=initial,iou=quality)
 for epoch in range(1,41):
  world.train();order=np.random.permutation(len(train));losses=[];metrics=[]
  for start_idx in range(0,len(order),16):
   ids=order[start_idx:start_idx+16];batch=train.batch(ids,np.random.randint(A,size=len(ids)))
   loss,metric=world_loss(world,evidence,batch,posterior_weight=.2,scale=scale)
   if not torch.isfinite(loss):raise FloatingPointError(metric)
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(world.parameters(),2.,error_if_nonfinite=True);optimizer.step()
   losses.append(float(loss.detach()));metrics.append(metric)
  entry=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.time()-start,
             **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
  if epoch==1 or epoch%2==0:
   value=utilities(world,evidence,val,scale=scale)
   report=action_metrics(quality,value,val.arrays['distance']);entry['validation_active_iou']=report['active']
   if report['active']>best:
    best=report['active'];best_epoch=epoch
    save_pt(folder/'world.pt',dict(model=world.state_dict(),config=world.config,epoch=epoch,
       validation=best,evidence_sha256=sha(checkpoint_dir(args)/'evidence.pt'),seed=args.seed))
    write_json(folder/'val_report.json',report);np.savez(folder/'val_predictions.npz',utility=value,iou=quality)
  history.append(entry);write_json(folder/'history.json',history);print(json.dumps(dict(stage=name,**entry)),flush=True)
  if epoch-best_epoch>=10:break
 write_json(folder/'complete.json',dict(name=name,best_epoch=best_epoch,validation_iou=best,test_used=False,
   evidence_sha256=sha(checkpoint_dir(args)/'evidence.pt'),world_sha256=sha(folder/'world.pt'),scale=scale))

def train_direct(train,val,args):
 folder=checkpoint_dir(args)
 if (folder/'direct_complete.json').exists():return
 evidence,eck=load_evidence(args);scale=eck['scale'];model=DirectValue(full_information=bool(args.revision)).cuda()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03)
 tr=evaluate_evidence(evidence,train,scales=[scale])['iou'][str(scale)]
 vy=evaluate_evidence(evidence,val,scales=[scale])['iou'][str(scale)]
 baseline=train.baseline_iou();best=-1.;best_epoch=0;history=[]
 for epoch in range(1,51):
  model.train();order=np.random.permutation(len(train));losses=[]
  for start in range(0,len(order),8):
   ids=order[start:start+8];batch=train.pairs(ids)
   prediction=model(batch,evidence).reshape(-1,A)
   quality=torch.from_numpy(tr[ids]).cuda();gain=quality-torch.from_numpy(baseline[ids]).cuda()[:,None]
   loss=torch.nn.functional.smooth_l1_loss(prediction,gain,beta=.05)
   loss=loss+.15*(-(quality/.08).softmax(-1)*(prediction/.08).log_softmax(-1)).sum(-1).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
   losses.append(float(loss.detach()))
  model.eval();values=[]
  with torch.no_grad():
   for start in range(0,len(val),8):values.append(model(val.pairs(np.arange(start,min(start+8,len(val)))),evidence).reshape(-1,A).cpu().numpy())
  values=np.concatenate(values);report=action_metrics(vy,values,val.arrays['distance'])
  entry=dict(epoch=epoch,loss=float(np.mean(losses)),validation_active_iou=report['active']);history.append(entry)
  if report['active']>best:
   best=report['active'];best_epoch=epoch
   save_pt(folder/'direct.pt',dict(model=model.state_dict(),config=model.config,epoch=epoch,validation=best,evidence_sha256=sha(folder/'evidence.pt')))
   write_json(report_dir(args)/'direct_validation.json',report)
  write_json(folder/'direct_history.json',history)
  if epoch==1 or epoch%5==0:print(json.dumps(dict(stage='direct',**entry)),flush=True)
  if epoch-best_epoch>=12:break
 write_json(folder/'direct_complete.json',dict(best_epoch=best_epoch,validation_iou=best,test_used=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['evidence','world','direct','all'],default='all');p.add_argument('--seed',type=int,default=1337)
 p.add_argument('--revision',default='',help='Nonempty revision writes new checkpoints and reports without mixing frozen runs')
 args=p.parse_args();seed_all(args.seed);train=Dataset('train');val=Dataset('val')
 if args.stage in ['evidence','all']:train_evidence(train,val,args)
 if args.stage in ['world','all']:train_world(train,val,args)
 if args.stage in ['direct','all']:train_direct(train,val,args)
