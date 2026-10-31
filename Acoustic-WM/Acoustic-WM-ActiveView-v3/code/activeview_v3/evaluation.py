import numpy as np,torch
from .data import A
from .models import imagined_utility

@torch.no_grad()
def evaluate_evidence(model,data,scales=(1.,),batch_size=8):
 model.eval();qualities={str(s):[] for s in scales};choices={str(s):[] for s in scales};post_quality=[]
 for start in range(0,len(data),batch_size):
  ids=np.arange(start,min(start+batch_size,len(data)));batch=data.pairs(ids)
  for scale in scales:
   score=model.scores(batch,scale=scale);choice=score.argmax(-1)
   iou=batch['iou'].gather(1,choice[:,None])[:,0]
   silent=batch['initial_silent']&(~batch['audible'])
   iou=torch.where(silent,batch['null_iou'],iou)
   qualities[str(scale)].append(iou.reshape(-1,A).cpu().numpy())
   choices[str(scale)].append(choice.reshape(-1,A).cpu().numpy())
 return dict(iou={k:np.concatenate(v) for k,v in qualities.items()},
             choices={k:np.concatenate(v) for k,v in choices.items()})

@torch.no_grad()
def utilities(world,evidence,data,scale=1.,draws=16,batch_size=2):
 result=[];world.eval();evidence.eval()
 for start in range(0,len(data),batch_size):
  ids=np.arange(start,min(start+batch_size,len(data)));batch=data.pairs(ids)
  result.append(imagined_utility(world,evidence,batch,draws,scale).reshape(-1,A).cpu().numpy())
 return np.concatenate(result)

def action_metrics(iou,utility,distance,cost=0.):
 choice=(utility-cost*distance).argmax(-1);active=iou[np.arange(len(iou)),choice]
 return dict(active=float(active.mean()),random=float(iou.mean()),oracle=float(iou.max(-1).mean()),
             chosen_action=choice.tolist(),sample_iou=active.tolist())

def scene_summary(values,rows,seed=1337):
 scenes=np.array([r.split('/')[1] for r in rows]);unique=sorted(set(scenes));values=np.asarray(values)
 total=np.array([values[scenes==s].sum() for s in unique]);count=np.array([(scenes==s).sum() for s in unique])
 indices=np.random.default_rng(seed).integers(len(unique),size=(5000,len(unique)))
 bootstrap=total[indices].sum(-1)/count[indices].sum(-1)
 return dict(mean=float(values.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
             scene_macro=float((total/count).mean()),scenes=len(unique),samples=len(values))
