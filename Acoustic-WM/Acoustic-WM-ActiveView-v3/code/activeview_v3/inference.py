"""One reference cloud, one initial recording, one continuously planned recording."""
from .runtime import ROOT
from .features import FrozenSingleView
from .embedding import AcousticEmbedding
from .models import EvidenceAdapter,ObservationWorld,DirectValue,imagined_utility
from .navigation import relative_pose
from .covariance import pack
import numpy as np
import torch

def mask_overlap(row):
 masks=np.asarray(row['masks'],dtype='float32')
 count=np.bincount(row['inverse'],minlength=masks.shape[1]).astype('float32')
 intersection=(masks*count)@masks.T;area=(masks*count).sum(-1)
 return intersection/(area[:,None]+area[None,:]-intersection).clip(1)

class Adapter:
 def __init__(self,world_path,device='cuda',direct_path=None,evidence_path=None):
  self.device=device;self.frozen=FrozenSingleView(device);self.embedding=AcousticEmbedding(device)
  ck=torch.load(ROOT/'checkpoints/evidence.pt' if evidence_path is None else evidence_path,map_location=device,weights_only=False)
  self.evidence=EvidenceAdapter(**ck['config']).to(device).eval().requires_grad_(False);self.evidence.load_state_dict(ck['model'])
  self.scale=ck['scale']
  ck=torch.load(world_path,map_location=device,weights_only=False)
  self.world=ObservationWorld(**ck['config']).to(device).eval().requires_grad_(False);self.world.load_state_dict(ck['model'])
  self.control=ObservationWorld(correction=False).to(device).eval().requires_grad_(False)
  self.direct=None
  if direct_path:
   ck=torch.load(direct_path,map_location=device,weights_only=False)
   self.direct=DirectValue(**ck['config']).to(device).eval().requires_grad_(False);self.direct.load_state_dict(ck['model'])

 @torch.no_grad()
 def prepare(self,row,actions,distances):
  # Whitelist inputs: no source metadata or target labels cross this boundary.
  allowed=['xyz','masks','inverse','pointbind_feature','rank_features','audio','observed','reliability','points','features','valid']
  # ReleasedModel has a few additional geometry fields; they are all reference observations.
  allowed+=['world_points','world_features','world_valid','full_point_features','hypotheses','baseline']
  reference={k:row[k] for k in allowed if k in row}
  feat,prepared=self.frozen.prepare(reference,reference['observed'],reference['reliability'],actions)
  embedding=self.embedding
  mu=embedding.encode_complex(torch.stack(prepared['q'])).half().float()
  var=embedding.variance(torch.stack(prepared['lv'])).half().float()
  residual=(embedding.encode_packed(feat['current_residual'])-embedding.encode_packed(torch.zeros_like(feat['current_residual']))).half().float()
  n=len(actions);k=len(feat['base'])
  batch={key:feat[key][None].expand(n,*feat[key].shape) for key in ['base','sites','latent','support','initial_silent']}
  batch.update(mu=mu,var=var,residual=residual[None].expand(n,-1,-1),actions=feat['actions'],
   distance=torch.as_tensor(distances,dtype=torch.float32,device=self.device),
   valid=torch.ones(n,k,dtype=torch.bool,device=self.device),
   overlap=torch.as_tensor(mask_overlap(reference),device=self.device)[None].expand(n,-1,-1))
  # ReleasedModel.current uses inference_mode; copy so an adapter may train on
  # freshly prepared features as well as on serialized feature caches.
  return {key:value.clone() for key,value in batch.items()},prepared

 @torch.no_grad()
 def utility(self,batch,method='world',draws=16):
  if method=='direct':
   if self.direct is None:raise ValueError('Direct baseline checkpoint was not loaded')
   return self.direct(batch,self.evidence)
  return imagined_utility(self.control if method=='frozen' else self.world,self.evidence,batch,draws,self.scale)

 @torch.no_grad()
 def update(self,batch,prepared,index,observed,reliability):
  one={k:v[index:index+1] for k,v in batch.items()}
  obs=torch.as_tensor(np.asarray(observed,np.float32),device=self.device)
  one['target']=self.embedding.encode_complex(torch.view_as_complex(obs.contiguous()))[None]
  one['raw_target']=pack(torch.view_as_complex(obs.contiguous()))[None]
  one['band_reliability']=torch.as_tensor(np.asarray(reliability,np.float32),device=self.device)[None]
  one['audible']=torch.tensor([np.asarray(reliability).sum()>0],device=self.device)
  scores=self.evidence.scores(one,scale=self.scale)[0];chosen=int(scores.argmax())
  silent=bool(one['initial_silent'][0]) and not bool(one['audible'][0])
  row=prepared['row'];mask=np.zeros(len(row['inverse']),dtype=bool) if silent else np.asarray(row['masks'][chosen],bool)[np.asarray(row['inverse'],np.int64)]
  return dict(mask=mask,candidate=-1 if silent else chosen,scores=scores.cpu().numpy())

def plan(adapter,row,listener,space,method='world',seed=1337,initial=32,rounds=2,population=16,elite=6,draws=16):
 """CEM in floor x/z and circular yaw. space only exposes geometry-based sampling."""
 rng=np.random.default_rng(seed)
 if method=='random':
  poses=space.sample(rng,1);values=np.zeros(1,dtype=float)
 else:
  poses=space.sample(rng,initial);values=None
  for iteration in range(rounds+1):
   fresh=poses if values is None else poses[len(values):]
   actions=np.stack([relative_pose(np.asarray(listener['position_world']),listener['yaw_rad'],p['position'],p['yaw']) for p in fresh])
   batch,_=adapter.prepare(row,actions,[p['path_distance_m'] for p in fresh]);chunks=[]
   for start in range(0,len(fresh),8):
    chunks.append(adapter.utility({k:v[start:start+8] for k,v in batch.items()},method,draws).cpu().numpy())
   new=np.concatenate(chunks);values=new if values is None else np.r_[values,new]
   if not np.isfinite(values).all():raise FloatingPointError('Non-finite planning utility')
   if iteration==rounds:break
   selected=np.argsort(values)[-min(elite,len(values)):]
   x=np.array([[poses[i]['floor'][0],poses[i]['floor'][2],poses[i]['yaw']] for i in selected])
   yaw=np.arctan2(np.sin(x[:,2]).mean(),np.cos(x[:,2]).mean())
   x[:,2]=yaw+np.arctan2(np.sin(x[:,2]-yaw),np.cos(x[:,2]-yaw))
   centre=np.r_[x[:,:2].mean(0),yaw];scale=np.maximum(x.std(0),[.12,.12,.15])
   poses+=space.sample(rng,population,centre=centre,scale=scale)
 chosen=int(values.argmax());pose=poses[chosen]
 action=relative_pose(np.asarray(listener['position_world']),listener['yaw_rad'],pose['position'],pose['yaw'])
 batch,prepared=adapter.prepare(row,action[None],[pose['path_distance_m']])
 return dict(pose=pose,utility=float(values[chosen]),evaluated_poses=len(poses),batch=batch,prepared=prepared,
             search_poses=poses,search_utility=values.tolist())
