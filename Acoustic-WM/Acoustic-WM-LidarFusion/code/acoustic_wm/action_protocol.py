"""Deployment-only action validation and geometric hypothesis grouping."""
import numpy as np
import torch


def validate_pose(pose):
 p=np.asarray(pose,np.float64)
 if p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0.,0.,0.,1.],atol=1e-6):
  raise ValueError('receiver_to_world must be a finite 4x4 rigid transform')
 r=p[:3,:3]
 if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(r),1.,atol=1e-4):
  raise ValueError('Receiver rotation must be proper orthogonal')
 return p


def actions_from_poses(reference_pose,destination_poses):
 reference=validate_pose(reference_pose)
 destinations=np.stack([validate_pose(p) for p in destination_poses])
 rotation=reference[:3,:3]
 translation=(destinations[:,:3,3]-reference[:3,3])@rotation
 relative=rotation.T[None]@destinations[:,:3,:3]
 return np.concatenate([translation,relative.reshape(-1,9)],1).astype(np.float32)


def validate_cloud(xyz):
 xyz=np.asarray(xyz,np.float32)
 if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
  raise ValueError('LiDAR xyz must be nonempty finite [N,3] in metres')
 return xyz

def valid_actions(actions,valid=None,costs=None,max_translation=3.0):
 if not np.isfinite(max_translation) or max_translation<0:raise ValueError('Translation bound must be finite and nonnegative')
 a=np.asarray(actions,np.float32)
 if a.ndim!=2 or a.shape[1]!=12 or not np.isfinite(a).all():raise ValueError('Expected finite [actions,12] poses')
 r=a[:,3:].reshape(-1,3,3)
 if not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(r),1,atol=1e-4):raise ValueError('Rotations must be proper orthogonal matrices')
 distance=np.linalg.norm(a[:,:3],axis=1)
 v=np.ones(len(a),bool) if valid is None else np.asarray(valid,bool).copy()
 c=distance.copy() if costs is None else np.asarray(costs,np.float32)
 if v.shape!=(len(a),) or c.shape!=(len(a),) or not np.isfinite(c).all() or (c<0).any():raise ValueError('Invalid validity or costs')
 v &= distance<=max_translation+1e-6
 return a,v,c,distance

def mask_groups(masks,threshold=.9):
 """Deterministic complete-link groups; exact masks count only once within groups.

 Groups are geometric proxies, not claimed physical-instance labels. Every pair
 must exceed the threshold, preventing transitive merging of distinct objects.
 """
 m=np.asarray(masks,bool)
 if m.ndim!=2 or not len(m):raise ValueError('Nonempty candidate mask bank required')
 packed=np.packbits(m,axis=1);keys=[p.tobytes() for p in packed]
 unique=sorted(set(keys));index={k:i for i,k in enumerate(unique)}
 inv=np.array([index[k] for k in keys]);first=np.array([keys.index(k) for k in unique])
 u=m[first].astype(np.float32);inter=u@u.T;area=u.sum(1)
 union=area[:,None]+area[None]-inter
 iou=np.divide(inter,union,out=np.ones_like(inter),where=union>0)
 groups=[];gid=np.empty(len(u),np.int64)
 for i in range(len(u)):
  matches=[j for j,g in enumerate(groups) if np.all(iou[i,g]>=threshold)]
  if matches:j=matches[0];groups[j].append(i)
  else:j=len(groups);groups.append([i])
  gid[i]=j
 counts=np.bincount(inv)
 return gid[inv],counts[inv],{'candidates':len(m),'unique_masks':len(u),'groups':len(groups),'threshold':threshold}

@torch.no_grad()
def grouped_information(base,sites,predicted,log_variance,reliability,groups,duplicates):
 groups=torch.as_tensor(groups,device=base.device);duplicates=torch.as_tensor(duplicates,device=base.device,dtype=base.dtype)
 log_variant=base-duplicates.log();objects=[];variances=[];group_scores=[]
 s=sites.exp()
 candidate_mean=(predicted*s[None,...,None,None,None]).sum(2)
 candidate_variance=(log_variance.exp()*s[None,...,None]).sum(2)
 candidate_variance+=((predicted-candidate_mean[:,:,None]).abs().square().sum((-1,-2))*s[None,...,None]).sum(2)
 for g in groups.unique(sorted=True):
  ids=groups==g;logits=log_variant[ids];w=logits.softmax(0)
  # Log mean exp over unique variants prevents accumulating prior mass by count.
  group_scores.append(torch.logsumexp(logits,0)-(1/duplicates[ids]).sum().log())
  mean=(candidate_mean[:,ids]*w[None,:,None,None,None]).sum(1)
  var=((candidate_variance[:,ids]+(candidate_mean[:,ids]-mean[:,None]).abs().square().sum((-1,-2)))*w[None,:,None]).sum(1)
  objects.append(mean);variances.append(var)
 q=torch.stack(objects,1);v=torch.stack(variances,1);p=torch.stack(group_scores).softmax(0)
 mean=(q*p[None,:,None,None,None]).sum(1)
 between=(q-mean[:,None]).abs().square().sum((-1,-2))
 return (between/v.clamp_min(1e-5)*p[None,:,None]*reliability[None,None]).sum((1,2))
