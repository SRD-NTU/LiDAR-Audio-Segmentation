from .runtime import ROOT,samples,feature_contract
import numpy as np,torch
K=193;H=4;A=8
STATIC={'base','sites','latent','support','residual','initial_silent','iou','null_iou','overlap','valid'}
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
def packed_observation(observed):
 return np.concatenate([np.stack([observed[...,i,i,0] for i in range(4)],-1),
  np.stack([observed[...,i,j,0] for i,j in PAIRS],-1),
  np.stack([observed[...,i,j,1] for i,j in PAIRS],-1)],-1).astype('float32')
class Dataset:
 def __init__(self,split):
  feature_contract()
  self.rows=samples(split);self.split=split;self.arrays={};self.counts=[]
  for i,row in enumerate(self.rows):
   with np.load(ROOT/'data/features'/(row+'.npz')) as z:
    count=len(z['base']);self.counts.append(count)
    for key in z.files:
     x=z[key]
     if key in ['base','sites','latent','support','residual','iou']:shape=(K,)+x.shape[1:]
     elif key=='overlap':shape=(K,K)
     elif key in ['mu','var']:shape=(A,K)+x.shape[2:]
     else:shape=x.shape
     if key not in self.arrays:self.arrays[key]=np.zeros((len(self.rows),)+shape,dtype=x.dtype)
     if key=='overlap':self.arrays[key][i,:count,:count]=x
     elif key in ['mu','var']:self.arrays[key][i,:,:count]=x
     elif key in ['base','sites','latent','support','residual','iou']:self.arrays[key][i,:count]=x
     else:self.arrays[key][i]=x
   # Existing feature archives predate the band-mask correction. Read the
   # authoritative render without altering or silently mixing old archives.
   with np.load(ROOT/'data/rendered'/(row+'.npz')) as render:
    for key,value in [('raw_target',packed_observation(render['observed'])),
                      ('band_reliability',render['weight'].astype('float32'))]:
     if key not in self.arrays:self.arrays[key]=np.zeros((len(self.rows),)+value.shape,dtype='float32')
     self.arrays[key][i]=value
   self.arrays['base'][i,count:]=-1e4
  self.arrays['valid']=np.arange(K)[None,:]<np.array(self.counts)[:,None]
  print('Loaded',split,len(self.rows),'references',round(sum(a.nbytes for a in self.arrays.values())/1e9,2),'GB',flush=True)
 def __len__(self):return len(self.rows)
 def batch(self,ids,actions,device='cuda'):
  result={}
  for key,a in self.arrays.items():
   x=a[ids] if key in STATIC else a[ids,actions]
   result[key]=torch.from_numpy(np.array(x,copy=True)).to(device)
  return result
 def pairs(self,ids,device='cuda'):
  return self.batch(np.repeat(ids,A),np.tile(np.arange(A),len(ids)),device)
 def baseline_iou(self):
  a=self.arrays;r=a['iou'][np.arange(len(self)),a['base'].argmax(-1)]
  return np.where(a['initial_silent'],a['null_iou'],r)
