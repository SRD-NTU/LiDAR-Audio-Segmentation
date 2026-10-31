"""TRAIN-fitted linear acoustic coordinates, shared by real and imagined observations."""
from .runtime import ROOT
from .covariance import pack
import numpy as np,torch
class AcousticEmbedding:
 def __init__(self,device='cuda'):
  data=np.load(ROOT/'checkpoints/embedding.npz')
  self.mean=torch.from_numpy(data['mean']).to(device)
  self.basis=torch.from_numpy(data['basis']).to(device)
  self.scale=torch.from_numpy(data['scale']).to(device)
  self.factor=torch.tensor([1.]*4+[2.**.5]*12,device=device)
  self.dimension=len(self.scale)
 def encode_packed(self,x):
  return (((x.float()*self.factor).flatten(-2)-self.mean)@self.basis)/self.scale
 def encode_complex(self,x):return self.encode_packed(pack(x))
 def variance(self,log_variance):
  # Frozen log_variance models total squared Hermitian error per frequency band.
  projection=self.basis.square().reshape(48,16,self.dimension).sum(1)/16/self.scale.square()
  return (log_variance.float().exp()@projection).clamp_min(.01)
