from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
import numpy as np

destination=output_path('checkpoints/embedding.npz')
if not destination.exists():
 # Acoustic observation coordinates use TRAIN observations only, never labels.
 source=ROOT.parent/'Acoustic-WM-ActiveView-v2/cache/train/target.npy'
 reference_weight=np.load(ROOT.parent/'Acoustic-WM-ActiveView-v2/cache/train/weight.npy',mmap_mode='r')
 x=np.load(source,mmap_mode='r')
 factor=np.array([1.]*4+[np.sqrt(2.)]*12,dtype=np.float64)
 count=0;total=np.zeros(768);second=np.zeros((768,768))
 for begin in range(0,len(x),64):
  a=(x[begin:begin+64].astype(np.float64)*factor).reshape(-1,768)
  valid=reference_weight[begin:begin+64].sum(-1).reshape(-1)>0;a=a[valid]
  total+=a.sum(0);second+=a.T@a;count+=len(a)
 mean=total/count;covariance=second/count-np.outer(mean,mean)
 values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1]
 dim=128;basis=vectors[:,order[:dim]];scale=np.sqrt(np.maximum(values[order[:dim]],.0025))
 np.savez(destination,mean=mean.astype('float32'),basis=basis.astype('float32'),scale=scale.astype('float32'))
 write_json('reports/embedding.json',dict(dimension=dim,training_observations=count,
  variance_explained=float(values[order[:dim]].sum()/values.clip(0).sum()),
  source=str(source),source_sha256=sha(source),test_used=False,labels_used=False))
print((ROOT/'reports/embedding.json').read_text())
