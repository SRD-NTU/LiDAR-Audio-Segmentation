from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.features import FrozenSingleView
from activeview_v3.embedding import AcousticEmbedding
import argparse,time,numpy as np,torch

def mask_overlap(row):
 masks=np.asarray(row['masks'],dtype='float32')
 count=np.bincount(row['inverse'],minlength=masks.shape[1]).astype('float32')
 intersection=(masks*count)@masks.T;area=(masks*count).sum(-1)
 return intersection/(area[:,None]+area[None,:]-intersection).clip(1)

def run(args):
 seed_all(1337);feature_contract();frozen=FrozenSingleView();embedding=AcousticEmbedding()
 all_rows=samples('train')+samples('val') if args.split=='all' else samples(args.split)
 rows=all_rows[args.rank::args.workers];start=time.time();done=0
 while rows:
  remaining=[];progress=False
  for sample in rows:
   destination=ROOT/'data/features'/(sample+'.npz')
   if destination.exists():continue
   if not (ROOT/'data/rendered'/(sample+'.json')).exists():remaining.append(sample);continue
   with np.load(ARCHIVE/'cache/reference'/(sample+'.npz')) as z:row={k:z[k] for k in z.files}
   with np.load(ROOT/'data/rendered'/(sample+'.npz')) as z:new={k:z[k] for k in z.files}
   with np.load(ARCHIVE/'cache/controlled'/(sample+'.npz')) as z:iou=z['iou'];null_iou=z['null_iou']
   with torch.no_grad():
    feat,prepared=frozen.prepare(row,row['observed'],row['reliability'],new['actions'])
    mu=embedding.encode_complex(torch.stack(prepared['q']))
    var=embedding.variance(torch.stack(prepared['lv']))
    target=embedding.encode_complex(torch.view_as_complex(torch.from_numpy(new['observed']).cuda().contiguous()))
    residual=embedding.encode_packed(feat['current_residual'])-embedding.encode_packed(torch.zeros_like(feat['current_residual']))
   arrays=dict(base=feat['base'].cpu().numpy(),sites=feat['sites'].cpu().numpy(),
    latent=feat['latent'].cpu().numpy(),support=feat['support'].cpu().numpy(),
    residual=residual.cpu().numpy().astype('float16'),actions=new['actions'],distance=new['distance'],
    mu=mu.cpu().numpy().astype('float16'),var=var.cpu().numpy().astype('float16'),
    target=target.cpu().numpy(),audible=(new['weight'].sum(-1)>0),
    initial_silent=feat['initial_silent'].cpu().numpy(),iou=iou,null_iou=null_iou,overlap=mask_overlap(row))
   for key,value in arrays.items():
    if np.asarray(value).dtype.kind=='f' and not np.isfinite(value).all():raise FloatingPointError(sample+': nonfinite '+key)
   destination=output_path(destination);temp=destination.with_suffix('.partial.npz')
   np.savez_compressed(temp,**arrays);temp.replace(destination)
   progress=True;done+=1
   if done==1 or done%25==0:
    status=dict(split=args.split,worker=args.rank,extracted=done,seconds=round(time.time()-start,1))
    write_json('reports/extract-%s-%d.json'%(args.split,args.rank),status);print(json.dumps(status),flush=True)
  rows=remaining
  if rows and not progress:
   if not args.watch:raise RuntimeError('Missing rendered observations for %d references'%len(rows))
   time.sleep(10)
 write_json('reports/extract-%s-%d-complete.json'%(args.split,args.rank),dict(complete=True,samples=len(all_rows[args.rank::args.workers])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','all'],required=True)
 p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--watch',action='store_true')
 run(p.parse_args())
