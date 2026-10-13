"""Test semantic-aware audio selection on frozen predicted masks.

The mask head, source selector and APEX are frozen. One scalar weight is selected
on TRAIN predictions before the full validation split is scored. Annotation
loading occurs after all candidate masks and scores have been computed.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from apex import data as D
from apex.run import tensor_inputs
from .inference import load_models,propose_and_score
from .linear_selector import LinearObjectSelector
from .objects import neighbors

WEIGHTS=(0.,.25,.5,1.,2.,4.)


@torch.inference_mode()
def semantic_scores(x,pred,models,device):
    masks,selector,_,_=models
    f,xyz,_,audio,_=tensor_inputs(x,device)
    knn=torch.as_tensor(neighbors(x['xyz']),device=device,dtype=torch.long)
    point=masks.encode(f,xyz,knn)['semantic'].softmax(-1)[:,:16].cpu().numpy()
    audio_logits=(selector.sound_logits(audio[None]) if isinstance(selector,LinearObjectSelector)
                  else selector.audio_class(selector.sound(audio[None])))
    sound=audio_logits.softmax(-1)[0].cpu().numpy()
    weights=pred['masks'].astype(np.float32)*x['count'][None]
    category=(weights@point)/weights.sum(1,keepdims=True).clip(1.)
    return np.log((category@sound).clip(1e-8)).astype(np.float32)


def gather(split,models,device,train_stride,geometry=None,posterior_dir=None):
    rs,post,emb,_=D.audio_bank(split)
    if posterior_dir:
        post=np.load(posterior_dir/f'adaptive_{split}_posterior.npy')
        if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Posterior mismatch')
    index=np.arange(len(rs))[::train_stride] if split=='train' else np.arange(len(rs))
    rows=[];start=time.time()
    for done,i in enumerate(index):
        rel=rs[i];x=D.load_inputs(rel,post[i],emb[i])
        pred=propose_and_score(x,models,device)
        semantic=semantic_scores(x,pred,models,device)
        base=pred['scores']['hybrid'].astype(np.float32)
        # Annotation access starts only after observation-only scores are fixed.
        target=D.load_targets(rel,x['inverse'],x['count'])
        masks=pred['masks'];positive=target['positive'];count=x['count']
        inter=(masks*positive[None]).sum(1)
        union=(masks*count[None]).sum(1)+positive.sum()-inter
        iou=inter/union.clip(1)
        rows.append(dict(rel=rel,semantic=semantic,base=base,iou=iou,
                         oracle=float(iou.max()),source_seed_hit=bool((positive[pred['seeds']]>0).any())))
        if (done+1)%100==0:
            print(f'{split} {done+1}/{len(index)} {time.time()-start:.1f}s',flush=True)
    return rows


def mean_iou(rows,weight):
    return float(np.mean([r['iou'][np.argmax(r['base']+weight*r['semantic'])] for r in rows]))


def main():
    p=argparse.ArgumentParser()
    for name in ('masks','selector','apex','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--geometry',type=Path);p.add_argument('--spatial',type=Path)
    p.add_argument('--train-stride',type=int,default=3)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.geometry:D.GEOMETRY=args.geometry
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    models=load_models(args.masks,args.selector,args.apex,args.device)
    tr=gather('train',models,args.device,args.train_stride,posterior_dir=args.spatial)
    train_scores={str(w):mean_iou(tr,w) for w in WEIGHTS}
    # Fixed tie break favors the simpler smaller correction.
    chosen=max(WEIGHTS,key=lambda w:(train_scores[str(w)],-w))
    va=gather('val',models,args.device,args.train_stride,posterior_dir=args.spatial)
    samples=[]
    for r in va:
        j=np.argmax(r['base']+chosen*r['semantic']);k=np.argmax(r['base'])
        samples.append(dict(rel=r['rel'],iou=float(r['iou'][j]),baseline_iou=float(r['iou'][k]),
                            oracle_candidate_iou=r['oracle'],source_seed_hit=r['source_seed_hit']))
    by_tier={}
    for name,check in (('easy',lambda d:d<=3),('medium',lambda d:3<d<=5),('hard',lambda d:d>5)):
        values=[]
        for row in samples:
            meta=json.loads((D.DATA/row['rel']/'metadata.json').read_text())
            if check(float(meta['relative_source']['distance_m'])):values.append(row['iou'])
        by_tier[name]=dict(n=len(values),mean_iou=float(np.mean(values)))
    result=dict(split='val',test_used=False,full_validation=True,
                method='audio conditioned point semantic candidate score',
                train_observations=len(tr),train_scene_count=len({r['rel'].split('/')[1] for r in tr}),
                train_score_by_weight=train_scores,selected_weight=chosen,
                validation_mean_iou=float(np.mean([r['iou'] for r in samples])),
                baseline_mean_iou=float(np.mean([r['baseline_iou'] for r in samples])),
                candidate_oracle_mean_iou=float(np.mean([r['oracle_candidate_iou'] for r in samples])),
                by_difficulty=by_tier,
                checkpoints={k:dict(path=str(getattr(args,k)),sha256=D.sha(getattr(args,k)))
                             for k in ('masks','selector','apex')},
                source_sha256=D.sha(Path(__file__)),
                geometry=str(D.GEOMETRY),
                spatial_manifest_sha256=D.sha(args.spatial/'COMPLETE.json') if args.spatial else None,
                samples=samples)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)


if __name__=='__main__':main()
