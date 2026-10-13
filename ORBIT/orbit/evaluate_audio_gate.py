"""Full observation-only evaluation of an audio-conditioned 3D object query."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from apex import data as D
from apex.run import tensor_inputs
from .audio_gate import AudioSeedGate
from .evaluate import report
from .inference import load_models,propose_and_score
from .objects import neighbors


@torch.inference_mode()
def gate_per_seed(x,models,gate,seeds,device):
    masks,selector,_,_=models
    f,xyz,acoustic,audio,count=tensor_inputs(x,device)
    knn=torch.as_tensor(neighbors(x['xyz']),device=device,dtype=torch.long)
    encoded=masks.encode(f,xyz,knn)
    sound=selector.sound_logits(audio[None]).softmax(-1)[0]
    scores={}
    for start in range(0,len(seeds),8):
        group=seeds[start:start+8]
        si=torch.as_tensor(group,device=device,dtype=torch.long)
        decoded=masks.decode(encoded,xyz,count,si)
        value=gate(encoded,decoded,acoustic,audio,count,si,sound)
        scores.update(zip(group,value.cpu().numpy().tolist()))
    return scores


def candidate_gate_values(x,models,gate,pred,device):
    """Align one learned sound-query score with each observation-only mask."""
    by_seed=gate_per_seed(x,models,gate,pred['seeds'],device)
    values=np.array([by_seed.get(int(origin.split(':')[1]),0.) if origin.startswith('prompt:')
                     else 0. for origin in pred['origins']],np.float32)
    valid=np.array([origin.startswith('prompt:') for origin in pred['origins']])
    if valid.any():values[valid]-=values[valid].mean()
    return values


def main():
    p=argparse.ArgumentParser()
    for name in ('masks','selector','apex','gate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--geometry',type=Path);p.add_argument('--spatial',type=Path)
    p.add_argument('--posterior',type=Path)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--reranker',type=Path)
    p.add_argument('--gate-weight',type=float,default=1.)
    p.add_argument('--gate2',type=Path)
    p.add_argument('--gate2-weight',type=float,default=0.)
    p.add_argument('--control',choices=['matched','shuffled','uniform-spatial'],default='matched')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.split=='test' and (args.geometry is None or args.posterior is None or
                               args.spatial is not None or args.control!='matched'):
        raise ValueError('Final TEST requires explicit frozen geometry and posterior paths')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.geometry:D.GEOMETRY=args.geometry
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    models=load_models(args.masks,args.selector,args.apex,args.device)
    ck=torch.load(args.gate,map_location='cpu',weights_only=False)
    gate=AudioSeedGate().to(args.device).eval();gate.load_state_dict(ck['model'])
    if bool(args.gate2) != bool(args.gate2_weight):
        raise ValueError('gate2 and nonzero gate2-weight must be specified together')
    gate2=None
    if args.gate2:
        ck2=torch.load(args.gate2,map_location='cpu',weights_only=False)
        gate2=AudioSeedGate().to(args.device).eval();gate2.load_state_dict(ck2['model'])
    ranker=None
    if args.reranker:
        from .rerank import Reranker
        checkpoint=torch.load(args.reranker,map_location='cpu',weights_only=False)
        ranker=Reranker(checkpoint['config']['n_features']).to(args.device).eval()
        ranker.load_state_dict(checkpoint['model'])
    rs,post,emb,provenance=D.audio_bank(args.split)
    if args.spatial:
        path=args.spatial/f'adaptive_{args.split}_posterior.npy'
        post=np.load(path)
        if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Posterior mismatch')
        provenance['replacement_posterior_sha256']=D.sha(path)
    if args.posterior:
        post=np.load(args.posterior)
        if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Posterior mismatch')
        provenance['replacement_posterior_sha256']=D.sha(args.posterior)
    rng=np.random.default_rng(1701)
    partners=[int(rng.choice([j for j,s in enumerate(rs) if s.split('/')[1]!=rel.split('/')[1]]))
              for rel in rs]
    rows=[];start=time.time()
    for done,rel in enumerate(rs):
        j=partners[done] if args.control=='shuffled' else done
        posterior=np.ones_like(post[j]) if args.control=='uniform-spatial' else post[j]
        x=D.load_inputs(rel,posterior,emb[j])
        pred=propose_and_score(x,models,args.device)
        values=candidate_gate_values(x,models,gate,pred,args.device)
        if gate2 is not None:
            values=values+args.gate2_weight*candidate_gate_values(x,models,gate2,pred,args.device)
        fused=pred['scores']['hybrid']+args.gate_weight*values
        selector_fused=pred['scores']['selector']+args.gate_weight*values
        gateonly=np.log(pred['quality'].clip(.01))+values
        # Targets are loaded only after all observations and candidate scores.
        targets=D.load_targets(rel,x['inverse'],x['count'])
        masks=pred['masks'];positive=targets['positive'];count=x['count']
        inter=(masks*positive[None]).sum(1)
        union=(masks*count[None]).sum(1)+positive.sum()-inter
        ious=inter/union.clip(1)
        meta=json.loads((D.DATA/rel/'metadata.json').read_text())
        distance=float(meta['relative_source']['distance_m'])
        selected=dict(baseline=int(pred['choices']['hybrid']),fused=int(fused.argmax()),
                      selector_fused=int(selector_fused.argmax()),
                      gate_only=int(gateonly.argmax()))
        if ranker is not None:
            features=np.column_stack([pred['rank_features'],values]).astype(np.float32)
            with torch.inference_mode():
                scores=ranker(torch.as_tensor(features,device=args.device),
                              torch.as_tensor(fused,device=args.device,dtype=torch.float32))
            selected['reranked']=int(scores.argmax())
        rows.append(dict(rel=rel,scene=rel.split('/')[1],category=D.CATS[targets['source']],
                         tier='easy' if distance<=3 else 'medium' if distance<=5 else 'hard',
                         **{key:float(ious[index]) for key,index in selected.items()},
                         oracle_candidate_iou=float(ious.max()),
                         choices={key:pred['origins'][index] for key,index in selected.items()}))
        if (done+1)%50==0:print(f'gate eval {done+1}/{len(rs)} fused={np.mean([r["fused"] for r in rows]):.4f}',flush=True)
    result=dict(split=args.split,test_used=args.split=='test',full_validation=args.split=='val',full_split=True,
                control=args.control,
                geometry=str(D.GEOMETRY),provenance=provenance,
                gate_weight=args.gate_weight,
                gate2_weight=args.gate2_weight,
                checkpoints={name:dict(path=str(getattr(args,name)),sha256=D.sha(getattr(args,name)))
                             for name in (('masks','selector','apex','gate')+
                                          (('gate2',) if args.gate2 else ())+
                                          (('reranker',) if args.reranker else ()))},
                metrics={name:report(rows,name) for name in
                         (('baseline','fused','selector_fused','gate_only','oracle_candidate_iou','reranked') if ranker is not None
                          else ('baseline','fused','selector_fused','gate_only','oracle_candidate_iou'))},
                source_sha256=D.sha(Path(__file__)),elapsed_seconds=time.time()-start,samples=rows)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','provenance')},indent=2),flush=True)


if __name__=='__main__':main()
