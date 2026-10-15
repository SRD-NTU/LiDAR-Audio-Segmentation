"""Full-resolution validation of observation-only predictions, labels loaded AFTER."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from apex import data as D
from .inference import load_models, propose_and_score


def report(rows, key):
    values=np.array([r[key] for r in rows])
    result=dict(n=len(rows),mean_iou=float(values.mean()),median_iou=float(np.median(values)),
                zero_iou_fraction=float((values==0).mean()),fraction_iou_ge_half=float((values>=.5).mean()),
                by_difficulty={name:dict(n=sum(r['tier']==name for r in rows),
                    mean_iou=float(np.mean([r[key] for r in rows if r['tier']==name])) if any(r['tier']==name for r in rows) else None)
                    for name in ('easy','medium','hard')},
                by_category={c:float(np.mean([r[key] for r in rows if r['category']==c])) for c in sorted({r['category'] for r in rows})},
                by_scene={s:float(np.mean([r[key] for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})})
    rng=np.random.default_rng(726);scenes=sorted(result['by_scene'])
    grouped={s:np.array([r[key] for r in rows if r['scene']==s]) for s in scenes}
    ci=[np.concatenate([grouped[s] for s in rng.choice(scenes,len(scenes))]).mean() for _ in range(2000)]
    result['scene_bootstrap_ci95']=np.percentile(ci,[2.5,97.5]).tolist()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--masks',type=Path,required=True);p.add_argument('--selector',type=Path,required=True)
    p.add_argument('--apex',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--seeds',type=int,default=48)
    p.add_argument('--geometry',type=Path);p.add_argument('--posterior',type=Path)
    p.add_argument('--reranker',type=Path)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--device',default='cuda');p.add_argument('--control',choices=['matched','shuffled','uniform-spatial'],default='matched')
    args=p.parse_args()
    if args.split=='test' and (args.limit is not None or args.control!='matched'):
        raise ValueError('Final test requires the complete matched split')
    if args.geometry is not None:D.GEOMETRY=args.geometry
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    models=load_models(args.masks,args.selector,args.apex,args.device)
    ranker=None
    if args.reranker:
        from .rerank import Reranker
        checkpoint=torch.load(args.reranker,map_location=args.device,weights_only=False)
        ranker=Reranker(checkpoint['config']['n_features']).to(args.device)
        ranker.load_state_dict(checkpoint['model']);ranker.eval()
    rs,post,emb,provenance=D.audio_bank(args.split)
    if args.posterior:
        post=D.load_posterior(args.posterior, rs)
        if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Wrong posterior bank shape')
        provenance['replacement_posterior_sha256']=D.sha(args.posterior)
    provenance['geometry']=str(D.GEOMETRY)
    idx=np.arange(len(rs)) if not args.limit else np.linspace(0,len(rs)-1,min(args.limit,len(rs)),dtype=int)
    rng=np.random.default_rng(1701)
    partners=[]
    if args.control=='shuffled':
        for r in rs:
            choices=[j for j,s in enumerate(rs) if s.split('/')[1]!=r.split('/')[1]]
            if not choices:raise ValueError('Shuffled control requires at least two scenes')
            partners.append(int(rng.choice(choices)))
    rows=[];start=time.time()
    for done,i in enumerate(idx):
        rel=rs[i];j=partners[i] if args.control=='shuffled' else i
        posterior=np.ones_like(post[j]) if args.control=='uniform-spatial' else post[j]
        x=D.load_inputs(rel,posterior,emb[j])
        pred=propose_and_score(x,models,args.device,args.seeds)
        if ranker is not None:
            with torch.inference_mode():
                scores=ranker(torch.as_tensor(pred['rank_features'],device=args.device),
                              torch.as_tensor(pred['scores']['hybrid'],device=args.device,dtype=torch.float32))
                pred['choices']['reranker']=int(scores.argmax())
        # Ground truth is accessed only after observation-only inference completes.
        targets=D.load_targets(rel,x['inverse'],x['count'])
        positive=targets['positive'];count=x['count'];masks=pred['masks']
        inter=(masks*positive[None]).sum(1)
        union=(masks*count[None]).sum(1)+positive.sum()-inter
        ious=np.where(union==0,1.,inter/union.clip(1))
        meta=json.loads((D.DATA/rel/'metadata.json').read_text())
        distance=float(meta['relative_source']['distance_m'])
        apexmask=pred['apex_probability']>=models[-1]['apex_threshold']
        ai=float((apexmask*positive).sum());au=float((apexmask*count).sum()+positive.sum()-ai)
        row=dict(rel=rel,scene=rel.split('/')[1],category=D.CATS[targets['source']] if targets['source']>=0 else 'none',distance=distance,
                 tier='easy' if distance<=3 else 'medium' if distance<=5 else 'hard',
                 **{k:float(ious[v]) for k,v in pred['choices'].items()},
                 apex=ai/max(au,1) if au else 1.,oracle_candidate_iou=float(ious.max()),candidates=len(masks),
                 source_seed_hit=bool((positive[pred['seeds']]>0).any()),
                 choices={k:pred['origins'][v] for k,v in pred['choices'].items()})
        rows.append(row)
        if (done+1)%50==0:
            print(json.dumps(dict(done=done+1,n=len(idx),hybrid_iou=float(np.mean([r['hybrid'] for r in rows])),
                oracle_coverage=float(np.mean([r['oracle_candidate_iou'] for r in rows])),elapsed=time.time()-start)),flush=True)
    result=dict(method_version='orbit-consistent-v2',split=args.split,test_evaluated=args.split=='test',control=args.control,
                full_validation=args.split=='val' and args.limit is None,full_split=args.limit is None,
                checkpoints={k:dict(path=str(getattr(args,k)),sha256=D.sha(getattr(args,k))) for k in
                             (('masks','selector','apex','reranker') if args.reranker else ('masks','selector','apex'))},
                source_hashes={f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')},provenance=provenance,
                metrics={k:report(rows,k) for k in ('selector','hybrid','apex_quality','apex','oracle_candidate_iou')+
                         (('reranker',) if args.reranker else ())},
                source_seed_recall=float(np.mean([r['source_seed_hit'] for r in rows])),samples=rows,
                note='oracle_candidate_iou and source_seed_recall use labels ONLY for diagnostics, not prediction.',
                elapsed_seconds=time.time()-start)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','source_hashes','provenance')},indent=2),flush=True)


if __name__=='__main__':main()
