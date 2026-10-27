"""Predicted-candidate training/evaluation bank. Masks are generated without labels."""
import argparse
import json
from pathlib import Path
import time
import sys
import numpy as np
import torch
from apex import data as D
from .inference import load_models,propose_and_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True);p.add_argument('--apex',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path)
    p.add_argument('--train-stride',type=int,default=3);p.add_argument('--device',default='cuda')
    p.add_argument('--fit-reranker',type=Path)
    p.add_argument('--spatial',type=Path)
    p.add_argument('--gate',type=Path)
    p.add_argument('--gate-weight',type=float,default=1.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.geometry:D.GEOMETRY=args.geometry
    if (args.output/'COMPLETE.json').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    models=load_models(args.masks,args.selector,args.apex,args.device);start=time.time()
    gate=None
    if args.gate:
        from .audio_gate import AudioSeedGate
        checkpoint=torch.load(args.gate,map_location='cpu',weights_only=False)
        gate=AudioSeedGate().to(args.device).eval()
        gate.load_state_dict(checkpoint['model'])
    posterior_hashes={}
    for split in ('train','val'):
        rs,post,emb,provenance=D.audio_bank(split);rows=[]
        if args.spatial:
            path=args.spatial/f'adaptive_{split}_posterior.npy'
            post=D.load_posterior(path, rs)
            if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Wrong posterior dimensions')
            posterior_hashes[split]=D.sha(path)
        indices=np.arange(len(rs))[::args.train_stride] if split=='train' else np.arange(len(rs))
        for done,i in enumerate(indices):
            rel=rs[i];x=D.load_inputs(rel,post[i],emb[i])
            pred=propose_and_score(x,models,args.device)
            features=pred['rank_features']
            baseline=pred['scores']['hybrid'].astype(np.float32)
            if gate is not None:
                from .evaluate_audio_gate import candidate_gate_values
                values=candidate_gate_values(x,models,gate,pred,args.device)
                features=np.column_stack([features,values]).astype(np.float32)
                baseline=baseline+args.gate_weight*values
            # Only now load supervision, after all proposals and features exist.
            target=D.load_targets(rel,x['inverse'],x['count']);masks=pred['masks']
            inter=(masks*target['positive'][None]).sum(1)
            union=(masks*x['count'][None]).sum(1)+target['positive'].sum()-inter
            ious=np.where(union==0,1.,inter/union.clip(1))
            rows.append(dict(rel=rel,features=features,baseline=baseline,
                             ious=ious.astype(np.float32),origins=pred['origins']))
            if (done+1)%100==0:print(f'candidates {split} {done+1}/{len(indices)} {time.time()-start:.1f}s',flush=True)
        torch.save(rows,args.output/f'{split}.pt')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(
        checkpoints={k:dict(path=str(getattr(args,k)),sha256=D.sha(getattr(args,k))) for k in ('masks','selector','apex')},
        train_stride=args.train_stride,geometry=str(D.GEOMETRY),test_used=False,
        gate_checkpoint_sha256=D.sha(args.gate) if args.gate else None,gate_weight=args.gate_weight,
        replacement_posterior_sha256=posterior_hashes,
        source_hashes={f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')}),indent=2))
    if args.fit_reranker:
        from .rerank import main as fit
        sys.argv=[sys.argv[0],'--bank',str(args.output),'--output',str(args.fit_reranker),'--device',args.device]
        fit()


if __name__=='__main__':main()
