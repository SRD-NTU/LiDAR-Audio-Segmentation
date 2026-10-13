"""Full train-instance learning and correctly labelled privileged prompt diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from apex import data as D
from .masks import PromptMasks, objective

THRESHOLDS = np.array([.2, .35, .5, .65, .8], np.float32)


def load_split(cache, split, limit=None):
    paths = sorted((cache / split).glob('*.npz'))
    if limit:
        paths = [paths[i] for i in np.linspace(0, len(paths)-1, min(len(paths), limit), dtype=int)]
    def read(path):
        with np.load(path, allow_pickle=False) as z:
            row = {k: z[k] for k in z.files}
        row['rel'] = str(row['rel'])
        return row
    start = time.time()
    with ThreadPoolExecutor(4) as pool:
        rows = []
        for row in pool.map(read, paths):
            rows.append(row)
            if len(rows)%400 == 0:
                print(f'load {split} {len(rows)}/{len(paths)} {time.time()-start:.1f}s', flush=True)
    return rows


def inputs(row, device):
    return [torch.as_tensor(row[k], device=device, dtype=torch.float32) for k in ('features','xyz','count')]+[
        torch.as_tensor(row['knn'], device=device, dtype=torch.long)]


def prompts(row, rng, n=8):
    ids = row['node_object']
    available = np.unique(ids[ids>=0])
    src = int(row['source_object'])
    others = available[available!=src]
    objects = np.r_[src, rng.choice(others, min(n-1, len(others)), replace=False)]
    seed, targets = [], []
    for j in objects:
        idx = np.flatnonzero(ids==j)
        if len(idx) == 0:
            continue
        seed.append(rng.choice(idx))
        # Exact fractional target for sounding object; majority voxel labels for
        # other TRAIN instances. No such annotations reach inference.
        targets.append(row['positive']/row['count'] if j==src else (ids==j).astype(np.float32))
    return np.asarray(seed, np.int64), np.asarray(targets, np.float32)


@torch.inference_mode()
def evaluate(model, rows, device):
    model.eval(); records=[]
    for r in rows:
        f, xyz, count, knn = inputs(r, device)
        enc = model.encode(f, xyz, knn)
        # This evaluator deliberately uses a correct point prompt. NOT deployable.
        positive = np.flatnonzero(r['positive'] > 0)
        centre = np.average(r['xyz'][positive], axis=0, weights=r['positive'][positive])
        seed = positive[np.argmin(np.linalg.norm(r['xyz'][positive]-centre, axis=1))]
        out = model.decode(enc, xyz, count, torch.tensor([seed], device=device))
        p = out['logits'][0].sigmoid().cpu().numpy()
        pred = p[:, None] >= THRESHOLDS[None]
        inter = (pred * r['positive'][:, None]).sum(0)
        union = (pred * r['count'][:, None]).sum(0) + r['positive'].sum() - inter
        records.append(dict(rel=r['rel'], ious=(inter/union.clip(1)).tolist()))
    all_iou = np.array([r['ious'] for r in records])
    j = int(all_iou.mean(0).argmax())
    return dict(n=len(records), mean_iou=float(all_iou[:, j].mean()), threshold=float(THRESHOLDS[j]),
                all_threshold_means=all_iou.mean(0).tolist(), privileged_correct_point_prompt=True,
                deployable_segmentation_result=False,
                samples=[dict(rel=r['rel'], iou=r['ious'][j]) for r in records])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--prompts',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=2);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    tr=load_split(args.cache,'train',args.train_limit);va=load_split(args.cache,'val',args.val_limit)
    assert not ({r['rel'].split('/')[1] for r in tr}&{r['rel'].split('/')[1] for r in va})
    config={**vars(args),'cache':str(args.cache),'output':str(args.output),'test_used':False,
            'train_samples':len(tr),'val_samples':len(va),'validation_uses_correct_point_prompt':True,
            'source_hashes':{f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    model=PromptMasks().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr*.1)
    rng=np.random.default_rng(args.seed);history=[];best=-1;start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();total=0;opt.zero_grad(set_to_none=True)
        for step,i in enumerate(rng.permutation(len(tr))):
            row=tr[i];f,xyz,count,knn=inputs(row,args.device)
            seed,target=prompts(row,rng,args.prompts)
            enc=model.encode(f,xyz,knn)
            out=model.decode(enc,xyz,count,torch.as_tensor(seed,device=args.device))
            value=objective(enc,out,torch.as_tensor(target,device=args.device),count,
                            torch.as_tensor(row['node_category'],device=args.device),
                            torch.as_tensor(row['node_object'],device=args.device))
            if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite mask loss {row["rel"]}')
            group_size=min(4,len(tr)-(step//4)*4)
            (value/group_size).backward();total+=float(value.detach())
            if (step+1)%4==0 or step+1==len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(),2)
                opt.step();opt.zero_grad(set_to_none=True)
            if (step+1)%500==0:print(f'mask epoch {epoch} {step+1}/{len(tr)} loss {total/(step+1):.4f}',flush=True)
        scheduler.step()
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            report=evaluate(model,va,args.device)
            train_report=evaluate(model,tr[::max(1,len(tr)//32)],args.device)
            row=dict(epoch=epoch,train_loss=total/len(tr),oracle_validation_iou=report['mean_iou'],
                     sampled_training_oracle_iou=train_report['mean_iou'],threshold=report['threshold'],
                     elapsed_seconds=time.time()-start)
            print(json.dumps(row),flush=True);history.append(row)
            if report['mean_iou']>best:
                best=report['mean_iou']
                torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,threshold=report['threshold']),args.output/'best.pt')
                (args.output/'correct_prompt_validation.json').write_text(json.dumps(report,indent=2))
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
        torch.save(dict(model=model.state_dict(),config=config,epoch=epoch),args.output/'last.pt')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(epochs=args.epochs,test_used=False,
        best_privileged_correct_prompt_iou=best,checkpoint_sha256=D.sha(args.output/'best.pt')),indent=2))


if __name__=='__main__':main()
