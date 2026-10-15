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
from .cache_contracts import instance_target, require_consistent_cache

THRESHOLDS = np.array([.2, .35, .5, .65, .8], np.float32)


def load_split(cache, split, limit=None):
    paths = sorted((cache / split).glob('*.npz'))
    if limit:
        paths = [paths[i] for i in np.linspace(0, len(paths)-1, min(len(paths), limit), dtype=int)]
    def read(path):
        with np.load(path, allow_pickle=False) as z:
            row = {k: z[k] for k in z.files}
        row['rel'] = str(row['rel'])
        require_consistent_cache(row)
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
    if n <= 0:
        raise ValueError('Prompt budget must be positive')
    objects = np.r_[([src] if src in available else []), rng.choice(others, min(n-1, len(others)), replace=False)]
    seed, targets = [], []
    for j in objects:
        target = instance_target(row, j)
        # A mixed voxel may support several instance targets. Use its dominant
        # instance for prompt identity so one seed cannot supervise two masks.
        idx = np.flatnonzero((ids==j) & (target > 0))
        if len(idx) == 0:
            continue
        seed.append(rng.choice(idx))
        targets.append(target)
    return np.asarray(seed, np.int64), np.asarray(targets, np.float32)


@torch.inference_mode()
def evaluate(model, rows, device):
    model.eval(); records=[]
    for r in rows:
        f, xyz, count, knn = inputs(r, device)
        enc = model.encode(f, xyz, knn)
        # This evaluator deliberately uses a correct point prompt. NOT deployable.
        positive = np.flatnonzero(r['positive'] > 0)
        if not len(positive):
            raise ValueError(f'No retained source prompt in validation: {r["rel"]}')
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
    p.add_argument('--deployment-selector',type=Path)
    p.add_argument('--deployment-apex',type=Path)
    p.add_argument('--deployment-posterior',type=Path)
    p.add_argument('--deployment-mask-threshold',type=float,default=.5,
                   help='Predeclared candidate threshold; hold fixed in controlled comparisons')
    args=p.parse_args()
    if not 0 < args.deployment_mask_threshold < 1:
        p.error('--deployment-mask-threshold must be between zero and one')
    if bool(args.deployment_selector)!=bool(args.deployment_apex):
        p.error('Supply both --deployment-selector and --deployment-apex')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    tr=load_split(args.cache,'train',args.train_limit);va=load_split(args.cache,'val',args.val_limit)
    tr = [row for row in tr if (row['node_object']>=0).any()]
    if not tr:raise ValueError('No retained object supervision in training cache')
    assert not ({r['rel'].split('/')[1] for r in tr}&{r['rel'].split('/')[1] for r in va})
    config={**vars(args),'cache':str(args.cache),'output':str(args.output),'test_used':False,
            'train_samples':len(tr),'val_samples':len(va),'validation_uses_correct_point_prompt':True,
            'source_hashes':{f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')}}
    config={key:str(value) if isinstance(value,Path) else value for key,value in config.items()}
    config['method_version']='orbit-consistent-v2'
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
            diagnostic=[row for row in tr if (row['positive']>0).any()]
            train_report=evaluate(model,diagnostic[::max(1,len(diagnostic)//32)],args.device) if diagnostic else None
            deployment=None
            if args.deployment_selector:
                from .inference import load_models
                from .validation import deployment_validation
                models=load_models(None,args.deployment_selector,args.deployment_apex,args.device,mask_model=model)
                models[-1]['mask_threshold']=args.deployment_mask_threshold
                deployment=deployment_validation(models,args.device,args.cache,posterior_path=args.deployment_posterior)
            selected_metric=deployment['mean_iou'] if deployment else report['mean_iou']
            row=dict(epoch=epoch,train_loss=total/len(tr),oracle_validation_iou=report['mean_iou'],
                     sampled_training_oracle_iou=train_report['mean_iou'] if train_report else None,threshold=report['threshold'],
                     selection_metric='deployable' if deployment else 'privileged-correct-prompt',selected_validation_metric=selected_metric,
                     elapsed_seconds=time.time()-start)
            print(json.dumps(row),flush=True);history.append(row)
            if selected_metric>best:
                best=selected_metric
                torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,
                                threshold=args.deployment_mask_threshold if deployment else report['threshold']),args.output/'best.pt')
                (args.output/'correct_prompt_validation.json').write_text(json.dumps(report,indent=2))
                if deployment:(args.output/'deployable_validation.json').write_text(json.dumps(deployment,indent=2))
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
        torch.save(dict(model=model.state_dict(),config=config,epoch=epoch),args.output/'last.pt')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(epochs=args.epochs,test_used=False,
        selection_metric='deployable' if args.deployment_selector else 'privileged-correct-prompt',
        best_selected_validation_metric=best,checkpoint_sha256=D.sha(args.output/'best.pt')),indent=2))


if __name__=='__main__':main()
