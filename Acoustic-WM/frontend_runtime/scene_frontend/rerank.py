"""Calibrate selection on imperfect TRAIN proposals, never perfect VAL objects."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from apex import data as D
from apex.model import mlp


class Reranker(nn.Module):
    def __init__(self,n_features=35):
        super().__init__()
        self.net=mlp([n_features,64,32,1],.15)
        self.register_buffer('mean',torch.zeros(n_features));self.register_buffer('std',torch.ones(n_features))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,features,baseline):
        return baseline+self.net(((features-self.mean)/self.std).clamp(-8,8)).squeeze(-1)


def batch(rows,device):
    b=len(rows);n=max(len(r['ious']) for r in rows);d=rows[0]['features'].shape[1]
    f=torch.zeros(b,n,d,device=device);base=torch.zeros(b,n,device=device)
    y=torch.zeros(b,n,device=device);valid=torch.zeros(b,n,dtype=torch.bool,device=device)
    for i,r in enumerate(rows):
        m=len(r['ious']);f[i,:m]=torch.as_tensor(r['features'],device=device)
        base[i,:m]=torch.as_tensor(r['baseline'],device=device);y[i,:m]=torch.as_tensor(r['ious'],device=device)
        valid[i,:m]=True
    return f,base,y,valid


@torch.inference_mode()
def evaluate(model,rows,device):
    model.eval();samples=[]
    for start in range(0,len(rows),64):
        group=rows[start:start+64];f,b,y,v=batch(group,device)
        scores=model(f,b).masked_fill(~v,-1e4);choice=scores.argmax(1)
        for i,r in enumerate(group):
            j=int(choice[i]);samples.append(dict(rel=r['rel'],iou=float(r['ious'][j]),origin=r['origins'][j],
                                               baseline_iou=float(r['ious'][r['baseline'].argmax()])))
    return dict(n=len(samples),mean_iou=float(np.mean([s['iou'] for s in samples])),
                baseline_iou=float(np.mean([s['baseline_iou'] for s in samples])),samples=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(1337)
    tr=torch.load(args.bank/'train.pt',weights_only=False);va=torch.load(args.bank/'val.pt',weights_only=False)
    # All-zero banks have no ranking supervision. They remain in validation.
    usable=[r for r in tr if r['ious'].max()>.05]
    g=np.concatenate([r['features'] for r in usable]);model=Reranker(g.shape[1]).to(args.device)
    model.mean.copy_(torch.tensor(g.mean(0),device=args.device));model.std.copy_(torch.tensor(g.std(0).clip(.02),device=args.device))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1);rng=np.random.default_rng(1337)
    best=-1;history=[];start=time.time()
    config=dict(bank=str(args.bank),bank_manifest_sha256=D.sha(args.bank/'COMPLETE.json'),epochs=args.epochs,
                training_observations=len(tr),usable_training_observations=len(usable),test_used=False,
                n_features=g.shape[1],source_sha256=D.sha(Path(__file__)))
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    for epoch in range(1,args.epochs+1):
        model.train();order=rng.permutation(len(usable))
        for start_idx in range(0,len(usable),32):
            rows=[usable[i] for i in order[start_idx:start_idx+32]];f,b,y,v=batch(rows,args.device)
            logits=model(f,b).masked_fill(~v,-1e4)
            target=y.pow(4);target=target/target.sum(1,keepdim=True).clamp_min(1e-9)
            loss=-(target*logits.log_softmax(1)).sum(1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if epoch==1 or epoch%5==0:
            report=evaluate(model,va,args.device);train=evaluate(model,tr,args.device)
            row=dict(epoch=epoch,train_iou=train['mean_iou'],validation_iou=report['mean_iou'],elapsed_seconds=time.time()-start)
            print(json.dumps(row),flush=True);history.append(row)
            if report['mean_iou']>best:
                best=report['mean_iou'];torch.save(dict(model=model.state_dict(),config=config,epoch=epoch),args.output/'best.pt')
                (args.output/'validation.json').write_text(json.dumps(report,indent=2))
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(test_used=False,best_validation_iou=best,
        checkpoint_sha256=D.sha(args.output/'best.pt')),indent=2))


if __name__=='__main__':main()
