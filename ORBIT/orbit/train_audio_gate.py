"""Train sounding-object query on TRAIN scenes with frozen geometric masks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from apex import data as D
from .audio_gate import AudioSeedGate,sample_seeds
from .inference import load_models
from .train_masks import load_split,inputs


def gate_logits(row,models,gate,device,seeds):
    masks,selector,_,_=models
    f,xyz,count,knn=inputs(row,device)
    acoustic=torch.as_tensor(row['acoustic'],device=device,dtype=torch.float32)
    audio=torch.as_tensor(row['audio'],device=device,dtype=torch.float32)
    with torch.no_grad():
        encoded=masks.encode(f,xyz,knn)
        decoded=masks.decode(encoded,xyz,count,torch.as_tensor(seeds,device=device))
        sound_prob=selector.sound_logits(audio[None]).softmax(-1)[0]
    return gate(encoded,decoded,acoustic,audio,count,
                torch.as_tensor(seeds,device=device),sound_prob)


@torch.inference_mode()
def evaluate(rows,models,gate,device):
    gate.eval();records=[]
    for i,row in enumerate(rows):
        rng=np.random.default_rng(9100+i)
        seeds=sample_seeds(row,rng)
        scores=gate_logits(row,models,gate,device,seeds)
        records.append(dict(rel=row['rel'],correct=int(scores.argmax())==0,n=len(seeds)))
    return dict(n=len(records),privileged_true_source_seed=True,
                deployable_segmentation_result=False,
                accuracy=float(np.mean([r['correct'] for r in records])),samples=records)


def main():
    p=argparse.ArgumentParser()
    for name in ('cache','masks','selector','apex','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    models=load_models(args.masks,args.selector,args.apex,args.device)
    for model in models[:3]:
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
    tr=load_split(args.cache,'train');va=load_split(args.cache,'val')
    if {r['rel'].split('/')[1] for r in tr}&{r['rel'].split('/')[1] for r in va}:
        raise AssertionError('TRAIN and VAL scenes overlap')
    gate=AudioSeedGate().to(args.device)
    opt=torch.optim.AdamW(gate.parameters(),lr=.0005,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.00005)
    config=dict(cache=str(args.cache),epochs=args.epochs,seed=args.seed,test_used=False,
                masks_sha256=D.sha(args.masks),selector_sha256=D.sha(args.selector),apex_sha256=D.sha(args.apex),
                source_sha256=D.sha(Path(__file__)),model_source_sha256=D.sha(Path(__file__).with_name('audio_gate.py')),
                train_samples=len(tr),val_samples=len(va))
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(args.seed);history=[];best=-1;start=time.time()
    for epoch in range(1,args.epochs+1):
        gate.train();loss_total=0
        for step,i in enumerate(rng.permutation(len(tr))):
            row=tr[i];seeds=sample_seeds(row,rng)
            scores=gate_logits(row,models,gate,args.device,seeds)
            loss=F.cross_entropy(scores[None],torch.zeros(1,device=args.device,dtype=torch.long))
            if not torch.isfinite(loss):raise FloatingPointError(row['rel'])
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),2.)
            opt.step();loss_total+=float(loss.detach())
            if (step+1)%500==0:print(f'gate epoch {epoch} {step+1}/{len(tr)} loss {loss_total/(step+1):.4f}',flush=True)
        scheduler.step()
        if epoch==1 or epoch%2==0 or epoch==args.epochs:
            report=evaluate(va,models,gate,args.device)
            info=dict(epoch=epoch,train_loss=loss_total/len(tr),privileged_validation_accuracy=report['accuracy'],
                      elapsed_seconds=time.time()-start)
            print(json.dumps(info),flush=True);history.append(info)
            if report['accuracy']>best:
                best=report['accuracy']
                torch.save(dict(model=gate.state_dict(),config=config,epoch=epoch),args.output/'best.pt')
                (args.output/'privileged_validation.json').write_text(json.dumps(report,indent=2))
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(test_used=False,epochs=args.epochs,
        best_privileged_validation_accuracy=best,checkpoint_sha256=D.sha(args.output/'best.pt')),indent=2))


if __name__=='__main__':main()
