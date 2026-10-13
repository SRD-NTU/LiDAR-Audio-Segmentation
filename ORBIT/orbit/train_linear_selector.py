import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from apex import data as D
from .selector import batch
from .linear_selector import LinearObjectSelector
from .train_selector import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--priors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError(args.output)
    torch.manual_seed(args.seed);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    tr=torch.load(args.cache/'perfect_objects_train.pt',weights_only=False)
    va=torch.load(args.cache/'perfect_objects_val.pt',weights_only=False)
    model=LinearObjectSelector().to(args.device)
    with np.load(args.priors) as z:
        model.text_features.copy_(torch.tensor(z['text_features'],device=args.device))
        model.text_scale.copy_(torch.tensor(z['logit_scale'],device=args.device))
    g=np.concatenate([r['geometry'] for r in tr])
    model.mean.copy_(torch.tensor(g.mean(0),device=args.device))
    model.std.copy_(torch.tensor(g.std(0).clip(.05),device=args.device))
    freq=np.bincount(np.concatenate([r['categories'] for r in tr]),minlength=17).clip(1)
    weights=(freq.sum()/freq)**.75;weights/=weights.mean()
    weights=torch.tensor(weights,dtype=torch.float32,device=args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    rng=np.random.default_rng(args.seed);best=-1;history=[];start=time.time()
    config={**vars(args),'cache':str(args.cache),'priors':str(args.priors),'output':str(args.output),
            'kind':'linear','test_used':False,'privileged_evaluation':True,
            'source_hashes':{f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    for epoch in range(1,args.epochs+1):
        model.train();total=0;order=rng.permutation(len(tr))
        for start_idx in range(0,len(tr),64):
            rows=[tr[i] for i in order[start_idx:start_idx+64]]
            inp,(cat,target,source)=batch(rows,args.device)
            out=model(*inp)
            # Padding slots have no target probability. Ordinary CE is correct;
            # generic label smoothing would allocate mass to masked -1e4 slots.
            value=(F.cross_entropy(out['scores'],target)
                +F.cross_entropy(out['geom_class'].transpose(1,2),cat,weight=weights,label_smoothing=.05)
                +F.cross_entropy(out['audio_class'],source,label_smoothing=.05)
                +.02*model.geom_class.weight.square().sum()+.1*model.audio_residual.weight.square().sum())
            opt.zero_grad(set_to_none=True);value.backward();opt.step();total+=float(value.detach())*len(rows)
        if epoch==1 or epoch%5==0:
            report=evaluate(model,va,args.device);train=evaluate(model,tr[::8],args.device)
            row=dict(epoch=epoch,train_loss=total/len(tr),train_accuracy=train['selection_accuracy'],
                     validation_accuracy=report['selection_accuracy'],audio_accuracy=report['audio_category_accuracy'],
                     geometry_accuracy=report['source_geometry_category_accuracy'],elapsed_seconds=time.time()-start)
            print(json.dumps(row),flush=True);history.append(row)
            if report['selection_accuracy']>best:
                best=report['selection_accuracy']
                torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,kind='linear'),args.output/'best.pt')
                (args.output/'perfect_mask_validation.json').write_text(json.dumps(report,indent=2))
            (args.output/'history.json').write_text(json.dumps(history,indent=2))
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(epochs=args.epochs,test_used=False,
        best_perfect_mask_validation_selection_accuracy=best,checkpoint_sha256=D.sha(args.output/'best.pt')),indent=2))


if __name__=='__main__':main()
