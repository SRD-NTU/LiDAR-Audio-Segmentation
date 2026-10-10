"""Train the actual LS3D graph on geometry-only world-model mask proposals."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lidarwm.engine import atomic_checkpoint,build_model,seed_everything
from lidarwm.suppression import MaskSuppression,choose_candidate


def load_split(root,split):
    scenes=[]
    for path in sorted((root/split).glob('*/*/*.npz')):
        with np.load(path,allow_pickle=False) as z:
            scene={key:z[key].copy() for key in ('features','positions','iou','intersection','areas','target','center','path')}
        scenes.append(scene)
    if not scenes:raise ValueError(f'No {split} candidates at {root}')
    return scenes


def collate(items):
    b=len(items);k=max(len(item['iou']) for item in items);d=items[0]['features'].shape[-1]
    features=torch.zeros(b,k,d);positions=torch.zeros(b,k,3);iou=torch.zeros(b,k)
    valid=torch.zeros(b,k,dtype=torch.bool);centers=torch.zeros(b,3)
    for i,item in enumerate(items):
        n=len(item['iou']);features[i,:n]=torch.from_numpy(item['features'])
        positions[i,:n]=torch.from_numpy(item['positions']);iou[i,:n]=torch.from_numpy(item['iou'])
        valid[i,:n]=True;centers[i]=torch.from_numpy(item['center'])
    return features,positions,iou,valid,centers


def measure(scenes,choices):
    intersections=np.asarray([s['intersection'][c] for s,c in zip(scenes,choices)],dtype='float64')
    areas=np.asarray([s['areas'][c] for s,c in zip(scenes,choices)],dtype='float64')
    targets=np.asarray([s['target'] for s in scenes],dtype='float64')
    union=areas+targets-intersections
    sample=(intersections+1e-9)/(union+1e-9)
    return dict(mean_iou=float(sample.mean()),micro_iou=float(intersections.sum()/max(union.sum(),1)),
      precision=float(intersections.sum()/max(areas.sum(),1)),recall=float(intersections.sum()/max(targets.sum(),1))),sample


@torch.inference_mode()
def score(model,scenes,mean,std,device):
    model.eval();results=[]
    loader=DataLoader(scenes,batch_size=32,shuffle=False,collate_fn=collate)
    for z,c,iou,valid,center in loader:
        z=((z.to(device)-mean)/std).clamp(-6,6)
        logit,_=model(z,c.to(device),valid.to(device))
        probabilities=logit.sigmoid().cpu().numpy()
        for row,mask in zip(probabilities,valid):results.append(row[mask.numpy()])
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',default='cache/suppression-v1')
    parser.add_argument('--output',default='runs/suppression-v1');parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seed',type=int,default=1337);parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--feature-mode',choices=['all','compact','physics'],default='all')
    parser.add_argument('--residual-quality',action='store_true')
    parser.add_argument('--dropout',type=float,default=.25)
    parser.add_argument('--hidden',type=int,default=96)
    parser.add_argument('--augment-features',action='store_true')
    parser.add_argument('--ranking-weight',type=float,default=.2)
    parser.add_argument('--regression-weight',type=float,default=1.)
    args=parser.parse_args();seed_everything(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('Run suppression training inside a Slurm GPU allocation')
    root=Path(args.cache);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Use a new output directory rather than overwrite an experiment')
    provenance=json.loads((root/'provenance.json').read_text())
    train=load_split(root,'train');validation=load_split(root,'val')
    if len(train)!=provenance['summary']['train']['samples'] or len(validation)!=provenance['summary']['val']['samples']:
        raise ValueError('Incomplete cache')
    features=np.concatenate([s['features'] for s in train]);mean_np=features.mean(0);std_np=features.std(0).clip(.05)
    device=torch.device('cuda');mean=torch.from_numpy(mean_np).to(device);std=torch.from_numpy(std_np).to(device)
    model=MaskSuppression(features.shape[-1],hidden=args.hidden,feature_mode=args.feature_mode,
      residual_quality=args.residual_quality,dropout=args.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=2e-5)
    loader=DataLoader(train,batch_size=32,shuffle=True,collate_fn=collate)
    margins=(0.,.01,.03,.05,.1,.2,.3,1.)
    baseline,baseline_samples=measure(validation,[0]*len(validation))
    oracle,_=measure(validation,[int(s['iou'].argmax()) for s in validation])
    print(json.dumps(dict(event='start',train=len(train),val=len(validation),baseline=baseline,
      oracle_candidate_bank=oracle,parameters=sum(p.numel() for p in model.parameters()),provenance=provenance)),flush=True)
    world=torch.load(root/'world.pt',map_location='cpu',weights_only=False)
    snapshot=out/'source'
    shutil.copytree(Path('src/lidarwm'),snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    best=-1;best_raw=-1;best_epoch=-1;best_state=None
    for epoch in range(args.epochs):
        model.train();losses=[];started=time.time()
        for z,c,iou,valid,center in loader:
            if args.augment_features:
                # Simulate the confidence/calibration shift of unseen scenes.
                # The perturbation is shared across all candidates in a frame,
                # preserving their order and leaving target overlap unchanged.
                scale=torch.empty(z.shape[0],1,1).uniform_(.35,1.3)
                bias=torch.empty(z.shape[0],1,1).uniform_(-4,1)
                for column in (1,2,4):
                    z[:,:,column:column+1]=(torch.logit(z[:,:,column:column+1].clamp(1e-4,1-1e-4))*scale+bias).sigmoid()
                z[:,:,3]*=scale.squeeze(-1)
            z=((z.to(device)-mean)/std).clamp(-6,6);c=c.to(device);iou=iou.to(device)
            valid=valid.to(device);center=center.to(device)
            logit,displacement=model(z,c,valid)
            regression=F.binary_cross_entropy_with_logits(logit[valid],iou[valid])
            target=F.softmax((iou/.08).masked_fill(~valid,-1e4),-1)
            ranking=-(target*F.log_softmax(logit.masked_fill(~valid,-1e4),-1)).sum(-1)
            informative=iou.max(-1).values>.01
            ranking=ranking[informative].mean() if informative.any() else logit.new_zeros(())
            positive=valid&(iou>.1)
            movement=F.smooth_l1_loss(displacement[positive],(center[:,None]-c)[positive],beta=.025) if positive.any() else logit.new_zeros(())
            loss=args.regression_weight*regression+args.ranking_weight*ranking+.25*movement
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite graph loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step();scores=score(model,validation,mean,std,device)
        options={margin:measure(validation,[choose_candidate(s,margin) for s in scores])[0] for margin in margins}
        margin=max(options,key=lambda m:options[m]['mean_iou']);metrics=options[margin]
        event=dict(event='epoch',epoch=epoch,loss=float(np.mean(losses)),validation=metrics,margin=margin,
          raw_selection=options[0.],seconds=time.time()-started)
        print(json.dumps(event),flush=True)
        with (out/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(event)+'\n')
        if metrics['mean_iou']>best+1e-12 or (abs(metrics['mean_iou']-best)<1e-12 and options[0.]['mean_iou']>best_raw+1e-12):
            best=metrics['mean_iou'];best_raw=options[0.]['mean_iou'];best_epoch=epoch
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            cfg=copy.deepcopy(world['config']);base_arch=cfg['model'].pop('architecture')
            cfg['model'].update(architecture='suppressed',base_architecture=base_arch,
              suppression_dim=features.shape[-1],suppression_hidden=args.hidden,suppression_dropout=args.dropout,
              suppression_feature_mode=args.feature_mode,suppression_residual_quality=args.residual_quality,
              suppression_context_radius=provenance.get('context_radius_metres',0.),
              baseline_threshold=world['validation']['threshold'],suppression_margin=margin)
            cfg['evaluation']['tta']=False;cfg['training']['resume']=None
            cfg['training'].pop('initialize_from',None)
            assembled=build_model(cfg)
            assembled.base.load_state_dict(world['model']);assembled.refiner.load_state_dict(best_state)
            assembled.feature_mean.copy_(torch.from_numpy(mean_np));assembled.feature_std.copy_(torch.from_numpy(std_np))
            choices=[choose_candidate(s,margin) for s in scores]
            _,sample_iou=measure(validation,choices)
            delta=sample_iou-baseline_samples
            scenes=sorted({str(s['path']).split('/')[1] for s in validation})
            sums=np.asarray([delta[[str(s['path']).split('/')[1]==scene for s in validation]].sum() for scene in scenes])
            counts=np.asarray([sum(str(s['path']).split('/')[1]==scene for s in validation) for scene in scenes])
            indices=np.random.default_rng(1337).integers(0,len(scenes),(10000,len(scenes)))
            ci=np.quantile(sums[indices].sum(1)/counts[indices].sum(1),[.025,.975]).tolist()
            report=dict(epoch=epoch,margin=margin,baseline=baseline,validation=metrics,
              raw_selection=options[0.],training_options=vars(args),
              mean_iou_gain=best-baseline['mean_iou'],scene_cluster_gain_95ci=ci,oracle_candidate_bank=oracle,
              validation_samples=len(validation),test_evaluated=False,provenance=provenance,
              selected_nonbaseline=sum(c!=0 for c in choices),
              samples=[dict(path=str(s['path']),baseline_iou=float(base),iou=float(value),candidate=int(c))
                for s,base,value,c in zip(validation,baseline_samples,sample_iou,choices)])
            (out/'report.json').write_text(json.dumps(report,indent=2))
            atomic_checkpoint(dict(model=assembled.state_dict(),config=cfg,epoch=epoch,
              validation={**metrics,'threshold':.5},best_iou=best,inference_only=True,
              provenance=report),out/'best.pt')
            del assembled
        elif epoch-best_epoch>=args.patience:
            print(json.dumps(dict(event='early_stop',epoch=epoch,best_epoch=best_epoch)),flush=True);break
    print(json.dumps(dict(event='complete',best_epoch=best_epoch,best_mean_iou=best,baseline=baseline,
      checkpoint=str(out/'best.pt'))),flush=True)


if __name__=='__main__':main()
