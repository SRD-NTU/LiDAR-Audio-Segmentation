"""TRAIN-supervised recurrent gate; VAL selects checkpoint, TEST remains unseen."""
from pathlib import Path
import sys, argparse, json, hashlib, time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import ROOT,BASE,DATA,ARCHIVE,samples,seed_all,fingerprints,sha,output_path,write_json,save_pt
from activeview_v3.navigation import rotation
from activeview_v3.recurrent import NeoMemoryFilter
from acoustic_wm.world_model.predictive_world import PredictiveAcousticWorld
import numpy as np
import torch


def world_model(device):
    microphones=json.loads((BASE/'assets/sensor/metadata/microphone_array.json').read_text())['positions_lidar_m']
    model=PredictiveAcousticWorld(microphones).to(device).eval().requires_grad_(False)
    ck=torch.load(BASE/'checkpoints/acoustic_world.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'])
    return model


def cache_contract(split,rows):
    return dict(version=1,split=split,rows_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                singleview_checkpoints=fingerprints(),feature_contract_sha256=sha(ROOT/'data/feature_contract.json'),
                supervision='best-IoU TRAIN/VAL mask and nearest source site; IoU >= .3, distance <= 1.5 m')


def sequence_cache(split,world,device,limit=0,smoke=False):
    rows=samples(split)
    if limit:rows=rows[:limit]
    contract=cache_contract(split,rows)
    path=ROOT/'data/neo_sequence_cache'/(split+'.npz')
    marker=path.with_suffix('.json')
    if not smoke and path.exists():
        if not marker.exists() or json.loads(marker.read_text())!=contract:
            raise RuntimeError('Existing Neo sequence cache has a different input contract')
        with np.load(path) as z:return {k:z[k] for k in z.files}
    arrays={k:[] for k in ['memory','source','actions','observed','weight']}
    reasons=dict(low_iou=0,far_site=0,short_sequence=0)
    started=time.time()
    with torch.no_grad():
        for index,row in enumerate(rows):
            meta=json.loads((DATA/row/'metadata.json').read_text())
            with np.load(ARCHIVE/'cache/controlled'/(row+'.npz')) as z:
                iou=z['iou'];k=int(iou.argmax());quality=float(iou[k])
            if quality<.3:
                reasons['low_iou']+=1;continue
            with np.load(ARCHIVE/'cache/reference'/(row+'.npz')) as z:
                hypotheses=z['hypotheses'][k]
                local=(np.asarray(meta['source']['position_world'])-
                       np.asarray(meta['listener']['position_world']))@rotation(meta['listener']['yaw_rad'])
                distances=np.linalg.norm(hypotheses-local,axis=-1)
                h=int(distances.argmin())
                if distances[h]>1.5:
                    reasons['far_site']+=1;continue
                source=torch.as_tensor(hypotheses[h].copy(),dtype=torch.float32,device=device)[None,None]
                points=torch.as_tensor(z['world_points'].copy(),device=device)[None]
                features=torch.as_tensor(z['world_features'].copy(),device=device)[None]
                valid=torch.as_tensor(z['world_valid'].copy(),device=device)[None]
                reference=torch.view_as_complex(torch.as_tensor(z['observed'].copy(),device=device).contiguous())
            with np.load(ROOT/'data/rendered'/(row+'.npz')) as z:
                actions=z['actions'].astype('float32').copy()
                observed=z['observed'].astype('float32').copy()
                weight=z['weight'].astype('float32').copy()
            if len(actions)<3 or (weight.sum(-1)>0).sum()<2:
                reasons['short_sequence']+=1;continue
            q0,_,latent=world(points,features,valid,source)
            memory=world.assimilate(latent,q0,reference[None,None].expand_as(q0))
            arrays['memory'].append(memory[0,0].cpu().numpy())
            arrays['source'].append(source[0,0].cpu().numpy())
            arrays['actions'].append(actions)
            arrays['observed'].append(observed)
            arrays['weight'].append(weight)
            if (index+1)%100==0:
                print(json.dumps(dict(stage='neo_cache',split=split,read=index+1,kept=len(arrays['memory']),
                                      seconds=round(time.time()-started,1))),flush=True)
    result={k:np.stack(v).astype('float32') for k,v in arrays.items()}
    if len(result['memory'])<2:raise RuntimeError('Too few supervised sequences')
    print(json.dumps(dict(stage='neo_cache_done',split=split,read=len(rows),kept=len(result['memory']),
                          excluded=reasons,seconds=round(time.time()-started,1))),flush=True)
    if not smoke:
        temporary=output_path(path).with_suffix('.partial.npz')
        np.savez(temporary,**result)
        temporary.replace(path)
        write_json(marker,contract)
    return result


def relative_batch(current,proposed):
    old=current[:,3:].reshape(-1,3,3)
    new=proposed[:,3:].reshape(-1,3,3)
    translation=torch.bmm((proposed[:,:3]-current[:,:3])[:,None],old)[:,0]
    turn=old.transpose(-1,-2)@new
    return torch.cat([translation,turn.reshape(-1,9)],-1)


def batch_loss(world,filter_model,data,ids,rng,device,views=5):
    b=len(ids)
    memory=torch.as_tensor(data['memory'][ids],device=device)[:,None]
    source=torch.as_tensor(data['source'][ids],device=device)[:,None]
    actions=torch.as_tensor(data['actions'][ids],device=device)
    observations=torch.as_tensor(data['observed'][ids],device=device)
    reliabilities=torch.as_tensor(data['weight'][ids],device=device)
    order=torch.as_tensor(np.argsort(rng.random((b,actions.shape[1])),axis=-1)[:,:views].copy(),device=device)
    current=torch.zeros(b,12,device=device)
    current[:,3:]=torch.eye(3,device=device).reshape(1,9)
    losses=torch.zeros(b,device=device)
    counts=torch.zeros(b,device=device)
    for step in range(order.shape[1]):
        index=order[:,step]
        row=torch.arange(b,device=device)
        action=actions[row,index]
        observed=torch.view_as_complex(observations[row,index].contiguous())[:,None]
        band=reliabilities[row,index].clamp(0,1)
        move=relative_batch(current,action)
        predicted,source,q,logvar=world.imagine(memory,source,move[:,None])
        if step:
            energy=(observed-q).abs().square().sum((-1,-2))[:,0]
            per_view=((energy*torch.exp(-logvar[:,0])+logvar[:,0])*band).sum(-1)/band.sum(-1).clamp_min(1)
            audible=band.sum(-1)>0
            losses=losses+torch.where(audible,per_view,0.)
            counts=counts+audible.float()
        measured=q+(observed-q)*band[:,None,:,None,None]
        assimilated=world.assimilate(predicted,q,measured)
        memory=torch.where((band.sum(-1)>0)[:,None,None],filter_model(predicted,assimilated),predicted)
        current=action
    usable=counts>0
    if not usable.any():raise RuntimeError('No audible held-out view in sequence batch')
    return (losses[usable]/counts[usable]).mean()


def run(args):
    seed_all(args.seed)
    device='cpu' if args.smoke else 'cuda'
    world=world_model(device)
    train=sequence_cache('train',world,device,limit=4 if args.smoke else 0,smoke=args.smoke)
    val=sequence_cache('val',world,device,limit=4 if args.smoke else 0,smoke=args.smoke)
    model=NeoMemoryFilter().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=.01)
    folder=ROOT/'checkpoints/neo1'
    if not args.smoke and (folder/'memory_filter_complete.json').exists():
        print('Neo memory training already complete',flush=True);return
    best=float('inf');best_epoch=0;history=[]
    for epoch in range(1,2 if args.smoke else args.epochs+1):
        model.train();rng=np.random.default_rng(args.seed+epoch*1009)
        order=rng.permutation(len(train['memory']));losses=[]
        for begin in range(0,len(order),args.batch_size):
            ids=order[begin:begin+args.batch_size]
            loss=batch_loss(world,model,train,ids,rng,device)
            loss=loss+.01*model.logit.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite Neo sequence loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();validation=[];rng=np.random.default_rng(args.seed+9917)
        with torch.no_grad():
            for begin in range(0,len(val['memory']),args.batch_size):
                ids=np.arange(begin,min(begin+args.batch_size,len(val['memory'])))
                validation.append((len(ids),float(batch_loss(world,model,val,ids,rng,device))))
        score=sum(n*x for n,x in validation)/sum(n for n,_ in validation)
        entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_predictive_loss=score,
                   gate_min=float((2*model.logit.sigmoid()).min().detach()),
                   gate_max=float((2*model.logit.sigmoid()).max().detach()))
        print(json.dumps(dict(stage='neo_memory',**entry)),flush=True)
        history.append(entry)
        if not args.smoke:write_json(folder/'memory_filter_history.json',history)
        if score<best:
            best=score;best_epoch=epoch
            if not args.smoke:
                save_pt(folder/'memory_filter.pt',dict(model=model.state_dict(),config=model.config,
                    epoch=epoch,val_predictive_loss=score,train_sequences=len(train['memory']),
                    val_sequences=len(val['memory']),singleview_checkpoints=fingerprints(),
                    supervised_site_selection='Best IoU mask, nearest true-source hypothesis on TRAIN/VAL only'))
        if epoch-best_epoch>=6:break
    if not args.smoke:
        write_json(folder/'memory_filter_complete.json',dict(best_epoch=best_epoch,
            val_predictive_loss=best,checkpoint_sha256=sha(folder/'memory_filter.pt'),test_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=1337)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
