"""Observation-only proposal generation and audio selection. No label loader imports."""
import numpy as np
import torch
from apex.model import APEX
from apex.run import tensor_inputs
from .masks import PromptMasks
from .selector import ObjectSelector, batch
from .linear_selector import LinearObjectSelector
from .objects import pool_masks, neighbors


def load_models(mask_checkpoint, selector_checkpoint, apex_checkpoint, device='cuda'):
    mc=torch.load(mask_checkpoint,map_location='cpu',weights_only=False)
    sc=torch.load(selector_checkpoint,map_location='cpu',weights_only=False)
    ac=torch.load(apex_checkpoint,map_location='cpu',weights_only=False)
    masks=PromptMasks().to(device).eval();masks.load_state_dict(mc['model'])
    selector=(LinearObjectSelector() if sc.get('kind')=='linear' else ObjectSelector()).to(device).eval()
    selector.load_state_dict(sc['model'])
    apex=APEX(expand=ac['expand']).to(device).eval();apex.load_state_dict(ac['model'])
    return masks,selector,apex,dict(mask_threshold=mc['threshold'],apex_threshold=ac['threshold'])


def diverse_seeds(xyz, score, n, min_distance=.18, excluded=None):
    seeds=[] if excluded is None else list(excluded)
    added=[]
    # A broad acoustic or point-score maximum often differs by just a few
    # floating-point ulps across GPU models. Quantize ranking at 0.1% of the
    # largest finite score and use voxel index as a deterministic tie break.
    # This stabilizes query identities without changing the spatial contract.
    score=np.asarray(score,np.float64)
    finite=np.nan_to_num(score,nan=-np.inf)
    step=max(float(np.max(np.abs(finite[np.isfinite(finite)]),initial=0))*1e-3,1e-8)
    rank=np.rint(finite/step)
    order=np.lexsort((np.arange(len(score)),-rank))
    for i in order:
        if not seeds or np.min(np.linalg.norm(xyz[seeds]-xyz[i],axis=1)) >= min_distance:
            seeds.append(int(i));added.append(int(i))
            if len(added)>=n:break
    return added


@torch.inference_mode()
def propose_and_score(x, models, device='cuda', n_seeds=48):
    """Consumes ONLY frozen features, XYZ, counts, observed acoustic/audio features.

    Labels are absent from this function signature and never loaded. Returns
    candidates and alternative predeclared scoring rules for an external evaluator.
    """
    mask_model,selector,apex,config=models
    f,xyz,acoustic,audio,count=tensor_inputs(x,device)
    apex_prob=apex(f,xyz,acoustic,audio,count)['logits'].sigmoid().cpu().numpy()
    knn=torch.as_tensor(neighbors(x['xyz']),device=device,dtype=torch.long)
    enc=mask_model.encode(f,xyz,knn)
    objectness=enc['objectness'].sigmoid().cpu().numpy()
    audio_logits=(selector.sound_logits(audio[None]) if isinstance(selector,LinearObjectSelector)
                  else selector.audio_class(selector.sound(audio[None])))
    audio_class=audio_logits.softmax(-1)[0]
    semantic_match=(enc['semantic'].softmax(-1)[:,:16]*audio_class[None]).sum(-1).cpu().numpy()
    # Independent seed sources reduce reliance on any single model's mistakes.
    n_main=n_seeds//2;n_spatial=n_seeds//3;n_general=n_seeds-n_main-n_spatial
    seeds=diverse_seeds(x['xyz'],apex_prob,n_main)
    spatial=np.sqrt(x['acoustic'][:,0].clip(1e-8))*semantic_match*np.sqrt(objectness)
    seeds+=diverse_seeds(x['xyz'],spatial,n_spatial,excluded=seeds)
    seeds+=diverse_seeds(x['xyz'],semantic_match*objectness,n_general,excluded=seeds)
    # One fixed threshold selected by the correct-prompt development diagnostic.
    # Include neighboring thresholds as candidates, not per-sample GT selection.
    thresholds=sorted(set([.35,.5,.65,round(config['mask_threshold'],3)]))
    masks=[];quality=[];origins=[]
    for start in range(0,len(seeds),8):
        group=seeds[start:start+8]
        out=mask_model.decode(enc,xyz,count,torch.tensor(group,device=device))
        probabilities=out['logits'].sigmoid().cpu().numpy()
        q=out['quality'].sigmoid().cpu().numpy()
        for j,prob in enumerate(probabilities):
            for threshold in thresholds:
                m=prob>=threshold
                if x['count'][m].sum()<10:continue
                masks.append(m);quality.append(float(q[j]));origins.append(f'prompt:{group[j]}:threshold:{threshold}')
    # A direct model prediction is a legitimate fallback candidate.
    fallback=apex_prob>=config['apex_threshold']
    if fallback.any():
        masks.append(fallback);quality.append(.5);origins.append('apex')
    if not masks:
        fallback=np.zeros(len(x['count']),bool);fallback[int(apex_prob.argmax())]=True
        masks=[fallback];quality=[.01];origins=['single_voxel_fallback']
    # Remove exact duplicates without labels, preserving their first occurrence.
    kept=[];seen=set()
    for j,m in enumerate(masks):
        key=np.packbits(m).tobytes()
        if key not in seen:seen.add(key);kept.append(j)
    masks=np.asarray([masks[j] for j in kept]);quality=np.asarray(quality)[kept]
    origins=[origins[j] for j in kept]
    memberships=[(np.flatnonzero(m),x['count'][m]) for m in masks]
    g,a=pool_masks(x,memberships)
    inp,_=batch([dict(geometry=g,acoustic=a,audio=x['audio'])],device)
    selector_score=selector(*inp)['scores'][0].cpu().numpy()
    mean_apex=np.array([np.average(apex_prob[m],weights=x['count'][m]) for m in masks])
    quality_log=np.log(quality.clip(.01,1))
    apex_log=np.log(mean_apex.clip(1e-6,1))
    scores=dict(selector=selector_score+quality_log,
                hybrid=selector_score+quality_log+apex_log,
                apex_quality=apex_log+quality_log)
    choices={k:int(v.argmax()) for k,v in scores.items()}
    rank_features=np.column_stack([g[:,512:],a,mean_apex,quality,selector_score/10]).astype(np.float32)
    return dict(masks=masks,choices=choices,scores=scores,seeds=seeds,origins=origins,
                quality=quality,apex_probability=apex_prob,rank_features=rank_features)
