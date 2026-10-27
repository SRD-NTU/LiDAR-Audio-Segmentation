"""Checkpoint selection on the actual observation-only proposal pipeline."""
import json
from pathlib import Path
import numpy as np
import torch
from apex import data as D
from .inference import propose_and_score


@torch.inference_mode()
def deployment_validation(models, device, cache, gate=None, posterior_path=None):
    for model in models[:3]:model.eval()
    if gate is not None:gate.eval()
    provenance=json.loads((Path(cache)/'COMPLETE.json').read_text())
    D.GEOMETRY=Path(provenance['geometry'])
    rows,post,emb,_=D.audio_bank('val')
    if posterior_path is None and provenance.get('spatial_directory'):
        posterior_path=Path(provenance['spatial_directory'])/'adaptive_val_posterior.npy'
    if posterior_path is not None:post=D.load_posterior(posterior_path,rows)
    results=[]
    for i,rel in enumerate(rows):
        x=D.load_inputs(rel,post[i],emb[i])
        prediction=propose_and_score(x,models,device)
        scores=prediction['scores']['hybrid'].copy()
        if gate is not None:
            from .evaluate_audio_gate import candidate_gate_values
            scores+=candidate_gate_values(x,models,gate,prediction,device)
        chosen=int(scores.argmax())
        target=D.load_targets(rel,x['inverse'],x['count'])
        masks=prediction['masks'];positive=target['positive']
        inter=(masks*positive[None]).sum(1)
        union=(masks*x['count'][None]).sum(1)+positive.sum()-inter
        ious=np.where(union==0,1.,inter/union.clip(1))
        results.append(dict(rel=rel,iou=float(ious[chosen]),oracle=float(ious.max())))
    return dict(n=len(results),mean_iou=float(np.mean([r['iou'] for r in results])),
                candidate_oracle_mean_iou=float(np.mean([r['oracle'] for r in results])),
                deployable_segmentation_result=True,selection_split='val',test_used=False,
                samples=results)
