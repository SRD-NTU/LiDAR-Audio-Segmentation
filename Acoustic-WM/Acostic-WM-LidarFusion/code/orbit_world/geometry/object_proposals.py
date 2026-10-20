"""Read-only adapter to the verified corrected ORBIT proposal pipeline.

Core ORBIT-WM training requires only exported observations, not ORBIT imports.
"""
import sys
from pathlib import Path
import numpy as np
import torch


class OrbitProposals:
    def __init__(self, config, device='cuda'):
        root=Path(config['paths']['orbit'])
        sys.dont_write_bytecode=True
        sys.path[:0]=[str(root),str(root/'runtime')]
        from apex import data as D
        from orbit.inference import load_models
        from orbit.audio_gate import AudioSeedGate
        self.D=D
        D.DATA=Path(config['paths']['dataset'])
        D.GEOMETRY=Path(config['paths']['geometry'])
        self.models=load_models(*[config['paths'][k] for k in ('masks','selector','apex')],device)
        self.gates=[]
        for k in ('gate','gate2'):
            model=AudioSeedGate().to(device).eval()
            model.load_state_dict(torch.load(config['paths'][k],map_location='cpu',weights_only=False)['model'])
            self.gates.append(model)
        self.device=device
        self.n_seeds=config.get('preparation',{}).get('seeds',48)

    @torch.inference_mode()
    def __call__(self, x):
        from orbit.inference import propose_and_score
        from orbit.evaluate_audio_gate import candidate_gate_values
        pred=propose_and_score(x,self.models,self.device,self.n_seeds)
        baseline=pred['scores']['hybrid'].copy()
        for gate in self.gates:
            baseline += candidate_gate_values(x,self.models,gate,pred,self.device)
        pred['baseline']=baseline.astype(np.float32)
        return pred


def source_hypotheses(xyz, counts, masks, n=4):
    """Candidate centre plus deterministic surface support points, never GT."""
    hypotheses=[]
    for mask in masks:
        p=xyz[mask];w=counts[mask]
        if not len(p):
            hypotheses.append(np.zeros((n,3),np.float32));continue
        center=np.average(p,axis=0,weights=w)
        # Centre and three farthest-point support sites cover extended objects.
        chosen=[int(np.argmin(np.linalg.norm(p-center,axis=1)))]
        for _ in range(n-2):
            dist=np.linalg.norm(p[:,None]-p[chosen][None],axis=-1).min(1)
            chosen.append(int(dist.argmax()))
        hypotheses.append(np.vstack([center,p[chosen]]))
    return np.asarray(hypotheses,np.float32)
