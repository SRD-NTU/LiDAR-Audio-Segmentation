"""Audio-conditioned source-object query over frozen 3D mask proposals."""
import torch
from torch import nn
from torch.nn import functional as F


class AudioSeedGate(nn.Module):
    """Score each geometric object query using waveform identity and DOA evidence.

    Mask generation stays class agnostic. This head learns which proposed
    complete instance is sounding, with negatives from other scene objects.
    """
    def __init__(self):
        super().__init__()
        self.audio=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Linear(128,96))
        self.score=nn.Sequential(nn.LayerNorm(343),nn.Linear(343,128),nn.GELU(),
                                 nn.Dropout(.15),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,encoded,decoded,acoustic,audio,count,seeds,sound_prob):
        """All arguments derive from observed XYZ/WAV and frozen model outputs."""
        h=encoded['h'];prob=decoded['logits'].sigmoid()
        w=prob*count[None];mass=w.sum(1,keepdim=True).clamp_min(1.)
        w=w/mass
        pooled=w@h
        seed_h=h[seeds]
        sound_h=self.audio(audio)[None].expand(len(seeds),-1)
        seed_ac=acoustic[seeds]
        pooled_ac=w@acoustic
        semantic=encoded['semantic'].softmax(-1)
        seed_sem=semantic[seeds]
        pooled_sem=w@semantic
        match_seed=(seed_sem[:,:16]*sound_prob[None]).sum(-1).clamp_min(1e-8).log()[:,None]
        match_mask=(pooled_sem[:,:16]*sound_prob[None]).sum(-1).clamp_min(1e-8).log()[:,None]
        objectness=encoded['objectness'][seeds].sigmoid()[:,None]
        quality=decoded['quality'].sigmoid()[:,None]
        size=mass.log1p()/10
        vector=torch.cat([seed_h,pooled,sound_h,seed_ac,pooled_ac,seed_sem,
                          pooled_sem,objectness,quality,size,match_seed,match_mask],-1)
        return self.score(vector).squeeze(-1)


def sample_seeds(row,rng,max_other=7,max_background=4):
    ids=row['node_object'];positive=np.flatnonzero(row['positive']>0)
    if len(positive)==0:raise ValueError('No source voxel')
    seeds=[int(rng.choice(positive))]
    valid=np.unique(ids[ids>=0]);valid=valid[valid!=int(row['source_object'])]
    for object_id in rng.choice(valid,min(len(valid),max_other),replace=False):
        nodes=np.flatnonzero(ids==object_id)
        seeds.append(int(rng.choice(nodes)))
    background=np.flatnonzero(ids<0)
    if len(background):
        seeds.extend(map(int,rng.choice(background,min(len(background),max_background),replace=False)))
    return seeds


# Numpy is needed only by the TRAIN sampling helper, never by forward().
import numpy as np
