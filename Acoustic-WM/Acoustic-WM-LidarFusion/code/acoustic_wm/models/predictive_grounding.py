import math
import torch
from torch import nn

class PredictiveReasoner(nn.Module):
    def __init__(self,bands=48,baseline_only=False):
        super().__init__();self.baseline_only=baseline_only
        self.log_prior=nn.Parameter(torch.tensor(0.))
        self.log_scale=nn.Parameter(torch.tensor(-1.))
        self.log_temperature=nn.Parameter(torch.tensor(0.))
        self.band_logits=nn.Parameter(torch.zeros(bands))
    def forward(self,energy,baseline,valid):
        score=self.log_prior.exp().clamp(.05,5)*baseline
        if not self.baseline_only:
            e=(energy*self.band_logits.softmax(-1)).sum(-1)
            ll=torch.logsumexp(-e/self.log_temperature.exp().clamp(.1,10),-1)-math.log(e.shape[-1])
            score=score+self.log_scale.exp().clamp(.01,20)*ll
        return score.masked_fill(~valid,torch.finfo(score.dtype).min)
