"""Learnable frequency reliability for counterfactual six-pair phase rendering."""
import torch
from torch import nn
from torch.nn import functional as F
from .audio import MIC_PAIRS


class ReliablePhaseRenderer(nn.Module):
    def __init__(self,n_fft,sample_rate):
        super().__init__()
        frequency=torch.fft.rfftfreq(n_fft,1/sample_rate)
        keep=(frequency>=200)&(frequency<=3500)
        self.register_buffer('keep',keep,persistent=False)
        self.register_buffer('frequency',frequency[keep],persistent=False)
        self.reliability=nn.Sequential(nn.Conv1d(4,16,5,padding=2),nn.GELU(),nn.Conv1d(16,1,1))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.zeros_(self.reliability[-1].bias)
        self.log_temperature=nn.Parameter(torch.tensor(0.))

    def forward(self,xyz,spectrum,microphones):
        with torch.autocast(device_type=xyz.device.type,enabled=False):
            xyz=xyz.float();positions=xyz.flatten(2).transpose(1,2)
            distance=(positions[:,:,None]-microphones[None,None]).norm(dim=-1)
            scores=[]
            for left,right in MIC_PAIRS:
                a=spectrum[:,left,self.keep];b=spectrum[:,right,self.keep]
                cross=a*b.conj();mean=cross.mean(-1)
                coherence=mean.abs()/cross.abs().mean(-1).clamp_min(1e-8)
                observed=mean/mean.abs().clamp_min(1e-8)
                ild=(a.abs().mean(-1).clamp_min(1e-5).log()-b.abs().mean(-1).clamp_min(1e-5).log()).abs()
                power=(a.abs()+b.abs()).mean(-1).clamp_min(1e-5).log()
                power=power-power.mean(-1,keepdim=True)
                stats=torch.stack((coherence,power,ild,(cross/cross.abs().clamp_min(1e-5)).real.var(-1,unbiased=False)),1)
                weight=coherence*(.1+F.softplus(self.reliability(stats).squeeze(1)))
                delay=(distance[:,:,right]-distance[:,:,left])/343
                predicted=torch.exp(2j*torch.pi*delay[:,:,None]*self.frequency[None,None])
                agreement=(predicted*observed[:,None].conj()).real
                scores.append((agreement*weight[:,None]).sum(-1)/weight.sum(-1,keepdim=True).clamp_min(1e-5))
            score=torch.stack(scores).mean(0)
            # Bound weak/multimodal likelihoods; never turn nearly flat acoustic
            # evidence into an arbitrarily confident source via division by tiny SD.
            centered=score-score.mean(-1,keepdim=True)
            scale=score.std(-1,keepdim=True,unbiased=False).clamp_min(.1)
            score=centered/scale*self.log_temperature.exp().clamp(.1,5.)
            return score.reshape(xyz.shape[0],*xyz.shape[2:])
