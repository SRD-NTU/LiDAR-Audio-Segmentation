"""Low-capacity alternative: class compatibility plus spatial likelihood.

Audio logits retain pretrained text priors and a trainable linear residual.
The full model is trained on TRAIN masks only. No inference category gates.
"""
import torch
from torch import nn
from .objects import GEOM_DIM


class LinearObjectSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.geom_class=nn.Linear(GEOM_DIM,17)
        self.audio_residual=nn.Linear(512,16)
        nn.init.zeros_(self.audio_residual.weight);nn.init.zeros_(self.audio_residual.bias)
        self.register_buffer('text_features',torch.zeros(16,512))
        self.register_buffer('text_scale',torch.tensor(1.))
        self.register_buffer('mean',torch.zeros(GEOM_DIM))
        self.register_buffer('std',torch.ones(GEOM_DIM))
        # Separate mean/peak evidence retains small sources and broad objects.
        self.weights=nn.Parameter(torch.tensor([1.,.6,.2]))

    def sound_logits(self,audio):
        return audio@self.text_features.T*self.text_scale+self.audio_residual(audio)

    def forward(self,geometry,acoustic,audio,valid):
        g=((geometry-self.mean)/self.std).clamp(-8,8)
        gc=self.geom_class(g)
        ac=self.sound_logits(audio)
        compatibility=torch.logsumexp(gc.log_softmax(-1)[...,:16]+ac.log_softmax(-1)[:,None],-1)
        scores=(self.weights[0].clamp(.1,3)*compatibility
                +self.weights[1].clamp(0,2)*acoustic[...,16]*14
                +self.weights[2].clamp(0,2)*acoustic[...,17]*14)
        return dict(scores=scores.masked_fill(~valid,-1e4),geom_class=gc,audio_class=ac)
