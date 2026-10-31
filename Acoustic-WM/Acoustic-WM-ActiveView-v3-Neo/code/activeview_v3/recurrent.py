"""Small supervised gate on the frozen SingleView recurrent assimilation cell."""
import torch
from torch import nn


class NeoMemoryFilter(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.config = dict(hidden=hidden)
        # Zero means exactly the frozen SingleView update used by code-only Neo.
        # The bounded gate can damp or strengthen each latent channel.
        self.logit = nn.Parameter(torch.zeros(hidden))

    def forward(self, predicted, assimilated):
        if predicted.shape != assimilated.shape or predicted.shape[-1] != len(self.logit):
            raise ValueError('Incompatible per-site recurrent states')
        gate = 2*torch.sigmoid(self.logit)
        return predicted + gate*(assimilated-predicted)
