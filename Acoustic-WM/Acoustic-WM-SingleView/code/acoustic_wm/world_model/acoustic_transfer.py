"""Spherical-wave propagation with matched band integration and receiver pose."""
import torch
from torch import nn
from ..audio.spatial_features import FREQUENCIES, POOL


def to_receiver(points, position, rotation):
    """rotation maps receiver basis to world; row vectors use (p-r) R."""
    return (points-position[...,None,:]) @ rotation


class DirectTransfer(nn.Module):
    def __init__(self, microphones):
        super().__init__()
        self.register_buffer('microphones',torch.tensor(microphones,dtype=torch.float32))
        self.register_buffer('frequencies',torch.tensor(FREQUENCIES,dtype=torch.float32))
        self.register_buffer('pool',torch.tensor(POOL,dtype=torch.float32))

    def forward(self, sources):
        d = (sources[...,None,:]-self.microphones).norm(dim=-1).clamp_min(.01)
        # Remove common phase to improve precision; covariance is unchanged.
        phase = -2*torch.pi*(d-d[...,:1])[...,None,:]*self.frequencies[:,None]/343
        h = torch.polar((1/d)[...,None,:].expand_as(phase),phase)
        h = h/(h.abs().square().sum(-1,keepdim=True).sqrt().clamp_min(1e-8))
        q = h[..., :, :, None]*h[..., :, None, :].conj()
        return torch.einsum('bf,...fmn->...bmn',self.pool.to(q.dtype),q)
