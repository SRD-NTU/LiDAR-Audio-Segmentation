"""Observed-future likelihood for each persistent instance/source-site hypothesis."""
import torch
from torch import nn


class InstanceVerifier(nn.Module):
    """Physics-backed covariance error plus a learned, data-grounded correction.

    The correction starts at zero. Training compares explanations of the *same*
    recorded future sound; no objective artificially separates predictions.
    """

    def __init__(self, v0_head=None):
        super().__init__()
        self.band_logits = nn.Parameter(torch.zeros(48))
        self.log_gain = nn.Parameter(torch.tensor(-1.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.correction = nn.Sequential(nn.Linear(150, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        if v0_head is not None:
            with torch.no_grad():
                self.band_logits.copy_(v0_head.band_logits)
                self.log_gain.copy_(v0_head.log_gain)
                self.log_temperature.copy_(v0_head.log_temperature)

    def forward(self, predicted, log_variance, observed, reliability, sites=None, action=None):
        if predicted.shape[-3:] != (48, 4, 4) or log_variance.shape != predicted.shape[:-2]:
            raise ValueError("Expected predicted covariance [...,48,4,4] and log scale [...,48]")
        if observed.shape[-3:] != (48, 4, 4):
            raise ValueError("Expected observed complex covariance [...,48,4,4]")
        error = (predicted - observed).abs().square().sum((-1, -2))
        scale = log_variance.float().clamp(-8, 8)
        energy = error.float() * (-scale).exp() + scale
        weight = torch.broadcast_to(reliability.float(), energy.shape)
        weighted = (energy * weight * 48 * self.band_logits.softmax(-1)).sum(-1)
        lead = energy.shape[:-1]
        source = energy.new_zeros((*lead, 3)) if sites is None else torch.broadcast_to(sites.float() / 10, (*lead, 3))
        delta = energy.new_zeros((*lead, 3)) if action is None else torch.broadcast_to(action[..., :3].float() / 3, (*lead, 3))
        features = torch.cat([energy.clamp(-20, 20) / 10, scale / 8,
                              weight * 48, source, delta], -1)
        correction = self.correction(features).squeeze(-1)
        return (-self.log_gain.exp().clamp(.01, 20) * weighted /
                self.log_temperature.exp().clamp(.1, 10) + correction)
