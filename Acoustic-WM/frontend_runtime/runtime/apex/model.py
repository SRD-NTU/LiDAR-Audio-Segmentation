"""APEX: direct masks with soft acoustic region expansion.

Frozen XYZ-only Sonata features are adapted to the task. Three soft foreground
prototypes at different acoustic temperatures provide region context. This is
a research candidate, not a claim of a novel or established SOTA architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(dims, dropout=0.):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers += [nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class APEX(nn.Module):
    def __init__(self, hidden=96, expand=True):
        super().__init__()
        self.hidden, self.expand = hidden, expand
        self.geometry = nn.Sequential(nn.LayerNorm(256), mlp([256, 192, hidden], .1))
        self.audio = mlp([512, 128, hidden], .1)
        self.semantic = nn.Linear(hidden, 17)
        self.audio_class = nn.Linear(hidden, 16)
        self.base = mlp([hidden * 2 + 12, 128, 64, 1], .1)
        self.refine = mlp([hidden * 2 + 12 + 9, 128, 64, 1], .1)

    def forward(self, features, xyz, acoustic, audio, count):
        h = self.geometry(features)
        a = self.audio(audio)
        geo = torch.cat([xyz / 10, xyz.norm(dim=1, keepdim=True) / 10], 1)
        x = torch.cat([h, h * a[None], geo, acoustic], 1)
        initial = self.base(x).squeeze(-1)
        if self.expand:
            # Area weighting prevents voxelization density from silently changing
            # the meaning of the pooled representation. Keep several temperatures.
            lp = torch.log(acoustic[:, 0].clamp_min(1e-5))
            weights = torch.stack([initial + t * lp + .5 * count.log()
                                   for t in (0., .5, 1.5)], 1).softmax(dim=0)
            prototypes = weights.T @ h
            centres = weights.T @ xyz
            affinity = h @ F.normalize(prototypes, dim=1).T / self.hidden ** .5
            distance = torch.stack([(xyz - c).norm(dim=1) for c in centres], 1)
            dot = F.normalize(xyz, dim=1) @ F.normalize(centres, dim=1).T
            # Learned residual can expand beyond acoustic peaks onto the object.
            residual = self.refine(torch.cat([x, affinity, distance / 5, dot], 1)).squeeze(-1)
            logits = initial + residual
        else:
            logits = initial
        return dict(logits=logits, initial=initial, semantic=self.semantic(h),
                    audio_class=self.audio_class(a))


def objective(out, target, count, category, source):
    """Per-observation balanced mask losses and auxiliary semantic supervision."""
    def mask_loss(logits):
        pos = count * target
        neg = count * (1 - target)
        bce = .5 * ((F.softplus(-logits) * pos).sum() / pos.sum().clamp_min(1)
                    + (F.softplus(logits) * neg).sum() / neg.sum().clamp_min(1))
        p = logits.sigmoid()
        dice = 1 - (2 * (p * pos).sum() + 1) / ((p * count).sum() + pos.sum() + 1)
        return bce + 2 * dice
    loss = mask_loss(out['logits']) + .25 * mask_loss(out['initial'])
    ce = F.cross_entropy(out['semantic'], category, reduction='none')
    class_losses = [ce[category == k].mean() * (1. if k < 16 else .25)
                    for k in category.unique().tolist()]
    semantic = torch.stack(class_losses).mean()
    audio = F.cross_entropy(out['audio_class'][None], source.reshape(1))
    return loss + .2 * semantic + .1 * audio
