"""Local-context, point-prompted instance decoder. Geometry only; no audio/labels."""
import torch
from torch import nn
from torch.nn import functional as F
from apex.model import mlp


class PromptMasks(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.geometry = nn.Sequential(nn.LayerNorm(256), mlp([256, 192, hidden], .1))
        self.neighbor_message = mlp([hidden + 4, 96, hidden])
        self.context = nn.LayerNorm(hidden)
        self.key = mlp([hidden, hidden, 64])
        self.refined_query = mlp([hidden * 2, hidden, 64])
        self.relative = mlp([7, 32, 1])
        self.scale = nn.Linear(hidden, 1)
        self.bias = nn.Linear(hidden, 1)
        self.quality = mlp([hidden * 2 + 3, 64, 1])
        self.semantic = nn.Linear(hidden, 17)
        self.objectness = nn.Linear(hidden, 1)
        self.temperature = nn.Parameter(torch.tensor(10.))

    def encode(self, features, xyz, knn):
        h = self.geometry(features)
        offset = xyz[knn] - xyz[:, None]
        distance = offset.norm(dim=-1, keepdim=True)
        # Short-range aggregation complements the frozen scene representation.
        msg = self.neighbor_message(torch.cat([h[knn] - h[:, None], offset, distance], -1))
        weight = torch.exp(-distance / .2)
        h = self.context(h + (msg * weight).sum(1) / weight.sum(1).clamp_min(1e-6))
        return dict(h=h, keys=F.normalize(self.key(h), dim=-1),
                    semantic=self.semantic(h), objectness=self.objectness(h).squeeze(-1))

    def decode(self, encoded, xyz, count, seeds):
        h, key = encoded['h'], encoded['keys']
        qh = h[seeds]
        offset = xyz[None] - xyz[seeds, None]
        distance = offset.norm(dim=-1)
        scale = .15 + F.softplus(self.scale(qh)).clamp(max=3)
        relative = torch.cat([offset, distance[..., None],
                              (offset / scale[:, None]).clamp(-10, 10)], -1)
        prior = self.relative(relative).squeeze(-1) - distance / scale
        similarity = key[seeds] @ key.T
        initial = self.temperature.clamp(1, 30) * (similarity - .5) + prior + self.bias(qh)
        # Label-free region completion: attend to a soft candidate, not a GT mask.
        weight = initial.sigmoid() * count[None] * torch.exp(-distance / 2)
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-6)
        prototype = weight @ h
        query = F.normalize(self.refined_query(torch.cat([qh, prototype], -1)) + key[seeds], dim=-1)
        logits = self.temperature.clamp(1, 30) * (query @ key.T - .5) + prior + self.bias(qh)
        prob = logits.sigmoid()
        stats = torch.stack([(weight * prob).sum(1), (weight * distance).sum(1),
                             torch.log1p((prob * count[None]).sum(1)) / 10], -1)
        quality = self.quality(torch.cat([qh, prototype, stats], -1)).squeeze(-1)
        return dict(logits=logits, initial=initial, quality=quality)


def mask_loss(logits, target, count):
    positive = target * count[None]
    negative = (1-target) * count[None]
    bce = .5 * ((F.softplus(-logits) * positive).sum(1) / positive.sum(1).clamp_min(1)
                + (F.softplus(logits) * negative).sum(1) / negative.sum(1).clamp_min(1))
    p = logits.sigmoid()
    dice = 1 - (2*(p*positive).sum(1)+1) / ((p*count[None]).sum(1)+positive.sum(1)+1)
    return (bce + 2*dice).mean()


def objective(encoded, decoded, targets, count, category, object_ids):
    loss = mask_loss(decoded['logits'], targets, count) + .25*mask_loss(decoded['initial'], targets, count)
    with torch.no_grad():
        pred = decoded['logits'] > 0
        inter = (pred * targets * count[None]).sum(1)
        union = ((pred + targets - pred*targets) * count[None]).sum(1)
        quality_target = inter / union.clamp_min(1)
    quality = F.mse_loss(decoded['quality'].sigmoid(), quality_target)
    ce = F.cross_entropy(encoded['semantic'], category, reduction='none')
    sem = torch.stack([ce[category==k].mean() * (1 if k < 16 else .3) for k in category.unique().tolist()]).mean()
    obj = object_ids >= 0
    objectness = .5*(F.softplus(-encoded['objectness'][obj]).mean() + F.softplus(encoded['objectness'][~obj]).mean()) if obj.any() and (~obj).any() else encoded['objectness'].sum()*0
    return loss + .3*sem + .2*objectness + .3*quality
