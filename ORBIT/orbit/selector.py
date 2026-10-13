"""Learn source selection from whole-object geometry and calibrated audio evidence."""
import torch
from torch import nn
from torch.nn import functional as F
from apex.model import mlp
from .objects import GEOM_DIM, ACOUSTIC_DIM


class ObjectSelector(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.geometry = nn.Sequential(nn.LayerNorm(512), mlp([512, 256, hidden], .15))
        self.sound = nn.Sequential(nn.LayerNorm(512), mlp([512, 256, hidden], .15))
        self.geom_class = nn.Linear(hidden, 17)
        self.audio_class = nn.Linear(hidden, 16)
        self.cross = mlp([hidden * 2 + 12 + ACOUSTIC_DIM + 3, 192, 96, 1], .15)
        self.prior_weights = nn.Parameter(torch.tensor([.5, .5]))
        self.register_buffer('mean', torch.zeros(GEOM_DIM))
        self.register_buffer('std', torch.ones(GEOM_DIM))

    def sound_logits(self, audio):
        """Return the audio class logits used by the selector and query gate."""
        return self.audio_class(self.sound(audio))

    def forward(self, geometry, acoustic, audio, valid):
        normalized = ((geometry - self.mean) / self.std).clamp(-10, 10)
        h = self.geometry(normalized[..., :512])
        s = self.sound(audio)
        gc, ac = self.geom_class(h), self.audio_class(s)
        log_g = gc.log_softmax(-1)[..., :16]
        log_a = ac.log_softmax(-1)[:, None]
        compatible = torch.logsumexp(log_g + log_a, -1)
        cosine = (F.normalize(h, dim=-1) * F.normalize(s, dim=-1)[:, None]).sum(-1)
        prior = acoustic[..., 16] * 14  # log mean SRP likelihood
        prior_relative = prior - prior.masked_fill(~valid, -1e4).amax(1, keepdim=True)
        z = torch.cat([h, h * s[:, None], normalized[..., 512:], acoustic,
                       compatible[..., None] / 8, cosine[..., None], prior_relative[..., None] / 14], -1)
        scores = self.cross(z).squeeze(-1)
        scores = scores + self.prior_weights[0] * compatible + self.prior_weights[1] * prior
        return dict(scores=scores.masked_fill(~valid, -1e4), geom_class=gc, audio_class=ac)


def batch(rows, device, augment=False):
    b, k = len(rows), max(len(r['geometry']) for r in rows)
    g = torch.zeros(b, k, GEOM_DIM, device=device)
    a = torch.zeros(b, k, ACOUSTIC_DIM, device=device)
    sound = torch.stack([torch.as_tensor(r['audio'], device=device) for r in rows])
    valid = torch.zeros(b, k, dtype=torch.bool, device=device)
    cat = torch.full((b, k), -100, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        n = len(r['geometry'])
        g[i, :n] = torch.as_tensor(r['geometry'], device=device)
        a[i, :n] = torch.as_tensor(r['acoustic'], device=device)
        valid[i, :n] = True
        if 'categories' in r:
            cat[i, :n] = torch.as_tensor(r['categories'], device=device)
    if augment:
        # Shared-channel semantic perturbation, not unphysical microphone phases.
        sound = sound + torch.randn_like(sound) * .005
    target = torch.tensor([r.get('target', -100) for r in rows], dtype=torch.long, device=device)
    source = torch.tensor([r.get('source', -100) for r in rows], dtype=torch.long, device=device)
    return (g, a, sound, valid), (cat, target, source)


def loss(output, labels, category_weight):
    cat, target, source = labels
    return (F.cross_entropy(output['scores'], target)
            + .5 * F.cross_entropy(output['geom_class'].transpose(1, 2), cat, weight=category_weight)
            + .5 * F.cross_entropy(output['audio_class'], source))
