"""A metric graph world with geometry-only latent acoustic hypotheses."""
import torch
from torch import nn
from torch.nn import functional as F


class SurfaceGraphBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(dim + 4, 32), nn.GELU(), nn.Linear(32, 1))
        self.update = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, centers, neighbors, valid):
        relative = (centers[neighbors] - centers[:, None]) / .35
        radius = relative.norm(dim=-1, keepdim=True)
        difference = (h[neighbors] - h[:, None]).abs()
        edge_logits = self.edge(torch.cat((difference, relative, radius), -1)).squeeze(-1)
        weight = edge_logits.sigmoid() * valid
        # Residual self-message makes isolated nodes well defined.
        pooled = (h + (h[neighbors] * weight[..., None]).sum(1)) / (1 + weight.sum(1, keepdim=True))
        return self.norm(h + self.update(torch.cat((h, pooled), -1))), edge_logits


class AcousticGraphWorld(nn.Module):
    def __init__(self, dim=96, layers=3, use_graph=True, audio_prior_dropout=.35):
        super().__init__()
        self.use_graph = use_graph
        self.audio_prior_dropout=float(audio_prior_dropout)
        self.geometry = nn.Sequential(nn.Linear(268, 128), nn.LayerNorm(128), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(128, dim), nn.LayerNorm(dim))
        self.graph = nn.ModuleList([SurfaceGraphBlock(dim) for _ in range(layers)])
        self.context = nn.Linear(dim, dim)
        self.semantic = nn.Linear(dim, 17)
        self.renderer = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 512))
        self.center_offsets = nn.Linear(dim, 3)
        nn.init.zeros_(self.center_offsets.weight); nn.init.zeros_(self.center_offsets.bias)
        self.audio = nn.Sequential(nn.Linear(533, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(.2))
        self.reliability = nn.Sequential(nn.Linear(75, 64), nn.GELU(), nn.Linear(64, 54))
        nn.init.zeros_(self.reliability[-1].weight); nn.init.zeros_(self.reliability[-1].bias)
        self.physics_trust = nn.Sequential(nn.Linear(75,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
        self.mask = nn.Sequential(nn.Linear(dim * 3 + 110, 128), nn.GELU(),
                                 nn.Dropout(.15), nn.Linear(128, 1))
        self.semantic_scale = nn.Parameter(torch.tensor(.1))
        self.render_scale = nn.Parameter(torch.tensor(.1))
        self.physics_scale = nn.Parameter(torch.tensor(.1))
        self.teacher_adapter = nn.Sequential(nn.Linear(41, dim), nn.LayerNorm(dim), nn.GELU())
        self.teacher_geometry_scale = nn.Parameter(torch.tensor(.1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.))

    def construct_world(self, features, centers, neighbors, edge_valid, teacher_geometry=None):
        """Construct reusable geometry-only state for static-scene audio queries."""
        h = self.geometry(features)
        if teacher_geometry is not None:
            h = h + self.teacher_geometry_scale.tanh() * self.teacher_adapter(teacher_geometry)
        edge_logits = None
        if self.use_graph:
            for layer in self.graph:
                h, edge_logits = layer(h, centers, neighbors, edge_valid)
        # Coarse metric cell context: not a scene ID or annotation partition.
        _, inverse = torch.unique(torch.floor(centers / .4).long(), dim=0, return_inverse=True)
        cells = int(inverse.max()) + 1
        coarse = h.new_zeros(cells, h.shape[1]).index_add(0, inverse, h)
        count = h.new_zeros(cells).index_add(0, inverse, h.new_ones(len(h)))
        h = h + self.context((coarse / count[:, None].clamp_min(1))[inverse])
        semantic = self.semantic(h)
        rendered = F.normalize(self.renderer(h), dim=-1)
        predicted_centers = centers + self.center_offsets(h).tanh() * 1.5
        return dict(h=h, semantic=semantic, rendered=rendered,
                    predicted_centers=predicted_centers, edge_logits=edge_logits)

    def forward(self, features, centers, neighbors, edge_valid, acoustic,
                embedding, text_logits, audio_context, delay_tables=None,
                teacher_geometry=None, teacher_probability=None, learned_audio_logits=None,
                world_state=None):
        if world_state is not None and self.training:
            raise ValueError('Reusable world state is inference-only; training reconstructs the current graph')
        world = self.construct_world(features,centers,neighbors,edge_valid,teacher_geometry) if world_state is None else world_state
        h,semantic,rendered = (world[k] for k in ('h','semantic','rendered'))
        predicted_centers,edge_logits = (world[k] for k in ('predicted_centers','edge_logits'))
        if len(h)!=len(features):
            raise ValueError('Cached world state does not match the observed node count')
        observed = F.normalize(embedding, dim=-1)
        render_score = (rendered * observed[None]).sum(-1) * 10
        prior = (text_logits / 2).softmax(-1) if learned_audio_logits is None else learned_audio_logits.softmax(-1)
        if self.training and learned_audio_logits is not None and torch.rand((),device=prior.device)<self.audio_prior_dropout:
            # Only two training recordings/class: avoid treating a memorized
            # supervised category posterior as an infallible source identifier.
            weight=torch.rand((),device=prior.device)
            prior=weight*prior+(1-weight)*(text_logits/2).softmax(-1)
        context = torch.cat((prior, audio_context))
        audio = self.audio(torch.cat((observed, context)))
        field = acoustic.float()
        if delay_tables is not None:
            from .features import MICROPHONES, PAIRS
            microphones = torch.as_tensor(MICROPHONES, device=centers.device)
            distance = (predicted_centers[:, None] - microphones[None]).norm(dim=-1)
            delays = torch.stack([(distance[:, right] - distance[:, left]) / 343 for left, right in PAIRS], -1)
            position = ((delays + .00036) / .00072 * 160).clamp(0, 159.999)
            lower = position.long(); upper = lower + 1
            pair = torch.arange(6, device=centers.device)[None, :, None]
            band = torch.arange(9, device=centers.device)[None, None, :]
            left = delay_tables[pair, band, lower[:, :, None]]
            right = delay_tables[pair, band, upper[:, :, None]]
            field = (left + (right - left) * (position - lower)[:, :, None]).flatten(1)
        standardized = (field - field.mean(0)) / field.std(0, unbiased=False).clamp_min(.1)
        coherence = delay_tables.float().square().mean(-1).sqrt().flatten() if delay_tables is not None else field.square().mean(0).sqrt()
        acoustic_context = torch.cat((context,coherence))
        reliability = self.reliability(acoustic_context).softmax(-1)
        physical = standardized @ reliability * self.physics_trust(acoustic_context).squeeze(-1)
        match = (semantic.float().softmax(-1)[:, 1:] * prior[None]).sum(-1)
        semantic_logit = torch.logit(match.clamp(1e-5, 1 - 1e-5))
        joint = torch.cat((h, audio.expand_as(h), h * audio[None], field, standardized,
                           render_score[:, None], semantic_logit[:, None]), -1)
        logits = self.mask(joint).squeeze(-1)
        logits = logits + self.semantic_scale * semantic_logit + self.render_scale * render_score + self.physics_scale * physical
        logit_delta = self.refinement_scale.tanh() * logits if teacher_probability is not None else logits
        if teacher_probability is not None:
            logits = torch.logit(teacher_probability.float().clamp(1e-5, 1 - 1e-5)) + logit_delta
        return dict(logits=logits, semantic=semantic, rendered=rendered,
                    render_score=render_score, edge_logits=edge_logits, predicted_centers=predicted_centers,
                    logit_delta=logit_delta)


def world_loss(output, targets, neighbors, edge_valid):
    logits = output['logits'].float()
    counts, positives = targets['counts'].float(), targets['positives'].float()
    fraction = positives / counts.clamp_min(1)
    error = F.binary_cross_entropy_with_logits(logits, fraction, reduction='none')
    probability = logits.sigmoid()
    focal = ((1 - probability) * fraction + probability * (1 - fraction)).square()
    segmentation = (error * focal * positives).sum() / positives.sum().clamp_min(1)
    segmentation += (error * focal * (counts - positives)).sum() / (counts - positives).sum().clamp_min(1)
    dice = 1 - (2 * (probability * positives).sum() + 1) / ((probability * counts).sum() + positives.sum() + 1)
    classes = targets['semantic'].long()
    ce = F.cross_entropy(output['semantic'].float(), classes, reduction='none')
    class_counts = torch.bincount(classes, minlength=17).float()
    weights = (class_counts.sum() / class_counts.clamp_min(1)).sqrt().clamp_max(20)
    weights[0] *= .2
    semantic = (ce * weights[classes]).mean()
    source_weights = positives / positives.sum().clamp_min(1)
    rendering = -(source_weights * output['render_score'].float().log_softmax(0)).sum()
    predicted_audio = (output['rendered'].float() * source_weights[:, None]).sum(0)
    consistency = 1 - F.cosine_similarity(predicted_audio[None], targets['embedding'][None].float()).mean()
    affinity = logits.sum() * 0
    if output['edge_logits'] is not None:
        instance = targets['instance'].long()
        same = (instance[:, None] == instance[neighbors]).float()
        valid = edge_valid * ((classes[:, None] > 0) | (classes[neighbors] > 0))
        edge_error = F.binary_cross_entropy_with_logits(output['edge_logits'].float(), same, reduction='none')
        affinity = (edge_error * valid).sum() / valid.sum().clamp_min(1)
    centers_loss = logits.sum() * 0
    if 'centers' in targets:
        instance = targets['instance'].long()
        foreground = (classes > 0) & (instance >= 0)
        if foreground.any():
            _, inverse = torch.unique(instance[foreground], return_inverse=True)
            groups = int(inverse.max()) + 1
            mass = counts[foreground]
            grouped_mass = mass.new_zeros(groups).index_add(0, inverse, mass)
            centroid = targets['centers'][foreground].new_zeros(groups, 3).index_add(
                0, inverse, targets['centers'][foreground] * mass[:, None]) / grouped_mass[:, None].clamp_min(1)
            expected = targets['instance_centers'][foreground] if 'instance_centers' in targets else centroid[inverse]
            error = F.smooth_l1_loss(output['predicted_centers'][foreground], expected,
                                     beta=.1, reduction='none').mean(-1)
            weighting = mass / grouped_mass[inverse].clamp_min(1)
            centers_loss = (error * weighting).sum() / groups
    total = segmentation + 2 * dice + .25 * semantic + .05 * rendering + .2 * consistency + .2 * affinity + .2 * centers_loss
    return dict(total=total, segmentation=segmentation, dice=dice, semantic=semantic,
                rendering=rendering, consistency=consistency, affinity=affinity, centers=centers_loss)
