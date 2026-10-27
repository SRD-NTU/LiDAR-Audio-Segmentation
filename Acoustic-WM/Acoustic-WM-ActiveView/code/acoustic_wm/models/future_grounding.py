import torch
from torch import nn

def initial_belief(calibrator, energy, baseline):
    """The original calibration and original four-site marginal, unchanged."""
    score = calibrator(energy[None], baseline[None],
                       torch.ones_like(baseline[None], dtype=torch.bool))[0]
    site_energy = (energy * calibrator.band_logits.softmax(-1)).sum(-1)
    temperature = calibrator.log_temperature.exp().clamp(.1, 10)
    return score, (-site_energy / temperature).log_softmax(-1)

class FutureEvidenceHead(nn.Module):
    """Fifty learned parameters; the original single-view scores are frozen."""
    def __init__(self, band_logits=None):
        super().__init__()
        self.band_logits = nn.Parameter(torch.zeros(48) if band_logits is None
                                        else band_logits.detach().clone())
        self.log_gain = nn.Parameter(torch.tensor(-1.))
        self.log_temperature = nn.Parameter(torch.tensor(0.))

    def forward(self, base, log_site, future_energy, reliability, valid=None):
        # Cache quantization is an explicit part of inference, not only training.
        energy = future_energy.to(torch.float16).float()
        energy = (energy * reliability[..., None, None, :] * 48
                  * self.band_logits.softmax(-1)).sum(-1)
        evidence = torch.logsumexp(log_site - energy /
                                  self.log_temperature.exp().clamp(.1, 10), -1)
        score = base + self.log_gain.exp().clamp(.01, 20) * evidence
        if valid is not None:
            score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        return score

class FutureAcousticReasoner:
    """Frozen world-model predictions, candidate disagreement and new evidence."""
    def __init__(self, world):
        self.world = world.eval()

    @torch.inference_mode()
    def predict(self, points, features, valid, hypotheses, observed, actions):
        k,h,_=hypotheses.shape
        source=hypotheses.reshape(1,k*h,3)
        q,_,state=self.world(points[None],features[None],valid[None],source)
        memory=self.world.assimilate(state,q,observed[None,None].expand_as(q))
        n=len(actions);action=actions[:,None].expand(n,k*h,12)
        state=memory.expand(n,-1,-1);source=source.expand(n,-1,-1)
        _,_,predicted,variance=self.world.imagine(state,source,action)
        return predicted.reshape(n,k,h,48,4,4),variance.reshape(n,k,h,48)

    @staticmethod
    @torch.inference_mode()
    def information(base, log_site, predicted, log_variance, reliability):
        """Between-object predictive disagreement; no future audio is an input."""
        probability = base.softmax(-1)
        sites = log_site.exp()
        # predicted: actions, objects, source sites, bands, microphones, microphones
        mean_per_object = (predicted * sites[None, ..., None, None, None]).sum(2)
        overall = (mean_per_object * probability[None, :, None, None, None]).sum(1)
        difference = (mean_per_object - overall[:, None]).abs().square().sum((-1, -2))
        variance = (log_variance.exp() * sites[None, ..., None]).sum(2)
        within = ((predicted - mean_per_object[:, :, None]).abs().square()
                  .sum((-1, -2)) * sites[None, ..., None]).sum(2)
        standardized = difference / (variance + within).clamp_min(1e-5)
        return (standardized * probability[None, :, None]
                * reliability[None, None]).sum((1, 2))

    @staticmethod
    @torch.inference_mode()
    def evidence(prediction, log_variance, observed):
        error = (prediction - observed).abs().square().sum((-1, -2))
        return (error * (-log_variance).exp() + log_variance).half()

def relative_actions(receiver, rotation, destinations, destination_rotations):
    """Known receiver poses only; no true source coordinates are consumed."""
    translation = (destinations - receiver) @ rotation
    relative_rotation = rotation.T[None] @ destination_rotations
    return torch.cat([translation, relative_rotation.flatten(1)], -1)
