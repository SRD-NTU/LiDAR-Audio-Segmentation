"""Training-supervised latent sound types with a marginal audio emission.

Type names/labels are never inputs at inference. Marginalizing over predicted
types avoids pretending that an object determines its exact recording.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PrototypeEmission(nn.Module):
    def __init__(self, feature_dim=256, types=16, active_types=None):
        super().__init__()
        self.register_buffer('centre',torch.zeros(feature_dim))
        self.register_buffer('coefficient',torch.zeros(feature_dim,types))
        self.register_buffer('bias',torch.zeros(types))
        self.register_buffer('audio_means',torch.zeros(types if active_types is None else active_types,512))
        self.register_buffer('temperature',torch.tensor(.2))

    def type_distribution(self,feature):
        logits=(F.normalize(feature.float(),dim=-1)-self.centre)@self.coefficient+self.bias
        return (logits/self.temperature.clamp(.01,10)).log_softmax(-1)

    def log_probability(self,feature,audio):
        observation=F.normalize(audio.float(),dim=-1)@F.normalize(self.audio_means,dim=-1).T*50
        # Any remaining type is a silent/non-source state. Observed nonzero
        # audio has no support under that state; its probability is retained
        # in the full type softmax rather than renormalized away.
        return torch.logsumexp(self.type_distribution(feature)[...,:len(self.audio_means)]+observation,-1)


class LibraryEmission(nn.Module):
    def __init__(self,feature_dim=256,types=17,active_types=16,components=80):
        super().__init__();self.classifier=PrototypeEmission(feature_dim,types,active_types)
        self.register_buffer('components',torch.zeros(active_types,components,512))
        self.register_buffer('valid',torch.zeros(active_types,components,dtype=torch.bool))

    def audio_likelihood(self,audio,excluded=None):
        observation=torch.einsum('...d,cmd->...cm',F.normalize(audio.float(),dim=-1),F.normalize(self.components,dim=-1))*50
        valid=self.valid.clone()
        if excluded is not None:valid[excluded[0],excluded[1]]=False
        return torch.logsumexp(observation.masked_fill(~valid,-torch.inf),-1)-valid.sum(-1).clamp_min(1).log()

    def log_probability(self,feature,audio,excluded=None):
        prior=self.classifier.type_distribution(feature)[...,:len(self.components)]
        return torch.logsumexp(prior+self.audio_likelihood(audio,excluded),-1)


def load_emission(checkpoint,device='cpu'):
    config=checkpoint['config']
    if config.get('library'):
        model=LibraryEmission(config['feature_dim'],config['types'],config['active_types'],config['components'])
    else:model=PrototypeEmission(config['feature_dim'],config['types'],config.get('active_types'))
    model.load_state_dict(checkpoint['model']);return model.to(device).eval()
