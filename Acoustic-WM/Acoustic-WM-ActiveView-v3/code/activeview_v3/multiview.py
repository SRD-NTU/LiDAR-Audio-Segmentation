"""Sequential rollout of frozen v3: retain a joint object/site posterior.

The SingleView prior is counted once. New observations are conditionally
independent given the object/site under the calibrated acoustic likelihood.
No sequence training is implied: the world retains its reference memory and
conditions forecasts on the updated posterior and its reweighted context.
"""
from .covariance import pack
from .inference import mask_overlap
import numpy as np
import torch


def component_increment(evidence, batch, target, scale):
    """Preserve the deployed object-mixture updater, including its calibration.

    The conditional site posterior uses the component likelihood. Object-level
    temperature/neural corrections are constant across that object's sites.
    At the selected scale=1, non-neural checkpoint this is simply log likelihood.
    """
    mean, variance = evidence.distribution(batch)
    ll = evidence.component_log_likelihood(batch,mean,variance,target,measured=True)
    mixture = torch.logsumexp(batch['sites'] + ll, -1)
    desired = evidence.scores(batch, target, scale, measured=True) - evidence.prior(batch)
    return ll + (desired - mixture)[..., None]


class JointBelief:
    def __init__(self, prior, sites):
        self.log_probability = prior.log_softmax(-1)[:, None] + sites.log_softmax(-1)
        self.seen = set()

    def marginals(self):
        objects = self.log_probability.logsumexp(-1)
        return objects, self.log_probability - objects[:, None]

    def update(self, likelihood, observation_id, audible=True):
        if observation_id in self.seen:
            raise ValueError('Observation already assimilated: ' + observation_id)
        if likelihood.shape != self.log_probability.shape or not torch.isfinite(likelihood).all():
            raise ValueError('Invalid component likelihood')
        if audible:
            joint = self.log_probability + likelihood
            self.log_probability = joint - joint.flatten().logsumexp(0)
        self.seen.add(observation_id)


class SequentialSession:
    @torch.no_grad()
    def __init__(self, adapter, row):
        self.adapter = adapter
        self.device = adapter.device
        identity = np.r_[np.zeros(3), np.eye(3).reshape(-1)].astype('float32')[None]
        initial, self.prepared = adapter.prepare(row, identity, [0.])
        self.row = self.prepared['row']
        self.state = self.prepared['state']
        self.k, self.h, _ = self.state['hypotheses'].shape
        model = adapter.frozen.model
        p, f, valid = self.state['geometry']
        self.source = self.state['hypotheses'].reshape(1, self.k*self.h, 3)
        q0, _, latent = model.world(p[None], f[None], valid[None], self.source)
        self.memory = model.world.assimilate(latent, q0, self.state['observed'][None, None].expand_as(q0))
        self.reference_q = q0.reshape(self.k, self.h, 48, 4, 4)
        self.belief = JointBelief(adapter.evidence.prior(initial)[0], initial['sites'][0])
        self.prior_scale = adapter.evidence.log_prior_scale.exp().clamp(.25, 4)
        self.overlap = torch.as_tensor(mask_overlap(self.row), device=self.device)
        self.audible = not bool(self.state['silent'])
        self.recordings = 1

    @torch.no_grad()
    def prepare(self, row, actions, distances):
        # row is deliberately ignored: only the whitelisted initial observation is retained.
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action.ndim != 2 or action.shape[-1] != 12 or not torch.isfinite(action).all():
            raise ValueError('Expected finite N by 12 reference-relative rigid poses')
        rotation = action[:, 3:].reshape(-1, 3, 3)
        if (not torch.allclose(rotation.mT @ rotation, torch.eye(3, device=self.device).expand_as(rotation), atol=1e-4)
                or not torch.allclose(torch.linalg.det(rotation), torch.ones(len(rotation), device=self.device), atol=1e-4)):
            raise ValueError('Invalid action rotation')
        mu, var = [], []
        for a in action:
            _, _, q, lv = self.adapter.frozen.model.world.imagine(
                self.memory, self.source, a[None, None].expand(1, self.k*self.h, -1))
            mu.append(self.adapter.embedding.encode_complex(q.reshape(self.k, self.h, 48, 4, 4)).half().float())
            var.append(self.adapter.embedding.variance(lv.reshape(self.k, self.h, 48)).half().float())
        objects, sites = self.belief.marginals()
        weights = sites.exp()
        context = (self.memory.reshape(self.k, self.h, -1)*weights[..., None]).sum(1).half()
        current = (self.reference_q*weights[..., None, None, None]).sum(1)
        residual = pack(self.state['observed'][None] - current).half()
        residual = (self.adapter.embedding.encode_packed(residual)
                    - self.adapter.embedding.encode_packed(torch.zeros_like(residual))).half().float()
        values = dict(base=objects/self.prior_scale, sites=sites, latent=context,
                      residual=residual, support=self.state['hypotheses'].flatten(1),
                      initial_silent=torch.tensor(not self.audible, device=self.device), overlap=self.overlap)
        n = len(action)
        batch = {key: value[None].expand(n, *value.shape) for key, value in values.items()}
        batch.update(mu=torch.stack(mu), var=torch.stack(var), actions=action,
                     distance=torch.as_tensor(distances, dtype=torch.float32, device=self.device),
                     valid=torch.ones(n, self.k, dtype=torch.bool, device=self.device))
        return batch, self.prepared

    def utility(self, batch, method='world', draws=16):
        return self.adapter.utility(batch, method, draws)

    @torch.no_grad()
    def observe(self, batch, observed, reliability, observation_id):
        if len(batch['base']) != 1:
            raise ValueError('Exactly one committed action must be assimilated')
        obs = torch.as_tensor(np.asarray(observed, np.float32), device=self.device)
        target = self.adapter.embedding.encode_complex(torch.view_as_complex(obs.contiguous()))[None]
        batch = dict(batch, raw_target=pack(torch.view_as_complex(obs.contiguous()))[None],
                     band_reliability=torch.as_tensor(np.asarray(reliability,np.float32),device=self.device)[None])
        increment = component_increment(self.adapter.evidence, batch, target, self.adapter.scale)[0]
        audible = bool(np.asarray(reliability).sum() > 0)
        self.belief.update(increment, observation_id, audible)
        self.audible |= audible
        self.recordings += 1
        return self.prediction()

    def prediction(self):
        scores, _ = self.belief.marginals()
        chosen = int(scores.argmax()) if self.audible else -1
        mask = (np.asarray(self.row['masks'][chosen], bool)[np.asarray(self.row['inverse'], np.int64)]
                if chosen >= 0 else np.zeros(len(self.row['inverse']), dtype=bool))
        return dict(candidate=chosen, mask=mask, scores=scores.cpu().numpy(),
                    joint_log_probability=self.belief.log_probability.cpu().numpy())
