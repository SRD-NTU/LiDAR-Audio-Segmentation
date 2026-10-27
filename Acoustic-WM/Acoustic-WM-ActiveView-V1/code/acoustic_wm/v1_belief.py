"""Persistent joint belief over measured mask versions and source sites.

Geometric groups are proxies for physical instances. Exact duplicate masks do
not gain prior mass; labels are never needed to build this inference belief.
"""
from dataclasses import dataclass

import torch


@dataclass
class JointBelief:
    log_joint: torch.Tensor  # [candidate, source site], normalized globally
    groups: torch.Tensor  # [candidate]
    group_count: int

    @classmethod
    def from_v0(cls, base, log_site, groups, duplicates):
        if base.ndim != 1 or log_site.ndim != 2 or log_site.shape[0] != len(base):
            raise ValueError("Expected candidate scores [K] and site scores [K,H]")
        gid = torch.as_tensor(groups, dtype=torch.long, device=base.device)
        dup = torch.as_tensor(duplicates, dtype=base.dtype, device=base.device)
        if gid.shape != base.shape or dup.shape != base.shape or (dup < 1).any():
            raise ValueError("Invalid geometric groups or duplicate counts")
        count = int(gid.max()) + 1
        if sorted(gid.unique().tolist()) != list(range(count)):
            raise ValueError("Group IDs must be consecutive")
        adjusted = base - dup.log()
        scores = []
        conditional = torch.empty_like(base)
        for g in range(count):
            members = gid == g
            local = adjusted[members]
            normalizer = torch.logsumexp(local, 0)
            unique_count = (1 / dup[members]).sum()
            scores.append(normalizer - unique_count.log())
            conditional[members] = local - normalizer
        group_prior = torch.stack(scores).log_softmax(0)
        joint = group_prior[gid, None] + conditional[:, None] + log_site.log_softmax(-1)
        return cls(joint - torch.logsumexp(joint.flatten(), 0), gid, count)

    def updated(self, site_log_likelihood):
        if site_log_likelihood.shape != self.log_joint.shape:
            raise ValueError("Evidence must have one value per candidate and source site")
        joint = self.log_joint + site_log_likelihood
        joint = joint - torch.logsumexp(joint.flatten(), 0)
        return JointBelief(joint, self.groups, self.group_count)

    def candidate_probability(self):
        return self.log_joint.exp().sum(-1)

    def group_probability(self):
        result = self.log_joint.new_zeros(self.group_count)
        return result.scatter_add(0, self.groups, self.candidate_probability())

    def entropy(self):
        probability = self.group_probability().clamp_min(1e-12)
        return -(probability * probability.log()).sum()

    def point_probability(self, masks, inverse):
        # This is a mask-posterior marginal, not a calibrated point probability.
        probability = self.candidate_probability()
        candidate_masks = torch.as_tensor(masks, dtype=probability.dtype, device=probability.device)
        mapping = torch.as_tensor(inverse, dtype=torch.long, device=probability.device)
        return (probability @ candidate_masks)[mapping]
