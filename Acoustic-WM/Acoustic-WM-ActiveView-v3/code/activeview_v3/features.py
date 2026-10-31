"""Read-only frozen model and strictly pre-acquisition feature computation."""
from .runtime import BASE
import numpy as np
import torch
from acoustic_wm.model import ReleasedModel
from .covariance import pack

MAX_CANDIDATES = 193

class FrozenSingleView:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = ReleasedModel(BASE, device)
        for module in (self.model.world, self.model.selector, self.model.emission):
            module.eval().requires_grad_(False)

    @torch.no_grad()
    def prepare(self, row, observed, reliability, actions):
        """No future observation, target mask, source position, or source ID input."""
        model = self.model
        if 'pointbind_feature' not in row:
            from acoustic_wm.models.pointbind_adapter import PointBindEncoder, proposal_points
            if model.pointbind is None:
                model.pointbind = PointBindEncoder(self.device)
            clouds = np.stack([proposal_points(row['xyz'], m) for m in row['masks']])
            features = []
            for begin in range(0, len(clouds), 16):
                with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(self.device).type == 'cuda'):
                    features.append(model.pointbind.points(model.tensor(clouds[begin:begin+16])).half().float())
            row = dict(row, pointbind_feature=torch.cat(features).cpu().numpy())
        state = model.current(row, observed, reliability)
        k, h, _ = state['hypotheses'].shape
        p, f, valid = state['geometry']
        source = state['hypotheses'].reshape(1, k*h, 3)
        q0, lv0, latent = model.world(p[None], f[None], valid[None], source)
        memory = model.world.assimilate(latent, q0, state['observed'][None, None].expand_as(q0))
        site = state['log_site'].exp()
        q0 = q0.reshape(k, h, 48, 4, 4)
        current = (q0 * site[..., None, None, None]).sum(1)
        context = (memory.reshape(k, h, -1) * site[..., None]).sum(1)
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action.ndim != 2 or action.shape[-1] != 12 or not torch.isfinite(action).all():
            raise ValueError('Actions must be finite N by 12 relative rigid poses')
        r = action[:, 3:].reshape(-1, 3, 3)
        if len(r) and (not torch.allclose(r.mT @ r, torch.eye(3, device=r.device).expand_as(r), atol=1e-4)
                       or not torch.allclose(torch.linalg.det(r), torch.ones(len(r), device=r.device), atol=1e-4)):
            raise ValueError('Action rotations must be proper orthogonal matrices')
        means, variances, full_q, full_lv = [], [], [], []
        for a in action:
            _, _, q, lv = model.world.imagine(memory, source, a[None, None].expand(1, k*h, -1))
            q = q.reshape(k, h, 48, 4, 4)
            lv = lv.reshape(k, h, 48)
            mean = (q * site[..., None, None, None]).sum(1)
            var = ((lv.exp() + (q - mean[:, None]).abs().square().sum((-1, -2))) * site[..., None]).sum(1)
            means.append(pack(mean)); variances.append(var.clamp_min(1e-5).log())
            full_q.append(q); full_lv.append(lv)
        body = model.tensor(np.asarray(row['pointbind_feature']).astype(np.float32))
        semantic = model.emission.log_probability(body, model.tensor(row['audio']))
        feature = dict(base=state['base'], sites=state['log_site'], latent=context,
                       rank=model.tensor(np.asarray(row['rank_features'], np.float32)),
                       support=state['hypotheses'].flatten(1), semantic=semantic,
                       current_residual=pack(state['observed'][None] - current),
                       reference_weight=state['reliability'], actions=action,
                       q=torch.stack(means) if means else torch.empty(0, k, 48, 16, device=self.device),
                       lv=torch.stack(variances) if variances else torch.empty(0, k, 48, device=self.device),
                       candidate_valid=torch.ones(k, dtype=torch.bool, device=self.device),
                       initial_silent=torch.tensor(state['silent'], device=self.device))
        # Cache precision is an explicit deployment contract, avoiding train/replay drift.
        for key in ('latent', 'current_residual', 'q', 'lv'):
            feature[key] = feature[key].half()
        return feature, dict(state=state, row=row, q=full_q, lv=full_lv, body=body)

    @torch.no_grad()
    def acquired(self, prepared, action_index, observed, reliability, audio_embedding):
        """Called only after choosing the action. Geometry always stays at reference."""
        model = self.model
        obs = torch.view_as_complex(model.tensor(np.asarray(observed, np.float32)).contiguous())
        q, lv = prepared['q'][action_index], prepared['lv'][action_index]
        energy = (q - obs).abs().square().sum((-1, -2)) * (-lv).exp() + lv
        sem = model.emission.log_probability(prepared['body'], model.tensor(np.asarray(audio_embedding, np.float32)))
        return dict(target=pack(obs), weight=model.tensor(np.asarray(reliability, np.float32)),
                    energy=energy.half(), future_semantic=sem)
