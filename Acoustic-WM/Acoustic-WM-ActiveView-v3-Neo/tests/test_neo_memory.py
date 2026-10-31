"""CPU checks for causal, pose-correct multi-view latent assimilation."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
import torch
from activeview_v3.embedding import AcousticEmbedding
from activeview_v3.models import EvidenceAdapter
from activeview_v3.multiview import SequentialSession, relative_transition
from activeview_v3.recurrent import NeoMemoryFilter
from acoustic_wm.world_model.predictive_world import move_source


def pose(translation, yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return torch.tensor(np.r_[translation, rotation.reshape(-1)], dtype=torch.float32)


class ToyWorld:
    """Deterministic stand-in that exposes transition and sensory updates."""
    def __init__(self):
        self.eye = torch.eye(4, dtype=torch.complex64)[None, None, None]

    def __call__(self, points, features, valid, source):
        n = source.shape[1]
        q = self.eye.expand(1, n, 48, 4, 4).clone()
        return q, torch.zeros(1, n, 48), torch.zeros(1, n, 128)

    def imagine(self, state, source, action):
        moved = move_source(source, action)
        next_state = state + action[..., :1]
        q = self.eye.expand(1, source.shape[1], 48, 4, 4).clone()
        q = q + .2 * next_state[..., 0, None, None, None] * self.eye
        return next_state, moved, q, torch.zeros(1, source.shape[1], 48)

    def assimilate(self, state, prediction, observation):
        innovation = (observation - prediction).real.mean((-3, -2, -1))
        return state + innovation[..., None]


class ToyAdapter:
    def __init__(self):
        self.device = 'cpu'
        self.embedding = AcousticEmbedding('cpu')
        self.evidence = EvidenceAdapter()
        self.scale = 1.
        self.frozen = type('Frozen', (), {})()
        self.frozen.model = type('Model', (), {})()
        self.frozen.model.world = ToyWorld()

    def prepare(self, row, actions, distances):
        q = torch.eye(4, dtype=torch.complex64).expand(48, 4, 4).clone()
        state = dict(hypotheses=torch.tensor([[[1., 0., 0.]], [[2., 0., 0.]]]),
                     geometry=(torch.zeros(1, 3), torch.zeros(1, 256), torch.ones(1, dtype=torch.bool)),
                     observed=q, reliability=torch.ones(48), silent=False)
        initial = dict(base=torch.zeros(1, 2), sites=torch.zeros(1, 2, 1),
                       valid=torch.ones(1, 2, dtype=torch.bool))
        return initial, dict(row=row, state=state)


class NeoMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.adapter = ToyAdapter()
        self.row = dict(masks=np.array([[1, 0], [0, 1]], dtype=bool), inverse=np.array([0, 1]))

    def session(self):
        return SequentialSession(self.adapter, self.row)

    def test_trainable_filter_starts_at_frozen_update(self):
        filter_model=NeoMemoryFilter()
        predicted=torch.randn(2,1,128)
        assimilated=torch.randn(2,1,128)
        torch.testing.assert_close(filter_model(predicted,assimilated),assimilated,atol=1e-7,rtol=1e-7)
        with torch.no_grad():filter_model.logit.fill_(-2.)
        self.assertFalse(torch.allclose(filter_model(predicted,assimilated),assimilated))

    def test_pose_composition_matches_direct_reference_transform(self):
        current = pose([1., .2, -.3], .6)
        proposed = pose([-1., .4, 2.], -.8)
        source = torch.tensor([[[2., 0., 1.], [-1., 1., 2.]]])
        via_current = move_source(move_source(source, current[None, None]),
                                  relative_transition(current, proposed)[None, None])
        direct = move_source(source, proposed[None, None])
        torch.testing.assert_close(via_current, direct, atol=1e-6, rtol=1e-6)

    def test_observation_changes_next_forecast_but_inactive_bands_do_not(self):
        a, b, c = self.session(), self.session(), self.session()
        first = pose([1., 0., 0.])
        batches = [session.prepare(None, first[None], [1.])[0] for session in (a, b, c)]
        q = torch.eye(4).expand(48, 4, 4).clone() * 1.2
        observed = torch.view_as_real(q.to(torch.complex64)).numpy()
        changed_inactive = observed.copy()
        changed_inactive[30:, :, :, 0] += 5.
        changed_active = observed.copy()
        changed_active[:24, :, :, 0] += .5
        reliability = np.r_[np.ones(24), np.zeros(24)].astype(np.float32)
        for session, batch, audio in zip((a, b, c), batches,
                                         (observed, changed_inactive, changed_active)):
            session.observe(batch, audio, reliability, 'view-2')
        torch.testing.assert_close(a.memory, b.memory, atol=0, rtol=0)
        torch.testing.assert_close(a.belief.log_probability, b.belief.log_probability, atol=0, rtol=0)
        self.assertFalse(torch.allclose(a.memory, c.memory))
        next_pose = pose([2., 0., 0.])
        forecast_a = a.prepare(None, next_pose[None], [1.])[0]['mu']
        forecast_b = b.prepare(None, next_pose[None], [1.])[0]['mu']
        forecast_c = c.prepare(None, next_pose[None], [1.])[0]['mu']
        torch.testing.assert_close(forecast_a, forecast_b, atol=0, rtol=0)
        self.assertFalse(torch.allclose(forecast_a, forecast_c))
        with self.assertRaisesRegex(ValueError, 'stale world state'):
            a.observe(batches[0], observed, reliability, 'view-3')

    def test_silent_acquisition_moves_state_without_sensory_update(self):
        session = self.session()
        first = pose([1., 0., 0.])
        batch = session.prepare(None, first[None], [1.])[0]
        q = torch.eye(4).expand(48, 4, 4).clone().to(torch.complex64)
        session.observe(batch, torch.view_as_real(q).numpy(), np.zeros(48, np.float32), 'silent')
        self.assertIsNone(session.last_observed)
        torch.testing.assert_close(session.memory, torch.ones_like(session.memory))
        self.assertEqual(session.recordings, 2)

    def test_two_acquisitions_replay_updated_latent_and_source(self):
        first, replay = self.session(), self.session()
        audio = torch.view_as_real((1.25*torch.eye(4)).to(torch.complex64).expand(48,4,4)).numpy()
        steps = (pose([1.,0.,0.], .2), pose([1.5,0.,.8], -.3))
        for index, action in enumerate(steps, start=2):
            old = first.memory.clone()
            for session in (first, replay):
                batch = session.prepare(None, action[None], [1.])[0]
                session.observe(batch, audio, np.ones(48,np.float32), 'view-'+str(index))
            self.assertFalse(torch.allclose(first.memory, old))
            torch.testing.assert_close(first.memory, replay.memory, atol=0, rtol=0)
            torch.testing.assert_close(first.source, replay.source, atol=0, rtol=0)
            torch.testing.assert_close(first.belief.log_probability, replay.belief.log_probability, atol=0, rtol=0)
        self.assertEqual(first.recordings, 3)


if __name__ == '__main__':
    unittest.main()
