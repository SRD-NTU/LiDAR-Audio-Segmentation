"""Behavioral tests for evidence accumulation and sequential movement."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import seed_all
from activeview_v3.multiview import JointBelief, component_increment
from activeview_v3.navigation import ReachableSpace
from activeview_v3.route import RouteSpace
from activeview_v3.models import EvidenceAdapter
import numpy as np
import torch
import unittest


class BeliefTests(unittest.TestCase):
    def setUp(self):
        seed_all(19)
        self.prior = torch.tensor([.2, -.4, 1.])
        self.sites = torch.randn(3, 4).log_softmax(-1)

    def test_prior_once_and_all_observations(self):
        belief = JointBelief(self.prior, self.sites)
        likelihoods = [torch.randn(3, 4) for _ in range(4)]
        for step, ll in enumerate(likelihoods):
            belief.update(ll, str(step))
            expected = self.prior.log_softmax(-1)[:, None]+self.sites+sum(likelihoods[:step+1])
            expected -= expected.flatten().logsumexp(0)
            torch.testing.assert_close(belief.log_probability, expected)
        self.assertFalse(torch.allclose(belief.marginals()[1], self.sites))

    def test_silence_and_duplicate_guard(self):
        belief = JointBelief(self.prior, self.sites)
        before = belief.log_probability.clone()
        belief.update(torch.randn(3, 4), 'silent', False)
        torch.testing.assert_close(belief.log_probability, before)
        with self.assertRaises(ValueError):
            belief.update(torch.randn(3, 4), 'silent')

    def test_first_observation_matches_two_view_mixture(self):
        evidence = EvidenceAdapter(dimension=5)
        batch = dict(base=self.prior[None], sites=self.sites[None], valid=torch.ones(1, 3, dtype=torch.bool),
                     mu=torch.randn(1, 3, 4, 5), var=torch.rand(1, 3, 4, 5))
        target = torch.randn(1, 5)
        for scale in [.25, .7, 1., 2.]:
            belief = JointBelief(evidence.prior(batch)[0], self.sites)
            belief.update(component_increment(evidence, batch, target, scale)[0], 'new')
            torch.testing.assert_close(belief.marginals()[0], evidence.scores(batch, target, scale)[0].log_softmax(-1))

    def test_replay_is_identical(self):
        likelihoods = [torch.randn(3, 4) for _ in range(4)]
        first, replay = JointBelief(self.prior, self.sites), JointBelief(self.prior, self.sites)
        for i, ll in enumerate(likelihoods):
            first.update(ll, str(i))
            replay.update(ll.clone(), str(i))
        torch.testing.assert_close(first.log_probability, replay.log_probability, rtol=0, atol=0)


class Path:
    pass


class FlatMap:
    def snap_point(self, point):
        return point

    def find_path(self, path):
        path.geodesic_distance = np.linalg.norm(path.requested_end-path.requested_start)
        path.points = [path.requested_start, path.requested_end]
        return True


class RouteTests(unittest.TestCase):
    def setUp(self):
        anchor = ReachableSpace(FlatMap(), Path, [0, 0, 0])
        self.route = RouteSpace(anchor, [2, 0, 0], [[0, 1.2, 0], [2, 1.2, 0]])

    def test_actual_leg_and_anchor_distance(self):
        pose = self.route.project([2.5, 0], 0)
        self.assertAlmostEqual(pose['path_distance_m'], .5)
        self.assertAlmostEqual(pose['anchor_distance_m'], 2.5)
        np.testing.assert_allclose(pose['path'][0], [2, 0, 0])

    def test_limits_and_no_revisits(self):
        self.assertIsNone(self.route.project([-2, 0], 0))
        self.assertIsNone(self.route.project([3.2, 0], 0))
        self.assertIsNone(self.route.project([2.1, 0], 1.))
        poses = self.route.sample(np.random.default_rng(4), 32)
        self.assertEqual(len(poses), 32)
        self.assertTrue(all(.15 <= p['path_distance_m'] <= 3 and p['anchor_distance_m'] <= 3 for p in poses))


if __name__ == '__main__':
    unittest.main()
