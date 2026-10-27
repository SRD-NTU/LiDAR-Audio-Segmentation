"""Tests of V1's scientific invariants and acquisition boundary."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from acoustic_wm.v1_belief import JointBelief
from acoustic_wm.v1_pipeline import ActiveViewV1
from acoustic_wm.v1_verifier import InstanceVerifier
from acoustic_wm.action_protocol import actions_from_poses, valid_actions


class BeliefTests(unittest.TestCase):
    def test_exact_duplicate_does_not_gain_instance_mass(self):
        original = JointBelief.from_v0(torch.zeros(2), torch.zeros(2, 1),
                                       [0, 1], [1, 1])
        duplicate = JointBelief.from_v0(torch.zeros(3), torch.zeros(3, 1),
                                        [0, 1, 0], [2, 1, 2])
        self.assertTrue(torch.allclose(original.group_probability(),
                                       duplicate.group_probability(), atol=1e-7))
        original = original.updated(torch.tensor([[1.0], [0.0]]))
        duplicate = duplicate.updated(torch.tensor([[1.0], [0.0], [1.0]]))
        self.assertTrue(torch.allclose(original.group_probability(),
                                       duplicate.group_probability(), atol=1e-7))

    def test_real_future_favors_better_explanation_and_backpropagates(self):
        torch.manual_seed(3)
        verifier = InstanceVerifier()
        identity = torch.eye(4, dtype=torch.complex64).expand(48, 4, 4)
        forecast = torch.stack([identity, identity * 0.5]).reshape(2, 1, 48, 4, 4)
        scale = torch.zeros(2, 1, 48)
        reliability = torch.full((48,), 1 / 48)
        likelihood = verifier(forecast, scale, identity, reliability)
        self.assertGreater(float(likelihood[0, 0].detach()),
                           float(likelihood[1, 0].detach()))
        (-likelihood[0, 0] + likelihood[1, 0]).backward()
        self.assertIsNotNone(verifier.correction[-1].weight.grad)
        self.assertTrue(torch.isfinite(verifier.correction[-1].weight.grad).all())


class AcquisitionTests(unittest.TestCase):
    def test_empty_action_menu_is_a_valid_zero_query_case(self):
        actions = actions_from_poses(np.eye(4), [])
        _, valid, _, _ = valid_actions(actions)
        self.assertEqual(actions.shape, (0, 12))
        self.assertEqual(valid.size, 0)

    def test_plan_precedes_only_selected_acquisition(self):
        events = []
        engine = ActiveViewV1.__new__(ActiveViewV1)
        engine.initialize = lambda *_args: events.append("initialize")
        engine.plan = lambda *_args, **_kwargs: (events.append("plan") or {"query": 0})
        engine.advance = lambda *_args: events.append("advance")
        engine.result = lambda *_args: {"events": list(events)}
        pose = np.eye(4)
        destination = np.eye(4)
        destination[0, 3] = 1.0

        def acquire(query):
            events.append("acquire")
            self.assertEqual(query, 0)
            return None, None

        result = engine.run({}, None, None, pose, [destination], acquire, max_queries=1)
        self.assertEqual(result["events"], ["initialize", "plan", "acquire", "advance"])


if __name__ == "__main__":
    unittest.main()
