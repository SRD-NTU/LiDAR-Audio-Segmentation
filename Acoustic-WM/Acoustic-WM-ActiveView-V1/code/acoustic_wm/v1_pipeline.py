"""V1 ActiveView: persistent instance/site belief and branch-value planning.

All action utilities depend on the current belief and predicted future sounds.
The acquisition callback is invoked only after an action has been selected.
"""
from pathlib import Path

import numpy as np
import torch

from .action_protocol import actions_from_poses, mask_groups, valid_actions
from .model import ReleasedModel
from .v1_belief import JointBelief
from .v1_verifier import InstanceVerifier


class ActiveViewV1:
    def __init__(self, directory, device="cuda", checkpoint=None, *, allow_v0_bootstrap=False,
                 grouping_threshold=0.9, imagined_outcomes=8):
        self.directory = Path(directory).resolve()
        self.device = device
        self.v0 = ReleasedModel(self.directory, device)
        self.world = self.v0.world
        self.verifier = InstanceVerifier(self.v0.head).to(device).eval()
        self.grouping_threshold = float(grouping_threshold)
        self.imagined_outcomes = int(imagined_outcomes)
        if not 0 < self.grouping_threshold <= 1 or self.imagined_outcomes < 1:
            raise ValueError("Invalid grouping threshold or imagined-outcome count")
        self.training_identity = "v0-bootstrap-untrained"
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("schema") != "activeview-v1-instance-verification-v1":
                raise ValueError("Incompatible V1 checkpoint")
            self.world.load_state_dict(saved["world"])
            self.verifier.load_state_dict(saved["verifier"])
            self.training_identity = str(Path(checkpoint).resolve())
        elif not allow_v0_bootstrap:
            raise FileNotFoundError("A trained V1 checkpoint is required; use --allow-v0-bootstrap only for functional checks")
        self.world.eval().requires_grad_(False)
        self.verifier.eval().requires_grad_(False)

    def _complex(self, value):
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.shape[-1] != 2:
            raise ValueError("Complex covariance must store real and imaginary channels")
        return torch.view_as_complex(tensor.contiguous())

    @torch.inference_mode()
    def initialize(self, row, observed, reliability):
        state = self.v0.current(row, observed, reliability)
        groups, duplicates, stats = mask_groups(state["masks"], self.grouping_threshold)
        self.belief = JointBelief.from_v0(state["base"], state["log_site"], groups, duplicates)
        self.state = state
        self.group_stats = stats
        self.sites = state["hypotheses"]
        k, h, _ = self.sites.shape
        initial, _, memory = self.world(*[x[None] for x in state["geometry"]],
                                         self.sites.reshape(1, k * h, 3))
        self.memory = self.world.assimilate(memory, initial,
                                            state["observed"][None, None].expand_as(initial))
        self.current_reliability = state["reliability"]
        self.silent = state["silent"]

    @torch.inference_mode()
    def _forecasts(self, actions):
        n = len(actions)
        k, h, _ = self.sites.shape
        action = torch.as_tensor(actions, device=self.device)
        source = self.sites.reshape(1, k * h, 3).expand(n, -1, -1)
        memory = self.memory.expand(n, -1, -1)
        expanded = action[:, None].expand(n, k * h, 12)
        moved_memory, moved_sites, q, lv = self.world.imagine(memory, source, expanded)
        return (moved_memory, moved_sites.reshape(n, k, h, 3),
                q.reshape(n, k, h, 48, 4, 4), lv.reshape(n, k, h, 48))

    @torch.inference_mode()
    def _expected_instance_entropy_reduction(self, prediction, log_scale, moved_sites, action):
        """Discrete predicted-outcome approximation; no future recording is read."""
        joint_probability = self.belief.log_joint.exp()
        h = prediction.shape[1]
        group_site = joint_probability.new_zeros((self.belief.group_count, h))
        group_site.index_add_(0, self.belief.groups, joint_probability)
        values, indices = group_site.flatten().topk(min(self.imagined_outcomes, group_site.numel()))
        outcome_weight = values / values.sum().clamp_min(1e-12)
        prior_entropy = self.belief.entropy()
        expected = prior_entropy.new_zeros(())
        for weight, index in zip(outcome_weight, indices):
            group = int(index) // h
            site = int(index) % h
            members = (self.belief.groups == group).nonzero(as_tuple=True)[0]
            candidate = int(members[self.belief.log_joint[members, site].argmax()])
            pseudo_observation = prediction[candidate, site]
            likelihood = self.verifier(prediction, log_scale, pseudo_observation,
                                       self.current_reliability, moved_sites, action)
            expected = expected + weight * self.belief.updated(likelihood).entropy()
        return (prior_entropy - expected).clamp_min(0)

    @torch.inference_mode()
    def plan(self, current_pose, destination_poses, available, costs, *, max_translation_m=3.0):
        actions = actions_from_poses(current_pose, destination_poses)
        actions, valid, costs, distances = valid_actions(actions, available, costs, max_translation_m)
        utilities = np.full(len(actions), -np.inf, np.float32)
        ids = np.flatnonzero(valid)
        if not len(ids):
            return {"query": -1, "utilities": utilities, "valid": valid,
                    "actions": actions, "distances": distances, "costs": costs}
        moved_memory, moved_sites, prediction, scale = self._forecasts(actions[ids])
        for j, original in enumerate(ids):
            utilities[original] = float(self._expected_instance_entropy_reduction(
                prediction[j], scale[j], moved_sites[j],
                torch.as_tensor(actions[original], device=self.device)))
        if not np.isfinite(utilities[ids]).all():
            raise FloatingPointError("Nonfinite V1 action utility")
        query = int(ids[np.argmax(utilities[ids])])
        selected = int(np.flatnonzero(ids == query)[0])
        return {"query": query, "utilities": utilities, "valid": valid,
                "actions": actions, "distances": distances, "costs": costs,
                "memory": moved_memory[selected:selected + 1],
                "sites": moved_sites[selected], "prediction": prediction[selected],
                "log_scale": scale[selected]}

    @torch.inference_mode()
    def advance(self, plan, observed, reliability):
        if plan["query"] < 0:
            raise ValueError("Cannot advance without an acquired observation")
        observed = self._complex(observed)
        reliability = torch.as_tensor(reliability, dtype=torch.float32, device=self.device)
        if observed.shape != (48, 4, 4) or reliability.shape != (48,):
            raise ValueError("Expected future covariance [48,4,4,2] and reliability [48]")
        likelihood = self.verifier(plan["prediction"], plan["log_scale"], observed,
                                   reliability, plan["sites"],
                                   torch.as_tensor(plan["actions"][plan["query"]], device=self.device))
        self.belief = self.belief.updated(likelihood)
        k, h, _ = self.sites.shape
        predicted = plan["prediction"].reshape(1, k * h, 48, 4, 4)
        self.memory = self.world.assimilate(plan["memory"], predicted,
                                            observed[None, None].expand_as(predicted))
        self.sites = plan["sites"]
        self.current_reliability = reliability
        self.silent = self.silent and not bool(reliability.any())

    @torch.inference_mode()
    def result(self, plans, max_translation_m):
        probability = self.belief.candidate_probability()
        selected = int(probability.argmax())
        masks = self.state["masks"][:, self.state["inverse"]]
        hard = masks[selected].astype(np.uint8).copy()
        soft = self.belief.point_probability(self.state["masks"], self.state["inverse"]).cpu().numpy().astype(np.float32)
        if self.silent:
            hard[:] = 0
            soft[:] = 0
        chosen = np.asarray([p["query"] for p in plans], np.int64)
        utilities = np.stack([p["utilities"] for p in plans]) if plans else np.empty((0, 0), np.float32)
        distances = np.asarray([p["distances"][p["query"]] for p in plans], np.float32)
        costs = np.asarray([p["costs"][p["query"]] for p in plans], np.float32)
        first = plans[0] if plans else None
        return dict(sound_mask=hard, point_probability=soft,
                    posterior_candidate=probability.cpu().numpy(),
                    posterior_group=self.belief.group_probability().cpu().numpy(),
                    posterior_site=self.belief.log_joint.exp().cpu().numpy(),
                    group_id=self.belief.groups.cpu().numpy(),
                    source_sites_current=self.sites.cpu().numpy(),
                    selected_object=np.int64(selected),
                    original_selected_object=np.int64(int(self.state["base"].argmax())),
                    selected_group=np.int64(int(self.belief.groups[selected])),
                    chosen_actions=chosen, chosen_action=np.int64(chosen[0] if len(chosen) else -1),
                    query_utilities=utilities, travel_distances_m=distances,
                    movement_costs=costs, total_distance_m=np.float32(distances.sum()),
                    total_movement_cost=np.float32(costs.sum()),
                    budget_used=np.int64(len(chosen)),
                    audio_observations=np.int64(1 + len(chosen)), point_clouds=np.int64(1),
                    max_translation_m=np.float32(max_translation_m),
                    grouping_threshold=np.float32(self.grouping_threshold),
                    geometric_group_count=np.int64(self.group_stats["groups"]),
                    action_policy=np.str_("predicted_instance_entropy_reduction_proxy"),
                    checkpoint_identity=np.str_(self.training_identity),
                    prior_predicted_covariance=(torch.view_as_real(first["prediction"]).cpu().numpy()
                                                if first else np.empty((0,), np.float32)),
                    prior_log_error_scale=(first["log_scale"].cpu().numpy()
                                           if first else np.empty((0,), np.float32)))

    @torch.inference_mode()
    def run(self, row, observed, reliability, reference_pose, destination_poses,
            acquire, *, valid=None, costs=None, max_queries=1, max_translation_m=3.0):
        if max_queries < 0 or max_queries > len(destination_poses):
            raise ValueError("Invalid query budget")
        self.initialize(row, observed, reliability)
        original = actions_from_poses(reference_pose, destination_poses)
        _, initial_valid, initial_costs, _ = valid_actions(original, valid, costs, max_translation=1e12)
        _, initially_local_valid, _, _ = valid_actions(
            original, initial_valid, initial_costs, max_translation=max_translation_m)
        visited = np.zeros(len(destination_poses), bool)
        current_pose = reference_pose
        plans = []
        for step in range(max_queries):
            # Menus supply costs from the reference. Later legs use measured
            # Euclidean displacement because pairwise path costs are absent.
            step_costs = (initial_costs if step == 0 else
                          np.linalg.norm(actions_from_poses(current_pose, destination_poses)[:, :3], axis=1))
            plan = self.plan(current_pose, destination_poses, initial_valid & ~visited,
                             step_costs, max_translation_m=max_translation_m)
            if plan["query"] < 0:
                break
            query = plan["query"]
            # No future sound or LiDAR is opened until this call.
            second, second_reliability = acquire(query)
            self.advance(plan, second, second_reliability)
            plans.append(plan)
            visited[query] = True
            current_pose = destination_poses[query]
        result = self.result(plans, max_translation_m)
        result.update(initial_relative_actions=original,
                      initial_action_valid=initially_local_valid,
                      supplied_movement_cost=initial_costs)
        return result
