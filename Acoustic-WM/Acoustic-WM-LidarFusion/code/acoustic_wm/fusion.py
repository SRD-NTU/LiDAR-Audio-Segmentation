"""Two measured LiDAR frames; row-vector transforms match the acoustic world.

An action stores [receiver translation in the reference frame, R_ref.T R_new].
New-frame points therefore map to reference coordinates as x_new @ R.T + t.
Registration is supplied by calibrated poses, not estimated by this module.
"""
from dataclasses import dataclass

import numpy as np


def validate_actions(actions):
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or not len(values):
        raise ValueError('Actions must have nonempty shape [N,12]')
    if not np.isfinite(values).all():
        raise ValueError('Actions contain nonfinite values')
    rotation = values[:, 3:].reshape(-1, 3, 3)
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-4)):
        raise ValueError('Receiver rotations must be orthonormal with determinant +1')
    return values.astype(np.float32)


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('receiver_to_world must be a finite 4x4 rigid transform')
    if not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-6):
        raise ValueError('Invalid homogeneous pose bottom row')
    validate_actions(np.r_[pose[:3, 3], pose[:3, :3].ravel()][None])
    return pose


def actions_from_poses(reference_pose, destination_poses):
    reference = validate_pose(reference_pose)
    destinations = np.stack([validate_pose(p) for p in destination_poses])
    rotation = reference[:3, :3]
    delta = (destinations[:, :3, 3] - reference[:3, 3]) @ rotation
    relative = rotation.T[None] @ destinations[:, :3, :3]
    return validate_actions(np.concatenate([delta, relative.reshape(-1, 9)], axis=1))


def validate_cloud(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('LiDAR xyz must be nonempty finite [N,3] in metres')
    return xyz


@dataclass
class FusedCloud:
    xyz: np.ndarray
    reference_to_fused: np.ndarray
    second_to_fused: np.ndarray
    view_membership: np.ndarray


def fuse_clouds(reference_xyz, second_xyz, action, voxel_size_m=0.02):
    """Keep one measured representative per voxel, prioritizing reference points.

    Membership bits are 1=reference, 2=second, 3=both. Inverse maps preserve the
    original point order of each scan. A zero voxel size retains every point.
    Voxel occupancy is a coverage proxy, not an occlusion/visibility label.
    """
    reference = validate_cloud(reference_xyz)
    second = validate_cloud(second_xyz)
    action = validate_actions(np.asarray(action).reshape(1, -1))[0]
    if not np.isfinite(voxel_size_m) or voxel_size_m < 0:
        raise ValueError('voxel_size_m must be finite and nonnegative')
    aligned = second @ action[3:].reshape(3, 3).T + action[:3]
    combined = np.concatenate([reference, aligned])
    if voxel_size_m:
        cells = np.floor(combined.astype(np.float64) / voxel_size_m)
        if np.abs(cells).max() >= np.iinfo(np.int64).max:
            raise ValueError('Voxel coordinates exceed int64 range')
        _, first, inverse = np.unique(cells.astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        # Sort by measurement order; np.unique alone sorts by voxel coordinates.
        order = np.argsort(first)
        remap = np.empty(len(order), dtype=np.int64)
        remap[order] = np.arange(len(order))
        inverse = remap[inverse]
        fused = combined[first[order]]
    else:
        fused = combined
        inverse = np.arange(len(combined), dtype=np.int64)
    reference_map, second_map = inverse[:len(reference)], inverse[len(reference):]
    membership = np.zeros(len(fused), dtype=np.uint8)
    np.bitwise_or.at(membership, reference_map, 1)
    np.bitwise_or.at(membership, second_map, 2)
    return FusedCloud(fused, reference_map, second_map, membership)


class LidarFusionPipeline:
    """Freeze the query using reference data, then acquire and fuse that view.

    The encoder must regenerate proposals and geometric features from fused
    points. This is not nearest-neighbour extension of an existing mask.
    The acoustic weights are reused without fusion-specific retraining.
    """
    def __init__(self, model, encoder, voxel_size_m=0.02):
        if not np.isfinite(voxel_size_m) or voxel_size_m < 0:
            raise ValueError('voxel_size_m must be finite and nonnegative')
        self.model = model
        self.encoder = encoder
        self.voxel_size_m = voxel_size_m

    def run(self, reference_xyz, reference_wave, actions, acquire,
            *, valid=None, movement_cost=None, cost_weight=0., rate=44100):
        from .audio.predictive_features import covariance48_from_wave
        if rate != 44100:
            raise ValueError('Expected 44100 Hz audio')
        reference_xyz = validate_cloud(reference_xyz)
        actions = validate_actions(actions)
        valid = np.ones(len(actions), bool) if valid is None else np.asarray(valid, bool)
        if valid.shape != (len(actions),) or not valid.any():
            raise ValueError('At least one valid receiver action is required')
        costs = (np.linalg.norm(actions[:, :3], axis=1) if movement_cost is None
                 else np.asarray(movement_cost, dtype=np.float32))
        if costs.shape != (len(actions),) or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError('Movement costs must be finite nonnegative [N]')
        if not np.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError('cost_weight must be finite and nonnegative')
        observed, reliability = covariance48_from_wave(reference_wave)
        reference_row = self.encoder.encode_arrays(reference_xyz, reference_wave, rate)
        available = np.flatnonzero(valid)
        plan = self.model.plan(reference_row, observed, reliability, actions[available])
        raw_utility = plan['utilities'].detach().cpu().numpy()
        utility = raw_utility - cost_weight * costs[available]
        local_query = int(utility.argmax())
        query = int(available[local_query])
        # If a movement penalty changes the query, obtain predictions for that
        # same selected action without reading either future sensor stream.
        if local_query != plan['query']:
            plan = self.model.plan(reference_row, observed, reliability, actions[query:query+1])
        plan['query'] = query
        initial_result = self.model.result(plan, plan['base'])

        # This is the ONLY acquisition call, after a final action is fixed.
        second_xyz, second_wave, second_rate = acquire(query)
        if second_rate != rate:
            raise ValueError('Both audio observations must be 44100 Hz')
        future_observed, future_reliability = covariance48_from_wave(second_wave)
        reference_result = self.model.complete(plan, future_observed, future_reliability)

        cloud = fuse_clouds(reference_xyz, second_xyz, actions[query], self.voxel_size_m)
        fused_row = self.encoder.encode_fused_arrays(cloud.xyz, reference_wave, rate,
                    second_receiver_position=actions[query, :3], view_membership=cloud.view_membership)
        # This second prediction is a post-acquisition rescore of the NEW fused
        # candidates. It is not the blind forecast that selected the query.
        fused_plan = self.model.plan(fused_row, observed, reliability, actions[query:query+1])
        fused_plan['silent'] = not bool(np.any(reliability) or np.any(future_reliability))
        fused_result = self.model.complete(fused_plan, future_observed, future_reliability)
        mask = np.asarray(fused_result['sound_mask'], dtype=np.uint8)
        probability = np.asarray(fused_result['point_probability'], dtype=np.float32)
        if mask.shape != (len(cloud.xyz),) or probability.shape != mask.shape:
            raise ValueError('Fused encoder inverse must map predictions to every fused input point')
        full_utility = np.full(len(actions), np.nan, dtype=np.float32)
        full_utility[available] = raw_utility
        scored_utility = np.full(len(actions), -np.inf, dtype=np.float32)
        scored_utility[available] = utility
        return dict(
            xyz=cloud.xyz, sound_mask=mask, point_probability=probability,
            posterior=fused_result['posterior'], selected_object=fused_result['selected_object'],
            reference_sound_mask=mask[cloud.reference_to_fused],
            second_sound_mask=mask[cloud.second_to_fused],
            reference_point_probability=probability[cloud.reference_to_fused],
            second_point_probability=probability[cloud.second_to_fused],
            reference_to_fused=cloud.reference_to_fused, second_to_fused=cloud.second_to_fused,
            view_membership=cloud.view_membership,
            second_only_voxel=(cloud.view_membership == 2),
            matched_activeview_sound_mask=reference_result['sound_mask'],
            matched_activeview_posterior=reference_result['posterior'],
            matched_singleview_sound_mask=initial_result['sound_mask'],
            chosen_action=np.int64(query), actions=actions, valid_actions=valid,
            movement_cost=costs, query_disagreement=full_utility,
            query_utility=scored_utility, cost_weight=np.float32(cost_weight),
            reference_band_reliability=reliability,
            second_band_reliability=future_reliability,
            # The blind pre-acquisition forecast remains on the reference bank.
            prior_predicted_covariance=np.stack([
                plan['prediction'].real.detach().cpu().numpy(),
                plan['prediction'].imag.detach().cpu().numpy()], axis=-1),
            prior_log_error_scale=plan['log_variance'].detach().cpu().numpy(),
            point_clouds=np.int64(2), audio_observations=np.int64(2),
            coordinate_frame=np.str_('reference_receiver_x_forward_y_left_z_up_metres'),
            geometry_source=np.str_('two_measured_scans_with_known_rigid_poses'),
            voxel_size_m=np.float32(self.voxel_size_m),
            fusion_specific_training=np.bool_(False),
        )
