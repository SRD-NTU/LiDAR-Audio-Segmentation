"""Validate committed 3D destinations without reselecting a navmesh floor.

The original x/z proposal projection can be non-idempotent on stairs. Keep its
successful behavior; on failure validate the already committed 3D point directly.
This repair neither resamples the policy action nor relaxes movement constraints.
"""
import numpy as np


def validate_committed_pose(space, requested):
    floor = np.asarray(requested['floor'], dtype=float)
    position = np.asarray(requested['position'], dtype=float)
    yaw = float(requested['yaw'])
    if floor.shape != (3,) or position.shape != (3,) or not np.isfinite(np.r_[floor, position, yaw]).all():
        raise ValueError('Invalid committed pose: non-finite rigid position')
    projected = space.project(floor[[0, 2]], yaw)
    if projected is not None and np.allclose(projected['position'], position, atol=1e-4):
        return projected
    anchor = getattr(space, 'anchor', space)
    current = getattr(space, 'current', space)
    snapped = np.asarray(anchor.pathfinder.snap_point(floor), dtype=float)
    if not np.allclose(snapped, floor, atol=1e-4, rtol=0):
        raise ValueError('Invalid committed pose: 3D floor is off the navmesh')
    if not np.allclose(position, floor + [0., anchor.height, 0.], atol=1e-4, rtol=0):
        raise ValueError('Invalid committed pose: receiver height')
    if anchor.clearance is not None and not anchor.clearance(position, .10):
        raise ValueError('Invalid committed pose: array clearance')
    visited = getattr(space, 'visited', np.empty((0, 3)))
    if len(visited) and np.min(np.linalg.norm(visited-position, axis=1)) < space.minimum_separation:
        raise ValueError('Invalid committed pose: repeated receiver position')
    for name, region in [('anchor', anchor), ('current', current)]:
        path = region.path_factory()
        path.requested_start = region.origin
        path.requested_end = floor
        if not region.pathfinder.find_path(path):
            raise ValueError('Invalid committed pose: disconnected '+name+' path')
        distance = float(path.geodesic_distance)
        if not np.isfinite(distance) or not .15 <= distance <= region.budget+1e-6:
            raise ValueError('Invalid committed pose: '+name+' movement budget')
        key = 'anchor_distance_m' if name == 'anchor' and hasattr(space, 'anchor') else 'path_distance_m'
        if not np.isclose(distance, requested[key], atol=1e-4, rtol=0):
            raise ValueError('Invalid committed pose: recorded '+name+' distance mismatch')
    return requested
