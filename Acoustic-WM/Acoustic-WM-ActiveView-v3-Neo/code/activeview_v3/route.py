"""Source-independent feasible paths for sequential acquisition."""
import numpy as np
from .navigation import ReachableSpace


class RouteSpace(ReachableSpace):
    def __init__(self, anchor, current_floor, visited_positions, minimum_separation=.15):
        super().__init__(anchor.pathfinder, anchor.path_factory, anchor.origin,
                         anchor.budget, anchor.height, anchor.clearance)
        self.anchor = anchor
        self.current = ReachableSpace(anchor.pathfinder, anchor.path_factory, current_floor,
                                      anchor.budget, anchor.height, anchor.clearance)
        self.visited = np.asarray(visited_positions, dtype=float).reshape(-1, 3)
        self.minimum_separation = minimum_separation

    def project(self, xz, yaw):
        anchor_pose = self.anchor.project(xz, yaw)
        if anchor_pose is None:
            return None
        pose = self.current.project(np.asarray(anchor_pose['floor'])[[0, 2]], yaw)
        if pose is None or not np.allclose(pose['position'], anchor_pose['position'], atol=1e-4):
            return None
        if len(self.visited) and np.min(np.linalg.norm(self.visited - pose['position'], axis=1)) < self.minimum_separation:
            return None
        pose['anchor_distance_m'] = anchor_pose['path_distance_m']
        return pose
