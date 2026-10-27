"""Habitat world Y-up; LiDAR X-forward, Y-left, Z-up; metres and radians."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

# LiDAR basis vectors in a zero-yaw Habitat frame: forward=-Z, left=-X, up=+Y.
LIDAR_TO_HABITAT = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])


def sensor_rotation(yaw: float) -> np.ndarray:
    return Rotation.from_rotvec([0., yaw, 0.]).as_matrix() @ LIDAR_TO_HABITAT


def to_world(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    return np.asarray(points) @ sensor_rotation(yaw).T + position


def to_local(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    return (np.asarray(points) - position) @ sensor_rotation(yaw)


def relative(source: np.ndarray, listener: np.ndarray, yaw: float) -> dict:
    p = to_local(np.asarray(source), listener, yaw)
    return {'position_lidar': p.tolist(), 'distance_m': float(np.linalg.norm(p)),
            'azimuth_deg': float(np.degrees(np.arctan2(p[1], p[0]))),
            'elevation_deg': float(np.degrees(np.arctan2(p[2], np.hypot(p[0], p[1]))))}


def quaternion_xyzw(yaw: float) -> list:
    return Rotation.from_rotvec([0., yaw, 0.]).as_quat().tolist()
