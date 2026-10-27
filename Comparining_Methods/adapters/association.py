"""Shared, source-blind association of acoustic localization with object masks."""
from __future__ import annotations

import numpy as np


def _validate(xyz: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if candidates.ndim != 2 or candidates.shape[1] != len(xyz) or len(candidates) == 0:
        raise ValueError("candidate masks must have shape [K,N]")
    if not np.all(candidates.any(axis=1)):
        raise ValueError("Empty candidate mask")
    return xyz, candidates


def direction_scores(xyz: np.ndarray, candidates: np.ndarray, direction_lidar: np.ndarray,
                     angular_sigma_deg: float, distance_m: float | None = None,
                     distance_sigma_m: float | None = None) -> np.ndarray:
    """Mean spatial likelihood over measured points of each candidate.

    A Gaussian angular likelihood is integrated over each candidate's measured
    surface points. An optional radial Gaussian is multiplied point-wise. Mean
    aggregation avoids an automatic preference for large masks; it remains a
    view-dependent approximation to object extent, not a source centroid rule.
    """
    xyz, candidates = _validate(xyz, candidates)
    direction = np.asarray(direction_lidar, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("direction_lidar must be a finite nonzero vector")
    if angular_sigma_deg <= 0:
        raise ValueError("angular_sigma_deg must be positive")
    radii = np.linalg.norm(xyz, axis=1)
    unit = xyz / np.maximum(radii[:, None], 1e-12)
    cosine = np.clip(unit @ (direction / np.linalg.norm(direction)), -1, 1)
    angle = np.arccos(cosine)
    likelihood = np.exp(-0.5 * (angle / np.deg2rad(angular_sigma_deg)) ** 2)
    if distance_m is not None:
        if distance_sigma_m is None or distance_sigma_m <= 0:
            raise ValueError("distance_sigma_m must be positive when distance_m is supplied")
        likelihood *= np.exp(-0.5 * ((radii - distance_m) / distance_sigma_m) ** 2)
    return (candidates @ likelihood) / candidates.sum(axis=1)


def panoramic_scores(xyz: np.ndarray, candidates: np.ndarray, likelihood_map: np.ndarray) -> np.ndarray:
    """Bilinearly integrate an equirectangular [elevation, azimuth] map.

    Columns cover azimuth [-180,180) and wrap; rows cover elevation [+90,-90].
    The map orientation must first be verified against the emitting model.
    """
    xyz, candidates = _validate(xyz, candidates)
    image = np.asarray(likelihood_map, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("likelihood_map must be a finite 2D map")
    h, w = image.shape
    radius = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arcsin(np.clip(xyz[:, 2] / np.maximum(radius, 1e-12), -1, 1))
    col = (azimuth / (2 * np.pi) + 0.5) * w
    row = (0.5 - elevation / np.pi) * (h - 1)
    c0 = np.floor(col).astype(int) % w
    c1 = (c0 + 1) % w
    r0 = np.floor(row).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    dc, dr = col - np.floor(col), row - r0
    values = ((1 - dr) * ((1 - dc) * image[r0, c0] + dc * image[r0, c1])
              + dr * ((1 - dc) * image[r1, c0] + dc * image[r1, c1]))
    return (candidates @ values) / candidates.sum(axis=1)
