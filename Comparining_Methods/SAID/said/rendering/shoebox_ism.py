"""Shoebox image-source geometry for SAID Online Scene Generation."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 CUHK
# Original NESD implementation: Qiuqiang Kong
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
# Upstream NESD revision:
# https://github.com/qiuqiangkong/nesd/tree/7475da725dcfc8d60761c5891644ca326a7226e7

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RoomSample:
    """Geometry and image paths for one simulated shoebox room."""

    dimensions_m: np.ndarray
    array_center_m: np.ndarray
    array_rotation: np.ndarray
    microphone_positions_m: np.ndarray
    microphone_directions: np.ndarray
    source_centers_m: np.ndarray
    subsource_positions_m: tuple[np.ndarray, ...]
    subsource_gains: tuple[np.ndarray, ...]
    subsource_delays_ms: tuple[np.ndarray, ...]
    source_is_extended: np.ndarray
    image_positions_m: tuple[
        tuple[tuple[np.ndarray, ...], ...], ...
    ]
    image_orders: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    wall_absorption: float
    reflection_order: int


def horizontal_rotation(random: np.random.Generator) -> np.ndarray:
    """Sample the horizontal array rotation used in the paper pipeline."""

    azimuth = float(random.uniform(0.0, 2.0 * np.pi))
    front = np.array(
        [np.cos(azimuth), np.sin(azimuth), 0.0], dtype=np.float64
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    side = np.cross(up, front)
    vertical = np.cross(front, side)
    return np.stack([front, side, vertical], axis=-1)


def _log_uniform(
    random: np.random.Generator, lower: float, upper: float
) -> float:
    return float(10.0 ** random.uniform(np.log10(lower), np.log10(upper)))


def _sample_position(
    random: np.random.Generator,
    dimensions: np.ndarray,
    *,
    margin_m: float = 0.2,
) -> np.ndarray:
    return random.uniform(margin_m, dimensions - margin_m).astype(np.float64)


def _sample_noncolliding_position(
    random: np.random.Generator,
    dimensions: np.ndarray,
    occupied: list[np.ndarray],
    *,
    minimum_distance_m: float = 1.0,
) -> np.ndarray:
    candidate = _sample_position(random, dimensions)
    for attempt in range(2048):
        threshold = minimum_distance_m if attempt < 1024 else 0.5
        candidate = _sample_position(random, dimensions)
        if all(np.linalg.norm(candidate - other) >= threshold for other in occupied):
            return candidate
    raise RuntimeError("could not sample non-colliding room positions")


def _build_image_source_room(
    pyroomacoustics: object,
    *,
    dimensions: np.ndarray,
    microphone_positions: np.ndarray,
    source_positions: list[np.ndarray],
    maximum_order: int,
):
    """Construct the extruded polygon room used by the frozen renderer."""

    corners = np.asarray(
        [
            [0.0, 0.0],
            [0.0, dimensions[1]],
            [dimensions[0], dimensions[1]],
            [dimensions[0], 0.0],
        ],
        dtype=np.float64,
    )
    room = pyroomacoustics.Room.from_corners(
        corners.T, max_order=int(maximum_order)
    )
    room.extrude(height=float(dimensions[2]))
    room.add_microphone_array(microphone_positions.T)
    for position in source_positions:
        room.add_source(position)
    room.image_source_model()
    return room


def _point_subsources(
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        center[None].astype(np.float64),
        np.ones(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )


def _sample_star_boundary(
    random: np.random.Generator,
    *,
    anchors: int = 12,
    radial_jitter_ratio: float = 0.55,
    radial_min_ratio: float = 0.45,
    smooth_iterations: int = 1,
    lobe_boost_ratio: float = 0.45,
    lobe_width: int = 2,
) -> np.ndarray:
    """Sample the star-shaped angular support used by the paper renderer."""

    minimum = min(max(float(radial_min_ratio), 0.05), 1.5)
    jitter = max(0.0, float(radial_jitter_ratio))
    lobe_boost = max(0.0, float(lobe_boost_ratio))
    radii = np.ones(int(anchors), dtype=np.float64)
    for index in range(int(anchors)):
        radii[index] = np.clip(
            1.0 + float(random.uniform(-jitter, jitter)),
            minimum,
            1.0 + lobe_boost,
        )
    if anchors >= 3 and lobe_boost > 0.0:
        maximum_lobes = min(3, anchors // 3 + 1)
        lobe_count = int(random.integers(1, maximum_lobes + 1))
        for _ in range(lobe_count):
            center = int(random.integers(0, anchors))
            boost = float(random.uniform(0.25 * lobe_boost, lobe_boost))
            width = int(random.integers(1, int(lobe_width) + 1))
            for offset in range(-width, width + 1):
                index = (center + offset) % anchors
                weight = max(0.0, 1.0 - abs(offset) / (width + 1))
                radii[index] = np.clip(
                    radii[index] + boost * weight,
                    minimum,
                    1.0 + lobe_boost,
                )
    for _ in range(int(smooth_iterations)):
        radii = (
            np.roll(radii, 1) + 2.0 * radii + np.roll(radii, -1)
        ) / 4.0
        radii = np.clip(radii, minimum, 1.0 + lobe_boost)
    return radii


def _interpolate_star_radius(
    angles: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    anchors = int(boundary.shape[0])
    anchor_angles = np.linspace(0.0, 2.0 * np.pi, anchors, endpoint=False)
    extended_angles = np.concatenate([anchor_angles, [2.0 * np.pi]])
    extended_boundary = np.concatenate([boundary, boundary[:1]])
    return np.interp(
        np.mod(angles, 2.0 * np.pi), extended_angles, extended_boundary
    )


def _center_angles(
    center: np.ndarray, listener: np.ndarray
) -> tuple[float, float, float]:
    vector = center - listener
    radius = float(np.linalg.norm(vector))
    if radius < 1.0e-6:
        return np.pi / 2.0, 0.0, radius
    theta = math.acos(float(np.clip(vector[2] / radius, -1.0, 1.0)))
    phi = math.atan2(float(vector[1]), float(vector[0])) % (2.0 * np.pi)
    return theta, phi, radius


def _sample_extent_metadata(
    random: np.random.Generator,
    *,
    extent_sizes_degrees: tuple[float, float, float],
    extent_probabilities: tuple[float, float, float],
) -> dict[str, object]:
    bucket = float(random.random())
    if bucket < extent_probabilities[0]:
        base_size = extent_sizes_degrees[0]
    elif bucket < extent_probabilities[0] + extent_probabilities[1]:
        base_size = extent_sizes_degrees[1]
    else:
        base_size = extent_sizes_degrees[2]
    azimuth_radius = max(
        1.0e-3, base_size * float(random.uniform(0.82, 1.18))
    )
    elevation_radius = max(
        1.0e-3,
        azimuth_radius * 0.5 * float(random.uniform(0.88, 1.12)),
    )
    return {
        "azimuth_radius_degrees": float(azimuth_radius),
        "elevation_radius_degrees": float(elevation_radius),
        "orientation_degrees": float(random.uniform(-180.0, 180.0)) % 360.0,
        "star_boundary": _sample_star_boundary(random),
    }


def _build_support_mask(
    *,
    center_theta: float,
    center_phi: float,
    metadata: dict[str, object],
) -> np.ndarray:
    elevation = np.linspace(0.0, np.pi, 181, dtype=np.float64)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 361, dtype=np.float64)
    grid_theta, grid_phi = np.meshgrid(elevation, azimuth, indexing="ij")
    azimuth_radius = math.radians(
        max(1.0e-6, float(metadata["azimuth_radius_degrees"]))
    )
    elevation_radius = math.radians(
        max(1.0e-6, float(metadata["elevation_radius_degrees"]))
    )
    rotation = math.radians(float(metadata["orientation_degrees"]))
    delta_azimuth = np.arctan2(
        np.sin(grid_phi - center_phi), np.cos(grid_phi - center_phi)
    )
    delta_elevation = grid_theta - center_theta
    local_azimuth = (
        delta_azimuth * math.cos(rotation)
        + delta_elevation * math.sin(rotation)
    )
    local_elevation = (
        -delta_azimuth * math.sin(rotation)
        + delta_elevation * math.cos(rotation)
    )
    normalized_radius = np.sqrt(
        (local_azimuth / azimuth_radius) ** 2
        + (local_elevation / elevation_radius) ** 2
    )
    polar_angle = np.arctan2(
        local_elevation / elevation_radius,
        local_azimuth / azimuth_radius,
    )
    boundary = _interpolate_star_radius(
        polar_angle,
        np.asarray(metadata["star_boundary"], dtype=np.float64),
    )
    return normalized_radius <= boundary


def _supports_overlap(
    previous: list[np.ndarray],
    candidate: np.ndarray,
    *,
    margin_degrees: float = 1.0,
) -> bool:
    margin = max(0, int(round(margin_degrees)))
    test = candidate.copy()
    if margin:
        dilated = test.copy()
        for step in range(1, margin + 1):
            dilated |= np.roll(test, step, axis=0) | np.roll(test, -step, axis=0)
            dilated |= np.roll(test, step, axis=1) | np.roll(test, -step, axis=1)
        test = dilated
    return any(np.any(item & test) for item in previous)


def _extended_subsources(
    center: np.ndarray,
    listener: np.ndarray,
    dimensions: np.ndarray,
    random: np.random.Generator,
    *,
    metadata: dict[str, object],
    point_spacing_degrees: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribute emitting points over a paper-configured star support."""

    theta, phi, distance = _center_angles(center, listener)
    if distance < 1.0e-6:
        return _point_subsources(center)
    azimuth_radius = math.radians(
        float(metadata["azimuth_radius_degrees"])
    )
    elevation_radius = math.radians(
        float(metadata["elevation_radius_degrees"])
    )
    orientation = math.radians(float(metadata["orientation_degrees"]))
    star_boundary = np.asarray(metadata["star_boundary"], dtype=np.float64)
    spacing = max(1.0e-4, math.radians(point_spacing_degrees))
    minimum_spacing = max(1.0e-4, 0.72 * spacing)
    support_area = (
        np.pi
        * azimuth_radius
        * elevation_radius
        * float(np.mean(star_boundary**2))
    )
    target_points = min(
        100,
        max(1, int(round(support_area / max(spacing**2, 1.0e-8)))),
    )
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    selected: list[tuple[np.ndarray, float, float]] = []

    def inside_support(delta_azimuth: float, delta_elevation: float) -> bool:
        normalized = math.sqrt(
            (delta_azimuth / max(azimuth_radius, 1.0e-8)) ** 2
            + (delta_elevation / max(elevation_radius, 1.0e-8)) ** 2
        )
        polar = math.atan2(
            delta_elevation / max(elevation_radius, 1.0e-8),
            delta_azimuth / max(azimuth_radius, 1.0e-8),
        )
        boundary = float(
            _interpolate_star_radius(np.asarray([polar]), star_boundary)[0]
        )
        return normalized <= boundary

    def to_world(
        delta_azimuth: float, delta_elevation: float
    ) -> np.ndarray | None:
        rotated_azimuth = delta_azimuth * cosine - delta_elevation * sine
        rotated_elevation = delta_azimuth * sine + delta_elevation * cosine
        theta_k = float(
            np.clip(theta + rotated_elevation, 1.0e-4, np.pi - 1.0e-4)
        )
        phi_k = (phi + rotated_azimuth) % (2.0 * np.pi)
        position = listener + distance * np.asarray(
            [
                math.sin(theta_k) * math.cos(phi_k),
                math.sin(theta_k) * math.sin(phi_k),
                math.cos(theta_k),
            ],
            dtype=np.float64,
        )
        if np.all(position > 0.2) and np.all(position < dimensions - 0.2):
            return position
        return None

    def try_add(delta_azimuth: float, delta_elevation: float) -> bool:
        if not inside_support(delta_azimuth, delta_elevation):
            return False
        if any(
            math.hypot(delta_azimuth - old_azimuth, delta_elevation - old_elevation)
            < minimum_spacing
            for _, old_azimuth, old_elevation in selected
        ):
            return False
        position = to_world(delta_azimuth, delta_elevation)
        if position is None:
            return False
        selected.append((position, delta_azimuth, delta_elevation))
        return True

    frontier: list[int] = []
    failures: dict[int, int] = {}
    normalized_azimuth = max(azimuth_radius, spacing)
    normalized_elevation = max(elevation_radius, spacing)

    def normalized_distance(first: int, second: int) -> float:
        _, azimuth_a, elevation_a = selected[first]
        _, azimuth_b, elevation_b = selected[second]
        return math.hypot(
            (azimuth_a - azimuth_b) / normalized_azimuth,
            (elevation_a - elevation_b) / normalized_elevation,
        )

    def preferred_angle(index: int) -> float:
        _, base_azimuth, base_elevation = selected[index]
        x = 0.85 * base_azimuth / normalized_azimuth
        y = 0.85 * base_elevation / normalized_elevation
        for other in range(len(selected)):
            if other == index:
                continue
            _, other_azimuth, other_elevation = selected[other]
            delta_x = (base_azimuth - other_azimuth) / normalized_azimuth
            delta_y = (
                base_elevation - other_elevation
            ) / normalized_elevation
            squared_distance = delta_x**2 + delta_y**2
            if squared_distance < 1.0e-6:
                continue
            weight = 1.0 / (squared_distance + 0.05)
            x += delta_x * weight
            y += delta_y * weight
        if abs(x) < 1.0e-6 and abs(y) < 1.0e-6:
            return float(random.uniform(0.0, 2.0 * np.pi))
        return math.atan2(y, x)

    def frontier_score(index: int) -> float:
        _, base_azimuth, base_elevation = selected[index]
        distances = [
            normalized_distance(index, other)
            for other in range(len(selected))
            if other != index
        ]
        nearest = min(distances, default=10.0)
        local_neighbors = sum(distance < 1.9 for distance in distances)
        radial_bias = math.hypot(
            base_azimuth / normalized_azimuth,
            base_elevation / normalized_elevation,
        )
        return (
            0.65 * nearest
            + 0.45 * radial_bias
            - 0.30 * local_neighbors
            - 0.35 * failures.get(index, 0)
        )

    if try_add(0.0, 0.0):
        frontier.append(0)
        failures[0] = 0
    else:
        for _ in range(48):
            angle = float(random.uniform(0.0, 2.0 * np.pi))
            step = spacing * float(random.uniform(0.15, 0.45))
            if try_add(step * math.cos(angle), step * math.sin(angle)):
                frontier.append(0)
                failures[0] = 0
                break
    if selected and target_points >= 4:
        seed_count = min(3, max(1, target_points // 5))
        seed_base = float(random.uniform(0.0, 2.0 * np.pi))
        for seed_index in range(seed_count):
            base_angle = seed_base + seed_index * 2.0 * np.pi / seed_count
            for scale in (1.05, 0.85, 1.25, 0.65):
                angle = base_angle + float(random.uniform(-0.35, 0.35))
                if try_add(
                    spacing * scale * math.cos(angle),
                    spacing * scale * math.sin(angle),
                ):
                    new_index = len(selected) - 1
                    frontier.append(new_index)
                    failures[new_index] = 0
                    break
    while frontier and len(selected) < target_points:
        frontier_index = max(
            frontier,
            key=lambda index: frontier_score(index)
            + float(random.uniform(-0.15, 0.15)),
        )
        _, base_azimuth, base_elevation = selected[frontier_index]
        preferred = preferred_angle(frontier_index)
        accepted = False
        for attempt in range(16):
            spread = 0.45 if attempt < 10 else 1.15
            angle = preferred + float(random.uniform(-spread, spread))
            step = spacing * float(random.uniform(0.90, 1.20))
            if try_add(
                base_azimuth + step * math.cos(angle),
                base_elevation + step * math.sin(angle),
            ):
                new_index = len(selected) - 1
                frontier.append(new_index)
                failures[new_index] = 0
                failures[frontier_index] = 0
                accepted = True
                break
        if accepted:
            continue
        failures[frontier_index] = failures.get(frontier_index, 0) + 1
        if failures[frontier_index] >= 5:
            frontier.remove(frontier_index)
    for _ in range(max(24, 5 * target_points)):
        if len(selected) >= target_points:
            break
        angle = float(random.uniform(0.0, 2.0 * np.pi))
        boundary = float(
            _interpolate_star_radius(np.asarray([angle]), star_boundary)[0]
        )
        radius = boundary * math.sqrt(float(random.random()))
        try_add(
            azimuth_radius * radius * math.cos(angle),
            elevation_radius * radius * math.sin(angle),
        )
    if not selected:
        selected.append((center.astype(np.float64), 0.0, 0.0))

    positions = np.stack([item[0] for item in selected[:100]])
    gains = []
    delays_ms = []
    for _, delta_azimuth, delta_elevation in selected[:100]:
        normalized = math.hypot(
            delta_azimuth / max(azimuth_radius, 1.0e-8),
            delta_elevation / max(elevation_radius, 1.0e-8),
        )
        gain = math.exp(-0.5 * normalized**2)
        gain *= 10.0 ** (float(random.uniform(-2.0, 2.0)) / 20.0)
        gains.append(gain)
        delays_ms.append(float(random.uniform(-0.2, 0.2)))
    gain_array = np.asarray(gains, dtype=np.float64)
    gain_array /= gain_array.sum()
    return positions, gain_array, np.asarray(delays_ms, dtype=np.float64)


class ShoeboxImageSourceModel:
    """Sample paper-configured rooms and expose their image-source paths."""

    def __init__(
        self,
        microphone_local_positions_m: np.ndarray,
        *,
        room_width_m: tuple[float, float] = (2.0, 10.0),
        room_length_m: tuple[float, float] = (2.0, 10.0),
        room_height_m: tuple[float, float] = (2.0, 4.0),
        wall_absorption: tuple[float, float] = (0.0, 0.5),
        maximum_reflection_order: int = 5,
        sample_rate: int = 48_000,
        point_spacing_degrees: float = 4.0,
        extent_sizes_degrees: tuple[float, float, float] = (3.0, 8.0, 13.0),
        extent_probabilities: tuple[float, float, float] = (0.30, 0.45, 0.25),
    ) -> None:
        positions = np.asarray(microphone_local_positions_m, dtype=np.float64)
        if positions.shape != (4, 3):
            raise ValueError("the paper renderer requires four microphone positions")
        self.microphone_local_positions_m = positions
        self.microphone_local_directions = positions / np.linalg.norm(
            positions, axis=-1, keepdims=True
        )
        self.room_width_m = tuple(float(value) for value in room_width_m)
        self.room_length_m = tuple(float(value) for value in room_length_m)
        self.room_height_m = tuple(float(value) for value in room_height_m)
        self.wall_absorption = tuple(float(value) for value in wall_absorption)
        self.maximum_reflection_order = int(maximum_reflection_order)
        self.sample_rate = int(sample_rate)
        self.point_spacing_degrees = float(point_spacing_degrees)
        self.extent_sizes_degrees = tuple(
            float(value) for value in extent_sizes_degrees
        )
        probabilities = np.asarray(extent_probabilities, dtype=np.float64)
        if (
            len(self.extent_sizes_degrees) != 3
            or probabilities.shape != (3,)
            or np.any(probabilities < 0.0)
            or float(probabilities.sum()) <= 0.0
        ):
            raise ValueError("source extent sizes and probabilities are invalid")
        self.extent_probabilities = tuple(
            float(value) for value in probabilities / probabilities.sum()
        )

    def sample(
        self,
        random: np.random.Generator,
        *,
        source_count: int,
        extended_sources: bool,
    ) -> RoomSample:
        try:
            import pyroomacoustics as pra
        except ImportError as error:
            raise RuntimeError(
                "Online Scene Generation requires the 'render' extra: "
                "pip install -e '.[render]'"
            ) from error
        length = _log_uniform(random, *self.room_length_m)
        width = _log_uniform(random, *self.room_width_m)
        height = _log_uniform(
            random,
            self.room_height_m[0],
            min(length, width, self.room_height_m[1]),
        )
        dimensions = np.array([length, width, height], dtype=np.float64)
        center = _sample_position(random, dimensions)
        rotation = horizontal_rotation(random)
        microphone_positions = (
            self.microphone_local_positions_m @ rotation.T + center
        )
        microphone_directions = (
            self.microphone_local_directions @ rotation.T
        )
        occupied = [position for position in microphone_positions]
        source_centers: list[np.ndarray] = []
        subsources: list[np.ndarray] = []
        subsource_gains: list[np.ndarray] = []
        subsource_delays_ms: list[np.ndarray] = []
        source_is_extended: list[bool] = []
        support_masks: list[np.ndarray] = []
        for _ in range(int(source_count)):
            if extended_sources:
                accepted = False
                for _attempt in range(128):
                    source = _sample_noncolliding_position(
                        random, dimensions, occupied + source_centers
                    )
                    metadata = _sample_extent_metadata(
                        random,
                        extent_sizes_degrees=self.extent_sizes_degrees,
                        extent_probabilities=self.extent_probabilities,
                    )
                    theta, phi, _radius = _center_angles(source, center)
                    support = _build_support_mask(
                        center_theta=theta,
                        center_phi=phi,
                        metadata=metadata,
                    )
                    if _supports_overlap(support_masks, support):
                        continue
                    points, gains, delays_ms = _extended_subsources(
                        source,
                        center,
                        dimensions,
                        random,
                        metadata=metadata,
                        point_spacing_degrees=self.point_spacing_degrees,
                    )
                    support_masks.append(support)
                    accepted = True
                    break
                if not accepted:
                    source = _sample_noncolliding_position(
                        random, dimensions, occupied + source_centers
                    )
                    points, gains, delays_ms = _point_subsources(source)
            else:
                source = _sample_noncolliding_position(
                    random, dimensions, occupied + source_centers
                )
                points, gains, delays_ms = _point_subsources(source)
            source_centers.append(source)
            subsources.append(points)
            subsource_gains.append(gains)
            subsource_delays_ms.append(delays_ms)
            source_is_extended.append(bool(extended_sources and accepted))
        # A Pyroomacoustics maximum order includes every image-source order
        # from direct sound through this value. The paper configuration uses
        # maximum order five for every scene.
        order = self.maximum_reflection_order
        absorption = float(random.uniform(*self.wall_absorption))
        flat_points = [point for group in subsources for point in group]
        room = _build_image_source_room(
            pra,
            dimensions=dimensions,
            microphone_positions=microphone_positions,
            source_positions=flat_points,
            maximum_order=order,
        )
        image_positions: list[tuple[tuple[np.ndarray, ...], ...]] = []
        image_orders: list[tuple[tuple[np.ndarray, ...], ...]] = []
        flat_index = 0
        for group in subsources:
            group_positions: list[tuple[np.ndarray, ...]] = []
            group_orders: list[tuple[np.ndarray, ...]] = []
            for _ in group:
                source = room.sources[flat_index]
                visible_positions: list[np.ndarray] = []
                visible_orders: list[np.ndarray] = []
                for microphone_index in range(
                    self.microphone_local_positions_m.shape[0]
                ):
                    visible = np.asarray(
                        room.visibility[flat_index][microphone_index],
                        dtype=np.bool_,
                    )
                    visible_positions.append(
                        source.images[:, visible].T.astype(np.float64)
                    )
                    visible_orders.append(
                        source.orders[visible].astype(np.int32)
                    )
                group_positions.append(tuple(visible_positions))
                group_orders.append(tuple(visible_orders))
                flat_index += 1
            image_positions.append(tuple(group_positions))
            image_orders.append(tuple(group_orders))
        return RoomSample(
            dimensions_m=dimensions,
            array_center_m=center,
            array_rotation=rotation,
            microphone_positions_m=microphone_positions,
            microphone_directions=microphone_directions,
            source_centers_m=np.asarray(source_centers, dtype=np.float64).reshape(-1, 3),
            subsource_positions_m=tuple(subsources),
            subsource_gains=tuple(subsource_gains),
            subsource_delays_ms=tuple(subsource_delays_ms),
            source_is_extended=np.asarray(source_is_extended, dtype=np.bool_),
            image_positions_m=tuple(image_positions),
            image_orders=tuple(image_orders),
            wall_absorption=absorption,
            reflection_order=order,
        )
