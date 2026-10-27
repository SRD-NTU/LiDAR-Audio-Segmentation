"""Spinning multi-beam rays; first surface hits, never depth-image backprojection."""
from __future__ import annotations
import numpy as np
from audiolidar.geometry.frames import sensor_rotation, to_local


def pattern(cfg: dict) -> tuple:
    elevations = np.radians(np.asarray(cfg['vertical_angles_deg'], dtype=float))
    if len(elevations) != cfg['num_beams'] or not np.isfinite(elevations).all():
        raise ValueError('One finite elevation required per beam')
    az = np.radians(np.arange(0., cfg['horizontal_fov_deg'], cfg['horizontal_resolution_deg']))
    azimuth = np.repeat(az, len(elevations))
    elevation = np.tile(elevations, len(az))
    beam = np.tile(np.arange(len(elevations)), len(az))
    directions = np.column_stack([np.cos(elevation)*np.cos(azimuth),
                                  np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    return directions, beam, azimuth, elevation


def scan(mesh, position: np.ndarray, yaw: float, cfg: dict) -> dict:
    directions, beams, az, el = pattern(cfg)
    world_directions = directions @ sensor_rotation(yaw).T
    origins = np.broadcast_to(position, world_directions.shape).copy()
    xyz, ray_id, tri = mesh.intersector.intersects_location(origins, world_directions, multiple_hits=False)
    order = np.argsort(ray_id)
    xyz, ray_id, tri = xyz[order], ray_id[order], tri[order]
    ranges = np.linalg.norm(xyz-position, axis=1)
    valid = (ranges >= cfg['min_range_m']) & (ranges <= cfg['max_range_m'])
    xyz, ray_id, tri, ranges = xyz[valid], ray_id[valid], tri[valid], ranges[valid]
    instances=mesh.labels_at(xyz,tri) if hasattr(mesh,'labels_at') else mesh.face_instances[tri]
    categories=mesh.categories_at(instances) if hasattr(mesh,'categories_at') else mesh.face_categories[tri]
    return {'xyz': to_local(xyz, position, yaw).astype('float32'),
            'xyz_world': xyz.astype('float32'), 'range': ranges.astype('float32'),
            'beam_id': beams[ray_id].astype('int16'), 'ray_id': ray_id.astype('int32'),
            'azimuth': az[ray_id].astype('float32'), 'elevation': el[ray_id].astype('float32'),
            'triangle_id': tri.astype('int32'),
            'instance_id': instances.astype('int32'),
            'semantic_id': categories.astype('int32')}
