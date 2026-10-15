"""Versioned geometry identity and sparse instance supervision contracts."""
import hashlib
import json
import numpy as np

CACHE_SCHEMA = 2
METHOD_VERSION = 'orbit-consistent-v2'


def geometry_uid(xyz_sha256, scene, listener, lidar):
    # Counts of source points are annotations, not sensor configuration.
    sensor = {k: v for k, v in lidar.items()
              if k not in ('num_source_points', 'source_point_fraction', 'num_points')}
    identity = dict(xyz_sha256=xyz_sha256, scene=scene,
                    position=listener['position_world'], rotation=listener['rotation_world'], lidar=sensor)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def shared_indices(uid, n, budget):
    if budget <= 0:
        raise ValueError('Training node budget must be positive')
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    seed = int(uid[:16], 16)
    return np.sort(np.random.default_rng(seed).choice(n, budget, replace=False))


def sparse_memberships(memberships, selected):
    lookup = np.full(int(selected.max()) + 1 if len(selected) else 0, -1, np.int64)
    lookup[selected] = np.arange(len(selected))
    objects, nodes, counts = [], [], []
    for object_id, (indices, mass) in enumerate(memberships):
        indices = np.asarray(indices, np.int64)
        inside = indices < len(lookup)
        indices, mass = indices[inside], np.asarray(mass)[inside]
        local = lookup[indices]
        keep = local >= 0
        objects.extend([object_id] * int(keep.sum()))
        nodes.extend(local[keep])
        counts.extend(mass[keep])
    return dict(instance_object=np.asarray(objects, np.int32),
                instance_voxel=np.asarray(nodes, np.int64),
                instance_count=np.asarray(counts, np.float32))


def instance_target(row, object_id):
    for key in ('instance_object', 'instance_voxel', 'instance_count'):
        if key not in row:
            raise ValueError('Legacy supervision cache: rebuild with orbit.prepare (schema 2)')
    keep = row['instance_object'] == object_id
    target = np.zeros(len(row['count']), np.float32)
    target[row['instance_voxel'][keep]] = row['instance_count'][keep] / row['count'][row['instance_voxel'][keep]]
    return target


def require_consistent_cache(row):
    if int(row.get('cache_schema', 0)) != CACHE_SCHEMA:
        raise ValueError('Legacy training cache is incompatible with corrected supervision; rebuild with orbit.prepare')
    required = ('geometry_uid', 'raw_xyz_sha256', 'selected_voxel_ids', 'full_voxel_count',
                'instance_ids', 'source_instance_id', 'instance_object', 'instance_voxel', 'instance_count')
    if any(key not in row for key in required):
        raise ValueError('Incomplete schema-2 supervision identity')
    n=len(row['count']);selected=np.asarray(row['selected_voxel_ids'])
    if len(selected)!=n or (selected<0).any() or (selected>=int(row['full_voxel_count'])).any() or (np.diff(selected)<=0).any():
        raise ValueError('inconsistent selected voxel identities')
    if any(len(row[key])!=n for key in ('xyz','node_object','positive')):
        raise ValueError('Inconsistent supervision array lengths')
    if not np.isfinite(row['positive']).all() or (row['positive']<0).any() or (row['positive']>row['count']).any():
        raise ValueError('Invalid sounding-point counts')
    if not np.isfinite(row['count']).all() or (row['count']<=0).any():
        raise ValueError('Invalid voxel point counts')
    objects=np.asarray(row['instance_object']);voxels=np.asarray(row['instance_voxel']);mass=np.asarray(row['instance_count'])
    if not (len(objects)==len(voxels)==len(mass)) or (objects<0).any() or (objects>=len(row['instance_ids'])).any() or (voxels<0).any() or (voxels>=n).any():
        raise ValueError('Invalid sparse instance identities')
    if not np.isfinite(mass).all() or (mass<=0).any() or (mass>row['count'][voxels]).any():
        raise ValueError('Invalid sparse instance point counts')
    if len(np.unique(objects*n+voxels))!=len(objects):raise ValueError('Duplicate sparse instance/voxel entries')
    if (np.bincount(voxels,weights=mass,minlength=n)>row['count']).any():raise ValueError('Instance mass exceeds voxel count')
    source=int(row['source_object'])
    if source<0 or source>=len(row['instance_ids']) or int(row['instance_ids'][source])!=int(row['source_instance_id']):
        raise ValueError('Source physical instance identity mismatch')


def sampling_statistics(rows):
    positive = sum(int((r['positive'] > 0).sum()) for r in rows)
    overlap = sum(int(((r['positive'] > 0) & (r['node_object'] != int(r['source_object']))).sum()) for r in rows)
    from .audio_gate import sample_seeds
    rng=np.random.default_rng(1337)
    sampled=[sample_seeds(r,rng) for r in rows if (r['positive']>0).any()]
    duplicate=sum(len(s)!=len(set(s)) for s in sampled)
    purity=[r['positive'][r['positive']>0]/r['count'][r['positive']>0] for r in rows]
    values=np.concatenate(purity) if positive else np.array([])
    return dict(rows=len(rows), positive_voxels=positive, sampled_rows=len(sampled), duplicate_seed_rows=duplicate,
                duplicate_seed_row_fraction=duplicate/max(len(sampled),1),
                positive_voxel_purity_mean=float(values.mean()) if len(values) else None,
                positive_negative_support_overlap_voxels=overlap,
                positive_negative_support_overlap_fraction=overlap / max(positive, 1),
                rows_with_support_overlap=sum(bool(((r['positive'] > 0) &
                    (r['node_object'] != int(r['source_object']))).any()) for r in rows))
