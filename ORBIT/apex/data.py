"""Observation-only feature construction; labels are loaded in a separate function."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('AUDIO_LIDAR_DATA_ROOT', ROOT / 'dataset-sample')).resolve()
GEOMETRY = Path(os.environ.get('ORBIT_GEOMETRY_CACHE', ROOT / 'cache/sonata-xyz-v2')).resolve()
AUDIO_BANK = Path(os.environ.get('ORBIT_AUDIO_BANK_DIR', ROOT / 'ORBIT/cache/audio-bank')).resolve()
CATS = ['refrigerator', 'fan', 'tv', 'door', 'toilet', 'washing machine',
        'vacuum cleaner', 'clock', 'microwave', 'dishwasher', 'toaster',
        'coffee machine', 'speaker', 'printer', 'keyboard', 'kettle']
import gate0_3d as acoustic


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rels(split, dataset=None):
    dataset = DATA if dataset is None else Path(dataset)
    if split not in ('train', 'val', 'test'):
        raise ValueError(split)
    rows = json.loads(Path(dataset, 'metadata', split + '.json').read_text())
    if len(rows) != len(set(rows)) or any(Path(r).parts[0] != split or '..' in Path(r).parts for r in rows):
        raise ValueError('Invalid sample manifest')
    return sorted(rows)


def acoustic_features(xyz, posterior):
    """Keep the full posterior plus four separated modes, not a single DoA.

    Inputs are observation-derived. No source positions or labels are accepted.
    Peak values in gate0 are max-normalized likelihoods, NOT probabilities.
    """
    p = np.asarray(posterior, np.float64)
    p = np.nan_to_num(p, nan=0., posinf=0., neginf=0.).clip(0)
    if p.max() <= 0:
        p = np.ones_like(p)
    p /= p.max()
    r = np.linalg.norm(xyz, axis=1).clip(1e-6)
    u = xyz / r[:, None]
    bins = acoustic.TREE.query(u)[1]
    modes = []
    for k in np.argsort(-p):
        if all(acoustic.DIRS[k] @ acoustic.DIRS[j] < np.cos(np.deg2rad(15)) for j in modes):
            modes.append(int(k))
        if len(modes) == 4:
            break
    cos = np.clip(u @ acoustic.DIRS[modes].T, -1, 1)
    angle = np.arccos(cos)
    spread = np.exp(-.5 * (angle / np.deg2rad(15)) ** 2) * p[modes]
    point_p = p[bins]
    prob = p / p.sum()
    entropy = -(prob * np.log(prob.clip(1e-12))).sum() / np.log(len(p))
    a = np.column_stack([point_p, np.log(point_p.clip(1e-6)) / 14,
                         spread, cos.max(1), np.full(len(xyz), entropy)])
    return a.astype(np.float32)


def geometry_inputs(xyz, features, inverse):
    """Voxel centroids, counts and frozen features. Preserves original point mapping."""
    m = len(features)
    count = np.bincount(inverse, minlength=m).astype(np.float32)
    if len(xyz) != len(inverse) or count.min() <= 0:
        raise ValueError('Invalid geometry cache inverse mapping')
    centre = np.stack([np.bincount(inverse, weights=xyz[:, d], minlength=m) / count
                       for d in range(3)], 1).astype(np.float32)
    return centre, count


def point_pixels(xyz):
    """Derive the cache's range-image pixel for each ORIGINAL input point."""
    r = np.linalg.norm(xyz, axis=1)
    az = np.mod(np.arctan2(xyz[:,1],xyz[:,0]), 2*np.pi)
    el = np.arcsin(np.clip(xyz[:,2]/r.clip(1e-6), -1, 1))
    col = np.rint(az/(2*np.pi)*1800).astype(np.int64) % 1800
    row = np.rint((el+np.pi/8)/(np.pi/4)*63).astype(np.int64)
    if ((row < 0) | (row >= 64)).any():
        raise ValueError('Input outside the calibrated LiDAR field of view')
    return row*1800+col


def align_inverse(xyz, cache_pixels, cache_inverse):
    """Cache inverse is in sorted valid-pixel order, NOT the NPZ point order."""
    pixels = point_pixels(xyz)
    if len(np.unique(pixels)) != len(pixels):
        raise ValueError('Multiple returns per calibrated ray')
    if len(cache_pixels) != len(cache_inverse) or len(np.unique(cache_pixels)) != len(cache_pixels):
        raise ValueError('Invalid cached pixel mapping')
    lookup = np.full(64*1800, -1, np.int64)
    lookup[cache_pixels] = cache_inverse
    inverse = lookup[pixels]
    if (inverse < 0).any():
        raise ValueError('Original point missing from cached range image')
    return inverse


def load_inputs(rel, posterior, embedding, dataset=None):
    dataset = DATA if dataset is None else Path(dataset)
    with np.load(Path(dataset, rel, 'lidar.npz'), allow_pickle=False) as z:
        xyz = z['xyz'].astype(np.float32)
    with np.load(GEOMETRY / (rel + '.npz'), allow_pickle=False) as z:
        if str(z['xyz_sha256']) != hashlib.sha256(xyz.tobytes()).hexdigest():
            raise ValueError(f'XYZ checksum mismatch: {rel}')
        features = z['features'].copy()
        inverse = align_inverse(xyz, z['pixels'].astype(np.int64), z['inverse'].astype(np.int64))
    centre, count = geometry_inputs(xyz, features, inverse)
    return dict(rel=rel, features=features, xyz=centre, count=count,
                acoustic=acoustic_features(centre, posterior),
                audio=np.asarray(embedding, np.float32), inverse=inverse)


def load_targets(rel, inverse, count, dataset=None):
    """Training/evaluator only. Fractions retain mixed-voxel intersections exactly."""
    dataset = DATA if dataset is None else Path(dataset)
    with np.load(Path(dataset, rel, 'lidar.npz'), allow_pickle=False) as z:
        mask = z['sound_mask'].astype(bool)
        sem = z['semantic_id']
        iid = z['instance_id']
    names = json.loads(Path(dataset, 'metadata/categories.json').read_text())
    mapping = {int(v): CATS.index(k) for k, v in names.items() if k in CATS}
    _, first = np.unique(inverse, return_index=True)
    category = np.array([mapping.get(int(s), 16) for s in sem[first]], np.int64)
    positive = np.bincount(inverse, weights=mask, minlength=len(count)).astype(np.float32)
    # No metadata dependency for targets: the sound mask and semantic IDs suffice.
    src_classes, counts = np.unique(sem[mask], return_counts=True)
    source = mapping[int(src_classes[counts.argmax()])]
    return dict(target=positive / count, positive=positive, category=category,
                source=source, n_points=len(mask), n_positive=int(mask.sum()),
                n_source_instances=int(len(np.unique(iid[mask]))))


def audio_bank(split):
    """Reuse tetra-only observations; fixed calibration and sample checksums identify SRP."""
    rs = rels(split)
    portable = AUDIO_BANK / f'{split}.npz'
    if portable.exists():
        with np.load(portable, allow_pickle=False) as z:
            if list(z['rels']) != rs:
                raise ValueError(f'Audio-bank observation order mismatch: {portable}')
            post = z['post'].copy()
            emb = z['emb'].copy()
        if post.shape != (len(rs), acoustic.N_DIRS) or emb.shape != (len(rs), 512):
            raise ValueError(f'Invalid audio-bank dimensions: {portable}')
        return rs, post, emb, dict(audio_bank_sha256=sha(portable),
                                   calibration_sha256=sha(DATA / 'metadata/microphone_array.json'))
    raise FileNotFoundError(f'Build {portable} with ORBIT/scripts/build_audio_bank.py')
