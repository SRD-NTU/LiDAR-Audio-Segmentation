"""Observation-only feature construction; labels are loaded in a separate function."""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get('AUDIO_LIDAR_DATA_ROOT', ROOT / 'dataset/v2-tetra4-4channel-full')).resolve()
GEOMETRY = Path(os.environ.get('ORBIT_GEOMETRY_CACHE', ROOT / 'ORBIT/cache/sonata-xyz-v2')).resolve()
CATS = ['refrigerator', 'fan', 'tv', 'door', 'toilet', 'washing machine',
        'vacuum cleaner', 'clock', 'microwave', 'dishwasher', 'toaster',
        'coffee machine', 'speaker', 'printer', 'keyboard', 'kettle']
sys.path.insert(0, str(ROOT / 'ORBIT/runtime'))
import gate0_3d as acoustic

_HASH_CACHE = {}


def current_sha(path):
    """Hash actual bytes, reusing a digest only while file identity is unchanged."""
    path = Path(path).resolve()
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    if signature not in _HASH_CACHE:
        _HASH_CACHE[signature] = sha(path)
    return _HASH_CACHE[signature]


def waveform_hashes(rows, dataset=None):
    dataset = DATA if dataset is None else Path(dataset)
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda rel: current_sha(dataset / rel / 'audio.wav'), rows))


def write_posterior_identity(path, rows, dependencies=None, dataset=None, provenance='generated-v2', replace=False):
    dataset = DATA if dataset is None else Path(dataset)
    path = Path(path)
    manifest = dict(schema=2, rels=list(rows), waveform_sha256=waveform_hashes(rows, dataset),
                    calibration_sha256=sha(dataset / 'metadata/microphone_array.json'),
                    posterior_sha256=sha(path), dependencies={str(Path(k).resolve()):v for k,v in (dependencies or {}).items()},
                    provenance=provenance)
    target = path.with_suffix(path.suffix + '.identity.json')
    if not replace and target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError(f'Refusing to overwrite a different posterior identity: {target}')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_dependencies(dependencies):
    for path, expected in dependencies.items():
        candidate = Path(path)
        if not candidate.is_absolute():candidate = ROOT / candidate
        if current_sha(candidate) != expected:
            raise ValueError(f'Cache dependency changed: {candidate}')


def load_posterior(path, rows, dataset=None):
    dataset = DATA if dataset is None else Path(dataset)
    path = Path(path)
    identity = path.with_suffix(path.suffix + '.identity.json')
    if not identity.is_file():
        raise ValueError(f'Missing posterior identity manifest: {identity}; regenerate or explicitly attest legacy data')
    manifest = json.loads(identity.read_text())
    if manifest.get('schema') != 2 or manifest.get('rels') != list(rows):
        raise ValueError('Posterior row identity/order mismatch')
    if manifest.get('calibration_sha256') != sha(dataset / 'metadata/microphone_array.json'):
        raise ValueError('Posterior calibration mismatch')
    if manifest.get('waveform_sha256') != waveform_hashes(rows, dataset):
        raise ValueError('Posterior waveform identity mismatch')
    if manifest.get('posterior_sha256') != current_sha(path):
        raise ValueError('Posterior array differs from its identity manifest')
    verify_dependencies(manifest.get('dependencies', {}))
    posterior = np.load(path, allow_pickle=False)
    if posterior.shape != (len(rows), acoustic.N_DIRS) or not np.isfinite(posterior).all():
        raise ValueError('Invalid posterior array')
    return posterior


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
    if p.shape != (acoustic.N_DIRS,):
        raise ValueError('Wrong directional posterior dimensions')
    if p.max() <= 0 or np.ptp(p) <= max(p.max(), 1.) * 1e-12:
        return neutral_acoustic_features(len(xyz))
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


def neutral_acoustic_features(n):
    """Direction-independent constants for all eight acoustic feature channels."""
    return np.broadcast_to(np.array([1., 0., 0., 0., 0., 0., 0., 1.], np.float32), (n, 8)).copy()


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
    geometry_contract()
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
                audio=np.asarray(embedding, np.float32), inverse=inverse,
                raw_xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
                silence=bool(np.max(np.nan_to_num(posterior), initial=0) <= 0))


def geometry_contract():
    """Verify weights/projection binding recorded by the geometry generator."""
    root=ROOT/'ORBIT'
    manifest=GEOMETRY/'COMPLETE.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        expected=record.get('checkpoint_sha256',record.get('utonia_checkpoint_sha256'))
        if expected is None:raise ValueError('Geometry manifest lacks encoder weight identity')
        if current_sha(root/'pretrained/utonia/utonia.pth')!=expected:
            raise ValueError('Geometry encoder weights differ from cache generation')
        projection_hash=record.get('projection_sha256',record.get('geometry_projection_sha256'))
        projection=Path(record.get('projection_path',GEOMETRY/'projection.pt'))
        if not projection.exists():projection=root/'cache/utonia-xyz-v1/projection.pt'
        if projection_hash is None or current_sha(projection)!=projection_hash:
            raise ValueError('Geometry projection differs from cache generation')
    else:
        manifest=GEOMETRY/'encoder.json'
        if not manifest.exists():raise ValueError('Missing geometry encoder provenance; regenerate the geometry cache')
        record=json.loads(manifest.read_text())
        if current_sha(root/'pretrained/sonata/sonata.pth')!=record.get('encoder_sha256'):
            raise ValueError('Sonata encoder weights differ from cache generation')
    return dict(manifest_sha256=current_sha(manifest),provenance=record)


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
    source = mapping[int(src_classes[counts.argmax()])] if len(src_classes) else -1
    return dict(target=positive / count, positive=positive, category=category,
                source=source, has_target=bool(mask.any()), n_points=len(mask), n_positive=int(mask.sum()),
                n_source_instances=int(len(np.unique(iid[mask]))))


def audio_bank(split):
    """Reuse tetra-only observations; fixed calibration and sample checksums identify SRP."""
    rs = rels(split)
    # Tier observations retain their original identities. Reuse the full bank
    # only after calibration and explicit row identities have been checked.
    full = ROOT / 'dataset/v2-tetra4-4channel-full'
    full_rs = rels(split, full)
    calibration = full / 'metadata/microphone_array.json'
    if sha(DATA / 'metadata/microphone_array.json') != sha(calibration):
        raise ValueError('Selected dataset calibration differs from the full audio bank')
    h = hashlib.sha256(calibration.read_bytes())
    h.update(json.dumps({'n_dirs': acoustic.N_DIRS, 'band': list(acoustic.BAND)}).encode())
    with ThreadPoolExecutor(8) as pool:
        for content in pool.map(lambda r: Path(full, r, 'COMPLETE.json').read_bytes(), full_rs):
            h.update(content)
    fingerprint = h.hexdigest()[:16]
    srp = ROOT / f'ORBIT/cache/audio-legacy/posteriors_{split}_{len(full_rs)}_{fingerprint}.npy'
    if not srp.exists():
        raise FileNotFoundError(f'Run preparation to compute tetra SRP: {srp}')
    post = np.load(srp)
    path = ROOT / f'ORBIT/cache/audio-legacy/clap_tetra_{split}.npz'
    with np.load(path, allow_pickle=False) as z:
        if list(z['rels']) != full_rs:
            raise ValueError('CLAP sample ordering mismatch')
        emb = z['emb'].copy()
    identity_path = path.with_suffix('.npz.identity.json')
    if not identity_path.exists():
        raise ValueError(f'Missing audio-bank identity: {identity_path}; run scripts/attest_legacy_caches.py or regenerate')
    identity = json.loads(identity_path.read_text())
    if identity.get('schema') != 2 or identity.get('rels') != full_rs:
        raise ValueError('Audio-bank identity/order mismatch')
    if identity.get('calibration_sha256') != sha(calibration):
        raise ValueError('Audio-bank calibration changed')
    if identity.get('waveform_sha256') != waveform_hashes(full_rs, full):
        raise ValueError('Audio-bank waveform changed')
    if identity.get('clap_sha256') != current_sha(path) or identity.get('srp_sha256') != current_sha(srp):
        raise ValueError('Audio-bank arrays changed')
    verify_dependencies(identity.get('dependencies', {}))
    if post.shape != (len(full_rs), acoustic.N_DIRS):
        raise ValueError('SRP shape mismatch')
    if emb.shape!=(len(full_rs),512) or not np.isfinite(emb).all() or not np.isfinite(post).all():
        raise ValueError('Invalid audio embedding or posterior bank')
    lookup = {rel: i for i, rel in enumerate(full_rs)}
    try:
        indices = np.asarray([lookup[rel] for rel in rs], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'Observation absent from full audio bank: {error}') from error
    if DATA != full.resolve():
        for rel in rs:
            if (DATA / rel / 'COMPLETE.json').read_bytes() != (full / rel / 'COMPLETE.json').read_bytes():
                raise ValueError(f'Tier observation identity differs from full dataset: {rel}')
        if waveform_hashes(rs) != [identity['waveform_sha256'][lookup[rel]] for rel in rs]:
            raise ValueError('Tier waveform differs from its source audio bank')
    post, emb = post[indices], emb[indices]
    return rs, post, emb, dict(srp_sha256=current_sha(srp), clap_sha256=current_sha(path),
                              identity_provenance=identity.get('provenance'), identity_sha256=sha(identity_path),
                              calibration_sha256=sha(calibration))
