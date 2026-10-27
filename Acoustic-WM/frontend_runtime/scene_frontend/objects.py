"""Mask pooling uses observations; annotation access is confined to supervision()."""
import json
import numpy as np
from scipy.spatial import cKDTree
from apex import data as D

GEOM_DIM = 524  # mean/max frozen features, centre, extent, std, range/count/volume
ACOUSTIC_DIM = 20


def pool_masks(x, memberships):
    """Pool nonnegative voxel memberships. No semantic/instance labels are read.

    Each membership is (voxel indices, ORIGINAL POINT counts in these voxels).
    The same routine accepts predicted masks and evaluator-only perfect masks.
    """
    g, a = [], []
    for idx, weights in memberships:
        idx = np.asarray(idx, np.int64)
        w = np.asarray(weights, np.float32)
        if len(idx) == 0 or w.sum() <= 0:
            raise ValueError('Empty candidate')
        w = w / w.sum()
        f = x['features'][idx].astype(np.float32)
        xyz = x['xyz'][idx]
        ac = x['acoustic'][idx]
        centre = w @ xyz
        extent = xyz.max(0) - xyz.min(0)
        std = np.sqrt(w @ ((xyz - centre) ** 2))
        geo = np.r_[w @ f, f.max(0), centre / 10, extent / 3, std / 3,
                    np.linalg.norm(centre) / 10, np.log1p(np.sum(weights)) / 10,
                    np.log1p(np.prod(extent))]
        mean = w @ ac
        maximum = ac.max(0)
        p = ac[:, 0]
        audio = np.r_[mean, maximum, np.log(max(float(w @ p), 1e-6)) / 14,
                      np.log(max(float(p.max()), 1e-6)) / 14,
                      np.sum(w[p > .5]), np.sum(w[p > .1])]
        g.append(geo); a.append(audio)
    return np.asarray(g, np.float32).reshape(-1, GEOM_DIM), np.asarray(a, np.float32).reshape(-1, ACOUSTIC_DIM)


def supervision(rel, x):
    """TRAIN targets / privileged evaluator only. Never call during inference."""
    with np.load(D.DATA / rel / 'lidar.npz', allow_pickle=False) as z:
        iid = z['instance_id'].astype(np.int64)
        sem = z['semantic_id']
        mask = z['sound_mask'].astype(bool)
    names = {int(v): k for k, v in json.loads((D.DATA / 'metadata/categories.json').read_text()).items()}
    ids, first, inv, counts = np.unique(iid, return_index=True, return_inverse=True, return_counts=True)
    valid = (ids > 0) & (counts >= 30) & np.array([names.get(int(s), '') not in D.acoustic.STRUCTURE for s in sem[first]])
    selected = np.flatnonzero(valid)
    mapping = {int(i): j for j, i in enumerate(selected)}
    point_object = np.array([mapping.get(int(i), -1) for i in inv], np.int32)
    nvox = len(x['count'])
    # True majority per voxel, never assume input scan order is sorted pixel order.
    keys = x['inverse'] * (len(ids) + 1) + inv
    u, ct = np.unique(keys, return_counts=True)
    vox, ii = u // (len(ids) + 1), u % (len(ids) + 1)
    rank = np.lexsort((-ct, vox))
    vv, first_vox = np.unique(vox[rank], return_index=True)
    dominant = ii[rank[first_vox]]
    node_object = np.full(nvox, -1, np.int32)
    node_object[vv] = np.array([mapping.get(int(i), -1) for i in dominant])
    node_category = np.full(nvox, 16, np.int64)
    labels = np.array([D.CATS.index(names[int(sem[first[i]])]) if names.get(int(sem[first[i]])) in D.CATS else 16 for i in selected], np.int64)
    ok = node_object >= 0
    node_category[ok] = labels[node_object[ok]]
    source_ids, source_counts = np.unique(iid[mask], return_counts=True)
    if not len(source_ids):
        raise ValueError(f'No sounding instance for supervised training: {rel}')
    source_id = source_ids[source_counts.argmax()]
    source_index = np.flatnonzero(ids[selected] == source_id)
    if len(source_index) != 1:
        raise ValueError(f'Source is missing from valid training/evaluation objects: {rel}')
    target = int(source_index[0])
    memberships = []
    for j in range(len(selected)):
        idx, cnt = np.unique(x['inverse'][point_object == j], return_counts=True)
        memberships.append((idx, cnt.astype(np.float32)))
    positives = np.bincount(x['inverse'], weights=mask, minlength=nvox).astype(np.float32)
    return dict(memberships=memberships, node_object=node_object, node_category=node_category,
                categories=labels, source_object=target, source_class=int(labels[target]),
                positive=positives, instance_ids=ids[selected], source_instance_id=int(source_id))


def neighbors(xyz, k=12):
    _, indices = cKDTree(xyz).query(xyz, k=min(len(xyz), k), workers=1)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices.astype(np.int32)
