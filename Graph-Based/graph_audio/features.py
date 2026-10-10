"""Observation-only graph and acoustic features; no annotation arguments."""
import hashlib
import itertools
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from scipy.spatial import cKDTree

CATEGORIES = ('refrigerator', 'fan', 'tv', 'door', 'toilet', 'washing machine',
              'vacuum cleaner', 'clock', 'microwave', 'dishwasher', 'toaster',
              'coffee machine', 'speaker', 'printer', 'keyboard', 'kettle')
MICROPHONES = np.array([[.04, .04, 0], [.04, -.04, 0],
                        [-.04, .04, 0], [-.04, -.04, 0]], np.float32)
PAIRS = tuple(itertools.combinations(range(4), 2))
FEATURE_VERSION = 'metric-graph-physics-v1'


def waveform(path):
    rate, audio = wavfile.read(path)
    if rate != 44100 or audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError(f'Expected synchronized 44100 Hz four-channel audio: {path}')
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio.astype(np.float32)


def build_graph(xyz, inverse, features, k=12):
    n = len(features)
    counts = np.bincount(inverse, minlength=n)
    centers = np.column_stack([np.bincount(inverse, weights=xyz[:, d], minlength=n)
                               / np.maximum(counts, 1) for d in range(3)]).astype(np.float32)
    distance, neighbors = cKDTree(centers).query(centers, k=min(k + 1, n), workers=1)
    neighbors = neighbors[:, 1:].astype(np.int32)
    distance = distance[:, 1:].astype(np.float32)
    # Nearest-point graph, constrained by a metric distance; no labels involved.
    edge_valid = distance < .35
    radial = np.linalg.norm(centers, axis=1, keepdims=True)
    direction = centers / np.maximum(radial, 1e-5)
    offsets = centers[neighbors] - centers[:, None]
    covariance = np.einsum('nki,nkj->nij', offsets, offsets) / max(1, neighbors.shape[1])
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)
    shape = eigenvalues / np.maximum(eigenvalues.sum(1, keepdims=True), 1e-6)
    geometric = np.concatenate((centers / 10, radial / 10, direction,
                                 centers[:, 2:3] - xyz[:, 2].min(),
                                 np.log1p(counts)[:, None] / 10, shape), 1)
    node_features = np.concatenate((features.astype(np.float32), geometric), 1).astype(np.float32)
    return dict(features=node_features, centers=centers, counts=counts.astype(np.int32),
                neighbors=neighbors, edge_valid=edge_valid.astype(np.uint8))


def acoustic_field(centers, audio, return_tables=False):
    """Eight-band near-field phase evidence and onset evidence at every node.

    Lookup tables evaluate the six spherical microphone delays efficiently.
    The source position is always the observed node center, never a label.
    """
    reduced = resample_poly(audio, 1, 3, axis=0)
    _, _, spectrum = stft(reduced.T, fs=14700, nperseg=512, noverlap=320,
                           boundary='zeros', padded=True, axis=-1)
    frequencies = np.fft.rfftfreq(512, 1 / 14700)
    keep = (frequencies >= 200) & (frequencies <= 3500)
    spectrum = spectrum[:, keep]
    frequencies = frequencies[keep]
    edges = np.geomspace(200, 3501, 9)
    delays = np.linspace(-.00036, .00036, 161)
    steering = np.exp(2j * np.pi * delays[:, None] * frequencies[None])
    distance = np.linalg.norm(centers[:, None] - MICROPHONES[None], axis=-1)
    evidence, lookup = [], []
    for left, right in PAIRS:
        pair_tables = []
        cross = spectrum[left] * spectrum[right].conj()
        unit = cross / np.maximum(np.abs(cross), 1e-8)
        average = cross.mean(-1)
        coherence = np.abs(average) / np.maximum(np.abs(cross).mean(-1), 1e-8)
        observed = average / np.maximum(np.abs(average), 1e-8)
        # Emphasize frames with positive spectral flux to reduce late reflections.
        power = np.abs(spectrum[left]) + np.abs(spectrum[right])
        flux = np.maximum(np.diff(power, axis=-1, prepend=power[:, :1]), 0).sum(0)
        onset = (unit * flux[None]).sum(-1) / max(float(flux.sum()), 1e-8)
        delta = (distance[:, right] - distance[:, left]) / 343
        for low, high in zip(edges[:-1], edges[1:]):
            band = (frequencies >= low) & (frequencies < high)
            weight = coherence[band]
            table = (steering[:, band] * observed[band].conj()[None]).real @ weight
            table /= max(float(weight.sum()), 1e-8)
            pair_tables.append(table)
            evidence.append(np.interp(delta, delays, table))
        onset_table = (steering * onset.conj()[None]).real.mean(-1)
        pair_tables.append(onset_table)
        lookup.append(pair_tables)
        evidence.append(np.interp(delta, delays, onset_table))
    field = np.column_stack(evidence).astype(np.float32)
    level = np.log(np.maximum(np.sqrt((audio ** 2).mean(0)), 1e-8))
    context = np.concatenate((level - level.mean(), [level.mean()])).astype(np.float32)
    if return_tables:
        return field, context, np.asarray(lookup, np.float32)
    return field, context


def cache_geometry(archive, foundation_cache):
    """Validate and align a frozen XYZ-only foundation cache to point order."""
    xyz = archive['xyz'].astype(np.float32)
    ray = archive['ray_id'].astype(np.int64)
    pixel = (ray % 64) * 1800 + ray // 64
    with np.load(foundation_cache, allow_pickle=False) as cached:
        if str(cached['xyz_sha256']) != hashlib.sha256(xyz.tobytes()).hexdigest():
            raise ValueError(f'XYZ hash mismatch: {foundation_cache}')
        mapping = np.full(64 * 1800, -1, dtype=np.int32)
        mapping[cached['pixels']] = cached['inverse']
        inverse = mapping[pixel]
        if (inverse < 0).any():
            raise ValueError('Foundation cache misses an observed point')
        features = cached['features'].astype(np.float32)
    return xyz, inverse, features
