"""Build training/validation observation graphs, with targets stored separately."""
import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from graph_audio.features import (CATEGORIES, FEATURE_VERSION, acoustic_field,
                                  build_graph, cache_geometry, waveform)

ROOT = Path(__file__).resolve().parent
WM = Path(os.environ.get('LIDAR_WM_ROOT', ROOT.parent / 'LiDAR-WM')).expanduser().resolve()
DATASET = Path(os.environ.get('AUDIO_LIDAR_DATA_ROOT', ROOT.parent / 'data')).expanduser().resolve()
FOUNDATION = Path(os.environ.get('LIDAR_WM_GEOMETRY_CACHE', WM / 'cache/sonata-xyz-v2')).expanduser().resolve()
CLAP = Path(os.environ.get('LIDAR_WM_CLAP_CACHE', WM / 'cache/clap')).expanduser().resolve()


def one(task):
    name, dataset, foundation, clap, output = task
    dataset, foundation, clap, output = map(Path, (dataset, foundation, clap, output))
    destination = output / (name + '.npz')
    if destination.exists():
        with np.load(destination) as a:
            if str(a['version']) != FEATURE_VERSION:
                raise ValueError('Incompatible existing feature cache')
        return name
    with np.load(dataset / name / 'lidar.npz', allow_pickle=False) as archive:
        xyz, inverse, feat = cache_geometry(archive, foundation / (name + '.npz'))
        observation = build_graph(xyz, inverse, feat)
        # Targets are only for loss/assessment. The feature builder never sees them.
        mask = archive['sound_mask']
        taxonomy = json.loads((dataset / 'metadata/categories.json').read_text())
        lookup = np.zeros(max(taxonomy.values()) + 1, np.int16)
        for i, category in enumerate(CATEGORIES, 1):
            lookup[taxonomy[category]] = i
        raw = archive['semantic_id']
        point_classes = lookup[np.clip(raw, 0, len(lookup) - 1)]
        point_classes[(raw < 0) | (raw >= len(lookup))] = 0
        n = len(feat)
        histogram = np.stack([np.bincount(inverse, weights=point_classes == c, minlength=n)
                              for c in range(17)], 1).astype(np.int32)
        semantic = histogram.argmax(1).astype(np.int16)
        # Majority instance per voxel; annotations never define nodes or edges.
        instance_ids = archive['instance_id'].astype(np.int64)
        pairs, pair_counts = np.unique(np.stack((inverse, instance_ids), 1), axis=0, return_counts=True)
        order = np.argsort(pair_counts, kind='stable')
        instance = np.full(n, -1, np.int32)
        instance[pairs[order, 0]] = pairs[order, 1]
        positives = np.bincount(inverse, weights=mask, minlength=n).astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field, audio_context = acoustic_field(observation['centers'], waveform(dataset / name / 'audio.wav'))
    with np.load(clap / (name + '.npz'), allow_pickle=False) as a:
        embedding, text_logits = a['embedding'].copy(), a['text_logits'].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.npz')
    observation['features'] = observation['features'].astype(np.float16)
    # Uncompressed half-precision arrays avoid spending most preparation time
    # compressing already compact foundation features. np.load supports both.
    np.savez(temporary, **observation, acoustic=field.astype(np.float16), audio_context=audio_context,
                        embedding=embedding, text_logits=text_logits, positives=positives,
                        semantic=semantic, instance=instance, class_counts=histogram,
                        inverse=inverse.astype(np.int32), version=FEATURE_VERSION,
                        xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest())
    os.replace(temporary, destination)
    return name


def audit(dataset):
    with (dataset / 'metadata/samples.csv').open() as f:
        rows = list(csv.DictReader(f))
    groups = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val', 'test')}
    for split, group in groups.items():
        manifest = json.loads((dataset / f'metadata/{split}.json').read_text())
        if len(manifest) != len(set(manifest)) or set(manifest) != {r['sample'] for r in group}:
            raise ValueError(f'Invalid {split} manifest')
    scene_sets = {k: {r['scene'] for r in v} for k, v in groups.items()}
    overlaps = {f'{a}/{b}': sorted(scene_sets[a] & scene_sets[b])
                for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(overlaps.values()):
        raise ValueError('Dataset scene overlap across splits')
    return dict(dataset=str(dataset), samples={k: len(v) for k, v in groups.items()},
                scenes={k: len(v) for k, v in scene_sets.items()}, scene_overlap=overlaps,
                feature_inputs=['XYZ', 'XYZ-only frozen Sonata features', 'four-channel audio',
                                'waveform-derived CLAP features', 'fixed microphone calibration'],
                labels_used_only_for='Training losses and assessment',
                pretrained_encoder_scene_overlap='Unresolved: Sonata pretraining includes HM3D',
                test_previously_evaluated_by_other_project=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    output = ROOT / 'cache/graphs'
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports/dataset_audit.json').write_text(json.dumps(audit(DATASET), indent=2))
    encoder = json.loads((FOUNDATION / 'encoder.json').read_text())
    clap_provenance=json.loads((CLAP/'provenance.json').read_text())
    if clap_provenance['classes']!=list(CATEGORIES):
        raise ValueError('Audio category ordering mismatch')
    if encoder.get('annotation_inputs') or encoder.get('observed_rgb'):
        raise ValueError('Foundation cache must contain only observed geometry')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(dict(version=FEATURE_VERSION,
          dataset=str(DATASET), foundation_cache=str(FOUNDATION), encoder=encoder,
          clap_provenance=clap_provenance,audio_cache_has_per_waveform_hash=False), indent=2))
    tasks = []
    for split in args.splits:
        names = json.loads((DATASET / f'metadata/{split}.json').read_text())
        if args.limit:
            names = names[:args.limit]
        tasks += [(name, str(DATASET), str(FOUNDATION), str(CLAP), str(output)) for name in names]
    # Interleave splits and scenes so preliminary runs have representative data.
    np.random.default_rng(1337).shuffle(tasks)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, name in enumerate(pool.map(one, tasks, chunksize=1), 1):
            if i % 32 == 0 or i == len(tasks):
                print(f'Prepared {i}/{len(tasks)} graphs; {time.monotonic()-start:.1f}s; {name}', flush=True)


if __name__ == '__main__':
    main()
