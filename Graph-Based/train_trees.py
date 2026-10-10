"""CPU modality baselines and graph refinement, selected on full validation."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import softmax

from graph_audio.metrics import THRESHOLDS, score, summarize
from prepare import DATASET, ROOT


def inputs(a, physics=True):
    n = len(a['counts'])
    categories = softmax(a['text_logits'].astype(np.float32) / 2)
    context = np.concatenate((categories, a['audio_context']))
    arrays = [a['features'].astype(np.float32), np.broadcast_to(context, (n, len(context)))]
    if physics:
        field = a['acoustic'].astype(np.float32)
        standardized = (field - field.mean(0)) / np.maximum(field.std(0), .1)
        arrays.extend((field, standardized))
    return np.concatenate(arrays, 1)


def refine_weights(a):
    neighbors = a['neighbors']
    centers = a['centers']
    features = a['features'][:, :48].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-5)
    cosine = (features[:, None] * features[neighbors]).sum(-1)
    distance = np.linalg.norm(centers[:, None] - centers[neighbors], axis=-1)
    weights = np.exp(-(distance / .15) ** 2 + 4 * (cosine - 1)) * a['edge_valid']
    return weights


def refine(probability, a, strength, base_weights=None):
    if strength == 0:
        return probability
    neighbors = a['neighbors']
    weights = refine_weights(a) if base_weights is None else base_weights.copy()
    # Prevent oversmoothing across a confident probability boundary.
    weights *= np.exp(-6 * np.abs(probability[:, None] - probability[neighbors]))
    average = (probability[neighbors] * weights).sum(1) / np.maximum(weights.sum(1), 1e-8)
    average[weights.sum(1) == 0] = probability[weights.sum(1) == 0]
    return (1 - strength) * probability + strength * average


def manifests(split, limit=None, available=False):
    names = json.loads((DATASET / f'metadata/{split}.json').read_text())
    if available:
        names = [name for name in names if (ROOT / 'cache/graphs' / (name + '.npz')).exists()]
    rng = np.random.default_rng(1337)
    rng.shuffle(names)
    return names[:limit] if limit else names


def wait_for_cache(names):
    while True:
        ready = sum((ROOT / 'cache/graphs' / (n + '.npz')).exists() for n in names)
        if ready == len(names):
            return
        print(f'Waiting for graph cache: {ready}/{len(names)}', flush=True)
        time.sleep(20)


def training_sample(task):
    name, index, per_positive, per_negative = task
    rng = np.random.default_rng(1337 + index)
    with np.load(ROOT / 'cache/graphs' / (name + '.npz')) as a:
        counts, positives = a['counts'], a['positives']
        label = positives / np.maximum(counts, 1) >= .5
        positive, negative = np.flatnonzero(label), np.flatnonzero(~label)
        ids = np.concatenate((rng.choice(positive, min(per_positive, len(positive)), replace=False),
                              rng.choice(negative, min(per_negative, len(negative)), replace=False)))
        # Read geometry once and materialize only sampled rows, rather than a
        # full-scene 397-channel dense training matrix that is mostly discarded.
        geometry = a['features'][ids].astype(np.float32)
        field = a['acoustic'].astype(np.float32)
        normalized = (field[ids] - field.mean(0)) / np.maximum(field.std(0), .1)
        context = np.concatenate((softmax(a['text_logits'] / 2), a['audio_context']))
        values = np.concatenate((geometry, np.broadcast_to(context, (len(ids), len(context))),
                                 field[ids], normalized), 1)
        return values, label[ids], a['semantic'][ids]


def collect_training(names, per_positive=64, per_negative=256):
    # Maximum allocation is bounded; array shape defines reproducible subsampling.
    rows = len(names) * (per_positive + per_negative)
    width = 397  # 268 geometry + 21 audio context + 108 phase evidence.
    path = ROOT / 'cache/training_tree_features.mmap'
    x = np.memmap(path, mode='w+', dtype=np.float32, shape=(rows, width))
    y = np.zeros(rows, np.uint8)
    semantic = np.memmap(ROOT / 'cache/training_semantic.mmap', mode='w+', dtype=np.uint8, shape=(rows,))
    position = 0
    tasks = [(name, i, per_positive, per_negative) for i, name in enumerate(names)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (values, labels, classes) in enumerate(pool.map(training_sample, tasks), 1):
            if values.shape[1] != width:
                raise ValueError(f'Feature schema mismatch: {values.shape}')
            x[position:position + len(values)] = values
            y[position:position + len(values)] = labels
            semantic[position:position + len(values)] = classes
            position += len(values)
            if i % 128 == 0 or i == len(names):
                print(f'Tree training observations {i}/{len(names)}; selected nodes {position}', flush=True)
    x.flush()
    semantic.flush()
    pool_path = ROOT / 'cache/training_pool.json'
    temporary = pool_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(dict(capacity=rows, nodes=position,
                  width=width, training_samples=len(names), names=names), indent=2))
    temporary.replace(pool_path)
    return x[:position], y[:position]


def assess(model, names, physics, run):
    with (DATASET / 'metadata/samples.csv').open() as f:
        metadata = {row['sample']: row for row in csv.DictReader(f)}
    choices = [(strength, float(threshold)) for strength in (0., .25, .5) for threshold in THRESHOLDS]
    rows = {choice: [] for choice in choices}
    oracle = []
    for i, name in enumerate(names, 1):
        with np.load(ROOT / 'cache/graphs' / (name + '.npz')) as a:
            probability = model.predict_proba(inputs(a, physics))[:, 1]
            for strength in (0., .25, .5):
                smoothed = refine(probability, a, strength)
                for threshold in THRESHOLDS:
                    row = dict(sample=name, scene=metadata[name]['scene'], category=metadata[name]['category'],
                               **score(smoothed, a['counts'], a['positives'], float(threshold)))
                    rows[strength, float(threshold)].append(row)
            oracle.append(score(a['positives'] / np.maximum(a['counts'], 1), a['counts'], a['positives'], .5)['iou'])
        if i % 128 == 0 or i == len(names):
            print(f'{run.name}: validation {i}/{len(names)}', flush=True)
    best = max(choices, key=lambda key: np.mean([row['iou'] for row in rows[key]]))
    selected = rows[best]
    categories = defaultdict(list)
    for row in selected:
        categories[row['category']].append(row)
    report = dict(model=run.name, split='val', full_split=len(names) == 768, physics=physics,
                  graph_strength=best[0], threshold=best[1], metrics=summarize(selected),
                  per_category={key: summarize(value) for key, value in categories.items()},
                  diagnostic_voxel_oracle_mean_iou=float(np.mean(oracle)),
                  voxel_oracle_uses_labels=True, samples_detail=selected)
    (run / 'validation.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--rounds', type=int, default=350)
    p.add_argument('--wait-cache', action='store_true')
    p.add_argument('--tag', default='full')
    p.add_argument('--available', action='store_true', help='Diagnostic run on graphs prepared so far')
    args = p.parse_args()
    train, val = (manifests('train', args.train_limit, args.available),
                  manifests('val', args.val_limit, args.available))
    if args.wait_cache:
        wait_for_cache(train + val)
    x, y = collect_training(train)
    trials = []
    for physics in (False, True):
        name = f'trees-{args.tag}-' + ('physics' if physics else 'semantic')
        run = ROOT / 'runs' / name
        run.mkdir(parents=True, exist_ok=True)
        (run / 'train_manifest.json').write_text(json.dumps(train))
        (run / 'val_manifest.json').write_text(json.dumps(val))
        values = x if physics else x[:, :289]
        model = lgb.LGBMClassifier(n_estimators=args.rounds, learning_rate=.05,
                    num_leaves=31, max_depth=10, min_child_samples=128,
                    colsample_bytree=.8, reg_lambda=2., n_jobs=3,
                    random_state=1337, verbosity=-1, force_col_wise=True)
        print(f'Training {name} on {len(values)} nodes from {len(train)} samples', flush=True)
        model.fit(values, y)
        model.booster_.save_model(str(run / 'model.txt'))
        report = assess(model, val, physics, run)
        report.update(training_samples=len(train), training_nodes=len(y), rounds=args.rounds,
                      model_sha256=hashlib.sha256((run / 'model.txt').read_bytes()).hexdigest())
        (run / 'validation.json').write_text(json.dumps(report, indent=2))
        trials.append(report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('samples_detail', 'per_category')}, indent=2), flush=True)
    best = max(trials, key=lambda r: r['metrics']['mean_iou'])
    report = dict(selection_split='val', full_training=len(train) == 3264, full_validation=len(val) == 768,
                  test_used_for_selection=False, best={k: v for k, v in best.items() if k != 'samples_detail'},
                  trials=[{k: v for k, v in r.items() if k not in ('samples_detail', 'per_category')} for r in trials])
    (ROOT / f'reports/trees-{args.tag}.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
