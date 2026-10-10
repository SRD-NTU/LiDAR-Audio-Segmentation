"""Strong geometry warm start, followed by calibrated acoustic graph grounding."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.nn import functional as F

from graph_audio.metrics import THRESHOLDS, score, summarize, threshold_weighted_scores
from graph_audio.model import AcousticGraphWorld
from prepare import ROOT, DATASET
from train_trees import refine, refine_weights, manifests, wait_for_cache


@torch.inference_mode()
def assess(model, run, learned_audio=False, audio_variant='clap_mlp'):
    model.eval()
    names = manifests('val')
    wait_for_cache(names)
    with (DATASET / 'metadata/samples.csv').open() as f:
        metadata = {r['sample']: r for r in csv.DictReader(f)}
    choices = [(alpha, strength, float(t)) for alpha in (0., .5, 1., 2.)
               for strength in (0., .5) for t in THRESHOLDS]
    rows = {choice: [] for choice in choices}
    for i, name in enumerate(names, 1):
        with np.load(ROOT / 'cache/graphs' / (name + '.npz')) as a:
            x = torch.from_numpy(a['features'].astype(np.float32))
            semantic = model.semantic(model.geometry(x)).softmax(-1).numpy()
            if learned_audio:
                audio_root = ROOT/('cache/audio_category_spectral' if audio_variant=='spectral' else 'cache/audio_category')
                with np.load(audio_root/(name+'.npz')) as audio:
                    prior = softmax(audio['logits'])
            else:
                prior = softmax(a['text_logits'] / 2)
            match = (semantic[:, 1:] * prior[None]).sum(-1)
            field = a['acoustic'].astype(np.float32)
            spatial = field.mean(1)
            spatial = (spatial - spatial.mean()) / max(float(spatial.std()), .1)
            base_weights = refine_weights(a)
            counts,positives = a['counts'],a['positives']
            for alpha in (0., .5, 1., 2.):
                probability = np.clip(match * np.exp(np.clip(alpha * spatial, -5, 5)), 0, 1)
                for strength in (0., .5):
                    refined = refine(probability, a, strength,base_weights)
                    for t,metric in zip(THRESHOLDS,threshold_weighted_scores(refined,counts,positives)):
                        rows[alpha, strength, float(t)].append(dict(sample=name,
                              scene=metadata[name]['scene'], category=metadata[name]['category'],
                              **metric))
        if i % 32 == 0 or i == len(names):
            print(f'Semantic/acoustic validation {i}/{len(names)}', flush=True)
    selected = max(choices, key=lambda key: np.mean([r['iou'] for r in rows[key]]))
    report = dict(model=run.name, split='val', full_split=True, training_samples=3264,learned_audio=learned_audio,
                  alpha=selected[0], graph_strength=selected[1], threshold=selected[2],
                  metrics=summarize(rows[selected]), samples_detail=rows[selected])
    (run / 'validation.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--wait-pool', action='store_true')
    args = p.parse_args()
    path = ROOT / 'cache/training_pool.json'
    while not path.exists() or json.loads(path.read_text())['training_samples'] != 3264:
        if not args.wait_pool:
            raise ValueError('Build the full training feature pool first')
        print('Waiting for full training feature pool', flush=True)
        time.sleep(20)
    meta = json.loads(path.read_text())
    torch.set_num_threads(3); torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    x = np.memmap(ROOT / 'cache/training_tree_features.mmap', mode='r', dtype=np.float32,
                  shape=(meta['capacity'], meta['width']))[:meta['nodes'], :268]
    y = np.memmap(ROOT / 'cache/training_semantic.mmap', mode='r', dtype=np.uint8,
                  shape=(meta['capacity'],))[:meta['nodes']]
    model = AcousticGraphWorld(use_graph=False)
    parameters = [*model.geometry.parameters(), *model.semantic.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    counts = np.bincount(y, minlength=17)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1)).clip(1, 20)
    class_weights[0] *= .3
    weights = torch.tensor(class_weights, dtype=torch.float32)
    run = ROOT / 'runs/semantic-head'; run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(dict(training_samples=3264, validation_samples=768,
                   epochs=args.epochs, foundation_frozen=True, type='semantic', no_graph=True), indent=2))
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(y)); total = 0; start = time.monotonic()
        for ids in np.array_split(order, max(1, len(order) // 2048)):
            values = torch.from_numpy(np.asarray(x[ids]).copy())
            labels = torch.from_numpy(np.asarray(y[ids]).astype(np.int64))
            logits = model.semantic(model.geometry(values))
            loss = F.cross_entropy(logits, labels, weight=weights, label_smoothing=.02)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())
        print(f'Geometry pretraining epoch {epoch}: loss {total/max(1,len(order)//2048):.4f}; '
              f'{time.monotonic()-start:.1f}s', flush=True)
    torch.save(dict(geometry=model.geometry.state_dict(), semantic=model.semantic.state_dict(),
                    training_samples=3264, epochs=args.epochs), run / 'geometry.pt')
    # Keep a deployable complete state, but only the geometry/class head is trained.
    torch.save(dict(model=model.state_dict(), config=dict(no_graph=True, type='semantic'), epoch=args.epochs-1),
               run / 'best.pt')
    report = assess(model, run)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples_detail'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
