"""Train/evaluate APEX with scene-disjoint data and exact all-point IoU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .model import APEX, objective

THRESHOLDS = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8], np.float32)


def load_split(split, limit, workers):
    rs, post, emb, provenance = D.audio_bank(split)
    if limit:
        # Evenly sample the sorted list: smoke coverage spans scenes/categories.
        idx = np.linspace(0, len(rs) - 1, min(limit, len(rs)), dtype=int)
    else:
        idx = np.arange(len(rs))
    def load(i):
        x = D.load_inputs(rs[i], post[i], emb[i])
        x.update(D.load_targets(rs[i], x['inverse'], x['count']))
        del x['inverse']
        return x
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        items = []
        kept_bytes = 0
        budget = (26 if split == 'train' else 6) * 1024 ** 3
        for x in pool.map(load, idx):
            x['n_features'] = len(x['features'])
            if kept_bytes + x['features'].nbytes <= budget:
                kept_bytes += x['features'].nbytes
            else:
                # Stream the remaining frozen features from the existing archive.
                # Avoid duplicating tens of GB or exceeding the site's 48 GB RAM cap.
                x['features'] = None
            items.append(x)
            if len(items) % 400 == 0:
                print(f'load {split} {len(items)}/{len(idx)} {time.time()-start:.1f}s', flush=True)
    print(f'loaded {split}: {len(items)} samples in {time.time()-start:.1f}s', flush=True)
    return items, provenance


def tensor_inputs(item, device, idx=None, audio_item=None):
    if idx is None:
        idx = slice(None)
    a = item if audio_item is None else audio_item
    # A shuffled control must project the partner's posterior onto THIS geometry.
    if a is not item:
        raise ValueError('Use intervention() to reproject foreign audio')
    features = item['features']
    if features is None:
        with np.load(D.GEOMETRY / (item['rel'] + '.npz'), allow_pickle=False) as z:
            features = z['features']
    return [torch.as_tensor(features[idx], device=device, dtype=torch.float32)] + [
            torch.as_tensor(item[k][idx], device=device, dtype=torch.float32)
            for k in ('xyz', 'acoustic')] + [
                torch.as_tensor(item['audio'], device=device, dtype=torch.float32),
                torch.as_tensor(item['count'][idx], device=device, dtype=torch.float32)]


def counts_at_thresholds(p, positive, count, thresholds=THRESHOLDS):
    pred = p[:, None] >= np.asarray(thresholds)[None]
    inter = (pred * positive[:, None]).sum(0)
    predicted = (pred * count[:, None]).sum(0)
    union = predicted + positive.sum() - inter
    return inter, union, predicted


def summarize(rows, threshold_index, thresholds=THRESHOLDS):
    j = threshold_index
    ious = np.array([r['ious'][j] for r in rows])
    scenes = sorted({r['rel'].split('/')[1] for r in rows})
    per_scene = {s: [float(ious[i]) for i, r in enumerate(rows) if r['rel'].split('/')[1] == s]
                 for s in scenes}
    rng = np.random.default_rng(726)
    boot = [float(np.mean(np.concatenate([per_scene[s] for s in rng.choice(scenes, len(scenes))])))
            for _ in range(2000)]
    return dict(n=len(rows), scenes=len(scenes), threshold=float(thresholds[j]),
                mean_iou=float(ious.mean()), median_iou=float(np.median(ious)),
                micro_iou=float(sum(r['inter'][j] for r in rows) / max(sum(r['union'][j] for r in rows), 1)),
                fraction_iou_ge_half=float((ious >= .5).mean()),
                zero_iou_fraction=float((ious == 0).mean()),
                scene_bootstrap_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                by_scene={s: float(np.mean(v)) for s, v in per_scene.items()},
                by_category={c: float(np.mean([ious[i] for i, r in enumerate(rows) if r['category'] == c]))
                             for c in sorted({r['category'] for r in rows})},
                samples=[dict(rel=r['rel'], iou=float(ious[i])) for i, r in enumerate(rows)])


@torch.inference_mode()
def evaluate(model, items, device, thresholds=THRESHOLDS):
    model.eval()
    rows = []
    for item in items:
        out = model(*tensor_inputs(item, device))
        p = out['logits'].sigmoid().cpu().numpy()
        inter, union, _ = counts_at_thresholds(p, item['positive'], item['count'], thresholds)
        rows.append(dict(rel=item['rel'], category=D.CATS[item['source']],
                         ious=(inter / union.clip(1)).tolist(), inter=inter.tolist(), union=union.tolist()))
    j = int(np.argmax(np.mean([r['ious'] for r in rows], 0)))
    return summarize(rows, j, thresholds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--max-nodes', type=int, default=8192)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--no-expansion', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'best.pt').exists():
        raise FileExistsError('Use a fresh run directory; existing checkpoints are preserved')
    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit the provided Slurm job')
    tr, tp = load_split('train', args.train_limit, args.workers)
    va, vp = load_split('val', args.val_limit, args.workers)
    assert not ({x['rel'].split('/')[1] for x in tr} & {x['rel'].split('/')[1] for x in va})
    config = {**vars(args), 'output': str(args.output), 'dataset': str(D.DATA),
              'train_samples': len(tr), 'val_samples': len(va),
              'train_provenance': tp, 'val_provenance': vp,
              'source_hashes': {str(f.relative_to(Path(__file__).parents[1])): D.sha(f)
                                for f in Path(__file__).parent.glob('*.py')},
              'torch_version': torch.__version__, 'test_used': False,
              'foundation_overlap': 'Sonata HM3D pretraining scene overlap unresolved'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    from .audit import loaded_audit
    (args.output / 'data_audit.json').write_text(json.dumps(
        dict(train=loaded_audit(tr), val=loaded_audit(va)), indent=2))
    model = APEX(expand=not args.no_expansion).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * .1)
    print('parameters', sum(v.numel() for v in model.parameters()), flush=True)
    best = -1
    rng = np.random.default_rng(args.seed)
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        loss_total = 0.
        opt.zero_grad(set_to_none=True)
        for step, i in enumerate(order):
            item = tr[i]
            n = item['n_features']
            idx = rng.choice(n, min(n, args.max_nodes), replace=False)
            inp = tensor_inputs(item, args.device, idx)
            out = model(*inp)
            target = torch.as_tensor(item['target'][idx], device=args.device)
            cat = torch.as_tensor(item['category'][idx], device=args.device)
            src = torch.tensor(item['source'], device=args.device)
            loss = objective(out, target, inp[-1], cat, src)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: epoch {epoch}, {item["rel"]}')
            group_size = min(args.accum, len(tr) - (step // args.accum) * args.accum)
            (loss / group_size).backward()
            loss_total += float(loss.detach())
            if (step + 1) % args.accum == 0 or step + 1 == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                opt.step(); opt.zero_grad(set_to_none=True)
            if (step + 1) % 500 == 0:
                print(f'epoch {epoch+1} step {step+1}/{len(tr)} loss {loss_total/(step+1):.4f}', flush=True)
        report = evaluate(model, va, args.device)
        row = dict(epoch=epoch+1, train_loss=loss_total/len(tr),
                   mean_iou=report['mean_iou'], threshold=report['threshold'],
                   elapsed_seconds=time.time()-start)
        history.append(row)
        print(json.dumps(row), flush=True)
        if report['mean_iou'] > best:
            best = report['mean_iou']
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch+1,
                            threshold=report['threshold'], expand=model.expand), args.output / 'best.pt')
            (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        schedule.step()
    print('TRAINING_COMPLETE', flush=True)
    checkpoint = torch.load(args.output / 'best.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(checkpoint['model'])
    from .controls import run_controls
    run_controls(model, va, checkpoint['threshold'], args.device, args.output / 'controls.json')
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(
        training_complete=True, epochs=args.epochs, checkpoint_sha256=D.sha(args.output/'best.pt'),
        full_training=args.train_limit is None, full_validation=args.val_limit is None,
        test_evaluated=False),indent=2))


if __name__ == '__main__':
    main()
