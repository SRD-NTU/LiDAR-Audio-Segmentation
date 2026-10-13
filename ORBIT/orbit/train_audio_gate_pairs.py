"""Contrastive sounding-object query training on identical-LiDAR audio swaps.

Pairs are selected from TRAIN scenes only. The validation pair metric is a
privileged checkpoint diagnostic, not deployable segmentation performance.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from apex import data as D
from .audio_gate import AudioSeedGate
from .masks import PromptMasks
from .selector import ObjectSelector
from .linear_selector import LinearObjectSelector
from .train_audio_gate import gate_logits
from .train_masks import load_split


def same_scan_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        rel = row['rel']
        meta = json.loads((D.DATA / rel / 'metadata.json').read_text())
        listener = meta['listener']
        key = (rel.split('/')[1], *listener['position_world'], *listener['rotation_world'])
        grouped[key].append(row)
    result = []
    for group in grouped.values():
        scans = defaultdict(list)
        for row in group:
            xyz = np.asarray(row['xyz'])
            key = (xyz.shape, hashlib.sha256(xyz.tobytes()).digest())
            scans[key].append(row)
        for scan in scans.values():
            if len({int(row['source_object']) for row in scan}) < 2:
                continue
            if any(not np.array_equal(row['xyz'], scan[0]['xyz']) for row in scan[1:]):
                raise AssertionError('XYZ hash collision or inconsistent cache')
            result.append(scan)
    return result


def pairs_from_group(group):
    return [(a, b) for i, a in enumerate(group) for b in group[i+1:]
            if int(a['source_object']) != int(b['source_object'])]


def query_seeds(a, b, rng):
    ids = a['node_object']
    seeds = [int(rng.choice(np.flatnonzero(a['positive'] > 0))),
             int(rng.choice(np.flatnonzero(b['positive'] > 0)))]
    other = np.unique(ids[ids >= 0])
    other = [int(v) for v in other if int(v) not in
             (int(a['source_object']), int(b['source_object']))]
    for object_id in rng.choice(other, min(4, len(other)), replace=False):
        seeds.append(int(rng.choice(np.flatnonzero(ids == object_id))))
    background = np.flatnonzero(ids < 0)
    if len(background):
        seeds.extend(map(int, rng.choice(background, min(2, len(background)), replace=False)))
    return seeds


def load_frozen(mask_checkpoint, selector_checkpoint, device):
    mc = torch.load(mask_checkpoint, map_location='cpu', weights_only=False)
    sc = torch.load(selector_checkpoint, map_location='cpu', weights_only=False)
    masks = PromptMasks().to(device).eval()
    masks.load_state_dict(mc['model'])
    selector = (LinearObjectSelector() if sc.get('kind') == 'linear'
                else ObjectSelector()).to(device).eval()
    selector.load_state_dict(sc['model'])
    return masks, selector, None, None


@torch.inference_mode()
def evaluate(groups, models, gate, device):
    gate.eval()
    correct = []
    for group in groups:
        for i, (a, b) in enumerate(pairs_from_group(group)):
            seeds = query_seeds(a, b, np.random.default_rng(8177 + i))
            sa = gate_logits(a, models, gate, device, seeds)
            sb = gate_logits(b, models, gate, device, seeds)
            correct.append((bool(sa[0] > sa[1]), bool(sb[1] > sb[0])))
    if not correct:
        raise ValueError('No identical-XYZ different-source validation pairs')
    return dict(n_pairs=len(correct), privileged_true_source_seeds=True,
                deployable_segmentation_result=False,
                pair_both_accuracy=float(np.mean([x and y for x, y in correct])),
                individual_accuracy=float(np.mean(correct)))


def main():
    p = argparse.ArgumentParser()
    for name in ('cache', 'masks', 'selector', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'best.pt').exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    models = load_frozen(args.masks, args.selector, args.device)
    for model in models[:2]:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    tr = load_split(args.cache, 'train')
    va = load_split(args.cache, 'val')
    train_groups = same_scan_groups(tr)
    val_groups = same_scan_groups(va)
    print(json.dumps(dict(train_exact_xyz_groups=len(train_groups),
                          val_exact_xyz_groups=len(val_groups))), flush=True)
    if not train_groups or not val_groups:
        raise ValueError('Missing shared-XYZ source pairs')
    if {row['rel'].split('/')[1] for group in train_groups for row in group} & \
       {row['rel'].split('/')[1] for group in val_groups for row in group}:
        raise AssertionError('TRAIN and VAL scene overlap')
    gate = AudioSeedGate().to(args.device)
    opt = torch.optim.AdamW(gate.parameters(), lr=.0005, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=.00005)
    config = dict(cache=str(args.cache), epochs=args.epochs, seed=args.seed,
                  test_used=False, train_same_scan_groups=len(train_groups),
                  val_same_scan_groups=len(val_groups),
                  masks_sha256=D.sha(args.masks), selector_sha256=D.sha(args.selector),
                  source_sha256=D.sha(Path(__file__)))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(args.seed)
    history = []
    best = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        gate.train()
        loss_total = 0.0
        for step, j in enumerate(rng.permutation(len(train_groups))):
            options = pairs_from_group(train_groups[j])
            a, b = options[int(rng.integers(len(options)))]
            seeds = query_seeds(a, b, rng)
            sa = gate_logits(a, models, gate, args.device, seeds)
            sb = gate_logits(b, models, gate, args.device, seeds)
            target_a = torch.zeros(1, device=args.device, dtype=torch.long)
            target_b = torch.ones(1, device=args.device, dtype=torch.long)
            # The same two geometric objects must swap ranks when the sound swaps.
            loss = (F.cross_entropy(sa[None], target_a) +
                    F.cross_entropy(sb[None], target_b) +
                    .5 * F.softplus(1.0 - ((sa[0] - sa[1]) - (sb[0] - sb[1]))))
            if not torch.isfinite(loss):
                raise FloatingPointError((a['rel'], b['rel']))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
            opt.step()
            loss_total += float(loss.detach())
            if (step + 1) % 250 == 0:
                print(f'pair epoch {epoch} {step+1}/{len(train_groups)} loss {loss_total/(step+1):.4f}', flush=True)
        scheduler.step()
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            result = evaluate(val_groups, models, gate, args.device)
            info = dict(epoch=epoch, train_loss=loss_total/len(train_groups),
                        **{k:v for k,v in result.items() if k not in ('privileged_true_source_seeds',
                                                                       'deployable_segmentation_result')},
                        elapsed_seconds=time.time()-start)
            print(json.dumps(info), flush=True)
            history.append(info)
            if result['pair_both_accuracy'] > best:
                best = result['pair_both_accuracy']
                torch.save(dict(model=gate.state_dict(), config=config, epoch=epoch),
                           args.output / 'best.pt')
                (args.output / 'privileged_validation.json').write_text(json.dumps(result, indent=2))
            (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(test_used=False,
        best_privileged_pair_both_accuracy=best,
        checkpoint_sha256=D.sha(args.output / 'best.pt')), indent=2))


if __name__ == '__main__':
    main()
