"""Observation-only predictions/features from the existing trained world model.

CPU fallback is supported while GPU allocations wait. Targets are never passed
to the teacher; cached point probabilities retain the original point order.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WM = Path(os.environ.get('LIDAR_WM_ROOT', ROOT.parent / 'LiDAR-WM')).expanduser().resolve()
DATASET = Path(os.environ.get('AUDIO_LIDAR_DATA_ROOT', ROOT.parent / 'data')).expanduser().resolve()


def load_teacher(device):
    from graph_audio.local_assets import configure_runtime
    configure_runtime()
    if device.type == 'cpu':
        # This host's oneDNN FP32 convolution path produced NaNs from finite
        # inputs/weights. The native CPU path is verified finite; CUDA is intact.
        torch.backends.mkldnn.enabled = False
    from lidarwm.engine import build_model
    path = WM / 'runs/foundation-joint-control/best.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    original = Path.cwd()
    try:
        os.chdir(WM)
        model = build_model(state['config']).to(device).eval()
    finally:
        os.chdir(original)
    model.load_state_dict(state['model'])
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--reverse', action='store_true', help='Cooperate with earlier cache workers from the opposite end')
    p.add_argument('--splits', nargs='+', choices=['train','val'], default=['train','val'])
    args = p.parse_args()
    torch.set_num_threads(3)
    teacher_root = ROOT / 'cache/teacher'
    teacher_root.mkdir(parents=True, exist_ok=True)
    (teacher_root / 'enabled').write_text('Observation-only pretrained world context\n')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Run CUDA inference inside a Slurm allocation')
    model, digest = load_teacher(device)
    from lidarwm.data import AudioLidarDataset, collate_samples
    names = sum((json.loads((DATASET / f'metadata/{s}.json').read_text()) for s in ('train', 'val')), [])
    np.random.default_rng(1337).shuffle(names)
    if device.type == 'cuda' or args.reverse:
        names.reverse()  # Reduce overlap with an already active CPU fallback.
    all_names = list(names)
    names = [name for name in names if name.split('/')[0] in args.splits]
    names = names[args.worker_index::args.workers]
    if args.limit:
        names = names[:args.limit]
    datasets = {s: AudioLidarDataset(DATASET, s, 3,
                   feature_root=WM / 'cache/clap', antialias=True,
                   geometry_feature_root=WM / 'cache/sonata-xyz-v2') for s in ('train', 'val')}
    indices = {record.relative_path: (s, i) for s, dataset in datasets.items()
               for i, record in enumerate(dataset.records)}
    captured = {}
    def capture(module, incoming):
        captured['geometry'] = incoming[0].detach()
    handle = model.object_features.register_forward_pre_hook(capture)
    start = time.monotonic()
    with torch.backends.nnpack.flags(enabled=False):
        for i, name in enumerate(names, 1):
            destination = teacher_root / (name + '.npz')
            if destination.exists():
                with np.load(destination) as a:
                    if str(a['teacher_sha256']) != digest:
                        raise ValueError('Teacher checkpoint provenance mismatch')
                    if not all(np.isfinite(a[k]).all() for k in ('geometry', 'probability', 'point_probability')):
                        raise FloatingPointError(f'Invalid cached teacher observations: {destination}')
                continue
            split, index = indices[name]
            batch = collate_samples([datasets[split][index]])
            # Explicit observation whitelist. No mask, category or semantic label.
            observations = {key: batch[key].to(device) for key in
                            ('lidar', 'audio', 'valid', 'audio_embedding', 'text_logits')}
            observations['geometry_features'] = [{k: v.to(device) for k, v in entry.items()}
                                                  for entry in batch['geometry_features']]
            output = model(**observations)
            ray = batch['ray_id'][0].numpy()
            row, col = ray % 64, ray // 64
            point_probability = output['mask_logits'][0, 0].sigmoid().float().cpu().numpy()[row, col]
            point_features = torch.cat((captured['geometry'], output['semantic_logits']), 1)[0]
            point_features = point_features.float().cpu().numpy()[:, row, col].T
            with np.load(ROOT / 'cache/graphs' / (name + '.npz')) as a:
                inverse, counts = a['inverse'], a['counts']
                xyz_digest = str(a['xyz_sha256'])
            if len(inverse) != len(point_probability):
                raise ValueError('Teacher/graph point ordering mismatch')
            n = len(counts)
            pooled = np.column_stack([np.bincount(inverse, weights=point_features[:, d], minlength=n)
                                       / np.maximum(counts, 1) for d in range(41)]).astype(np.float16)
            probability = (np.bincount(inverse, weights=point_probability, minlength=n)
                           / np.maximum(counts, 1)).astype(np.float32)
            if not all(np.isfinite(v).all() for v in (pooled, probability, point_probability)):
                raise FloatingPointError(f'Nonfinite teacher observations: {name}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(temporary, geometry=pooled, probability=probability,
                     point_probability=point_probability, teacher_sha256=digest,
                     xyz_sha256=xyz_digest, backend=args.device, precision='float32')
            os.replace(temporary, destination)
            if i % 32 == 0 or i == len(names):
                print(f'Teacher worker {args.worker_index}: {i}/{len(names)}; '
                      f'{time.monotonic()-start:.1f}s', flush=True)
    handle.remove()
    if all((teacher_root / (name + '.npz')).exists() for name in all_names):
        (teacher_root / 'teacher.ready').write_text(json.dumps(dict(samples=len(all_names), checkpoint_sha256=digest)))


if __name__ == '__main__':
    main()
