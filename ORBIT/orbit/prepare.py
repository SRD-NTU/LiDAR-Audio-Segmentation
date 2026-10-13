"""Build compact train/val supervision caches with the verified point mapping.

No test split is accepted. Perfect mask descriptors are marked diagnostic-only;
they are not inputs to the deployable pipeline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from apex import data as D
from .objects import supervision, pool_masks, neighbors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--train-nodes', type=int, default=10000)
    p.add_argument('--geometry', type=Path)
    p.add_argument('--spatial', type=Path)
    args = p.parse_args()
    if args.geometry is not None:
        D.GEOMETRY = args.geometry
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for split in ('train', 'val'):
        rs, post, emb, prov = D.audio_bank(split)
        if args.spatial:
            posterior_file=args.spatial/f'adaptive_{split}_posterior.npy'
            post=np.load(posterior_file)
            if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Wrong posterior shape')
            prov['replacement_posterior_sha256']=D.sha(posterior_file)
        provenance[split] = prov
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        start = time.time()
        def load(i):
            rel = rs[i]
            dest = split_dir / f'{i:05d}.npz'
            if dest.exists():
                raise FileExistsError(f'Use a fresh preparation directory: {dest}')
            x = D.load_inputs(rel, post[i], emb[i])
            y = supervision(rel, x)
            g, a = pool_masks(x, y['memberships'])
            # Deterministic instance-aware TRAIN sampling. All validation voxels
            # remain present. This is supervised training sampling, not inference.
            rng = np.random.default_rng(3307 + i)
            n = len(x['count'])
            take = np.arange(n)
            if split == 'train' and n > args.train_nodes:
                mandatory = []
                for j in range(len(y['categories'])):
                    idx = np.flatnonzero(y['node_object'] == j)
                    if len(idx):
                        mandatory.append(rng.choice(idx, min(96, len(idx)), replace=False))
                src = np.flatnonzero(y['positive'] > 0)
                mandatory.append(rng.choice(src, min(512, len(src)), replace=False))
                keep = np.unique(np.concatenate(mandatory))
                if len(keep) > args.train_nodes // 2:
                    keep = rng.choice(keep, args.train_nodes // 2, replace=False)
                rest = np.setdiff1d(take, keep, assume_unique=True)
                take = np.sort(np.r_[keep, rng.choice(rest, args.train_nodes - len(keep), replace=False)])
            obs = {k: x[k][take] for k in ('features', 'xyz', 'count', 'acoustic')}
            # Voxel memberships are separated from the observation feature keys.
            np.savez(dest, rel=rel, **obs, audio=x['audio'], knn=neighbors(obs['xyz']),
                     node_object=y['node_object'][take], node_category=y['node_category'][take],
                     positive=y['positive'][take], source_class=y['source_class'],
                     source_object=y['source_object'], object_categories=y['categories'])
            return dict(rel=rel, geometry=g, acoustic=a, audio=x['audio'],
                        categories=y['categories'], target=y['source_object'], source=y['source_class'],
                        n_original_voxels=n, n_retained_voxels=len(take))
        rows = []
        with ThreadPoolExecutor(args.workers) as pool:
            for i, row in enumerate(pool.map(load, range(len(rs)))):
                rows.append(row)
                if (i + 1) % 200 == 0:
                    print(f'prepare {split} {i+1}/{len(rs)} {time.time()-start:.1f}s', flush=True)
        # Ragged pooled descriptors are safe internally generated torch objects.
        import torch
        torch.save(rows, args.output / f'perfect_objects_{split}.pt')
        (args.output / f'{split}_manifest.json').write_text(json.dumps(rs))
        print(f'PREPARED {split} {len(rs)} in {time.time()-start:.1f}s', flush=True)
    report = dict(provenance=provenance, train_nodes=args.train_nodes, test_used=False, geometry=str(D.GEOMETRY),
                  corrected_alignment=True, source_hashes={f.name:D.sha(f) for f in Path(__file__).parent.glob('*.py')},
                  warning='perfect_objects caches contain ground truth masks: privileged diagnostics ONLY')
    (args.output / 'COMPLETE.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
