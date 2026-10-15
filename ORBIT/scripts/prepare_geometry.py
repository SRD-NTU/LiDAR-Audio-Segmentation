"""Extract Sonata geometry from raw XYZ for any selected indoor V2 split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from apex import data as D
from apex.predict import range_grid
from lidarwm.sonata_geometry import SonataGeometry


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--output', type=Path, default=D.GEOMETRY)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    encoder = SonataGeometry(checkpoint=str(D.ROOT / 'ORBIT/pretrained/sonata/sonata.pth')).to(args.device).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(encoder='facebook/sonata',encoder_sha256=encoder.checkpoint_sha256,
                  adapter_version='deterministic-pooling-v2',feature_dim=256,
                  source_sha256=D.sha(Path(__file__)))
    identity=args.output/'encoder.json'
    if identity.exists():
        previous=json.loads(identity.read_text())
        if any(previous.get(key)!=manifest[key] for key in ('encoder','encoder_sha256','adapter_version','feature_dim')):
            raise ValueError('Existing encoder contract differs; use a new geometry output directory')
    if not identity.exists():identity.write_text(json.dumps(manifest,indent=2)+'\n')
    for rel in D.rels(args.split):
        with np.load(D.DATA / rel / 'lidar.npz', allow_pickle=False) as item:
            xyz = item['xyz'].astype(np.float32)
        digest = hashlib.sha256(xyz.tobytes()).hexdigest()
        output = args.output / (rel + '.npz')
        if output.exists():
            with np.load(output, allow_pickle=False) as item:
                if str(item['xyz_sha256']) != digest:
                    raise ValueError(f'Existing XYZ cache differs: {rel}')
            continue
        grid, valid, _ = range_grid(xyz)
        result = encoder._encode(torch.from_numpy(grid)[None].to(args.device),
                                 torch.from_numpy(valid)[None].to(args.device), sparse=True)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, features=result['features'].half().cpu().numpy(),
                            pixels=result['pixels'].cpu().numpy(), inverse=result['inverse'].cpu().numpy(),
                            xyz_sha256=digest)
        print(rel, flush=True)


if __name__ == '__main__':
    main()
