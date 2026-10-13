"""Find controlled same-LiDAR, different-sounding-source observations.

Reads TRAIN/VAL source metadata and observation XYZ, but not segmentation masks.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from apex import data as D


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('train', 'val'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = defaultdict(list)
    for rel in D.rels(args.split):
        meta = json.loads((D.DATA / rel / 'metadata.json').read_text())
        listener = meta['listener']
        key = (rel.split('/')[1], *listener['position_world'], *listener['rotation_world'])
        groups[key].append((rel, int(meta['source']['instance_id'])))
    candidates = [items for items in groups.values() if len({src for _, src in items}) > 1]
    matched = []
    for items in candidates:
        by_hash = defaultdict(list)
        for rel, source in items:
            with np.load(D.DATA / rel / 'lidar.npz', allow_pickle=False) as z:
                xyz = z['xyz']
                digest = hashlib.sha256(xyz.tobytes()).hexdigest()
            by_hash[digest].append((rel, source))
        for digest, scan in by_hash.items():
            if len({src for _, src in scan}) > 1:
                matched.append(dict(xyz_sha256=digest,
                                    observations=[dict(rel=rel, source_instance_id=src)
                                                  for rel, src in scan]))
    output = dict(split=args.split, test_used=False,
                  source_metadata_read=True, segmentation_masks_read=False,
                  total_observations=len(D.rels(args.split)),
                  same_listener_multi_source_groups=len(candidates),
                  same_xyz_multi_source_groups=len(matched),
                  observations_in_same_xyz_groups=sum(len(x['observations']) for x in matched),
                  pairs_with_distinct_sources=sum(sum(a['source_instance_id'] != b['source_instance_id']
                                                      for i, a in enumerate(x['observations'])
                                                      for b in x['observations'][i+1:]) for x in matched),
                  groups=matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='groups'}, indent=2))


if __name__ == '__main__':
    main()
