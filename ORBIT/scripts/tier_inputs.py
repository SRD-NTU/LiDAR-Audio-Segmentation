"""Select existing frozen full-dataset posteriors by explicit observation IDs."""
import argparse
import json
from pathlib import Path

import numpy as np
from apex import data as D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = D.ROOT / 'ORBIT'
    spatial = root / 'cache/spatial-v1'
    heldout = root / 'cache/utonia-heldout-test-v1'
    report = {'dataset': str(D.DATA), 'splits': {}}
    for split in ('train', 'val', 'test'):
        if split == 'test':
            source = heldout / 'adaptive_test_posterior.npy'
            rows = json.loads((heldout / 'rels.json').read_text())
        else:
            source = spatial / f'adaptive_{split}_posterior.npy'
            with np.load(spatial / f'{split}_observation_features.npz', allow_pickle=False) as bank:
                rows = list(bank['rels'])
        selected = D.rels(split)
        lookup = {rel: i for i, rel in enumerate(rows)}
        if len(lookup) != len(rows):
            raise ValueError('Duplicate source posterior identities')
        if any(rel not in lookup for rel in selected):
            raise ValueError('Selected observations missing from frozen posterior bank')
        full = D.load_posterior(source, rows, dataset=D.ROOT/'dataset/v2-tetra4-4channel-full')
        if full.shape != (len(rows), D.acoustic.N_DIRS):
            raise ValueError('Source posterior shape differs from its identities')
        target = args.output / f'adaptive_{split}_posterior.npy'
        values = full[[lookup[rel] for rel in selected]]
        if target.exists():
            if not np.array_equal(np.load(target), values):
                raise ValueError(f'Existing tier posterior differs: {target}')
        else:
            np.save(target, values)
        (args.output / f'{split}_rels.json').write_text(json.dumps(selected, indent=2) + '\n')
        D.write_posterior_identity(target, selected, dependencies={str(source):D.sha(source)})
        report['splits'][split] = {'rows': len(selected), 'source_sha256': D.sha(source),
                                   'posterior_sha256': D.sha(target)}
    (args.output / 'COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
