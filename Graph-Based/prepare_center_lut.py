"""Upgrade observation caches with differentiable acoustic delay lookup tables."""
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from graph_audio.features import acoustic_field, waveform
from prepare import ROOT, DATASET


def update(name):
    path = ROOT / 'cache/graphs' / (name + '.npz')
    destination = ROOT / 'cache/phase_lut' / (name + '.npy')
    if destination.exists():
        return name
    with np.load(path) as a:
        if 'delay_tables' in a.files:
            tables = a['delay_tables']
        else:
            centers = a['centers']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, _, tables = acoustic_field(centers, waveform(DATASET / name / 'audio.wav'), True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.npy')
    np.save(temporary, tables.astype(np.float16))
    os.replace(temporary, destination)
    return name


def main():
    names = sum((json.loads((DATASET / f'metadata/{split}.json').read_text()) for split in ('train', 'val')), [])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, _ in enumerate(pool.map(update, names), 1):
            if i % 256 == 0 or i == len(names):
                print(f'Center phase lookup tables {i}/{len(names)}', flush=True)
    (ROOT / 'cache/graphs/center_lut.ready').write_text(str(len(names)))


if __name__ == '__main__':
    main()
