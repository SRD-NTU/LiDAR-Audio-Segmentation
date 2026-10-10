"""Precompute label-free SRP-PHAT for all dataset samples once."""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .spatial import srp_features


def one(item):
    root, name = item
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        rate, audio = wavfile.read(Path(root) / name / "audio.wav")
    return srp_features(audio, rate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        names = json.loads((args.root / "metadata" / f"{split}.json").read_text())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            values = list(pool.map(one, ((str(args.root), name) for name in names), chunksize=8))
        likelihood = np.stack([v[0] for v in values])
        coherence = np.asarray([v[1] for v in values], dtype=np.float32)
        path = args.output / f"{split}.npz"
        np.savez_compressed(path, samples=np.asarray(names), likelihood=likelihood,
                            coherence=coherence)
        print(json.dumps({"split": split, "samples": len(names), "coherence_median": float(np.median(coherence)),
                          "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
