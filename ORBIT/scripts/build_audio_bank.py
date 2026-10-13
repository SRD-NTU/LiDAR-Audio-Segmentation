"""Cache observation-only SRP and CLAP features for ORBIT evaluation/training."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from apex import data as D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--output', type=Path, default=D.AUDIO_BANK)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f'{args.split}.npz'
    if destination.exists():
        raise FileExistsError(destination)
    from lidarwm.foundation import FrozenClap

    calibration_file = D.DATA / 'metadata/microphone_array.json'
    calibration = json.loads(calibration_file.read_text())
    positions = np.asarray(calibration['positions_lidar_m'])
    rels = D.rels(args.split)
    old_cwd = Path.cwd()
    try:
        os.chdir(D.ROOT / 'LiDAR-WM')
        clap = FrozenClap(args.device)
        posteriors, embeddings = [], []
        for index, rel in enumerate(rels, 1):
            wave, rate = sf.read(D.DATA / rel / 'audio.wav', dtype='float32')
            if rate != 44100 or wave.ndim != 2 or wave.shape[1] != 4:
                raise ValueError(f'Expected four 44100 Hz channels: {rel}')
            posteriors.append(D.acoustic.srp_phat_3d(wave.astype(np.float64), rate, positions))
            embedding, _ = clap([wave], sample_rate=rate)
            embeddings.append(embedding[0].cpu().numpy())
            if index % 50 == 0:
                print(f'{args.split}: {index}/{len(rels)}', flush=True)
    finally:
        os.chdir(old_cwd)
    np.savez_compressed(destination, rels=np.asarray(rels),
                        post=np.asarray(posteriors, dtype=np.float32),
                        emb=np.asarray(embeddings, dtype=np.float32))
    print(json.dumps(dict(output=str(destination), observations=len(rels),
                          calibration_sha256=D.sha(calibration_file),
                          annotation_fields_opened=False), indent=2))


if __name__ == '__main__':
    main()
