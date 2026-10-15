"""Build observation-only full V2 audio caches without the archived L2I project."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from apex import data as D
from lidarwm.foundation import FrozenClap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # Shared full banks are row-selected for easy/medium/hard at load time.
    full = D.ROOT / 'dataset/v2-tetra4-4channel-full'
    rows = D.rels(args.split, full)
    calibration = full / 'metadata/microphone_array.json'
    identity = hashlib.sha256(calibration.read_bytes())
    identity.update(json.dumps({'n_dirs': D.acoustic.N_DIRS, 'band': list(D.acoustic.BAND)}).encode())
    with ThreadPoolExecutor(8) as pool:
        for content in pool.map(lambda rel: (full / rel / 'COMPLETE.json').read_bytes(), rows):
            identity.update(content)
    output = D.ROOT / 'ORBIT/cache/audio-legacy'
    output.mkdir(parents=True, exist_ok=True)
    srp = output / f'posteriors_{args.split}_{len(rows)}_{identity.hexdigest()[:16]}.npy'
    clap_path = output / f'clap_tetra_{args.split}.npz'
    if srp.exists() or clap_path.exists():
        raise FileExistsError('Audio caches already exist; preserve them or select a fresh dataset version')
    cal = json.loads(calibration.read_text())
    clap = FrozenClap(args.device)
    posteriors, embeddings, waveform_hashes = [], [], []
    for i, rel in enumerate(rows, 1):
        path = full / rel / 'audio.wav'
        wave, rate = sf.read(path, dtype='float32')
        if rate != 44100 or wave.ndim != 2 or wave.shape[1] != 4:
            raise ValueError(f'Expected four physical 44100 Hz channels: {rel}')
        post = D.acoustic.srp_phat_3d(wave.astype(np.float64), rate, np.asarray(cal['positions_lidar_m']))
        if not np.isfinite(post).all():
            raise ValueError(f'Non-finite SRP posterior: {rel}')
        embedding, _ = clap([wave], sample_rate=rate)
        posteriors.append(post.astype(np.float32))
        embeddings.append(embedding[0].cpu().numpy())
        waveform_hashes.append(D.sha(path))
        if i % 100 == 0:
            print(f'{args.split}: {i}/{len(rows)}', flush=True)
    np.save(srp, np.asarray(posteriors, np.float32))
    np.savez(clap_path, rels=rows, emb=np.asarray(embeddings, np.float32),
             waveform_sha256=waveform_hashes, calibration_sha256=D.sha(calibration))
    (output / f'{args.split}_identity.json').write_text(json.dumps({
        'rels': rows, 'waveform_sha256': waveform_hashes,
        'calibration_sha256': D.sha(calibration), 'source_sha256': D.sha(Path(__file__)),
        'srp_sha256': D.sha(srp), 'clap_sha256': D.sha(clap_path),
    }, indent=2) + '\n')
    dependencies = {str(Path(__file__).resolve()): D.sha(Path(__file__)),
                    str(Path(D.acoustic.__file__).resolve()): D.sha(D.acoustic.__file__)}
    model_root = D.ROOT / 'ORBIT/pretrained/clap-hub/models--laion--clap-htsat-unfused'
    for path in sorted(model_root.rglob('*')):
        if path.is_file() and ('snapshots' in path.parts or 'refs' in path.parts):
            dependencies[str(path)] = D.current_sha(path)
    identity = dict(schema=2, rels=rows, waveform_sha256=waveform_hashes,
                    calibration_sha256=D.sha(calibration), srp_sha256=D.sha(srp),
                    clap_sha256=D.sha(clap_path), dependencies=dependencies, provenance='generated-v2')
    clap_path.with_suffix('.npz.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(srp, clap_path)


if __name__ == '__main__':
    main()
