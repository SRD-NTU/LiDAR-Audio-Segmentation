"""Evaluate the retained indoor ensemble on a selected V2 difficulty tier."""
import argparse
from pathlib import Path
import subprocess
import sys

from apex import data as D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = D.ROOT / 'ORBIT'
    tier = D.DATA.name.rsplit('-', 1)[-1]
    inputs = root / 'cache/tier-inputs' / tier
    subprocess.run([sys.executable, str(root / 'scripts/tier_inputs.py'), '--output', str(inputs)], check=True)
    geometry = root / 'cache' / ('utonia-heldout-test-v1' if args.split == 'test' else 'utonia-xyz-v1')
    command = [sys.executable, '-m', 'orbit.evaluate_audio_gate', '--split', args.split,
               '--masks', str(root / 'runs/masks-utonia-spatial-1337/best.pt'),
               '--selector', str(root / 'runs/selector-utonia-spatial-1337/best.pt'),
               '--apex', str(root / 'runs/apex-utonia-spatial-1337/best.pt'),
               '--gate', str(root / 'runs/audio-gate-utonia-1337/best.pt'),
               '--gate2', str(root / 'runs/audio-gate-pairs-utonia-1337-r2/best.pt'),
               '--gate2-weight', '1', '--geometry', str(geometry),
               '--posterior', str(inputs / f'adaptive_{args.split}_posterior.npy'),
               '--device', args.device, '--output', str(args.output)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
