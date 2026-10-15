"""Predeclared full-data consistency study; run GPU stages only through Slurm.

Keep the frozen Utonia/spatial encoder, selector, APEX, seed, candidate thresholds
and epoch budgets fixed. This first study evaluates the mask/gate corrections,
not a new backbone or a learned world model. TEST cannot select checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from apex import data as D

ROOT = D.ROOT
ORBIT = ROOT / 'ORBIT'
STUDY = ORBIT / 'reports/consistent-v2-full-20261014'
CACHE = ORBIT / 'cache/aligned-utonia-consistent-v2'
POST = ORBIT / 'cache/tier-inputs/full'
FROZEN_MASK = ORBIT / 'runs/masks-utonia-spatial-1337/best.pt'
MASK = ORBIT / 'runs/masks-consistent-v2-full-1337/best.pt'
SELECTOR = ORBIT / 'runs/selector-utonia-spatial-1337/best.pt'
APEX = ORBIT / 'runs/apex-utonia-spatial-1337/best.pt'
FROZEN_GATE = ORBIT / 'runs/audio-gate-utonia-1337/best.pt'
FROZEN_PAIR = ORBIT / 'runs/audio-gate-pairs-utonia-1337-r2/best.pt'
GATE = ORBIT / 'runs/audio-gate-consistent-v2-full-1337/best.pt'
PAIR = ORBIT / 'runs/audio-gate-pairs-consistent-v2-full-1337/best.pt'
FGATE = ORBIT / 'runs/audio-gate-consistent-v2-frozen-masks-1337/best.pt'
FPAIR = ORBIT / 'runs/audio-gate-pairs-consistent-v2-frozen-masks-1337/best.pt'


def run(module, *arguments, script=False):
    command = [sys.executable, '-u'] + ([str(ORBIT / 'scripts' / module)] if script else ['-m', module])
    command += list(map(str, arguments))
    print(json.dumps({'command': command, 'time': time.time()}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate(name, masks, gate, pair=None, split='val', control='matched'):
    arguments = ['--masks', masks, '--selector', SELECTOR, '--apex', APEX,
                 '--gate', gate, '--split', split, '--control', control,
                 '--geometry', ORBIT / 'cache' / ('utonia-heldout-test-v1' if split == 'test' else 'utonia-xyz-v1'),
                 '--posterior', POST / f'adaptive_{split}_posterior.npy', '--output', STUDY / f'{name}.json']
    if pair:
        arguments += ['--gate2', pair, '--gate2-weight', '1']
    run('orbit.evaluate_audio_gate', *arguments)


def gate_train(masks, output, paired=False):
    run('orbit.train_audio_gate_pairs' if paired else 'orbit.train_audio_gate',
        '--cache', CACHE, '--masks', masks, '--selector', SELECTOR, '--apex', APEX,
        '--output', output, '--epochs', 12 if paired else 16, '--seed', 1337,
        '--selection-metric', 'deployable')


def raw_check(name, masks, gate, pair):
    run('orbit.raw_check', '--masks', masks, '--selector', SELECTOR, '--apex', APEX,
        '--geometry', ORBIT / 'cache/utonia-xyz-v1',
        '--projection', ORBIT / 'cache/utonia-xyz-v1/projection.pt',
        '--spatial', ORBIT / 'cache/spatial-v1/adaptive.pt',
        '--posterior', POST / 'adaptive_val_posterior.npy',
        '--gate', gate, '--gate2', pair, '--gate2-weight', '1',
        '--output', STUDY / f'{name}.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('masks', 'frozen-standard', 'frozen-paired',
                                         'baseline', 'new-standard', 'new-paired', 'evaluate'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU campaign must run inside a Slurm allocation')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Slurm allocation has no usable CUDA device')
    STUDY.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.stage == 'masks':
        run('consistency_cache_audit.py', '--cache', CACHE,
            '--output', STUDY / 'corrected-full-cache-audit.json', script=True)
        run('orbit.train_masks', '--cache', CACHE, '--output', MASK.parent,
            '--epochs', 24, '--seed', 1337, '--prompts', 8,
            '--deployment-selector', SELECTOR, '--deployment-apex', APEX,
            '--deployment-mask-threshold', '.8')
    elif args.stage == 'frozen-standard':
        gate_train(FROZEN_MASK, FGATE.parent)
        evaluate('frozen-mask-corrected-standard-val', FROZEN_MASK, FGATE)
    elif args.stage == 'frozen-paired':
        gate_train(FROZEN_MASK, FPAIR.parent, paired=True)
        evaluate('frozen-mask-corrected-paired-val', FROZEN_MASK, FPAIR)
    elif args.stage == 'baseline':
        evaluate('frozen-weights-current-evaluator-val', FROZEN_MASK, FROZEN_GATE, FROZEN_PAIR)
        evaluate('frozen-weights-neutral-spatial-val', FROZEN_MASK, FROZEN_GATE, FROZEN_PAIR,
                 control='uniform-spatial')
        evaluate('frozen-weights-current-evaluator-test', FROZEN_MASK, FROZEN_GATE, FROZEN_PAIR, split='test')
        # Raw equivalence failure is evidence to retain, not a reason to delete
        # a report or stop the independent controlled training jobs.
        try:
            raw_check('frozen-raw-cached-equivalence', FROZEN_MASK, FROZEN_GATE, FROZEN_PAIR)
        except subprocess.CalledProcessError as error:
            (STUDY / 'frozen-raw-check-failure.json').write_text(json.dumps({'exit_code': error.returncode}))
    elif args.stage == 'new-standard':
        gate_train(MASK, GATE.parent)
    elif args.stage == 'new-paired':
        gate_train(MASK, PAIR.parent, paired=True)
    else:
        evaluate('corrected-mask-frozen-gates-val', MASK, FROZEN_GATE, FROZEN_PAIR)
        evaluate('corrected-mask-standard-val', MASK, GATE)
        evaluate('corrected-mask-paired-val', MASK, PAIR)
        evaluate('corrected-mask-ensemble-val', MASK, GATE, PAIR)
        evaluate('corrected-mask-neutral-spatial-val', MASK, GATE, PAIR, control='uniform-spatial')
        # Predeclared ensemble and unit weights: no TEST-based choice or tuning.
        evaluate('corrected-mask-ensemble-test', MASK, GATE, PAIR, split='test')
        raw_check('corrected-raw-cached-equivalence', MASK, GATE, PAIR)
    (STUDY / f'{args.stage}-COMPLETE.json').write_text(json.dumps(dict(
        stage=args.stage, job_id=os.environ['SLURM_JOB_ID'], elapsed_seconds=time.time()-started,
        device=torch.cuda.get_device_name(), source_sha256=D.sha(Path(__file__)),
        comparison='fixed-backbone-selector-apex-spatial; corrected-mask-and-gate-contracts',
        causal_isolation_of_each_individual_fix=False), indent=2))


if __name__ == '__main__':main()
