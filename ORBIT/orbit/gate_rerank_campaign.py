"""Fit a residual ranker to imperfect TRAIN masks with audio-query evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    masks = root / 'runs/masks-utonia-spatial-1337/best.pt'
    selector = root / 'runs/selector-utonia-spatial-1337/best.pt'
    apex = root / 'runs/apex-utonia-spatial-1337/best.pt'
    apex_complete = root / 'runs/apex-utonia-spatial-1337/COMPLETE.json'
    gate = root / 'runs/audio-gate-pairs-utonia-1337-r2/best.pt'
    geometry = root / 'cache/utonia-xyz-v1'
    spatial = root / 'cache/spatial-v1'
    bank = root / 'cache/candidates-utonia-pair-gate-v1'
    reranker = root / 'runs/reranker-utonia-pair-gate-1337'
    report = root / 'reports/utonia_pair_gate_reranker_val.json'
    journal = root / 'reports/campaign-pair-gate-reranker.json'
    if journal.exists() or report.exists() or (bank / 'COMPLETE.json').exists():
        raise FileExistsError(journal)
    for path in (masks, selector, apex, apex_complete, gate,
                 geometry / 'COMPLETE.json', spatial / 'COMPLETE.json'):
        if not path.exists():
            raise FileNotFoundError(path)
    common = ['--masks', str(masks), '--selector', str(selector), '--apex', str(apex),
              '--gate', str(gate), '--geometry', str(geometry)]
    commands = [
        ['-m', 'orbit.candidate_bank', *common, '--spatial', str(spatial),
         '--output', str(bank), '--fit-reranker', str(reranker)],
        ['-m', 'orbit.evaluate_audio_gate', *common, '--spatial', str(spatial),
         '--reranker', str(reranker / 'best.pt'), '--output', str(report)],
    ]
    for control in ('shuffled', 'uniform-spatial'):
        commands.append(['-m', 'orbit.evaluate_audio_gate', *common, '--spatial', str(spatial),
                         '--reranker', str(reranker / 'best.pt'), '--control', control,
                         '--output', str(root / f'reports/utonia_pair_gate_{control}_val.json')])
    state = dict(test_used=False, status='running', stages=[])
    for command in commands:
        stage = dict(command=[sys.executable, '-u', *command], status='running',
                     started=time.time())
        state['stages'].append(stage)
        journal.write_text(json.dumps(state, indent=2))
        print('START', stage['command'], flush=True)
        try:
            subprocess.run(stage['command'], check=True)
        except BaseException:
            stage['status'] = 'failed'
            state['status'] = 'failed'
            raise
        else:
            stage['status'] = 'complete'
        finally:
            stage['elapsed_seconds'] = time.time() - stage['started']
            journal.write_text(json.dumps(state, indent=2))
    state['status'] = 'complete'
    journal.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
