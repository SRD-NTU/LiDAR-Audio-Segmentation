"""TRAIN-only end-to-end evaluator recovery check, including artifact hashes."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
import subprocess

command = [sys.executable, '-B', '-u', str(ROOT/'scripts/evaluate_multiview.py'),
           '--split', 'train', '--limit', '1', '--methods', 'random-0']
subprocess.run(command, check=True)
sample = sorted(samples('train'))[0]
directory = ROOT/'data/multiview'/sample
before = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
# Simulate an interruption after the fourth saved acquisition but before the
# episode completion markers. Every observation/commitment/prediction stays intact.
(directory/'complete.json').unlink()
(directory/'random-0/complete.json').unlink()
subprocess.run(command, check=True)
after = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
if before != after:
    raise RuntimeError('Resume altered durable inference artifacts')
write_json('reports/multiview-resume-smoke.json',
           dict(passed=True, sample=sample, files_unchanged=len(before),
                resumed_views=[2, 3, 4, 5], protocol_sha256=json.loads((ROOT/'config/frozen_multiview.json').read_text())['protocol_sha256']))
print('Evaluator resume passed; all %d artifacts unchanged.' % len(before), flush=True)
