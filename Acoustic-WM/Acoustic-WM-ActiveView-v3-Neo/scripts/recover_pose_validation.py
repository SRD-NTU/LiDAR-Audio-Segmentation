"""Audited acquisition-only repair around the unchanged frozen evaluators."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.multiview_simulator import RouteSimulator
import subprocess
import argparse


class RecoveredSimulator(RouteSimulator):
    def __init__(self, log):
        self.log = open(output_path(log), 'a', buffering=1)
        self.process = subprocess.Popen(
            [str(UTILITY/'environments/audiolidar/bin/python'), '-B', '-u', str(ROOT/'scripts/renderer_server_posefix.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            env=dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OPENBLAS_NUM_THREADS='1'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--two-view', action='store_true')
    args = p.parse_args()
    manifest = json.loads((ROOT/'config/pose_validation_repair.json').read_text())
    for name, digest in manifest['code_sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Acquisition repair changed: '+name)
    seed_all(1337)
    if args.two_view:
        import evaluate as evaluation
        evaluation.Simulator = RecoveredSimulator
        args.workers = 1
    else:
        import evaluate_multiview as evaluation
        evaluation.RouteSimulator = RecoveredSimulator
    args.split = 'test'
    args.limit = 0
    args.methods = ''
    args.stop_file = ''
    print(json.dumps(dict(acquisition_repair=manifest['revision'], job_id=os.getenv('SLURM_JOB_ID'), rank=args.rank)), flush=True)
    evaluation.run(args)
    if args.two_view:
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/report.py'), '--split', 'test'], check=True)
