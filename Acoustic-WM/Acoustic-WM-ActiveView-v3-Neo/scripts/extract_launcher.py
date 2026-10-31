"""Durable Slurm client for extraction on the existing rendering GPU allocations.

This process does no GPU computation itself. If a rendering allocation finishes,
move extraction to another live allocation. The training job handles any final
remainder after all rendering allocations have exited.
"""
from pathlib import Path
import sys,subprocess,shlex,time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import ROOT,PROJECT,samples,write_json

jobs=['231375','231424','231425','231376']
rows=samples('train')+samples('val')
def complete():
 return all((ROOT/'data/features'/(row+'.npz')).exists() for row in rows)
def running():
 query=subprocess.run(['squeue','-h','-j',','.join(jobs),'-o','%i %T'],capture_output=True,text=True)
 return [line.split()[0] for line in query.stdout.splitlines() if line.split()[-1]=='RUNNING']

while not complete():
 available=running()
 if not available:
  write_json('reports/persistent_extraction.json',dict(state='remaining_extraction_deferred_to_training',
   reason='No rendering allocations remain; train.sbatch extracts remaining files before optimization'))
  sys.exit(0)
 job=next(j for j in jobs if j in available)
 write_json('reports/persistent_extraction.json',dict(state='extracting',gpu_allocation=job))
 python=PROJECT/'Acoustic-WM/.venv/bin/python'
 # One extractor uses one allocated device. Restrict visibility before importing
 # torch so NVML/lazy initialization cannot enumerate a second inaccessible GPU.
 command='cd '+shlex.quote(str(ROOT))+' && : "${CUDA_VISIBLE_DEVICES:?Slurm GPU visibility is required}" && export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" && exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 '+shlex.quote(str(python))+' -B -u scripts/extract.py --split all --workers 1 --watch'
 print('Starting extraction inside GPU allocation '+job,flush=True)
 result=subprocess.run(['srun','--jobid='+job,'--overlap','--ntasks=1','--cpus-per-task=2','bash','-c',command])
 if result.returncode and job in running():
  time.sleep(3)
  if job in running():
   write_json('reports/persistent_extraction.json',dict(state='failed',gpu_allocation=job,returncode=result.returncode))
   raise RuntimeError('Extraction failed in a live allocation; inspect this job stderr')
write_json('reports/persistent_extraction.json',dict(state='complete',references=len(rows)))
