"""Shared durable tasks; persistent GPU workers with OS locks and explicit errors."""
from pathlib import Path
import sys,os,subprocess,time,json,fcntl,traceback,argparse
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'src'))
from common import write_json,PYTHON,PROJECT
SEEDS=[1337,2027,3407];METHODS=['world','analytic_direct','repeat_current','no_assimilation','no_transition','standard']

def claim(name,command,env=None):
 root=BASE/'runs/tasks';root.mkdir(parents=True,exist_ok=True);done=root/(name+'.done.json');error=root/(name+'.error.json')
 if done.exists():return False
 with (root/(name+'.lock')).open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  if done.exists():return False
  start=time.time();write_json(root/(name+'.running.json'),dict(pid=os.getpid(),started=start,job=os.environ.get('SLURM_JOB_ID'),command=command))
  print('START',name,flush=True)
  logfile=BASE/'logs'/(name+'.log')
  with logfile.open('a') as out:
   proc=subprocess.run(command,cwd=BASE,env=env,stdout=out,stderr=subprocess.STDOUT)
  if proc.returncode:
   write_json(error,dict(command=command,exitcode=proc.returncode,log=str(logfile),time=time.time()));raise RuntimeError(f'{name} failed; see {logfile}')
  write_json(done,dict(command=command,seconds=time.time()-start,job=os.environ.get('SLURM_JOB_ID')));error.unlink(missing_ok=True)
  print('DONE',name,flush=True);return True

def extracted(scope):
 root=BASE if scope=='historical' else BASE/'scopes/common'
 for s in ['train','val','test']:
  manifest=root/'manifests'/f'{s}.json'
  if not manifest.exists():return False
  for r in json.loads(manifest.read_text()):
   p=root/'cache/controlled'/(r+'.npz')
   if not p.exists():return False
  # The pre-campaign pilot cache has schema 1 and must be regenerated.
  pilot=root/'cache/controlled/test/00025-ixTj1aTMup2/18/00000.npz'
  if pilot.exists():
   import numpy as np
   with np.load(pilot) as z:
    if 'schema' not in z:return False
 return True

def run(rank,skip_generate=False):
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU work requires Slurm')
 for _ in range(600):
  if (BASE/'configs/READY').exists():break
  time.sleep(1)
 else:raise RuntimeError('Source freeze not ready')
 # Each hardware family is checked, and every worker obeys the same frozen source.
 freeze=json.loads((BASE/'manifests/campaign_source.json').read_text())
 import hashlib
 for name,digest in freeze.items():
  if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed after freeze: '+name)
 for name,value in json.loads((BASE/'manifests/external_dependencies.json').read_text()).items():
  p=Path(name)
  if not p.exists() or p.stat().st_size!=value['bytes'] or p.stat().st_mtime_ns!=value['mtime_ns']:raise RuntimeError('External dependency changed: '+name)
 for seed in SEEDS:claim(f'fusion-{seed}',[str(PYTHON),'-B','-u','src/fusion_controls.py','--seed',str(seed)])
 idle=0
 while True:
  work=False
  for scope in ['historical','common']:
   if scope=='common' and not (BASE/'generated/common/COMPLETE.json').exists():continue
   env=dict(os.environ,EXPERIMENT_SCOPE=scope)
   if not extracted(scope):
    subprocess.run([str(PYTHON),'-B','-u','src/extract.py','--rank',str(rank)],cwd=BASE,env=env,check=True);work=True
   if extracted(scope):
    for seed in SEEDS:
     for method in METHODS:work=claim(f'{scope}-{method}-{seed}',[str(PYTHON),'-B','-u','src/train_controls.py','--method',method,'--seed',str(seed)],env) or work
    work=claim(f'{scope}-prediction',[str(PYTHON),'-B','-u','src/evaluate_world.py'],env) or work
    work=claim(f'{scope}-frozen',[str(PYTHON),'-B','-u','src/evaluate_frozen.py'],env) or work
    # Reproduce source-supervised training without replacing the selected method.
    work=claim(f'{scope}-world-reproduction-1337',[str(PYTHON),'-B','-u','src/retrain_world.py','--seed','1337','--epochs','120'],env) or work
    expected=[BASE/'runs/tasks'/f'{scope}-{m}-{s}.done.json' for m in METHODS for s in SEEDS]+[BASE/'runs/tasks'/f'{scope}-frozen.done.json',BASE/'runs/tasks'/f'{scope}-prediction.done.json',BASE/'runs/tasks'/f'{scope}-world-reproduction-1337.done.json']
    if all(p.exists() for p in expected):work=claim(f'{scope}-report',[str(PYTHON),'-B','-u','src/report.py'],env) or work
  if not skip_generate and not (BASE/'generated/common/COMPLETE.json').exists():
   prefix=PROJECT/'.redun/dataset-utility/utility/environments/audiolidar';env=dict(os.environ)
   env.update(PYTHONPATH=str(BASE/'snapshots/generator'),LD_LIBRARY_PATH=str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',EXPERIMENT_SCOPE='historical')
   subprocess.run([str(prefix/'bin/python'),'-B','-u','scripts/generate_common_menu.py','--rank',str(rank)],cwd=BASE,env=env,check=True);work=True
  required=[BASE/'runs/tasks'/f'{s}-report.done.json' for s in (['historical'] if skip_generate else ['historical','common'])]
  if all(p.exists() for p in required):return
  idle=0 if work else idle+1
  if idle>120:raise RuntimeError('No progress for one hour; incomplete shared tasks retained for restart')
  time.sleep(30)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--skip-generate',action='store_true');a=p.parse_args();run(a.rank,a.skip_generate)
