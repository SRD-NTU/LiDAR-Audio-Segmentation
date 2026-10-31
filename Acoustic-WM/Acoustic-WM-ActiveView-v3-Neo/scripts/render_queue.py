"""Resumable scene work queue with exclusive locks across Slurm allocations."""
from render_training import *
import fcntl
from types import SimpleNamespace

p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);args=p.parse_args()
queue=[]
for split in (['val','train'] if args.worker>=6 else ['train','val']):
 scenes=sorted({r.split('/')[1] for r in samples(split)})
 for scene in scenes:queue.append((split,scene))
for split,scene in queue:
 lockpath=output_path('runtime/locks/'+scene+'.lock')
 with open(lockpath,'a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:continue
  run(SimpleNamespace(split=split,rank=args.worker,workers=1,poses=8,scenes=[scene]))
write_json('reports/queue-worker-%d-complete.json'%args.worker,dict(complete=True,worker=args.worker))
