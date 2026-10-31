"""Committed continuous acquisition with durable, verified restart semantics."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.inference import Adapter,plan
from activeview_v3.navigation import relative_pose
from activeview_v3.simulator_client import Simulator
from activeview_v3.provenance import verify_manifest
import argparse,numpy as np,time

def run(args):
 protocol=json.loads((ROOT/args.protocol).read_text())
 if 'provenance' not in protocol:
  raise RuntimeError('Legacy frozen protocol has no code identity; use its archived evaluator, not revised code')
 provenance=verify_manifest(protocol,check_dependencies=True,
  learning_python=sys.executable,renderer_python=UTILITY/'environments/audiolidar/bin/python')
 if sha(ROOT/protocol['renderer_server'])!=protocol['renderer_sha256']:
  raise RuntimeError('Frozen renderer changed')
 for path,digest in protocol['checkpoints'].items():
  if sha(ROOT/path)!=digest:raise RuntimeError('Frozen evaluation checkpoint changed: '+path)
 if fingerprints()!=protocol['singleview_checkpoints']:raise RuntimeError('SingleView checkpoint changed')
 adapter=Adapter(ROOT/protocol['world'],direct_path=ROOT/protocol['direct'],evidence_path=ROOT/protocol['evidence'])
 rows=samples(args.split);scenes=sorted({r.split('/')[1] for r in rows})[args.rank::args.workers]
 rows=[r for r in rows if r.split('/')[1] in scenes]
 if args.split=='test' and (args.limit or args.methods):raise ValueError('TEST uses all methods and references')
 if args.limit:rows=rows[:args.limit]
 methods=protocol['methods'] if not args.methods else args.methods.split(',')
 if any(m not in protocol['methods'] for m in methods):raise ValueError('Unregistered method')
 simulator=Simulator('logs/eval-%s-renderer-%s-%d.log'%(protocol['revision'],args.split,args.rank),
  server=protocol['renderer_server'])
 start=time.time();completed=0
 try:
  for sample in rows:
   resultdir=protocol['results_root']+'/'+sample
   remaining=[]
   for method in methods:
    path=ROOT/resultdir/(method+'.json')
    if path.exists():
     previous=json.loads(path.read_text())
     if previous['protocol_sha256']!=protocol['protocol_sha256']:raise RuntimeError('Stale result: '+str(path))
    else:remaining.append(method)
   if not remaining:continue
   reference_path=ARCHIVE/'cache/reference'/(sample+'.npz')
   reference_sha256=sha(reference_path)
   with np.load(reference_path) as z:row={k:z[k] for k in z.files}
   listener=simulator.reference(sample)
   seed_key=json.dumps(dict(listener=listener,scene=sample.split('/')[1],seed=protocol['seed']),sort_keys=True)
   seed=int(hashlib.sha256(seed_key.encode()).hexdigest()[:8],16)
   for method in remaining:
    name='random' if method.startswith('random-') else method
    trial=int(method.split('-')[1]) if name=='random' else 0
    commit_path=ROOT/resultdir/(method+'-commit.json')
    audio_path=ROOT/resultdir/(method+'-audio.npz')
    receipt_path=ROOT/resultdir/(method+'-receipt.json')
    if commit_path.exists():
     commitment=json.loads(commit_path.read_text())
     if (commitment['protocol_sha256']!=protocol['protocol_sha256'] or commitment['method']!=method or
         commitment['sample']!=sample or commitment['listener']!=listener or
         commitment['reference_sha256']!=reference_sha256 or commitment['provenance']!=provenance):
      raise RuntimeError('Stale commitment: '+str(commit_path))
     pose=commitment['pose']
     action=relative_pose(np.asarray(listener['position_world']),listener['yaw_rad'],pose['position'],pose['yaw'])
     batch,prepared=adapter.prepare(row,action[None],[pose['path_distance_m']])
    else:
     choice=plan(adapter,row,listener,simulator,name,seed=(seed+trial)%(2**32),**protocol['search'])
     batch,prepared=choice['batch'],choice['prepared']
     commitment={k:v for k,v in choice.items() if k not in ['batch','prepared']}
     commitment.update(method=method,sample=sample,listener=listener,reference_sha256=reference_sha256,
                       seed=(seed+trial)%(2**32),protocol_sha256=protocol['protocol_sha256'],provenance=provenance)
     write_json(commit_path,commitment)
     pose=commitment['pose']
    if receipt_path.exists():
     receipt=json.loads(receipt_path.read_text())
     if (receipt['protocol_sha256']!=protocol['protocol_sha256'] or receipt['commit_sha256']!=sha(commit_path)
         or receipt['observation_sha256']!=sha(audio_path) or receipt['provenance']!=provenance
         or receipt['renderer_sha256']!=protocol['renderer_sha256']):
      raise RuntimeError('Acquisition receipt mismatch: '+str(receipt_path))
    else:
     if audio_path.exists():raise RuntimeError('Unreceipted observation; preserve for manual recovery: '+str(audio_path))
     observation=simulator.acquire(pose,str(audio_path.relative_to(ROOT)),save_audio=completed<2)
     receipt=dict(observation,protocol_sha256=protocol['protocol_sha256'],commit_sha256=sha(commit_path),
                  observation_sha256=sha(audio_path),provenance=provenance,renderer_server=protocol['renderer_server'])
     receipt['renderer_sha256']=protocol['renderer_sha256']
     write_json(receipt_path,receipt)
    with np.load(audio_path) as z:observed=z['observed'];weight=z['weight']
    prediction=adapter.update(batch,prepared,0,observed,weight)
    # Ground-truth IoU enters only after action, acquisition, and prediction.
    with np.load(ARCHIVE/'cache/controlled'/(sample+'.npz')) as z:iou=z['iou'];null_iou=float(z['null_iou'])
    chosen=prediction['candidate'];quality=null_iou if chosen<0 else float(iou[chosen])
    state=prepared['state'];baseline=null_iou if state['silent'] else float(iou[int(state['base'].argmax())])
    path=output_path(resultdir+'/'+method+'-prediction.npz')
    if path.exists():
     with np.load(path) as z:
      np.testing.assert_array_equal(z['mask'],prediction['mask'])
      np.testing.assert_allclose(z['scores'],prediction['scores'],rtol=1e-5,atol=1e-5)
    else:
     temporary=path.with_suffix('.partial.npz')
     np.savez_compressed(temporary,mask=prediction['mask'],scores=prediction['scores']);temporary.replace(path)
    write_json(resultdir+'/'+method+'.json',dict(sample=sample,method=method,iou=quality,singleview_iou=baseline,
      candidate=chosen,predicted_utility=commitment['utility'],evaluated_poses=commitment['evaluated_poses'],pose=pose,
      rir_sha256=receipt['rir_sha256'],nonzero_channels=receipt['nonzero_channels'],
      observation_sha256=receipt['observation_sha256'],receipt_sha256=sha(receipt_path),commit_sha256=sha(commit_path),
      prediction_sha256=sha(path),original_point_count=len(prediction['mask']),recordings=2,lidar_scans=1,
      protocol_sha256=protocol['protocol_sha256'],provenance=provenance,
      renderer_sha256=protocol['renderer_sha256']))
   completed+=1
   if completed==1 or completed%10==0:
    status=dict(split=args.split,worker=args.rank,completed=completed,total=len(rows),seconds=time.time()-start,
                protocol_sha256=protocol['protocol_sha256'])
    print(json.dumps(status),flush=True)
    write_json('reports/'+protocol['revision']+'/eval-%s-%d-progress.json'%(args.split,args.rank),status)
 finally:simulator.close()
 write_json('reports/'+protocol['revision']+'/eval-%s-%d-complete.json'%(args.split,args.rank),
  dict(complete=True,rows=len(rows),methods=methods,protocol_sha256=protocol['protocol_sha256'],provenance=provenance))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='test')
 p.add_argument('--protocol',default='config/frozen_evaluation_auditfix1.json')
 p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=1)
 p.add_argument('--limit',type=int,default=0);p.add_argument('--methods',default='')
 args=p.parse_args();seed_all(1337);run(args)
