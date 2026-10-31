#!/usr/bin/env python3
"""Plan from one observation; complete only when the selected real audio arrives."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from activeview_v3.runtime import *
from activeview_v3.inference import Adapter,plan
from activeview_v3.navigation import relative_pose
from activeview_v3.simulator_client import Simulator
from activeview_v3.provenance import verify_manifest
from acoustic_wm.audio.predictive_features import covariance48_from_wave
import argparse,numpy as np,soundfile as sf

def wave(path):
 data,rate=sf.read(path,dtype='float32',always_2d=True)
 if rate!=44100 or data.shape[1]!=4 or not np.isfinite(data).all():raise ValueError('Expected finite 44100 Hz four-channel WAV')
 return data

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',default='config/frozen_evaluation_auditfix1.json')
 sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('plan');q.add_argument('--prepared');q.add_argument('--lidar');q.add_argument('--audio')
 q.add_argument('--scene',required=True);q.add_argument('--listener',required=True,help='JSON with position_world, floor_position_world, yaw_rad (Habitat world frame)')
 q.add_argument('--method',choices=['world','frozen','direct','random'],default='world');q.add_argument('--seed',type=int,default=1337)
 q.add_argument('--output',required=True,help='New plan directory inside v3')
 q=sub.add_parser('complete');q.add_argument('--plan',required=True);q.add_argument('--audio',required=True);q.add_argument('--output',required=True)
 args=p.parse_args();seed_all(1337)
 protocol=json.loads((ROOT/args.protocol).read_text())
 if 'provenance' not in protocol:raise RuntimeError('Use archived inference code for the legacy protocol')
 verify_manifest(protocol)
 for path,digest in protocol['checkpoints'].items():
  if sha(ROOT/path)!=digest:raise RuntimeError('Changed checkpoint: '+path)
 if fingerprints()!=protocol['singleview_checkpoints']:raise RuntimeError('Changed SingleView checkpoint')
 adapter=Adapter(ROOT/protocol['world'],direct_path=ROOT/protocol['direct'],evidence_path=ROOT/protocol['evidence'])
 if args.command=='plan':
  destination=output_path(args.output)
  if destination.exists():raise FileExistsError(destination)
  if args.prepared:
   if args.lidar or args.audio:p.error('Use --prepared or --lidar with --audio')
   with np.load(args.prepared,allow_pickle=False) as z:row={k:z[k] for k in z.files}
  else:
   if not args.lidar or not args.audio:p.error('Supply --prepared or both --lidar and --audio')
   from acoustic_wm.frontend import RawObservationEncoder
   encoder=RawObservationEncoder(json.loads((BASE/'config.json').read_text()))
   with np.load(args.lidar,allow_pickle=False) as z:xyz=z['xyz'].astype('float32')
   audio=wave(args.audio);row=encoder.encode_arrays(xyz,audio,44100)
   row['observed'],row['reliability']=covariance48_from_wave(audio)
  listener=json.loads(Path(args.listener).read_text())
  listener={k:listener[k] for k in ['position_world','floor_position_world','yaw_rad']}
  sim=Simulator(server=protocol['renderer_server'])
  try:
   sim.geometry(args.scene,listener)
   result=plan(adapter,row,listener,sim,args.method,args.seed,**protocol['search'])
  finally:sim.close()
  destination.mkdir()
  reference=destination/'reference.npz';np.savez_compressed(reference,**result['prepared']['row'])
  record={k:v for k,v in result.items() if k not in ['batch','prepared']}
  record.update(listener=listener,scene=args.scene,method=args.method,reference_sha256=sha(reference),
   protocol_sha256=protocol['protocol_sha256'],protocol_path=args.protocol,
   provenance={k:protocol['provenance'][k] for k in
               ['source_manifest_sha256','config_manifest_sha256','dependency_manifest_sha256']},recordings_consumed=1)
  write_json(destination/'plan.json',record)
  print(json.dumps(dict(plan=str(destination/'plan.json'),next_receiver_pose=result['pose'],expected_utility=result['utility'])))
 else:
  folder=Path(args.plan);record=json.loads((folder/'plan.json').read_text())
  if record['protocol_sha256']!=protocol['protocol_sha256'] or sha(folder/'reference.npz')!=record['reference_sha256']:raise RuntimeError('Plan inputs changed')
  destination=output_path(args.output)
  if destination.exists():raise FileExistsError(destination)
  with np.load(folder/'reference.npz',allow_pickle=False) as z:row={k:z[k] for k in z.files}
  listener=record['listener'];pose=record['pose']
  action=relative_pose(np.asarray(listener['position_world']),listener['yaw_rad'],pose['position'],pose['yaw'])
  batch,prepared=adapter.prepare(row,action[None],[pose['path_distance_m']])
  observed,weight=covariance48_from_wave(wave(args.audio))
  result=adapter.update(batch,prepared,0,observed,weight)
  np.savez_compressed(destination,mask=result['mask'],scores=result['scores'],candidate=result['candidate'],audio_observations=2,lidar_scans=1)
  print(json.dumps(dict(output=str(destination),original_points=len(result['mask']),foreground_points=int(result['mask'].sum()),candidate=result['candidate'])))

if __name__=='__main__':main()
