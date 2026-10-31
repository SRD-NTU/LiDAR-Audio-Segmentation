"""Freeze a separate sequential rollout of the revised two-view model."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.provenance import make_manifest,verify_manifest

p=argparse.ArgumentParser()
p.add_argument('--revision',default='neo1')
p.add_argument('--parent-revision',default='auditfix1')
p.add_argument('--results-root',default='')
args=p.parse_args()
source=json.loads((ROOT/'config'/('frozen_evaluation_'+args.parent_revision+'.json')).read_text())
if source.get('revision') != args.parent_revision:
 raise RuntimeError('Parent two-view revision does not match the requested revision')
verify_manifest(source,check_dependencies=True,learning_python=sys.executable,
                renderer_python=UTILITY/'environments/audiolidar/bin/python')
memory_filter=ROOT/'checkpoints'/args.revision/'memory_filter.pt'
if not (ROOT/'checkpoints'/args.revision/'memory_filter_complete.json').exists() or not memory_filter.exists():
 raise RuntimeError('Neo recurrent-memory training is incomplete')
protocol=dict(version=3,experiment='recurrent-acoustic-memory-rollout-of-'+args.parent_revision,revision=args.revision,
 parent_protocol_sha256=source['protocol_sha256'],world=source['world'],evidence=source['evidence'],
 memory_filter=str(memory_filter.relative_to(ROOT)),
 checkpoints={**source['checkpoints'],str(memory_filter.relative_to(ROOT)):sha(memory_filter)},
 singleview_checkpoints=source['singleview_checkpoints'],
 search=source['search'],methods=['world','frozen','random-0','random-1','random-2'],
 report_views=[3,4,5],max_views=5,anchor_radius_m=3.,max_leg_distance_m=3.,
 minimum_position_separation_m=.15,receiver_height_m=1.2,seed=1337,
 results_root=args.results_root or 'data/multiview_'+args.revision,renderer_server='scripts/renderer_server_posefix.py',
 renderer_sha256=sha(ROOT/'scripts/renderer_server_posefix.py'),
 fusion='Joint object/site posterior with measured-band likelihood; prior counted once.',
 memory_update='trained_gate_on_frozen_singleview_gru_at_acquired_pose_v1',
 planning='Each acquired recording updates the frozen SingleView latent through a TRAIN-supervised, VAL-selected gate at the committed pose; future forecasts use that latent and the current site posterior.',
 limitations=['The global memory gate is supervised using the nearest source site of the best-IoU mask on TRAIN; this is an approximate oracle component.',
              'Conditional independence of recordings is approximate; by-view calibration remains unmeasured.',
              'Imagined outcomes do not predict a future reliability mask.'],
 provenance=make_manifest(PROJECT/'Acoustic-WM/.venv/bin/python',UTILITY/'environments/audiolidar/bin/python'))
protocol['protocol_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
path=ROOT/'config'/('frozen_multiview_'+args.revision+'.json')
if path.exists():
 if json.loads(path.read_text())!=protocol:raise RuntimeError('Existing frozen multiview revision differs')
 print('Already frozen:',path);sys.exit(0)
for name,digest in protocol['checkpoints'].items():
 if sha(ROOT/name)!=digest:raise RuntimeError('Checkpoint changed: '+name)
if fingerprints()!=protocol['singleview_checkpoints']:raise RuntimeError('SingleView changed')
write_json(path,protocol)
print(json.dumps(dict(path=str(path),protocol_sha256=protocol['protocol_sha256'])))
