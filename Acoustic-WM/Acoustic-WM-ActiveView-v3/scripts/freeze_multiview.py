"""Freeze a separate sequential rollout of the revised two-view model."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.provenance import make_manifest

p=argparse.ArgumentParser();p.add_argument('--revision',default='auditfix1')
p.add_argument('--results-root',default='');args=p.parse_args()
source=json.loads((ROOT/'config'/('frozen_evaluation_'+args.revision+'.json')).read_text())
protocol=dict(version=2,experiment='sequential-rollout-of-'+args.revision,revision=args.revision,
 parent_protocol_sha256=source['protocol_sha256'],world=source['world'],evidence=source['evidence'],
 checkpoints=source['checkpoints'],singleview_checkpoints=source['singleview_checkpoints'],
 search=source['search'],methods=['world','frozen','random-0','random-1','random-2'],
 report_views=[3,4,5],max_views=5,anchor_radius_m=3.,max_leg_distance_m=3.,
 minimum_position_separation_m=.15,receiver_height_m=1.2,seed=1337,
 results_root=args.results_root or 'data/multiview_'+args.revision,renderer_server='scripts/renderer_server_posefix.py',
 renderer_sha256=sha(ROOT/'scripts/renderer_server_posefix.py'),
 fusion='Joint object/site posterior with measured-band likelihood; prior counted once.',
 planning='Frozen reference memory, reweighted context and sequential belief updates.',
 limitations=['No sequence retraining; conditional independence of recordings is approximate.',
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
