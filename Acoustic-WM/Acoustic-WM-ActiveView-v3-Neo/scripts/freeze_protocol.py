"""Select on VAL and seal a new evaluation revision without overwriting v3."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.provenance import make_manifest

p=argparse.ArgumentParser();p.add_argument('--revision',default='auditfix1')
p.add_argument('--results-root',default='');args=p.parse_args()
if not args.revision or '/' in args.revision or '..' in args.revision:raise ValueError('Use a simple nonempty revision name')
path=ROOT/'config'/('frozen_evaluation_'+args.revision+'.json')
if path.exists():
 print('Protocol already frozen:',path);sys.exit(0)
folder=ROOT/'checkpoints'/args.revision
worlds=[]
for seed in [1337,2027]:
 child=folder/('world-s%d'%seed)
 result=json.loads((child/'complete.json').read_text());worlds.append((result['validation_iou'],child/'world.pt'))
_,world=max(worlds,key=lambda x:x[0])
for required in ['evidence_complete.json','direct_complete.json']:
 if not (folder/required).exists():raise RuntimeError('Training incomplete: '+required)
evidence=folder/'evidence.pt';direct=folder/'direct.pt'
checkpoints=[evidence,ROOT/'checkpoints/embedding.npz',direct,world]
provenance=make_manifest(PROJECT/'Acoustic-WM/.venv/bin/python',UTILITY/'environments/audiolidar/bin/python')
selection=dict(version='ActiveView-v3-'+args.revision,revision=args.revision,
 world=str(world.relative_to(ROOT)),evidence=str(evidence.relative_to(ROOT)),direct=str(direct.relative_to(ROOT)),
 checkpoints={str(q.relative_to(ROOT)):sha(q) for q in checkpoints},singleview_checkpoints=fingerprints(),
 selection_split='val',test_used_for_selection=False,
 search=dict(initial=32,rounds=2,population=16,elite=6,draws=16),budget_m=3.,height_m=1.2,seed=1337,
 renderer_server='scripts/renderer_server_posefix.py',
 renderer_sha256=sha(ROOT/'scripts/renderer_server_posefix.py'),
 results_root=args.results_root or 'data/evaluation_'+args.revision,
 methods=['world','frozen','direct','random-0','random-1','random-2'],reference_lidar_scans=1,
 recordings_per_policy_episode=2,
 world_selection='Highest mean VAL IoU over eight newly rendered feasible continuous poses per reference',
 evidence_likelihood='Measured 48-band reliability projected into 128-D acoustic evidence; simulated outcomes assume all bands active',
 direct_baseline='Full-information site encoder with matching pre-acquisition fields',provenance=provenance,
 limitations=['Historical source-conditioned anchors and exposed TEST scenes.',
              'Masked likelihood uses diagonal projected covariance approximation.',
              'Imagined outcomes do not predict a future reliability mask.'])
selection['protocol_sha256']=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
write_json(path,selection)
print(json.dumps(dict(path=str(path),protocol_sha256=selection['protocol_sha256'],
                      source_manifest_sha256=provenance['source_manifest_sha256'],
                      dependency_manifest_sha256=provenance['dependency_manifest_sha256']),indent=2))
