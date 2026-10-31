"""Lightweight live status; no model loading or GPU computation."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parents[1]
print(subprocess.run(['squeue','-u','soumyara004','-o','%i|%j|%T|%M|%R'],capture_output=True,text=True,check=True).stdout)
status={}
for split,total in [('train',3264),('val',768)]:
 rendered=len(list((root/'data/rendered'/split).rglob('*.json')))
 features=len([p for p in (root/'data/features'/split).rglob('*.npz') if not p.name.endswith('.partial.npz')])
 status[split]=dict(rendered_references=rendered,total_references=total,recordings=8*rendered,extracted_features=features)
for name in ['evidence','direct']:
 path=root/'checkpoints'/(name+'_history.json')
 if path.exists():status[name+'_latest']=json.loads(path.read_text())[-1]
for seed in [1337,2027]:
 path=root/'checkpoints'/('world-s%d'%seed)/'history.json'
 if path.exists():status['world_s%d_latest'%seed]=json.loads(path.read_text())[-1]
status['test_report_complete']=(root/'reports/test_results.json').exists()
print(json.dumps(status,indent=2))
