"""Verify committed render files before recovering a failed job dependency."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,zipfile,datetime,io
root=Path(__file__).resolve().parents[1]
data=root.parents[1]/'dataset/v2-tetra4-4channel-full'
splits={s:json.loads((data/'metadata'/(s+'.json')).read_text()) for s in ['train','val']}
rows=splits['train']+splits['val']
def verify(row):
 meta=json.loads((root/'data/rendered'/(row+'.json')).read_text())
 assert meta['sample']==row and len(meta['poses'])==8,row
 assert all(.15 <= p['path_distance_m'] <= 3.000001 for p in meta['poses']),row
 content=(root/'data/rendered'/(row+'.npz')).read_bytes()
 assert hashlib.sha256(content).hexdigest()==meta['observations_sha256'],row
 with zipfile.ZipFile(io.BytesIO(content)) as z:assert z.testzip() is None,row
 return row
with ThreadPoolExecutor(max_workers=16) as pool:
 for n,_ in enumerate(pool.map(verify,rows),1):
  if n%500==0:print('Verified %d / %d references'%(n,len(rows)),flush=True)
report=dict(passed=True,verified_references={s:len(r) for s,r in splits.items()},
 checks=['8 feasible poses per reference','observation SHA256 matches committed manifest','NPZ archive integrity'],
 time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path=root/'reports/render_completion_audit.json';temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
print(json.dumps(report),flush=True)
