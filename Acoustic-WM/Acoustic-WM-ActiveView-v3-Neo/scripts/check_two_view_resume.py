"""One VAL episode: interrupted result writing must not change acquisition artifacts."""
from pathlib import Path
import sys,subprocess,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import ROOT,samples,sha,write_json
sample=samples('val')[0]
base=ROOT/'data/evaluation_auditfix1'/sample
args=[sys.executable,'-B',str(ROOT/'scripts/evaluate.py'),'--split','val','--limit','1',
      '--methods','random-0','--protocol','config/frozen_evaluation_auditfix1.json']
subprocess.run(args,check=True)
paths=[base/'random-0-commit.json',base/'random-0-audio.npz',base/'random-0-receipt.json']
before={p.name:sha(p) for p in paths}
result=base/'random-0.json';saved=result.read_bytes();result.unlink()
try:subprocess.run(args,check=True)
except Exception:
 result.write_bytes(saved)
 raise
after={p.name:sha(p) for p in paths}
if before!=after:raise RuntimeError('Resume changed committed action or observation')
if json.loads(result.read_text())['protocol_sha256']!=json.loads(saved)['protocol_sha256']:
 raise RuntimeError('Resume changed protocol')
write_json('reports/auditfix1/two_view_resume.json',dict(passed=True,sample=sample,unchanged_sha256=after))
print('Two-view resume preserved commitment, observation, and receipt',flush=True)
