"""Train-split only end-to-end contract smoke. Never creates a model checkpoint."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.inference import *
from activeview_v3.models import evidence_loss,world_loss
from activeview_v3.simulator_client import Simulator
import time

seed_all(17);start=time.time();sample=next(s for s in samples('train') if (ROOT/'data/features'/(s+'.npz')).exists())
with np.load(ARCHIVE/'cache/reference'/(sample+'.npz')) as z:row={k:z[k] for k in z.files}
with np.load(ROOT/'data/features'/(sample+'.npz')) as z:cached={k:z[k] for k in z.files}
a=Adapter.__new__(Adapter);a.device='cuda';a.frozen=FrozenSingleView();a.embedding=AcousticEmbedding()
a.evidence=EvidenceAdapter().cuda();a.world=ObservationWorld().cuda();a.control=ObservationWorld(correction=False).cuda();a.direct=None;a.scale=.25
b,prepared=a.prepare(row,cached['actions'],cached['distance'])
parity={}
for key in ['base','sites','latent','support','initial_silent','residual','mu','var','actions','distance','overlap']:
 actual=b[key].cpu().numpy();expected=cached[key]
 if key in ['base','sites','latent','support','initial_silent','residual','overlap']:expected=np.broadcast_to(expected,actual.shape)
 np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6);parity[key]=True
b.update(target=torch.as_tensor(cached['target'],device='cuda'),audible=torch.as_tensor(cached['audible'],device='cuda'),
 iou=torch.as_tensor(cached['iou'],device='cuda')[None].expand(8,-1))
opt=torch.optim.AdamW(a.evidence.parameters(),lr=.001);evidence_losses=[]
for _ in range(5):
 loss=evidence_loss(a.evidence,b);opt.zero_grad();loss.backward();opt.step();evidence_losses.append(float(loss.detach()))
a.evidence.eval().requires_grad_(False);opt=torch.optim.AdamW(a.world.parameters(),lr=.0003);world_losses=[]
for _ in range(5):
 loss,metric=world_loss(a.world,a.evidence,b,scale=a.scale);opt.zero_grad();loss.backward();opt.step();world_losses.append(float(loss.detach()))
if not np.isfinite(evidence_losses+world_losses).all():raise RuntimeError('Nonfinite smoke training')
sim=Simulator('logs/integration-renderer.log')
try:
 listener=sim.reference(sample)
 chosen=plan(a,row,listener,sim,seed=17,initial=8,rounds=1,population=4,elite=3,draws=8)
 acquired=sim.acquire(chosen['pose'],'examples/integration-observation.npz',save_audio=True)
 with np.load(acquired['output']) as z:prediction=a.update(chosen['batch'],chosen['prepared'],0,z['observed'],z['weight'])
 assert len(prediction['mask'])==len(row['inverse'])
 write_json('reports/integration_smoke.json',dict(passed=True,sample=sample,feature_parity=parity,
  evidence_losses=evidence_losses,world_losses=world_losses,pose=chosen['pose'],evaluated_poses=chosen['evaluated_poses'],
  original_point_count=len(prediction['mask']),predicted_points=int(prediction['mask'].sum()),candidate=prediction['candidate'],
  seconds=time.time()-start,untrained_smoke_only=True))
finally:sim.close()
print((ROOT/'reports/integration_smoke.json').read_text())
