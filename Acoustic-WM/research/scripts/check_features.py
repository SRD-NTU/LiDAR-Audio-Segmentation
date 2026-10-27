from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
from train_controls import StandardFusion,observation_vector
from acoustic_wm.models.future_grounding import FutureEvidenceHead
import numpy as np,torch
seed_all()
p=next((ROOT/'cache/controlled').rglob('*.npz'))
with np.load(p) as z:
 safe=np.maximum(z['queries'],0);obs=torch.as_tensor(observation_vector(z,safe),device='cuda');feat=torch.as_tensor(z['standard'][safe],device='cuda');base=torch.as_tensor(z['base'],device='cuda');sites=torch.as_tensor(z['sites'],device='cuda');energy=torch.as_tensor(z['energy'][:,safe],device='cuda');weight=torch.as_tensor(z['weight'][safe],device='cuda')
 assert bool(z['prediction_completed_at']<=z['acquisition_completed_at'])
 assert np.isfinite(z['candidate_mse']).all()
 group_error=float(np.max(np.abs(z['group_utilities'][1]-z['duplicate_group_utilities'])))
model=StandardFusion(feat.shape[-1]).cuda();scores=model(base[None].expand(6,-1),feat,obs)
loss=torch.nn.functional.cross_entropy(scores,torch.zeros(6,dtype=torch.long,device='cuda'));loss.backward()
head=FutureEvidenceHead().cuda()
for i in range(5):
 score=head(base[None],sites[None],energy[i],weight)
 assert score.shape==scores.shape and torch.isfinite(score).all()
assert group_error<1e-4
# Check resident tensor packing and inference with a temporary one-row fixture.
import tempfile,train_controls as tc
with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as tmp:
 tmp=Path(tmp);(tmp/'manifests').mkdir();(tmp/'cache/controlled/train').mkdir(parents=True)
 (tmp/'manifests/train.json').write_text(json.dumps(['train/fixture']))
 with np.load(p) as z:fixture={k:z[k] for k in z.files}
 fixture['null_iou']=np.float32(0);save_npz(tmp/'cache/controlled/train/fixture.npz',**fixture)
 original_root=tc.ROOT;tc.ROOT=tmp
 try:
  for method in ['world','standard']:
   data,_=tc.load('train',method)
   _,choices,quality=tc.evaluate(model if method=='standard' else head,data,method)
   assert choices.shape==(1,6) and np.isfinite(quality).all()
 finally:tc.ROOT=original_root
write_json(ROOT/'reports/feature_check.json',dict(success=True,sample=str(p.relative_to(ROOT)),standard_features=feat.shape[-1],observation_dimensions=obs.shape[-1],standard_parameters=sum(p.numel() for p in model.parameters()),duplicate_group_error=group_error,labels_used_for_gradient_test=False))
print('FEATURE_CHECK_OK',flush=True)
