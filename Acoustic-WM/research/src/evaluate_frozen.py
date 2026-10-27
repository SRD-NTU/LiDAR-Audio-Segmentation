"""Evaluate the unchanged released evidence head on the matched policy matrix."""
from common import *
from train_controls import load,evaluate
from acoustic_wm.models.future_grounding import FutureEvidenceHead
from extract import POLICIES
import numpy as np,torch
seed_all();folder=ROOT/'runs/controls/frozen-world-1337';folder.mkdir(parents=True,exist_ok=True)
model=FutureEvidenceHead().cuda();model.load_state_dict(torch.load(ACTIVE/'checkpoints/future_evidence.pt',map_location='cpu',weights_only=False)['model'])
for split in ['val','test']:
 if (folder/f'{split}.json').exists():continue
 data,rows=load(split,'world');metric,choice,quality=evaluate(model,data,'world');start=data['base'].argmax(-1);base=torch.where(data['initial_silent'],data['null_iou'],data['quality'].gather(1,start[:,None])[:,0]).cpu().tolist()
 write_json(folder/f'{split}.json',dict(method='frozen-world',seed=1337,policies=POLICIES,mean_iou=metric.tolist(),single_iou=float(np.mean(base)),samples=[dict(sample=r,scene=r.split('/')[1],choices=choice[i].tolist(),iou=quality[i].tolist(),single_iou=base[i]) for i,r in enumerate(rows)]));del data;torch.cuda.empty_cache()
write_json(folder/'complete.json',dict(training_performed=False,parameters=50,checkpoint_sha256=sha(ACTIVE/'checkpoints/future_evidence.pt')))
