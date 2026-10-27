from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT.parent/'Acoustic-WM-ActiveView'
FRONTEND_ROOT=ROOT.parent/'frontend_runtime'
sys.path[:0]=[str(ACTIVE/'code'),str(FRONTEND_ROOT),str(FRONTEND_ROOT/'runtime')]
import torch,numpy as np
from acoustic_wm.model import ReleasedModel
print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(),flush=True)
torch.set_num_threads(1)
model=ReleasedModel(ACTIVE,'cuda')
row_path=ACTIVE/'examples/reference.npz'
with np.load(row_path) as z:
 print('example',row_path,{k:z[k].shape for k in z.files},flush=True)
 row={k:z[k] for k in z.files}
if 'observed' in row:
 state=model.current(row,row['observed'],row['reliability'])
 print('loaded and inferred',len(state['base']),flush=True)
print('SMOKE_OK',flush=True)
(ROOT/'reports/gpu_smoke.json').write_text(json.dumps({'success':True,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'time':time.time()},indent=2))
