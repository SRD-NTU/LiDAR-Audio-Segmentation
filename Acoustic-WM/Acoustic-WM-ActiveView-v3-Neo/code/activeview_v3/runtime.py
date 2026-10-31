from pathlib import Path
import os,sys,json,hashlib,tempfile
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT.parents[1]
EXPERIMENT_OUTPUT_ROOT=PROJECT/'Experiments_20261030'
BASE=ROOT.parent/'Acoustic-WM-SingleView'
DATA=PROJECT/'dataset/v2-tetra4-4channel-full'
ARCHIVE=PROJECT/'.importants/Experiments_20261024'
UTILITY=PROJECT/'.redun/dataset-utility/utility'
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'code'))
for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),
 ('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TRITON_CACHE_DIR','triton'),
 ('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda'),('CUPY_CACHE_DIR','cupy'),
 ('TORCHINDUCTOR_CACHE_DIR','inductor'),('HF_HOME','huggingface')]:
 p=ROOT/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  USE_TF='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def output_path(path):
 p=Path(path);p=(ROOT/p if not p.is_absolute() else p).resolve()
 if not (p.is_relative_to(ROOT) or p.is_relative_to(EXPERIMENT_OUTPUT_ROOT)):
  raise ValueError('Output must stay under the repository or '+str(EXPERIMENT_OUTPUT_ROOT))
 p.parent.mkdir(parents=True,exist_ok=True);return p
def write_json(path,value):
 p=output_path(path)
 with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');temp=Path(f.name)
 temp.replace(p)
def save_pt(path,value):
 import torch
 p=output_path(path)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:torch.save(value,f);temp=Path(f.name)
 temp.replace(p)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fingerprints():return {p.name:sha(p) for p in sorted((BASE/'checkpoints').glob('*.pt'))}
def feature_contract():
 value=dict(singleview_checkpoints=fingerprints(),embedding_sha256=sha(ROOT/'checkpoints/embedding.npz'),
            feature_version=1,dimension=128,sites=4,poses_per_reference=8)
 path=ROOT/'data/feature_contract.json'
 if path.exists():
  if json.loads(path.read_text())!=value:raise RuntimeError('Feature cache model/embedding contract changed; use a new cache')
 else:write_json(path,value)
 return value
def samples(split):return json.loads((DATA/'metadata'/(split+'.json')).read_text())
def seed_all(seed):
 import random,numpy as np,torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False
