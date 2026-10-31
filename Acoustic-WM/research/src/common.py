"""Experiment paths, atomic writes, deterministic identities and runtime setup."""
from pathlib import Path
import os,sys,json,hashlib,tempfile
BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parents[1]
SCOPE=os.environ.get('EXPERIMENT_SCOPE','historical')
if SCOPE not in ['historical','common']:raise ValueError(SCOPE)
ROOT=BASE if SCOPE=='historical' else BASE/'scopes/common'
DATA=PROJECT/'dataset/v2-tetra4-4channel-full' if SCOPE=='historical' else BASE/'generated/common'
FRONTEND_ROOT=BASE.parent/'frontend_runtime'
ACTIVE=BASE.parent/'Acoustic-WM-ActiveView'
FUSION=BASE.parent/'Acoustic-WM-LidarFusion'
ARCHIVE=PROJECT/'.redun/20261020/Acostic-WM-LidarFusion/outputs'
PYTHON=BASE.parent/'.venv/bin/python'
sys.dont_write_bytecode=True
sys.path[:0]=[str(ACTIVE/'code'),str(FRONTEND_ROOT),str(FRONTEND_ROOT/'runtime')]
for key,folder in [('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp'),('TORCH_EXTENSIONS_DIR','extensions'),('TRITON_CACHE_DIR','triton')]:
 p=ROOT/'runtime'/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');tmp=Path(f.name)
 tmp.replace(path)
def save_npz(path,**values):
 import numpy as np
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:np.savez_compressed(f,**values);tmp=Path(f.name)
 tmp.replace(path)
def save_pt(path,value):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:torch.save(value,f);tmp=Path(f.name)
 tmp.replace(path)
def stable_int(value):return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8],'little')
def seed_all(seed=1337):
 import random,numpy as np,torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
