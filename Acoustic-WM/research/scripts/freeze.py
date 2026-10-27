"""Pin runnable sources, configs, external weights and the historical caches."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
files=[]
for folder in ['src','scripts','tests','configs','deployment']:
 files.extend(p for p in (BASE/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.sbatch','.yaml'] and not p.is_symlink())
for package in [ACTIVE,FUSION,BASE.parent/'Acoustic-WM-SingleView']:
 files.extend(p for p in package.rglob('*.py') if 'outputs' not in p.parts and 'assets' not in p.parts)
 files.append(package/'config.json')
for folder in ['scene_frontend','runtime','third_party/Utonia','third_party/sonata']:
 files.extend((FRONTEND_ROOT/folder).rglob('*.py'))
source={os.path.relpath(p,BASE):sha(p) for p in sorted(set(files))};write_json(BASE/'manifests/campaign_source.json',source)
external=[];cfg=json.loads((ACTIVE/'config.json').read_text())
for key in ['masks','selector','apex','gate','gate2']:external.append(Path(cfg['paths'][key]).resolve())
external.extend([FRONTEND_ROOT/'cache/utonia-xyz-v1/projection.pt',FRONTEND_ROOT/'cache/spatial-v1/adaptive.pt',FRONTEND_ROOT/'pretrained/utonia/utonia.pth',ACTIVE/'assets/pointbind/point-audio.pt'])
for package in [ACTIVE,FUSION,BASE.parent/'Acoustic-WM-SingleView']:
 external.extend((package/'checkpoints').glob('*.pt'))
 pointbind=package/'assets/pointbind/point-audio.pt'
 if pointbind.exists():external.append(pointbind)
clap=FRONTEND_ROOT/'pretrained/clap-hub/models--laion--clap-htsat-unfused';snap=clap/'snapshots'/(clap/'refs/main').read_text().strip()
external.extend(p for p in snap.iterdir() if p.is_file())
external.extend((FRONTEND_ROOT/'third_party/Utonia').rglob('*.py'))
external.extend((FRONTEND_ROOT/'third_party/sonata').rglob('*.py'))
manifest={}
for p in sorted(set(external)):
 if not p.is_file():raise FileNotFoundError(p)
 p=p.resolve();manifest[str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
write_json(BASE/'manifests/external_dependencies.json',manifest)
# Cache identity map avoids a hidden mutable training corpus. Size+mtime are
# checked on workers; cache provenance/source hashes stay in the copied archive.
cache={}
for folder in [ARCHIVE/'fusion_recovery_v2/extended',ARCHIVE/'fusion_training_v1/cache',ARCHIVE/'fusion_recovery_v2/geometry',ARCHIVE/'fusion_training_v1/geometry']:
 for p in folder.rglob('*'):
  if p.is_file():cache[str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
write_json(BASE/'manifests/historical_cache_identity.json',cache)
(BASE/'configs/READY').write_text('Frozen campaign sources and read-only dependency identities are in manifests/.\n')
print('Frozen',len(source),'source/config files,',len(manifest),'external dependencies,',len(cache),'cached records')
