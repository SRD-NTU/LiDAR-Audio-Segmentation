"""Privileged simulator used only to generate actual observations, never policy features."""
from .runtime import *
import numpy as np
sys.path.insert(0,str(ARCHIVE/'snapshots/generator'))
sys.path.insert(0,str(ARCHIVE/'src'))
from audiolidar.habitat.scenes import discover,annotations,SemanticMesh
from common_menu_backend import CommonMenuBackend
from .navigation import rotation,ReachableSpace
import yaml,soundfile as sf,math
from scipy.signal import fftconvolve,resample_poly

def dry_segment(metadata):
 wave,rate=sf.read(DATA/metadata['audio']['dry_audio_file'],dtype='float32',always_2d=True)
 wave=wave.mean(-1)
 if rate!=44100:
  g=math.gcd(rate,44100);wave=resample_poly(wave,44100//g,rate//g)
 n=88200
 if len(wave)>n:
  energy=np.r_[0.,np.cumsum(wave.astype(float)**2)];start=int((energy[n:]-energy[:-n]).argmax());wave=wave[start:start+n]
 else:wave=np.pad(wave,(0,n-len(wave)))
 return wave.astype('float32')

class Renderer:
 def __init__(self,scene_id):
  scenes={s['scene_id']:s for s in discover(UTILITY/'data/hm3d-v02')};scene=scenes[scene_id]
  cfg=yaml.safe_load((ARCHIVE/'snapshots/generator/base_config.yaml').read_text())
  cfg.update(seed=20261028,config_path=str(ROOT/'config/renderer.yaml'),
   dataset_config=str(UTILITY/'data/hm3d-v02/hm3d_annotated_basis.scene_dataset_config.json'))
  cfg['audio']['mode']='mono';cfg['soundspaces']['materials_json']=str(DATA/'assets/mp3d_material_config.json')
  cfg['soundspaces']['acoustics']['threadCount']=2
  categories=json.loads((DATA/'metadata/categories.json').read_text())
  for record in annotations(Path(scene['labels'])).values():
   if record['category'] not in categories:categories[record['category']]=len(categories)
  self.mesh=SemanticMesh(scene,categories,np.asarray(cfg['mesh_to_world_rotation']))
  self.backend=CommonMenuBackend(scene,cfg,mesh=self.mesh)
  self.offsets=np.asarray(json.loads((DATA/'metadata/microphone_array.json').read_text())['positions_lidar_m'])
  self.scene=scene
 def space(self,listener,budget=3.):
  return ReachableSpace(self.backend.sim.pathfinder,self.backend.hs.ShortestPath,
   listener['floor_position_world'],budget=budget,clearance=self.mesh.clearance)
 def acquire(self,emitter,dry,pose):
  positions=self.offsets@rotation(pose['yaw']).T+np.asarray(pose['position'])
  responses=[self.backend.render(emitter,p,pose['yaw'])[0] for p in positions]
  n=max(map(len,responses));rir=np.stack([np.pad(h,(0,n-len(h))) for h in responses]).astype('float32')
  wave=np.stack([fftconvolve(dry,h)[:len(dry)] for h in rir],-1).astype('float32')
  return wave,rir
 def close(self):self.backend.close()
