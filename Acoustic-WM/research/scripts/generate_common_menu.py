"""Scene-level source-independent receiver menus and physical branching data."""
from pathlib import Path
import sys,os,json,hashlib,fcntl,time,copy,math,tempfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'snapshots/generator'))
import numpy as np,yaml,soundfile as sf
from scipy.signal import fftconvolve,resample_poly
from audiolidar.habitat.scenes import discover,SemanticMesh
sys.path.insert(0,str(ROOT/'src'))
from common_menu_backend import CommonMenuBackend as Backend
from audiolidar.lidar.rays import scan
from audiolidar.geometry.frames import sensor_rotation,quaternion_xyzw,relative
HIST=PROJECT/'dataset/v2-tetra4-4channel-full';OUT=ROOT/'generated/common';UTILITY=PROJECT/'.redun/dataset-utility/utility'
PLAN=json.loads((ROOT/'configs/common_menu.json').read_text())
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
  stream.write(json.dumps(value,indent=2)+'\n');tmp=Path(stream.name)
 tmp.replace(path)
def generate(rank=0,limit=0):
 scenes={s['scene_id']:s for s in discover(UTILITY/'data/hm3d-v02')};tasks={}
 for split in ['train','val','test']:
  for sample in json.loads((HIST/'metadata'/f'{split}.json').read_text()):
   _,scene,source,index=sample.split('/')
   if index=='00000':tasks.setdefault((split,scene),[]).append(sample)
 ordered=sorted(tasks,key=lambda x:({'train':0,'val':1,'test':2}[x[0]],x[1]));ordered=ordered[rank%len(ordered):]+ordered[:rank%len(ordered)]
 count=0
 for split,scene_id in ordered:
  dest=OUT/'scenes'/scene_id
  if (dest/'COMPLETE.json').exists():continue
  dest.mkdir(parents=True,exist_ok=True)
  with (dest/'lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (dest/'COMPLETE.json').exists():continue
   seed=(PLAN['seed']+int(hashlib.sha256(scene_id.encode()).hexdigest()[:8],16))%(2**32);rng=np.random.default_rng(seed)
   scene=scenes[scene_id];cfg=yaml.safe_load((ROOT/'snapshots/generator/base_config.yaml').read_text())
   cfg['seed']=seed;cfg['config_path']=str(ROOT/'configs/generator.yaml');cfg['dataset_config']=str(UTILITY/'data/hm3d-v02/hm3d_annotated_basis.scene_dataset_config.json');cfg['audio']['mode']='mono';cfg['soundspaces']['materials_json']=str(UTILITY/cfg['soundspaces']['materials_json']);cfg['soundspaces']['acoustics']['threadCount']=4
   from audiolidar.habitat.scenes import annotations
   cats=sorted({r['category'] for r in annotations(Path(scene['labels'])).values()});category_ids={c:i for i,c in enumerate(cats)}
   mesh=SemanticMesh(scene,category_ids,np.asarray(cfg['mesh_to_world_rotation']));backend=Backend(scene,cfg,mesh=mesh)
   try:
    menu_path=dest/'menu.json'
    if menu_path.exists():poses=json.loads(menu_path.read_text())['poses']
    else:
     sys.path.insert(0,str(ROOT/'src'))
     from geometry_menu import sample_geometry_menu
     poses=sample_geometry_menu(backend.sim.pathfinder,mesh.clearance,seed,PLAN)
     write(menu_path,{'scene':scene_id,'seed':seed,'poses':poses,'source_inputs_used':False,'rule':PLAN})
    menu_hash=hashlib.sha256(menu_path.read_bytes()).hexdigest()
    # Geometry is raycast and stored BEFORE any source is assigned to this menu.
    for j,p in enumerate(poses):
     path=dest/f'geometry-{j:05d}.npz'
     if not path.exists():np.savez_compressed(path,**scan(mesh,np.asarray(p['position']),p['yaw'],cfg['lidar']))
    for sample in sorted(tasks[(split,scene_id)]):
     meta=json.loads((HIST/sample/'metadata.json').read_text());source=meta['source'];emitter=np.asarray(source['position_world']);ident=source['instance_id']
     dry,rate=sf.read(HIST/meta['audio']['dry_audio_file'],dtype='float32',always_2d=True);dry=dry.mean(1);n=round(meta['audio']['duration_sec']*44100)
     if rate!=44100:
      g=math.gcd(rate,44100);dry=resample_poly(dry,44100//g,rate//g).astype(np.float32)
     if len(dry)>n:
      c=np.r_[0.,np.cumsum(dry.astype(float)**2)];start=int((c[n:]-c[:-n]).argmax());dry=dry[start:start+n]
     else:dry=np.pad(dry,(0,n-len(dry)))
     for j,p in enumerate(poses):
      target=OUT/split/scene_id/str(ident)/f'{j:05d}'
      if (target/'COMPLETE.json').exists():continue
      target.mkdir(parents=True,exist_ok=True);position=np.asarray(p['position']);yaw=p['yaw'];rotation=sensor_rotation(yaw)
      offsets=np.asarray(meta['audio']['microphone_array']['positions_lidar_m']);microphones=offsets@rotation.T+position
      responses=[backend.render(emitter,mic,yaw)[0] for mic in microphones];length=max(map(len,responses));rir=np.stack([np.pad(h,(0,length-len(h))) for h in responses])
      wave=np.stack([fftconvolve(dry,h)[:n] for h in rir],1).astype(np.float32)
      with np.load(dest/f'geometry-{j:05d}.npz') as z:points={k:z[k] for k in z.files}
      points['sound_mask']=(points['instance_id']==ident).astype(np.uint8)
      m=copy.deepcopy(meta);m['version']='common-menu-20261024';m['listener']={'position_world':position.tolist(),'floor_position_world':p['floor'],'rotation_world':quaternion_xyzw(yaw),'yaw_rad':yaw};m['relative_source']=relative(emitter,position,yaw);m['audio']['microphone_array']['positions_world_m']=microphones.tolist();m['audio']['rir_file']='rir.npy';m['lidar']['num_points']=len(points['xyz']);m['lidar']['num_source_points']=int(points['sound_mask'].sum());m['lidar']['source_point_fraction']=float(points['sound_mask'].mean());m['menu_sha256']=menu_hash;m['availability_basis']='shared scene geometry menu before source assignment'
      m['audio']['zero_response_channels']=[i for i,h in enumerate(responses) if not np.any(h)];m['audio']['zero_response_interpretation']='Raw finite simulator output retained; not proof of physical silence or simulator correctness'
      np.savez_compressed(target/'lidar.npz',**points);np.save(target/'rir.npy',rir);sf.write(target/'audio.wav',wave,44100,subtype='FLOAT');write(target/'metadata.json',m);write(target/'COMPLETE.json',{'complete':True,'menu_sha256':menu_hash})
    write(dest/'COMPLETE.json',{'scene':scene_id,'split':split,'sources':len(tasks[(split,scene_id)]),'receivers':len(poses),'menu_sha256':menu_hash,'source_independent':True});print('COMMON_MENU_DONE',scene_id,flush=True);count+=1
   finally:backend.close()
   if limit and count>=limit:return
 # A full manifest is written only when every declared scene is complete.
 if all((OUT/'scenes'/scene/'COMPLETE.json').exists() for _,scene in tasks):
  with (OUT/'finalize.lock').open('a') as final_lock:
   fcntl.flock(final_lock,fcntl.LOCK_EX)
   if (OUT/'COMPLETE.json').exists():return
   for split in ['train','val','test']:
    rows=sorted(str(p.parent.relative_to(OUT)) for p in (OUT/split).glob('*/*/*/COMPLETE.json'))
    write(OUT/'metadata'/f'{split}.json',rows);write(ROOT/'scopes/common/manifests'/f'{split}.json',rows)
   import shutil
   (OUT/'metadata').mkdir(exist_ok=True);shutil.copy2(HIST/'metadata/microphone_array.json',OUT/'metadata/microphone_array.json')
   (OUT/'assets').mkdir(exist_ok=True)
   if not (OUT/'assets/dry_audio').exists():(OUT/'assets/dry_audio').symlink_to(HIST/'assets/dry_audio',target_is_directory=True)
   write(OUT/'COMPLETE.json',{'complete':True,'scenes':len(tasks),'rule':PLAN})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int,default=0);a=p.parse_args();generate(a.rank,a.limit)
