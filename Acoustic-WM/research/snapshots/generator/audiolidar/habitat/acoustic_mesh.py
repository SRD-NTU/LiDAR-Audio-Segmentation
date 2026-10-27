"""Lossless triangle geometry, centroid category approximation for acoustic materials.

The pinned AudioSensor accepts only vertex-labeled semantic geometry. This derived
PLY duplicates triangle vertices and labels each triangle by its original texture
centroid. It is NEVER the LiDAR label source; those use original per-hit UVs.
"""
from pathlib import Path
import numpy as np
from audiolidar.utils.io import write_json,sha256,read_json


def prepare(mesh,scene,cfg):
 root=Path(cfg['config_path']).parent.parent/'cache/acoustic_mesh'/scene['scene_id'];root.mkdir(parents=True,exist_ok=True)
 ply=root/'acoustic.semantic.ply';stage=root/'acoustic.stage_config.json';manifest=root/'manifest.json'
 signature=dict(semantic_sha256=sha256(Path(scene['semantic'])),labels_sha256=sha256(Path(scene['labels'])),
                label_implementation_sha256=sha256(Path(__file__).with_name('scenes.py')),
                adapter_implementation_sha256=sha256(Path(__file__)),
                mesh_to_world_rotation=cfg['mesh_to_world_rotation'],
                method='original triangles; centroid texture label; duplicated vertices; raw Z-up',version=2)
 stage_data=dict(render_asset=str(Path(scene['visual']).resolve()),semantic_asset=str(ply.resolve()),
                 semantic_descriptor_filename=str(Path(scene['labels']).resolve()),has_semantic_textures=False,
                 up=[0,0,1],front=[0,1,0],origin=[0,0,0],shader_type='flat')
 if manifest.exists() and read_json(manifest)==signature and ply.exists():
  # Cache geometry can survive a repository move; absolute asset paths cannot.
  if not stage.exists() or read_json(stage)!=stage_data:write_json(stage,stage_data)
  return stage
 vertices=mesh.mesh.triangles.reshape(-1,3)@np.asarray(cfg['mesh_to_world_rotation'])
 colors=np.array([mesh.annotations.get(int(i),{'color':0})['color'] for i in mesh.face_instances],dtype='uint32')
 rgb=np.column_stack([(colors>>16)&255,(colors>>8)&255,colors&255]).astype('uint8')
 data=np.empty(len(vertices),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
 for i,k in enumerate(['x','y','z']):data[k]=vertices[:,i]
 for i,k in enumerate(['red','green','blue']):data[k]=np.repeat(rgb[:,i],3)
 faces=np.empty(len(rgb),dtype=[('count','u1'),('indices','<i4',(3,))]);faces['count']=3;faces['indices']=np.arange(len(vertices)).reshape(-1,3)
 header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(rgb)}\nproperty list uchar int vertex_indices\nend_header\n'
 with ply.open('wb') as f:f.write(header.encode());data.tofile(f);faces.tofile(f)
 write_json(stage,stage_data)
 write_json(manifest,signature);return stage
