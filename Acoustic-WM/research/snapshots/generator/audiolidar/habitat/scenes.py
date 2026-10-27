"""HM3DSem scene discovery and semantic-mesh triangle labels."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


def discover(root: Path) -> list:
    scenes = []
    # ``data/hm3d-v02`` uses split-level symlinks to avoid duplicating the
    # verified HM3D archives.  pathlib.rglob does not descend through those
    # links, so walk with followlinks enabled and retain deterministic order.
    label_files = []
    for directory, _, filenames in os.walk(Path(root), followlinks=True):
        label_files.extend(Path(directory) / name for name in filenames
                           if name.endswith('.semantic.txt'))
    for label_file in sorted(label_files):
        stem = label_file.name.removesuffix('.semantic.txt')
        files = {k: label_file.parent/(stem+ext) for k, ext in
                 [('visual', '.basis.glb'), ('semantic', '.semantic.glb'), ('navmesh', '.navmesh')]}
        if not files['navmesh'].exists(): files['navmesh'] = label_file.parent/(stem+'.basis.navmesh')
        files['labels'] = label_file
        if all(p.is_file() for p in files.values()):
            scenes.append({'scene_id': label_file.parent.name, **{k: str(v.resolve()) for k,v in files.items()}})
    return scenes


def annotations(path: Path) -> dict:
    result = {}
    with Path(path).open() as f:
        header = next(f)
        if 'HM3D Semantic Annotations' not in header:
            raise ValueError('Not an HM3D semantic annotation file')
        for row in csv.reader(f, skipinitialspace=True):
            if not row: continue
            idx, color, category, region = row[:4]
            result[int(idx)] = {'instance_id': int(idx), 'color': int(color,16),
                                'category': category.strip(), 'region_id': int(region)}
    return result


class SemanticMesh:
    """Use original semantic mesh first-hit triangles and exact annotation colors."""
    def __init__(self, scene: dict, category_ids: dict, rotation: np.ndarray):
        self.annotations = annotations(Path(scene['labels']))
        loaded = trimesh.load(scene['semantic'], force='scene', process=False)
        parts=list(loaded.dump())
        self.parts=parts
        self.ends=np.cumsum([len(g.faces) for g in parts])
        self.starts=np.r_[0,self.ends[:-1]]
        self.lookup={v['color']:k for k,v in self.annotations.items()}
        self.category_ids=category_ids
        self.mesh=trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices,faces=g.faces,process=False) for g in parts])
        self.mesh.vertices=self.mesh.vertices@np.asarray(rotation).T
        self.intersector=RayMeshIntersector(self.mesh)
        faces=np.arange(len(self.mesh.faces))
        # Centroid labels guide source proposals only. Every LiDAR hit is sampled
        # independently from the original semantic texture at its barycentric UV.
        self.face_instances=self.labels_at(self.mesh.triangles_center,faces)
        self.face_categories=self.categories_at(self.face_instances)
        self.objects={}
        for idx,record in self.annotations.items():
            selected=np.flatnonzero(self.face_instances==idx)
            if not len(selected):continue
            vertices=self.mesh.vertices[np.unique(self.mesh.faces[selected])]
            lo,hi=vertices.min(axis=0),vertices.max(axis=0)
            self.objects[idx]=dict(record,semantic_id=category_ids[record['category']],
                center=((lo+hi)/2).tolist(),bbox_min=lo.tolist(),bbox_max=hi.tolist(),
                bbox_dimensions=(hi-lo).tolist(),bbox_method='bounds of triangles with instance-labeled centroids')

    def categories_at(self, instances):
        return np.array([self.category_ids.get(self.annotations.get(int(i),{}).get('category'),-1) for i in instances],dtype='int32')

    def labels_at(self, points, faces):
        faces=np.asarray(faces);result=np.full(len(faces),-1,dtype='int32')
        groups=np.searchsorted(self.ends,faces,side='right')
        for group in np.unique(groups):
            selected=np.flatnonzero(groups==group);g=self.parts[group]
            local_faces=faces[selected]-self.starts[group]
            # Zero-area input triangles have no valid barycentric coordinates.
            # Preserve the original triangle indexing, but never turn their
            # NaNs into texture pixel zero (which would invent an instance).
            with np.errstate(divide='ignore', invalid='ignore'):
                bary=trimesh.triangles.points_to_barycentric(self.mesh.triangles[faces[selected]],np.asarray(points)[selected])
            valid=np.isfinite(bary).all(axis=1)
            selected=selected[valid];local_faces=local_faces[valid];bary=bary[valid]
            if not len(selected):continue
            if g.visual.kind=='texture':
                uv=np.einsum('ni,nij->nj',bary,g.visual.uv[g.faces[local_faces]])
                valid=np.isfinite(uv).all(axis=1)
                selected=selected[valid];uv=uv[valid]
                if not len(selected):continue
                material=g.visual.material
                image=np.asarray(material.baseColorTexture if hasattr(material,'baseColorTexture') else material.image)
                height,width=image.shape[:2]
                x=np.clip(np.floor(uv[:,0]*width).astype(int),0,width-1)
                y=np.clip(np.floor((1-uv[:,1])*height).astype(int),0,height-1)
                rgb=image[y,x,:3].astype('int64')
            else:
                colors=g.visual.vertex_colors[g.faces[local_faces],:3]
                rgb=colors[np.arange(len(colors)),np.argmax(bary,axis=1)].astype('int64')
            packed=rgb[:,0]*65536+rgb[:,1]*256+rgb[:,2]
            result[selected]=[self.lookup.get(int(c),-1) for c in packed]
        return result

    def clearance(self, position: np.ndarray, radius: float = .08) -> bool:
        directions = np.vstack([np.eye(3),-np.eye(3)])
        xyz, _, _ = self.intersector.intersects_location(np.tile(position,(6,1)), directions, multiple_hits=False)
        if len(xyz) and np.any(np.linalg.norm(xyz-position,axis=1)<=radius):return False
        # Exact nearest-surface clearance also catches obstacles between probe rays.
        _,distance,_=trimesh.proximity.closest_point(self.mesh,np.asarray(position)[None,:])
        return bool(np.isfinite(distance[0]) and distance[0]>radius)

    def source_position(self, instance: int, listener: np.ndarray, offset: float = .06) -> np.ndarray:
        """Find a measured instance surface, then offset into its visible free side.

        Position stays fixed across receiver samples. This avoids a potentially solid
        bbox centre; labels still come exclusively from triangle instance IDs.
        """
        ids = np.flatnonzero(self.face_instances == instance)
        if not len(ids): raise ValueError('Instance has no labeled triangles')
        centers = self.mesh.triangles_center[ids]
        target = np.asarray(self.objects[instance]['center'])
        order = np.argsort(np.linalg.norm(centers-target,axis=1))
        for i in order[::max(1,len(order)//1000)]:
            center = centers[i]
            delta = center-listener
            distance = np.linalg.norm(delta)
            if distance < .2: continue
            direction = delta/distance
            xyz, _, faces = self.intersector.intersects_location(listener[None,:], direction[None,:], multiple_hits=False)
            if not len(faces) or self.labels_at(xyz,faces)[0] != instance: continue
            point = xyz[0] - offset*direction
            if self.clearance(point, offset*.45): return point
        raise ValueError('No free-side emitting surface found for this instance/listener')
