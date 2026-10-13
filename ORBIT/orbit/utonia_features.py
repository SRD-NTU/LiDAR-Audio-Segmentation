"""XYZ-only Utonia features with train-fitted PCA and exact original point mapping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from apex import data as D
from apex.predict import range_grid

HERE=Path(__file__).resolve().parents[1]


class UtoniaFeatures:
    def __init__(self,checkpoint,projection=None,device='cuda',patch_size=256):
        sys.path.insert(0,str(HERE/'third_party/Utonia'))
        sys.path.insert(0,str(D.ROOT/'LiDAR-WM/src'))
        from utonia.model import load
        from lidarwm.sonata_geometry import geometry_normals
        self.normals=geometry_normals;self.device=device
        self.encoder=load(str(checkpoint),custom_config=dict(enable_flash=False,enc_patch_size=[patch_size]*5,
            shuffle_orders=False,drop_path=0.,upcast_attention=True,upcast_softmax=True)).to(device).eval()
        for m in self.encoder.modules():
            if hasattr(m,'shuffle_orders'):m.shuffle_orders=False
        self.projection=projection
        if projection is not None:
            self.projection={k:torch.as_tensor(v,device=device) for k,v in projection.items()}

    @torch.inference_mode()
    def encode(self,xyz):
        grid,valid,pixel=range_grid(xyz)
        image=torch.from_numpy(grid)[None].to(self.device)*20
        normals=self.normals(image,torch.from_numpy(valid)[None].to(self.device))
        # Match Sonata's calibrated pixel order and its physical 5 cm voxel grid.
        order=np.argsort(pixel);sorted_pixel=pixel[order]
        # Reproduce the legacy adapter's float32 range-grid round trip before
        # gridding; tiny roundoff at voxel boundaries otherwise changes cells.
        npixel=torch.tensor(sorted_pixel,device=self.device,dtype=torch.long)
        points=image[0].flatten(1).T[npixel]
        lower=points.amin(0);upper=points.amax(0)
        shift=torch.stack(((lower[0]+upper[0])/2,(lower[1]+upper[1])/2,lower[2]))
        centered=points-shift
        cells=torch.floor((centered-centered.amin(0))/.05).long()
        _,inverse=torch.unique(cells,dim=0,return_inverse=True)
        n=int(inverse.max())+1
        first=torch.full((n,),len(points),device=self.device,dtype=torch.long)
        first.scatter_reduce_(0,inverse,torch.arange(len(points),device=self.device),reduce='amin')
        position=centered[first]*.5  # official indoor demo's coordinate scale
        normal=normals[0].flatten(1).T[npixel[first]]
        point=self.encoder(dict(coord=position,grid_coord=cells[first].int(),
            feat=torch.cat([position,torch.zeros_like(position),normal],-1),
            batch=torch.zeros(n,dtype=torch.long,device=self.device)))
        while 'pooling_parent' in point:
            parent=point.pop('pooling_parent');inv=point.pop('pooling_inverse')
            parent.feat=torch.cat([parent.feat,point.feat[inv]],-1);point=parent
        feat=point.feat
        if self.projection is not None:
            norm=(feat-self.projection['mean'])/self.projection['std']
            tail=norm[:,54:]@self.projection['components']
            feat=torch.cat([norm[:,:54],tail/self.projection['scale']],-1)
        return feat,sorted_pixel,inverse.cpu().numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=HERE/'pretrained/utonia/utonia.pth')
    p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda')
    p.add_argument('--projection',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=True
    model=UtoniaFeatures(args.checkpoint,device=args.device)
    start=time.time();rng=np.random.default_rng(991)
    projection_rels=[]
    if args.projection:
        projection=torch.load(args.projection,map_location=args.device,weights_only=True)
    else:
        tr=D.rels('train');samples=[]
        for i in np.linspace(0,len(tr)-1,64,dtype=int):
            rel=tr[i];projection_rels.append(rel)
            with np.load(D.DATA/rel/'lidar.npz') as z:xyz=z['xyz'].astype(np.float32)
            full,_,_=model.encode(xyz)
            idx=rng.choice(len(full),min(1024,len(full)),replace=False)
            samples.append(full[idx].clone())
        samples=torch.cat(samples);mean=samples.mean(0);std=samples.std(0).clamp_min(.02)
        normalized=(samples-mean)/std
        covariance=normalized[:,54:].T@normalized[:,54:]/(len(samples)-1)
        eigenvalues,components=torch.linalg.eigh(covariance)
        projection=dict(mean=mean,std=std,components=components[:,-202:],scale=eigenvalues[-202:].clamp_min(.01).sqrt())
        del samples,normalized,covariance
    torch.save({k:v.cpu() for k,v in projection.items()},args.output/'projection.pt')
    model.projection=projection
    print(f'PCA_READY {time.time()-start:.1f}s',flush=True)
    checked=0;written={}
    for split in ('train','val'):
        rs=D.rels(split)
        if args.limit:rs=[rs[i] for i in np.linspace(0,len(rs)-1,min(len(rs),args.limit),dtype=int)]
        written[split]=rs
        for i,rel in enumerate(rs):
            path=args.output/(rel+'.npz')
            if path.exists():raise FileExistsError(path)
            with np.load(D.DATA/rel/'lidar.npz') as z:xyz=z['xyz'].astype(np.float32)
            features,pixels,inverse=model.encode(xyz)
            # Validate that the fixed grid actually matches the legacy physical
            # partition; ordering is checked by pixel identity, not array length.
            with np.load(D.GEOMETRY/(rel+'.npz')) as z:
                if not np.array_equal(pixels,z['pixels']) or not np.array_equal(inverse,z['inverse']):
                    raise AssertionError(f'Voxel mapping changed: {rel}')
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,features=features.half().cpu().numpy(),pixels=pixels,inverse=inverse,
                xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest())
            checked+=1
            if (i+1)%100==0:print(f'utonia {split} {i+1}/{len(rs)} {time.time()-start:.1f}s',flush=True)
    report=dict(written=written,checked_mappings=checked,test_used=False,feature_dim=256,
        backbone='Pointcept/Utonia',huggingface_revision='fb39b40d33d215d29d1d4717ffb86287f4700c4d',
        code_revision='da776a0bd3a48c6df83ac2ae0e27b26141cc7e31',checkpoint_sha256=D.sha(args.checkpoint),
        projection_sha256=D.sha(args.output/'projection.pt'),projection_training_samples=projection_rels,
        inherited_projection=str(args.projection) if args.projection else None,
        source_sha256=D.sha(Path(__file__)),inputs=['xyz','xyz-derived normals','zero color'],
        physical_voxel_size=.05,coordinate_scale=.5,patch_size=256,
        pretraining_scene_overlap='Unresolved: Utonia pretraining includes HM3D; disclose this limitation',
        elapsed_seconds=time.time()-start)
    (args.output/'COMPLETE.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='written'},indent=2),flush=True)


if __name__=='__main__':main()
