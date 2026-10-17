import numpy as np

def scene_tokens(xyz, count=192):
    """Deterministic spherical binning, independent of labels/source identity."""
    xyz = np.asarray(xyz, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite nonempty XYZ')
    # 48 azimuth x 4 elevation sectors retain measured surface geometry.
    az = np.floor((np.arctan2(xyz[:,1],xyz[:,0]) + np.pi) / (2*np.pi) * 48).astype(int) % 48
    el = np.clip(((np.arctan2(xyz[:,2],np.linalg.norm(xyz[:,:2],axis=1))+np.pi/2)/np.pi*4).astype(int),0,3)
    ids = az * 4 + el
    n = np.bincount(ids,minlength=192)
    points = np.stack([np.bincount(ids,weights=xyz[:,d],minlength=192)/np.maximum(n,1) for d in range(3)],1)
    return points.astype(np.float32), (n>0)
