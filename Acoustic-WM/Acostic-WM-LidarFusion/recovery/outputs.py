"""Map both measured scans back onto the registered fused representatives."""
import numpy as np
from acoustic_wm.fusion import fuse_clouds


def fused_outputs(reference_xyz,second_xyz,action,reference_mask,second_mask,reference_probability,second_probability):
    cloud=fuse_clouds(reference_xyz,second_xyz,action,.02)
    mask=np.zeros(len(cloud.xyz),np.uint8);probability=np.zeros(len(cloud.xyz),np.float32)
    ids,first=np.unique(cloud.second_to_fused,return_index=True)
    mask[ids]=second_mask[first];probability[ids]=second_probability[first]
    # The representative in a shared voxel is a reference measurement.
    ids,first=np.unique(cloud.reference_to_fused,return_index=True)
    mask[ids]=reference_mask[first];probability[ids]=reference_probability[first]
    return dict(xyz=cloud.xyz,sound_mask=mask,point_probability=probability,
                reference_to_fused=cloud.reference_to_fused,second_to_fused=cloud.second_to_fused,
                view_membership=cloud.view_membership,second_only_voxel=cloud.view_membership==2)
