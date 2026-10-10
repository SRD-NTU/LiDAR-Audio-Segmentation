import numpy as np
from predict import project_xyz

def test_xyz_projection_preserves_sensor_rays():
    angles=np.asarray([0,2*np.pi/1800,2*np.pi*1799/1800])
    elevations=-np.pi/8+np.asarray([0,31,63])*np.pi/4/63
    xyz=np.stack((np.cos(angles)*np.cos(elevations),np.sin(angles)*np.cos(elevations),np.sin(elevations)),1).astype('float32')*3
    lidar,valid,row,col=project_xyz(xyz)
    np.testing.assert_array_equal(col,[0,1,1799])
    np.testing.assert_array_equal(row,[0,31,63])
    np.testing.assert_allclose(lidar[:3,row,col].T*20,xyz,atol=1e-6)

def test_narrow_target_direction_mass_survives_downsampling():
    import torch
    from lidarwm.losses import soft_azimuth_target
    mask=torch.zeros(1,1,64,1800);mask[:,:,30,4]=1
    target=soft_azimuth_target(mask,225)
    assert target[0,0]>.99
