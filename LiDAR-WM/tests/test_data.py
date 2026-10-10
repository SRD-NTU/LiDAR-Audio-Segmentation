from pathlib import Path
import os

import numpy as np

from lidarwm.data import AudioLidarDataset


DATA_ROOT = Path(os.environ.get("AUDIO_LIDAR_DATA_ROOT", "data"))


def test_real_dataset_contract():
    dataset = AudioLidarDataset(DATA_ROOT, "val", max_samples=1)
    item = dataset[0]
    assert item["lidar"].shape == (7, 64, 1800)
    assert item["valid"].shape == (1, 64, 1800)
    assert item["audio"].shape == (4, 29400)
    assert item["mask"].sum() > 0
    assert item["semantic"].shape == (64, 1800)
    assert (item["semantic"] != 255).sum() == item["valid"].sum()
    assert int(item["valid"].sum()) == len(item["ray_id"])


def test_instance_masks_are_annotation_only_targets():
    item=AudioLidarDataset(DATA_ROOT,'val',max_samples=1,instance_masks=True)[0]
    assert item['instance_map'].shape==(64,1800)
    assert item['instance_map'].max()>0
    assert ((item['instance_map']>0)&(item['valid'][0]==0)).sum()==0
    assert (item['instance_map'][item['mask'][0]>0]>0).all()
