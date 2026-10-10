# Graph-Based AudioLiDAR segmentation

This method constructs a graph over geometry-derived LiDAR nodes and fuses a
four-channel acoustic field, frozen geometry features, and CLAP audio features.
It predicts a sounding-object mask at the original LiDAR points.

## Requirements

Install the sibling `LiDAR-WM` package and this directory's requirements from
the repository root:

```bash
python -m pip install -e './LiDAR-WM[foundation,geometry]'
python -m pip install -r Graph-Based/requirements.txt
export AUDIO_LIDAR_DATA_ROOT=/absolute/path/to/dataset
```

`prepare.py` reads frozen Sonata geometry and CLAP caches produced by LiDAR-WM.
By default it finds them below the sibling `LiDAR-WM/` directory. Override those
locations when needed:

```bash
export LIDAR_WM_ROOT=/absolute/path/to/LiDAR-WM
export LIDAR_WM_GEOMETRY_CACHE="$LIDAR_WM_ROOT/cache/sonata-xyz-v2"
export LIDAR_WM_CLAP_CACHE="$LIDAR_WM_ROOT/cache/clap"
```

## Training

```bash
cd Graph-Based
python prepare.py --workers 2
python prepare_center_lut.py
python train_audio_semantic.py
python train_spatial_audio.py
python train_graph.py --device cuda --epochs 20 --run graph-world
```

Optional teacher-context training requires a compatible LiDAR-WM checkpoint at
`$LIDAR_WM_ROOT/runs/foundation-joint-control/best.pt`:

```bash
python prepare_teacher.py --device cuda
```

The training scripts write their regenerated caches and checkpoints below this
directory; those paths are ignored by Git.

## Inference

`predict_cached.py` evaluates a sample already prepared in the graph cache.
`predict.py` performs raw XYZ/four-channel-WAV inference and requires CUDA,
Sonata code/weights, a CLAP snapshot, and the selected Graph-Based checkpoint.
No trained checkpoints are included in this release.
