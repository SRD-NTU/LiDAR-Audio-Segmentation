# Point-Bind-Based AudioLiDAR segmentation

This spatial-audio baseline rasterizes LiDAR geometry and derives a label-free
SRP-PHAT likelihood from the synchronized four-channel waveform. The model then
predicts a sounding-object probability for each original LiDAR point.

## Install

```bash
python -m pip install -e ./Point-Bind-Based
export AUDIO_LIDAR_DATA_ROOT=/absolute/path/to/dataset
```

Use a CUDA-enabled PyTorch build for training. The source itself does not assume
a specific cluster, GPU type, account, or Python installation.

## Train, evaluate, and predict

```bash
cd Point-Bind-Based
python -m audiolidar_seg.precompute_spatial \
  --root "$AUDIO_LIDAR_DATA_ROOT" --output runs/spatial-cache

python -m audiolidar_seg.run_v2 train \
  --root "$AUDIO_LIDAR_DATA_ROOT" --cache runs/spatial-cache \
  --output runs/model --epochs 30 --batch-size 2

python -m audiolidar_seg.run_v2 eval \
  --root "$AUDIO_LIDAR_DATA_ROOT" --cache runs/spatial-cache \
  --checkpoint runs/model/best.pt --split val

python -m audiolidar_seg.run_v2 predict \
  --root "$AUDIO_LIDAR_DATA_ROOT" --checkpoint runs/model/best.pt \
  --output predictions --split test
```

The spatial cache is needed for training/evaluation but prediction derives its
spatial audio feature directly from each `audio.wav`. Checkpoints, caches, and
predictions are excluded through the repository-level `.gitignore`.
