# LiDAR-WM

LiDAR-WM is a latent acoustic world model for sounding-object segmentation from
an observed LiDAR point cloud and synchronized four-channel audio. It does not
use RGB, semantic IDs, instance IDs, source coordinates, RIRs, or labels as
inference inputs.

## Install

From the repository root:

```bash
python -m pip install -e './LiDAR-WM[foundation,geometry,diagnostics]'
export AUDIO_LIDAR_DATA_ROOT=/absolute/path/to/dataset
```

Every supplied YAML configuration reads its data root from
`$AUDIO_LIDAR_DATA_ROOT`. A command-line override is also available, for example
`--set data.root=/absolute/path/to/dataset`.

## Train and evaluate

```bash
cd LiDAR-WM
python train.py --config configs/smoke.yaml
python train.py --config configs/base.yaml
python evaluate.py --config configs/base.yaml --checkpoint runs/lawm-semantic/best.pt --split val
python infer.py --config configs/base.yaml --checkpoint runs/lawm-semantic/best.pt \
  --split test --sample 0 --output prediction.npz
```

The foundation-world variants additionally need the Sonata source and a
separately obtained compatible checkpoint. CLAP preparation needs Transformers
and downloads or a locally provided model snapshot. See
[`third_party/README.md`](third_party/README.md); no weights are distributed in
this repository.

## Optional external methods

The Point-SAM adapter and LS3D suppression adapter load their upstream source
only when used. Clone the pinned upstream repositories following the third-party
instructions. Their code and licenses remain separate from this source release.
