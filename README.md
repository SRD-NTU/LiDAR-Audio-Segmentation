# LiDAR-Audio-Segmentation

Three implementations for sounding-object segmentation from a LiDAR point
cloud and synchronized four-channel audio:

- `Graph-Based/`: graph and acoustic-field model.
- `LiDAR-WM/`: latent acoustic world model.
- `Point-Bind-Based/`: spatial-audio raster baseline.

The repository includes a compact, ready-to-run dataset at `dataset-sample/`:
100 train, 50 validation, and 50 test observations. It preserves all 16 sound
categories, uses disjoint original splits, and has a deterministic selection
record in `dataset-sample/metadata/subset_manifest.json`.

## Create the environment

Python 3.10+ is required. The portable requirements install the three local
packages and their common dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
export AUDIO_LIDAR_DATA_ROOT="$PWD/dataset-sample"
```

Alternatively, use Conda:

```bash
conda env create -f environment/environment.yml
conda activate lidar-audio-segmentation
export AUDIO_LIDAR_DATA_ROOT="$PWD/dataset-sample"
```

For raw foundation-geometry features, install the optional CUDA dependencies
and follow the pinned third-party instructions:

```bash
python -m pip install -r environment/requirements-foundation.txt
```

See [environment/README.md](environment/README.md) for CUDA and platform notes.

## Smoke test

```bash
python LiDAR-WM/train.py --config LiDAR-WM/configs/smoke.yaml
```

The normal training commands are documented in each method directory. Generated
caches, checkpoints, and predictions are ignored by Git.

## Included dataset subset

`dataset-sample/` contains the model-required `lidar.npz` and `audio.wav`
files, plus manifests, class taxonomy, microphone calibration, and audio
attribution. It intentionally omits RIRs, previews, source assets, and
generation-only metadata. The LiDAR files retain labels for supervised
training and evaluation; those labels are never inference inputs.

The original recordings and HM3D-derived content retain their own terms.
Review [dataset-sample/README.md](dataset-sample/README.md) and the bundled
`metadata/audio_manifest.json` before redistributing or using the data.

## License and third-party components

Choose and add a license for your own contributions before publishing. The
Point-Bind license is preserved in `Point-Bind-Based/LICENSE`. Optional
Point-SAM, Sonata, and learned-suppression-3d code is not redistributed; see
[LiDAR-WM/third_party/README.md](LiDAR-WM/third_party/README.md).
