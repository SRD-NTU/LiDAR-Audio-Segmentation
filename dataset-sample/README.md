# AudioLiDAR-HM3D demonstration subset

This is a deterministic, class-stratified subset of the four-channel
AudioLiDAR-HM3D package. It has 100 training, 50 validation, and 50 test
observations across all 16 categories. Original train/validation/test scene
separation is preserved because samples are selected only within their original
split.

Each sample includes only `lidar.npz` and `audio.wav`, the two observation files
needed by the included method loaders. `lidar.npz` retains training labels and
evaluator fields. RIRs, per-sample metadata, checksums, previews, source assets,
and generation artifacts are deliberately omitted to keep the demonstration
package compact. `metadata/subset_manifest.json` lists every selected sample and
SHA-256 digest for its included files.

Use this directory as the dataset root:

```bash
export AUDIO_LIDAR_DATA_ROOT="$PWD/dataset-sample"
```

The selection was made with seed `20261010`. Regenerate it from the full dataset
with:

```bash
python tools/create_dataset_subset.py \
  --source /path/to/v1-expanded16-4channel --output dataset-sample
```

## Attribution and use

The audio recording attribution and licence information is retained in
`metadata/audio_manifest.json`. This subset does not assign a blanket licence
over the source recordings or HM3D-derived content. Review the original terms
and attribution before redistribution or use.
