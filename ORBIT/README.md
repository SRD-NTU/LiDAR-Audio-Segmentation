# ORBIT: audio-prompted LiDAR segmentation

ORBIT predicts complete 3D object masks from XYZ LiDAR points, then uses
four-channel audio to select the sounding object. This directory contains the
model, training and evaluation code, tests, the small APEX support package used
by ORBIT, and the upstream Utonia source with its original license. Generated
features, datasets, checkpoints, and downloaded weights are excluded.

## Saved result

The frozen ensemble scored **47.10%** mean original-point foreground IoU on
768 validation observations and **37.77%** on 768 scene-disjoint TEST
observations. These are archived results from the full V2 dataset, not scores
on the included `dataset-sample/`. The corresponding records are in
[`evaluation/`](evaluation/); verify their counts, reported means, frozen
checkpoint identifiers, and evaluator hash with:

```bash
python ORBIT/scripts/verify_evaluation.py
```

The verifier checks saved records. It cannot rerun inference because model
checkpoints and pretrained weights are not distributed here. The matching raw
XYZ/WAV comparison covers one validation observation. Correct-point prompts,
perfect instance masks, and best-candidate oracles in the research notes are
privileged diagnostics, not deployed scores. The 60–80% target was not reached.

## Environment and tests

From the repository root, create the environment described in the top-level
README, then install ORBIT:

```bash
python -m pip install -e ORBIT --no-deps
export AUDIO_LIDAR_DATA_ROOT="$PWD/dataset-sample"
python -m unittest discover -s ORBIT/tests -v
```

Use an editable install from this checkout: the raw encoder adapters resolve
the sibling `LiDAR-WM/` directory and local third-party source by path.

The 14 tests cover original-point mapping, mask/selector gradients, candidate
padding, deterministic seeds, audio-gate scoring, audio-bank ordering, and
tetrahedral array direction. The included sample is useful for inspecting the input contract.
It is not the full benchmark used for the archived result.

## Inputs and preparation

`AUDIO_LIDAR_DATA_ROOT` selects a dataset containing `metadata/{train,val,test}.json`,
`metadata/categories.json`, `metadata/microphone_array.json`, and
`<split>/<scene>/<pose>/<observation>/{lidar.npz,audio.wav}`. The waveform must
have four channels at 44,100 Hz. LiDAR inference reads only `xyz`; labels are
used by training and evaluation functions after prediction.

The following local assets must be obtained separately for the frozen Utonia
method:

- Sonata and Utonia pretrained weights, plus the CLAP model.
- ORBIT mask, selector, APEX, audio-gate, spatial-adapter, and projection
  checkpoints matching `evaluation/final_method_frozen.json`.
- A compatible CUDA/PyTorch setup for the sparse encoders. See
  `../environment/requirements-foundation.txt` and Utonia's included setup.

Feature caches are generated outside Git. Set `ORBIT_GEOMETRY_CACHE` to the
Sonata cache directory if it differs from the default `cache/sonata-xyz-v2`.
For an observation-only audio bank, run the following after installing the
LiDAR-WM package and its optional foundation dependencies:

```bash
python ORBIT/scripts/build_audio_bank.py --split train
python ORBIT/scripts/build_audio_bank.py --split val
```

This stores ordered CLAP embeddings and SRP scores in
`ORBIT/cache/audio-bank/{train,val}.npz`. `ORBIT_AUDIO_BANK_DIR` selects another
directory. The full research training pipeline also needs per-observation
source metadata for spatial supervision, which the compact sample intentionally
omits. Run it on a complete, authorized V2 dataset. CLI entry points are under
`orbit/`; for example `python -m orbit.prepare --help` and
`python -m orbit.predict --help`. The optional `scripts/gpu.sbatch` accepts
cluster account, partition, and GPU flags through `sbatch`; it contains no
site-specific allocation.

## Scope and provenance

`apex/` and `gate0_3d.py` are the minimal support sources copied from this
project's original APEX and gate0 implementations. Their data roots were
adapted to this shareable repository; ORBIT's model and evaluator math was
preserved. `requirements-observed.txt` records the original working versions.
`RESEARCH_2026_10_12.md` and `CAMPAIGN_2026_10_12.txt` are historical notes;
paths to full experiment artifacts in them refer to the private working tree.

The V2 validation and TEST scenes are distinct, but Sonata and Utonia may have
HM3D pretraining overlap, and the benchmark reuses only two independent dry
audio clips per category. Do not claim generalization to fully unseen scenes or
new recordings from this evaluation alone.

No license has yet been selected for this project's own code. Utonia code
retains its Apache 2.0 license; Utonia and Sonata weights have separate
noncommercial terms. Review the dataset, checkpoint, and recording rights
before publishing or redistributing additional assets.
