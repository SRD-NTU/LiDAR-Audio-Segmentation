#!/usr/bin/env python3
"""Create the deterministic, stratified AudioLiDAR demonstration subset.

The output contains only assets required by the model loaders: lidar.npz,
audio.wav, split manifests, taxonomy, calibration, and audio attribution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_COUNTS = {"train": 100, "val": 50, "test": 50}
ASSETS = ("lidar.npz", "audio.wav")


def select_rows(rows: list[dict[str, str]], count: int, split: str, seed: int) -> list[dict[str, str]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    categories = sorted(by_category)
    if count < len(categories):
        raise ValueError(f"{split}: requested {count} samples for {len(categories)} categories")
    base, remainder = divmod(count, len(categories))
    extra_order = categories[:]
    random.Random(f"{seed}:{split}:allocation").shuffle(extra_order)
    requested = {category: base + int(category in extra_order[:remainder]) for category in categories}
    chosen: list[dict[str, str]] = []
    for category in categories:
        candidates = sorted(by_category[category], key=lambda row: row["sample"])
        if requested[category] > len(candidates):
            raise ValueError(f"{split}/{category}: insufficient samples")
        chooser = random.Random(f"{seed}:{split}:{category}")
        chosen.extend(chooser.sample(candidates, requested[category]))
    return sorted(chosen, key=lambda row: row["sample"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Full v1-expanded16-4channel dataset root")
    parser.add_argument("--output", type=Path, default=Path("dataset-sample"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "metadata" / "samples.csv").is_file():
        raise FileNotFoundError(f"Not an AudioLiDAR dataset: {source}")
    permitted_existing = {output / "README.md"}
    if output.exists() and any(path not in permitted_existing for path in output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    requested = {"train": args.train, "val": args.val, "test": args.test}
    with (source / "metadata" / "samples.csv").open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
        fields = list(all_rows[0])
    selected: dict[str, list[dict[str, str]]] = {}
    for split, count in requested.items():
        selected[split] = select_rows([row for row in all_rows if row["split"] == split], count, split, args.seed)
    metadata = output / "metadata"
    metadata.mkdir(parents=True)
    for name in ("categories.json", "microphone_array.json", "audio_manifest.json"):
        shutil.copy2(source / "metadata" / name, metadata / name)
    with (metadata / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for split in requested for row in selected[split])
    files: dict[str, dict[str, dict[str, str | int]]] = {}
    for split, rows in selected.items():
        names = [row["sample"] for row in rows]
        (metadata / f"{split}.json").write_text(json.dumps(names, indent=2) + "\n")
        for row in rows:
            sample = row["sample"]
            destination = output / sample
            destination.mkdir(parents=True)
            files[sample] = {}
            for asset in ASSETS:
                origin = source / sample / asset
                target = destination / asset
                shutil.copy2(origin, target)
                files[sample][asset] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    summary = {
        "name": "AudioLiDAR-HM3D demonstration subset",
        "source_version": "0.3-four-microphone-portable",
        "selection_seed": args.seed,
        "counts": {split: len(rows) for split, rows in selected.items()},
        "category_counts": {split: dict(sorted(Counter(row["category"] for row in rows).items())) for split, rows in selected.items()},
        "included_assets": list(ASSETS),
        "omitted_assets": ["rir.npy", "metadata.json", "COMPLETE.json", "preview/"],
        "files": files,
    }
    (metadata / "subset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
