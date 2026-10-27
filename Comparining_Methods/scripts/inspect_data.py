#!/usr/bin/env python3
"""Audit the complete read-only dataset without loading audio or all point clouds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from evaluation.io import write_unchanged_or_new


def run(data_root: Path) -> dict:
    data = AudioLiDAR(data_root)
    rows = list(csv.DictReader((data_root / "metadata/samples.csv").open(newline="")))
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    result = {"samples": len(rows), "splits": {}, "scene_disjoint": True,
              "dry_audio_disjoint": True, "source_independent_original_menus": True,
              "source_dependent_scenes": [], "example": None}
    scene_sets, dry_sets = {}, {}
    for split in ("train", "val", "test"):
        part = by_split[split]
        scene_sets[split] = {r["scene"] for r in part}
        dry_sets[split] = {r["dry_sha256"] for r in part}
        result["splits"][split] = {
            "samples": len(part), "scenes": len(scene_sets[split]),
            "source_groups": len({(r["scene"], r["source"]) for r in part}),
            "categories": dict(sorted(Counter(r["category"] for r in part).items()))}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        result["scene_disjoint"] &= scene_sets[left].isdisjoint(scene_sets[right])
        result["dry_audio_disjoint"] &= dry_sets[left].isdisjoint(dry_sets[right])

    # Compare saved receiver poses across all source objects in each scene.
    for split in ("train", "val", "test"):
        for scene in sorted(scene_sets[split]):
            source_dirs = sorted(p for p in (data_root / split / scene).iterdir() if p.is_dir())
            baseline = None
            mismatches = 0
            for source_dir in source_dirs:
                poses = []
                for view in sorted(p for p in source_dir.iterdir() if p.is_dir() and p.name.isdigit()):
                    metadata = json.loads((view / "metadata.json").read_text())
                    # Rounded only to remove JSON float formatting differences.
                    poses.append((tuple(round(v, 5) for v in metadata["listener"]["position_world"]),
                                  tuple(round(v, 5) for v in metadata["listener"]["rotation_world"])))
                if baseline is None:
                    baseline = poses
                elif poses != baseline:
                    mismatches += 1
            if mismatches:
                result["source_independent_original_menus"] = False
                result["source_dependent_scenes"].append(
                    {"split": split, "scene": scene, "sources": len(source_dirs), "different_from_first_source": mismatches})
    sample = data.samples("train")[0]
    observation = data.observation(sample)
    target = data.target_for_evaluation(sample)
    result["example"] = {"sample": sample, "xyz_shape": list(observation["xyz"].shape),
                         "audio_shape": list(observation["audio"].shape),
                         "sample_rate": observation["sample_rate"],
                         "target_points": int(target["sound_mask"].sum()),
                         "candidate_instance_ids_in_reference": int(len(set(target["instance_id"].tolist()))),
                         "same_recording_future_views": len(data.recorded_views(sample))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dataset_audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Audit output must stay inside Comparing_Methods")
    report = run(args.data_root.resolve())
    write_unchanged_or_new(output, json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("samples", "scene_disjoint", "dry_audio_disjoint", "source_independent_original_menus", "example")}, indent=2))


if __name__ == "__main__":
    main()
