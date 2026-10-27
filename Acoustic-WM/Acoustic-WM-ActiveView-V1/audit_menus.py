"""Audit whether recorded receiver menus are identical across scene sources."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE / "code"))

import numpy as np

from acoustic_wm.v1_dataset import calibrated_pose


def audit(data_root, split):
    scenes = sorted(p for p in (data_root / split).iterdir() if p.is_dir())
    report = []
    total_sources = 0
    total_observations = 0
    for scene in scenes:
        baseline = None
        baseline_valid = None
        source_count = 0
        mismatches = []
        for source in sorted(p for p in scene.iterdir() if p.is_dir()):
            views = sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())
            if not views:
                continue
            poses = np.stack([calibrated_pose(json.loads((p / "metadata.json").read_text()))
                              for p in views])
            positions = poses[:, :3, 3]
            valid = np.linalg.norm(positions[:, None] - positions[None], axis=-1) <= 3.0
            if baseline is None:
                baseline, baseline_valid = poses, valid
            elif poses.shape != baseline.shape or not np.allclose(poses, baseline, atol=1e-5) or not np.array_equal(valid, baseline_valid):
                mismatches.append(source.name)
            source_count += 1
            total_observations += len(views)
        total_sources += source_count
        report.append({"scene": scene.name, "sources": source_count,
                       "views_per_source": int(len(baseline)) if baseline is not None else 0,
                       "mismatched_sources": mismatches})
    return {"split": split, "scenes": len(report), "sources": total_sources,
            "observations": total_observations,
            "source_independent": all(not row["mismatched_sources"] for row in report),
            "scene_details": report,
            "limitation": "This checks saved receiver poses and local validity. It does not certify arbitrary external menus or acoustic independence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT / ".importants/Experiments_20261024/generated/common")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/menu_audit.json")
    args = parser.parse_args()
    result = audit(args.data_root.resolve(), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("split", "scenes", "sources", "observations", "source_independent")}))
    if not result["source_independent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
