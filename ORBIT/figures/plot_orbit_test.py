"""ORBIT V1 test performance summary, read from ORBIT/reports/final_test.json (read-only)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT.parent / "LiDAR_Segmentation/ORBIT/reports/final_test.json").read_text()) \
    if (ROOT.parent / "LiDAR_Segmentation/ORBIT/reports/final_test.json").exists() \
    else json.loads(Path("/projects/soumyaratna/LiDAR_Segmentation/ORBIT/reports/final_test.json").read_text())
m = report["metrics"]["fused"]
n = m["n"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
fig.suptitle(f"ORBIT: V1 test performance ({n} samples)", fontsize=15, fontweight="bold")

# Aggregate metrics (ORBIT reports IoU statistics; precision/recall/F1 are not part of its results).
names = ["mIoU", "Median IoU", "IoU ≥ 0.5", "IoU > 0"]
values = [m["mean_iou"], m["median_iou"], m["fraction_iou_ge_half"], 1 - m["zero_iou_fraction"]]
bars = axes[0].bar(names, values, color="#2878b5")
axes[0].set_ylim(0, 0.8)
axes[0].set_ylabel("Score (fraction of samples or mean IoU)")
axes[0].set_title("Aggregate metrics")
axes[0].grid(axis="y", alpha=0.25)
for bar, value in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value * 100:.1f}%", ha="center", fontsize=10)

# Category mean IoU.
ordered = sorted(m["by_category"].items(), key=lambda item: item[1], reverse=True)
labels = [name for name, _ in ordered]
ious = [value for _, value in ordered]
axes[1].barh(labels[::-1], ious[::-1], color="#43a2ca")
axes[1].set_xlim(0, max(ious) * 1.16)
axes[1].set_xlabel("Mean IoU")
axes[1].set_title("mIoU by source category")
axes[1].grid(axis="x", alpha=0.25)
for y, value in enumerate(ious[::-1]):
    axes[1].text(value + 0.004, y, f"{value * 100:.1f}%", va="center", fontsize=8)

# Outcome distribution.
zero = round(m["zero_iou_fraction"] * n)
success = round(m["fraction_iou_ge_half"] * n)
partial = n - zero - success
outcome_labels = ["IoU = 0", "0 < IoU < 0.5", "IoU ≥ 0.5"]
outcome_values = [zero, partial, success]
colors = ["#d95f02", "#e6ab02", "#1b9e77"]
wedges, _ = axes[2].pie(outcome_values, colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
axes[2].legend(
    wedges,
    [f"{label}: {value} ({value / n * 100:.1f}%)" for label, value in zip(outcome_labels, outcome_values)],
    loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=9,
)
axes[2].set_title("Per-sample IoU outcomes")

fig.savefig(ROOT / "orbit_test_performance.png", dpi=200, bbox_inches="tight")
print("saved", ROOT / "orbit_test_performance.png", outcome_values)
