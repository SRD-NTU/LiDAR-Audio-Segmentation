#!/usr/bin/env python3
"""Build explicit non-result rows until a full valid evaluation is available."""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.io import write_unchanged_or_new
status = json.loads((ROOT / "configs/method_status.json").read_text())
results = ROOT / "results"
fields = ("method", "variant", "protocol", "status", "test_examples", "mean_iou", "candidate_selection_accuracy", "ci95_low", "ci95_high", "reason")
for protocol, basename in (("passive", "cvpr_passive_comparison.csv"), ("active", "cvpr_active_comparison.csv")):
    file = StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for key, entry in status.items():
        if entry["protocol"] == protocol:
            writer.writerow({"method": entry["published_name"], "variant": entry["variant"],
                             "protocol": protocol, "status": entry["status"], "reason": entry["reason"]})
    if protocol == "active":
        writer.writerow({"method": "Acoustic-WM", "variant": "existing active selection",
                         "protocol": "active", "status": "not_run_on_matched_common_menu",
                         "reason": "No new matched frozen-backend evaluation was run in this workspace; archived evidence is a distinct protocol."})
    write_unchanged_or_new(results / basename, file.getvalue())
lines = ["# CVPR comparison status", "", "No publication-quality cross-dataset scores are available from the five released implementations. Empty CSV score fields are intentional.", "", "## Passive grounding", "", "| Method | Variant | Status |", "|---|---|---|"]
for entry in status.values():
    if entry["protocol"] == "passive":
        lines.append(f"| {entry['published_name']} | {entry['variant']} | {entry['status']} |")
lines += ["", "## Active grounding", "", "| Method | Variant | Status |", "|---|---|---|"]
for entry in status.values():
    if entry["protocol"] == "active":
        lines.append(f"| {entry['published_name']} | {entry['variant']} | {entry['status']} |")
lines.append("| Acoustic-WM | existing active selection | not_run_on_matched_common_menu |")
lines += ["", "The original dataset has source-conditioned receiver menus. The separate source-independent common-menu dataset is a different protocol and needs its own matched evaluation.", ""]
write_unchanged_or_new(results / "cvpr_main_table.md", "\n".join(lines))
print("Wrote status tables; no scores were fabricated.")
