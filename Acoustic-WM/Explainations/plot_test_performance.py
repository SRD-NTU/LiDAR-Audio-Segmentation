#!/usr/bin/env python3
"""Create acoustic frontend-style test-performance plots from the released predictions."""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = Path("/projects/soumyaratna/LiDAR_Segmentation/dataset/v2-tetra4-4channel-full")
SCALE = 2
WIDTH, HEIGHT = 2400, 900


def scaled(value: float) -> int:
    return round(value * SCALE)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", scaled(size))


F_TITLE = font(28, True)
F_PANEL = font(21)
F_LABEL = font(16)
F_SMALL = font(13)
F_TICK = font(15)


def read_npy_scalar(archive: Path, member: str) -> float:
    """Read a numeric scalar from an NPY member without requiring NumPy."""
    with zipfile.ZipFile(archive) as handle:
        payload = handle.read(member)
    if payload[:6] != b"\x93NUMPY":
        raise ValueError(f"Invalid NPY payload in {archive}:{member}")
    major = payload[6]
    if major == 1:
        header_size = struct.unpack("<H", payload[8:10])[0]
        data_start = 10 + header_size
    else:
        header_size = struct.unpack("<I", payload[8:12])[0]
        data_start = 12 + header_size
    header_start = 10 if major == 1 else 12
    header = ast.literal_eval(payload[header_start:data_start].decode("latin1").strip())
    if header["shape"] != ():
        raise ValueError(f"Expected scalar {member} in {archive}, got {header['shape']}")
    formats = {"<f8": "<d", "<f4": "<f", "<i8": "<q", "<i4": "<i"}
    try:
        fmt = formats[header["descr"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported scalar dtype {header['descr']}") from exc
    return float(struct.unpack_from(fmt, payload, data_start)[0])


def sample_categories() -> dict[str, str]:
    result: dict[str, str] = {}
    with (DATASET / "metadata/samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                result[row["sample"]] = row["category"]
    return result


def load_results(release: str, categories: dict[str, str]) -> list[tuple[str, str, float]]:
    directory = ROOT / release / "outputs/predictions/test"
    rows = []
    for path in sorted(directory.glob("*/*/*.npz")):
        sample = "test/" + path.relative_to(directory).with_suffix("").as_posix()
        rows.append((sample, categories[sample], read_npy_scalar(path, "iou.npy")))
    if len(rows) != 768:
        raise ValueError(f"Expected 768 test predictions for {release}, found {len(rows)}")
    reported = json.loads((ROOT / release / "outputs/metrics.json").read_text())["test_iou"]
    calculated = math.fsum(row[2] for row in rows) / len(rows)
    # ActiveView's summary was accumulated in float32, while this script sums
    # the stored scalar IoUs in double precision.
    if not math.isclose(calculated, reported, abs_tol=1e-7):
        raise ValueError(f"Computed mIoU {calculated} differs from reported {reported}")
    return rows


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str,
                text_font: ImageFont.FreeTypeFont, fill: str = "#1f1f1f") -> None:
    draw.text((scaled(xy[0]), scaled(xy[1])), value, font=text_font, fill=fill,
              anchor="mm")


def rotated_text(image: Image.Image, xy: tuple[float, float], value: str,
                 text_font: ImageFont.FreeTypeFont, angle: int = 90) -> None:
    box = text_font.getbbox(value)
    layer = Image.new("RGBA", (box[2] - box[0] + scaled(12), box[3] - box[1] + scaled(12)), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((scaled(6) - box[0], scaled(6) - box[1]), value, font=text_font, fill="#1f1f1f")
    layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(layer, (scaled(xy[0]) - layer.width // 2, scaled(xy[1]) - layer.height // 2))


def axes(draw: ImageDraw.ImageDraw, left: float, top: float, right: float, bottom: float) -> None:
    draw.rectangle((scaled(left), scaled(top), scaled(right), scaled(bottom)), outline="#222222", width=scaled(1))


def aggregate_panel(image: Image.Image, values: list[float]) -> None:
    draw = ImageDraw.Draw(image)
    text_center(draw, (380, 92), "Aggregate metrics", F_PANEL)
    left, top, right, bottom = 76, 119, 690, 756
    limit = 0.8
    for i in range(9):
        value = i / 10
        y = bottom - value / limit * (bottom - top)
        draw.line((scaled(left), scaled(y), scaled(right), scaled(y)), fill="#d5d5d5", width=scaled(1))
        draw.text((scaled(left - 11), scaled(y)), f"{value:.1f}", font=F_TICK, fill="#202020", anchor="rm")
    axes(draw, left, top, right, bottom)
    labels = ["mIoU", "Median IoU", "IoU ≥ 0.5", "IoU > 0"]
    metrics = [
        math.fsum(values) / len(values),
        statistics.median(values),
        sum(value >= 0.5 for value in values) / len(values),
        sum(value > 0 for value in values) / len(values),
    ]
    slot = (right - left) / len(metrics)
    bar_width = slot * 0.78
    for index, (label, value) in enumerate(zip(labels, metrics)):
        cx = left + slot * (index + 0.5)
        y = bottom - value / limit * (bottom - top)
        draw.rectangle((scaled(cx - bar_width / 2), scaled(y), scaled(cx + bar_width / 2), scaled(bottom)), fill="#337fb6")
        draw.text((scaled(cx), scaled(y - 7)), f"{value:.1%}", font=F_LABEL, fill="#202020", anchor="ms")
        draw.text((scaled(cx), scaled(bottom + 13)), label, font=F_LABEL, fill="#202020", anchor="ma")
    rotated_text(image, (22, (top + bottom) / 2), "Score (fraction of samples or mean IoU)", F_LABEL)


def category_panel(image: Image.Image, grouped: dict[str, list[float]]) -> None:
    draw = ImageDraw.Draw(image)
    text_center(draw, (1175, 92), "mIoU by source category", F_PANEL)
    ranked = sorted(((name, math.fsum(vals) / len(vals)) for name, vals in grouped.items()), key=lambda item: item[1], reverse=True)
    left, top, right, bottom = 867, 119, 1480, 756
    limit = max(0.8, math.ceil(max(value for _, value in ranked) * 10) / 10)
    for i in range(round(limit * 10) + 1):
        value = i / 10
        x = left + value / limit * (right - left)
        draw.line((scaled(x), scaled(top), scaled(x), scaled(bottom)), fill="#d5d5d5", width=scaled(1))
        draw.text((scaled(x), scaled(bottom + 13)), f"{value:.1f}", font=F_TICK, fill="#202020", anchor="ma")
    axes(draw, left, top, right, bottom)
    row_height = (bottom - top) / len(ranked)
    bar_height = row_height * 0.78
    for index, (name, value) in enumerate(ranked):
        cy = top + row_height * (index + 0.5)
        endpoint = left + value / limit * (right - left)
        draw.rectangle((scaled(left), scaled(cy - bar_height / 2), scaled(endpoint), scaled(cy + bar_height / 2)), fill="#49a6cc")
        draw.text((scaled(left - 8), scaled(cy)), name, font=F_LABEL, fill="#202020", anchor="rm")
        draw.text((scaled(endpoint + 7), scaled(cy)), f"{value:.1%}", font=F_SMALL, fill="#202020", anchor="lm")
    text_center(draw, ((left + right) / 2, 810), "Mean IoU", F_LABEL)


def pie_panel(image: Image.Image, values: list[float]) -> None:
    draw = ImageDraw.Draw(image)
    text_center(draw, (1905, 92), "Per-sample IoU outcomes", F_PANEL)
    counts = [sum(value == 0 for value in values), sum(0 < value < 0.5 for value in values), sum(value >= 0.5 for value in values)]
    colors = ["#e66100", "#edaf00", "#1b9e77"]
    box = tuple(map(scaled, (1624, 165, 2198, 739)))
    start = 90.0
    for count, color in zip(counts, colors):
        extent = count / len(values) * 360
        draw.pieslice(box, start=round(start), end=round(start + extent), fill=color, outline="white", width=scaled(2))
        start += extent
    legend_left, legend_top, legend_right, legend_bottom = 1655, 775, 2315, 887
    draw.rounded_rectangle(tuple(map(scaled, (legend_left, legend_top, legend_right, legend_bottom))), radius=scaled(4), fill="white", outline="#d4d4d4", width=scaled(1))
    labels = ["IoU = 0", "0 < IoU < 0.5", "IoU ≥ 0.5"]
    for index, (label, count, color) in enumerate(zip(labels, counts, colors)):
        cy = legend_top + 22 + index * 32
        draw.rectangle(tuple(map(scaled, (legend_left + 16, cy - 7, legend_left + 48, cy + 7))), fill=color)
        draw.text((scaled(legend_left + 59), scaled(cy)), f"{label}: {count} ({count / len(values):.1%})", font=F_SMALL, fill="#202020", anchor="lm")


def render(release: str, output: str, rows: list[tuple[str, str, float]]) -> None:
    image = Image.new("RGBA", (scaled(WIDTH), scaled(HEIGHT)), "white")
    draw = ImageDraw.Draw(image)
    title = release.replace("Acoustic-WM-", "Acoustic-WM ")
    text_center(draw, (1200, 34), f"{title}: test performance ({len(rows)} samples)", F_TITLE, "#111111")
    values = [row[2] for row in rows]
    grouped: dict[str, list[float]] = defaultdict(list)
    for _, category, value in rows:
        grouped[category].append(value)
    aggregate_panel(image, values)
    category_panel(image, grouped)
    pie_panel(image, values)
    image = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image.save(HERE / output, quality=95, dpi=(180, 180))


def main() -> None:
    categories = sample_categories()
    releases = [
        ("Acoustic-WM-SingleView", "acoustic_wm_singleview_test_performance.png"),
        ("Acoustic-WM-ActiveView", "acoustic_wm_activeview_test_performance.png"),
    ]
    for release, output in releases:
        rows = load_results(release, categories)
        render(release, output, rows)
        values = [row[2] for row in rows]
        print(f"{output}: mIoU={statistics.fmean(values):.6f}, median={statistics.median(values):.6f}, "
              f"IoU>=0.5={sum(v >= 0.5 for v in values)}, IoU>0={sum(v > 0 for v in values)}")


if __name__ == "__main__":
    main()
