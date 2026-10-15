#!/usr/bin/env python3
"""Gate 0 with a full 3D SRP-PHAT scorer, applied identically to any four-channel dataset.

Ground-truth whole instances are the proposals. The acoustic scorer returns a posterior over
directions on the sphere (Fibonacci grid). Each candidate instance is scored by the posterior
at its points. The top-1 instance is compared with the sounding instance.

Evaluator fields (instance_id, semantic_id, sound_mask, metadata) are used only to build
proposals and score predictions. The scorer never sees them. The test split is not read.

Usage:
    python gate0/gate0_3d.py --dataset dataset/v2-tetra4-4channel-full --name tetra --split val
"""
import argparse
import csv
import glob
import hashlib
import json
import multiprocessing
import os
import time

import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPEED = 343.0
N_DIRS = 2000
MIN_POINTS = 30
N_BOOT = 2000
BAND = (150.0, 3000.0)  # analysis band in Hz; set from --band in main

STRUCTURE = {"wall", "floor", "ceiling", "window", "column", "pillar", "beam",
             "stairs", "stair", "staircase", "railing", "balustrade", "banister",
             "arch", "archway", "doorframe", "door frame", "window frame"}
METHODS = ["mean", "centroid", "oracle_3d", "oracle_direction", "shuffled_audio", "mirrored_azimuth"]

_CTX = {}


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


DIRS = fibonacci_sphere(N_DIRS)
TREE = cKDTree(DIRS)


def srp_phat_3d(audio, rate, positions):
    """Six-pair SRP-PHAT over the sphere. Same sign convention and band as array_doa.py."""
    frequency, _, spectrum = stft(audio.T, fs=rate, nperseg=2048, noverlap=1024)
    valid = (frequency >= BAND[0]) & (frequency <= BAND[1])
    frequency, spectrum = frequency[valid], spectrum[:, valid]
    energy = np.sum(np.abs(spectrum) ** 2, axis=(0, 1))
    active = energy > max(energy.max() * 0.05, 1e-20)
    if not active.any():
        # No directional evidence. Zero distinguishes silence from a flat
        # nonzero posterior and lets observation-only inference reject it.
        return np.zeros(len(DIRS), np.float64)
    spectrum = spectrum[:, :, active]
    score = np.zeros(len(DIRS))
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            cross = spectrum[i] * spectrum[j].conj()
            cross /= np.maximum(np.abs(cross), 1e-12)
            cross = cross.mean(axis=1)
            delay = -(DIRS @ (positions[i] - positions[j])) / SPEED
            score += np.real(np.exp(2j * np.pi * delay[:, None] * frequency) @ cross) / len(frequency)
    score /= n * (n - 1) / 2
    post = np.exp((score - score.max()) / 0.05)
    post /= post.sum()
    return post / post.max()


def cache_fingerprint(dataset, dirs, array_json):
    """Changes whenever the array calibration, band, direction grid, or any sample's checksums change."""
    h = hashlib.sha256()
    with open(array_json, "rb") as f:
        h.update(f.read())
    h.update(json.dumps({"n_dirs": N_DIRS, "band": list(BAND)}).encode())
    for d in dirs:
        with open(os.path.join(dataset, d, "COMPLETE.json"), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def audio_job(d):
    audio, rate = sf.read(os.path.join(d, "audio.wav"), dtype="float64")
    assert audio.shape[1] == 4 and rate == 44100
    return srp_phat_3d(audio, rate, _CTX["positions"])


def load_names(dataset):
    with open(os.path.join(dataset, "metadata", "categories.json")) as f:
        return {int(v): k for k, v in json.load(f).items()}


def instance_scores(iid, xyz, post, true_dir, true_pos):
    ids, first, inv, counts = np.unique(iid, return_index=True, return_inverse=True, return_counts=True)
    u = unit(xyz)
    idx = TREE.query(u)[1]
    mean = np.bincount(inv, weights=post[idx]) / counts
    cx = np.bincount(inv, weights=xyz[:, 0]) / counts
    cy = np.bincount(inv, weights=xyz[:, 1]) / counts
    cz = np.bincount(inv, weights=xyz[:, 2]) / counts
    cent = post[TREE.query(unit(np.stack([cx, cy, cz], 1)))[1]]
    dist = np.linalg.norm(xyz - true_pos[None, :], axis=1)
    d3 = np.full(len(ids), np.inf)
    np.minimum.at(d3, inv, dist)
    ang = 1 - u @ true_dir
    da = np.full(len(ids), np.inf)
    np.minimum.at(da, inv, ang)
    return dict(ids=ids, first=first, counts=counts, mean=mean, centroid=cent, dist3=d3, angle=da)


def process(job):
    i, rel = job
    d = os.path.join(_CTX["dataset"], rel)
    post = _CTX["posts"][i]
    partner = _CTX["posts"][_CTX["partners"][i]]
    z = np.load(os.path.join(d, "lidar.npz"))
    xyz, iid, sem, mask = z["xyz"].astype(np.float64), z["instance_id"], z["semantic_id"], z["sound_mask"].astype(bool)
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    p = np.array(meta["relative_source"]["position_lidar"], dtype=float)
    true_dir = p / np.linalg.norm(p)
    sound_inst = int(meta["source"]["instance_id"])
    if not mask.any():
        raise ValueError(f"Empty sound_mask in {rel}: the IoU convention for empty targets must be defined before scoring")
    u, c = np.unique(iid[mask], return_counts=True)
    purity = float(c.max() / mask.sum())
    own = instance_scores(iid, xyz, post, true_dir, p)
    ids, counts = own["ids"], own["counts"]
    sound_count = int(counts[ids == sound_inst][0])
    inter = int((mask & (iid == sound_inst)).sum())
    iou = inter / (mask.sum() + sound_count - inter)
    names = _CTX["names"]
    structural = np.array([names.get(int(s), "") in STRUCTURE for s in sem[own["first"]]])
    valid = (ids > 0) & (counts >= MIN_POINTS)
    sets = {"primary": valid & ~structural, "all": valid}
    flip = xyz.copy()
    flip[:, 1] *= -1
    fdir = true_dir.copy()
    fdir[1] *= -1
    controls = {
        "mean": own, "centroid": own,
        "oracle_3d": own, "oracle_direction": own,
        "shuffled_audio": instance_scores(iid, xyz, partner, true_dir, p),
        "mirrored_azimuth": instance_scores(iid, flip, post, fdir, p * np.array([1, -1, 1])),
    }
    key = {"mean": "mean", "centroid": "centroid", "oracle_3d": "dist3", "oracle_direction": "angle",
           "shuffled_audio": "mean", "mirrored_azimuth": "mean"}
    peak = DIRS[int(np.argmax(post))]
    peak_err = float(np.rad2deg(np.arccos(np.clip(peak @ true_dir, -1, 1))))
    rec = dict(sample=rel, scene=meta["scene_id"], category=meta["source"]["category"],
               sound_instance=sound_inst, sound_purity=purity, iou_true=iou, peak_error_deg=peak_err,
               distance_m=float(meta["relative_source"]["distance_m"]))
    for set_name, cand in sets.items():
        rec[f"K_{set_name}"] = int(cand.sum())
        is_c = bool(cand[ids == sound_inst].any())
        rec[f"candidate_{set_name}"] = int(is_c)
        idx = np.where(cand)[0]
        for m in METHODS:
            sc = controls[m][key[m]]
            if not is_c:
                pred = -1
            elif key[m] in ("dist3", "angle"):
                pred = int(ids[idx[np.argmin(sc[idx])]])
            else:
                pred = int(ids[idx[np.argmax(sc[idx])]])
            rec[f"correct_{m}_{set_name}"] = int(pred == sound_inst)
    return rec


def choose_partners(scenes, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(len(scenes), dtype=int)
    for i, s in enumerate(scenes):
        choices = [j for j, t in enumerate(scenes) if t != s]
        out[i] = choices[rng.integers(len(choices))]
    return out


def boot(values, scenes, rng):
    vals = np.asarray(values, float)
    by = {s: np.where(np.asarray(scenes) == s)[0] for s in sorted(set(scenes))}
    keys = list(by)
    stats = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([by[keys[k]] for k in pick])
        stats.append(vals[idx].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(recs, tiers):
    rng = np.random.default_rng(0)
    out = {"n": len(recs), "empty_targets": 0, "empty_target_convention": "not applicable: scoring raises on an empty sound_mask",
           "scenes": len({r["scene"] for r in recs}),
           "iou_perfect_selection": float(np.mean([r["iou_true"] for r in recs])),
           "peak_error_deg_median": float(np.median([r["peak_error_deg"] for r in recs])),
           "peak_error_deg_p90": float(np.percentile([r["peak_error_deg"] for r in recs], 90)),
           "sound_purity_min": float(min(r["sound_purity"] for r in recs)), "sets": {}}
    scenes = [r["scene"] for r in recs]
    for set_name in ["primary", "all"]:
        s = {"K_median": float(np.median([r[f"K_{set_name}"] for r in recs])),
             "chance": float(np.mean([1 / max(r[f"K_{set_name}"], 1) for r in recs])), "methods": {}}
        for m in METHODS:
            corr = np.array([r[f"correct_{m}_{set_name}"] for r in recs], float)
            lo, hi = boot(corr, scenes, rng)
            s["methods"][m] = {"top1": float(corr.mean()), "ci95": [lo, hi],
                               "expected_iou": float(np.mean(corr * np.array([r["iou_true"] for r in recs])))}
        out["sets"][set_name] = s
    out["tiers"] = {}
    by_sample = {r["sample"]: r for r in recs}
    for tier, samples in tiers.items():
        rs = [by_sample[x] for x in samples if x in by_sample]
        if not rs:
            continue
        out["tiers"][tier] = {"n": len(rs), "scenes": len({r["scene"] for r in rs}),
                              "top1_mean": float(np.mean([r["correct_mean_primary"] for r in rs])),
                              "top1_oracle_direction": float(np.mean([r["correct_oracle_direction_primary"] for r in rs])),
                              "expected_iou_mean": float(np.mean([r["correct_mean_primary"] * r["iou_true"] for r in rs]))}
    return out


def write_md(path, name, split, summary, elapsed):
    p = summary["sets"]["primary"]["methods"]
    e = p["mean"]["expected_iou"]
    lines = [f"# Gate 0 (3D SRP-PHAT): {name}, {split}, band {int(BAND[0])}-{int(BAND[1])} Hz", "",
             f"{summary['n']} samples, {summary['scenes']} scenes. Runtime {elapsed:.0f} s. "
             "Ground-truth whole-instance proposals; no learned model.", "",
             "## Verdict", "",
             ("**PASS:** selection alone does not rule out 80% mean IoU." if e >= 0.80 else
              f"**FAIL:** expected mean IoU with perfect proposals is {e:.3f}, below 0.80."), "",
             "## Selection methods (primary candidate set)", "",
             "| Method | Top-1 | 95% CI (scene bootstrap) | Expected mean IoU |", "|---|---|---|---|"]
    labels = {"mean": "3D SRP-PHAT, mean over instance points (primary)",
              "centroid": "3D SRP-PHAT at instance centroid",
              "oracle_3d": "Oracle: nearest instance in 3D to true source (not achievable from audio)",
              "oracle_direction": "Oracle: instance with a point nearest the true direction",
              "shuffled_audio": "Control: audio from another scene",
              "mirrored_azimuth": "Control: mirrored azimuth"}
    for m in METHODS:
        r = p[m]
        lines.append(f"| {labels[m]} | {r['top1']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['expected_iou']:.3f} |")
    lines += ["", f"Chance top-1: {summary['sets']['primary']['chance']:.3f}. "
              f"Median candidates: {summary['sets']['primary']['K_median']:.0f}.", "",
              f"Global peak error to true direction: median {summary['peak_error_deg_median']:.1f}°, "
              f"p90 {summary['peak_error_deg_p90']:.1f}°.", "",
              "## By difficulty tier (primary set, 3D SRP-PHAT)", "",
              "| Tier | n | Scenes | Top-1 | Expected mean IoU |", "|---|---|---|---|---|"]
    for tier in ["easy", "medium", "hard"]:
        if tier in summary["tiers"]:
            t = summary["tiers"][tier]
            lines.append(f"| {tier} | {t['n']} | {t['scenes']} | {t['top1_mean']:.3f} | {t['expected_iou_mean']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--tiers", default=os.path.join(HERE, "tiers", "tiers_val.json"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--band", type=float, nargs=2, default=[150.0, 3000.0], metavar=("LO", "HI"))
    args = ap.parse_args()
    global BAND
    BAND = (args.band[0], args.band[1])
    out = os.path.join(HERE, "reports")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(HERE, "cache")
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()

    dataset = os.path.abspath(args.dataset)
    with open(os.path.join(dataset, "metadata", "microphone_array.json")) as f:
        positions = np.asarray(json.load(f)["positions_lidar_m"], dtype=float)
    dirs = sorted(os.path.relpath(os.path.dirname(p), dataset) for p in
                  glob.glob(os.path.join(dataset, args.split, "*", "*", "*", "lidar.npz")))
    scenes = [os.path.basename(os.path.dirname(os.path.dirname(d))) for d in dirs]
    print(f"{args.name} {args.split}: {len(dirs)} samples, {len(set(scenes))} scenes", flush=True)

    _CTX.update(positions=positions, dataset=dataset, names=load_names(dataset))
    fp = cache_fingerprint(dataset, dirs, os.path.join(dataset, "metadata", "microphone_array.json"))
    post_cache = os.path.join(cache, f"post3d_{args.name}_{args.split}_{int(BAND[0])}-{int(BAND[1])}_{fp}.npy")
    ctx = multiprocessing.get_context("fork")
    if os.path.exists(post_cache):
        posts = np.load(post_cache)
    else:
        with ctx.Pool(args.workers) as pool:
            posts = np.stack(pool.map(audio_job, [os.path.join(dataset, d) for d in dirs], chunksize=4))
        np.save(post_cache, posts)
    print(f"audio done {time.time() - t0:.0f}s", flush=True)

    _CTX.update(posts=posts, partners=choose_partners(scenes))
    with ctx.Pool(args.workers) as pool:
        recs = pool.map(process, list(enumerate(dirs)), chunksize=4)
    print(f"scoring done {time.time() - t0:.0f}s", flush=True)

    with open(args.tiers) as f:
        tiers = json.load(f)["tiers"]
    summary = summarize(recs, tiers)
    base = os.path.join(out, f"gate0_3d_{args.name}_{args.split}_band{int(BAND[0])}-{int(BAND[1])}")
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(base + "_samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    write_md(base + ".md", args.name, args.split, summary, time.time() - t0)
    print(f"done -> {base}.md", flush=True)


if __name__ == "__main__":
    main()
