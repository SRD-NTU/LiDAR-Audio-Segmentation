"""Train and evaluate the spatial-audio LiDAR segmenter."""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data_v2 import AudioLidarV2Dataset, load_sample_v2
from .model_v2 import AudioLidarV2

THRESHOLDS = np.arange(.1, .901, .05).round(2)


def make_loader(root, split, cache, batch_size, workers, limit=0, augment=False):
    dataset = AudioLidarV2Dataset(root, split, cache, augment=augment)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def lovasz_hinge(logits, target, valid):
    """Per-sample Lovasz hinge: a convex surrogate for binary mask IoU."""
    losses = []
    for sample_logits, sample_target, sample_valid in zip(logits, target, valid):
        values = sample_logits[sample_valid]
        labels = sample_target[sample_valid]
        signs = 2 * labels - 1
        errors = 1 - values * signs
        errors, permutation = torch.sort(errors, descending=True)
        sorted_labels = labels[permutation]
        positives = sorted_labels.sum()
        intersection = positives - sorted_labels.cumsum(0)
        union = positives + (1 - sorted_labels).cumsum(0)
        gradient = 1 - intersection / union.clamp_min(1)
        if len(gradient) > 1:
            gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
        losses.append(torch.dot(F.relu(errors), gradient))
    return torch.stack(losses).mean()


def loss_function(logits, target, valid, pos_weight, dice_weight, gamma, lovasz_weight=0.0):
    logits = logits.float()
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if gamma:
        pt = torch.where(target > .5, probability, 1 - probability)
        bce = bce * (1 - pt).pow(gamma)
    weight = 1 + (pos_weight - 1) * target
    bce = (bce * weight * valid).sum() / valid.sum().clamp_min(1)
    probability = probability * valid
    intersection = (probability * target).sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1) / (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    total = bce + dice_weight * dice.mean()
    if lovasz_weight:
        total = total + lovasz_weight * lovasz_hinge(logits, target, valid)
    return total


def metric_result(counts, macro_sum, samples):
    tp, fp, fn = [int(x) for x in counts]
    return {"samples": samples, "tp": tp, "fp": fp, "fn": fn,
            "point_iou": tp / max(1, tp + fp + fn),
            "mean_iou": float(macro_sum / samples),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn)}


def evaluate(model, data, device, pos_weight, dice_weight, gamma, lovasz_weight=0.0, thresholds=THRESHOLDS):
    model.eval()
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    macro_sum = np.zeros(len(thresholds), dtype=np.float64)
    loss_sum = 0.0
    th = torch.tensor(thresholds, dtype=torch.float32, device=device)[:, None, None, None]
    with torch.inference_mode():
        for grid, wave, target, _ in data:
            grid, wave, target = (x.to(device, non_blocking=True) for x in (grid, wave, target))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(grid, wave)
            valid = grid[:, 4] > 0
            loss_sum += float(loss_function(logits, target, valid, pos_weight, dice_weight, gamma, lovasz_weight)) * len(grid)
            pred = (logits.float().sigmoid()[None] >= th) & valid[None]
            truth = (target[None] > .5) & valid[None]
            tp = (pred & truth).sum(dim=(2, 3))
            fp = (pred & ~truth).sum(dim=(2, 3))
            fn = (~pred & truth).sum(dim=(2, 3))
            sample_counts = torch.stack((tp, fp, fn), dim=2).cpu().numpy()
            counts += sample_counts.sum(axis=1)
            macro_sum += (sample_counts[:, :, 0] / np.maximum(1, sample_counts.sum(axis=2))).sum(axis=1)
    results = {str(float(t)): {"threshold": float(t),
                               **metric_result(counts[i], macro_sum[i], len(data.dataset))}
               for i, t in enumerate(thresholds)}
    best = max(results.values(), key=lambda x: x["mean_iou"])
    return {"loss": loss_sum / len(data.dataset), "best": best, "thresholds": results}


def save(path, model, optimizer, epoch, best_mean_iou, threshold, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
             "epoch": epoch, "best_mean_iou": best_mean_iou, "threshold": threshold,
             "config": vars(args)}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def restore(path, model, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def epoch_lr(epoch, epochs, lr, warmup):
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    fraction = (epoch - warmup) / max(1, epochs - warmup - 1)
    return lr * (.03 + .97 * (1 + math.cos(math.pi * fraction)) / 2)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA allocation")
    device = torch.device("cuda")
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "args": vars(args)}, default=str), flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = AudioLidarV2(audio_branch=not args.no_audio_branch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best, best_threshold = 0, 0.0, .5
    if args.resume:
        state = restore(args.resume, model, optimizer)
        start, best, best_threshold = state["epoch"], state["best_mean_iou"], state["threshold"]
    train_data = make_loader(args.root, "train", args.cache, args.batch_size, args.workers, args.train_limit, True)
    val_data = make_loader(args.root, "val", args.cache, args.batch_size, args.workers, args.val_limit)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    no_improve = 0
    for epoch in range(start, args.epochs):
        learning_rate = epoch_lr(epoch, args.epochs, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        running = 0.0
        for step, (grid, wave, target, _) in enumerate(train_data, 1):
            grid, wave, target = (x.to(device, non_blocking=True) for x in (grid, wave, target))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(grid, wave)
                loss = loss_function(logits, target, grid[:, 4] > 0,
                                     args.pos_weight, args.dice_weight, args.focal_gamma, args.lovasz_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach())
            if args.max_steps and step >= args.max_steps:
                break
        validation = evaluate(model, val_data, device, args.pos_weight, args.dice_weight,
                              args.focal_gamma, args.lovasz_weight)
        result = {"epoch": epoch + 1, "lr": learning_rate, "train_loss": running / step,
                  "val_loss": validation["loss"], "val": validation["best"]}
        print(json.dumps(result), flush=True)
        improved = validation["best"]["mean_iou"] > best + 1e-5
        if improved:
            best = validation["best"]["mean_iou"]
            best_threshold = validation["best"]["threshold"]
            no_improve = 0
            save(output / "best.pt", model, None, epoch + 1, best, best_threshold, args)
        else:
            no_improve += 1
        save(output / "last.pt", model, optimizer, epoch + 1, best, best_threshold, args)
        if no_improve >= args.patience:
            print(json.dumps({"early_stop": epoch + 1, "best_mean_iou": best}), flush=True)
            break


def load_for_inference(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AudioLidarV2(audio_branch=not state["config"]["no_audio_branch"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def eval_command(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_for_inference(args.checkpoint, device)
    config = state["config"]
    data = make_loader(args.root, args.split, args.cache, args.batch_size, args.workers, args.limit)
    threshold = args.threshold if args.threshold is not None else state["threshold"]
    result = evaluate(model, data, device, config["pos_weight"], config["dice_weight"],
                      config["focal_gamma"], config.get("lovasz_weight", 0.0), np.asarray([threshold]))
    print(json.dumps({"epoch": state["epoch"], "split": args.split, "loss": result["loss"],
                      **result["best"]}, indent=2), flush=True)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_for_inference(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else state["threshold"]
    names = json.loads((Path(args.root) / "metadata" / f"{args.split}.json").read_text())
    if args.sample:
        if args.sample not in names:
            raise ValueError("Sample is not in selected split")
        names = [args.sample]
    if args.limit:
        names = names[:args.limit]
    output = Path(args.output)
    with torch.inference_mode():
        for name in names:
            item = load_sample_v2(args.root, name, labeled=False)
            grid = torch.from_numpy(item["grid"]).unsqueeze(0).to(device)
            wave = torch.from_numpy(item["audio"]).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                probabilities = model(grid, wave).float().sigmoid()[0].cpu().numpy()
            ray_id = item["ray_id"]
            probability = probabilities[ray_id % 64, ray_id // 64]
            path = output / (name + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=probability.astype(np.float32),
                                sound_mask=(probability >= threshold).astype(np.uint8), ray_id=ray_id)
            print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "eval", "predict"):
        p = sub.add_parser(command)
        p.add_argument("--root", required=True, type=Path)
        p.add_argument("--cache", type=Path, default=Path("runs/audiolidar-v2/spatial-cache"))
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=2)
        if command == "train":
            p.add_argument("--output", required=True, type=Path)
            p.add_argument("--epochs", type=int, default=35)
            p.add_argument("--patience", type=int, default=10)
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--warmup", type=int, default=2)
            p.add_argument("--weight-decay", type=float, default=.01)
            p.add_argument("--pos-weight", type=float, default=15.0)
            p.add_argument("--dice-weight", type=float, default=1.0)
            p.add_argument("--focal-gamma", type=float, default=0.0)
            p.add_argument("--lovasz-weight", type=float, default=0.0)
            p.add_argument("--no-audio-branch", action="store_true")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--resume", type=Path)
            p.add_argument("--train-limit", type=int, default=0)
            p.add_argument("--val-limit", type=int, default=0)
            p.add_argument("--max-steps", type=int, default=0)
        elif command == "eval":
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--split", choices=("train", "val", "test"), default="val")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--threshold", type=float)
        else:
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--split", choices=("train", "val", "test"), default="test")
            p.add_argument("--output", required=True, type=Path)
            p.add_argument("--sample")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--threshold", type=float)
    args = parser.parse_args()
    {"train": train, "eval": eval_command, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()
