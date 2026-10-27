"""Train V1 instance verification on recorded branching observations.

The V0 frontend/selector remain frozen. The world model and verifier are
fine-tuned on TRAIN only; checkpoint selection uses VAL physical-instance NLL.
Ground-truth instance IDs are read here solely to construct supervision.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from acoustic_wm.action_protocol import mask_groups
from acoustic_wm.model import ReleasedModel
from acoustic_wm.v1_belief import JointBelief
from acoustic_wm.v1_verifier import InstanceVerifier
from freeze import verify_source_manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_action(first, second):
    """Convert two reference-relative receiver poses to a first-to-second action."""
    ra = first[3:].reshape(3, 3)
    rb = second[3:].reshape(3, 3)
    delta = (second[:3] - first[:3]) @ ra
    return np.concatenate([delta, (ra.T @ rb).reshape(-1)]).astype(np.float32)


def majority_instance(mask, inverse, instance_ids):
    selected = instance_ids[mask[inverse]]
    selected = selected[selected >= 0]
    if not len(selected):
        return -1, 0.0
    ids, counts = np.unique(selected, return_counts=True)
    winner = int(counts.argmax())
    return int(ids[winner]), float(counts[winner] / len(selected))


def prepare_sample(sample, data_root, cache_root, max_candidates, min_positive_iou):
    if (cache_root / "skipped" / (sample + ".json")).is_file():
        return None
    reference = cache_root / "reference" / (sample + ".npz")
    controlled = cache_root / "controlled" / (sample + ".npz")
    if not reference.is_file() or not controlled.is_file():
        raise FileNotFoundError(f"Missing prepared branch or reference: {sample}")
    with np.load(reference, allow_pickle=False) as z:
        ref = {key: z[key] for key in ("masks", "inverse", "hypotheses")}
    with np.load(controlled, allow_pickle=False) as z:
        branch = {key: z[key] for key in ("base", "sites", "iou", "actions", "valid",
                                          "observed", "future_observed", "weight",
                                          "reference_world_points", "reference_world_features",
                                          "reference_world_valid")}
    metadata = json.loads((data_root / sample / "metadata.json").read_text())
    source_id = int(metadata["source"]["instance_id"])
    with np.load(data_root / sample / "lidar.npz", allow_pickle=False) as z:
        instance_ids = z["instance_id"].astype(np.int64)
    if not bool(branch["valid"].any()) or not (instance_ids == source_id).any():
        return None
    quality = branch["iou"]
    positives = np.flatnonzero(quality >= min_positive_iou)
    negatives = np.flatnonzero(quality <= 0.1)
    positives = positives[np.argsort(-quality[positives])][:min(4, max_candidates // 2)]
    negatives = negatives[np.argsort(-branch["base"][negatives])][:max_candidates * 3]
    chosen = []
    physical = []
    for k in np.concatenate([positives, negatives]):
        identity, purity = majority_instance(ref["masks"][k], ref["inverse"], instance_ids)
        if purity < 0.6:
            continue
        if k in positives and identity == source_id:
            chosen.append(int(k)); physical.append(identity)
        elif k in negatives and identity != source_id:
            chosen.append(int(k)); physical.append(identity)
        if len(chosen) >= max_candidates:
            break
    if not chosen or source_id not in physical or all(value == source_id for value in physical):
        return None
    chosen = np.asarray(chosen, np.int64)
    groups, duplicates, _ = mask_groups(ref["masks"][chosen], 0.9)
    return dict(sample=sample, source_id=source_id, candidate_ids=chosen,
                physical=np.asarray(physical, np.int64), groups=groups,
                duplicates=duplicates, base=branch["base"][chosen],
                log_site=branch["sites"][chosen], sites=ref["hypotheses"][chosen],
                geometry=(branch["reference_world_points"],
                          branch["reference_world_features"],
                          branch["reference_world_valid"]),
                observed=branch["observed"], future=branch["future_observed"],
                reliability=branch["weight"], actions=branch["actions"],
                valid=branch["valid"])


def complex_tensor(array, device):
    return torch.view_as_complex(torch.as_tensor(array, device=device).contiguous())


def episode_loss(world, verifier, item, action_indices, device):
    base = torch.as_tensor(item["base"], device=device)
    log_site = torch.as_tensor(item["log_site"], device=device)
    belief = JointBelief.from_v0(base, log_site, item["groups"], item["duplicates"])
    sites = torch.as_tensor(item["sites"], device=device)
    points, features, valid = [torch.as_tensor(x, device=device)[None] for x in item["geometry"]]
    k, h, _ = sites.shape
    initial = complex_tensor(item["observed"], device)
    initial_prediction, _, memory = world(points, features, valid, sites.reshape(1, k * h, 3))
    memory = world.assimilate(memory, initial_prediction, initial[None, None].expand_as(initial_prediction))
    target = torch.as_tensor(item["physical"] == item["source_id"], device=device)
    losses = []
    physical_nll = []
    previous = None
    for index in action_indices:
        raw_action = item["actions"][index]
        step_action = raw_action if previous is None else relative_action(previous, raw_action)
        action = torch.as_tensor(step_action, device=device)
        expanded = action[None, None].expand(1, k * h, 12)
        predicted_memory, moved, q, lv = world.imagine(memory, sites.reshape(1, k * h, 3), expanded)
        moved = moved.reshape(k, h, 3)
        q = q.reshape(k, h, 48, 4, 4)
        lv = lv.reshape(k, h, 48)
        actual = complex_tensor(item["future"][index], device)
        weight = torch.as_tensor(item["reliability"][index], device=device)
        if not bool(weight.any()):
            previous = raw_action
            continue
        likelihood = verifier(q, lv, actual, weight, moved, action)
        conditional_site = belief.log_joint - torch.logsumexp(belief.log_joint, -1, keepdim=True)
        belief = belief.updated(likelihood)
        correct_probability = belief.candidate_probability()[target].sum().clamp_min(1e-8)
        nll = -correct_probability.log()
        # Contrast explanations of the same actually acquired sound, rather
        # than forcing candidate forecasts to diverge synthetically.
        candidate_evidence = torch.logsumexp(conditional_site + likelihood, -1)
        positive = torch.logsumexp(candidate_evidence[target], 0) - target.sum().float().log()
        negative = torch.logsumexp(candidate_evidence[~target], 0) - (~target).sum().float().log()
        ranking = F.softplus(0.5 - (positive - negative))
        error = (q - actual).abs().square().sum((-1, -2))
        forecast = (error[target] * weight[None] * conditional_site[target].exp()[..., None]).sum((-1, -2)).mean()
        losses.append(nll + 0.25 * ranking + 0.05 * forecast)
        physical_nll.append(nll.detach())
        memory = world.assimilate(predicted_memory, q.reshape(1, k * h, 48, 4, 4),
                                  actual[None, None].expand(1, k * h, 48, 4, 4))
        sites = moved
        previous = raw_action
    if not losses:
        return None
    return torch.stack(losses).mean(), torch.stack(physical_nll).mean()


def choose_actions(item, sequence_length, rng):
    ids = np.flatnonzero(item["valid"] & (item["reliability"].sum(-1) > 0))
    if not len(ids):
        return []
    first = rng.choice(ids.tolist())
    order = [first]
    while len(order) < sequence_length:
        options = [i for i in ids if i not in order and
                   np.linalg.norm(relative_action(item["actions"][order[-1]],
                                                  item["actions"][i])[:3]) <= 3.0]
        if not options:
            break
        order.append(int(rng.choice(options)))
    return order


def run_epoch(world, verifier, samples, data_root, cache_root, args, optimizer=None, epoch=0):
    training = optimizer is not None
    world.train(training and not args.freeze_world)
    verifier.train(training)
    rng = random.Random(args.seed + epoch if training else args.seed)
    order = list(samples)
    if training:
        rng.shuffle(order)
    total = 0.0
    used = 0
    skipped = 0
    for number, sample in enumerate(order, 1):
        item = prepare_sample(sample, data_root, cache_root, args.max_candidates, args.min_positive_iou)
        if item is None:
            skipped += 1
            continue
        actions = choose_actions(item, args.sequence_length, rng)
        if not actions:
            skipped += 1
            continue
        with torch.set_grad_enabled(training):
            result = episode_loss(world, verifier, item, actions, args.device)
            if result is None:
                skipped += 1
                continue
            loss, nll = result
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {sample}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(world.parameters()) + list(verifier.parameters()), 2.0)
                optimizer.step()
        total += float(nll)
        used += 1
        if number % 100 == 0:
            print(json.dumps({"epoch": epoch, "training": training, "processed": number,
                              "eligible": used, "physical_nll": total / used if used else None}), flush=True)
    if not used:
        raise RuntimeError("No eligible positive-versus-wrong-instance branches were found")
    return {"physical_instance_nll": total / used, "eligible": used,
            "skipped": skipped, "total": len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT / ".importants/Experiments_20261024/generated/common")
    parser.add_argument("--cache-root", type=Path,
                        default=HERE / "prepared/common/cache")
    parser.add_argument("--manifest-root", type=Path,
                        default=HERE / "prepared/common/manifests")
    parser.add_argument("--output", type=Path, default=HERE / "checkpoints/v1_instance_world.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=2,
                        help="Number of additional observations to use in training")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--min-positive-iou", type=float, default=0.3)
    parser.add_argument("--freeze-world", action="store_true")
    parser.add_argument("--max-train", type=int, default=0, help="Development limit; zero uses full TRAIN")
    parser.add_argument("--max-val", type=int, default=0, help="Development limit; zero uses full VAL")
    args = parser.parse_args()
    if args.epochs < 1 or args.sequence_length < 1 or args.max_candidates < 4:
        parser.error("epochs and sequence length must be positive; max candidates >=4")
    if (HERE / "SOURCE_MANIFEST.json").exists():
        verify_source_manifest()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    released = ReleasedModel(HERE, args.device)
    world = released.world
    world.requires_grad_(not args.freeze_world)
    verifier = InstanceVerifier(released.head).to(args.device)
    parameters = [{"params": verifier.parameters(), "lr": 1e-3}]
    if not args.freeze_world:
        parameters.append({"params": world.parameters(), "lr": 2e-5})
    optimizer = torch.optim.AdamW(parameters, weight_decay=1e-4)
    train_manifest = args.manifest_root / "train.json"
    val_manifest = args.manifest_root / "val.json"
    train = json.loads(train_manifest.read_text())
    val = json.loads(val_manifest.read_text())
    full_train_count = sum(1 for _ in (args.data_root / "train").rglob("audio.wav"))
    full_val_count = sum(1 for _ in (args.data_root / "val").rglob("audio.wav"))
    if args.max_train:
        train = train[:args.max_train]
    if args.max_val:
        val = val[:args.max_val]
    development_limited = (len(train) != full_train_count or len(val) != full_val_count or
                           bool(args.max_train or args.max_val))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(world, verifier, train, args.data_root, args.cache_root,
                             args, optimizer, epoch)
        validation = run_epoch(world, verifier, val, args.data_root, args.cache_root,
                               args, None, epoch)
        row = {"epoch": epoch, "train": training, "val": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["physical_instance_nll"] < best:
            best = validation["physical_instance_nll"]
            checkpoint = {
                "schema": "activeview-v1-instance-verification-v1",
                "world": {k: v.detach().cpu() for k, v in world.state_dict().items()},
                "verifier": {k: v.detach().cpu() for k, v in verifier.state_dict().items()},
                "epoch": epoch, "seed": args.seed, "val_physical_instance_nll": best,
                "training_objective": "physical-instance NLL + 0.25 same-future explanation ranking + 0.05 correct-instance covariance error",
                "source": {"train_manifest_sha256": sha(train_manifest),
                           "val_manifest_sha256": sha(val_manifest),
                           "v0_world_sha256": sha(HERE / "checkpoints/acoustic_world.pt"),
                           "training_script_sha256": sha(__file__),
                           "v1_source_manifest_sha256": (sha(HERE / "SOURCE_MANIFEST.json")
                                                          if (HERE / "SOURCE_MANIFEST.json").exists() else None)},
                "eligibility": {"max_candidates": args.max_candidates,
                                "min_positive_iou": args.min_positive_iou,
                                "sequence_length": args.sequence_length,
                                "world_fine_tuned": not args.freeze_world,
                                "max_train": args.max_train, "max_val": args.max_val},
                "training_scope": {"train_samples": len(train), "val_samples": len(val),
                                   "full_dataset_train_samples": full_train_count,
                                   "full_dataset_val_samples": full_val_count,
                                   "development_limited": development_limited},
            }
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
        (args.output.parent / "v1_training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.output), "sha256": sha(args.output),
                      "best_val_physical_instance_nll": best}))


if __name__ == "__main__":
    main()
