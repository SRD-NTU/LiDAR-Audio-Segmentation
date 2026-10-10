from __future__ import annotations

import json
import math
import os
import random
import time
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader,WeightedRandomSampler

from .config import save_config
from .data import AudioLidarDataset, collate_samples
from .losses import WorldModelLoss
from .metrics import BinarySegmentationMetrics
from .model import AcousticWorldModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> AcousticWorldModel:
    kwargs=dict(config['model'])
    architecture=kwargs.pop('architecture','original')
    if architecture=='foundation_world':
        from .foundation_world import FoundationWorldModel
        return FoundationWorldModel(**kwargs)
    if architecture=='suppressed':
        from .suppression import SuppressedWorldModel
        return SuppressedWorldModel(**kwargs)
    if architecture=='graph_world':
        from .graph_world import GraphLatentWorldModel
        return GraphLatentWorldModel(**kwargs)
    if architecture=='object_world':
        from .object_world import ObjectQueryWorldModel
        return ObjectQueryWorldModel(**kwargs)
    if architecture=='revised':
        from .revised import RevisedWorldModel
        return RevisedWorldModel(**kwargs)
    if architecture=='instance_field':
        from .instance_field import InstanceFieldWorldModel
        return InstanceFieldWorldModel(**kwargs)
    return AcousticWorldModel(**kwargs)


def forward_batch(model, batch):
    extra={key:batch[key] for key in ('audio_embedding','text_logits','geometry_features') if key in batch}
    return model(batch['lidar'],batch['audio'],batch['valid'],**extra)


def to_device(batch: dict, device: torch.device) -> dict:
    def move(value):
        if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,list):return [move(v) for v in value]
        return value
    return {
        key: move(value) if key=='geometry_features' else (value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value)
        for key, value in batch.items()
    }


def atomic_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        if config['model'].get('architecture')=='suppressed':
            raise ValueError('The suppressed model is inference-only. Train its frozen-world refiner with train_mask_suppression.py.')
        train_cfg = config["training"]
        seed_everything(int(train_cfg.get("seed", 1337)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if train_cfg.get("require_cuda", True) and self.device.type != "cuda":
            raise RuntimeError("CUDA is required by this config. Use training.require_cuda=false for a CPU smoke test.")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.run_dir = Path(config["output_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.yaml")
        snapshot=self.run_dir/'source'
        if not snapshot.exists():
            shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.glob('*.py'))}
            (self.run_dir/'source_hashes.json').write_text(json.dumps(files,indent=2))
        self.log_path = self.run_dir / "metrics.jsonl"

        data_cfg = config["data"]
        self.train_set = AudioLidarDataset(
            data_cfg["root"], "train", data_cfg.get("audio_stride", 3), True,
            data_cfg.get("max_train_samples"),
            data_cfg.get('feature_root'), data_cfg.get('equivariant_augmentation',False),data_cfg.get('antialias',False),
            data_cfg.get('instance_centers',False),
            data_cfg.get('instance_masks',False),
            data_cfg.get('geometry_feature_root'),
        )
        self.val_set = AudioLidarDataset(
            data_cfg["root"], "val", data_cfg.get("audio_stride", 3), False,
            data_cfg.get("max_val_samples"),
            data_cfg.get('feature_root'), False,data_cfg.get('antialias',False),
            data_cfg.get('instance_centers',False),
            data_cfg.get('instance_masks',False),
            data_cfg.get('geometry_feature_root'),
        )
        loader_args = dict(
            num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=self.device.type == "cuda",
            collate_fn=collate_samples, persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
        )
        category_counts=Counter(record.category for record in self.train_set.records)
        self.train_loader = DataLoader(
            self.train_set, batch_size=int(train_cfg["batch_size"]), shuffle=not train_cfg.get('balance_categories',False),
            sampler=(WeightedRandomSampler(
                [category_counts[record.category]**-.5 for record in self.train_set.records],
                len(self.train_set),replacement=True) if train_cfg.get('balance_categories',False) else None),drop_last=True,
            **loader_args,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
            shuffle=False, drop_last=False, **loader_args,
        )
        self.model = build_model(config).to(self.device)
        if train_cfg.get('initialize_from'):
            state=torch.load(train_cfg['initialize_from'],map_location='cpu',weights_only=False)
            missing,unexpected=self.model.load_state_dict(state['model'],strict=False)
            if any(not (name.startswith(('center_head.','geometry_graph.','object_','geometry_','phase_renderer.')) or name=='field_strength') for name in missing) or unexpected:
                raise ValueError(f'Unexpected initialization mismatch: {missing}, {unexpected}')
        self.criterion = WorldModelLoss(**config.get("loss", {})).to(self.device)
        self.stage=train_cfg.get('stage','joint')
        if self.stage not in ('geometry','selection','joint'):raise ValueError('Unknown training stage')
        if self.stage!='joint':
            self.model.requires_grad_(False)
            geometry_prefixes=('stem.','down.','scene_context.','geo_decoder.','semantic_head.',
                'geometry_adapter.','geometry_context.','geometry_fine.','geometry_gate',
                'object_features.','object_local','object_position.','object_decoder.','object_mask.',
                'object_class.','object_offset.','object_radius.','object_graph.')
            for name,parameter in self.model.named_parameters():
                is_geometry=name.startswith(geometry_prefixes)
                if (self.stage=='geometry' and is_geometry) or (self.stage=='selection' and not is_geometry and not name.startswith('geometry_foundation.')):
                    parameter.requires_grad_(True)
            if self.stage=='geometry':
                for key in self.criterion.weights:
                    if key not in ('semantic','object_instances','geometry_foreground','boundary'):
                        self.criterion.weights[key]=0.
        # Keep frozen dropout modules deterministic in the staged curriculum.
        self.frozen_module_names=[]
        for name,module in self.model.named_modules():
            parameters=list(module.parameters())
            if name and parameters and not any(p.requires_grad for p in parameters):
                self.frozen_module_names.append(name)
        if self.stage=='selection' and hasattr(self.model,'geometry_feature_dropout'):
            self.frozen_module_names.append('geometry_feature_dropout')
        if data_cfg.get('geometry_feature_root'):
            metadata=json.loads((Path(data_cfg['geometry_feature_root'])/'encoder.json').read_text())
            encoder=self.model.geometry_foundation
            if metadata['encoder_sha256']!=encoder.checkpoint_sha256 or metadata['grid_size']!=encoder.grid_size or metadata.get('adapter_version')!='deterministic-pooling-v2':
                raise ValueError('Geometry cache and raw-XYZ encoder mismatch')
        learning_rate=float(train_cfg['learning_rate'])
        backbone_scale=float(train_cfg.get('backbone_lr_scale',1.0))
        if backbone_scale<=0:
            raise ValueError('backbone_lr_scale must be positive; world core must remain trainable')
        def group(name):
            if name.startswith('geometry_foundation.'):return 'encoder'
            if name.startswith(('object_','geometry_','phase_renderer.')):return 'heads'
            return 'backbone'
        scales={'encoder':float(train_cfg.get('encoder_lr_scale',.01)),'heads':1.,'backbone':backbone_scale}
        parameters=[{'params':[p for name,p in self.model.named_parameters() if p.requires_grad and group(name)==key],
            'lr':learning_rate*scale} for key,scale in scales.items()]
        parameters=[g for g in parameters if len(g['params'])]
        self.optimizer = AdamW(
            parameters, lr=learning_rate,
            weight_decay=float(train_cfg.get("weight_decay", 0.05)), betas=(0.9, 0.95),
        )
        accumulation = int(train_cfg.get("gradient_accumulation", 1))
        self.steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accumulation))
        total_steps = self.steps_per_epoch * int(train_cfg["epochs"])
        warmup_steps = self.steps_per_epoch * int(train_cfg.get("warmup_epochs", 5))

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return max(1e-3, step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = LambdaLR(self.optimizer, schedule)
        self.use_amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.start_epoch = 0
        self.best_iou = -1.0
        resume = train_cfg.get("resume")
        if resume:
            state = load_checkpoint(resume, self.model, self.optimizer, self.scheduler)
            self.start_epoch = int(state["epoch"]) + 1
            self.best_iou = float(state.get("best_iou", -1.0))

    def _log(self, payload: dict[str, Any]) -> None:
        payload = {"time": time.time(), **payload}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        modules=dict(self.model.named_modules())
        for name in self.frozen_module_names:modules[name].eval()
        accumulation = int(self.config["training"].get("gradient_accumulation", 1))
        clip_norm = float(self.config["training"].get("clip_grad_norm", 1.0))
        running: dict[str, float] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(self.train_loader):
            batch = to_device(raw_batch, self.device)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = forward_batch(self.model,batch)
                losses = self.criterion(output, batch)
                scaled_loss = losses["total"] / accumulation
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}, samples {raw_batch["path"]}')
            self.scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), clip_norm, error_if_nonfinite=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach())
        return {name: value / len(self.train_loader) for name, value in running.items()}

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        metrics = BinarySegmentationMetrics(tuple(self.config["evaluation"]["thresholds"]))
        total_loss = 0.0
        oracle_sum=0.;query_count=0
        semantic_intersection=torch.zeros(17,device=self.device)
        semantic_union=torch.zeros(17,device=self.device)
        for raw_batch in self.val_loader:
            batch = to_device(raw_batch, self.device)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = forward_batch(self.model,batch)
                total_loss += float(self.criterion(output, batch)["total"])
            metrics.update(output["mask_logits"], batch["mask"], batch["valid"])
            if self.stage=='geometry':
                prediction=output['semantic_logits'].argmax(1)
                for c in range(1,17):
                    p=(prediction==c)&batch['valid'][:,0].bool();t=batch['semantic']==c
                    semantic_intersection[c]+=(p&t).sum();semantic_union[c]+=(p|t).sum()
                if 'object_mask_logits' in output:
                    candidates=(output['object_mask_logits'].sigmoid()>=.5)&batch['valid'].bool()
                    truth=batch['mask'].bool()
                    iou=(candidates&truth).sum((-2,-1))/(candidates|truth).sum((-2,-1)).clamp_min(1)
                    oracle_sum+=float(iou.amax(-1).sum());query_count+=len(iou)
        results={"loss": total_loss / len(self.val_loader), **metrics.compute()}
        if self.stage=='geometry':
            present=semantic_union[1:]>0
            results['geometry_semantic_miou']=float((semantic_intersection[1:]/semantic_union[1:].clamp_min(1))[present].mean())
            results['oracle_query_coverage_iou']=oracle_sum/max(query_count,1)
        return results

    def fit(self) -> None:
        epochs = int(self.config["training"]["epochs"])
        patience = int(self.config["training"].get("early_stopping_patience", epochs))
        epochs_without_improvement = 0
        self._log({
            "event": "start", "device": str(self.device), "train_samples": len(self.train_set),
            "val_samples": len(self.val_set), "parameters": sum(p.numel() for p in self.model.parameters()),
        })
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            selection_key='oracle_query_coverage_iou' if self.stage=='geometry' else 'mean_iou'
            improved = val_metrics[selection_key] > self.best_iou
            if improved:
                self.best_iou = val_metrics[selection_key]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            state = {
                "epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "best_iou": self.best_iou,
                "config": self.config, "validation": val_metrics,"selection_metric":selection_key,
            }
            atomic_checkpoint(state, self.run_dir / "last.pt")
            if improved:
                atomic_checkpoint(state, self.run_dir / "best.pt")
            self._log({
                "event": "epoch", "epoch": epoch, "learning_rate": self.scheduler.get_last_lr()[0],
                "train": train_metrics, "validation": val_metrics, "best_iou": self.best_iou,
            })
            if epochs_without_improvement >= patience:
                self._log({"event": "early_stop", "epoch": epoch, "patience": patience})
                break


@torch.inference_mode()
def evaluate(config: dict[str, Any], checkpoint_path: str | Path, split: str = "val") -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()
    data_cfg, train_cfg = config["data"], config["training"]
    dataset = AudioLidarDataset(data_cfg["root"], split, data_cfg.get("audio_stride", 3), False,
        feature_root=data_cfg.get('feature_root'),antialias=data_cfg.get('antialias',False),
        geometry_feature_root=data_cfg.get('geometry_feature_root'))
    loader = DataLoader(
        dataset, batch_size=int(train_cfg.get("val_batch_size", 2)), shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=device.type == "cuda",
        collate_fn=collate_samples,
    )
    if split == "test":
        # Test labels must not choose a threshold. Reuse the selection stored by
        # the best validation checkpoint.
        validation_threshold = float(checkpoint.get("validation", {}).get("threshold", 0.5))
        thresholds = (validation_threshold,)
    else:
        thresholds = tuple(config["evaluation"]["thresholds"])
    metrics = BinarySegmentationMetrics(thresholds)
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        from .inference import predict_mask_logits
        extra={key:batch[key] for key in ('audio_embedding','text_logits') if key in batch}
        logits=predict_mask_logits(model,batch['lidar'],batch['audio'],batch['valid'],extra,
            tta=config.get('evaluation',{}).get('tta',False))
        metrics.update(logits, batch["mask"], batch["valid"])
    return metrics.compute()
