from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    positives = (target * valid).sum()
    negatives = ((1.0 - target) * valid).sum()
    pos_weight = (negatives / positives.clamp_min(1.0)).sqrt().clamp(2.0, 80.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    probability = logits.sigmoid()
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    return (((1.0 - pt).pow(gamma) * bce) * valid).sum() / valid.sum().clamp_min(1.0)


def soft_azimuth_target(mask: torch.Tensor, bins: int) -> torch.Tensor:
    column_mass = mask.sum(dim=-2).squeeze(1)
    # Pooling conserves target mass. Strided linear interpolation can completely
    # miss a narrow target such as a fan, clock or keyboard and supervise a
    # uniform direction distribution instead.
    column_mass = F.adaptive_avg_pool1d(column_mass[:, None], bins).squeeze(1)
    column_mass = column_mass + 1e-6
    return column_mass / column_mass.sum(dim=-1, keepdim=True)


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        segmentation: float = 1.0,
        dice: float = 1.0,
        renderer: float = 0.25,
        consistency: float = 0.1,
        doa: float = 0.15,
        category: float = 0.1,
        semantic: float = 0.25,
        position: float = 0.2,
        smoothness: float = 0.02,
        sparsity: float = 0.02,
        ranking: float = 0.0,
        balanced_focal: bool = False,
        physical_ranking: float = 0.0,
        instance_center: float = 0.0,
        balanced_semantic: bool = False,
        lovasz: float = 0.0,
        object_instances: float = 0.0,
        object_source: float = 0.0,
        object_reconstruction: float = 0.0,
        audio_distillation: float = 0.0,
        query_segmentation: float = 0.0,
        geometry_foreground: float = 0.0,
        boundary: float = 0.0,
        hard_source: float = 0.0,
    ) -> None:
        super().__init__()
        self.balanced_focal=balanced_focal
        self.balanced_semantic=balanced_semantic
        self.weights = {
            "segmentation": segmentation, "dice": dice, "renderer": renderer,
            "consistency": consistency, "doa": doa, "category": category,
            "semantic": semantic, "position": position,
            "smoothness": smoothness, "sparsity": sparsity,
            "ranking": ranking,
            "physical_ranking":physical_ranking,
            "instance_center":instance_center,
            "lovasz":lovasz,"object_instances":object_instances,
            "object_source":object_source,"object_reconstruction":object_reconstruction,
            "audio_distillation":audio_distillation,"query_segmentation":query_segmentation,
            'geometry_foreground':geometry_foreground,'boundary':boundary,'hard_source':hard_source,
        }

    def forward(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits, target, valid = output["mask_logits"], batch["mask"], batch["valid"]
        losses: dict[str, torch.Tensor] = {}
        losses["segmentation"] = focal_bce_loss(logits, target, valid)
        if self.balanced_focal:
            bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
            prob=logits.sigmoid(); pt=prob*target+(1-prob)*(1-target)
            error=bce*(1-pt).square()*valid
            axes=(-3,-2,-1)
            positive=(error*target).sum(axes)/(target*valid).sum(axes).clamp_min(1)
            negative=(error*(1-target)).sum(axes)/((1-target)*valid).sum(axes).clamp_min(1)
            losses['segmentation']=(positive+negative).mean()
        losses["dice"] = soft_dice_loss(logits, target, valid)

        render_shape = output["render_logits"].shape[-2:]
        region_target = F.adaptive_max_pool2d(target, render_shape).squeeze(1)
        region_valid = F.adaptive_max_pool2d(valid, render_shape).squeeze(1)
        render_bce = F.binary_cross_entropy_with_logits(
            output["render_logits"], region_target, reduction="none",
            pos_weight=torch.tensor(8.0, device=logits.device),
        )
        losses["renderer"] = (render_bce * region_valid).sum() / region_valid.sum().clamp_min(1.0)

        flat_target = region_target.flatten(1)
        weights = flat_target / flat_target.sum(dim=1, keepdim=True).clamp_min(1.0)
        losses['ranking']=-(weights*F.log_softmax(output['render_logits'].flatten(1),dim=-1)).sum(-1).mean()
        losses['physical_ranking']=torch.zeros((),device=logits.device)
        if 'physical_render_logits' in output:
            losses['physical_ranking']=-(weights*F.log_softmax(output['physical_render_logits'].flatten(1),dim=-1)).sum(-1).mean()
        source_audio = torch.einsum("bn,bnd->bd", weights, output["rendered_audio"])
        losses["consistency"] = (1.0 - F.cosine_similarity(source_audio, output["observed_audio"], dim=-1)).mean()

        azimuth_target = soft_azimuth_target(target * valid, output["doa_logits"].shape[-1])
        losses["doa"] = -(azimuth_target * F.log_softmax(output["doa_logits"], dim=-1)).sum(dim=-1).mean()
        losses["category"] = F.cross_entropy(output["category_logits"], batch["category"])
        semantic_weights = torch.ones(output["semantic_logits"].shape[1], device=logits.device)
        semantic_weights[0] = 0.05
        losses["semantic"] = F.cross_entropy(
            output["semantic_logits"], batch["semantic"], weight=semantic_weights, ignore_index=255,
        )
        if self.balanced_semantic:
            # Equalize visible class contributions per scene rather than letting
            # large walls/appliances overwhelm clocks, keyboards and other tiny objects.
            with torch.autocast(device_type=logits.device.type,enabled=False):
                scores=output['semantic_logits'].float()
                labels=batch['semantic']; known=labels!=255
                safe=labels.masked_fill(~known,0)
                classes=scores.shape[1]
                nll=-scores.log_softmax(1).gather(1,safe[:,None]).squeeze(1)*known
                counts=torch.zeros(scores.shape[0],classes,device=logits.device)
                totals=torch.zeros_like(counts)
                counts.scatter_add_(1,safe.flatten(1),known.float().flatten(1))
                totals.scatter_add_(1,safe.flatten(1),nll.flatten(1))
                present=(counts>0).float(); present[:,0]*=.25
                ce=((totals/counts.clamp_min(1))*present).sum(1)/present.sum(1).clamp_min(1)
                onehot=F.one_hot(safe,num_classes=classes).permute(0,3,1,2).float()*known[:,None]
                probability=scores.softmax(1)*known[:,None]
                intersection=(probability*onehot).sum((-2,-1))
                denominator=(probability+onehot).sum((-2,-1))
                dice=1-(2*intersection+1)/(denominator+1)
                foreground=present.clone();foreground[:,0]=0
                dice=(dice*foreground).sum(1)/foreground.sum(1).clamp_min(1)
                losses['semantic']=(ce+dice).mean()
        xyz = batch["lidar"][:, :3]
        source_weight = target * valid
        target_position = (xyz * source_weight).sum(dim=(-2, -1)) / source_weight.sum(dim=(-2, -1)).clamp_min(1.0)
        losses["position"] = F.smooth_l1_loss(output["source_position"], target_position, beta=0.05)
        losses['instance_center']=torch.zeros((),device=logits.device)
        if 'instance_center' in output and 'instance_center' in batch:
            error=F.smooth_l1_loss(output['instance_center'],batch['instance_center'],reduction='none',beta=.025).mean(1,keepdim=True)
            weight=batch['instance_weight']
            losses['instance_center']=(error*weight).sum()/weight.sum().clamp_min(1)
            if 'instance_anchor' in output:
                losses['instance_center']=losses['instance_center']+F.smooth_l1_loss(output['instance_anchor'],target_position,beta=.025)

        probability = logits.sigmoid()
        depth = batch["lidar"][:, 3:4]
        horizontal_edge = torch.exp(-20.0 * (depth[..., 1:] - depth[..., :-1]).abs())
        vertical_edge = torch.exp(-20.0 * (depth[..., 1:, :] - depth[..., :-1, :]).abs())
        horizontal = ((probability[..., 1:] - probability[..., :-1]).abs() * horizontal_edge).mean()
        vertical = ((probability[..., 1:, :] - probability[..., :-1, :]).abs() * vertical_edge).mean()
        seam_edge = torch.exp(-20.0 * (depth[..., :1] - depth[..., -1:]).abs())
        seam = ((probability[..., :1] - probability[..., -1:]).abs() * seam_edge).mean()
        losses["smoothness"] = horizontal + vertical + seam
        predicted_fraction = (probability * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1.0)
        target_fraction = (target * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1.0)
        losses["sparsity"] = F.smooth_l1_loss(predicted_fraction, target_fraction)
        if self.weights['lovasz']:
            from .object_losses import lovasz_hinge
            losses['lovasz']=lovasz_hinge(logits,target,valid)
        if 'object_mask_logits' in output:
            from .object_losses import object_world_losses
            losses.update(object_world_losses(output,batch))
            query=output['query_mask_logits']
            losses['query_segmentation']=soft_dice_loss(query,target,valid)
        if self.weights['geometry_foreground'] or self.weights['boundary']:
            semantic=output['semantic_logits'].float()
            # Exactly log(P(foreground)/P(background)); avoids sigmoid saturation.
            foreground=torch.logsumexp(semantic[:,1:],dim=1,keepdim=True)-semantic[:,:1]
            truth=((batch['semantic']>0)&(batch['semantic']!=255)).float()[:,None]
            losses['geometry_foreground']=soft_dice_loss(foreground,truth,valid)
            if self.weights['boundary']:
                instance=batch['instance_map'][:,None]
                edge=(instance!=instance.roll(1,-1))|(instance!=instance.roll(-1,-1))
                edge=edge|(instance!=instance.roll(1,-2))|(instance!=instance.roll(-1,-2))
                pixel_weight=(1+edge.float()*4)*valid
                errors=F.binary_cross_entropy_with_logits(foreground,truth,reduction='none')
                losses['boundary']=(errors*pixel_weight).sum()/pixel_weight.sum().clamp_min(1)
        losses["total"] = sum(self.weights[name] * value for name, value in losses.items())
        return losses
