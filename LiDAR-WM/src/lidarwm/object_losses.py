"""Training-only instance matching and audio-following objectives."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def lovasz_hinge(logits, target, valid):
    """Per-example binary Lovasz hinge, directly targeting foreground IoU."""
    terms=[]
    for prediction, truth, keep in zip(logits.flatten(1),target.flatten(1),valid.flatten(1)):
        keep=keep>0
        prediction=prediction[keep].float();truth=truth[keep].float()
        if not prediction.numel():
            terms.append(prediction.sum());continue
        errors=1-prediction*(2*truth-1)
        errors,order=errors.sort(descending=True)
        truth=truth[order]
        intersection=truth.sum()-truth.cumsum(0)
        union=truth.sum()+(1-truth).cumsum(0)
        gradient=1-intersection/union.clamp_min(1)
        gradient=torch.cat((gradient[:1],gradient[1:]-gradient[:-1]))
        terms.append(torch.dot(errors.relu(),gradient))
    return torch.stack(terms).mean()


def object_world_losses(output,batch):
    instance_terms=[];source_terms=[];reconstruction_terms=[];hard_terms=[]
    for b in range(batch['mask'].shape[0]):
        ids=batch['instance_map'][b].flatten()
        valid=batch['valid'][b,0].flatten()>0
        gt_ids=ids.unique();gt_ids=gt_ids[gt_ids>0]
        predictions=output['object_mask_logits'][b].float().flatten(1)
        classes=output['object_class_logits'][b].float()
        if len(gt_ids)>len(predictions):
            raise ValueError(f'{len(gt_ids)} visible object targets exceed {len(predictions)} queries; increase model.max_queries')
        if not len(gt_ids):
            instance_terms.append(F.cross_entropy(classes,torch.zeros(len(classes),device=ids.device,dtype=torch.long)))
            continue
        targets=(ids[None]==gt_ids[:,None]).float()
        # Include every object's positives so tiny targets survive matching.
        # Deterministic subsampling keeps validation loss reproducible.
        indices=[]
        for target in targets:
            points=target.nonzero().flatten()
            indices.append(points[torch.linspace(0,len(points)-1,min(64,len(points)),device=ids.device).long()])
        background=((ids==0)&valid).nonzero().flatten()
        if len(background):
            indices.append(background[torch.linspace(0,len(background)-1,min(2048,len(background)),device=ids.device).long()])
        sample=torch.cat(indices)
        selected=predictions[:,sample];truth=targets[:,sample]
        with torch.no_grad():
            probability=selected.sigmoid()
            dice_cost=1-(2*(probability@truth.T)+1)/(probability.sum(1)[:,None]+truth.sum(1)[None]+1)
            # Balanced positive/negative matching costs, not background-dominated.
            positive_cost=-(F.logsigmoid(selected)@truth.T)/truth.sum(1).clamp_min(1)[None]
            negative_cost=-(F.logsigmoid(-selected)@(1-truth).T)/(1-truth).sum(1).clamp_min(1)[None]
            semantic=batch['semantic'][b].flatten()
            category=torch.stack([semantic[ids==value].mode().values for value in gt_ids])
            cost=dice_cost+positive_cost+negative_cost-classes.softmax(-1)[:,category]
            row,col=linear_sum_assignment(cost.cpu().numpy())
            row=torch.as_tensor(row,device=ids.device);col=torch.as_tensor(col,device=ids.device)
        matched=predictions[row][:,valid];truth_full=targets[col][:,valid]
        error=F.binary_cross_entropy_with_logits(matched,truth_full,reduction='none')
        pos=(error*truth_full).sum(1)/truth_full.sum(1).clamp_min(1)
        neg=(error*(1-truth_full)).sum(1)/(1-truth_full).sum(1).clamp_min(1)
        probability=matched.sigmoid()
        dice=1-(2*(probability*truth_full).sum(1)+1)/(probability.sum(1)+truth_full.sum(1)+1)
        class_target=torch.zeros(len(classes),device=ids.device,dtype=torch.long)
        class_target[row]=category[col]
        class_weights=torch.ones(classes.shape[-1],device=ids.device);class_weights[0]=.1
        instance_terms.append((pos+neg+dice*2).mean()+F.cross_entropy(classes,class_target,weight=class_weights))
        source=batch['mask'][b,0].flatten()*valid
        # Sounding label selects a training target only, never the forward query.
        overlap=(targets*source).sum(1)
        if overlap.max()>0:
            source_id=overlap.argmax()
            source_slot=row[col==source_id][0]
            source_terms.append(F.cross_entropy(output['object_scores'][b:b+1],source_slot[None]))
            # Explicit competing silent instances in this SAME scene and category.
            # Labels define training negatives only; inference ranks all slots.
            source_category=category[source_id]
            same=row[category[col]==source_category]
            same=same[same!=source_slot]
            if len(same):
                scores=output['object_scores'][b]
                hard_terms.append(F.softplus(1+scores[same]-scores[source_slot]).mean())
            reconstruction_terms.append(1-F.cosine_similarity(output['object_rendered_audio'][b,source_slot][None].float(),output['observed_audio'][b:b+1].float()).mean())
    zero=output['mask_logits'].sum()*0
    return {
        'object_instances':torch.stack(instance_terms).mean() if instance_terms else zero,
        'object_source':torch.stack(source_terms).mean() if source_terms else zero,
        'object_reconstruction':torch.stack(reconstruction_terms).mean() if reconstruction_terms else zero,
        'audio_distillation':(1-F.cosine_similarity(output['student_audio'].float(),output['observed_audio'].float())).mean(),
        'hard_source':torch.stack(hard_terms).mean() if hard_terms else zero,
    }
