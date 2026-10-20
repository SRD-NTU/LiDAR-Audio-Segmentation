"""Keep reference masks, relate fused masks by measured geometric overlap."""
import torch


@torch.no_grad()
def assemble(reference_x, reference_score, fused_x, fused_score, overlap, reference_sites, fused_sites):
    """No labels: an exact reference fallback plus geometric cross-bank evidence."""
    nr,nf=overlap.shape
    ref_match=overlap.argmax(1);fused_match=overlap.argmax(0)
    def descriptors(x,score,other_x,other_score,relation,match,sites,other_sites,bank):
        quality=relation.max(1).values
        if relation.shape[1]>1:
            second=relation.topk(2,dim=1).values[:,1]
        else:second=torch.zeros_like(quality)
        matched=other_score[match]-other_score.max()
        compatible=relation@other_score.softmax(0)
        distance=(sites[:,0]-other_sites[match,0]).norm(dim=-1)/5
        fields=torch.stack([quality,quality-second,matched,compatible,distance,
                            torch.full_like(quality,bank),score-score.max()],-1)
        # Rank descriptors and acoustic likelihood summaries from the matched bank.
        return torch.cat([x,other_x[match,:45],fields],-1)
    rx=descriptors(reference_x,reference_score,fused_x,fused_score,overlap,ref_match,reference_sites,fused_sites,0.)
    fx=descriptors(fused_x,fused_score,reference_x,reference_score,overlap.T,fused_match,fused_sites,reference_sites,1.)
    # At zero correction the selected reference mask is exactly the original MAP.
    base=torch.cat([reference_score-reference_score.max(),fused_score-fused_score.max()-2.])
    return torch.cat([rx,fx]),base,ref_match,fused_match


@torch.no_grad()
def geometric_overlap(reference_masks, reference_inverse, fused_masks, fused_reference_map):
    """Exact overlap on original points, compressed by shared voxel-pair counts."""
    width=fused_masks.shape[1]
    pairs=reference_inverse.long()*width+fused_reference_map.long()
    unique,counts=torch.unique(pairs,return_counts=True)
    ri=torch.div(unique,width,rounding_mode='floor');fi=unique%width
    reference=reference_masks[:,ri].float()
    fused=fused_masks[:,fi].float()
    intersection=(reference*counts.float())@fused.T
    area_r=(reference*counts.float()).sum(1)
    area_f=(fused*counts.float()).sum(1)
    return intersection/(area_r[:,None]+area_f[None]-intersection).clamp_min(1)
