"""Deployable two-scan recovery with no label or future-observation planning inputs."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from acoustic_wm.audio.predictive_features import covariance48_from_wave
from acoustic_wm.fusion import validate_actions,fuse_clouds
from recovery.runtime import bank_input
from recovery.features import acoustic_features
from recovery.union import assemble,geometric_overlap


class RecoveryPipeline:
    def __init__(self,model,encoder,ranker,kind='union'):
        self.model=model;self.encoder=encoder;self.ranker=ranker;self.kind=kind

    @torch.inference_mode()
    def run(self,xyz,wave,actions,acquire,*,rate=44100,valid=None,movement_cost=None,max_translation_m=3.0):
        if rate!=44100:raise ValueError('Expected four-channel 44100 Hz audio')
        model=self.model;actions=validate_actions(actions)
        valid=np.ones(len(actions),bool) if valid is None else np.asarray(valid,bool)
        distance_m=np.linalg.norm(actions[:,:3],axis=1)
        costs=distance_m.copy() if movement_cost is None else np.asarray(movement_cost,np.float32)
        if valid.shape!=(len(actions),) or costs.shape!=(len(actions),) or not np.isfinite(costs).all() or (costs<0).any():raise ValueError('Invalid validity/cost arrays')
        if not np.isfinite(max_translation_m) or max_translation_m<0:raise ValueError('Invalid translation bound')
        valid=valid & (distance_m<=max_translation_m+1e-6)
        available=np.flatnonzero(valid)
        observed,reliability=covariance48_from_wave(wave)
        reference_row=self.encoder.encode_arrays(xyz,wave,rate)
        if not len(available):
            state=model.current(reference_row,observed,reliability);single=model.result(state,state['base'])
            return dict(reference_sound_mask=single['sound_mask'],reference_point_probability=single['point_probability'],reference_xyz=xyz,second_xyz=np.empty((0,3),np.float32),second_sound_mask=np.empty(0,np.uint8),second_point_probability=np.empty(0,np.float32),posterior=single['posterior'],selected_object=single['selected_object'],selected_bank=np.str_('reference'),chosen_action=np.int64(-1),movement_cost=costs,chosen_movement_cost=np.float32(0),distance_m=np.float32(0),budget_used=np.int64(0),audio_observations=np.int64(1),point_clouds=np.int64(1),action_valid=valid,relative_actions=actions,max_translation_m=np.float32(max_translation_m),planning_policy=np.str_('grouped'))
        plan=model.plan(reference_row,observed,reliability,actions[available])
        query=int(available[int(plan['query'])]);action=actions[query]
        # Action and its forecast are frozen BEFORE acquisition of either sensor.
        second_xyz,second_wave,second_rate=acquire(query)
        if second_rate!=rate:raise ValueError('Mismatched audio sample rates')
        next_observed,next_reliability=covariance48_from_wave(second_wave)
        reference_row,ref_input=bank_input(self.encoder,model,xyz,wave,rate,action,observed,reliability,
            next_observed,next_reliability,row=reference_row,plan=plan)
        cloud=fuse_clouds(xyz,second_xyz,action,.02)
        fused_row=self.encoder.encode_fused_arrays(cloud.xyz,wave,rate,
            second_receiver_position=action[:3],view_membership=cloud.view_membership)
        _,fused_input=bank_input(self.encoder,model,cloud.xyz,wave,rate,action,observed,reliability,
            next_observed,next_reliability,row=fused_row)
        rx,rs=acoustic_features(model,ref_input);fx,fs=acoustic_features(model,fused_input)
        r=model.tensor(reference_row['masks']);f=model.tensor(fused_row['masks'])
        ri=model.tensor(reference_row['inverse']);fm=model.tensor(fused_row['inverse'][cloud.reference_to_fused])
        sm=model.tensor(fused_row['inverse'][cloud.second_to_fused])
        overlap=geometric_overlap(r,ri,f,fm)
        if self.kind=='union':
            x,base,match,_=assemble(rx,rs,fx,fs,overlap,ref_input['hypotheses'],fused_input['hypotheses'])
        else:x,base=fx,fs;match=overlap.argmax(1)
        score,_=self.ranker(x[None],base[None],torch.ones(1,len(base),dtype=torch.bool,device=x.device))
        posterior=score[0].softmax(0);index=int(score.argmax());nr=len(r)
        aligned=second_xyz@action[3:].reshape(3,3).T+action[:3]
        distance,nearest=cKDTree(reference_row['xyz']).query(aligned,workers=4)
        covered=model.tensor(distance<=.1);nearest=model.tensor(nearest)
        allow=overlap.max(1).values>=.1
        if self.kind=='union':
            if index<nr:
                ref_mask=r[index,ri];second_mask=r[index,nearest].clone()
                second_mask[~covered]=f[match[index],sm[~covered]] if allow[index] else False
            else:ref_mask=f[index-nr,fm];second_mask=f[index-nr,sm]
            rp=posterior[:nr]@r.float();fp=posterior[nr:]@f.float()
            ref_probability=rp[ri]+fp[fm]
            extended=torch.zeros(len(f),device=x.device).scatter_add_(0,match,posterior[:nr]*allow)
            second_probability=fp[sm]+torch.where(covered,rp[nearest],(extended@f.float())[sm])
        else:
            ref_mask=f[index,fm];second_mask=f[index,sm];point_probability=posterior@f.float()
            ref_probability=point_probability[fm];second_probability=point_probability[sm]
        if not (np.any(reliability) or np.any(next_reliability)):
            ref_mask=torch.zeros_like(ref_mask);second_mask=torch.zeros_like(second_mask)
            ref_probability=torch.zeros_like(ref_probability);second_probability=torch.zeros_like(second_probability)
        cpu=lambda t:t.detach().cpu().numpy()
        result=dict(reference_sound_mask=cpu(ref_mask).astype(np.uint8),second_sound_mask=cpu(second_mask).astype(np.uint8),
            reference_point_probability=cpu(ref_probability),second_point_probability=cpu(second_probability),
            reference_xyz=xyz,second_xyz=second_xyz,posterior=cpu(posterior),selected_object=np.int64(index),
            selected_bank=np.str_('reference' if self.kind=='union' and index<nr else 'fused'),
            reference_candidates=np.int64(nr),fused_candidates=np.int64(len(f)),
            matched_activeview_sound_mask=cpu(r[int(rs.argmax()),ri]).astype(np.uint8),
            chosen_action=np.int64(query),relative_actions=actions,action_valid=valid,
            movement_cost=costs,chosen_movement_cost=np.float32(costs[query]),distance_m=np.float32(distance_m[query]),budget_used=np.int64(1),
            max_translation_m=np.float32(max_translation_m),planning_policy=np.str_(plan['planning_policy']),
            query_utilities=cpu(plan['utilities']),audio_observations=np.int64(2),point_clouds=np.int64(2),
            prior_predicted_covariance=torch.view_as_real(plan['prediction']).cpu().numpy(),
            prior_log_error_scale=cpu(plan['log_variance']),geometry_source=np.str_('two_measured_scans'))
        from recovery.outputs import fused_outputs
        result.update(fused_outputs(xyz,second_xyz,action,result['reference_sound_mask'],result['second_sound_mask'],
                                   result['reference_point_probability'],result['second_point_probability']))
        return result
