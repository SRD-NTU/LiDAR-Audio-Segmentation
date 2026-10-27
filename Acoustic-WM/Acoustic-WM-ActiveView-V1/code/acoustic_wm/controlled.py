"""One-query ActiveView deployment with explicit validity and grouped planning.

The action objective is acoustic disagreement under a fixed observation budget.
Movement cost is validated and recorded, but is not an optimization objective.
The default grouping is a geometric proxy for instance hypotheses; it cannot
establish physical-instance identity without labels.
"""
import numpy as np
import torch

from .action_protocol import valid_actions,mask_groups,grouped_information
from .model import ReleasedModel


class ControlledActiveView(ReleasedModel):
    @torch.inference_mode()
    def plan(self,row,observed,reliability,actions,*,valid=None,costs=None,
             max_translation=3.0,policy='grouped'):
        if policy not in ('grouped','legacy','fixed'):
            raise ValueError(f'Unknown planning policy {policy!r}')
        a,v,c,d=valid_actions(actions,valid,costs,max_translation)
        state=self.current(row,observed,reliability)
        state.update(query=-1,cost=0.,distance_m=0.,budget_used=0,valid=v,
                     actions=a,movement_cost=c,query_utilities=np.full(len(a),-np.inf,np.float32),
                     planning_policy=policy,max_translation_m=float(max_translation))
        if not v.any():return state
        q,lv=self.future.predict(*state['geometry'],state['hypotheses'],state['observed'],self.tensor(a))
        if policy=='grouped':
            groups,duplicates,stats=mask_groups(state['masks'])
            utility=grouped_information(state['base'],state['log_site'],q,lv,
                                        state['reliability'],groups,duplicates)
            state['grouping']=stats
        elif policy=='legacy':
            utility=self.future.information(state['base'],state['log_site'],q,lv,state['reliability'])
        elif policy=='fixed':utility=-torch.arange(len(a),device=q.device,dtype=torch.float32)
        utility=utility.masked_fill(~self.tensor(v),-torch.inf)
        query=int(utility.argmax())
        state.update(query=query,prediction=q[query],log_variance=lv[query],
                     utilities=utility,query_utilities=utility.cpu().numpy(),cost=float(c[query]),
                     distance_m=float(d[query]),budget_used=1)
        return state

    def result(self,state,score):
        selected=int(score.argmax());posterior=score.softmax(-1).cpu().numpy()
        masks=state['masks'][:,state['inverse']]
        mask=masks[selected].astype(np.uint8)
        probability=(posterior@masks.astype(np.float32)).astype(np.float32)
        if state['silent']:mask[:]=0;probability[:]=0
        return dict(sound_mask=mask,point_probability=probability,posterior=posterior,
                    selected_object=np.int64(selected),
                    original_selected_object=np.int64(int(state['base'].argmax())))

    @torch.inference_mode()
    def complete(self,state,observed=None,reliability=None):
        if state['query']<0:result=self.result(state,state['base'])
        else:
            # A valid second segment may recover an initially silent observation.
            updated=dict(state);updated['silent']=state['silent'] and not bool(np.any(reliability))
            result=super().complete(updated,observed,reliability)
            result['prior_predicted_covariance']=torch.view_as_real(state['prediction']).cpu().numpy()
            result['prior_log_error_scale']=state['log_variance'].cpu().numpy()
        result.update(chosen_action=np.int64(state['query']),
                      query_utilities=state['query_utilities'],
                      action_valid=state['valid'],relative_actions=state['actions'],
                      movement_cost=state['movement_cost'],
                      chosen_movement_cost=np.float32(state['cost']),
                      distance_m=np.float32(state['distance_m']),
                      max_translation_m=np.float32(state['max_translation_m']),
                      budget_used=np.int64(state['budget_used']),
                      audio_observations=np.int64(1+state['budget_used']),
                      point_clouds=np.int64(1),
                      planning_policy=np.str_(state['planning_policy']))
        return result

    def run(self,row,observed,reliability,actions,acquire,**kwargs):
        plan=self.plan(row,observed,reliability,actions,**kwargs)
        if plan['query']<0:return self.complete(plan)
        second,weight=acquire(plan['query'])
        return self.complete(plan,second,weight)
