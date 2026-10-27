"""Frozen acoustic model with posterior-marginalized point scores for fusion."""
import numpy as np
import torch

from .model import ReleasedModel
from .fusion import validate_actions
from .action_protocol import mask_groups,grouped_information


class FusionModel(ReleasedModel):
    @torch.inference_mode()
    def plan(self, row, observed, reliability, actions, *, planning_policy='grouped'):
        """Select a supplied pose from pre-acquisition evidence only.

        Geometric groups make exact mask duplication invariant. They are proxies,
        not verified physical-instance labels. The legacy policy is retained for
        replay of the original release results.
        """
        actions=validate_actions(actions)
        state=self.current(row,observed,reliability)
        q,lv=self.future.predict(*state['geometry'],state['hypotheses'],state['observed'],self.tensor(actions))
        if planning_policy=='grouped':
            groups,duplicates,stats=mask_groups(state['masks'])
            utility=grouped_information(state['base'],state['log_site'],q,lv,
                                        state['reliability'],groups,duplicates)
            state['grouping']=stats
        elif planning_policy=='legacy':
            utility=self.future.information(state['base'],state['log_site'],q,lv,state['reliability'])
        else:raise ValueError(f'Unknown planning policy {planning_policy!r}')
        query=int(utility.argmax())
        state.update(prediction=q[query],log_variance=lv[query],query=query,
                     utilities=utility,planning_policy=planning_policy)
        return state

    def result(self, state, score):
        posterior = score.softmax(-1).detach().cpu().numpy()
        selected = int(posterior.argmax())
        masks = state['masks']
        inverse = state['inverse']
        mask = masks[selected][inverse].astype(np.uint8)
        # A marginal over candidate masks; not an empirically calibrated score.
        probability = (posterior @ masks.astype(np.float32))[inverse]
        if state['silent']:
            mask[:] = 0
            probability[:] = 0
        return dict(sound_mask=mask, point_probability=probability.astype(np.float32),
                    posterior=posterior, selected_object=selected,
                    original_selected_object=int(state['base'].argmax()))
