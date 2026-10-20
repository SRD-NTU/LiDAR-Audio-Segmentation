"""Frozen acoustic model with posterior-marginalized point scores for fusion."""
import numpy as np

from .model import ReleasedModel
from .fusion import validate_actions


class FusionModel(ReleasedModel):
    def plan(self, row, observed, reliability, actions):
        return super().plan(row, observed, reliability, validate_actions(actions))

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
