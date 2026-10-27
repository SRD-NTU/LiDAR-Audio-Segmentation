"""Retain raw zero-energy responses without conditioning menus on acoustic success."""
import numpy as np
from audiolidar.habitat.backend import Backend


def validate_response(result, channels):
    result = np.asarray(result, dtype='float32')
    if (result.ndim != 2 or result.shape[0] != channels or
            result.shape[1] == 0 or not np.isfinite(result).all()):
        raise RuntimeError('Malformed SoundSpaces response: shape=%s, finite=%s' %
                           (result.shape, bool(np.isfinite(result).all())))
    return result


class CommonMenuBackend(Backend):
    def render(self, source, position, yaw):
        self.pose(position, yaw)
        self.audio.setAudioSourceTransform(np.asarray(source, dtype=float))
        result = self.sim.get_sensor_observations()['audio_sensor']
        # No fabricated signal, microphone displacement, or source-specific retry.
        # Finite zeros are recorded explicitly by the generator for later auditing.
        return validate_response(result, self.channel_count)
