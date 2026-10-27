from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'snapshots/generator')]
from common_menu_backend import validate_response, CommonMenuBackend


class ResponseTests(unittest.TestCase):
    def test_preserves_zero_and_nonzero_simulator_output(self):
        for response in [np.zeros((1, 12), np.float32), np.ones((1, 12), np.float32)]:
            calls = []
            backend = object.__new__(CommonMenuBackend)
            backend.channel_count = 1
            backend.pose = lambda position, yaw: calls.append((position, yaw))
            backend.audio = type('Audio', (), {'setAudioSourceTransform': lambda self, source: calls.append(source)})()
            backend.sim = type('Sim', (), {'get_sensor_observations': lambda self: {'audio_sensor': response}})()
            actual = backend.render([1, 2, 3], [4, 5, 6], .5)
            np.testing.assert_array_equal(actual, response)
            self.assertEqual(len(calls), 2)

    def test_rejects_malformed_and_nonfinite_responses(self):
        for response in [np.zeros((1, 0)), np.zeros((2, 12)), np.zeros(12), [[np.nan]], [[np.inf]]]:
            with self.assertRaises(RuntimeError):
                validate_response(response, 1)


if __name__ == '__main__':
    unittest.main()
