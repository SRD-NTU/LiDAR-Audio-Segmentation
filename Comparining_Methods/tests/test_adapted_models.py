"""Sensor geometry and action-order checks for the tetra/LiDAR adaptations."""
import unittest

import numpy as np
import torch

from adapters.active_menu import pose_only_menu, train_parallax_teacher
from adapters.audiolidar import AudioLiDAR
from models.passive import tetra_to_lowband_foa


class AdaptedModelTest(unittest.TestCase):
    def test_tetra_gradient_points_toward_synthetic_plane_wave(self):
        data = AudioLiDAR()
        microphones = torch.tensor(data.calibration["positions_lidar_m"], dtype=torch.float32)
        rate, frequency, speed = 16000, 400, 343
        time = torch.arange(16000, dtype=torch.float32) / rate
        # A wave from +x arrives first at the +x microphone.
        delays = microphones[:, 0] / speed
        wave = torch.sin(2 * torch.pi * frequency * (time[None] + delays[:, None]))[None]
        foa = tetra_to_lowband_foa(wave, microphones)
        w, x, y, z = foa[0]
        self.assertGreater(torch.corrcoef(torch.stack([w[1000:], x[1000:]]))[0, 1].item(), .9)
        self.assertLess(y[1000:].square().mean().item(), .02 * x[1000:].square().mean().item())
        self.assertLess(z[1000:].square().mean().item(), .02 * x[1000:].square().mean().item())

    def test_pose_menu_contains_only_pose_features_and_rejects_test_teacher(self):
        data = AudioLiDAR()
        initial, menu = next((sample, options) for sample in data.samples("val")
                             if (options := pose_only_menu(data, sample)))
        self.assertTrue(menu)
        self.assertEqual(set(menu[0]), {"sample", "pose", "features", "distance_m"})
        self.assertEqual(menu[0]["features"].shape, (13,))
        self.assertTrue(np.isfinite(menu[0]["features"]).all())
        test, test_menu = next((sample, options) for sample in data.samples("test")
                               if (options := pose_only_menu(data, sample)))
        with self.assertRaises(ValueError):
            train_parallax_teacher(data, test, test_menu)


if __name__ == "__main__":
    unittest.main()
