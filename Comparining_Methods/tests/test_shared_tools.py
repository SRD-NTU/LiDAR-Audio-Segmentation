import unittest

import numpy as np

from adapters.association import direction_scores, panoramic_scores
from evaluation.scoring import foreground_iou, selected_candidate_accuracy, aggregate


class SharedToolsTest(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[2, 0, 0], [2, .1, 0], [-2, 0, 0], [-2, -.1, 0]], dtype=float)
        self.candidates = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)

    def test_direction_selects_extent_in_correct_direction(self):
        scores = direction_scores(self.xyz, self.candidates, np.array([1, 0, 0]), 10)
        self.assertGreater(scores[0], scores[1])
        with self.assertRaises(ValueError):
            direction_scores(self.xyz, self.candidates, np.zeros(3), 10)

    def test_panoramic_map_wrap_and_orientation(self):
        image = np.zeros((5, 8), dtype=float)
        image[2, 4] = 1.0  # x-forward, azimuth 0, elevation 0
        scores = panoramic_scores(self.xyz, self.candidates, image)
        self.assertGreater(scores[0], scores[1])

    def test_scoring_is_aligned_and_scene_clustered(self):
        truth = self.candidates[0]
        self.assertEqual(foreground_iou(truth, truth), 1.0)
        self.assertEqual(selected_candidate_accuracy(self.candidates, 0, truth), 1.0)
        self.assertEqual(selected_candidate_accuracy(self.candidates, 1, truth), 0.0)
        with self.assertRaises(ValueError):
            foreground_iou(truth[:-1], truth)
        summary = aggregate([{"scene": "a", "foreground_iou": 1.0,
                              "candidate_selection_accuracy": 1.0},
                             {"scene": "b", "foreground_iou": 0.0,
                              "candidate_selection_accuracy": 0.0}], draws=100)
        self.assertEqual(summary["mean_iou"], 0.5)
        self.assertEqual(summary["scenes"], 2)


if __name__ == "__main__":
    unittest.main()
