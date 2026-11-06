import unittest

import numpy as np

from ratiss_entrelacs.topology import psig_from_correlation, psig_from_points, rips_persistence


class TopologyTests(unittest.TestCase):
    def test_square_point_cloud_has_a_finite_h1_cycle(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = rips_persistence(np.linalg.norm(square[:, None] - square[None, :], axis=2))
        self.assertGreater(result["psig"], 0.0)
        self.assertEqual(result["n_finite_h1"], 1)
        self.assertAlmostEqual(psig_from_points(square), result["psig"])

    def test_collapsed_points_have_no_finite_cycle(self):
        points = np.zeros((5, 3))
        self.assertEqual(psig_from_points(points), 0.0)

    def test_correlation_profiles_can_be_read(self):
        corr = np.eye(4, dtype=np.complex128)
        self.assertGreaterEqual(psig_from_correlation(corr), 0.0)

    def test_bad_distance_matrices_are_rejected(self):
        with self.assertRaises(ValueError):
            rips_persistence(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            rips_persistence([[0, 1], [2, 0]])
        with self.assertRaises(ValueError):
            rips_persistence([[1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
