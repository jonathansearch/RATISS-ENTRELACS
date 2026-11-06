import unittest

import numpy as np

from ratiss_entrelacs.couplings import density_phase_bias, local_density, phase_to_radial_anchors


class CrossDomainCouplingTests(unittest.TestCase):
    def test_phase_changes_radial_anchor_geometry(self):
        phases = np.zeros(8)
        baseline = phase_to_radial_anchors(phases)
        phases[0] = np.pi / 2
        displaced = phase_to_radial_anchors(phases)
        self.assertAlmostEqual(float(np.linalg.norm(baseline[0])), 2.2, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(displaced[0])), 2.35, places=12)
        np.testing.assert_allclose(baseline[1:], displaced[1:])

    def test_geometry_density_can_feed_back_to_phase_bias(self):
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        environment = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        density = local_density(positions, environment, neighbors=2)
        self.assertAlmostEqual(float(density[0]), 1.0, places=12)
        self.assertAlmostEqual(float(density[1]), 0.0, places=12)
        np.testing.assert_allclose(density_phase_bias(density), [0.5, -0.5])

    def test_uniform_density_has_zero_bias(self):
        np.testing.assert_allclose(density_phase_bias([0.5, 0.5, 0.5], gain=3.0), 0.0)

    def test_invalid_cross_domain_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            phase_to_radial_anchors([0.0, np.nan, 1.0])
        with self.assertRaises(ValueError):
            local_density([[0, 0]], [[1, 2, 3]])
        with self.assertRaises(ValueError):
            density_phase_bias([1.2])


if __name__ == "__main__":
    unittest.main()
