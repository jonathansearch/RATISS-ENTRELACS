import unittest

import numpy as np

from ratiss_entrelacs.engine import run
from ratiss_entrelacs.metrics import phase_coherence
from ratiss_entrelacs.baselines.phase_ring import ForcedRing, ring_coupling


class ForcedRingTests(unittest.TestCase):
    def test_phase_coherence_known_cases(self):
        self.assertAlmostEqual(phase_coherence(np.zeros(8)), 1.0, places=12)
        spread = np.arange(8) * (2.0 * np.pi / 8.0)
        self.assertLess(phase_coherence(spread), 1e-12)

    def test_internal_ring_coupling_cannot_change_total_phase_by_itself(self):
        phases = np.array([-2.1, -0.7, 0.2, 1.6, 2.4, 0.9])
        self.assertAlmostEqual(float(np.sum(ring_coupling(phases))), 0.0, places=12)

    def test_no_interaction_no_drive_no_noise_is_exact_linear_drift(self):
        omega = np.array([1.0, 2.0, -0.5])
        initial = np.array([0.2, -0.1, 1.3])
        dt = 0.05
        model = ForcedRing(omega)
        result = run(model, initial, steps=7, dt=dt, seed=11)
        expected = initial[None, :] + result.times[:, None] * omega[None, :]
        np.testing.assert_allclose(result.states, expected, atol=1e-13)

    def test_seed_replays_noisy_trajectory(self):
        model = ForcedRing.homogeneous(8, frequency=1.0, coupling=0.4, noise_sigma=0.2)
        initial = np.linspace(-1.0, 1.0, 8)
        first = run(model, initial, steps=20, dt=0.01, seed=99)
        second = run(model, initial, steps=20, dt=0.01, seed=99)
        np.testing.assert_array_equal(first.states, second.states)

    def test_invalid_network_and_state_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            ForcedRing.homogeneous(2)
        model = ForcedRing.homogeneous(4)
        with self.assertRaisesRegex(ValueError, "state shape"):
            run(model, np.zeros(3), steps=1, dt=0.01)

    def test_inputs_are_not_mutated(self):
        initial = np.array([0.1, 0.2, 0.3])
        saved = initial.copy()
        result = run(ForcedRing.homogeneous(3, coupling=0.2), initial, steps=4, dt=0.02)
        np.testing.assert_array_equal(initial, saved)
        self.assertFalse(np.shares_memory(result.states, initial))


if __name__ == "__main__":
    unittest.main()
