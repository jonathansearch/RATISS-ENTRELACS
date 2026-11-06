import unittest

import numpy as np

from ratiss_entrelacs.domains.ssh_reference import (
    correlation_matrix,
    edge_weight,
    evolve_orbitals,
    ground_state_orbitals,
    spectral_gap,
    ssh_hamiltonian,
)


class SSHReferenceTests(unittest.TestCase):
    def test_hamiltonian_is_hermitian_and_has_expected_shape(self):
        h = ssh_hamiltonian(8, delta=-0.2)
        self.assertEqual(h.shape, (8, 8))
        np.testing.assert_allclose(h, h.T)
        self.assertEqual(np.count_nonzero(np.triu(h, 1)), 7)

    def test_ground_state_orbitals_are_orthonormal_and_half_filled(self):
        h = ssh_hamiltonian(8, delta=0.3)
        orbitals = ground_state_orbitals(h)
        self.assertEqual(orbitals.shape, (8, 4))
        np.testing.assert_allclose(orbitals.conj().T @ orbitals, np.eye(4), atol=1e-12)
        corr = correlation_matrix(orbitals)
        self.assertAlmostEqual(float(np.trace(corr).real), 4.0, places=12)
        self.assertGreaterEqual(spectral_gap(h), 0.0)
        self.assertGreaterEqual(edge_weight(corr), 0.0)

    def test_rk4_evolution_preserves_norm_with_small_step(self):
        orbitals = ground_state_orbitals(ssh_hamiltonian(6, 0.2))
        trajectory = evolve_orbitals(orbitals, lambda t: 0.2 + 0.001 * t, dt=0.002, steps=20)
        self.assertEqual(trajectory.shape, (21, 6, 3))
        norm0 = np.linalg.norm(trajectory[0])
        norms = np.linalg.norm(trajectory, axis=(1, 2))
        self.assertLess(float(np.max(np.abs(norms - norm0))), 1e-9)

    def test_invalid_odd_chain_is_rejected(self):
        with self.assertRaises(ValueError):
            ssh_hamiltonian(5, 0.1)


if __name__ == "__main__":
    unittest.main()
