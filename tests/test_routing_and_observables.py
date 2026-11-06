import unittest

import numpy as np

from ratiss_entrelacs.metrics import graph_cycle_rank, hysteresis_area, phase_coherence
from ratiss_entrelacs.routing import sparse_mix, topk_route


class CrossDomainObservableTests(unittest.TestCase):
    def test_phase_order_control_cases(self):
        self.assertAlmostEqual(phase_coherence(np.zeros(8)), 1.0, places=12)
        uniform = np.arange(8) * 2 * np.pi / 8
        self.assertLess(phase_coherence(uniform), 1e-12)

    def test_cycle_rank_on_tree_ring_and_disconnected_graph(self):
        tree = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        ring = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        ring[0, 2] = ring[2, 0] = 1
        disconnected = np.zeros((5, 5), dtype=int)
        disconnected[:3, :3] = 1 - np.eye(3, dtype=int)  # triangle + two isolates
        self.assertEqual(graph_cycle_rank(tree), 0)
        self.assertEqual(graph_cycle_rank(ring), 1)
        self.assertEqual(graph_cycle_rank(disconnected), 1)

    def test_hysteresis_area_zero_for_same_path_and_positive_for_loop(self):
        x = np.linspace(0, 1, 9)
        self.assertAlmostEqual(hysteresis_area(x, x, x[::-1], x[::-1]), 0.0, places=12)
        area = hysteresis_area(x, x, x[::-1], -x[::-1])
        self.assertAlmostEqual(area, 1.0, places=12)

    def test_sparse_router_is_stable_and_weights_sum_to_one(self):
        indices, weights = topk_route([0.5, 1.0, 1.0, -2.0], 2)
        np.testing.assert_array_equal(indices, [1, 2])  # stable tie break
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertGreater(weights[0], 0.0)
        self.assertGreater(weights[1], 0.0)

    def test_sparse_mix_uses_only_the_selected_paths(self):
        values = np.array([[10.0], [20.0], [100.0]])
        mixed, indices, weights = sparse_mix(values, [3.0, 2.0, -5.0], k=2)
        self.assertEqual(set(indices.tolist()), {0, 1})
        self.assertAlmostEqual(float(mixed[0]), float(10 * weights[0] + 20 * weights[1]))
        self.assertLess(float(mixed[0]), 100.0)

    def test_invalid_routing_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            topk_route([1.0, np.nan], 1)
        with self.assertRaises(ValueError):
            topk_route([1.0], 0)
        with self.assertRaises(ValueError):
            topk_route([1.0], 1, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
