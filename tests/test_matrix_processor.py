import unittest

import numpy as np

from ratiss_entrelacs.matrix_processor import (
    DynamicMatrixProcessor,
    MatrixProcessorConfig,
    weighted_fold,
)


class WeightedFoldTests(unittest.TestCase):
    def test_fold_keeps_top_k_signed_weighted_relations(self):
        weights = np.array([[0.1, -0.8, 0.4], [0.0, 0.3, 0.0], [0.2, 0.0, 0.5]])
        folded, active = weighted_fold(
            weights, np.zeros(3), np.zeros(3), top_k=1,
            rhythm_strength=0.0, coactivity_floor=1.0,
            temperature=0.01, row_budget=2.0,
        )
        np.testing.assert_array_equal(active[0], [False, True, False])
        self.assertAlmostEqual(float(folded[0, 1]), -0.8)
        self.assertEqual(int(active.sum()), 3)

    def test_phase_compatibility_can_reconfigure_the_fold(self):
        weights = np.zeros((3, 3))
        weights[0, 1] = 0.8
        weights[0, 2] = 0.6
        _, active = weighted_fold(
            weights, np.zeros(3), [0.0, np.pi, 0.0], top_k=1,
            rhythm_strength=1.0, coactivity_floor=1.0,
            temperature=0.01, row_budget=2.0,
        )
        self.assertFalse(active[0, 1])
        self.assertTrue(active[0, 2])

    def test_fully_incompatible_relation_is_suppressed(self):
        weights = np.array([[0.0, 1.0], [0.0, 0.0]])
        folded, active = weighted_fold(
            weights, np.zeros(2), [0.0, np.pi], top_k=1,
            rhythm_strength=1.0, coactivity_floor=1.0,
            temperature=0.1, row_budget=1.0,
        )
        self.assertFalse(active[0, 1])
        self.assertEqual(float(folded[0, 1]), 0.0)


class DynamicMatrixProcessorTests(unittest.TestCase):
    def test_external_rhythm_gates_continuous_integration(self):
        config = MatrixProcessorConfig(
            units=4, top_k=2, dt=0.1, tau=0.1,
            rhythm_strength=1.0, matrix_learning_rate=0.0, seed=4,
        )
        input_weights = np.ones((4, 1))
        off = DynamicMatrixProcessor(1, config=config, input_weights=input_weights,
                                     initial_weights=np.zeros((4, 4)))
        on = DynamicMatrixProcessor(1, config=config, input_weights=input_weights,
                                    initial_weights=np.zeros((4, 4)))
        off_result = off.step([1.0], rhythm=0.0)
        on_result = on.step([1.0], rhythm=1.0)
        np.testing.assert_allclose(off_result.activity, 0.0)
        self.assertTrue(np.all(on_result.activity > 0.0))

    def test_local_matrix_rule_changes_weights_within_row_budget(self):
        config = MatrixProcessorConfig(
            units=3, top_k=2, dt=0.1, tau=0.2, rhythm_strength=0.0,
            matrix_learning_rate=1.0, weight_decay=0.0, row_budget=0.05, seed=1,
        )
        processor = DynamicMatrixProcessor(
            1, config=config, input_weights=np.ones((3, 1)),
            initial_weights=np.zeros((3, 3)),
        )
        initial = processor.weights
        processor.step([1.0])
        processor.step([1.0])
        self.assertFalse(np.allclose(initial, processor.weights))
        self.assertTrue(np.all(np.sum(np.abs(processor.weights), axis=1) <= 0.05 + 1e-12))

    def test_threshold_event_is_a_crossing_not_a_level_flag(self):
        config = MatrixProcessorConfig(
            units=2, top_k=1, dt=0.1, tau=0.2, rhythm_strength=0.0,
            matrix_learning_rate=0.0, threshold=0.1, seed=2,
        )
        processor = DynamicMatrixProcessor(
            1, config=config, input_weights=np.full((2, 1), 5.0),
            initial_weights=np.zeros((2, 2)),
        )
        first = processor.step([1.0])
        second = processor.step([1.0])
        self.assertTrue(np.all(first.events))
        self.assertFalse(np.any(second.events))

    def test_readout_calibration_reduces_error(self):
        processor = DynamicMatrixProcessor(
            2, output_dim=1,
            config=MatrixProcessorConfig(units=3, top_k=1, seed=3),
        )
        activity = np.array([1.0, 0.5, -0.25])
        target = np.array([0.7])
        initial_error = abs(float(target[0] - (processor.readout_weights @ activity)[0]))
        for _ in range(12):
            processor.calibrate_readout(target, activity=activity, learning_rate=0.5)
        final_error = abs(float(target[0] - (processor.readout_weights @ activity)[0]))
        self.assertLess(final_error, initial_error * 1e-3)

    def test_runs_are_reproducible_for_equal_seeds(self):
        config = MatrixProcessorConfig(units=8, top_k=3, seed=42)
        signals = np.random.default_rng(11).normal(size=(20, 3))
        first = DynamicMatrixProcessor(3, config=config).process(signals)
        second = DynamicMatrixProcessor(3, config=config).process(signals)
        np.testing.assert_allclose(first.activity, second.activity)
        np.testing.assert_allclose(first.final_weights, second.final_weights)
        np.testing.assert_array_equal(first.events, second.events)

    def test_reset_clears_state_but_keeps_adapted_matrix(self):
        processor = DynamicMatrixProcessor(
            1,
            config=MatrixProcessorConfig(units=2, top_k=1, matrix_learning_rate=0.5, rhythm_strength=0.0),
            input_weights=np.ones((2, 1)), initial_weights=np.zeros((2, 2)),
        )
        processor.step([1.0])
        processor.step([1.0])
        learned = processor.weights
        processor.reset()
        np.testing.assert_allclose(processor.activity, 0.0)
        np.testing.assert_allclose(processor.weights, learned)

    def test_toy_pattern_association_runs_through_matrix_and_readout(self):
        config = MatrixProcessorConfig(
            units=16, top_k=4, dt=0.1, tau=0.5, rhythm_strength=0.5,
            matrix_learning_rate=0.02, weight_decay=0.01, seed=17,
        )
        processor = DynamicMatrixProcessor(3, output_dim=2, config=config)
        patterns = (
            (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0])),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0])),
        )
        for _ in range(20):
            for signal, target in patterns:
                processor.reset()
                for _ in range(6):
                    result = processor.step(signal, learn_matrix=True)
                processor.calibrate_readout(target, activity=result.activity, learning_rate=0.3)
        predictions = []
        for signal, _ in patterns:
            processor.reset()
            for _ in range(6):
                result = processor.step(signal, learn_matrix=False)
            predictions.append(result.output)
        expected = np.stack([target for _, target in patterns])
        rmse = float(np.sqrt(np.mean((np.stack(predictions) - expected) ** 2)))
        self.assertLess(rmse, 0.02)

    def test_same_final_input_can_read_different_histories(self):
        config = MatrixProcessorConfig(
            units=16, top_k=4, dt=0.1, tau=0.5, rhythm_strength=0.5,
            matrix_learning_rate=0.02, weight_decay=0.01, seed=17,
        )
        processor = DynamicMatrixProcessor(3, output_dim=2, config=config)
        first = np.array([1.0, 0.0, 0.0])
        second = np.array([0.0, 1.0, 0.0])
        shared_final = np.array([0.0, 0.0, 1.0])
        sequences = (
            (np.stack([first] * 4 + [shared_final] * 4), np.array([1.0, 0.0])),
            (np.stack([second] * 4 + [shared_final] * 4), np.array([0.0, 1.0])),
        )
        for _ in range(30):
            for sequence, target in sequences:
                processor.reset()
                for signal in sequence:
                    result = processor.step(signal, learn_matrix=True)
                processor.calibrate_readout(target, activity=result.activity, learning_rate=0.3)
        predictions = []
        states = []
        for sequence, _ in sequences:
            processor.reset()
            for signal in sequence:
                result = processor.step(signal, learn_matrix=False)
            predictions.append(result.output)
            states.append(result.activity)
        expected = np.stack([target for _, target in sequences])
        rmse = float(np.sqrt(np.mean((np.stack(predictions) - expected) ** 2)))
        self.assertLess(rmse, 0.1)
        self.assertGreater(float(np.linalg.norm(states[0] - states[1])), 0.1)

    def test_invalid_dimensions_and_rhythm_values_are_rejected(self):
        with self.assertRaises(ValueError):
            DynamicMatrixProcessor(0)
        processor = DynamicMatrixProcessor(2, config=MatrixProcessorConfig(units=4, top_k=2))
        with self.assertRaises(ValueError):
            processor.step([0.0, 1.0], rhythm=1.5)
        with self.assertRaises(ValueError):
            processor.process(np.zeros((3, 2)), rhythms=np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            processor.process(np.zeros((3, 2)), targets=np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            processor.step([1.0 + 1.0j, 0.0])


if __name__ == "__main__":
    unittest.main()
