import unittest

import numpy as np

from ratiss_entrelacs.cells import DendriticConfig, DendriticPopulation
from ratiss_entrelacs.matrix_processor import (
    DynamicMatrixProcessor,
    MatrixProcessorConfig,
    STDPConfig,
    weighted_fold_channels,
)


class BranchAndPathwayTests(unittest.TestCase):
    def test_dendritic_branches_keep_separate_ei_currents_and_local_transfer(self):
        cells = DendriticPopulation(
            1,
            DendriticConfig(
                tau_dendrite=0.1, tau_membrane=0.1, tau_adaptation=1.0,
                threshold=10.0, branch_nonlinearity="tanh",
            ),
            branches=2,
            branch_gains=[1.0, 2.0],
        )
        result = cells.step(
            excitatory_current=[[2.0, 0.0]],
            inhibitory_current=[[0.0, 1.0]],
            dt=0.1,
        )
        np.testing.assert_allclose(result.dendritic_potential, [[2.0, -1.0]])
        np.testing.assert_allclose(result.excitatory_current, [[2.0, 0.0]])
        np.testing.assert_allclose(result.inhibitory_current, [[0.0, 1.0]])
        expected_soma_drive = np.tanh(2.0) - 2.0 * np.tanh(1.0)
        self.assertAlmostEqual(float(result.pre_reset_membrane[0]), expected_soma_drive)
        self.assertEqual(result.dendritic_potential.shape, (1, 2))

    def test_channel_fold_keeps_simultaneous_excitation_and_inhibition(self):
        excitatory = np.array([[0.0, 0.8], [0.0, 0.0]])
        inhibitory = np.array([[0.0, 0.4], [0.0, 0.0]])
        folded_e, folded_i, active_e, active_i = weighted_fold_channels(
            excitatory, inhibitory, np.zeros(2), np.zeros(2),
            top_k=1, rhythm_strength=0.0, coactivity_floor=1.0,
            temperature=0.01, row_budget=2.0,
        )
        self.assertTrue(active_e[0, 1])
        self.assertTrue(active_i[0, 1])
        self.assertAlmostEqual(float(folded_e[0, 1]), 0.8)
        self.assertAlmostEqual(float(folded_i[0, 1]), 0.4)

    def test_branch_specific_input_is_preserved_in_processor_trace(self):
        layer = DendriticPopulation(
            2,
            DendriticConfig(
                tau_dendrite=0.1, tau_membrane=0.1, tau_adaptation=1.0,
                threshold=10.0,
            ),
            branches=2,
        )
        branch_input = np.zeros((2, 2, 1))
        branch_input[0, 0, 0] = 1.0
        branch_input[1, 1, 0] = 1.0
        processor = DynamicMatrixProcessor(
            1,
            config=MatrixProcessorConfig(
                units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
                matrix_learning_rate=0.0,
            ),
            dendritic_layer=layer,
            input_branch_weights=branch_input,
            initial_weights=np.zeros((2, 2)),
        )
        result = processor.step([1.0])
        self.assertEqual(result.compartments["dendritic_potential"].shape, (2, 2))
        np.testing.assert_allclose(
            result.compartments["dendritic_potential"], [[1.0, 0.0], [0.0, 1.0]]
        )
        trace = processor.process(np.ones((2, 1)), learn_matrix=False)
        self.assertEqual(trace.compartments["dendritic_potential"].shape, (2, 2, 2))

    def test_independent_ei_matrices_drive_soma_and_obey_combined_budget(self):
        layer = DendriticPopulation(
            2,
            DendriticConfig(
                tau_dendrite=0.1, tau_membrane=0.1, tau_adaptation=1.0,
                threshold=10.0,
            ),
        )
        config = MatrixProcessorConfig(
            units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
            matrix_learning_rate=0.0, row_budget=1.0,
        )
        processor = DynamicMatrixProcessor(
            1,
            config=config,
            dendritic_layer=layer,
            input_weights=[[1.0], [0.0]],
            initial_excitatory_weights=[[0.0, 0.0], [0.8, 0.0]],
            initial_inhibitory_weights=[[0.0, 0.0], [0.3, 0.0]],
        )
        self.assertGreater(float(processor.excitatory_weights[1, 0]), 0.0)
        self.assertGreater(float(processor.inhibitory_weights[1, 0]), 0.0)
        self.assertLessEqual(
            float(np.sum(processor.excitatory_weights[1] + processor.inhibitory_weights[1])),
            1.0 + 1e-12,
        )
        processor.step([1.0])
        result = processor.step([0.0])
        e_drive = float(result.compartments["excitatory_current"][1])
        i_drive = float(result.compartments["inhibitory_current"][1])
        self.assertGreater(e_drive, i_drive)
        self.assertGreater(float(result.activity[1]), 0.0)
        self.assertAlmostEqual(float(processor.weights[1, 0]), e_drive - i_drive)

    def test_coactivity_update_preserves_coexisting_ei_channels(self):
        processor = DynamicMatrixProcessor(
            2,
            config=MatrixProcessorConfig(
                units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
                matrix_learning_rate=1.0, weight_decay=0.0,
            ),
            input_weights=np.eye(2) * 5.0,
            initial_excitatory_weights=[[0.0, 0.0], [0.2, 0.0]],
            initial_inhibitory_weights=[[0.0, 0.0], [0.1, 0.0]],
        )
        processor.step([1.0, 1.0])
        processor.step([1.0, 1.0])
        self.assertGreater(float(processor.excitatory_weights[1, 0]), 0.2)
        self.assertAlmostEqual(float(processor.inhibitory_weights[1, 0]), 0.1)
        self.assertAlmostEqual(
            float(processor.weights[1, 0]),
            float(processor.excitatory_weights[1, 0] - processor.inhibitory_weights[1, 0]),
        )

    def test_invalid_branch_and_pathway_shapes_or_signs_are_rejected(self):
        cells = DendriticPopulation(2, branches=2)
        with self.assertRaises(ValueError):
            cells.step(np.zeros((2, 3)), dt=0.05)
        with self.assertRaises(ValueError):
            cells.step(excitatory_current=[-1.0, 0.0], dt=0.05)
        with self.assertRaises(ValueError):
            DynamicMatrixProcessor(
                1,
                config=MatrixProcessorConfig(units=2, top_k=1),
                initial_excitatory_weights=-np.ones((2, 2)),
            )
        with self.assertRaises(ValueError):
            DynamicMatrixProcessor(
                1,
                config=MatrixProcessorConfig(units=2, top_k=1),
                initial_weights=np.zeros((2, 2)),
                initial_inhibitory_weights=np.zeros((2, 2)),
            )
        with self.assertRaises(ValueError):
            MatrixProcessorConfig(units=2, top_k=1, plasticity_rule="unknown")


class STDPTests(unittest.TestCase):
    def _processor(self, *, inhibitory=False):
        config = MatrixProcessorConfig(
            units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
            matrix_learning_rate=1.0, weight_decay=0.0, threshold=0.5,
            plasticity_rule="stdp", row_budget=1.0,
        )
        initial = np.zeros((2, 2))
        initial[1, 0] = 0.2
        options = (
            {"initial_inhibitory_weights": initial}
            if inhibitory else {"initial_excitatory_weights": initial}
        )
        return DynamicMatrixProcessor(
            2,
            config=config,
            input_weights=[[10.0, 0.0], [0.0, 10.0]],
            stdp_config=STDPConfig(
                tau_pre=0.5, tau_post=0.5,
                a_plus=0.2, a_minus=0.2,
                inhibitory_scale=1.5,
            ),
            **options,
        )

    def test_pre_before_post_potentiates_excitatory_weight(self):
        processor = self._processor()
        before = float(processor.excitatory_weights[1, 0])
        pre = processor.step([1.0, 0.0])
        post = processor.step([0.0, 1.0])
        self.assertTrue(bool(pre.events[0]))
        self.assertTrue(bool(post.events[1]))
        self.assertGreater(float(processor.excitatory_weights[1, 0]), before)
        self.assertGreater(float(post.compartments["pre_spike_trace"][0]), 0.0)

    def test_post_before_pre_depresses_excitatory_weight(self):
        processor = self._processor()
        before = float(processor.excitatory_weights[1, 0])
        post = processor.step([0.0, 1.0])
        pre = processor.step([1.0, 0.0])
        self.assertTrue(bool(post.events[1]))
        self.assertTrue(bool(pre.events[0]))
        self.assertGreaterEqual(float(processor.excitatory_weights[1, 0]), 0.0)
        self.assertLess(float(processor.excitatory_weights[1, 0]), before)

    def test_inhibitory_pathway_has_its_own_timing_scaled_update(self):
        processor = self._processor(inhibitory=True)
        before = float(processor.inhibitory_weights[1, 0])
        processor.step([1.0, 0.0])
        processor.step([0.0, 1.0])
        inhibitory_after = float(processor.inhibitory_weights[1, 0])
        excitatory_after = float(processor.excitatory_weights[1, 0])
        self.assertGreater(inhibitory_after, before)
        self.assertGreater(excitatory_after, 0.0)
        self.assertAlmostEqual(inhibitory_after - before, 1.5 * excitatory_after, places=6)

    def test_simultaneous_events_have_no_zero_lag_pair_update_and_reset_clears_traces(self):
        processor = self._processor()
        before = processor.excitatory_weights
        result = processor.step([1.0, 1.0])
        np.testing.assert_allclose(processor.excitatory_weights, before)
        self.assertTrue(np.all(result.events))
        self.assertTrue(np.any(processor.pre_spike_trace))
        processor.reset()
        np.testing.assert_allclose(processor.pre_spike_trace, 0.0)
        np.testing.assert_allclose(processor.post_spike_trace, 0.0)
        np.testing.assert_allclose(processor.excitatory_weights, before)

    def test_stdp_parameter_values_are_validated(self):
        with self.assertRaises(ValueError):
            STDPConfig(tau_pre=0.0)
        with self.assertRaises(ValueError):
            STDPConfig(a_minus=-0.1)


if __name__ == "__main__":
    unittest.main()
