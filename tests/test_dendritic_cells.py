import unittest

import numpy as np

from ratiss_entrelacs.cells import DendriticConfig, DendriticPopulation
from ratiss_entrelacs.matrix_processor import DynamicMatrixProcessor, MatrixProcessorConfig


class DendriticPopulationTests(unittest.TestCase):
    def test_separate_dendrite_soma_and_adaptation_generate_spikes(self):
        cells = DendriticPopulation(
            1,
            DendriticConfig(
                tau_dendrite=0.1, tau_membrane=0.2, tau_adaptation=1.0,
                adaptation_gain=0.2, adaptation_jump=0.4,
                threshold=0.5, reset_value=0.0,
            ),
        )
        first = cells.step([1.0], dt=0.1)
        self.assertTrue(bool(first.events[0]))
        self.assertAlmostEqual(float(first.dendritic_potential[0]), 1.0)
        self.assertAlmostEqual(float(first.pre_reset_membrane[0]), 0.5)
        self.assertEqual(float(first.membrane[0]), 0.0)
        self.assertGreater(float(first.adaptation[0]), 0.0)

        second = cells.step([1.0], dt=0.1)
        self.assertFalse(bool(second.events[0]))
        self.assertGreater(float(second.membrane[0]), 0.0)

    def test_inhibitory_current_and_rhythm_gate_affect_dendrite(self):
        cells = DendriticPopulation(2, DendriticConfig(threshold=2.0))
        inhibited = cells.step([-1.0, 1.0], dt=0.1, rhythm_gate=[1.0, 0.0])
        self.assertLess(float(inhibited.dendritic_potential[0]), 0.0)
        self.assertEqual(float(inhibited.dendritic_potential[1]), 0.0)
        self.assertFalse(np.any(inhibited.events))

    def test_reset_and_bad_time_step(self):
        cells = DendriticPopulation(2)
        cells.step([1.0, 1.0], dt=0.1)
        cells.reset()
        np.testing.assert_allclose(cells.dendritic_potential, 0.0)
        np.testing.assert_allclose(cells.membrane, 0.0)
        np.testing.assert_allclose(cells.adaptation, 0.0)
        with self.assertRaises(ValueError):
            cells.step([1.0, 0.0], dt=0.3)

    def test_matrix_processor_can_use_compartment_dynamics_and_trace_them(self):
        layer = DendriticPopulation(
            4,
            DendriticConfig(
                tau_dendrite=0.1, tau_membrane=0.2, tau_adaptation=1.0,
                threshold=0.25, reset_value=0.0,
            ),
        )
        config = MatrixProcessorConfig(
            units=4, top_k=2, dt=0.05, tau=0.5,
            rhythm_strength=0.0, matrix_learning_rate=0.0, seed=8,
        )
        processor = DynamicMatrixProcessor(
            1,
            config=config,
            dendritic_layer=layer,
            input_weights=np.ones((4, 1)),
            initial_weights=np.zeros((4, 4)),
        )
        trace = processor.process(np.ones((8, 1)))
        self.assertEqual(trace.activity.shape, (8, 4))
        self.assertEqual(trace.compartments["dendritic_potential"].shape, (8, 4))
        self.assertEqual(trace.compartments["pre_reset_membrane"].shape, (8, 4))
        self.assertEqual(trace.compartments["adaptation"].shape, (8, 4))
        self.assertTrue(np.any(trace.events))


if __name__ == "__main__":
    unittest.main()
