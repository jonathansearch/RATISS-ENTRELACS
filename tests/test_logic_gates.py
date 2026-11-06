import unittest

import numpy as np

from ratiss_entrelacs.logic import LogicGate, LogicGateBindings, LogicGateLayer
from ratiss_entrelacs.matrix_processor import DynamicMatrixProcessor, MatrixProcessorConfig


class LogicGateLayerTests(unittest.TestCase):
    def test_boolean_truth_functions_and_chained_gate(self):
        circuit = LogicGateLayer(
            3,
            [
                LogicGate("both", "AND", (0, 1)),
                LogicGate("either", "OR", ("both", 2)),
                LogicGate("exclusive", "XOR", (0, 1)),
                LogicGate("not_first", "NOT", (0,)),
            ],
        )
        values = circuit.evaluate([1, 1, 0])
        self.assertEqual(values, {"both": True, "either": True, "exclusive": False, "not_first": False})
        self.assertEqual(circuit.evaluate([1, 0, 0])["either"], False)

    def test_invalid_gate_graph_and_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            LogicGateLayer(2, [LogicGate("bad", "AND", (0, "later"))])
        with self.assertRaises(ValueError):
            LogicGateLayer(2, [LogicGate("bad", "NOT", (0, 1))])
        circuit = LogicGateLayer(2, [LogicGate("both", "AND", (0, 1))])
        with self.assertRaises(ValueError):
            circuit.evaluate([1, 2])
        with self.assertRaises(ValueError):
            circuit.evaluate([1])


class ProcessorGateFeedbackTests(unittest.TestCase):
    def test_events_control_next_tick_routing_learning_and_readout(self):
        config = MatrixProcessorConfig(
            units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
            matrix_learning_rate=1.0, weight_decay=0.0, threshold=0.1, seed=5,
        )
        circuit = LogicGateLayer(2, [LogicGate("both_crossed", "AND", (0, 1))])
        processor = DynamicMatrixProcessor(
            1,
            output_dim=1,
            config=config,
            input_weights=np.full((2, 1), 5.0),
            initial_weights=np.array([[0.0, 0.5], [0.5, 0.0]]),
            readout_weights=np.ones((1, 2)),
            logic_layer=circuit,
            gate_bindings=LogicGateBindings(
                route="both_crossed", plasticity="both_crossed", readout="both_crossed"
            ),
        )
        initial_weights = processor.weights
        first = processor.step([1.0])
        self.assertFalse(first.logic_outputs["both_crossed"])
        self.assertEqual(first.active_edges, 0)
        self.assertEqual(float(first.output[0]), 0.0)
        self.assertTrue(np.all(first.events))
        np.testing.assert_allclose(processor.weights, initial_weights)

        second = processor.step([1.0])
        self.assertTrue(second.logic_outputs["both_crossed"])
        self.assertEqual(second.active_edges, 2)
        self.assertGreater(float(second.output[0]), 0.0)
        self.assertFalse(np.allclose(processor.weights, initial_weights))

    def test_process_records_gate_outputs_per_tick(self):
        config = MatrixProcessorConfig(
            units=2, top_k=1, dt=0.1, tau=0.1, rhythm_strength=0.0,
            matrix_learning_rate=0.0, threshold=0.1,
        )
        circuit = LogicGateLayer(2, [LogicGate("both", "AND", (0, 1))])
        processor = DynamicMatrixProcessor(
            1, config=config, input_weights=np.full((2, 1), 5.0),
            initial_weights=np.array([[0.0, 0.4], [0.4, 0.0]]), logic_layer=circuit,
            gate_bindings=LogicGateBindings(route="both"),
        )
        trace = processor.process(np.ones((3, 1)))
        np.testing.assert_array_equal(trace.logic_outputs["both"], [False, True, False])
        np.testing.assert_array_equal(trace.active_edges, [0, 2, 0])

    def test_reset_clears_event_memory_and_binding_names_are_checked(self):
        config = MatrixProcessorConfig(units=2, top_k=1)
        circuit = LogicGateLayer(2, [LogicGate("go", "OR", (0, 1))])
        processor = DynamicMatrixProcessor(
            1, config=config, logic_layer=circuit,
            gate_bindings=LogicGateBindings(route="go"),
        )
        processor.step([1.0])
        processor.reset()
        self.assertFalse(processor.step([0.0]).logic_outputs["go"])
        with self.assertRaises(ValueError):
            DynamicMatrixProcessor(
                1, config=config, logic_layer=circuit,
                gate_bindings=LogicGateBindings(route="missing"),
            )


if __name__ == "__main__":
    unittest.main()
