import unittest

import numpy as np

from ratiss_entrelacs.engine import run
from ratiss_entrelacs.system import Coupling, InterlacedSystem


class ComplexRotation:
    def step(self, time, state, dt, rng):
        return state * np.exp(1j * dt)


class ComplexRelay:
    def step(self, time, state, incoming, dt, rng):
        total = np.zeros_like(state, dtype=np.complex128)
        for value in incoming.values():
            total += value
        return state + dt * total


class ComplexStateTests(unittest.TestCase):
    def test_single_domain_engine_keeps_complex_phase(self):
        result = run(ComplexRotation(), np.array([1.0 + 0.0j]), steps=10, dt=0.1)
        self.assertTrue(np.iscomplexobj(result.states))
        self.assertAlmostEqual(abs(result.final_state[0]), 1.0, places=12)
        self.assertAlmostEqual(np.angle(result.final_state[0]), 1.0, places=12)

    def test_complex_and_real_domains_can_exchange_messages(self):
        system = InterlacedSystem(
            {"phase": ComplexRelay(), "field": ComplexRelay()},
            [
                Coupling("phase", "field", lambda x: np.exp(1j * x), "phase_to_field"),
                Coupling("field", "phase", lambda x: np.abs(x), "field_to_phase"),
            ],
        )
        result = system.run({"phase": [0.2], "field": [1.0 + 0.0j]}, steps=2, dt=0.1, seed=3)
        self.assertTrue(np.iscomplexobj(result.series("field")))
        self.assertAlmostEqual(float(np.real(result.series("phase")[-1, 0])), 0.4098, places=4)
        self.assertGreater(float(abs(result.series("field")[-1, 0])), 1.0)


if __name__ == "__main__":
    unittest.main()
