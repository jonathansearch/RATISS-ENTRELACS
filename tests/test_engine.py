import unittest

import numpy as np

from ratiss_entrelacs.engine import run


class AddDrift:
    def step(self, time, state, dt, rng):
        return state + dt


class WrongShape:
    def step(self, time, state, dt, rng):
        return np.zeros((state.size + 1,))


class NonFinite:
    def step(self, time, state, dt, rng):
        return state * np.nan


class EngineTests(unittest.TestCase):
    def test_records_initial_and_final_state(self):
        initial = np.array([0.0, 2.0])
        result = run(AddDrift(), initial, steps=4, dt=0.25)
        np.testing.assert_allclose(result.times, [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result.states[0], [0.0, 2.0])
        np.testing.assert_allclose(result.final_state, [1.0, 3.0])
        np.testing.assert_allclose(initial, [0.0, 2.0])  # caller input is untouched

    def test_zero_steps_still_records_initial_state(self):
        result = run(AddDrift(), np.array([3.0]), steps=0, dt=0.1)
        self.assertEqual(result.states.shape, (1, 1))
        np.testing.assert_allclose(result.final_state, [3.0])

    def test_bad_parameters_are_rejected(self):
        for steps, dt in [(-1, 0.1), (1.5, 0.1), (2, 0.0), (2, np.nan)]:
            with self.subTest(steps=steps, dt=dt), self.assertRaises(ValueError):
                run(AddDrift(), np.array([0.0]), steps=steps, dt=dt)

    def test_shape_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed state shape"):
            run(WrongShape(), np.array([0.0, 1.0]), steps=1, dt=0.1)

    def test_non_finite_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run(NonFinite(), np.array([1.0]), steps=1, dt=0.1)


if __name__ == "__main__":
    unittest.main()
