import unittest

import numpy as np

from ratiss_entrelacs.system import Coupling, InterlacedSystem, RepeatingSchedule


class IntegrateIncoming:
    def step(self, time, state, incoming, dt, rng):
        total = np.zeros_like(state)
        for value in incoming.values():
            total += value
        return state + dt * total


class SystemTests(unittest.TestCase):
    def test_two_domains_exchange_state_through_directed_couplers(self):
        system = InterlacedSystem(
            {"field": IntegrateIncoming(), "memory": IntegrateIncoming()},
            [
                Coupling("field", "memory", lambda x: x, label="field_to_memory"),
                Coupling("memory", "field", lambda x: 0.5 * x, label="memory_to_field"),
            ],
        )
        result = system.run({"field": [1.0], "memory": [0.0]}, steps=2, dt=0.1, seed=5)
        np.testing.assert_allclose(result.series("field")[:, 0], [1.0, 1.0, 1.005])
        np.testing.assert_allclose(result.series("memory")[:, 0], [0.0, 0.1, 0.2])
        self.assertEqual(result.frames[-1].tick, 2)

    def test_same_seed_replays_multi_domain_run(self):
        system = InterlacedSystem({"a": IntegrateIncoming(), "b": IntegrateIncoming()}, [
            Coupling("a", "b", lambda x: x)
        ])
        a = system.run({"a": [0.2], "b": [0.0]}, steps=3, dt=0.1, seed=12)
        b = system.run({"a": [0.2], "b": [0.0]}, steps=3, dt=0.1, seed=12)
        np.testing.assert_array_equal(a.series("b"), b.series("b"))

    def test_frame_is_read_only_snapshot(self):
        system = InterlacedSystem({"a": IntegrateIncoming()})
        result = system.run({"a": [1.0]}, steps=1, dt=0.1)
        with self.assertRaises(ValueError):
            result.frames[0].fields["a"][0] = 4.0

    def test_event_schedule_can_update_domains_at_different_ticks(self):
        system = InterlacedSystem(
            {"a": IntegrateIncoming(), "b": IntegrateIncoming()},
            [Coupling("a", "b", lambda x: x)],
            scheduler=RepeatingSchedule((("a",), ("b",))),
        )
        result = system.run({"a": [1.0], "b": [0.0]}, steps=3, dt=0.1)
        # a updates on ticks 0 and 2; b updates on tick 1 from a's previous state.
        np.testing.assert_allclose(result.series("a")[:, 0], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(result.series("b")[:, 0], [0.0, 0.0, 0.1, 0.1])

    def test_unknown_component_and_mismatched_initial_state_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown component"):
            InterlacedSystem({"a": IntegrateIncoming()}, [Coupling("a", "b", lambda x: x)])
        system = InterlacedSystem({"a": IntegrateIncoming()})
        with self.assertRaisesRegex(ValueError, "keys must exactly match"):
            system.run({"wrong": [1.0]}, steps=1, dt=0.1)


if __name__ == "__main__":
    unittest.main()
