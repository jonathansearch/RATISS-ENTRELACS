"""Single-domain trajectory engine; state may be real or complex."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

NumericArray = NDArray[np.generic]


def _numeric_state(value: ArrayLike, label: str) -> NumericArray:
    state = np.asarray(value)
    if state.dtype.kind not in "fciub":
        raise ValueError(f"{label} must be a real or complex numeric array")
    dtype = np.complex128 if state.dtype.kind == "c" else np.float64
    state = np.asarray(state, dtype=dtype).copy()
    if state.size == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{label} contains non-finite values")
    return state


class Stepper(Protocol):
    """A component that advances a state by one time step."""

    def step(
        self,
        time: float,
        state: NumericArray,
        dt: float,
        rng: np.random.Generator,
    ) -> ArrayLike: ...


@dataclass(frozen=True)
class SimulationResult:
    """Recorded time grid and states, shape ``(steps + 1, *state_shape)``."""

    times: NDArray[np.float64]
    states: NumericArray

    @property
    def final_state(self) -> NumericArray:
        """Return a copy so callers cannot mutate the stored trajectory."""
        return self.states[-1].copy()


def run(
    model: Stepper,
    initial_state: ArrayLike,
    *,
    steps: int,
    dt: float,
    seed: int = 0,
) -> SimulationResult:
    """Run ``model.step`` repeatedly and record every real/complex state.

    The engine owns the random generator so identical models, inputs and seeds
    replay identically. The component law itself remains injectable.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    state = _numeric_state(initial_state, "initial_state")
    trajectory = [state.copy()]
    times = np.arange(int(steps) + 1, dtype=np.float64) * float(dt)
    rng = np.random.default_rng(int(seed))

    for index in range(int(steps)):
        next_state = _numeric_state(
            model.step(float(times[index]), state.copy(), float(dt), rng),
            f"state at step {index + 1}",
        )
        if next_state.shape != state.shape:
            raise ValueError(
                f"model changed state shape from {state.shape} to {next_state.shape}"
            )
        state = next_state
        trajectory.append(state.copy())

    return SimulationResult(times=times, states=np.stack(trajectory, axis=0))
