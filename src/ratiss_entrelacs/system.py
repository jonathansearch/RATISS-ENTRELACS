"""Composable in-silico substrate for heterogeneous state domains.

Components own their update law; couplings translate state between domains;
a scheduler decides which components update on a tick. The default scheduler
is synchronous only as a reference. None of these choices encodes Jonathan's
rhythmic algorithm. Real and complex-valued state fields are supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

NumericArray = NDArray[np.generic]


def _numeric_copy(value: ArrayLike, label: str) -> NumericArray:
    array = np.asarray(value)
    if array.dtype.kind not in "fciub":
        raise ValueError(f"{label} must be a real or complex numeric array")
    dtype = np.complex128 if array.dtype.kind == "c" else np.float64
    array = np.asarray(array, dtype=dtype).copy()
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be non-empty and finite")
    return array


class Component(Protocol):
    """A computational domain with a pluggable state-transition rule."""

    def step(
        self,
        time: float,
        state: NumericArray,
        incoming: Mapping[str, NumericArray],
        dt: float,
        rng: np.random.Generator,
    ) -> ArrayLike: ...


class Scheduler(Protocol):
    """Select domains that update at a particular tick."""

    def active(self, tick: int, time: float, names: tuple[str, ...]) -> Sequence[str]: ...


@dataclass(frozen=True)
class SynchronousScheduler:
    """Update every domain from the same previous-state snapshot."""

    def active(self, tick: int, time: float, names: tuple[str, ...]) -> tuple[str, ...]:
        return names


@dataclass(frozen=True)
class RepeatingSchedule:
    """Small explicit event pattern; useful as a scheduler control, not a law."""

    pattern: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.pattern or any(not isinstance(row, tuple) for row in self.pattern):
            raise ValueError("pattern must contain one or more tuples of component names")

    def active(self, tick: int, time: float, names: tuple[str, ...]) -> Sequence[str]:
        return self.pattern[tick % len(self.pattern)]


@dataclass(frozen=True)
class Coupling:
    """Directed adapter from one state domain to another.

    ``transform`` is intentionally injectable: phase, geometry, graph,
    energy, symbolic and complex-valued states need not share a representation.
    """

    source: str
    target: str
    transform: Callable[[NumericArray], ArrayLike]
    label: str = ""

    def transfer(self, state: NumericArray) -> NumericArray:
        message = _numeric_copy(
            self.transform(state.copy()),
            f"coupling {self.label or self.source + '->' + self.target} message",
        )
        return message


@dataclass(frozen=True)
class Snapshot:
    """Read-only bus snapshot; all components see the same tick's state."""

    tick: int
    time: float
    fields: Mapping[str, NumericArray]

    def __post_init__(self) -> None:
        frozen: dict[str, NumericArray] = {}
        for name, value in self.fields.items():
            array = _numeric_copy(value, f"state field {name!r}")
            array.setflags(write=False)
            frozen[name] = array
        object.__setattr__(self, "fields", MappingProxyType(frozen))


@dataclass(frozen=True)
class SystemRun:
    """A reproducible sequence of bus snapshots from a modular system."""

    frames: tuple[Snapshot, ...]

    def series(self, name: str) -> NumericArray:
        if not self.frames or name not in self.frames[0].fields:
            raise KeyError(name)
        return np.stack([frame.fields[name] for frame in self.frames], axis=0)


class InterlacedSystem:
    """Run heterogeneous components joined by directed couplers.

    This default uses a snapshot/Jacobi update: couplers read the old frame,
    then selected components update. A custom Scheduler can replace the
    synchronous schedule with a pulse/event pattern without changing the
    component equations.
    """

    def __init__(
        self,
        components: Mapping[str, Component],
        couplings: Sequence[Coupling] = (),
        scheduler: Scheduler | None = None,
    ) -> None:
        if not components or any(not isinstance(name, str) or not name for name in components):
            raise ValueError("components must be a non-empty mapping with non-empty names")
        self.components = dict(components)
        self.couplings = tuple(couplings)
        self.scheduler = scheduler or SynchronousScheduler()
        for edge in self.couplings:
            if edge.source not in self.components or edge.target not in self.components:
                raise ValueError(f"unknown component in coupling {edge.source!r}->{edge.target!r}")

    def run(
        self,
        initial_states: Mapping[str, ArrayLike],
        *,
        steps: int,
        dt: float,
        seed: int = 0,
    ) -> SystemRun:
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if set(initial_states) != set(self.components):
            raise ValueError("initial_states keys must exactly match component names")

        states = {
            name: _numeric_copy(initial_states[name], f"initial state for {name!r}")
            for name in self.components
        }
        names = tuple(self.components)
        child_seeds = np.random.SeedSequence(int(seed)).spawn(len(names))
        rngs = {name: np.random.default_rng(child) for name, child in zip(names, child_seeds)}
        frames: list[Snapshot] = [Snapshot(0, 0.0, states)]

        for tick in range(int(steps)):
            time = tick * float(dt)
            incoming: dict[str, dict[str, NumericArray]] = {name: {} for name in names}
            for index, edge in enumerate(self.couplings):
                key = edge.label or f"{edge.source}->{edge.target}#{index}"
                if key in incoming[edge.target]:
                    raise ValueError(f"duplicate coupling label {key!r} into {edge.target!r}")
                incoming[edge.target][key] = edge.transfer(states[edge.source])

            active = tuple(self.scheduler.active(tick, time, names))
            if len(set(active)) != len(active) or any(name not in self.components for name in active):
                raise ValueError(f"scheduler returned invalid component names at tick {tick}")

            next_states = {name: state.copy() for name, state in states.items()}
            for name in active:
                result = _numeric_copy(
                    self.components[name].step(
                        time,
                        states[name].copy(),
                        {key: value.copy() for key, value in incoming[name].items()},
                        float(dt),
                        rngs[name],
                    ),
                    f"component {name!r} state",
                )
                if result.shape != states[name].shape:
                    raise ValueError(f"component {name!r} changed state shape")
                next_states[name] = result

            states = next_states
            frames.append(Snapshot(tick + 1, (tick + 1) * float(dt), states))

        return SystemRun(tuple(frames))
