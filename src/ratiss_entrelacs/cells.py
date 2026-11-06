"""Abstract multi-branch dendrite/soma/adaptation dynamics.

This compact in-silico model makes branch-specific analog integration,
excitatory/inhibitory drive, somatic threshold events, and adaptation explicit.
It is a functional hypothesis, not a full biological-neuron model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _vector(value: ArrayLike, size: int, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiub":
        raise ValueError(f"{label} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite vector of shape ({size},)")
    return array.copy()


@dataclass(frozen=True)
class DendriticConfig:
    """Time constants and transfer rules for abstract dendritic compartments."""

    tau_dendrite: float = 0.2
    tau_membrane: float = 0.5
    tau_adaptation: float = 4.0
    dendrite_gain: float = 1.0
    adaptation_gain: float = 0.15
    adaptation_jump: float = 0.12
    threshold: float = 0.5
    reset_value: float = 0.0
    branch_nonlinearity: str = "linear"

    def __post_init__(self) -> None:
        for name in ("tau_dendrite", "tau_membrane", "tau_adaptation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "dendrite_gain", "adaptation_gain", "adaptation_jump", "threshold", "reset_value"
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.adaptation_gain < 0.0 or self.adaptation_jump < 0.0:
            raise ValueError("adaptation parameters must be non-negative")
        if self.reset_value >= self.threshold:
            raise ValueError("reset_value must be below threshold")
        if not isinstance(self.branch_nonlinearity, str) or self.branch_nonlinearity not in {"linear", "tanh"}:
            raise ValueError("branch_nonlinearity must be 'linear' or 'tanh'")


@dataclass(frozen=True)
class DendriticStep:
    """Per-tick branch and soma signals, including pre-reset voltage/events."""

    dendritic_potential: FloatArray
    membrane: FloatArray
    pre_reset_membrane: FloatArray
    adaptation: FloatArray
    events: BoolArray
    excitatory_current: FloatArray
    inhibitory_current: FloatArray


class DendriticPopulation:
    """Vectorized population with independently evolving dendritic branches.

    ``current`` is the backward-compatible signed-current interface. New code
    can pass ``excitatory_current`` and ``inhibitory_current`` separately; the
    latter is subtracted from the former before each branch is integrated.
    Inputs may be per-unit vectors (distributed evenly across branches) or
    explicit ``(units, branches)`` arrays. Branch potentials are combined at
    the soma, where threshold/reset and adaptation act.
    """

    def __init__(
        self,
        units: int,
        config: DendriticConfig | None = None,
        *,
        branches: int = 1,
        branch_gains: ArrayLike | None = None,
    ) -> None:
        if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units < 1:
            raise ValueError("units must be a positive integer")
        if isinstance(branches, bool) or not isinstance(branches, (int, np.integer)) or branches < 1:
            raise ValueError("branches must be a positive integer")
        self.units = int(units)
        self.branches = int(branches)
        self.config = config or DendriticConfig()
        if branch_gains is None:
            self._branch_gains = np.ones(self.branches, dtype=np.float64)
        else:
            self._branch_gains = _vector(branch_gains, self.branches, "branch_gains")
            if np.any(self._branch_gains < 0.0):
                raise ValueError("branch_gains must be non-negative")
        self._dendrite = np.zeros((self.units, self.branches), dtype=np.float64)
        self._membrane = np.full(self.units, self.config.reset_value, dtype=np.float64)
        self._adaptation = np.zeros(self.units, dtype=np.float64)

    @property
    def branch_gains(self) -> FloatArray:
        return self._branch_gains.copy()

    @property
    def dendritic_potential(self) -> FloatArray:
        return self._format_branches(self._dendrite)

    @property
    def membrane(self) -> FloatArray:
        return self._membrane.copy()

    @property
    def adaptation(self) -> FloatArray:
        return self._adaptation.copy()

    def _format_branches(self, values: FloatArray) -> FloatArray:
        return values[:, 0].copy() if self.branches == 1 else values.copy()

    def _branch_array(self, value: ArrayLike, label: str) -> FloatArray:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiub":
            raise ValueError(f"{label} must be a real numeric array")
        array = np.asarray(raw, dtype=np.float64)
        if array.shape == (self.units,):
            array = np.repeat(array[:, None] / self.branches, self.branches, axis=1)
        if array.shape != (self.units, self.branches) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{label} must be a finite vector of shape ({self.units},) or "
                f"an array of shape ({self.units}, {self.branches})"
            )
        return array.copy()

    def reset(self) -> None:
        self._dendrite.fill(0.0)
        self._membrane.fill(self.config.reset_value)
        self._adaptation.fill(0.0)

    def step(
        self,
        current: ArrayLike | None = None,
        *,
        dt: float,
        rhythm_gate: ArrayLike | None = None,
        excitatory_current: ArrayLike | None = None,
        inhibitory_current: ArrayLike | None = None,
    ) -> DendriticStep:
        """Advance all branches and emit upward somatic threshold events."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if dt > min(self.config.tau_dendrite, self.config.tau_membrane):
            raise ValueError("dt must not exceed dendrite or membrane time constants")
        if current is not None:
            if excitatory_current is not None or inhibitory_current is not None:
                raise ValueError("use either signed current or separate E/I currents")
            signed = self._branch_array(current, "current")
            excitatory = np.maximum(signed, 0.0)
            inhibitory = np.maximum(-signed, 0.0)
        else:
            if excitatory_current is None:
                raise ValueError("provide current or excitatory_current")
            excitatory = self._branch_array(excitatory_current, "excitatory_current")
            inhibitory = (
                np.zeros_like(excitatory)
                if inhibitory_current is None
                else self._branch_array(inhibitory_current, "inhibitory_current")
            )
            if np.any(excitatory < 0.0) or np.any(inhibitory < 0.0):
                raise ValueError("separate E/I currents must be non-negative")
        drive = excitatory - inhibitory

        if rhythm_gate is None:
            gate = np.ones((self.units, self.branches), dtype=np.float64)
        else:
            raw_gate = np.asarray(rhythm_gate)
            if raw_gate.dtype.kind not in "fiub":
                raise ValueError("rhythm_gate must be a real numeric array")
            gate_values = np.asarray(raw_gate, dtype=np.float64)
            if gate_values.shape == (self.units,):
                gate_values = np.repeat(gate_values[:, None], self.branches, axis=1)
            if (
                gate_values.shape != (self.units, self.branches)
                or not np.all(np.isfinite(gate_values))
                or np.any(gate_values < 0.0)
                or np.any(gate_values > 1.0)
            ):
                raise ValueError(
                    f"rhythm_gate must have values in [0, 1] and shape ({self.units},) "
                    f"or ({self.units}, {self.branches})"
                )
            gate = gate_values.copy()

        gated_drive = drive * gate
        dendrite = self._dendrite + (dt / self.config.tau_dendrite) * (
            gated_drive - self._dendrite
        )
        adaptation_decay = self._adaptation * np.exp(-dt / self.config.tau_adaptation)
        if self.config.branch_nonlinearity == "tanh":
            branch_to_soma = np.tanh(dendrite) * self._branch_gains[None, :]
        else:
            branch_to_soma = dendrite * self._branch_gains[None, :]
        somatic_drive = np.sum(branch_to_soma, axis=1)
        membrane_pre = self._membrane + (dt / self.config.tau_membrane) * (
            self.config.dendrite_gain * somatic_drive
            - self.config.adaptation_gain * adaptation_decay
            - self._membrane
        )
        events = membrane_pre >= self.config.threshold
        membrane = np.where(events, self.config.reset_value, membrane_pre)
        adaptation = adaptation_decay + self.config.adaptation_jump * events.astype(np.float64)

        self._dendrite = dendrite
        self._membrane = membrane
        self._adaptation = adaptation
        return DendriticStep(
            dendritic_potential=self._format_branches(dendrite),
            membrane=membrane.copy(),
            pre_reset_membrane=membrane_pre.copy(),
            adaptation=adaptation.copy(),
            events=events.copy(),
            excitatory_current=self._format_branches(excitatory),
            inhibitory_current=self._format_branches(inhibitory),
        )
