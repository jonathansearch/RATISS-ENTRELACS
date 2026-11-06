"""Hypothesis-level adaptive weighted-matrix processor (in silico).

The core combines a continuous state, branch-resolved excitatory/inhibitory
relations, rhythmic folding, event timing, and an optional dendrite/soma layer.
These equations are a candidate algorithm, not claims of biological fidelity,
physical implementation, or task-level superiority.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cells import DendriticPopulation
from .logic import LogicGateBindings, LogicGateLayer

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _real_array(value: ArrayLike, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiub":
        raise ValueError(f"{label} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _finite_vector(value: ArrayLike, size: int, label: str) -> FloatArray:
    array = _real_array(value, label)
    if array.shape != (size,):
        raise ValueError(f"{label} must be a finite vector of shape ({size},)")
    return array.copy()


def _bound_branch_channels(
    excitatory: FloatArray, inhibitory: FloatArray, budget: float
) -> tuple[FloatArray, FloatArray]:
    """Bound the combined E+I mass over every receiver's branches and sources."""
    norms = np.sum(excitatory + inhibitory, axis=(1, 2), keepdims=True)
    factors = np.minimum(1.0, budget / np.maximum(norms, 1e-15))
    return excitatory * factors, inhibitory * factors


def _relation_branches(
    value: ArrayLike,
    units: int,
    branches: int,
    label: str,
    *,
    nonnegative: bool = False,
) -> FloatArray:
    """Accept an aggregate matrix or explicit receiver×branch×source tensor."""
    array = _real_array(value, label)
    if array.shape == (units, units):
        array = np.repeat(array[:, None, :] / branches, branches, axis=1)
    elif array.shape != (units, branches, units):
        raise ValueError(
            f"{label} must have shape ({units}, {units}) or "
            f"({units}, {branches}, {units})"
        )
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{label} must be non-negative")
    return array.copy()


def weighted_fold_channels(
    excitatory_weights: ArrayLike,
    inhibitory_weights: ArrayLike,
    activity: ArrayLike,
    phases: ArrayLike,
    *,
    top_k: int,
    rhythm_strength: float,
    coactivity_floor: float = 0.1,
    temperature: float = 0.2,
    row_budget: float = 1.0,
) -> tuple[FloatArray, FloatArray, BoolArray, BoolArray]:
    """Fold separate non-negative E/I matrices while preserving pathway identity.

    Top-k selection is over source links after combining their E+I magnitude;
    the selected effective mass is split back in each link's E/I proportion.
    A link can therefore carry both pathways without cancelling during routing.
    """
    excitatory = _real_array(excitatory_weights, "excitatory_weights")
    inhibitory = _real_array(inhibitory_weights, "inhibitory_weights")
    if (
        excitatory.ndim != 2
        or excitatory.shape[0] == 0
        or excitatory.shape[0] != excitatory.shape[1]
        or inhibitory.shape != excitatory.shape
    ):
        raise ValueError("E/I weights must be equally sized, non-empty square matrices")
    if np.any(excitatory < 0.0) or np.any(inhibitory < 0.0):
        raise ValueError("E/I weights must be non-negative")
    n = excitatory.shape[0]
    state = _finite_vector(activity, n, "activity")
    phase = _finite_vector(phases, n, "phases")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= n:
        raise ValueError("top_k must be an integer between 1 and the matrix size")
    if not np.isfinite(rhythm_strength) or not 0.0 <= rhythm_strength <= 1.0:
        raise ValueError("rhythm_strength must lie in [0, 1]")
    if not np.isfinite(coactivity_floor) or coactivity_floor < 0.0:
        raise ValueError("coactivity_floor must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(row_budget) or row_budget <= 0.0:
        raise ValueError("row_budget must be finite and positive")

    total_weights = excitatory + inhibitory
    phase_compatibility = 0.5 * (1.0 + np.cos(phase[:, None] - phase[None, :]))
    phase_gate = (1.0 - rhythm_strength) + rhythm_strength * phase_compatibility
    source_use = coactivity_floor + np.abs(state)[None, :]
    scores = total_weights * phase_gate * source_use
    folded_e = np.zeros_like(excitatory)
    folded_i = np.zeros_like(inhibitory)

    for receiver in range(n):
        candidates = np.flatnonzero(total_weights[receiver] > 0.0)
        if candidates.size == 0:
            continue
        order = np.argsort(-scores[receiver, candidates], kind="stable")
        chosen = candidates[order[: min(int(top_k), candidates.size)]]
        mass = min(float(np.sum(total_weights[receiver, chosen])), float(row_budget))
        logits = scores[receiver, chosen] / float(temperature)
        probabilities = np.exp(logits - float(np.max(logits)))
        probabilities /= float(np.sum(probabilities))
        allocated = mass * probabilities * phase_gate[receiver, chosen]
        selected_total = total_weights[receiver, chosen]
        scale = allocated / selected_total
        folded_e[receiver, chosen] = excitatory[receiver, chosen] * scale
        folded_i[receiver, chosen] = inhibitory[receiver, chosen] * scale

    active_e = folded_e > 1e-15
    active_i = folded_i > 1e-15
    return folded_e, folded_i, active_e, active_i


def weighted_fold(
    weights: ArrayLike,
    activity: ArrayLike,
    phases: ArrayLike,
    *,
    top_k: int,
    rhythm_strength: float,
    coactivity_floor: float = 0.1,
    temperature: float = 0.2,
    row_budget: float = 1.0,
) -> tuple[FloatArray, BoolArray]:
    """Build a sparse signed effective matrix from one dense relation matrix."""
    matrix = _real_array(weights, "weights")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weights must be a non-empty square matrix")
    folded_e, folded_i, active_e, active_i = weighted_fold_channels(
        np.maximum(matrix, 0.0),
        np.maximum(-matrix, 0.0),
        activity,
        phases,
        top_k=top_k,
        rhythm_strength=rhythm_strength,
        coactivity_floor=coactivity_floor,
        temperature=temperature,
        row_budget=row_budget,
    )
    return folded_e - folded_i, active_e | active_i


@dataclass(frozen=True)
class STDPConfig:
    """Pair-based spike-timing rule; coefficients are abstract model units."""

    tau_pre: float = 0.2
    tau_post: float = 0.2
    a_plus: float = 0.01
    a_minus: float = 0.012
    inhibitory_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("tau_pre", "tau_post"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("a_plus", "a_minus", "inhibitory_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MatrixProcessorConfig:
    """Parameters for the candidate weighted-fold algorithm."""

    units: int = 32
    top_k: int = 4
    dt: float = 0.05
    tau: float = 0.5
    rhythm_strength: float = 0.65
    rhythm_rate: float = 2.0 * np.pi
    matrix_learning_rate: float = 0.02
    weight_decay: float = 0.01
    row_budget: float = 1.0
    coactivity_floor: float = 0.1
    temperature: float = 0.2
    threshold: float = 0.5
    plasticity_rule: str = "coactivity"
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.units, bool) or not isinstance(self.units, (int, np.integer)) or self.units < 1:
            raise ValueError("units must be a positive integer")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, (int, np.integer)) or not 1 <= self.top_k <= self.units:
            raise ValueError("top_k must be between 1 and units")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.tau) or self.tau <= 0.0 or self.dt > self.tau:
            raise ValueError("tau must be positive and dt must not exceed tau")
        if not np.isfinite(self.rhythm_strength) or not 0.0 <= self.rhythm_strength <= 1.0:
            raise ValueError("rhythm_strength must lie in [0, 1]")
        if not np.isfinite(self.rhythm_rate):
            raise ValueError("rhythm_rate must be finite")
        if not np.isfinite(self.matrix_learning_rate) or self.matrix_learning_rate < 0.0:
            raise ValueError("matrix_learning_rate must be finite and non-negative")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not np.isfinite(self.row_budget) or self.row_budget <= 0.0:
            raise ValueError("row_budget must be finite and positive")
        if not np.isfinite(self.coactivity_floor) or self.coactivity_floor < 0.0:
            raise ValueError("coactivity_floor must be finite and non-negative")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not isinstance(self.plasticity_rule, str) or self.plasticity_rule not in {"coactivity", "stdp"}:
            raise ValueError("plasticity_rule must be 'coactivity' or 'stdp'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class ProcessorStep:
    """One tick; ``active_edges`` counts branch×channel links separately."""

    activity: FloatArray
    output: FloatArray
    events: BoolArray
    active_edges: int
    logic_outputs: Mapping[str, bool]
    compartments: Mapping[str, FloatArray]


@dataclass(frozen=True)
class ProcessorTrace:
    """Recorded run; only the final aggregate matrix is retained."""

    activity: FloatArray
    output: FloatArray
    events: BoolArray
    active_edges: NDArray[np.int64]
    final_weights: FloatArray
    logic_outputs: Mapping[str, BoolArray]
    compartments: Mapping[str, FloatArray]


class DynamicMatrixProcessor:
    """In-silico processor with branch-resolved E/I routes and event dynamics.

    ``weights[i, j]`` remains a signed, aggregate view of pathways from source
    ``j`` to receiver ``i``. Internally, non-negative excitatory and inhibitory
    matrices are stored independently for each dendritic branch. Matrix
    plasticity may use the legacy local co-activity rule or pair-based STDP.
    Event logic still gates the following tick's routes, updates, and readout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        *,
        config: MatrixProcessorConfig | None = None,
        input_weights: ArrayLike | None = None,
        input_branch_weights: ArrayLike | None = None,
        initial_weights: ArrayLike | None = None,
        initial_excitatory_weights: ArrayLike | None = None,
        initial_inhibitory_weights: ArrayLike | None = None,
        readout_weights: ArrayLike | None = None,
        phase_rates: ArrayLike | None = None,
        dendritic_layer: DendriticPopulation | None = None,
        stdp_config: STDPConfig | None = None,
        logic_layer: LogicGateLayer | None = None,
        gate_bindings: LogicGateBindings | None = None,
    ) -> None:
        self.config = config or MatrixProcessorConfig()
        if isinstance(input_dim, bool) or not isinstance(input_dim, (int, np.integer)) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(output_dim, bool) or not isinstance(output_dim, (int, np.integer)) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.units = int(self.config.units)
        if dendritic_layer is not None:
            if not isinstance(dendritic_layer, DendriticPopulation):
                raise TypeError("dendritic_layer must be a DendriticPopulation")
            if dendritic_layer.units != self.units:
                raise ValueError("dendritic_layer units must match processor units")
            if self.config.dt > min(dendritic_layer.config.tau_dendrite, dendritic_layer.config.tau_membrane):
                raise ValueError("processor dt must not exceed dendritic or membrane time constants")
            self.branches = dendritic_layer.branches
        else:
            self.branches = 1
        self.dendritic_layer = dendritic_layer
        if stdp_config is not None and not isinstance(stdp_config, STDPConfig):
            raise TypeError("stdp_config must be an STDPConfig")
        self.stdp_config = stdp_config or STDPConfig()
        self._rng = np.random.default_rng(int(self.config.seed))

        if input_weights is not None and input_branch_weights is not None:
            raise ValueError("provide input_weights or input_branch_weights, not both")
        if input_branch_weights is not None:
            branch_input = _real_array(input_branch_weights, "input_branch_weights")
            expected = (self.units, self.branches, self.input_dim)
            if branch_input.shape != expected:
                raise ValueError(f"input_branch_weights must have shape {expected}")
            self._input_branch_weights = branch_input.copy()
        else:
            if input_weights is None:
                base_input = self._rng.normal(
                    0.0, 1.0 / np.sqrt(self.input_dim), (self.units, self.input_dim)
                )
            else:
                base_input = self._finite_matrix(
                    input_weights, (self.units, self.input_dim), "input_weights"
                )
            self._input_branch_weights = np.repeat(
                base_input[:, None, :] / self.branches, self.branches, axis=1
            )

        separate_channels = (
            initial_excitatory_weights is not None or initial_inhibitory_weights is not None
        )
        if initial_weights is not None and separate_channels:
            raise ValueError("initial_weights cannot be combined with initial E/I matrices")
        if initial_weights is None and not separate_channels:
            raw_initial = self._rng.normal(
                0.0, 0.15 / np.sqrt(self.units), (self.units, self.units)
            )
            signed = _relation_branches(raw_initial, self.units, self.branches, "initial_weights")
            excitatory = np.maximum(signed, 0.0)
            inhibitory = np.maximum(-signed, 0.0)
        elif initial_weights is not None:
            signed = _relation_branches(
                initial_weights, self.units, self.branches, "initial_weights"
            )
            excitatory = np.maximum(signed, 0.0)
            inhibitory = np.maximum(-signed, 0.0)
        else:
            if initial_excitatory_weights is None:
                excitatory = np.zeros((self.units, self.branches, self.units), dtype=np.float64)
            else:
                excitatory = _relation_branches(
                    initial_excitatory_weights,
                    self.units,
                    self.branches,
                    "initial_excitatory_weights",
                    nonnegative=True,
                )
            if initial_inhibitory_weights is None:
                inhibitory = np.zeros((self.units, self.branches, self.units), dtype=np.float64)
            else:
                inhibitory = _relation_branches(
                    initial_inhibitory_weights,
                    self.units,
                    self.branches,
                    "initial_inhibitory_weights",
                    nonnegative=True,
                )
        self._branch_weights_e, self._branch_weights_i = _bound_branch_channels(
            excitatory, inhibitory, self.config.row_budget
        )

        if readout_weights is None:
            self._readout = np.zeros((self.output_dim, self.units), dtype=np.float64)
        else:
            self._readout = self._finite_matrix(
                readout_weights, (self.output_dim, self.units), "readout_weights"
            )

        if phase_rates is None:
            self._phase_rates = np.full(self.units, self.config.rhythm_rate, dtype=np.float64)
        else:
            self._phase_rates = _finite_vector(phase_rates, self.units, "phase_rates")
        self._initial_phases = 2.0 * np.pi * np.arange(self.units, dtype=np.float64) / self.units
        self._activity = np.zeros(self.units, dtype=np.float64)
        if self.dendritic_layer is not None:
            self._activity = self.dendritic_layer.membrane
        self._phases = self._initial_phases.copy()
        self._last_events = np.zeros(self.units, dtype=bool)
        self._pre_trace = np.zeros(self.units, dtype=np.float64)
        self._post_trace = np.zeros(self.units, dtype=np.float64)
        if gate_bindings is not None and not isinstance(gate_bindings, LogicGateBindings):
            raise TypeError("gate_bindings must be LogicGateBindings")
        self.logic_layer = logic_layer
        self.gate_bindings = gate_bindings or LogicGateBindings()
        if logic_layer is None:
            if any((self.gate_bindings.route, self.gate_bindings.plasticity, self.gate_bindings.readout)):
                raise ValueError("gate_bindings require a logic_layer")
        else:
            if not isinstance(logic_layer, LogicGateLayer):
                raise TypeError("logic_layer must be a LogicGateLayer")
            if logic_layer.n_inputs != self.units:
                raise ValueError("logic_layer input count must match processor units")
            unknown = {
                name for name in (
                    self.gate_bindings.route,
                    self.gate_bindings.plasticity,
                    self.gate_bindings.readout,
                ) if name is not None and name not in logic_layer.gate_names
            }
            if unknown:
                raise ValueError(f"gate bindings reference unknown outputs: {sorted(unknown)}")

    @staticmethod
    def _finite_matrix(value: ArrayLike, shape: tuple[int, int], label: str) -> FloatArray:
        array = _real_array(value, label)
        if array.shape != shape:
            raise ValueError(f"{label} must be a finite matrix of shape {shape}")
        return array.copy()

    @property
    def weights(self) -> FloatArray:
        """Defensive copy of the aggregate signed (E minus I) weight matrix."""
        return np.sum(self._branch_weights_e - self._branch_weights_i, axis=1)

    @property
    def excitatory_weights(self) -> FloatArray:
        """Aggregate non-negative excitatory matrix across branches."""
        return np.sum(self._branch_weights_e, axis=1)

    @property
    def inhibitory_weights(self) -> FloatArray:
        """Aggregate non-negative inhibitory magnitudes across branches."""
        return np.sum(self._branch_weights_i, axis=1)

    @property
    def branch_weights(self) -> FloatArray:
        """Signed per-branch relations with shape ``(units, branches, units)``."""
        return self._branch_weights_e - self._branch_weights_i

    @property
    def excitatory_branch_weights(self) -> FloatArray:
        return self._branch_weights_e.copy()

    @property
    def inhibitory_branch_weights(self) -> FloatArray:
        return self._branch_weights_i.copy()

    @property
    def input_branch_weights(self) -> FloatArray:
        return self._input_branch_weights.copy()

    @property
    def activity(self) -> FloatArray:
        return self._activity.copy()

    @property
    def phases(self) -> FloatArray:
        return self._phases.copy()

    @property
    def readout_weights(self) -> FloatArray:
        return self._readout.copy()

    @property
    def pre_spike_trace(self) -> FloatArray:
        return self._pre_trace.copy()

    @property
    def post_spike_trace(self) -> FloatArray:
        return self._post_trace.copy()

    def reset(self) -> None:
        """Reset dynamic states/traces; preserve learned pathway matrices."""
        self._activity.fill(0.0)
        if self.dendritic_layer is not None:
            self.dendritic_layer.reset()
            self._activity = self.dendritic_layer.membrane
        self._phases = self._initial_phases.copy()
        self._last_events.fill(False)
        self._pre_trace.fill(0.0)
        self._post_trace.fill(0.0)

    def folded_channels(
        self,
    ) -> tuple[FloatArray, FloatArray, BoolArray, BoolArray]:
        """Return branch-resolved effective E/I weights and their active masks."""
        folded_e = np.zeros_like(self._branch_weights_e)
        folded_i = np.zeros_like(self._branch_weights_i)
        for branch in range(self.branches):
            e, i, _, _ = weighted_fold_channels(
                self._branch_weights_e[:, branch, :],
                self._branch_weights_i[:, branch, :],
                self._activity,
                self._phases,
                top_k=int(self.config.top_k),
                rhythm_strength=float(self.config.rhythm_strength),
                coactivity_floor=float(self.config.coactivity_floor),
                temperature=float(self.config.temperature),
                row_budget=float(self.config.row_budget),
            )
            folded_e[:, branch, :] = e
            folded_i[:, branch, :] = i
        folded_e, folded_i = _bound_branch_channels(
            folded_e, folded_i, float(self.config.row_budget)
        )
        active_e = folded_e > 1e-15
        active_i = folded_i > 1e-15
        return folded_e, folded_i, active_e, active_i

    def folded_matrix(self) -> tuple[FloatArray, BoolArray]:
        """Inspect the aggregate signed fold and active edges without advancing."""
        folded_e, folded_i, active_e, active_i = self.folded_channels()
        signed = np.sum(folded_e - folded_i, axis=1)
        active = np.any(active_e | active_i, axis=1)
        return signed, active

    def _phase_gate(self) -> FloatArray:
        compatibility = 0.5 * (
            1.0 + np.cos(self._phases[:, None] - self._phases[None, :])
        )
        return (1.0 - self.config.rhythm_strength) + self.config.rhythm_strength * compatibility

    def _coactivity_update(self, next_activity: FloatArray) -> None:
        decay = np.exp(-self.config.weight_decay * self.config.dt)
        coactivity = np.outer(next_activity, self._activity)
        delta = (
            self.config.matrix_learning_rate
            * self.config.dt
            * self._phase_gate()
            * coactivity
        )[:, None, :] / self.branches
        # Positive local correlations grow E weights; negative correlations grow
        # I magnitudes. Existing E and I channels decay independently and are not
        # collapsed merely because their signed aggregate is used for inspection.
        delta_e = np.maximum(delta, 0.0)
        delta_i = np.maximum(-delta, 0.0)
        self._branch_weights_e, self._branch_weights_i = _bound_branch_channels(
            self._branch_weights_e * decay + delta_e,
            self._branch_weights_i * decay + delta_i,
            self.config.row_budget,
        )

    def _stdp_update(self, events: BoolArray, pre_decay: FloatArray, post_decay: FloatArray) -> None:
        timing_delta = (
            self.stdp_config.a_plus * np.outer(events.astype(np.float64), pre_decay)
            - self.stdp_config.a_minus * np.outer(post_decay, events.astype(np.float64))
        )
        timing_delta *= self._phase_gate()
        scale = self.config.matrix_learning_rate
        decay = np.exp(-self.config.weight_decay * self.config.dt)
        delta_e = (scale * timing_delta / self.branches)[:, None, :]
        delta_i = (
            scale * self.stdp_config.inhibitory_scale * timing_delta / self.branches
        )[:, None, :]
        updated_e = np.maximum(self._branch_weights_e * decay + delta_e, 0.0)
        updated_i = np.maximum(self._branch_weights_i * decay + delta_i, 0.0)
        self._branch_weights_e, self._branch_weights_i = _bound_branch_channels(
            updated_e, updated_i, self.config.row_budget
        )

    def step(
        self,
        signal: ArrayLike,
        *,
        rhythm: ArrayLike | float | None = None,
        learn_matrix: bool = True,
    ) -> ProcessorStep:
        """Integrate one input vector and optionally update relation matrices.

        ``rhythm`` may be a scalar pulse shared by all units or a per-unit
        vector in [0, 1]. Internal phases generate the pulse when omitted.
        Logic outputs use the previous tick's threshold events. In STDP mode,
        pre/post traces use exponentially decaying histories and signed event
        intervals to potentiate causal or depress anti-causal pathway weights.
        """
        incoming = _finite_vector(signal, self.input_dim, "signal")
        if rhythm is None:
            pulse = 0.5 * (1.0 + np.sin(self._phases))
        else:
            pulse_array = _real_array(rhythm, "rhythm")
            if pulse_array.ndim == 0:
                pulse = np.full(self.units, float(pulse_array), dtype=np.float64)
            else:
                pulse = _finite_vector(pulse_array, self.units, "rhythm")
            if not np.all(np.isfinite(pulse)) or np.any(pulse < 0.0) or np.any(pulse > 1.0):
                raise ValueError("rhythm pulse values must lie in [0, 1]")

        logic_outputs = (
            {} if self.logic_layer is None
            else dict(self.logic_layer.evaluate(self._last_events))
        )
        route_open = (
            True if self.gate_bindings.route is None
            else logic_outputs[self.gate_bindings.route]
        )
        folded_e, folded_i, active_e, active_i = self.folded_channels()
        if not route_open:
            folded_e.fill(0.0)
            folded_i.fill(0.0)
            active_e.fill(False)
            active_i.fill(False)

        # Somatic state is continuous and may be signed; use its rectified part
        # as a non-negative release proxy so E/I pathway polarity stays explicit.
        released_activity = np.maximum(self._activity, 0.0)
        recurrent_e = np.einsum("ibj,j->ib", folded_e, released_activity, optimize=True)
        recurrent_i = np.einsum("ibj,j->ib", folded_i, released_activity, optimize=True)
        input_drive = np.einsum(
            "ibf,f->ib", self._input_branch_weights, incoming, optimize=True
        )
        excitatory_current = recurrent_e + np.maximum(input_drive, 0.0)
        inhibitory_current = recurrent_i + np.maximum(-input_drive, 0.0)
        current = np.sum(excitatory_current - inhibitory_current, axis=1)
        gate = (1.0 - self.config.rhythm_strength) + self.config.rhythm_strength * pulse
        compartments: dict[str, FloatArray] = {
            "excitatory_current": (
                excitatory_current[:, 0].copy()
                if self.branches == 1 else excitatory_current.copy()
            ),
            "inhibitory_current": (
                inhibitory_current[:, 0].copy()
                if self.branches == 1 else inhibitory_current.copy()
            ),
        }
        if self.dendritic_layer is None:
            target_activity = np.tanh(current)
            next_activity = self._activity + (self.config.dt / self.config.tau) * gate * (
                target_activity - self._activity
            )
            events = (self._activity < self.config.threshold) & (next_activity >= self.config.threshold)
        else:
            cell_step = self.dendritic_layer.step(
                excitatory_current=excitatory_current,
                inhibitory_current=inhibitory_current,
                dt=self.config.dt,
                rhythm_gate=gate,
            )
            next_activity = cell_step.membrane
            events = cell_step.events
            compartments.update({
                "dendritic_potential": cell_step.dendritic_potential,
                "pre_reset_membrane": cell_step.pre_reset_membrane,
                "adaptation": cell_step.adaptation,
            })

        readout_open = (
            True if self.gate_bindings.readout is None
            else logic_outputs[self.gate_bindings.readout]
        )
        output = self._readout @ next_activity if readout_open else np.zeros(self.output_dim)
        plasticity_open = (
            True if self.gate_bindings.plasticity is None
            else logic_outputs[self.gate_bindings.plasticity]
        )

        pre_decay = self._pre_trace
        post_decay = self._post_trace
        if self.config.plasticity_rule == "stdp":
            pre_decay = self._pre_trace * np.exp(-self.config.dt / self.stdp_config.tau_pre)
            post_decay = self._post_trace * np.exp(-self.config.dt / self.stdp_config.tau_post)
        if learn_matrix and plasticity_open and self.config.matrix_learning_rate > 0.0:
            if self.config.plasticity_rule == "stdp":
                self._stdp_update(events, pre_decay, post_decay)
            else:
                self._coactivity_update(next_activity)
        if self.config.plasticity_rule == "stdp":
            event_values = events.astype(np.float64)
            self._pre_trace = pre_decay + event_values
            self._post_trace = post_decay + event_values
            compartments["pre_spike_trace"] = self._pre_trace.copy()
            compartments["post_spike_trace"] = self._post_trace.copy()

        self._activity = next_activity
        self._last_events = events.copy()
        self._phases = np.remainder(
            self._phases + self._phase_rates * self.config.dt,
            2.0 * np.pi,
        )
        return ProcessorStep(
            activity=self._activity.copy(),
            output=output.copy(),
            events=events.copy(),
            active_edges=int(np.count_nonzero(active_e) + np.count_nonzero(active_i)),
            logic_outputs=MappingProxyType(logic_outputs),
            compartments=MappingProxyType(compartments),
        )

    def calibrate_readout(
        self,
        target: ArrayLike,
        *,
        activity: ArrayLike | None = None,
        learning_rate: float = 0.2,
    ) -> FloatArray:
        """Apply a normalized least-mean-squares update to the output map."""
        desired = _finite_vector(target, self.output_dim, "target")
        state = self._activity if activity is None else _finite_vector(activity, self.units, "activity")
        if not np.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        error = desired - self._readout @ state
        denominator = 1e-12 + float(state @ state)
        self._readout += float(learning_rate) * np.outer(error, state) / denominator
        return error

    def process(
        self,
        signals: ArrayLike,
        *,
        rhythms: ArrayLike | float | None = None,
        learn_matrix: bool = True,
        targets: ArrayLike | None = None,
        readout_learning_rate: float = 0.2,
    ) -> ProcessorTrace:
        """Process a time-by-input matrix; optionally calibrate the readout."""
        sequence = _real_array(signals, "signals")
        if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] != self.input_dim:
            raise ValueError(f"signals must have shape (time>0, {self.input_dim})")

        rhythm_sequence: NDArray[np.float64] | None = None
        rhythm_constant: float | None = None
        if rhythms is not None:
            rhythm_array = _real_array(rhythms, "rhythms")
            if rhythm_array.ndim == 0:
                rhythm_constant = float(rhythm_array)
                if not 0.0 <= rhythm_constant <= 1.0:
                    raise ValueError("rhythm pulse values must lie in [0, 1]")
            elif rhythm_array.shape == (sequence.shape[0],):
                rhythm_sequence = rhythm_array
            elif rhythm_array.shape == (sequence.shape[0], self.units):
                rhythm_sequence = rhythm_array
            else:
                raise ValueError("rhythms must be scalar, shape (time,), or shape (time, units)")
            if rhythm_sequence is not None and (
                not np.all(np.isfinite(rhythm_sequence))
                or np.any(rhythm_sequence < 0.0)
                or np.any(rhythm_sequence > 1.0)
            ):
                raise ValueError("rhythm pulse values must lie in [0, 1]")

        desired: NDArray[np.float64] | None = None
        if targets is not None:
            desired = _real_array(targets, "targets")
            if desired.shape != (sequence.shape[0], self.output_dim):
                raise ValueError(
                    f"targets must be finite with shape ({sequence.shape[0]}, {self.output_dim})"
                )

        activities = np.empty((sequence.shape[0], self.units), dtype=np.float64)
        outputs = np.empty((sequence.shape[0], self.output_dim), dtype=np.float64)
        events = np.zeros((sequence.shape[0], self.units), dtype=bool)
        edge_counts = np.empty(sequence.shape[0], dtype=np.int64)
        logic_trace = {
            name: np.zeros(sequence.shape[0], dtype=bool)
            for name in (() if self.logic_layer is None else self.logic_layer.gate_names)
        }
        branch_shape = (self.units,) if self.branches == 1 else (self.units, self.branches)
        compartment_shapes: dict[str, tuple[int, ...]] = {
            "excitatory_current": branch_shape,
            "inhibitory_current": branch_shape,
        }
        if self.dendritic_layer is not None:
            compartment_shapes.update({
                "dendritic_potential": branch_shape,
                "pre_reset_membrane": (self.units,),
                "adaptation": (self.units,),
            })
        if self.config.plasticity_rule == "stdp":
            compartment_shapes.update({
                "pre_spike_trace": (self.units,),
                "post_spike_trace": (self.units,),
            })
        compartment_trace = {
            name: np.empty((sequence.shape[0],) + shape, dtype=np.float64)
            for name, shape in compartment_shapes.items()
        }
        for index, signal in enumerate(sequence):
            if rhythm_constant is not None:
                rhythm = rhythm_constant
            elif rhythm_sequence is None:
                rhythm = None
            elif rhythm_sequence.ndim == 1:
                rhythm = float(rhythm_sequence[index])
            else:
                rhythm = rhythm_sequence[index]
            result = self.step(signal, rhythm=rhythm, learn_matrix=learn_matrix)
            activities[index] = result.activity
            outputs[index] = result.output
            events[index] = result.events
            edge_counts[index] = result.active_edges
            for name, value in result.logic_outputs.items():
                logic_trace[name][index] = value
            for name, value in result.compartments.items():
                compartment_trace[name][index] = value
            readout_open = (
                self.gate_bindings.readout is None
                or result.logic_outputs[self.gate_bindings.readout]
            )
            if desired is not None and readout_open:
                self.calibrate_readout(
                    desired[index],
                    activity=result.activity,
                    learning_rate=readout_learning_rate,
                )
        return ProcessorTrace(
            activity=activities,
            output=outputs,
            events=events,
            active_edges=edge_counts,
            final_weights=self.weights,
            logic_outputs=MappingProxyType(logic_trace),
            compartments=MappingProxyType(compartment_trace),
        )
