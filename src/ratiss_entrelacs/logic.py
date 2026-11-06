"""Boolean control fabric for event-driven matrix-processor decisions.

Gates consume threshold-crossing events and produce control bits for the next
matrix tick. They are a distinct logic layer, not replacements for the
continuous weighted-state computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

GateSource: TypeAlias = int | str


@dataclass(frozen=True)
class LogicGate:
    """One Boolean gate; integer sources address events, names address prior gates."""

    name: str
    operation: str
    sources: tuple[GateSource, ...]


@dataclass(frozen=True)
class LogicGateBindings:
    """Optional named outputs mapped to processor control points."""

    route: str | None = None
    plasticity: str | None = None
    readout: str | None = None


class LogicGateLayer:
    """Ordered acyclic Boolean circuit driven by a vector of event bits.

    Gates may refer to event indices or to gates declared earlier in the list.
    This enforces an explicit feed-forward logic graph; feedback to the
    processor occurs through the control bindings on the following tick.
    """

    _ARITIES = {
        "AND": (2, None),
        "NAND": (2, None),
        "OR": (2, None),
        "NOR": (2, None),
        "XOR": (2, None),
        "XNOR": (2, None),
        "NOT": (1, 1),
        "IMPLIES": (2, 2),
        "MAJORITY": (3, None),
    }

    def __init__(self, n_inputs: int, gates: tuple[LogicGate, ...] | list[LogicGate]) -> None:
        if isinstance(n_inputs, bool) or not isinstance(n_inputs, (int, np.integer)) or n_inputs < 1:
            raise ValueError("n_inputs must be a positive integer")
        self.n_inputs = int(n_inputs)
        self.gates = tuple(gates)
        names: set[str] = set()
        for gate in self.gates:
            if not isinstance(gate, LogicGate):
                raise TypeError("gates must contain LogicGate instances")
            if not isinstance(gate.name, str) or not gate.name or gate.name in names:
                raise ValueError("gate names must be non-empty and unique")
            operation = gate.operation.upper()
            if operation not in self._ARITIES:
                raise ValueError(f"unsupported gate operation {gate.operation!r}")
            minimum, maximum = self._ARITIES[operation]
            if len(gate.sources) < minimum or (maximum is not None and len(gate.sources) != maximum):
                raise ValueError(f"{operation} has invalid arity {len(gate.sources)}")
            for source in gate.sources:
                if isinstance(source, bool):
                    raise ValueError("gate sources must be event indices or prior gate names")
                if isinstance(source, (int, np.integer)):
                    if not 0 <= int(source) < self.n_inputs:
                        raise ValueError(f"event source {source} is outside [0, {self.n_inputs})")
                elif isinstance(source, str):
                    if source not in names:
                        raise ValueError(f"gate source {source!r} must name an earlier gate")
                else:
                    raise ValueError("gate sources must be event indices or prior gate names")
            names.add(gate.name)
        self.gate_names = tuple(gate.name for gate in self.gates)

    def evaluate(self, events: ArrayLike) -> Mapping[str, bool]:
        """Evaluate the circuit on Boolean threshold-crossing events."""
        raw = np.asarray(events)
        if raw.shape != (self.n_inputs,) or raw.dtype.kind not in "biuf":
            raise ValueError(f"events must be a Boolean/numeric vector of shape ({self.n_inputs},)")
        if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
            raise ValueError("event inputs must contain only 0/1 values")
        event_values = raw.astype(bool, copy=False)
        values: dict[str, bool] = {}

        for gate in self.gates:
            inputs = [
                bool(event_values[int(source)]) if isinstance(source, (int, np.integer)) else values[source]
                for source in gate.sources
            ]
            op = gate.operation.upper()
            if op == "AND":
                result = all(inputs)
            elif op == "NAND":
                result = not all(inputs)
            elif op == "OR":
                result = any(inputs)
            elif op == "NOR":
                result = not any(inputs)
            elif op == "XOR":
                result = sum(inputs) % 2 == 1
            elif op == "XNOR":
                result = sum(inputs) % 2 == 0
            elif op == "NOT":
                result = not inputs[0]
            elif op == "IMPLIES":
                result = (not inputs[0]) or inputs[1]
            else:  # MAJORITY
                result = sum(inputs) > len(inputs) / 2.0
            values[gate.name] = bool(result)
        return values
