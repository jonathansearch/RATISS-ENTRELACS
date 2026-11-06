"""RATISS-ENTRELACS: in-silico adaptive weighted-matrix processor prototype."""

from .baselines.phase_ring import ForcedRing, ring_coupling
from .cells import DendriticConfig, DendriticPopulation, DendriticStep
from .couplings import density_phase_bias, local_density, phase_to_radial_anchors
from .domains.ssh_reference import ssh_hamiltonian, evolve_orbitals
from .engine import SimulationResult, run
from .logic import LogicGate, LogicGateBindings, LogicGateLayer
from .matrix_processor import (
    DynamicMatrixProcessor,
    MatrixProcessorConfig,
    ProcessorStep,
    ProcessorTrace,
    STDPConfig,
    weighted_fold,
    weighted_fold_channels,
)
from .metrics import graph_cycle_rank, hysteresis_area, phase_coherence
from .routing import sparse_mix, topk_route
from .system import Coupling, InterlacedSystem, RepeatingSchedule, Snapshot, SystemRun
from .topology import psig_from_correlation, psig_from_points, rips_persistence

__all__ = [
    "Coupling",
    "DendriticConfig",
    "DendriticPopulation",
    "DendriticStep",
    "DynamicMatrixProcessor",
    "ForcedRing",
    "MatrixProcessorConfig",
    "ProcessorStep",
    "ProcessorTrace",
    "STDPConfig",
    "InterlacedSystem",
    "LogicGate",
    "LogicGateBindings",
    "LogicGateLayer",
    "RepeatingSchedule",
    "SimulationResult",
    "Snapshot",
    "SystemRun",
    "density_phase_bias",
    "graph_cycle_rank",
    "hysteresis_area",
    "local_density",
    "phase_coherence",
    "phase_to_radial_anchors",
    "psig_from_correlation",
    "psig_from_points",
    "ring_coupling",
    "rips_persistence",
    "run",
    "sparse_mix",
    "ssh_hamiltonian",
    "evolve_orbitals",
    "topk_route",
    "weighted_fold",
    "weighted_fold_channels",
]
