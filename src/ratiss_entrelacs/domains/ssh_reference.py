"""Driven SSH reference field, adapted from RATISS/Travaux (Apache-2.0).

Modified in RATISS-ENTRELACS: input validation, typed API, and naming. This is
a computational toy Hamiltonian with an exact phase boundary; it is one
couplable field-domain, not a material prediction or the whole processor.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ssh_hamiltonian(n_sites: int, delta: float, hopping: float = 1.0) -> NDArray[np.float64]:
    """Open-boundary SSH chain; even site count required."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 2 or n_sites % 2:
        raise ValueError("n_sites must be a positive even integer >= 2")
    if not np.isfinite(delta) or not np.isfinite(hopping):
        raise ValueError("delta and hopping must be finite")
    t1, t2 = hopping * (1.0 + delta), hopping * (1.0 - delta)
    h = np.zeros((n_sites, n_sites), dtype=np.float64)
    for i in range(n_sites - 1):
        value = t1 if i % 2 == 0 else t2
        h[i, i + 1] = h[i + 1, i] = value
    return h


def ground_state_orbitals(hamiltonian: ArrayLike) -> NDArray[np.complex128]:
    """Lowest half-filled single-particle orbitals of a Hermitian matrix."""
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2 or h.shape[0] % 2:
        raise ValueError("hamiltonian must be a square even-sized matrix")
    if not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    _, vectors = np.linalg.eigh(h)
    return vectors[:, : h.shape[0] // 2]


def correlation_matrix(orbitals: ArrayLike) -> NDArray[np.complex128]:
    """Projector ``C = orbitals @ orbitals†``."""
    values = np.asarray(orbitals, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("orbitals must be a finite non-empty matrix")
    return values @ values.conj().T


def spectral_gap(hamiltonian: ArrayLike) -> float:
    """Finite-chain gap at half filling."""
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2 or h.shape[0] % 2:
        raise ValueError("hamiltonian must be a square even-sized matrix")
    eigs = np.linalg.eigvalsh(h)
    return float(eigs[h.shape[0] // 2] - eigs[h.shape[0] // 2 - 1])


def edge_weight(correlation: ArrayLike) -> float:
    """Boundary-to-boundary correlation magnitude ``|C[0, -1]|``."""
    values = np.asarray(correlation)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("correlation must be a square matrix of size >= 2")
    return float(abs(values[0, -1]))


def evolve_orbitals(
    orbitals0: ArrayLike,
    delta_of_time,
    *,
    dt: float,
    steps: int,
    time0: float = 0.0,
) -> NDArray[np.complex128]:
    """Propagate ``i dψ/dt = H(t)ψ`` with RK4; return the full trajectory."""
    psi0 = np.asarray(orbitals0, dtype=np.complex128)
    if psi0.ndim != 2 or psi0.shape[0] < 2 or not np.all(np.isfinite(psi0)):
        raise ValueError("orbitals0 must be a finite matrix with at least 2 sites")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(time0):
        raise ValueError("dt must be positive and time0 finite")

    output = np.empty((int(steps) + 1,) + psi0.shape, dtype=np.complex128)
    output[0] = psi0

    def rhs(time: float, psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return -1j * (ssh_hamiltonian(psi.shape[0], float(delta_of_time(time))) @ psi)

    psi = psi0.copy()
    time = float(time0)
    for index in range(1, int(steps) + 1):
        k1 = rhs(time, psi)
        k2 = rhs(time + dt / 2.0, psi + dt * k1 / 2.0)
        k3 = rhs(time + dt / 2.0, psi + dt * k2 / 2.0)
        k4 = rhs(time + dt, psi + dt * k3)
        psi = psi + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        time += dt
        output[index] = psi
    return output
