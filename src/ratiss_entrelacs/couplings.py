"""Cross-domain adapters for phase ↔ geometry links seen in RATISS-FOCAL.

These functions translate representations. They do not prescribe a full
coupled dynamics; they can be attached to ``Coupling`` edges and ablated.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def phase_to_radial_anchors(
    phases: ArrayLike,
    *,
    radius: float = 2.2,
    displacement: float = 0.15,
) -> NDArray[np.float64]:
    """Map phases to points on a ring with phase-dependent radial displacement.

    This small adapter follows the geometry/phase construction in
    ``ratiss-focal/experiences/exp22_kuramoto_fort.py``.
    """
    values = np.asarray(phases, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("phases must be a finite 1-D array with at least 3 entries")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(displacement):
        raise ValueError("radius must be positive and parameters finite")
    angles = np.linspace(0.0, 2.0 * np.pi, values.size, endpoint=False)
    base = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(values.size)))
    radial = base / radius
    return base + radial * (float(displacement) * np.sin(values))[:, None]


def local_density(
    positions: ArrayLike,
    environment: ArrayLike,
    *,
    neighbors: int = 5,
) -> NDArray[np.float64]:
    """Estimate normalized local density from nearest environment points.

    Density is the inverse mean distance to the nearest ``neighbors`` points,
    min-max normalized to ``[0, 1]``. This is an adapter inspired by the
    density-to-phase coupling explored in FOCAL exp68, not a physical density.
    """
    query = np.asarray(positions, dtype=np.float64)
    field = np.asarray(environment, dtype=np.float64)
    if query.ndim != 2 or field.ndim != 2 or query.shape[1] != field.shape[1]:
        raise ValueError("positions and environment must be 2-D arrays with equal coordinate dimension")
    if query.shape[0] == 0 or field.shape[0] == 0 or query.shape[1] == 0:
        raise ValueError("positions and environment must be non-empty")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(field)):
        raise ValueError("positions and environment must be finite")
    if isinstance(neighbors, bool) or not isinstance(neighbors, (int, np.integer)) or not 1 <= neighbors <= field.shape[0]:
        raise ValueError("neighbors must be an integer between 1 and len(environment)")

    distances = np.linalg.norm(query[:, None, :] - field[None, :, :], axis=2)
    nearest = np.partition(distances, int(neighbors) - 1, axis=1)[:, : int(neighbors)]
    inverse_mean = 1.0 / (np.mean(nearest, axis=1) + 1e-9)
    span = float(inverse_mean.max() - inverse_mean.min())
    return (inverse_mean - inverse_mean.min()) / max(span, 1e-9)


def density_phase_bias(density: ArrayLike, gain: float = 1.0) -> NDArray[np.float64]:
    """Convert normalized local density to a frequency/phase increment bias."""
    values = np.asarray(density, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("density must be a finite non-empty 1-D array")
    if not np.isfinite(gain):
        raise ValueError("gain must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("density values must lie in [0, 1]")
    return float(gain) * (values - 0.5)
