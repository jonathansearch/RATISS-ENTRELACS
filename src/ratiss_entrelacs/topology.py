"""Persistent-graph observables adapted from RATISS/Travaux (Apache-2.0).

Modified in RATISS-ENTRELACS: explicit input/diagonal validation, typed API,
and a separate point-cloud entry point. The source pipeline is preserved:
correlation profiles -> Euclidean distances -> Vietoris-Rips filtration ->
longest finite H1 lifetime (P_sig). It is one observer, not a universal
measure of computation quality.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _ordered_simplices(distance: NDArray[np.float64], max_edge: float) -> list[tuple[tuple[int, ...], float]]:
    n = distance.shape[0]
    simplices: list[tuple[tuple[int, ...], float]] = [((i,), 0.0) for i in range(n)]
    for i, j in combinations(range(n), 2):
        birth = float(distance[i, j])
        if birth <= max_edge:
            simplices.append(((i, j), birth))
    for i, j, k in combinations(range(n), 3):
        birth = float(max(distance[i, j], distance[i, k], distance[j, k]))
        if birth <= max_edge:
            simplices.append(((i, j, k), birth))
    return sorted(simplices, key=lambda item: (item[1], len(item[0]), item[0]))


def rips_persistence(distance: ArrayLike, max_edge: float | None = None) -> dict[str, Any]:
    """Compute Vietoris-Rips H0/H1 persistence by GF(2) boundary reduction.

    Simplices are included through dimension 2, so H1 cycles can be born and
    killed by triangles. This small implementation is intended for modest
    state graphs and reproducible software experiments.
    """
    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be a square matrix")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-9):
        raise ValueError("distance must be finite and symmetric")
    if np.any(matrix < -1e-9):
        raise ValueError("distance cannot contain negative values")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-9):
        raise ValueError("distance diagonal must be zero")

    n = matrix.shape[0]
    if n == 0:
        return {"diagrams": {"H0": [], "H1": []}, "betti": [0, 0, 0], "psig": 0.0}
    if max_edge is None:
        upper = matrix[np.triu_indices(n, 1)]
        max_edge = float(np.max(upper)) if upper.size else 0.0
    if not np.isfinite(max_edge) or max_edge < 0:
        raise ValueError("max_edge must be finite and non-negative")

    simplices = _ordered_simplices(matrix, float(max_edge))
    index = {simplex: position for position, (simplex, _) in enumerate(simplices)}
    births = [birth for _, birth in simplices]
    dimensions = [len(simplex) - 1 for simplex, _ in simplices]
    boundaries: list[set[int]] = []
    for simplex, _ in simplices:
        if len(simplex) == 1:
            boundaries.append(set())
        else:
            boundaries.append({
                index[tuple(vertex for pos, vertex in enumerate(simplex) if pos != removed)]
                for removed in range(len(simplex))
            })

    reduced: dict[int, set[int]] = {}
    low_to_column: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    creators: set[int] = set()
    for column, raw_boundary in enumerate(boundaries):
        boundary = set(raw_boundary)
        while boundary and max(boundary) in low_to_column:
            boundary.symmetric_difference_update(reduced[low_to_column[max(boundary)]])
        if not boundary:
            creators.add(column)
        else:
            low = max(boundary)
            reduced[column] = boundary
            low_to_column[low] = column
            pairs.append((low, column))
            creators.discard(low)

    diagrams: dict[str, list[list[float | None]]] = {"H0": [], "H1": []}
    for birth_index, death_index in pairs:
        dimension = dimensions[birth_index]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[birth_index]), float(births[death_index])])
    for creator in creators:
        dimension = dimensions[creator]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[creator]), None])

    tolerance = 1e-9
    finite_h1 = [
        death - birth
        for birth, death in diagrams["H1"]
        if death is not None and death - birth > tolerance
    ]
    betti = [sum(1 for _, death in diagrams[f"H{dim}"] if death is None) for dim in (0, 1)] + [0]
    return {
        "diagrams": diagrams,
        "betti": betti,
        "psig": float(max(finite_h1, default=0.0)),
        "n_finite_h1": len(finite_h1),
        "max_edge": float(max_edge),
    }


def psig_from_correlation(correlation: ArrayLike, max_edge: float | None = None) -> float:
    """P_sig from absolute correlation profiles, following the Travaux pipeline."""
    values = np.asarray(correlation)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError("correlation must be a finite square matrix")
    profiles = np.abs(values)
    differences = profiles[:, None, :] - profiles[None, :, :]
    distance = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distance, 0.0)
    return float(rips_persistence(distance, max_edge=max_edge)["psig"])


def psig_from_points(points: ArrayLike, max_edge: float | None = None) -> float:
    """P_sig from a finite point cloud; kept distinct from correlation P_sig."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("points must be a finite non-empty 2-D array")
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distance, 0.0)
    return float(rips_persistence(distance, max_edge=max_edge)["psig"])
