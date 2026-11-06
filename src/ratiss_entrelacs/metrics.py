"""Independent observables for phase, graph structure and path dependence."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def phase_coherence(phases: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """Circular order parameter ``|mean(exp(i*phase))|`` along ``axis``."""
    values = np.asarray(phases, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("phases must be non-empty and finite")
    result = np.abs(np.mean(np.exp(1j * values), axis=axis))
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=np.float64)


def graph_cycle_rank(adjacency: ArrayLike) -> int:
    """Undirected graph cycle rank ``β₁ = E − V + C`` (isolated nodes count)."""
    matrix = np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.all(np.isfinite(matrix.astype(np.float64))):
        raise ValueError("adjacency must be finite")
    edges = matrix != 0
    np.fill_diagonal(edges, False)
    if not np.array_equal(edges, edges.T):
        raise ValueError("adjacency must be symmetric for an undirected graph")

    n = edges.shape[0]
    seen: set[int] = set()
    components = 0
    for start in range(n):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(edges[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    n_edges = int(np.count_nonzero(np.triu(edges, 1)))
    return n_edges - n + components


def hysteresis_area(
    x_forward: ArrayLike,
    y_forward: ArrayLike,
    x_return: ArrayLike,
    y_return: ArrayLike,
) -> float:
    """Area between two measured branches over their shared ``x`` interval.

    Inputs may be ordered either way, but each branch must have distinct,
    finite x-values. This is a path-dependence observable, not memory proof.
    """
    xf = np.asarray(x_forward, dtype=np.float64)
    yf = np.asarray(y_forward, dtype=np.float64)
    xr = np.asarray(x_return, dtype=np.float64)
    yr = np.asarray(y_return, dtype=np.float64)
    if xf.ndim != 1 or xr.ndim != 1 or yf.shape != xf.shape or yr.shape != xr.shape:
        raise ValueError("each branch must be a pair of same-length 1-D arrays")
    if xf.size < 2 or xr.size < 2:
        raise ValueError("each branch must contain at least two samples")
    if not all(np.all(np.isfinite(a)) for a in (xf, yf, xr, yr)):
        raise ValueError("branch samples must be finite")
    if np.unique(xf).size != xf.size or np.unique(xr).size != xr.size:
        raise ValueError("x values in each branch must be distinct")

    lo = max(float(xf.min()), float(xr.min()))
    hi = min(float(xf.max()), float(xr.max()))
    if hi <= lo:
        return 0.0
    grid = np.unique(np.concatenate(([lo, hi], xf[(xf >= lo) & (xf <= hi)], xr[(xr >= lo) & (xr <= hi)])))
    of = np.argsort(xf)
    or_ = np.argsort(xr)
    f = np.interp(grid, xf[of], yf[of])
    r = np.interp(grid, xr[or_], yr[or_])
    return float(np.sum(0.5 * (np.abs(f[:-1] - r[:-1]) + np.abs(f[1:] - r[1:])) * np.diff(grid)))
