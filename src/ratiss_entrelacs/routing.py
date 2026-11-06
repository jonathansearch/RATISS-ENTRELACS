"""Sparse route selection as an optional operator (VEGA/X family)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def topk_route(scores: ArrayLike, k: int, temperature: float = 1.0) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Select the ``k`` highest scores and softmax only those routes.

    Stable ordering makes ties deterministic. This is a reusable routing
    primitive, not the full RATISS-X or VEGA router.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite 1-D array")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= values.size:
        raise ValueError("k must be an integer between 1 and len(scores)")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    indices = np.argsort(-values, kind="stable")[: int(k)]
    logits = values[indices] / float(temperature)
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    return indices.astype(np.int64, copy=False), weights.astype(np.float64, copy=False)


def sparse_mix(values: ArrayLike, scores: ArrayLike, k: int, temperature: float = 1.0) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Mix selected routes, returning output, route indices and route weights."""
    candidates = np.asarray(values, dtype=np.float64)
    if candidates.ndim < 1 or candidates.shape[0] == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError("values must be a non-empty finite array with routes on axis 0")
    route_scores = np.asarray(scores, dtype=np.float64)
    if route_scores.ndim != 1 or route_scores.size != candidates.shape[0]:
        raise ValueError("scores length must match values axis 0")
    indices, weights = topk_route(route_scores, k, temperature)
    mixed = np.tensordot(weights, candidates[indices], axes=(0, 0))
    return np.asarray(mixed, dtype=np.float64), indices, weights
