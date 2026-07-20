"""Spearman / Pearson correlations for local scores vs ground-truth importance.

Mask-restricted edge Spearman (GT-2 subsample), per-layer metrics, and overall
metrics with optional per-layer z-score before concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import stats

DEFAULT_SPEARMAN_MIN_PAIRS = 16


def _as_1d_float(x: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr


def _apply_mask(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} vs {y.shape}")
    if mask is None:
        return x, y
    m = np.asarray(mask).ravel()
    if m.dtype == bool:
        if m.shape != x.shape:
            raise ValueError(f"boolean mask shape {m.shape} != data shape {x.shape}")
        return x[m], y[m]
    # integer index mask
    idx = m.astype(np.int64, copy=False)
    return x[idx], y[idx]


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def spearman_rho(
    x: np.ndarray | list | tuple,
    y: np.ndarray | list | tuple,
    *,
    mask: np.ndarray | None = None,
    min_pairs: int = DEFAULT_SPEARMAN_MIN_PAIRS,
) -> float:
    """Spearman ρ between two vectors, optionally restricted to ``mask``.

    Returns ``nan`` when fewer than ``min_pairs`` finite pairs remain after
    masking (default ``spearman_min_pairs=16``).
    """
    xa, ya = _apply_mask(np.asarray(x), np.asarray(y), mask)
    xa, ya = _finite_pairs(xa, ya)
    if xa.size < min_pairs:
        return float("nan")
    # Constant vectors → undefined rank correlation
    if np.unique(xa).size < 2 or np.unique(ya).size < 2:
        return float("nan")
    rho, _ = stats.spearmanr(xa, ya)
    return float(rho)


def pearson_r(
    x: np.ndarray | list | tuple,
    y: np.ndarray | list | tuple,
    *,
    mask: np.ndarray | None = None,
    min_pairs: int = DEFAULT_SPEARMAN_MIN_PAIRS,
) -> float:
    """Pearson r (secondary metric); same mask / min_pairs conventions as Spearman."""
    xa, ya = _apply_mask(np.asarray(x), np.asarray(y), mask)
    xa, ya = _finite_pairs(xa, ya)
    if xa.size < min_pairs:
        return float("nan")
    if np.unique(xa).size < 2 or np.unique(ya).size < 2:
        return float("nan")
    r, _ = stats.pearsonr(xa, ya)
    return float(r)


def _zscore(v: np.ndarray) -> np.ndarray:
    v = _as_1d_float(v)
    mu = np.nanmean(v)
    sigma = np.nanstd(v)
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.zeros_like(v, dtype=np.float64)
    return (v - mu) / sigma


@dataclass(frozen=True)
class LayerCorrelation:
    """Correlation results for a single layer."""

    layer: str
    spearman: float
    pearson: float
    n_pairs: int


@dataclass
class CorrelationResult:
    """Per-layer and overall correlations for one (local, GT) pair."""

    per_layer: dict[str, LayerCorrelation] = field(default_factory=dict)
    overall_spearman: float = float("nan")
    overall_pearson: float = float("nan")
    overall_n_pairs: int = 0
    skipped_layers: list[str] = field(default_factory=list)


def correlate_layers(
    local: Mapping[str, np.ndarray],
    gt: Mapping[str, np.ndarray],
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    layer_zscore_before_concat: bool = True,
    min_pairs: int = DEFAULT_SPEARMAN_MIN_PAIRS,
) -> CorrelationResult:
    """Compute per-layer and overall Spearman/Pearson between local and GT scores.

    Parameters
    ----------
    local, gt
        Mapping layer_name → score array (any shape; flattened). Equal layer keys
        required for each layer included; layers missing from either side are
        skipped and recorded in ``skipped_layers``.
    masks
        Optional per-layer mask (boolean or integer indices). Applied before
        correlation (edge subsample protocol for GT-2).
    layer_zscore_before_concat
        If True (default), z-score each layer's local and GT vectors before
        concatenating for the overall metric.
    min_pairs
        Minimum finite pairs for a defined Spearman (default 16).
    """
    result = CorrelationResult()
    local_parts: list[np.ndarray] = []
    gt_parts: list[np.ndarray] = []

    common = sorted(set(local.keys()) & set(gt.keys()))
    only_local = set(local.keys()) - set(gt.keys())
    only_gt = set(gt.keys()) - set(local.keys())
    for name in sorted(only_local | only_gt):
        result.skipped_layers.append(name)

    for layer in common:
        lx = _as_1d_float(local[layer])
        gy = _as_1d_float(gt[layer])
        mask = None if masks is None else masks.get(layer)
        try:
            xa, ya = _apply_mask(lx, gy, mask)
        except ValueError:
            result.skipped_layers.append(layer)
            continue
        xa, ya = _finite_pairs(xa, ya)
        n = int(xa.size)
        sp = spearman_rho(xa, ya, min_pairs=min_pairs) if n else float("nan")
        pe = pearson_r(xa, ya, min_pairs=min_pairs) if n else float("nan")
        # spearman_rho already handles min_pairs on the unmasked path; re-apply
        # here with already-masked arrays for clarity.
        if n < min_pairs:
            sp = float("nan")
            pe = float("nan")
        result.per_layer[layer] = LayerCorrelation(
            layer=layer, spearman=sp, pearson=pe, n_pairs=n
        )
        if n == 0:
            continue
        if layer_zscore_before_concat:
            local_parts.append(_zscore(xa))
            gt_parts.append(_zscore(ya))
        else:
            local_parts.append(xa)
            gt_parts.append(ya)

    if local_parts:
        x_all = np.concatenate(local_parts)
        y_all = np.concatenate(gt_parts)
        result.overall_n_pairs = int(x_all.size)
        result.overall_spearman = spearman_rho(x_all, y_all, min_pairs=min_pairs)
        result.overall_pearson = pearson_r(x_all, y_all, min_pairs=min_pairs)
    return result
