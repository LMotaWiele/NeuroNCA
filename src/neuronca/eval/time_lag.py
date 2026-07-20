"""Time-lag recovery analysis: first-crossing t*_abs and rho_rec.

Primary recovery (verdict) uses absolute first crossing of threshold 0.60.
``rho_rec`` is ρ at the first crossing, or ρ at the window end if none —
**never** max-in-window. Relative recovery (0.9 * rho_pre_switch) is secondary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_RECOVERY_THRESHOLD = 0.60
DEFAULT_LAG_WINDOW_STEPS = 500
DEFAULT_RELATIVE_RECOVERY_FACTOR = 0.9


@dataclass(frozen=True)
class LagRecoveryResult:
    """Per-seed lag recovery summary for one (local, GT) pair."""

    t_star_abs: float
    """First t with rho(t) >= threshold, or +inf if none within window."""

    rho_rec: float
    """ρ at t*_abs if finite, else ρ at the last lag point (window end)."""

    rho_max_in_window: float
    """max_t ρ(t) for reporting only — not used for rho_rec or pass predicates."""

    recovered_abs: bool
    """True iff t*_abs is finite (crossed absolute threshold within window)."""

    t_star_rel: float
    """First t meeting absolute and relative thresholds, or +inf."""

    recovered_rel: bool
    """True iff t*_rel is finite."""

    rho_pre_switch: float
    """Pre-switch (late_a) Spearman used for relative recovery; may be nan."""

    rho_at_window_end: float
    """ρ at the last lag point (typically t=500)."""


def first_crossing(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
) -> float:
    """Return min { t <= max_t : rho(t) >= threshold }, or +inf if none.

    Times are scanned in the order provided (must be sorted ascending for
    standard lag schedules). Only points with t <= max_t are considered.
    Non-finite ρ values never count as crossings.
    """
    t_arr = np.asarray(times, dtype=np.float64)
    r_arr = np.asarray(rhos, dtype=np.float64)
    if t_arr.shape != r_arr.shape:
        raise ValueError("times and rhos must have the same shape")
    if t_arr.ndim != 1:
        raise ValueError("times and rhos must be 1-D")

    best = float("inf")
    for t, r in zip(t_arr.tolist(), r_arr.tolist()):
        if not np.isfinite(t) or t > max_t:
            continue
        if np.isfinite(r) and r >= threshold:
            if t < best:
                best = float(t)
    return best


def rho_at_time(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    t_star: float,
) -> float:
    """Return ρ at the first index whose time equals ``t_star``; nan if missing."""
    if not np.isfinite(t_star):
        return float("nan")
    t_arr = np.asarray(times, dtype=np.float64)
    r_arr = np.asarray(rhos, dtype=np.float64)
    matches = np.where(np.isclose(t_arr, t_star) | (t_arr == t_star))[0]
    if matches.size == 0:
        # exact equality fallback for integer schedules
        matches = np.where(t_arr == t_star)[0]
    if matches.size == 0:
        return float("nan")
    return float(r_arr[int(matches[0])])


def rho_at_window_end(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    *,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
) -> float:
    """ρ at the largest t <= max_t present in the series (window end)."""
    t_arr = np.asarray(times, dtype=np.float64)
    r_arr = np.asarray(rhos, dtype=np.float64)
    valid = np.isfinite(t_arr) & (t_arr <= max_t)
    if not np.any(valid):
        return float("nan")
    # Prefer exact max_t if present; else last point with t <= max_t
    idx_exact = np.where(valid & (t_arr == max_t))[0]
    if idx_exact.size:
        return float(r_arr[int(idx_exact[-1])])
    idxs = np.where(valid)[0]
    # take the one with largest t
    best_i = idxs[int(np.argmax(t_arr[idxs]))]
    return float(r_arr[int(best_i)])


def compute_rho_rec(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
) -> tuple[float, float]:
    """Return (t_star_abs, rho_rec) using first-crossing semantics.

    rho_rec = rho(t*) if t* finite else rho(window_end).
    Never uses max-in-window.
    """
    t_star = first_crossing(times, rhos, threshold=threshold, max_t=max_t)
    if np.isfinite(t_star):
        return t_star, rho_at_time(times, rhos, t_star)
    return t_star, rho_at_window_end(times, rhos, max_t=max_t)


def relative_first_crossing(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    *,
    rho_pre_switch: float,
    abs_threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    relative_factor: float = DEFAULT_RELATIVE_RECOVERY_FACTOR,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
) -> float:
    """Secondary relative recovery: first t with rho >= abs_threshold and
    rho >= relative_factor * rho_pre_switch when rho_pre_switch >= abs_threshold.

    If rho_pre_switch is nan or < abs_threshold, relative recovery requires only
    the absolute threshold (still logged as secondary; does not affect verdict).
    """
    t_arr = np.asarray(times, dtype=np.float64)
    r_arr = np.asarray(rhos, dtype=np.float64)

    if np.isfinite(rho_pre_switch) and rho_pre_switch >= abs_threshold:
        rel_thr = relative_factor * rho_pre_switch
    else:
        # Design: relative criterion is meaningful when pre-switch was strong;
        # otherwise fall back to absolute-only for the secondary metric.
        rel_thr = abs_threshold

    best = float("inf")
    for t, r in zip(t_arr.tolist(), r_arr.tolist()):
        if not np.isfinite(t) or t > max_t:
            continue
        if np.isfinite(r) and r >= abs_threshold and r >= rel_thr:
            if t < best:
                best = float(t)
    return best


def analyze_lag_curve(
    times: Sequence[float] | np.ndarray,
    rhos: Sequence[float] | np.ndarray,
    *,
    rho_pre_switch: float = float("nan"),
    threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
    relative_factor: float = DEFAULT_RELATIVE_RECOVERY_FACTOR,
) -> LagRecoveryResult:
    """Full lag recovery analysis for one seed / candidate lag curve.

    Parameters
    ----------
    times, rhos
        Dense lag schedule (e.g. t = 0, 25, ..., 500) and corresponding Spearman ρ.
    rho_pre_switch
        Edge Spearman vs GT-1 at late_a / pre_switch_a (for secondary relative recovery).
    """
    t_arr = np.asarray(times, dtype=np.float64)
    r_arr = np.asarray(rhos, dtype=np.float64)
    if t_arr.shape != r_arr.shape or t_arr.ndim != 1:
        raise ValueError("times and rhos must be equal-length 1-D sequences")

    t_star_abs, rho_rec = compute_rho_rec(
        t_arr, r_arr, threshold=threshold, max_t=max_t
    )
    # Reporting only — never used for rho_rec
    valid = np.isfinite(t_arr) & (t_arr <= max_t) & np.isfinite(r_arr)
    if np.any(valid):
        rho_max = float(np.max(r_arr[valid]))
    else:
        rho_max = float("nan")

    t_star_rel = relative_first_crossing(
        t_arr,
        r_arr,
        rho_pre_switch=rho_pre_switch,
        abs_threshold=threshold,
        relative_factor=relative_factor,
        max_t=max_t,
    )
    rho_end = rho_at_window_end(t_arr, r_arr, max_t=max_t)

    return LagRecoveryResult(
        t_star_abs=float(t_star_abs),
        rho_rec=float(rho_rec),
        rho_max_in_window=rho_max,
        recovered_abs=bool(np.isfinite(t_star_abs)),
        t_star_rel=float(t_star_rel),
        recovered_rel=bool(np.isfinite(t_star_rel)),
        rho_pre_switch=float(rho_pre_switch),
        rho_at_window_end=float(rho_end),
    )
