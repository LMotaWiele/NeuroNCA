"""Multi-seed verdict aggregation (pre-registered Success Criteria).

Seeds S = {0, 1, 2}. Implements best_local selection, stationary_pass,
catastrophic_fail, recovered, weak_shift, phase1_go, and primary verdict enum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from neuronca.eval.time_lag import (
    DEFAULT_LAG_WINDOW_STEPS,
    DEFAULT_RECOVERY_THRESHOLD,
    analyze_lag_curve,
)

# Tie-break preference for best_local (higher index = preferred on ties)
CANDIDATE_TIE_ORDER: tuple[str, ...] = (
    "activation",
    "hebbian_corr",
    "eligibility_residual_energy",
    "eligibility_gevb_rf",
)

DEFAULT_STATIONARY_MEAN = 0.60
DEFAULT_STATIONARY_SEED_FLOOR = 0.50
DEFAULT_STATIONARY_SEED_FLOOR_MIN_COUNT = 2
DEFAULT_FAIL_RHO_CEILING = 0.30
DEFAULT_SHIFT_RECOVERY_MIN_SEED_COUNT = 2
DEFAULT_WEAK_SHIFT_CEILING = 0.90

VERDICT_FAIL = "FAIL"
VERDICT_PASS = "PASS"
VERDICT_PASS_WEAK_SHIFT = "PASS_WEAK_SHIFT"
VERDICT_PARTIAL_PASS = "PARTIAL_PASS"

VERDICT_VALUES = (
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PASS_WEAK_SHIFT,
    VERDICT_PARTIAL_PASS,
)


@dataclass
class SeedLagCurve:
    """Lag ρ series for one seed (best_local vs grad_abs)."""

    seed: int
    times: list[float]
    rhos: list[float]
    rho_pre_switch: float = float("nan")


@dataclass
class VerdictThresholds:
    stationary_edge_spearman: float = DEFAULT_STATIONARY_MEAN
    stationary_seed_floor: float = DEFAULT_STATIONARY_SEED_FLOOR
    stationary_seed_floor_min_count: int = DEFAULT_STATIONARY_SEED_FLOOR_MIN_COUNT
    shift_recovery_spearman: float = DEFAULT_RECOVERY_THRESHOLD
    shift_recovery_max_steps: int = DEFAULT_LAG_WINDOW_STEPS
    shift_recovery_min_seed_count: int = DEFAULT_SHIFT_RECOVERY_MIN_SEED_COUNT
    fail_rho_ceiling: float = DEFAULT_FAIL_RHO_CEILING
    gt_shift_disagreement_ceiling: float = DEFAULT_WEAK_SHIFT_CEILING


@dataclass
class VerdictResult:
    verdict: str
    phase1_go: bool
    best_local: str | None
    stationary_pass: bool
    recovered: bool
    weak_shift: bool
    catastrophic_fail: bool
    mean_rho_by_candidate: dict[str, float]
    seed_rho_by_candidate: dict[str, dict[int, float]]
    recovery_by_seed: dict[int, dict[str, float | bool]]
    recovery_rho_mean: float
    rho_shift_mean: float
    rho_shift_by_seed: dict[int, float]
    thresholds: dict[str, float | int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tie_rank(name: str) -> int:
    try:
        return CANDIDATE_TIE_ORDER.index(name)
    except ValueError:
        return -1  # unknown candidates lose ties to known ones


def mean_rho_per_candidate(
    seed_rho: Mapping[str, Mapping[int, float]],
) -> dict[str, float]:
    """Step 1: per-candidate aggregate mean_s ρ_s(c)."""
    out: dict[str, float] = {}
    for cand, by_seed in seed_rho.items():
        vals = [float(v) for v in by_seed.values() if np.isfinite(v)]
        out[cand] = float(np.mean(vals)) if vals else float("nan")
    return out


def select_best_local(mean_rho: Mapping[str, float]) -> str | None:
    """Step 2: argmax mean ρ; ties prefer gevb_rf > residual_energy > hebbian > activation."""
    if not mean_rho:
        return None
    # Filter to finite means; if all nan, still pick by tie order among all keys
    items = list(mean_rho.items())
    finite = [(c, r) for c, r in items if np.isfinite(r)]
    pool = finite if finite else items

    def key(item: tuple[str, float]) -> tuple[float, int]:
        name, rho = item
        r = float(rho) if np.isfinite(rho) else float("-inf")
        return (r, _tie_rank(name))

    best = max(pool, key=key)
    return best[0]


def stationary_pass_check(
    seed_rhos: Mapping[int, float],
    *,
    mean_threshold: float = DEFAULT_STATIONARY_MEAN,
    seed_floor: float = DEFAULT_STATIONARY_SEED_FLOOR,
    min_seeds_above_floor: int = DEFAULT_STATIONARY_SEED_FLOOR_MIN_COUNT,
) -> tuple[bool, float, int]:
    """Step 3: mean >= 0.60 AND >= min_seeds_above_floor seeds with ρ >= 0.50."""
    vals = [float(v) for v in seed_rhos.values()]
    finite = [v for v in vals if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    n_above = sum(1 for v in finite if v >= seed_floor)
    passed = bool(
        np.isfinite(mean)
        and mean >= mean_threshold
        and n_above >= min_seeds_above_floor
    )
    return passed, mean, n_above


def is_catastrophic_fail(
    mean_rho: Mapping[str, float],
    *,
    fail_rho_ceiling: float = DEFAULT_FAIL_RHO_CEILING,
) -> bool:
    """Step 4: max_c mean ρ(c) < fail_rho_ceiling (annotation; implies FAIL path)."""
    finite = [float(v) for v in mean_rho.values() if np.isfinite(v)]
    if not finite:
        return True
    return float(max(finite)) < fail_rho_ceiling


def compute_recovery_for_seed(
    times: Sequence[float],
    rhos: Sequence[float],
    *,
    rho_pre_switch: float = float("nan"),
    threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
) -> dict[str, float | bool]:
    """First-crossing recovery for one seed; never uses max-in-window for rho_rec."""
    result = analyze_lag_curve(
        times,
        rhos,
        rho_pre_switch=rho_pre_switch,
        threshold=threshold,
        max_t=max_t,
    )
    return {
        "t_star_abs": result.t_star_abs,
        "rho_rec": result.rho_rec,
        "rho_max_in_window": result.rho_max_in_window,
        "recovered_abs": result.recovered_abs,
        "t_star_rel": result.t_star_rel,
        "recovered_rel": result.recovered_rel,
        "rho_pre_switch": result.rho_pre_switch,
        "rho_at_window_end": result.rho_at_window_end,
        # Diagnostic only — not used by recovered_check / decide_verdict.
        "t_star_sustained": result.t_star_sustained,
        "recovered_sustained": result.recovered_sustained,
    }


def recovered_check(
    recovery_by_seed: Mapping[int, Mapping[str, float | bool]],
    *,
    max_t: float = DEFAULT_LAG_WINDOW_STEPS,
    min_seed_count: int = DEFAULT_SHIFT_RECOVERY_MIN_SEED_COUNT,
) -> bool:
    """Step 8: recovered iff >= min_seed_count seeds have finite t*_abs <= max_t."""
    n = 0
    for rec in recovery_by_seed.values():
        t_star = float(rec["t_star_abs"])  # type: ignore[arg-type]
        if np.isfinite(t_star) and t_star <= max_t:
            n += 1
    return n >= min_seed_count


def weak_shift_check(
    rho_shift_by_seed: Mapping[int, float],
    *,
    ceiling: float = DEFAULT_WEAK_SHIFT_CEILING,
) -> tuple[bool, float]:
    """Step 9: weak_shift if mean_s ρ_shift,s >= ceiling (default 0.90)."""
    vals = [float(v) for v in rho_shift_by_seed.values() if np.isfinite(v)]
    mean = float(np.mean(vals)) if vals else float("nan")
    weak = bool(np.isfinite(mean) and mean >= ceiling)
    return weak, mean


def decide_verdict(
    *,
    stationary_pass: bool,
    recovered: bool,
    weak_shift: bool,
) -> tuple[str, bool]:
    """Steps 10: primary verdict enum + phase1_go.

    phase1_go is true ONLY when verdict == PASS.
    """
    if not stationary_pass:
        return VERDICT_FAIL, False
    if recovered and not weak_shift:
        return VERDICT_PASS, True
    if recovered and weak_shift:
        return VERDICT_PASS_WEAK_SHIFT, False
    return VERDICT_PARTIAL_PASS, False


def aggregate_verdict(
    seed_rho: Mapping[str, Mapping[int, float]],
    *,
    lag_curves: Mapping[int, SeedLagCurve] | None = None,
    rho_shift_by_seed: Mapping[int, float] | None = None,
    thresholds: VerdictThresholds | None = None,
) -> VerdictResult:
    """Full multi-seed aggregation per Success Criteria.

    Parameters
    ----------
    seed_rho
        candidate → {seed → edge Spearman vs grad_abs at stationary late}.
    lag_curves
        seed → lag series for the global best_local (frozen id). If omitted,
        recovery fields are empty / recovered=False.
    rho_shift_by_seed
        seed → Spearman(I_grad late_a, I_grad switch_b0). If omitted, weak_shift=False.
    thresholds
        Optional overrides; defaults match configs/default.yaml.
    """
    thr = thresholds or VerdictThresholds()
    mean_rho = mean_rho_per_candidate(seed_rho)
    best = select_best_local(mean_rho)

    if best is None:
        notes = ["no candidates provided"]
        return VerdictResult(
            verdict=VERDICT_FAIL,
            phase1_go=False,
            best_local=None,
            stationary_pass=False,
            recovered=False,
            weak_shift=False,
            catastrophic_fail=True,
            mean_rho_by_candidate={},
            seed_rho_by_candidate={},
            recovery_by_seed={},
            recovery_rho_mean=float("nan"),
            rho_shift_mean=float("nan"),
            rho_shift_by_seed={},
            thresholds=_thresholds_dict(thr),
            notes=notes,
        )

    best_seed_rhos = dict(seed_rho.get(best, {}))
    stat_pass, _mean_best, _n_floor = stationary_pass_check(
        best_seed_rhos,
        mean_threshold=thr.stationary_edge_spearman,
        seed_floor=thr.stationary_seed_floor,
        min_seeds_above_floor=thr.stationary_seed_floor_min_count,
    )
    cat_fail = is_catastrophic_fail(mean_rho, fail_rho_ceiling=thr.fail_rho_ceiling)

    # Normalize seed_rho to plain dicts for serialization
    seed_rho_out: dict[str, dict[int, float]] = {
        c: {int(s): float(v) for s, v in by_s.items()} for c, by_s in seed_rho.items()
    }

    recovery_by_seed: dict[int, dict[str, float | bool]] = {}
    if lag_curves:
        for seed, curve in lag_curves.items():
            recovery_by_seed[int(seed)] = compute_recovery_for_seed(
                curve.times,
                curve.rhos,
                rho_pre_switch=curve.rho_pre_switch,
                threshold=thr.shift_recovery_spearman,
                max_t=float(thr.shift_recovery_max_steps),
            )

    recovered = recovered_check(
        recovery_by_seed,
        max_t=float(thr.shift_recovery_max_steps),
        min_seed_count=thr.shift_recovery_min_seed_count,
    ) if recovery_by_seed else False

    rho_recs = [
        float(r["rho_rec"])  # type: ignore[arg-type]
        for r in recovery_by_seed.values()
        if np.isfinite(float(r["rho_rec"]))  # type: ignore[arg-type]
    ]
    recovery_rho_mean = float(np.mean(rho_recs)) if rho_recs else float("nan")

    shift_map = {int(s): float(v) for s, v in (rho_shift_by_seed or {}).items()}
    weak, rho_shift_mean = weak_shift_check(
        shift_map, ceiling=thr.gt_shift_disagreement_ceiling
    )
    # If no shift measurements, weak_shift stays false
    if not shift_map:
        weak = False
        rho_shift_mean = float("nan")

    verdict, phase1_go = decide_verdict(
        stationary_pass=stat_pass,
        recovered=recovered,
        weak_shift=weak,
    )

    # Catastrophic fail is annotation; if it triggers, stationary should have failed
    notes: list[str] = []
    if cat_fail:
        notes.append(
            "catastrophic_fail: max mean rho < fail_rho_ceiling; rethink info access."
        )
    if verdict == VERDICT_PASS_WEAK_SHIFT:
        notes.append(
            "weak_shift: mean rho_shift >= ceiling; run B2/B3 contingency before Phase 1."
        )
    notes.append("phase1_go is true only when verdict == PASS.")
    notes.append("recovery_rho_definition: first_crossing_or_rho_at_window_end")

    # Safety: phase1_go only when PASS
    assert phase1_go == (verdict == VERDICT_PASS)

    return VerdictResult(
        verdict=verdict,
        phase1_go=phase1_go,
        best_local=best,
        stationary_pass=stat_pass,
        recovered=recovered,
        weak_shift=weak,
        catastrophic_fail=cat_fail,
        mean_rho_by_candidate=mean_rho,
        seed_rho_by_candidate=seed_rho_out,
        recovery_by_seed=recovery_by_seed,
        recovery_rho_mean=recovery_rho_mean,
        rho_shift_mean=rho_shift_mean,
        rho_shift_by_seed=shift_map,
        thresholds=_thresholds_dict(thr),
        notes=notes,
    )


def _thresholds_dict(thr: VerdictThresholds) -> dict[str, float | int]:
    return {
        "stationary_edge_spearman": thr.stationary_edge_spearman,
        "stationary_seed_floor": thr.stationary_seed_floor,
        "stationary_seed_floor_min_count": thr.stationary_seed_floor_min_count,
        "shift_recovery_spearman": thr.shift_recovery_spearman,
        "shift_recovery_max_steps": thr.shift_recovery_max_steps,
        "shift_recovery_min_seed_count": thr.shift_recovery_min_seed_count,
        "fail_rho_ceiling": thr.fail_rho_ceiling,
        "gt_shift_disagreement_ceiling": thr.gt_shift_disagreement_ceiling,
    }
