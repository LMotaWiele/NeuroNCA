"""Unit tests for multi-seed verdict aggregation and first-crossing recovery."""

from __future__ import annotations

import numpy as np
import pytest

from neuronca.eval.time_lag import analyze_lag_curve, compute_rho_rec
from neuronca.eval.verdict import (
    VERDICT_FAIL,
    VERDICT_PARTIAL_PASS,
    VERDICT_PASS,
    VERDICT_PASS_WEAK_SHIFT,
    SeedLagCurve,
    VerdictThresholds,
    aggregate_verdict,
    decide_verdict,
    is_catastrophic_fail,
    mean_rho_per_candidate,
    recovered_check,
    select_best_local,
    stationary_pass_check,
    weak_shift_check,
)


# ---------------------------------------------------------------------------
# Design-required non-monotonic lag curve (Success Criteria §13)
# ---------------------------------------------------------------------------


def test_non_monotonic_lag_curve_first_crossing():
    """ρ: 0.4 → 0.65 → 0.55 → 0.70 → t*_abs is second point; rho_rec=0.65 not 0.70.

    Never use max-in-window for rho_rec.
    """
    times = [0, 25, 50, 75]
    rhos = [0.4, 0.65, 0.55, 0.70]
    t_star, rho_rec = compute_rho_rec(times, rhos, threshold=0.60, max_t=500)
    assert t_star == 25.0
    assert rho_rec == pytest.approx(0.65)
    assert rho_rec != pytest.approx(0.70)

    result = analyze_lag_curve(times, rhos)
    assert result.t_star_abs == 25.0
    assert result.rho_rec == pytest.approx(0.65)
    assert result.rho_max_in_window == pytest.approx(0.70)


def test_non_monotonic_via_aggregate_verdict_recovery_fields():
    """Same non-monotonic curve wired through aggregate_verdict recovery path."""
    seed_rho = {
        "eligibility_gevb_rf": {0: 0.70, 1: 0.72, 2: 0.68},
        "activation": {0: 0.20, 1: 0.22, 2: 0.18},
    }
    times = [0.0, 25.0, 50.0, 75.0]
    rhos = [0.4, 0.65, 0.55, 0.70]
    lag_curves = {
        s: SeedLagCurve(seed=s, times=times, rhos=rhos, rho_pre_switch=0.75)
        for s in (0, 1, 2)
    }
    out = aggregate_verdict(seed_rho, lag_curves=lag_curves, rho_shift_by_seed={0: 0.5, 1: 0.5, 2: 0.5})
    for s in (0, 1, 2):
        rec = out.recovery_by_seed[s]
        assert rec["t_star_abs"] == 25.0
        assert rec["rho_rec"] == pytest.approx(0.65)
        assert rec["rho_max_in_window"] == pytest.approx(0.70)
        assert rec["recovered_abs"] is True


# ---------------------------------------------------------------------------
# best_local selection and tie-break
# ---------------------------------------------------------------------------


def test_mean_rho_and_best_local():
    seed_rho = {
        "activation": {0: 0.4, 1: 0.5, 2: 0.6},
        "hebbian_corr": {0: 0.55, 1: 0.55, 2: 0.55},
        "eligibility_residual_energy": {0: 0.7, 1: 0.7, 2: 0.7},
        "eligibility_gevb_rf": {0: 0.65, 1: 0.65, 2: 0.65},
    }
    means = mean_rho_per_candidate(seed_rho)
    assert means["eligibility_residual_energy"] == pytest.approx(0.7)
    assert select_best_local(means) == "eligibility_residual_energy"


def test_best_local_tie_break_prefers_gevb():
    """On equal mean ρ, prefer gevb_rf > residual_energy > hebbian > activation."""
    means = {
        "activation": 0.60,
        "hebbian_corr": 0.60,
        "eligibility_residual_energy": 0.60,
        "eligibility_gevb_rf": 0.60,
    }
    assert select_best_local(means) == "eligibility_gevb_rf"

    means2 = {
        "activation": 0.60,
        "hebbian_corr": 0.60,
        "eligibility_residual_energy": 0.60,
    }
    assert select_best_local(means2) == "eligibility_residual_energy"

    means3 = {"activation": 0.60, "hebbian_corr": 0.60}
    assert select_best_local(means3) == "hebbian_corr"


# ---------------------------------------------------------------------------
# stationary_pass, catastrophic_fail
# ---------------------------------------------------------------------------


def test_stationary_pass_mean_and_seed_floor():
    # mean 0.62, all seeds >= 0.50
    ok, mean, n = stationary_pass_check({0: 0.55, 1: 0.65, 2: 0.66})
    assert ok is True
    assert mean == pytest.approx(0.62)
    assert n == 3

    # mean high but only 1/3 seeds above floor
    ok2, _, n2 = stationary_pass_check({0: 0.90, 1: 0.40, 2: 0.40})
    assert ok2 is False
    assert n2 == 1

    # mean below 0.60 even if all above floor
    ok3, mean3, n3 = stationary_pass_check({0: 0.50, 1: 0.51, 2: 0.52})
    assert ok3 is False
    assert mean3 < 0.60
    assert n3 == 3


def test_catastrophic_fail():
    assert is_catastrophic_fail({"a": 0.10, "b": 0.20}) is True
    assert is_catastrophic_fail({"a": 0.10, "b": 0.35}) is False
    assert is_catastrophic_fail({"a": 0.29, "b": 0.25}, fail_rho_ceiling=0.30) is True


# ---------------------------------------------------------------------------
# recovery aggregation, weak_shift, decide_verdict
# ---------------------------------------------------------------------------


def test_recovered_requires_two_of_three_finite_t_star():
    rec = {
        0: {"t_star_abs": 25.0, "rho_rec": 0.65},
        1: {"t_star_abs": 50.0, "rho_rec": 0.70},
        2: {"t_star_abs": float("inf"), "rho_rec": 0.40},
    }
    assert recovered_check(rec) is True

    rec2 = {
        0: {"t_star_abs": 25.0, "rho_rec": 0.65},
        1: {"t_star_abs": float("inf"), "rho_rec": 0.40},
        2: {"t_star_abs": float("inf"), "rho_rec": 0.30},
    }
    assert recovered_check(rec2) is False


def test_weak_shift_mean_ceiling():
    weak, mean = weak_shift_check({0: 0.95, 1: 0.92, 2: 0.91})
    assert weak is True
    assert mean >= 0.90

    weak2, _ = weak_shift_check({0: 0.5, 1: 0.6, 2: 0.4})
    assert weak2 is False


def test_decide_verdict_and_phase1_go():
    assert decide_verdict(stationary_pass=False, recovered=True, weak_shift=False) == (
        VERDICT_FAIL,
        False,
    )
    assert decide_verdict(stationary_pass=True, recovered=True, weak_shift=False) == (
        VERDICT_PASS,
        True,
    )
    assert decide_verdict(stationary_pass=True, recovered=True, weak_shift=True) == (
        VERDICT_PASS_WEAK_SHIFT,
        False,
    )
    assert decide_verdict(stationary_pass=True, recovered=False, weak_shift=False) == (
        VERDICT_PARTIAL_PASS,
        False,
    )
    # phase1_go true ONLY when PASS
    for sp, rec, weak in [
        (False, True, False),
        (True, False, False),
        (True, True, True),
        (True, False, True),
    ]:
        v, p1 = decide_verdict(stationary_pass=sp, recovered=rec, weak_shift=weak)
        assert p1 == (v == VERDICT_PASS)


# ---------------------------------------------------------------------------
# Full aggregate_verdict scenarios
# ---------------------------------------------------------------------------


def _full_seed_rho(mean_level: float = 0.70) -> dict[str, dict[int, float]]:
    levels = {
        "activation": mean_level - 0.3,
        "hebbian_corr": mean_level - 0.2,
        "eligibility_residual_energy": mean_level - 0.1,
        "eligibility_gevb_rf": mean_level,
    }
    return {name: {0: val, 1: val, 2: val} for name, val in levels.items()}


def _lag_all_recover(t_star: float = 50.0):
    # Build series that first crosses 0.60 at t_star
    times = [0.0, 25.0, 50.0, 100.0, 250.0, 500.0]
    rhos = [0.2, 0.4, 0.5, 0.5, 0.55, 0.58]
    # set the point at t_star
    if t_star in times:
        i = times.index(t_star)
        rhos[i] = 0.62
    else:
        times = sorted(times + [t_star])
        rhos = [0.62 if t == t_star else 0.3 for t in times]
    return {
        s: SeedLagCurve(seed=s, times=list(times), rhos=list(rhos), rho_pre_switch=0.75)
        for s in (0, 1, 2)
    }


def test_aggregate_pass():
    seed_rho = _full_seed_rho(mean_level=0.70)
    out = aggregate_verdict(
        seed_rho,
        lag_curves=_lag_all_recover(50.0),
        rho_shift_by_seed={0: 0.4, 1: 0.5, 2: 0.45},
    )
    assert out.best_local == "eligibility_gevb_rf"
    assert out.stationary_pass is True
    assert out.recovered is True
    assert out.weak_shift is False
    assert out.verdict == VERDICT_PASS
    assert out.phase1_go is True
    assert out.catastrophic_fail is False


def test_aggregate_partial_pass_recovery_fails():
    seed_rho = _full_seed_rho(mean_level=0.70)
    times = [0.0, 100.0, 250.0, 500.0]
    rhos = [0.1, 0.2, 0.3, 0.4]  # never crosses 0.60
    lag = {
        s: SeedLagCurve(seed=s, times=times, rhos=rhos) for s in (0, 1, 2)
    }
    out = aggregate_verdict(
        seed_rho,
        lag_curves=lag,
        rho_shift_by_seed={0: 0.3, 1: 0.3, 2: 0.3},
    )
    assert out.stationary_pass is True
    assert out.recovered is False
    assert out.verdict == VERDICT_PARTIAL_PASS
    assert out.phase1_go is False
    # rho_rec is end-window, not max (all increasing so same, but field present)
    for s in (0, 1, 2):
        assert out.recovery_by_seed[s]["rho_rec"] == pytest.approx(0.4)
        assert out.recovery_by_seed[s]["t_star_abs"] == float("inf")


def test_aggregate_pass_weak_shift():
    seed_rho = _full_seed_rho(mean_level=0.70)
    out = aggregate_verdict(
        seed_rho,
        lag_curves=_lag_all_recover(25.0),
        rho_shift_by_seed={0: 0.95, 1: 0.93, 2: 0.91},
    )
    assert out.recovered is True
    assert out.weak_shift is True
    assert out.verdict == VERDICT_PASS_WEAK_SHIFT
    assert out.phase1_go is False  # NOT a Phase 1 green light


def test_aggregate_fail_and_catastrophic():
    seed_rho = {
        "activation": {0: 0.10, 1: 0.12, 2: 0.11},
        "hebbian_corr": {0: 0.15, 1: 0.14, 2: 0.13},
        "eligibility_residual_energy": {0: 0.18, 1: 0.17, 2: 0.16},
        "eligibility_gevb_rf": {0: 0.20, 1: 0.19, 2: 0.21},
    }
    out = aggregate_verdict(seed_rho)
    assert out.stationary_pass is False
    assert out.verdict == VERDICT_FAIL
    assert out.phase1_go is False
    assert out.catastrophic_fail is True
    assert out.best_local == "eligibility_gevb_rf"


def test_aggregate_fail_not_catastrophic_mid_rho():
    """Stationary fail but max mean rho above fail_rho_ceiling → not catastrophic."""
    seed_rho = {
        "activation": {0: 0.35, 1: 0.36, 2: 0.34},
        "eligibility_gevb_rf": {0: 0.40, 1: 0.42, 2: 0.41},
    }
    out = aggregate_verdict(seed_rho)
    assert out.verdict == VERDICT_FAIL
    assert out.catastrophic_fail is False
    assert out.phase1_go is False


def test_phase1_go_only_when_pass():
    """Explicit invariant across all enum outcomes."""
    cases = []
    # PASS
    cases.append(
        aggregate_verdict(
            _full_seed_rho(0.70),
            lag_curves=_lag_all_recover(),
            rho_shift_by_seed={0: 0.4, 1: 0.4, 2: 0.4},
        )
    )
    # PARTIAL
    lag_fail = {
        s: SeedLagCurve(seed=s, times=[0, 500], rhos=[0.1, 0.2]) for s in (0, 1, 2)
    }
    cases.append(
        aggregate_verdict(
            _full_seed_rho(0.70),
            lag_curves=lag_fail,
            rho_shift_by_seed={0: 0.4, 1: 0.4, 2: 0.4},
        )
    )
    # WEAK
    cases.append(
        aggregate_verdict(
            _full_seed_rho(0.70),
            lag_curves=_lag_all_recover(),
            rho_shift_by_seed={0: 0.95, 1: 0.95, 2: 0.95},
        )
    )
    # FAIL
    cases.append(aggregate_verdict({"activation": {0: 0.1, 1: 0.1, 2: 0.1}}))

    for out in cases:
        assert out.phase1_go == (out.verdict == VERDICT_PASS)


def test_recovery_uses_global_best_local_not_per_seed_repick():
    """best_local is frozen from stationary means; lag is only for that id."""
    seed_rho = {
        "eligibility_gevb_rf": {0: 0.80, 1: 0.80, 2: 0.80},
        "activation": {0: 0.50, 1: 0.50, 2: 0.50},
    }
    # Only provide lag for best_local path (caller responsibility)
    out = aggregate_verdict(
        seed_rho,
        lag_curves=_lag_all_recover(),
        rho_shift_by_seed={0: 0.2, 1: 0.2, 2: 0.2},
    )
    assert out.best_local == "eligibility_gevb_rf"
    assert out.verdict == VERDICT_PASS


def test_to_dict_serializable():
    out = aggregate_verdict(
        _full_seed_rho(0.70),
        lag_curves=_lag_all_recover(),
        rho_shift_by_seed={0: 0.4, 1: 0.4, 2: 0.4},
    )
    d = out.to_dict()
    assert d["verdict"] == VERDICT_PASS
    assert d["phase1_go"] is True
    assert "mean_rho_by_candidate" in d
    assert isinstance(d["thresholds"], dict)


def test_custom_thresholds():
    thr = VerdictThresholds(
        stationary_edge_spearman=0.80,
        fail_rho_ceiling=0.10,
    )
    seed_rho = _full_seed_rho(0.70)
    out = aggregate_verdict(seed_rho, thresholds=thr)
    assert out.stationary_pass is False  # 0.70 < 0.80
    assert out.verdict == VERDICT_FAIL
