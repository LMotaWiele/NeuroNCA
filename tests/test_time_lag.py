"""Unit tests for first-crossing lag recovery (t*_abs, rho_rec)."""

from __future__ import annotations

import numpy as np
import pytest

from neuronca.eval.schedule import lag_times, switch_snapshot_tags
from neuronca.eval.time_lag import (
    analyze_lag_curve,
    compute_rho_rec,
    first_crossing,
    relative_first_crossing,
)


def test_first_crossing_simple():
    times = [0, 25, 50, 75, 100]
    rhos = [0.1, 0.2, 0.61, 0.7, 0.8]
    assert first_crossing(times, rhos, threshold=0.60, max_t=500) == 50


def test_first_crossing_none_returns_inf():
    times = [0, 25, 50, 500]
    rhos = [0.1, 0.2, 0.3, 0.59]
    assert first_crossing(times, rhos, threshold=0.60, max_t=500) == float("inf")


def test_non_monotonic_first_crossing_not_max():
    """ρ goes 0.4 → 0.65 → 0.55 → 0.70: t* is second point; rho_rec=0.65 not 0.70."""
    times = [0, 25, 50, 75]
    rhos = [0.4, 0.65, 0.55, 0.70]
    t_star, rho_rec = compute_rho_rec(times, rhos, threshold=0.60, max_t=500)
    assert t_star == 25
    assert rho_rec == pytest.approx(0.65)
    assert rho_rec != pytest.approx(0.70)

    result = analyze_lag_curve(times, rhos, threshold=0.60, max_t=500)
    assert result.t_star_abs == 25
    assert result.rho_rec == pytest.approx(0.65)
    assert result.rho_max_in_window == pytest.approx(0.70)
    assert result.recovered_abs is True


def test_rho_rec_at_window_end_when_no_crossing():
    times = [0, 100, 250, 500]
    rhos = [0.1, 0.2, 0.3, 0.45]
    t_star, rho_rec = compute_rho_rec(times, rhos, threshold=0.60, max_t=500)
    assert t_star == float("inf")
    assert rho_rec == pytest.approx(0.45)
    result = analyze_lag_curve(times, rhos)
    assert result.recovered_abs is False
    assert result.rho_rec == pytest.approx(0.45)
    assert result.rho_max_in_window == pytest.approx(0.45)


def test_relative_recovery_secondary():
    times = [0, 25, 50, 100]
    # abs threshold 0.60 crossed at t=25, but relative 0.9*0.80=0.72 needs t=100
    rhos = [0.50, 0.62, 0.65, 0.75]
    rho_pre = 0.80
    t_rel = relative_first_crossing(
        times, rhos, rho_pre_switch=rho_pre, abs_threshold=0.60, relative_factor=0.9
    )
    assert t_rel == 100
    result = analyze_lag_curve(times, rhos, rho_pre_switch=rho_pre)
    assert result.t_star_abs == 25
    assert result.t_star_rel == 100
    assert result.recovered_abs is True
    assert result.recovered_rel is True


def test_lag_schedule_times():
    times = lag_times(500, 25)
    assert times[0] == 0
    assert times[-1] == 500
    assert 25 in times
    assert all(t % 25 == 0 for t in times)


def test_switch_tags():
    tags = [s.tag for s in switch_snapshot_tags()]
    assert tags == ["late_a", "switch_b0"]
