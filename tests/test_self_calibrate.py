"""Unit tests for self-calibrating constant trackers."""

from __future__ import annotations

import pytest
import torch

from neuronca.plasticity.self_calibrate import (
    LayerRateTracker,
    PruneCutoffTracker,
    SelfCalibrateFlags,
    SelfCalibrateState,
    StraddleRefTracker,
    detect_rate_runaway,
    detect_stat_oscillation,
)


def test_layer_rate_tracker_requires_warm_start() -> None:
    t = LayerRateTracker(mu=0.9)
    with pytest.raises(RuntimeError, match="warm_start"):
        t.update(
            {"h1": torch.ones(4), "h2": torch.ones(2), "h3": torch.ones(2)},
            {
                "h1": torch.ones(4, dtype=torch.bool),
                "h2": torch.ones(2, dtype=torch.bool),
                "h3": torch.ones(2, dtype=torch.bool),
            },
        )


def test_layer_rate_tracker_rejects_zero_warm_start() -> None:
    t = LayerRateTracker(mu=0.9)
    with pytest.raises(ValueError, match="> 0"):
        t.warm_start({"h1": 0.0, "h2": 1.0, "h3": 1.0})


def test_layer_rate_tracker_ema_moves_toward_data() -> None:
    t = LayerRateTracker(mu=0.5)
    t.warm_start({"h1": 1.0, "h2": 1.0, "h3": 1.0})
    s = {
        "h1": torch.full((4,), 2.0),
        "h2": torch.full((2,), 2.0),
        "h3": torch.full((2,), 2.0),
    }
    a = {h: torch.ones(s[h].numel(), dtype=torch.bool) for h in s}
    r1 = t.update(s, a)
    assert abs(r1["h1"] - 1.5) < 1e-5
    r2 = t.update(s, a)
    assert r2["h1"] > r1["h1"]
    assert r2["h1"] < 2.0


def test_hybrid_period_skips_updates() -> None:
    t = LayerRateTracker(mu=0.5, hybrid_period=3)
    t.warm_start({"h1": 1.0, "h2": 1.0, "h3": 1.0})
    s = {
        "h1": torch.full((4,), 3.0),
        "h2": torch.full((2,), 3.0),
        "h3": torch.full((2,), 3.0),
    }
    a = {h: torch.ones(s[h].numel(), dtype=torch.bool) for h in s}
    r0 = t.snapshot()
    t.update(s, a)  # step 1 — skip
    t.update(s, a)  # step 2 — skip
    assert t.snapshot() == r0
    r3 = t.update(s, a)  # step 3 — update
    assert r3["h1"] != r0["h1"]


def test_prune_cutoff_quantile_below_mean() -> None:
    t = PruneCutoffTracker(mu=0.0, target_quantile=0.1587)  # mu=0 → pure empirical
    t.warm_start(-1.0)
    # Standard normal sample approx: quantile ~ -1
    z = {"h1": torch.randn(500), "h2": torch.randn(200), "h3": torch.randn(100)}
    a = {h: torch.ones(z[h].numel(), dtype=torch.bool) for h in z}
    cut = t.update(z, a)
    assert cut < 0.0
    assert -2.5 < cut < 0.0


def test_straddle_fallback_after_birth() -> None:
    t = StraddleRefTracker(mu=0.5)
    t.warm_start()
    t.update_from_birth("h1", mean_gap=-5.0, mean_nongap=-11.0)
    b = t.fallback_straddle_B("h1")
    assert b is not None
    assert abs(b - 8.0) < 1e-5


def test_self_cal_state_flags() -> None:
    st = SelfCalibrateState.from_config(
        {"r_star": True, "prune_cutoff": False, "straddle": False},
        mu_r_star=0.99,
    )
    assert st.flags.r_star
    assert not st.flags.prune_cutoff
    assert st.rates is not None
    assert st.prune is None
    assert st.needs_warm_start()


def test_detect_runaway_and_oscillation() -> None:
    hist = [{"h1": 1.0 + 0.1 * i, "h2": 1.0, "h3": 1.0} for i in range(12)]
    run = detect_rate_runaway(hist, monotonic_epochs=8, rel_drift=0.3)
    assert run["runaway"] is True
    osc = detect_stat_oscillation(
        [1.0, 1.2, 1.0, 1.2, 1.0, 1.2],
        rel_step_change=0.1,
        consecutive_steps=3,
        last_n=5,
        range_over_mean=0.15,
    )
    assert osc["oscillating"] is True


def test_flags_from_mapping() -> None:
    f = SelfCalibrateFlags.from_mapping({"r_star": 1, "prune_cutoff": 0})
    assert f.r_star and not f.prune_cutoff and not f.straddle
    assert f.any_enabled()
