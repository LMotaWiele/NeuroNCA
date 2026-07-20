"""Unit tests for mask-restricted Spearman and min-pairs handling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from neuronca.eval.correlation import (
    DEFAULT_SPEARMAN_MIN_PAIRS,
    correlate_layers,
    pearson_r,
    spearman_rho,
)


def test_spearman_matches_scipy_unmasked():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = x + 0.1 * rng.normal(size=64)
    expected, _ = stats.spearmanr(x, y)
    assert spearman_rho(x, y) == pytest.approx(float(expected))


def test_mask_restriction_boolean():
    """Spearman must use only masked pairs (GT-2 edge subsample protocol)."""
    n = 100
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    # Build a subset where x and y are tightly correlated; rest is noise
    mask = np.zeros(n, dtype=bool)
    mask[:40] = True
    x[:40] = np.linspace(-1, 1, 40)
    y[:40] = x[:40] + 0.01 * rng.normal(size=40)
    # Outside mask: anti-correlated garbage that would ruin unmasked Spearman
    x[40:] = np.linspace(1, -1, 60)
    y[40:] = -x[40:]

    rho_masked = spearman_rho(x, y, mask=mask)
    rho_unmasked = spearman_rho(x, y)
    expected_sub, _ = stats.spearmanr(x[mask], y[mask])

    assert rho_masked == pytest.approx(float(expected_sub))
    assert rho_masked > 0.9
    assert rho_unmasked < rho_masked


def test_mask_restriction_integer_indices():
    x = np.arange(50, dtype=np.float64)
    y = x.copy()
    idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    rho = spearman_rho(x, y, mask=idx)
    assert rho == pytest.approx(1.0)


def test_nan_on_too_few_pairs():
    """Fewer than spearman_min_pairs (default 16) → nan."""
    x = np.arange(10, dtype=np.float64)
    y = x * 2.0
    assert np.isnan(spearman_rho(x, y))
    assert np.isnan(spearman_rho(x, y, min_pairs=DEFAULT_SPEARMAN_MIN_PAIRS))
    # Explicit lower threshold allows the correlation
    assert spearman_rho(x, y, min_pairs=5) == pytest.approx(1.0)


def test_nan_on_masked_too_few_pairs():
    x = np.arange(100, dtype=np.float64)
    y = x.copy()
    mask = np.zeros(100, dtype=bool)
    mask[:8] = True  # only 8 pairs
    assert np.isnan(spearman_rho(x, y, mask=mask, min_pairs=16))


def test_pearson_secondary_min_pairs():
    x = np.arange(8, dtype=np.float64)
    y = 2 * x
    assert np.isnan(pearson_r(x, y, min_pairs=16))
    assert pearson_r(x, y, min_pairs=4) == pytest.approx(1.0)


def test_correlate_layers_overall_zscore_and_mask():
    # Distinct per-layer scales; exact copies so z-score concat ranks match.
    rng = np.random.default_rng(42)
    local = {
        "fc1": rng.normal(size=20),
        "fc2": rng.normal(loc=50.0, scale=3.0, size=20),
    }
    gt = {k: v.copy() for k, v in local.items()}
    # Mask only first 16 of each layer
    masks = {
        "fc1": np.arange(16),
        "fc2": np.arange(16),
    }
    res = correlate_layers(local, gt, masks=masks, layer_zscore_before_concat=True)
    assert "fc1" in res.per_layer
    assert res.per_layer["fc1"].n_pairs == 16
    assert res.per_layer["fc1"].spearman == pytest.approx(1.0, abs=1e-9)
    assert res.overall_n_pairs == 32
    assert res.overall_spearman == pytest.approx(1.0, abs=1e-9)

    # Mask restriction: correlating only masked indices equals full call with mask
    rho_direct = spearman_rho(local["fc1"], gt["fc1"], mask=masks["fc1"])
    assert rho_direct == pytest.approx(res.per_layer["fc1"].spearman)


def test_correlate_layers_skips_missing_keys():
    local = {"fc1": np.arange(20, dtype=float), "fc2": np.arange(20, dtype=float)}
    gt = {"fc1": np.arange(20, dtype=float) * 2}
    res = correlate_layers(local, gt, layer_zscore_before_concat=False)
    assert "fc2" in res.skipped_layers
    assert "fc1" in res.per_layer


def test_correlate_layers_nan_when_layer_below_min_pairs():
    local = {"fc1": np.arange(5, dtype=float)}
    gt = {"fc1": np.arange(5, dtype=float)}
    res = correlate_layers(local, gt, min_pairs=16)
    assert np.isnan(res.per_layer["fc1"].spearman)
    assert np.isnan(res.overall_spearman)
