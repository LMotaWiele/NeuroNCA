"""Smoke / unit tests for the stationary experiment runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neuronca.experiments.stationary import _apply_smoke_overrides, run_stationary
from neuronca.io.config import SmokeConfigError, load_config_named


def _tiny_smoke_cfg() -> dict:
    """Shared smoke-capped config for multi-seed / smoke tests."""
    cfg = load_config_named("stationary", smoke=True)
    train = dict(cfg.get("train") or {})
    train["max_steps_per_epoch"] = 2
    train["batch_size"] = 32
    cfg["train"] = train
    data = dict(cfg.get("data") or {})
    data["probe_size"] = 64
    data["num_workers"] = 0
    data["root"] = "data/mnist"
    cfg["data"] = data
    gt = dict(cfg.get("gt") or {})
    gt["probe_size"] = 64
    gt["edge_ablation_sample"] = 0
    gt["fisher"] = False
    cfg["gt"] = gt
    eval_cfg = dict(cfg.get("eval") or {})
    eval_cfg["coarse_gt_measures"] = ["grad_abs"]
    eval_cfg["spearman_min_pairs"] = 8
    cfg["eval"] = eval_cfg
    # Multi-seed list from Phase 0 defaults (shared ledger identity).
    exp = dict(cfg.get("experiment") or {})
    exp["seeds"] = [0, 1, 2]
    cfg["experiment"] = exp
    return cfg


@pytest.fixture(scope="module")
def smoke_config() -> dict:
    """Resolved stationary config with smoke overrides + tiny step budget."""
    cfg = _tiny_smoke_cfg()
    train = dict(cfg.get("train") or {})
    train["max_steps_per_epoch"] = 4
    cfg["train"] = train
    return cfg


def test_run_stationary_smoke_artifacts(tmp_path: Path, smoke_config: dict):
    """Smoke path: short real-MNIST run writes correlations.csv + summary.json."""
    results_root = tmp_path / "phase0"
    summary = run_stationary(
        smoke_config,
        results_root,
        seed=0,
        smoke=True,
    )

    run_dir = results_root / "stationary" / "seed_0"
    corr_path = run_dir / "correlations.csv"
    summary_path = run_dir / "summary.json"

    assert corr_path.is_file(), f"missing {corr_path}"
    assert summary_path.is_file(), f"missing {summary_path}"
    assert summary["condition"] == "stationary"
    assert summary["seed"] == 0
    assert summary["smoke"] is True
    assert summary["global_step"] > 0
    assert "experiment_id" in summary and summary["experiment_id"]

    # Ledger frozen under results root
    assert (results_root / "meta" / "experiment_id.txt").is_file()
    assert (results_root / "meta" / "thresholds.json").is_file()

    with corr_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "correlations.csv should have data rows"

    # Expected columns present
    required = {
        "seed",
        "condition",
        "snapshot_tag",
        "local_signal",
        "gt_measure",
        "layer",
        "granularity",
        "spearman",
    }
    assert required.issubset(set(rows[0].keys()))

    tags = {r["snapshot_tag"] for r in rows}
    # Smoke epochs=1 → early/mid/late all land on epoch 1
    assert "late" in tags
    assert "early" in tags

    signals = {r["local_signal"] for r in rows}
    for name in (
        "activation",
        "hebbian_corr",
        "eligibility_residual_energy",
        "eligibility_gevb_rf",
    ):
        assert name in signals

    assert any(r["gt_measure"] == "grad_abs" for r in rows)
    assert any(r["layer"] == "overall" for r in rows)
    assert any(r["granularity"] == "edge" for r in rows)

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 0
    assert loaded["artifacts"]["correlations_csv"]
    assert Path(loaded["artifacts"]["correlations_csv"]).is_file()
    assert "late_edge_grad_abs" in loaded


def test_run_stationary_smoke_flag_merges_overrides(tmp_path: Path):
    """smoke=True applies base smoke block even if config is full-budget."""
    cfg = load_config_named("stationary", smoke=False)
    # Force a full-looking budget; smoke flag should shrink epochs.
    train = dict(cfg.get("train") or {})
    train["epochs"] = 20
    train["max_steps_per_epoch"] = 2
    train["batch_size"] = 32
    cfg["train"] = train
    data = dict(cfg.get("data") or {})
    data["num_workers"] = 0
    data["root"] = "data/mnist"
    cfg["data"] = data
    eval_cfg = dict(cfg.get("eval") or {})
    eval_cfg["coarse_gt_measures"] = ["grad_abs"]
    cfg["eval"] = eval_cfg
    gt = dict(cfg.get("gt") or {})
    gt["edge_ablation_sample"] = 0
    cfg["gt"] = gt

    summary = run_stationary(cfg, tmp_path / "phase0", seed=0, smoke=True)
    # Base smoke sets epochs: 1
    assert summary["epochs"] == 1
    assert summary["global_step"] <= 4  # max_steps kept from overrides merge order
    assert (tmp_path / "phase0" / "stationary" / "seed_0" / "correlations.csv").is_file()


def test_multi_seed_shares_ledger_without_hash_clash(tmp_path: Path):
    """Two seeds on the same results_root share one meta/ (no LedgerConflictError)."""
    cfg = _tiny_smoke_cfg()
    results_root = tmp_path / "phase0"

    s0 = run_stationary(cfg, results_root, seed=0, smoke=True)
    s1 = run_stationary(cfg, results_root, seed=1, smoke=True)

    assert s0["experiment_id"] == s1["experiment_id"]
    assert s0["content_sha256"] == s1["content_sha256"]
    assert s0["seed"] == 0
    assert s1["seed"] == 1

    exp_id = (results_root / "meta" / "experiment_id.txt").read_text(encoding="utf-8").strip()
    assert exp_id == s0["experiment_id"]

    assert (results_root / "stationary" / "seed_0" / "correlations.csv").is_file()
    assert (results_root / "stationary" / "seed_1" / "correlations.csv").is_file()
    assert (results_root / "stationary" / "seed_0" / "summary.json").is_file()
    assert (results_root / "stationary" / "seed_1" / "summary.json").is_file()

    # Per-seed GEVB matrices may both exist under shared meta/
    assert (results_root / "meta" / "gevb_B_seed0.pt").is_file()
    assert (results_root / "meta" / "gevb_B_seed1.pt").is_file()


def test_apply_smoke_overrides_raises_when_defaults_missing():
    """smoke=True must hard-fail (SmokeConfigError) if base defaults are gone."""
    cfg = {"train": {"epochs": 20}}
    with patch(
        "neuronca.experiments.stationary.load_yaml",
        side_effect=FileNotFoundError("missing"),
    ):
        with pytest.raises(SmokeConfigError):
            _apply_smoke_overrides(cfg)


def test_apply_smoke_overrides_raises_when_smoke_block_empty():
    """Empty / null smoke mapping must not silently keep full budget."""
    cfg = {"train": {"epochs": 20}}
    with patch(
        "neuronca.experiments.stationary.load_yaml",
        return_value={"seed": 0, "smoke": None},
    ):
        with pytest.raises(SmokeConfigError):
            _apply_smoke_overrides(cfg)

    with patch(
        "neuronca.experiments.stationary.load_yaml",
        return_value={"seed": 0, "smoke": {}},
    ):
        with pytest.raises(SmokeConfigError):
            _apply_smoke_overrides(cfg)
