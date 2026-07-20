"""Tests for YAML config load, deep-merge, smoke flag, and device resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuronca.io.config import (
    SmokeConfigError,
    canonical_yaml,
    deep_merge,
    extract_thresholds,
    load_config,
    load_config_named,
    load_yaml,
    resolve_device,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def test_deep_merge_nested_and_replace_lists():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]}
    over = {"b": {"y": 9, "z": 3}, "c": [9], "d": 4}
    out = deep_merge(base, over)
    assert out["a"] == 1
    assert out["b"] == {"x": 1, "y": 9, "z": 3}
    assert out["c"] == [9]
    assert out["d"] == 4
    # base not mutated
    assert base["b"]["y"] == 2


def test_load_default_config_resolves_device():
    cfg = load_config(defaults_path=CONFIGS / "default.yaml")
    assert cfg["device"] in ("cpu", "cuda")
    assert cfg["device"] != "auto"
    assert "smoke" not in cfg
    assert cfg["train"]["epochs"] == 20
    assert cfg["experiment"]["seeds"] == [0, 1, 2]
    assert cfg["model"]["hidden"] == [256, 128, 64]


def test_load_stationary_override():
    cfg = load_config(
        CONFIGS / "stationary.yaml",
        defaults_path=CONFIGS / "default.yaml",
    )
    assert cfg["experiment"]["conditions"] == ["stationary"]
    assert cfg["tasks"]["shift_type"] is None
    assert cfg["train"]["epochs"] == 20
    # defaults preserved
    assert cfg["broadcast"]["per_sample"] is True


def test_load_task_shift_override():
    cfg = load_config_named("task_shift", configs_dir=CONFIGS)
    assert cfg["experiment"]["conditions"] == ["task_shift"]
    assert cfg["tasks"]["shift_type"] == "label_permutation"
    assert cfg["train"]["epochs_a"] == 15


def test_smoke_merges_and_strips_block():
    cfg = load_config(
        CONFIGS / "stationary.yaml",
        defaults_path=CONFIGS / "default.yaml",
        smoke=True,
    )
    assert "smoke" not in cfg
    assert cfg["train"]["epochs"] == 1
    assert cfg["train"]["epochs_a"] == 1
    assert cfg["train"]["batch_size"] == 64
    assert cfg["data"]["probe_size"] == 128
    assert cfg["gt"]["probe_size"] == 128
    assert cfg["gt"]["edge_ablation_sample"] == 0
    assert cfg["eval"]["lag_window_steps"] == 50
    assert cfg["experiment"]["seeds"] == [0]
    # stationary condition preserved through smoke
    assert cfg["experiment"]["conditions"] == ["stationary"]


def test_in_memory_overrides_last():
    cfg = load_config(
        defaults_path=CONFIGS / "default.yaml",
        overrides={"train": {"lr": 0.5}, "seed": 99},
    )
    assert cfg["train"]["lr"] == 0.5
    assert cfg["seed"] == 99
    assert cfg["train"]["momentum"] == 0.9


def test_resolve_device_auto_and_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    auto = resolve_device("auto")
    assert auto in ("cpu", "cuda")
    assert resolve_device(None) in ("cpu", "cuda")


def test_extract_thresholds_includes_weak_shift_ceiling():
    cfg = load_config(defaults_path=CONFIGS / "default.yaml")
    thr = extract_thresholds(cfg)
    assert thr["stationary_edge_spearman"] == 0.60
    assert thr["fail_rho_ceiling"] == 0.30
    assert thr["gt_shift_disagreement_ceiling"] == 0.90


def test_canonical_yaml_stable_for_hashing():
    a = {"b": 2, "a": {"z": 1, "y": 0}}
    b = {"a": {"y": 0, "z": 1}, "b": 2}
    assert canonical_yaml(a) == canonical_yaml(b)


def test_load_yaml_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml(CONFIGS / "does_not_exist.yaml")


def test_load_yaml_non_mapping_raises(tmp_path: Path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_yaml(p)


def test_smoke_null_override_still_applies_base_smoke(tmp_path: Path):
    """Overrides that set smoke: null must not disable CI smoke."""
    defaults = tmp_path / "default.yaml"
    defaults.write_text(
        "train:\n  epochs: 20\n"
        "experiment:\n  seeds: [0, 1, 2]\n"
        "smoke:\n  train:\n    epochs: 1\n  experiment:\n    seeds: [0]\n",
        encoding="utf-8",
    )
    override = tmp_path / "null_smoke.yaml"
    override.write_text("smoke: null\n", encoding="utf-8")
    cfg = load_config(override, defaults_path=defaults, smoke=True)
    assert cfg["train"]["epochs"] == 1
    assert cfg["experiment"]["seeds"] == [0]
    assert "smoke" not in cfg


def test_smoke_true_without_base_smoke_raises(tmp_path: Path):
    """Custom base without smoke: + smoke=True must error, not pull project defaults."""
    defaults = tmp_path / "bare.yaml"
    defaults.write_text("train:\n  epochs: 20\n", encoding="utf-8")
    with pytest.raises(SmokeConfigError, match="no usable smoke"):
        load_config(defaults_path=defaults, smoke=True)


def test_smoke_false_leaves_full_budget():
    cfg = load_config(defaults_path=CONFIGS / "default.yaml", smoke=False)
    assert cfg["train"]["epochs"] == 20
    assert cfg["experiment"]["seeds"] == [0, 1, 2]
