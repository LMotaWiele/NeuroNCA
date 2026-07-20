"""Integration / unit tests for task-shift ordered A→B switch and lag."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuronca.data.mnist import build_mnist_loaders
from neuronca.data.tasks import (
    TaskContext,
    make_identity_transform,
    make_label_permutation_transform,
)
from neuronca.experiments.task_shift import (
    _gt_shift_disagreement,
    _lag_target_b_steps,
    run_task_shift,
)
from neuronca.importance.gradient_saliency import GradientSaliency
from neuronca.io.config import load_config_named
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.training.callbacks import InstrumentedTrainer, capture_signal_traces, traces_bitwise_equal
from neuronca.training.optim import build_optimizer
from neuronca.broadcast.error import BroadcastErrorEncoder
from neuronca.signals import ActivationMagnitude


def _tiny_task_shift_cfg() -> dict:
    """CI-capped task-shift config (few steps, short lag).

    Already reduced — pass ``smoke=False`` to :func:`run_task_shift` so the base
    smoke block does not overwrite lag_window / probe sizes.
    """
    cfg = load_config_named("task_shift", smoke=False)
    train = dict(cfg.get("train") or {})
    train["max_steps_per_epoch"] = 3
    train["batch_size"] = 32
    train["epochs_a"] = 1
    train["epochs_b"] = 1
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
    # Lag times [0,1,2,3] with 3 B steps → t=0@step1, t=1@step1, t=2@step2, t=3@step3
    eval_cfg["lag_window_steps"] = 3
    eval_cfg["lag_stride"] = 1
    eval_cfg["lag_gt_measures"] = ["grad_abs"]
    eval_cfg["coarse_gt_measures"] = ["grad_abs"]
    eval_cfg["spearman_min_pairs"] = 8
    cfg["eval"] = eval_cfg
    exp = dict(cfg.get("experiment") or {})
    exp["seeds"] = [0]
    exp["conditions"] = ["task_shift"]
    cfg["experiment"] = exp
    return cfg


@pytest.fixture(scope="module")
def smoke_config() -> dict:
    return _tiny_task_shift_cfg()


def test_lag_target_b_steps_mapping():
    """t=0 → after first B step; t>0 → t optimizer steps into B."""
    assert _lag_target_b_steps(0) == 1
    assert _lag_target_b_steps(25) == 25
    assert _lag_target_b_steps(500) == 500


def test_switch_sequence_probe_labels_differ_frozen_weights(tmp_path: Path):
    """B1 integration: frozen weights, probe labels under B differ from A.

    After late_a → set_transform(B) → switch_b0: labels are π(raw); weights
    unchanged; local traces bitwise-identical across both GT calls.
    """
    set_seed(0)
    transform_a = make_identity_transform()
    transform_b = make_label_permutation_transform(num_classes=10, seed=123)
    task_ctx = TaskContext(transform_a)
    loaders = build_mnist_loaders(
        task_ctx,
        root="data/mnist",
        probe_size=64,
        seed=0,
        batch_size=32,
        num_workers=0,
        download=True,
    )
    probe_loader = loaders["probe_loader"]
    probe_ds = loaders["probe_dataset"]

    # Collect A labels
    labels_a: list[int] = []
    for i in range(min(32, len(probe_ds))):
        _x, y = probe_ds[i]
        labels_a.append(int(y))

    model = MLPReasoningNet(input_dim=784, hidden=(32, 16, 8), num_classes=10)
    optimizer = build_optimizer(model, name="sgd", lr=0.01)
    signals = [ActivationMagnitude(decay=0.99)]
    trainer = InstrumentedTrainer(
        model,
        optimizer,
        signals,
        encoder=BroadcastErrorEncoder(),
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
    )
    # A few train steps on A so signals are non-empty
    trainer.train_steps(loaders["train_loader"], max_steps=2)

    gt = [GradientSaliency()]
    weights_pre = {n: p.detach().clone() for n, p in model.named_parameters()}
    traces_pre_gt = [capture_signal_traces(s) for s in signals]

    # late_a GT under A
    snap_a = trainer.snapshot(gt, probe_loader, check_isolation=True)
    assert "grad_abs" in snap_a.gt_results

    traces_after_a = [capture_signal_traces(s) for s in signals]
    for before, after in zip(traces_pre_gt, traces_after_a):
        assert traces_bitwise_equal(before, after)

    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), weights_pre[n])

    # Atomic flip
    task_ctx.set_transform(transform_b)
    labels_b: list[int] = []
    for i in range(min(32, len(probe_ds))):
        _x, y = probe_ds[i]
        labels_b.append(int(y))

    # B1: at least some labels must change under non-identity π
    pi = transform_b.permutation
    assert not torch.equal(pi, torch.arange(10)), "test requires non-identity π"
    assert labels_a != labels_b, "probe labels must differ A vs B under B1"

    # switch_b0 GT under B, same weights
    traces_before_b0 = [capture_signal_traces(s) for s in signals]
    snap_b = trainer.snapshot(gt, probe_loader, check_isolation=True)
    assert "grad_abs" in snap_b.gt_results

    traces_after_b0 = [capture_signal_traces(s) for s in signals]
    for before, after in zip(traces_before_b0, traces_after_b0):
        assert traces_bitwise_equal(before, after)

    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), weights_pre[n]), "weights must stay frozen"

    # gt_shift_disagreement is a defined Spearman (may be high or low)
    rho = _gt_shift_disagreement(
        snap_a.gt_results["grad_abs"],
        snap_b.gt_results["grad_abs"],
        layer_zscore=True,
        min_pairs=8,
    )
    assert rho is not None
    # finite or nan both acceptable for tiny probe; just ensure callable path works
    assert isinstance(rho, float)


def test_run_task_shift_smoke_artifacts(tmp_path: Path, smoke_config: dict):
    """Smoke path: ordered switch writes correlations, time_lag, summary."""
    results_root = tmp_path / "phase0"
    # Config is already CI-capped; smoke=False avoids re-merge overwriting lag.
    summary = run_task_shift(
        smoke_config,
        results_root,
        seed=0,
        smoke=False,
    )

    run_dir = results_root / "task_shift" / "seed_0"
    corr_path = run_dir / "correlations.csv"
    lag_path = run_dir / "time_lag.csv"
    summary_path = run_dir / "summary.json"

    assert corr_path.is_file(), f"missing {corr_path}"
    assert lag_path.is_file(), f"missing {lag_path}"
    assert summary_path.is_file(), f"missing {summary_path}"
    assert summary["condition"] == "task_shift"
    assert summary["seed"] == 0
    assert summary["global_step"] > 0
    assert summary["b_steps"] > 0
    assert summary["b_steps"] == 3  # max_steps_per_epoch=3, epochs_b=1
    assert "experiment_id" in summary and summary["experiment_id"]
    assert "gt_shift_disagreement" in summary
    assert summary["task_version_final"] >= 1
    assert 0 in summary["lag_times_observed"]

    # Ledger frozen
    assert (results_root / "meta" / "experiment_id.txt").is_file()

    with corr_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    tags = {r["snapshot_tag"] for r in rows}
    assert "late_a" in tags, tags
    assert "switch_b0" in tags, tags
    assert "lag_t0" in tags, tags
    # mid_b / late_b land on epoch 1 when epochs_b=1
    assert "late_b" in tags or "mid_b" in tags, tags

    conditions = {r["condition"] for r in rows}
    assert conditions == {"task_shift"}

    with lag_path.open(encoding="utf-8", newline="") as f:
        lag_rows = list(csv.DictReader(f))
    assert len(lag_rows) > 0
    lag_cols = set(lag_rows[0].keys())
    for col in (
        "steps_after_switch",
        "seed",
        "local_signal",
        "gt_measure",
        "spearman",
        "rho_pre_switch",
        "t_star_abs",
        "rho_rec",
    ):
        assert col in lag_cols, f"missing {col} in time_lag.csv"

    # t=0 present
    steps = {int(float(r["steps_after_switch"])) for r in lag_rows}
    assert 0 in steps

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["switch_sequence"][0] == "late_a"
    assert "set_transform_B" in loaded["switch_sequence"]
    assert loaded["artifacts"]["time_lag_csv"]
    assert Path(loaded["artifacts"]["time_lag_csv"]).is_file()
    assert "lag_recovery" in loaded


def test_run_task_shift_smoke_flag_merges_overrides(tmp_path: Path):
    """smoke=True applies base smoke block (epochs_a/b=1, short lag)."""
    cfg = load_config_named("task_shift", smoke=False)
    train = dict(cfg.get("train") or {})
    train["epochs_a"] = 15
    train["epochs_b"] = 15
    train["max_steps_per_epoch"] = 2
    train["batch_size"] = 32
    cfg["train"] = train
    data = dict(cfg.get("data") or {})
    data["num_workers"] = 0
    data["root"] = "data/mnist"
    cfg["data"] = data
    eval_cfg = dict(cfg.get("eval") or {})
    eval_cfg["coarse_gt_measures"] = ["grad_abs"]
    eval_cfg["lag_gt_measures"] = ["grad_abs"]
    cfg["eval"] = eval_cfg
    gt = dict(cfg.get("gt") or {})
    gt["edge_ablation_sample"] = 0
    cfg["gt"] = gt

    summary = run_task_shift(cfg, tmp_path / "phase0", seed=0, smoke=True)
    assert summary["epochs_a"] == 1
    assert summary["epochs_b"] == 1
    assert summary["lag_window_steps"] == 50  # base smoke
    assert (tmp_path / "phase0" / "task_shift" / "seed_0" / "correlations.csv").is_file()
    assert (tmp_path / "phase0" / "task_shift" / "seed_0" / "time_lag.csv").is_file()


def test_switch_sequence_order_in_summary(tmp_path: Path, smoke_config: dict):
    """Snapshot tags in summary respect ordered late_a → switch_b0 → lag_t0."""
    summary = run_task_shift(
        smoke_config, tmp_path / "phase0", seed=0, smoke=False
    )
    tags = [s["tag"] for s in summary["snapshots"]]
    assert "late_a" in tags
    assert "switch_b0" in tags
    i_late = tags.index("late_a")
    i_b0 = tags.index("switch_b0")
    assert i_late < i_b0
    if "lag_t0" in tags:
        assert i_b0 < tags.index("lag_t0")
    # No optimizer between late_a and switch_b0: same global step
    late_rec = next(s for s in summary["snapshots"] if s["tag"] == "late_a")
    b0_rec = next(s for s in summary["snapshots"] if s["tag"] == "switch_b0")
    assert late_rec["step"] == b0_rec["step"]
