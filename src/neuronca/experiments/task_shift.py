"""Task-shift runner: A→B switch snapshots, then lag GT-1 after the first B step.

Drain loaders before ``set_transform``; ``switch_b0`` is same-weights, no
optimizer step. Stationary lives in ``stationary.py``.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from neuronca.broadcast.error import BroadcastErrorEncoder
from torch.utils.data import DataLoader, Subset

from neuronca.data.mnist import (
    TaskAwareDataset,
    build_mnist_datasets,
    build_mnist_loaders,
    _mnist_targets,
)
from neuronca.data.tasks import (
    TaskContext,
    make_digit_split_transform,
    make_identity_transform,
    make_task_transform,
)
from neuronca.eval.correlation import correlate_layers
from neuronca.eval.schedule import lag_times, task_shift_schedule
from neuronca.eval.time_lag import analyze_lag_curve
from neuronca.importance.base import ImportanceResult
from neuronca.io.artifacts import (
    correlation_rows_from_result,
    ensure_results_tree,
    seed_run_dir,
    write_correlations_csv,
)
from neuronca.io.ledger import LedgerMeta, freeze_ledger, meta_dir_for
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.signals.base import ScoreBundle
from neuronca.training.callbacks import InstrumentedTrainer, SnapshotResult
from neuronca.training.loop import _unpack_batch, resolve_device
from neuronca.training.optim import build_optimizer

# Reuse stationary scaffolding (smoke, ledger, signals, GT, masks).
from neuronca.experiments.stationary import (
    _apply_smoke_overrides,
    _build_gt_measures,
    _build_signals,
    _config_for_ledger,
    _eval_isolation_guard,
    _finite_or_none,
    _gt_measure_names,
    _mask_map_to_numpy,
    _maybe_record_gevb_alignment,
    _prepare_edge_mask,
    _tensor_map_to_numpy,
    _write_csv,
)
from neuronca.signals.gevb_alignment import GEVB_ALIGNMENT_CSV_COLUMNS


CONDITION = "task_shift"


def _filter_indices_by_digit(
    indices: Tensor | Sequence[int],
    targets: Tensor,
    low: int,
    high: int,
) -> list[int]:
    """Keep base indices whose raw MNIST labels lie in ``[low, high]``."""
    out: list[int] = []
    for i in indices:
        ii = int(i)
        y = int(targets[ii].item())
        if low <= y <= high:
            out.append(ii)
    return out


def _loader_from_indices(
    train_full: Any,
    indices: Sequence[int],
    task_context: TaskContext,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    base = Subset(train_full, list(indices))
    ds = TaskAwareDataset(base, task_context)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _build_digit_split_phase_loaders(
    task_ctx: TaskContext,
    *,
    root: str | Path,
    probe_size: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Build Task A (digits 0–4) and Task B (digits 5–9) loaders for B2.

    Design B2: restricted 5-way split — A trains on labels 0–4; B on raw 5–9
    remapped to 0–4 via :class:`~neuronca.data.tasks.DigitSplitTransform`.
    Shared train/probe index draw first, then filtered by digit range.
    """
    # Temporary identity context only to build the stratified split once.
    tmp_ctx = TaskContext(make_identity_transform())
    bundles = build_mnist_datasets(
        tmp_ctx,
        root=root,
        probe_size=probe_size,
        seed=seed,
        download=True,
    )
    train_full = bundles["train_full"]
    targets = _mnist_targets(train_full)
    train_idx = bundles["train_indices"]
    probe_idx = bundles["probe_indices"]

    a_train = _filter_indices_by_digit(train_idx, targets, 0, 4)
    a_probe = _filter_indices_by_digit(probe_idx, targets, 0, 4)
    b_train = _filter_indices_by_digit(train_idx, targets, 5, 9)
    b_probe = _filter_indices_by_digit(probe_idx, targets, 5, 9)
    if not a_train or not a_probe or not b_train or not b_probe:
        raise RuntimeError(
            f"digit_split filter empty: a_train={len(a_train)} a_probe={len(a_probe)} "
            f"b_train={len(b_train)} b_probe={len(b_probe)}"
        )

    pin_memory = torch.cuda.is_available()
    # Phase A: labels already in 0–4; Identity is fine (DigitSplit(0,4) also ok).
    task_ctx.set_transform(make_identity_transform())
    train_a = _loader_from_indices(
        train_full,
        a_train,
        task_ctx,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    probe_a = _loader_from_indices(
        train_full,
        a_probe,
        task_ctx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # probe GT always single-process
        pin_memory=pin_memory,
    )
    # Phase B loaders share the same TaskContext object; transform flipped at switch.
    train_b = _loader_from_indices(
        train_full,
        b_train,
        task_ctx,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    probe_b = _loader_from_indices(
        train_full,
        b_probe,
        task_ctx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return {
        "train_loader_a": train_a,
        "probe_loader_a": probe_a,
        "train_loader_b": train_b,
        "probe_loader_b": probe_b,
        "n_train_a": len(a_train),
        "n_probe_a": len(a_probe),
        "n_train_b": len(b_train),
        "n_probe_b": len(b_probe),
        "num_classes": 5,
        "transform_b": make_digit_split_transform(low=5, high=9),
    }


TIME_LAG_CSV_COLUMNS: tuple[str, ...] = (
    "steps_after_switch",
    "seed",
    "local_signal",
    "gt_measure",
    "spearman",
    "recovered_abs",
    "recovered_rel",
    "t_star_abs",
    "t_star_rel",
    "t_star_sustained",
    "recovered_sustained",
    "rho_pre_switch",
    "rho_rec",
    "rho_max_in_window",
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _correlation_rows_for_snapshot(
    snap: SnapshotResult,
    *,
    seed: int,
    snapshot_tag: str,
    layer_zscore: bool,
    min_pairs: int,
    edge_mask: dict[str, Tensor] | None,
    condition: str = CONDITION,
) -> list[dict[str, Any]]:
    """Correlate each local × each GT at edge and neuron granularity."""
    rows: list[dict[str, Any]] = []
    for sig_name, bundle in snap.local_scores.items():
        assert isinstance(bundle, ScoreBundle)
        local_edge = _tensor_map_to_numpy(bundle.edge)
        local_neuron = _tensor_map_to_numpy(bundle.neuron)

        for gt_name, gt_res in snap.gt_results.items():
            assert isinstance(gt_res, ImportanceResult)
            gt_edge = _tensor_map_to_numpy(gt_res.edge)
            gt_neuron = _tensor_map_to_numpy(gt_res.neuron)

            masks = None
            mask_restricted = False
            if gt_res.edge_index:
                masks = _mask_map_to_numpy(gt_res.edge_index)
                mask_restricted = True
            elif edge_mask is not None and gt_name in ("ablation_relu", "ablation"):
                masks = _mask_map_to_numpy(edge_mask)
                mask_restricted = True

            if gt_edge:
                edge_corr = correlate_layers(
                    local_edge,
                    gt_edge,
                    masks=masks,
                    layer_zscore_before_concat=layer_zscore,
                    min_pairs=min_pairs,
                )
                rows.extend(
                    correlation_rows_from_result(
                        per_layer=edge_corr.per_layer,
                        overall_spearman=edge_corr.overall_spearman,
                        overall_pearson=edge_corr.overall_pearson,
                        overall_n_pairs=edge_corr.overall_n_pairs,
                        seed=seed,
                        condition=condition,
                        snapshot_tag=snapshot_tag,
                        step=snap.step,
                        local_signal=sig_name,
                        gt_measure=gt_name,
                        granularity="edge",
                        mask_restricted=mask_restricted,
                    )
                )

            if gt_neuron:
                neuron_corr = correlate_layers(
                    local_neuron,
                    gt_neuron,
                    masks=None,
                    layer_zscore_before_concat=layer_zscore,
                    min_pairs=min_pairs,
                )
                rows.extend(
                    correlation_rows_from_result(
                        per_layer=neuron_corr.per_layer,
                        overall_spearman=neuron_corr.overall_spearman,
                        overall_pearson=neuron_corr.overall_pearson,
                        overall_n_pairs=neuron_corr.overall_n_pairs,
                        seed=seed,
                        condition=condition,
                        snapshot_tag=snapshot_tag,
                        step=snap.step,
                        local_signal=sig_name,
                        gt_measure=gt_name,
                        granularity="neuron",
                        mask_restricted=False,
                    )
                )
    return rows


def _overall_edge_rho(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_tag: str,
    local_signal: str,
    gt_measure: str = "grad_abs",
) -> float | None:
    """Extract overall edge Spearman for one (local, GT, tag) from correlation rows."""
    for row in rows:
        if (
            row.get("snapshot_tag") == snapshot_tag
            and row.get("local_signal") == local_signal
            and row.get("gt_measure") == gt_measure
            and row.get("granularity") == "edge"
            and row.get("is_overall")
        ):
            return _finite_or_none(row.get("spearman"))
    return None


def _gt_shift_disagreement(
    pre: ImportanceResult,
    post: ImportanceResult,
    *,
    layer_zscore: bool,
    min_pairs: int,
) -> float:
    """Spearman(I_pre, I_immediate_post) edge overall for GT-1."""
    pre_edge = _tensor_map_to_numpy(pre.edge)
    post_edge = _tensor_map_to_numpy(post.edge)
    corr = correlate_layers(
        pre_edge,
        post_edge,
        masks=None,
        layer_zscore_before_concat=layer_zscore,
        min_pairs=min_pairs,
    )
    return float(corr.overall_spearman)


def _write_time_lag_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TIME_LAG_CSV_COLUMNS)
    extra: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames and k not in extra:
                extra.append(k)
    fieldnames = fieldnames + extra
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _write_metrics_train_csv(path: Path, history: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("phase,epoch,train_loss,train_acc,train_n,train_steps,global_step\n")
        return path
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = (
        "phase",
        "epoch",
        "train_loss",
        "train_acc",
        "train_n",
        "train_steps",
        "global_step",
        "b_steps",
    )
    for k in preferred:
        if any(k in h for h in history):
            fieldnames.append(k)
            seen.add(k)
    for h in history:
        for k in h:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for h in history:
            writer.writerow({k: h.get(k, "") for k in fieldnames})
    return path


def _snapshot_epochs_a(
    epochs_a: int, *, include_a_early_mid: bool
) -> dict[int, list[str]]:
    """Epoch (1-based) → tags during Task A phase (excluding late_a switch tag)."""
    by_epoch: dict[int, list[str]] = {}
    if not include_a_early_mid or epochs_a < 1:
        return by_epoch
    by_epoch.setdefault(1, []).append("early")
    if epochs_a >= 2:
        mid = max(1, epochs_a // 2)
        by_epoch.setdefault(mid, []).append("mid")
    return by_epoch


def _lag_target_b_steps(lag_t: int) -> int:
    """Map lag tag time t → total B optimizer steps completed at that snapshot.

    ``t=0`` is after the **first** B optimizer step (1 total B step).
    ``t>0`` means ``t`` optimizer steps into B.
    """
    if lag_t <= 0:
        return 1
    return int(lag_t)


def run_task_shift(
    config: Mapping[str, Any],
    results_root: str | Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the task-shift (A→B) Phase 0 condition for one seed.

    Parameters
    ----------
    config
        Resolved runtime config (defaults ⊕ task_shift ⊕ optional smoke).
        When ``smoke=True``, the base ``smoke:`` block is merged if not already;
        raises :class:`~neuronca.io.config.SmokeConfigError` if no usable smoke
        block exists.
    results_root
        Experiment root (e.g. ``results/phase0``). Ledger freezes under
        ``{results_root}/meta/`` once per shared experiment config (seed-invariant
        content hash); artifacts under ``{results_root}/task_shift/seed_{seed}/``.
    seed
        Run seed (model init, data split, GEVB B, edge mask). Recorded in per-seed
        artifacts only — **not** stamped into the ledger content hash.
    smoke
        If True, apply CI smoke overrides (1 epoch A/B, probe 128, short lag).

    Returns
    -------
    dict
        Summary payload also written to ``summary.json``.
    """
    results_root = Path(results_root)
    cfg = copy.deepcopy(dict(config))
    if smoke:
        cfg = _apply_smoke_overrides(cfg)
    cfg.pop("smoke", None)

    ledger_cfg = _config_for_ledger(cfg, results_root=results_root, run_seed=int(seed))
    cfg["experiment"] = copy.deepcopy(ledger_cfg["experiment"])
    if "seed" in ledger_cfg:
        cfg["seed"] = ledger_cfg["seed"]

    ensure_results_tree(results_root)
    ledger: LedgerMeta = freeze_ledger(results_root, ledger_cfg)
    meta_dir = meta_dir_for(results_root)
    run_dir = seed_run_dir(results_root, CONDITION, int(seed))

    set_seed(int(seed))

    train_cfg = dict(cfg.get("train") or {})
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    gt_cfg = dict(cfg.get("gt") or {})
    tasks_cfg = dict(cfg.get("tasks") or {})
    thr_cfg = dict(eval_cfg.get("thresholds") or {})

    epochs_a = int(train_cfg.get("epochs_a", train_cfg.get("epochs", 15)))
    epochs_b = int(train_cfg.get("epochs_b", 15))
    batch_size = int(train_cfg.get("batch_size", 128))
    probe_size = int(gt_cfg.get("probe_size") or data_cfg.get("probe_size") or 1024)
    data_root = data_cfg.get("root", "data/mnist")
    num_workers = int(data_cfg.get("num_workers", 0))
    # Prefer 0 workers so atomic set_transform is immediately visible.
    train_workers = 0 if smoke else num_workers
    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)

    lag_window_steps = int(eval_cfg.get("lag_window_steps", 500))
    lag_stride = int(eval_cfg.get("lag_stride", 25))
    lag_gt_names = list(eval_cfg.get("lag_gt_measures") or ["grad_abs"])
    recovery_threshold = float(thr_cfg.get("shift_recovery_spearman", 0.60))
    relative_factor = float(thr_cfg.get("relative_recovery_factor", 0.9))
    shift_ceiling = float(
        eval_cfg.get("gt_shift_disagreement_ceiling")
        or thr_cfg.get("gt_shift_disagreement_ceiling")
        or 0.90
    )

    device = resolve_device(cfg.get("device", "auto"))
    layer_zscore = bool(eval_cfg.get("layer_zscore_before_concat", True))
    min_pairs = int(eval_cfg.get("spearman_min_pairs", 16))

    shift_type = str(tasks_cfg.get("shift_type", "label_permutation"))
    perm_seed = int(tasks_cfg.get("permutation_seed", 123))
    num_classes = int(model_cfg.get("num_classes", 10))
    shift_key = shift_type.lower().replace("-", "_")
    is_digit_split = shift_key in ("digit_split", "b2")

    # --- data: shared TaskContext (A identity first) ---
    transform_a = make_identity_transform()
    task_ctx = TaskContext(transform_a)
    digit_meta: dict[str, Any] | None = None
    train_loader_b: Any = None
    probe_loader_b: Any = None

    if is_digit_split:
        # B2: 5-way digit-range split (A: 0–4, B: 5–9 → 0–4). Overrides head size.
        digit_meta = _build_digit_split_phase_loaders(
            task_ctx,
            root=data_root,
            probe_size=probe_size,
            seed=int(seed),
            batch_size=batch_size,
            num_workers=train_workers,
        )
        train_loader = digit_meta["train_loader_a"]
        probe_loader = digit_meta["probe_loader_a"]
        train_loader_b = digit_meta["train_loader_b"]
        probe_loader_b = digit_meta["probe_loader_b"]
        transform_b = digit_meta["transform_b"]
        num_classes = int(digit_meta["num_classes"])
    else:
        transform_b = make_task_transform(
            shift_type, seed=perm_seed, num_classes=num_classes
        )
        loaders = build_mnist_loaders(
            task_ctx,
            root=data_root,
            probe_size=probe_size,
            seed=int(seed),
            batch_size=batch_size,
            num_workers=train_workers,
            download=True,
        )
        train_loader = loaders["train_loader"]
        probe_loader = loaders["probe_loader"]

    # --- model / optim ---
    hidden = tuple(model_cfg.get("hidden") or (256, 128, 64))
    model = MLPReasoningNet(
        input_dim=int(model_cfg.get("input_dim", 784)),
        hidden=hidden,  # type: ignore[arg-type]
        num_classes=num_classes,
    )
    optimizer = build_optimizer(
        model,
        name=str(train_cfg.get("optimizer", "sgd")),
        lr=float(train_cfg.get("lr", 0.01)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    # --- signals + GT suites ---
    full_measure_names = _gt_measure_names(cfg, smoke=smoke)
    lag_measure_names = [n for n in lag_gt_names if n]
    if not lag_measure_names:
        lag_measure_names = ["grad_abs"]
    # Ensure lag measures are buildable; default to grad_abs subset of full.
    for n in lag_measure_names:
        if n not in full_measure_names and n not in ("grad_abs", "gradient_saliency"):
            # still build lag-only measures even if not in coarse list
            pass

    signals = _build_signals(cfg, seed=int(seed), model=model, meta_dir=meta_dir)
    gt_full = _build_gt_measures(cfg, seed=int(seed), measure_names=full_measure_names)
    gt_lag = _build_gt_measures(cfg, seed=int(seed), measure_names=lag_measure_names)
    edge_mask = _prepare_edge_mask(
        model,
        cfg,
        seed=int(seed),
        meta_dir=meta_dir,
        measure_names=full_measure_names,
    )

    trainer = InstrumentedTrainer(
        model,
        optimizer,
        signals,
        encoder=BroadcastErrorEncoder(),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    schedule = task_shift_schedule(
        epochs_a=epochs_a,
        epochs_b=epochs_b,
        lag_window_steps=lag_window_steps,
        lag_stride=lag_stride,
        include_a_early_mid=True,
    )
    a_tags_by_epoch = _snapshot_epochs_a(epochs_a, include_a_early_mid=True)
    lag_schedule_times = lag_times(lag_window_steps, lag_stride, include_t0=True)

    all_rows: list[dict[str, Any]] = []
    train_history: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    gevb_alignment_rows: list[dict[str, Any]] = []
    isolation_guard = _eval_isolation_guard(cfg)
    isolation_checks: list[dict[str, Any]] = []

    # lag collection: (local, gt) → list of (t, rho_edge_overall)
    lag_curves: dict[tuple[str, str], list[tuple[int, float]]] = {}
    rho_pre_by_local: dict[str, float | None] = {}

    def _snap(
        gt_measures: Sequence[Any],
        *,
        tag: str,
        epoch: int | None = None,
        steps_after_switch: int | None = None,
    ) -> SnapshotResult:
        """Snapshot with live isolation assert + GEVB alignment logging."""
        check_iso = isolation_guard.for_snapshot()
        snap = trainer.snapshot(
            gt_measures,
            probe_loader,
            edge_sample_mask=edge_mask,
            check_isolation=check_iso,
        )
        if check_iso:
            isolation_checks.append(
                {
                    "tag": tag,
                    "step": snap.step,
                    "epoch": epoch,
                    "steps_after_switch": steps_after_switch,
                    "passed": True,
                }
            )
        align = _maybe_record_gevb_alignment(
            model=model,
            signals=signals,
            probe_loader=probe_loader,
            config=cfg,
            device=device,
            seed=int(seed),
            step=snap.step,
            snapshot_tag=tag,
            epoch=epoch,
            rows_out=gevb_alignment_rows,
        )
        # Stash for record helper
        snap._gevb_alignment = align  # type: ignore[attr-defined]
        snap._check_isolation = check_iso  # type: ignore[attr-defined]
        return snap

    def _record_snapshot(
        snap: SnapshotResult,
        tag: str,
        *,
        epoch: int | None = None,
        steps_after_switch: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = _correlation_rows_for_snapshot(
            snap,
            seed=int(seed),
            snapshot_tag=tag,
            layer_zscore=layer_zscore,
            min_pairs=min_pairs,
            edge_mask=edge_mask,
        )
        all_rows.extend(rows)
        snapshot_records.append(
            {
                "tag": tag,
                "epoch": epoch,
                "step": snap.step,
                "steps_after_switch": steps_after_switch,
                "n_correlation_rows": len(rows),
                "gt_measures": list(snap.gt_results.keys()),
                "local_signals": list(snap.local_scores.keys()),
                "task_version": task_ctx.version,
                "check_isolation": getattr(snap, "_check_isolation", None),
                "gevb_alignment": getattr(snap, "_gevb_alignment", None),
            }
        )
        return rows

    # ------------------------------------------------------------------
    # Phase A: train to late_a
    # ------------------------------------------------------------------
    for epoch_idx in range(epochs_a):
        epoch_1based = epoch_idx + 1
        epoch_metrics = trainer.train_steps(
            train_loader,
            max_steps=max_steps_per_epoch,
        )
        train_history.append(
            {
                "phase": "A",
                "epoch": epoch_1based,
                "train_loss": epoch_metrics["loss"],
                "train_acc": epoch_metrics["acc"],
                "train_n": epoch_metrics["n"],
                "train_steps": epoch_metrics["steps"],
                "global_step": trainer.global_step,
            }
        )
        for tag in a_tags_by_epoch.get(epoch_1based, []):
            # Skip mid/early tags that collide with late_a epoch — late_a is
            # taken once after the A loop with full switch semantics.
            if tag in ("late_a", "switch_b0"):
                continue
            snap = _snap(gt_full, tag=tag, epoch=epoch_1based)
            _record_snapshot(snap, tag, epoch=epoch_1based)

    # ------------------------------------------------------------------
    # Ordered switch boundary (weights frozen across steps 1–4)
    # ------------------------------------------------------------------
    # (1) late_a: transform A, full GT → I_pre
    weights_before = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    snap_late_a = _snap(gt_full, tag="late_a", epoch=epochs_a)
    late_a_rows = _record_snapshot(
        snap_late_a, "late_a", epoch=epochs_a, steps_after_switch=None
    )
    # rho_pre_switch per local vs GT-1
    for sig in signals:
        rho_pre_by_local[sig.name] = _overall_edge_rho(
            late_a_rows,
            snapshot_tag="late_a",
            local_signal=sig.name,
            gt_measure="grad_abs",
        )

    i_pre = snap_late_a.gt_results.get("grad_abs")
    if i_pre is None:
        # fall back to first GT result if grad_abs missing
        i_pre = next(iter(snap_late_a.gt_results.values()), None)

    # (2) atomic flip to B
    task_ctx.set_transform(transform_b)
    assert task_ctx.version >= 1
    if is_digit_split and train_loader_b is not None and probe_loader_b is not None:
        # B2: also switch to digit-filtered loaders (new classes in B).
        train_loader = train_loader_b
        probe_loader = probe_loader_b

    # (3) switch_b0: transform B, SAME weights, NO optimizer, full GT
    for name, p in model.named_parameters():
        if not torch.equal(p.detach(), weights_before[name]):
            raise RuntimeError(
                f"weights changed between late_a and switch_b0 at {name!r}"
            )
    snap_b0 = _snap(
        gt_full, tag="switch_b0", epoch=epochs_a, steps_after_switch=0
    )
    _record_snapshot(snap_b0, "switch_b0", epoch=epochs_a, steps_after_switch=0)

    # Confirm weights still frozen after switch_b0 GT
    for name, p in model.named_parameters():
        if not torch.equal(p.detach(), weights_before[name]):
            raise RuntimeError(
                f"weights changed during switch_b0 GT at {name!r}"
            )

    i_post = snap_b0.gt_results.get("grad_abs")
    if i_post is None:
        i_post = next(iter(snap_b0.gt_results.values()), None)

    # (4) gt_shift_disagreement
    gt_shift_rho: float | None = None
    if i_pre is not None and i_post is not None:
        gt_shift_rho = _gt_shift_disagreement(
            i_pre,
            i_post,
            layer_zscore=layer_zscore,
            min_pairs=min_pairs,
        )
    weak_shift = (
        gt_shift_rho is not None
        and math.isfinite(gt_shift_rho)
        and gt_shift_rho >= shift_ceiling
    )

    # ------------------------------------------------------------------
    # Phase B: first B step → lag t=0 → dense lag (GT-1) within B epochs
    # ------------------------------------------------------------------
    b_steps = 0
    lag_done: set[int] = set()
    max_b_for_lag = max(
        (_lag_target_b_steps(t) for t in lag_schedule_times), default=0
    )

    def _maybe_lag_snapshot() -> None:
        """Take GT-1 lag snapshots when b_steps hits scheduled lag targets."""
        candidate_ts: list[int] = []
        # t=0 after the first B optimizer step
        if b_steps == 1 and 0 in lag_schedule_times:
            candidate_ts.append(0)
        for t in lag_schedule_times:
            if t > 0 and _lag_target_b_steps(t) == b_steps:
                candidate_ts.append(t)
        for t in candidate_ts:
            if t in lag_done:
                continue
            snap = _snap(gt_lag, tag=f"lag_t{t}", steps_after_switch=t)
            rows = _record_snapshot(
                snap,
                f"lag_t{t}",
                steps_after_switch=t,
            )
            for row in rows:
                if (
                    row.get("granularity") == "edge"
                    and row.get("is_overall")
                    and row.get("snapshot_tag") == f"lag_t{t}"
                ):
                    key = (str(row["local_signal"]), str(row["gt_measure"]))
                    rho = _finite_or_none(row.get("spearman"))
                    lag_curves.setdefault(key, []).append(
                        (t, float(rho) if rho is not None else float("nan"))
                    )
            lag_done.add(t)

    mid_b_epoch = max(1, epochs_b // 2) if epochs_b >= 1 else None
    late_b_epoch = epochs_b if epochs_b >= 1 else None

    # B training is epoch-structured; lag samples fire on matching b_steps
    # during those epochs (not as a separate pre-epoch prefix).
    for epoch_idx in range(epochs_b):
        epoch_1based = epoch_idx + 1
        steps_this_epoch = 0
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_n = 0.0
        epoch_cap = int(max_steps_per_epoch) if max_steps_per_epoch is not None else None

        for batch in train_loader:
            if epoch_cap is not None and steps_this_epoch >= epoch_cap:
                break
            x, y = _unpack_batch(batch)
            m = trainer.train_step(x, y)
            b_steps += 1
            steps_this_epoch += 1
            epoch_loss += m["loss"] * m["n"]
            epoch_correct += m["acc"] * m["n"]
            epoch_n += m["n"]
            # (5)+(6)+(7): lag t=0 after first B step; then dense lag
            if b_steps <= max_b_for_lag:
                _maybe_lag_snapshot()

        train_history.append(
            {
                "phase": "B",
                "epoch": epoch_1based,
                "train_loss": (epoch_loss / epoch_n) if epoch_n else float("nan"),
                "train_acc": (epoch_correct / epoch_n) if epoch_n else float("nan"),
                "train_n": epoch_n,
                "train_steps": float(steps_this_epoch),
                "global_step": trainer.global_step,
                "b_steps": b_steps,
            }
        )

        # B epoch-end snapshots (design):
        # - coarse mid_b / late_b → full GT (GT-1 + GT-2)
        # - all other B epoch ends → GT-1 only (tag b_epoch_{k})
        coarse_tags: list[str] = []
        if mid_b_epoch is not None and epoch_1based == mid_b_epoch:
            coarse_tags.append("mid_b")
        if late_b_epoch is not None and epoch_1based == late_b_epoch:
            coarse_tags.append("late_b")

        if coarse_tags:
            # One full-GT compute; emit a correlation row set per coarse tag
            # (mid_b == late_b when epochs_b == 1).
            primary = coarse_tags[0]
            snap = _snap(
                gt_full,
                tag=primary,
                epoch=epoch_1based,
                steps_after_switch=b_steps,
            )
            seen_tag: set[str] = set()
            for tag in coarse_tags:
                if tag in seen_tag:
                    continue
                seen_tag.add(tag)
                _record_snapshot(
                    snap, tag, epoch=epoch_1based, steps_after_switch=b_steps
                )
        else:
            # Non-coarse B epoch end: GT-1 only
            tag = f"b_epoch_{epoch_1based}"
            snap = _snap(
                gt_lag,
                tag=tag,
                epoch=epoch_1based,
                steps_after_switch=b_steps,
            )
            _record_snapshot(
                snap,
                tag,
                epoch=epoch_1based,
                steps_after_switch=b_steps,
            )

    # ------------------------------------------------------------------
    # Analyze lag curves → time_lag.csv
    # ------------------------------------------------------------------
    time_lag_rows: list[dict[str, Any]] = []
    lag_recovery_summary: dict[str, Any] = {}

    # Ensure every (local, lag_gt) has a curve entry even if empty
    for sig in signals:
        for gt_name in lag_measure_names:
            key = (sig.name, gt_name)
            points = sorted(lag_curves.get(key, []), key=lambda p: p[0])
            times = [p[0] for p in points]
            rhos = [p[1] for p in points]
            rho_pre = rho_pre_by_local.get(sig.name)
            rho_pre_f = float(rho_pre) if rho_pre is not None else float("nan")

            if times:
                rec = analyze_lag_curve(
                    times,
                    rhos,
                    rho_pre_switch=rho_pre_f,
                    threshold=recovery_threshold,
                    max_t=float(lag_window_steps),
                    relative_factor=relative_factor,
                )
            else:
                rec = None

            lag_recovery_summary[f"{sig.name}__{gt_name}"] = {
                "t_star_abs": (rec.t_star_abs if rec else None),
                "rho_rec": (rec.rho_rec if rec else None),
                "rho_pre_switch": rho_pre_f if math.isfinite(rho_pre_f) else None,
                "recovered_abs": (rec.recovered_abs if rec else False),
                "recovered_rel": (rec.recovered_rel if rec else False),
                "t_star_rel": (rec.t_star_rel if rec else None),
                "rho_max_in_window": (rec.rho_max_in_window if rec else None),
                # Diagnostic sustained-hold metric (not a pass/fail gate).
                "t_star_sustained": (
                    rec.t_star_sustained
                    if rec and math.isfinite(rec.t_star_sustained)
                    else None
                ),
                "recovered_sustained": (rec.recovered_sustained if rec else False),
                "n_lag_points": len(times),
            }

            for t, rho in points:
                time_lag_rows.append(
                    {
                        "steps_after_switch": t,
                        "seed": int(seed),
                        "local_signal": sig.name,
                        "gt_measure": gt_name,
                        "spearman": rho if math.isfinite(rho) else None,
                        "recovered_abs": rec.recovered_abs if rec else False,
                        "recovered_rel": rec.recovered_rel if rec else False,
                        "t_star_abs": (
                            rec.t_star_abs
                            if rec and math.isfinite(rec.t_star_abs)
                            else None
                        ),
                        "t_star_rel": (
                            rec.t_star_rel
                            if rec and math.isfinite(rec.t_star_rel)
                            else None
                        ),
                        "t_star_sustained": (
                            rec.t_star_sustained
                            if rec and math.isfinite(rec.t_star_sustained)
                            else None
                        ),
                        "recovered_sustained": (
                            rec.recovered_sustained if rec else False
                        ),
                        "rho_pre_switch": (
                            rho_pre_f if math.isfinite(rho_pre_f) else None
                        ),
                        "rho_rec": (
                            rec.rho_rec if rec and math.isfinite(rec.rho_rec) else None
                        ),
                        "rho_max_in_window": (
                            rec.rho_max_in_window
                            if rec and math.isfinite(rec.rho_max_in_window)
                            else None
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    corr_path = run_dir / "correlations.csv"
    write_correlations_csv(corr_path, all_rows)

    lag_path = run_dir / "time_lag.csv"
    _write_time_lag_csv(lag_path, time_lag_rows)

    metrics_path = run_dir / "metrics_train.csv"
    _write_metrics_train_csv(metrics_path, train_history)

    align_path = run_dir / "gevb_alignment.csv"
    if gevb_alignment_rows:
        _write_csv(align_path, GEVB_ALIGNMENT_CSV_COLUMNS, gevb_alignment_rows)

    final_train = next(
        (h for h in reversed(train_history) if h.get("phase") == "B"),
        train_history[-1] if train_history else {},
    )

    # Late-A alignment snapshot (pre-switch) for summary inspection
    late_a_align = None
    if gevb_alignment_rows:
        late_a_rows_align = [
            r for r in gevb_alignment_rows if r.get("snapshot_tag") == "late_a"
        ]
        if late_a_rows_align:
            late_a_align = {
                str(r["layer"]): {
                    "fa_signal_angle_deg": r.get("fa_signal_angle_deg"),
                    "subspace_principal_angle_deg": r.get(
                        "subspace_principal_angle_deg"
                    ),
                }
                for r in late_a_rows_align
            }

    summary: dict[str, Any] = {
        "seed": int(seed),
        "condition": CONDITION,
        "smoke": bool(smoke),
        "experiment_id": ledger.experiment_id,
        "content_sha256": ledger.content_sha256,
        "epochs_a": epochs_a,
        "epochs_b": epochs_b,
        "probe_size": probe_size,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "device": str(device),
        "global_step": trainer.global_step,
        "b_steps": b_steps,
        "digit_split": (
            {
                "a_digits": [0, 1, 2, 3, 4],
                "b_digits": [5, 6, 7, 8, 9],
                "n_train_a": digit_meta.get("n_train_a") if digit_meta else None,
                "n_probe_a": digit_meta.get("n_probe_a") if digit_meta else None,
                "n_train_b": digit_meta.get("n_train_b") if digit_meta else None,
                "n_probe_b": digit_meta.get("n_probe_b") if digit_meta else None,
            }
            if is_digit_split
            else None
        ),
        "max_steps_per_epoch": max_steps_per_epoch,
        "lag_window_steps": lag_window_steps,
        "lag_stride": lag_stride,
        "lag_times_scheduled": lag_schedule_times,
        "lag_times_observed": sorted(lag_done),
        "gt_measures_full": full_measure_names,
        "gt_measures_lag": lag_measure_names,
        "local_signals": [s.name for s in signals],
        "shift_type": shift_type,
        "permutation_seed": perm_seed,
        "task_version_final": task_ctx.version,
        "snapshot_tags": [s.tag for s in schedule.snapshots],
        "snapshots": snapshot_records,
        "train_history": train_history,
        "final_train_loss": final_train.get("train_loss"),
        "final_train_acc": final_train.get("train_acc"),
        "gt_shift_disagreement": gt_shift_rho,
        "weak_shift": weak_shift,
        "gt_shift_disagreement_ceiling": shift_ceiling,
        "rho_pre_switch": {
            k: (v if v is not None else None) for k, v in rho_pre_by_local.items()
        },
        "lag_recovery": lag_recovery_summary,
        "check_isolation_mode": isolation_guard.mode,
        "isolation_checks": isolation_checks,
        "gevb_alignment_late_a": late_a_align,
        "artifacts": {
            "correlations_csv": str(corr_path),
            "time_lag_csv": str(lag_path),
            "metrics_train_csv": str(metrics_path),
            "gevb_alignment_csv": str(align_path) if gevb_alignment_rows else None,
            "run_dir": str(run_dir),
            "meta_dir": str(meta_dir),
        },
        "thresholds": dict(ledger.thresholds),
        "switch_sequence": [
            "late_a",
            "set_transform_B",
            "switch_b0",
            "first_b_step",
            "lag_t0",
            "dense_lag",
            "b_epoch_end_gt1_or_coarse_full",
        ],
    }

    summary_path = run_dir / "summary.json"
    summary["artifacts"]["summary_json"] = str(summary_path)
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary
