"""Stationary condition experiment runner.

Wires config → MNIST + MLP + SGD + local signals + GT → InstrumentedTrainer
with scheduled early/mid/late snapshots, then writes correlations.csv and
summary.json under ``{results_root}/stationary/seed_{k}/``.

Does **not** implement task-shift (see ``task_shift.py``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from neuronca.broadcast.error import BroadcastErrorEncoder
from neuronca.data.mnist import build_mnist_loaders
from neuronca.data.tasks import TaskContext, make_identity_transform
from neuronca.eval.correlation import correlate_layers
from neuronca.eval.schedule import SnapshotSpec, stationary_schedule
from neuronca.importance.ablation import (
    AblationImportance,
    build_edge_ablation_mask,
    default_edge_mask_path,
    save_edge_ablation_mask,
)
from neuronca.importance.base import GroundTruthImportance, ImportanceResult
from neuronca.importance.fisher import FisherImportance
from neuronca.importance.gradient_saliency import GradientSaliency
from neuronca.io.artifacts import (
    correlation_rows_from_result,
    ensure_results_tree,
    seed_run_dir,
    write_correlations_csv,
)
from neuronca.io.config import (
    DEFAULT_CONFIG_PATH,
    SmokeConfigError,
    deep_merge,
    load_yaml,
)
from neuronca.io.ledger import LedgerMeta, freeze_ledger, meta_dir_for
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.signals import build_signal
from neuronca.signals.base import ScoreBundle
from neuronca.signals.gevb_eligibility import default_gevb_persist_path
from neuronca.training.callbacks import InstrumentedTrainer, SnapshotResult
from neuronca.training.loop import resolve_device
from neuronca.training.optim import build_optimizer


def _apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge the base ``smoke:`` block onto a copy of ``config``.

    Aligns with :func:`neuronca.io.config.load_config`: if ``smoke=True`` but
    no usable base smoke mapping exists, raise :class:`SmokeConfigError` so CI
    cannot silently run a full-budget config.
    """
    cfg = copy.deepcopy(dict(config))
    try:
        base = load_yaml(DEFAULT_CONFIG_PATH)
    except FileNotFoundError as exc:
        raise SmokeConfigError(
            f"smoke=True but defaults config not found at {DEFAULT_CONFIG_PATH}. "
            "Define a non-empty top-level smoke: mapping on the defaults file."
        ) from exc
    smoke_block = base.get("smoke")
    if not isinstance(smoke_block, Mapping) or not smoke_block:
        raise SmokeConfigError(
            f"smoke=True but no usable smoke: block found in base config "
            f"{DEFAULT_CONFIG_PATH}. Define a non-empty top-level smoke: mapping "
            "on the defaults file; overrides must not be relied on to supply it."
        )
    cfg = deep_merge(cfg, smoke_block)
    cfg.pop("smoke", None)
    return cfg


def _config_for_ledger(
    config: Mapping[str, Any],
    *,
    results_root: Path,
    run_seed: int,
) -> dict[str, Any]:
    """Build a **seed-invariant** config for primary ledger freeze.

    Multi-seed Phase 0 layout is one shared ``{results_root}/meta/`` plus
    per-seed artifact dirs ``stationary/seed_{k}/``. The content hash must
    **not** include the per-run seed, or a second seed on the same root raises
    :class:`~neuronca.io.ledger.LedgerConflictError`.

    Top-level ``seed`` is left as in the shared experiment config (or the first
    entry of ``experiment.seeds``). The per-run seed is applied only via
    :func:`set_seed` and seed-dependent constructors after freeze.
    """
    cfg = copy.deepcopy(dict(config))
    cfg.pop("smoke", None)
    exp = dict(cfg.get("experiment") or {})
    exp.setdefault("output_dir", str(results_root))
    seeds = exp.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        # Single-seed caller: record the run seed in the shared list only.
        exp["seeds"] = [int(run_seed)]
    else:
        exp["seeds"] = [int(s) for s in seeds]
    cfg["experiment"] = exp
    # Prefer a stable top-level seed from the experiment seeds list (not run_seed
    # alone) so freeze is identical across seeds 0/1/2 for the same config.
    if "seed" not in cfg or cfg.get("seed") is None:
        cfg["seed"] = int(exp["seeds"][0])
    # Never overwrite cfg["seed"] with run_seed — that would change the hash.
    return cfg


def _tensor_map_to_numpy(scores: Mapping[str, Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().float().cpu().numpy() for k, v in scores.items()}


def _mask_map_to_numpy(
    masks: Mapping[str, Tensor] | None,
) -> dict[str, np.ndarray] | None:
    if not masks:
        return None
    return {k: v.detach().cpu().numpy() for k, v in masks.items()}


def _build_signals(
    config: Mapping[str, Any],
    *,
    seed: int,
    model: MLPReasoningNet,
    meta_dir: Path,
) -> list:
    """Construct all configured local signal candidates."""
    sig_cfg = dict(config.get("signals") or {})
    decay = float(sig_cfg.get("decay", 0.99))
    names: Sequence[str] = list(
        sig_cfg.get("candidates")
        or [
            "activation",
            "hebbian_corr",
            "eligibility_residual_energy",
            "eligibility_gevb_rf",
        ]
    )
    broadcast_cfg = dict(config.get("broadcast") or {})
    gevb_cfg = dict(broadcast_cfg.get("gevb_feedback") or {})
    model_cfg = dict(config.get("model") or {})
    num_classes = int(model_cfg.get("num_classes", 10))
    layer_shapes = model.layer_shapes()

    signals = []
    for name in names:
        kwargs: dict[str, Any] = {}
        if name == "eligibility_gevb_rf":
            kwargs = {
                "experiment_seed": int(seed),
                "gevb_feedback_seed": int(gevb_cfg.get("seed", 0)),
                "num_classes": num_classes,
                "layer_shapes": layer_shapes,
                "persist_path": default_gevb_persist_path(seed, meta_dir),
                "init": str(gevb_cfg.get("init", "gaussian_scaled")),
            }
        signals.append(build_signal(name, decay=decay, **kwargs))
    return signals


def _gt_measure_names(config: Mapping[str, Any], *, smoke: bool) -> list[str]:
    """Resolve which GT measure names to run for stationary snapshots."""
    eval_cfg = dict(config.get("eval") or {})
    gt_cfg = dict(config.get("gt") or {})
    # Full suite default; smoke / coarse list can shrink.
    names = list(
        eval_cfg.get("coarse_gt_measures")
        or ["grad_abs", "ablation_relu"]
    )
    if smoke and not eval_cfg.get("coarse_gt_measures"):
        names = ["grad_abs"]
    if gt_cfg.get("fisher") and "fisher" not in names:
        names = list(names) + ["fisher"]
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _build_gt_measures(
    config: Mapping[str, Any],
    *,
    seed: int,
    measure_names: Sequence[str],
) -> list[GroundTruthImportance]:
    gt_cfg = dict(config.get("gt") or {})
    measures: list[GroundTruthImportance] = []
    for name in measure_names:
        if name in ("grad_abs", "gradient_saliency"):
            measures.append(
                GradientSaliency(
                    grad_batch_size=gt_cfg.get("grad_batch_size"),
                )
            )
        elif name in ("ablation_relu", "ablation"):
            measures.append(
                AblationImportance(
                    edge_ablation_sample=int(gt_cfg.get("edge_ablation_sample", 512)),
                    neuron_ablation=str(gt_cfg.get("neuron_ablation", "all_hidden")),
                    rank_transform=str(gt_cfg.get("ablation_rank_transform", "relu")),
                    seed=int(seed),
                    edge_mask_seed_offset=int(gt_cfg.get("edge_mask_seed_offset", 17)),
                )
            )
        elif name == "fisher":
            measures.append(FisherImportance())
        else:
            raise ValueError(f"unknown gt measure {name!r}")
    return measures


def _prepare_edge_mask(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    seed: int,
    meta_dir: Path,
    measure_names: Sequence[str],
) -> dict[str, Tensor] | None:
    """Build/persist fixed edge ablation mask when GT-2 edges are requested."""
    if "ablation_relu" not in measure_names and "ablation" not in measure_names:
        return None
    gt_cfg = dict(config.get("gt") or {})
    sample = int(gt_cfg.get("edge_ablation_sample", 512))
    if sample <= 0:
        return None
    mask = build_edge_ablation_mask(
        model,
        seed=int(seed),
        sample_per_layer=sample,
        edge_mask_seed_offset=int(gt_cfg.get("edge_mask_seed_offset", 17)),
    )
    path = default_edge_mask_path(meta_dir, seed)
    if not path.is_file():
        save_edge_ablation_mask(mask, path)
    return mask


def _correlation_rows_for_snapshot(
    snap: SnapshotResult,
    *,
    seed: int,
    snapshot_tag: str,
    layer_zscore: bool,
    min_pairs: int,
    edge_mask: dict[str, Tensor] | None,
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

            # Edge correlations
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
                        condition="stationary",
                        snapshot_tag=snapshot_tag,
                        step=snap.step,
                        local_signal=sig_name,
                        gt_measure=gt_name,
                        granularity="edge",
                        mask_restricted=mask_restricted,
                    )
                )

            # Neuron correlations (skip empty GT neuron maps)
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
                        condition="stationary",
                        snapshot_tag=snapshot_tag,
                        step=snap.step,
                        local_signal=sig_name,
                        gt_measure=gt_name,
                        granularity="neuron",
                        mask_restricted=False,
                    )
                )
    return rows


def _snapshot_epochs(specs: Sequence[SnapshotSpec]) -> dict[int, list[SnapshotSpec]]:
    """Group stationary snapshot specs by 1-indexed epoch."""
    by_epoch: dict[int, list[SnapshotSpec]] = {}
    for spec in specs:
        if spec.epoch is None:
            continue
        by_epoch.setdefault(int(spec.epoch), []).append(spec)
    return by_epoch


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _best_late_edge_rho(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick best local candidate by overall edge Spearman vs grad_abs at late."""
    best_name: str | None = None
    best_rho = float("-inf")
    per_candidate: dict[str, float | None] = {}
    for row in rows:
        if (
            row.get("snapshot_tag") != "late"
            or row.get("granularity") != "edge"
            or row.get("gt_measure") != "grad_abs"
            or not row.get("is_overall")
        ):
            continue
        name = str(row.get("local_signal") or "")
        rho = _finite_or_none(row.get("spearman"))
        per_candidate[name] = rho
        if rho is not None and rho > best_rho:
            best_rho = rho
            best_name = name
    return {
        "best_local": best_name,
        "best_rho": best_rho if best_name is not None else None,
        "per_candidate": per_candidate,
    }


def run_stationary(
    config: Mapping[str, Any],
    results_root: str | Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the stationary (Task A only) Phase 0 condition for one seed.

    Parameters
    ----------
    config
        Resolved runtime config (defaults ⊕ stationary ⊕ optional smoke).
        When ``smoke=True``, the base ``smoke:`` block is merged if not already;
        raises :class:`~neuronca.io.config.SmokeConfigError` if no usable smoke
        block exists (same contract as :func:`~neuronca.io.config.load_config`).
    results_root
        Experiment root (e.g. ``results/phase0``). Ledger freezes under
        ``{results_root}/meta/`` once per shared experiment config (seed-invariant
        content hash); artifacts under ``{results_root}/stationary/seed_{seed}/``.
        Multiple seeds may share the same ``results_root``.
    seed
        Run seed (model init, data split, GEVB B, edge mask). Recorded in per-seed
        artifacts only — **not** stamped into the ledger content hash.
    smoke
        If True, apply CI smoke overrides (1 epoch, probe 128, skip edge
        ablation when configured).

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

    # Freeze ledger once per experiment with a seed-invariant shared config.
    # Per-run seed is applied after freeze (set_seed / loaders / GEVB / masks)
    # and recorded only in per-seed artifacts (CSV / summary), not the content
    # hash — so seeds 0/1/2 share one {results_root}/meta/.
    ledger_cfg = _config_for_ledger(cfg, results_root=results_root, run_seed=int(seed))
    # Keep runtime cfg aligned with ledger (output_dir / seeds list) but do not
    # stamp the per-run seed into top-level cfg["seed"].
    cfg["experiment"] = copy.deepcopy(ledger_cfg["experiment"])
    if "seed" in ledger_cfg:
        cfg["seed"] = ledger_cfg["seed"]

    ensure_results_tree(results_root)
    ledger: LedgerMeta = freeze_ledger(results_root, ledger_cfg)
    meta_dir = meta_dir_for(results_root)
    run_dir = seed_run_dir(results_root, "stationary", int(seed))

    # Per-run RNG only — after freeze so multi-seed callers share one ledger.
    set_seed(int(seed))

    train_cfg = dict(cfg.get("train") or {})
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    gt_cfg = dict(cfg.get("gt") or {})

    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 128))
    # Prefer gt.probe_size (smoke overrides both); fall back to data.probe_size.
    probe_size = int(gt_cfg.get("probe_size") or data_cfg.get("probe_size") or 1024)
    data_root = data_cfg.get("root", "data/mnist")
    num_workers = int(data_cfg.get("num_workers", 0))
    # Prefer 0 workers for probe/GT stability; training can use configured workers.
    train_workers = 0 if smoke else num_workers
    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)

    device = resolve_device(cfg.get("device", "auto"))
    layer_zscore = bool(eval_cfg.get("layer_zscore_before_concat", True))
    min_pairs = int(eval_cfg.get("spearman_min_pairs", 16))

    # --- data (Task A identity only) ---
    task_ctx = TaskContext(make_identity_transform())
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
        num_classes=int(model_cfg.get("num_classes", 10)),
    )
    optimizer = build_optimizer(
        model,
        name=str(train_cfg.get("optimizer", "sgd")),
        lr=float(train_cfg.get("lr", 0.01)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    # --- signals + GT ---
    measure_names = _gt_measure_names(cfg, smoke=smoke)
    signals = _build_signals(cfg, seed=int(seed), model=model, meta_dir=meta_dir)
    gt_measures = _build_gt_measures(cfg, seed=int(seed), measure_names=measure_names)
    edge_mask = _prepare_edge_mask(
        model,
        cfg,
        seed=int(seed),
        meta_dir=meta_dir,
        measure_names=measure_names,
    )

    trainer = InstrumentedTrainer(
        model,
        optimizer,
        signals,
        encoder=BroadcastErrorEncoder(),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    schedule = stationary_schedule(epochs=epochs)
    by_epoch = _snapshot_epochs(schedule.snapshots)

    all_rows: list[dict[str, Any]] = []
    train_history: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []

    for epoch_idx in range(epochs):
        epoch_1based = epoch_idx + 1
        epoch_metrics = trainer.train_steps(
            train_loader,
            max_steps=max_steps_per_epoch,
        )
        row = {
            "epoch": epoch_1based,
            "train_loss": epoch_metrics["loss"],
            "train_acc": epoch_metrics["acc"],
            "train_n": epoch_metrics["n"],
            "train_steps": epoch_metrics["steps"],
            "global_step": trainer.global_step,
        }
        train_history.append(row)

        if epoch_1based not in by_epoch:
            continue

        # One isolated snapshot (freeze → GT signals-off → resume); emit rows
        # for every schedule tag that lands on this epoch (smoke: early=mid=late).
        snap = trainer.snapshot(
            gt_measures,
            probe_loader,
            edge_sample_mask=edge_mask,
            check_isolation=False,
        )
        tags = [s.tag for s in by_epoch[epoch_1based]]
        for tag in tags:
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
                    "epoch": epoch_1based,
                    "step": snap.step,
                    "n_correlation_rows": len(rows),
                    "gt_measures": list(snap.gt_results.keys()),
                    "local_signals": list(snap.local_scores.keys()),
                }
            )

    corr_path = run_dir / "correlations.csv"
    write_correlations_csv(corr_path, all_rows)

    late_best = _best_late_edge_rho(all_rows)
    final_train = train_history[-1] if train_history else {}

    summary: dict[str, Any] = {
        "seed": int(seed),
        "condition": "stationary",
        "smoke": bool(smoke),
        "experiment_id": ledger.experiment_id,
        "content_sha256": ledger.content_sha256,
        "epochs": epochs,
        "probe_size": probe_size,
        "batch_size": batch_size,
        "device": str(device),
        "global_step": trainer.global_step,
        "max_steps_per_epoch": max_steps_per_epoch,
        "gt_measures": measure_names,
        "local_signals": [s.name for s in signals],
        "snapshot_tags": [s.tag for s in schedule.snapshots],
        "snapshots": snapshot_records,
        "train_history": train_history,
        "final_train_loss": final_train.get("train_loss"),
        "final_train_acc": final_train.get("train_acc"),
        "late_edge_grad_abs": late_best,
        "artifacts": {
            "correlations_csv": str(corr_path),
            "run_dir": str(run_dir),
            "meta_dir": str(meta_dir),
        },
        "thresholds": dict(ledger.thresholds),
    }

    # JSON-safe: replace non-finite floats
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

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    summary["artifacts"]["summary_json"] = str(summary_path)
    # Re-write with summary path included
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary
