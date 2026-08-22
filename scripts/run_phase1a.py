#!/usr/bin/env python3
"""Phase 1a structural plasticity runner (fixed / sliding / baseline configs)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import (  # noqa: E402
    TaskContext,
    make_identity_transform,
    make_label_permutation_transform,
)
from neuronca.io.config import deep_merge, load_yaml, resolve_device  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet  # noqa: E402
from neuronca.models.mlp import MLPReasoningNet  # noqa: E402
from neuronca.plasticity.diagnostics import (  # noqa: E402
    length_histogram,
    prune_origin_summary,
    sparsity_report,
    stability_check,
)
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.plasticity.structure_snapshot import (  # noqa: E402
    StructureSnapshotRecorder,
)
from neuronca.training.loop import evaluate, train_epoch  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import train_plastic_model  # noqa: E402


def _structure_snapshot_recorder(
    cfg: dict[str, Any],
    seed_dir: Path,
    epochs: int,
) -> StructureSnapshotRecorder | None:
    """Opt-in structure snapshot export (default off)."""
    exp = cfg.get("experiment") or {}
    log = cfg.get("logging") or {}
    record = bool(
        exp.get(
            "record_structure_snapshots",
            log.get("record_structure_snapshots", False),
        )
    )
    cadence = exp.get(
        "structure_snapshot_cadence",
        log.get("structure_snapshot_cadence", "every_epoch"),
    )
    return StructureSnapshotRecorder.from_config(
        seed_dir,
        total_epochs=epochs,
        record=record,
        cadence=cadence,
    )


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    smoke = cfg.get("smoke") or {}
    if not smoke:
        return cfg
    out = deep_merge(cfg, smoke)
    out.pop("smoke", None)
    return out


def _plasticity_config(cfg: dict[str, Any]) -> PlasticityConfig:
    p = cfg.get("plasticity") or {}
    raw_rates = p.get("target_rates")
    target_rates: dict[str, float] | None = None
    if isinstance(raw_rates, dict):
        # Accept h1/h2/h3 or fc1/fc2/fc3 keys.
        aliases = {"fc1": "h1", "fc2": "h2", "fc3": "h3"}
        target_rates = {}
        for k, v in raw_rates.items():
            key = aliases.get(str(k), str(k))
            target_rates[key] = float(v)
    target_rate = p.get("target_rate", None)
    return PlasticityConfig(
        radius=float(p.get("radius", 0.35)),
        target_rates=target_rates or {},
        target_rate=float(target_rate) if target_rate is not None and not target_rates else None,
        tau=float(p.get("tau", 0.5)),
        threshold_kind=str(p.get("threshold_kind", "fixed")),
        mu=float(p.get("mu", 0.999)),
        prune_edge_fraction=float(p.get("prune_edge_fraction", 0.05)),
        prune_neuron_z=float(p.get("prune_neuron_z", -1.0)),
        max_new_edges_per_layer=int(p.get("max_new_edges_per_layer", 64)),
        max_new_neurons_per_layer=int(p.get("max_new_neurons_per_layer", 4)),
        seed_edges_per_new_neuron=int(p.get("seed_edges_per_new_neuron", 4)),
        edge_eligibility_mode=str(
            p.get("edge_eligibility_mode", "neither_contraction")
        ),
        newborn_grace_steps=int(p.get("newborn_grace_steps", 3)),
        newborn_ema_init=str(p.get("newborn_ema_init", "layer_target")),
        newborn_birth_bias=float(p.get("newborn_birth_bias", 0.5)),
        newborn_birth_bias_steps=int(p.get("newborn_birth_bias_steps", 3)),
        newborn_weight_init=str(p.get("newborn_weight_init", "kaiming")),
        newborn_parent_noise=float(p.get("newborn_parent_noise", 0.1)),
        newborn_copy_outgoing=bool(p.get("newborn_copy_outgoing", True)),
        newborn_parent_coord_jitter=float(p.get("newborn_parent_coord_jitter", 0.05)),
        newborn_lr_mult=float(p.get("newborn_lr_mult", 1.0)),
        neuron_growth_trigger=str(p.get("neuron_growth_trigger", "activation")),
        novelty_k_min=int(p.get("novelty_k_min", 11)),
        novelty_rank_fraction=float(p.get("novelty_rank_fraction", 1.0)),
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_structure_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Flatten nested counts for CSV.
    flat_rows = []
    for r in rows:
        flat = {
            "step": r.get("step"),
            "epoch": r.get("epoch"),
            "edges_pruned_init_dense": r.get("edges_pruned_init_dense"),
            "edges_pruned_incentive_grown": r.get("edges_pruned_incentive_grown"),
            "edges_grown": r.get("edges_grown"),
            "neurons_pruned": r.get("neurons_pruned"),
            "neurons_grown": r.get("neurons_grown"),
            "cooldown_blocks": r.get("cooldown_blocks"),
        }
        hc = r.get("hidden_counts") or {}
        for k, v in hc.items():
            flat[f"n_{k}"] = v
        flat_rows.append(flat)
    keys = list(flat_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat_rows)


def run_plastic_stationary(
    cfg: dict[str, Any],
    results_root: Path,
    seed: int,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    net = DynamicMLPReasoningNet(
        input_dim=int(mcfg.get("input_dim", 784)),
        hidden_init=tuple(mcfg.get("hidden_init", [256, 128, 64])),
        hidden_max=tuple(mcfg.get("hidden_max", [512, 256, 128])),
        num_classes=int(mcfg.get("num_classes", 10)),
        coord_seed=seed,
    )
    opt = build_optimizer(
        net,
        tcfg.get("optimizer", "sgd"),
        lr=float(tcfg.get("lr", 0.01)),
        momentum=float(tcfg.get("momentum", 0.9)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    orch = PlasticityOrchestrator(_plasticity_config(cfg))

    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=cfg.get("data", {}).get("root", "data/mnist"),
        batch_size=int(tcfg.get("batch_size", 128)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 0 if smoke else 2)),
        seed=seed,
        probe_size=128 if smoke else 1024,
    )

    epochs = int(tcfg.get("epochs", 20))
    max_steps = 5 if smoke else None
    seed_dir = results_root / "stationary" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    snap_rec = _structure_snapshot_recorder(cfg, seed_dir, epochs)
    out = train_plastic_model(
        net,
        loaders["train_loader"],
        opt,
        orch,
        epochs=epochs,
        device=device,
        val_loader=loaders["test_loader"],
        max_steps_per_epoch=max_steps,
        probe_loader=loaders.get("probe_loader"),
        structure_snapshot_recorder=snap_rec,
    )

    test_m = evaluate(net, loaders["test_loader"], device=device, max_steps=max_steps)

    summary = {
        "seed": seed,
        "variant": (cfg.get("experiment") or {}).get("variant"),
        "threshold_kind": (cfg.get("plasticity") or {}).get("threshold_kind"),
        "test_acc": test_m["acc"],
        "test_loss": test_m["loss"],
        "active_params": net.active_parameter_count(),
        "capacity_params": net.capacity_parameter_count(),
        "hidden_counts": net.hidden_counts(),
        "edge_counts": net.edge_counts(by_origin=True),
        "sparsity": sparsity_report(net),
        "length_hist": length_histogram(net, max_r=float((cfg.get("plasticity") or {}).get("radius", 0.35)) * 3),
        "prune_origin": prune_origin_summary(orch),
        "stability": stability_check(
            orch.history,
            max_hidden=sum(mcfg.get("hidden_max", [512, 256, 128])),
            init_hidden_sum=sum(mcfg.get("hidden_init", [256, 128, 64])),
        ),
        "train_history": out["train"],
        "val_history": out["val"],
    }
    _write_json(seed_dir / "summary.json", summary)
    _write_structure_csv(seed_dir / "structure_timeseries.csv", out["structure"])
    _write_json(seed_dir / "length_hist.json", summary["length_hist"])
    _write_json(seed_dir / "sparsity.json", summary["sparsity"])
    torch.save(
        {
            "active_h1": net.active_h1.cpu(),
            "active_h2": net.active_h2.cpu(),
            "active_h3": net.active_h3.cpu(),
            "edge_origin_fc1": net.edge_origin_fc1.cpu(),
            "edge_origin_fc2": net.edge_origin_fc2.cpu(),
            "edge_origin_fc3": net.edge_origin_fc3.cpu(),
            "edge_origin_fc_out": net.edge_origin_fc_out.cpu(),
        },
        seed_dir / "structure_final.pt",
    )
    return summary


def run_plastic_task_shift(
    cfg: dict[str, Any],
    results_root: Path,
    seed: int,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train on task A then switch labels (B1) while plasticity continues."""
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    tasks = cfg.get("tasks") or {}

    net = DynamicMLPReasoningNet(
        input_dim=int(mcfg.get("input_dim", 784)),
        hidden_init=tuple(mcfg.get("hidden_init", [256, 128, 64])),
        hidden_max=tuple(mcfg.get("hidden_max", [512, 256, 128])),
        num_classes=int(mcfg.get("num_classes", 10)),
        coord_seed=seed,
    )
    opt = build_optimizer(
        net,
        tcfg.get("optimizer", "sgd"),
        lr=float(tcfg.get("lr", 0.01)),
        momentum=float(tcfg.get("momentum", 0.9)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    orch = PlasticityOrchestrator(_plasticity_config(cfg))

    epochs_a = int(tcfg.get("epochs_a", 15))
    epochs_b = int(tcfg.get("epochs_b", 15))
    max_steps = 5 if smoke else None
    batch_size = int(tcfg.get("batch_size", 128))
    data_root = cfg.get("data", {}).get("root", "data/mnist")
    nw = int(cfg.get("data", {}).get("num_workers", 0 if smoke else 2))

    seed_dir = results_root / "task_shift" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Snapshots use phase-A epoch indices; B continues from epoch 0 of out_b.
    # When recording is on, dump every A epoch (and separately B if reconfigured).
    snap_rec = _structure_snapshot_recorder(cfg, seed_dir, epochs_a + epochs_b)

    task_a = TaskContext(make_identity_transform())
    loaders_a = build_mnist_loaders(
        task_a,
        root=data_root,
        batch_size=batch_size,
        num_workers=nw,
        seed=seed,
        probe_size=128 if smoke else 1024,
    )
    out_a = train_plastic_model(
        net,
        loaders_a["train_loader"],
        opt,
        orch,
        epochs=epochs_a,
        device=device,
        val_loader=loaders_a["test_loader"],
        max_steps_per_epoch=max_steps,
        probe_loader=loaders_a.get("probe_loader"),
        structure_snapshot_recorder=snap_rec,
    )
    acc_a = evaluate(net, loaders_a["test_loader"], device=device, max_steps=max_steps)

    perm_seed = int(tasks.get("permutation_seed", 123))
    task_b = TaskContext(make_label_permutation_transform(seed=perm_seed))
    loaders_b = build_mnist_loaders(
        task_b,
        root=data_root,
        batch_size=batch_size,
        num_workers=nw,
        seed=seed,
        probe_size=128 if smoke else 1024,
    )
    # Continue same net/opt/orch through phase B. Snapshot epochs continue
    # if recorder cadence covers epochs_a..total-1 — remap via a wrapper.
    if snap_rec is not None:
        # Offset B-phase local epochs so files don't collide with A.
        offset = epochs_a
        orig_maybe = snap_rec.maybe_record

        def _maybe_record_b(epoch, model, **kwargs):  # type: ignore[no-untyped-def]
            return orig_maybe(epoch + offset, model, **kwargs)

        snap_rec.maybe_record = _maybe_record_b  # type: ignore[method-assign]
    out_b = train_plastic_model(
        net,
        loaders_b["train_loader"],
        opt,
        orch,
        epochs=epochs_b,
        device=device,
        val_loader=loaders_b["test_loader"],
        max_steps_per_epoch=max_steps,
        probe_loader=loaders_b.get("probe_loader"),
        structure_snapshot_recorder=snap_rec,
    )
    acc_b = evaluate(net, loaders_b["test_loader"], device=device, max_steps=max_steps)
    structure_all = out_a["structure"] + [
        {**s, "epoch": (s.get("epoch") or 0) + epochs_a} for s in out_b["structure"]
    ]
    # Structure event rate around switch (last A epoch vs first B epochs).
    switch_idx = epochs_a
    near = [
        s
        for s in structure_all
        if s.get("epoch") is not None and abs(int(s["epoch"]) - switch_idx) <= 2
    ]
    summary = {
        "seed": seed,
        "variant": (cfg.get("experiment") or {}).get("variant"),
        "acc_end_a": acc_a["acc"],
        "acc_end_b": acc_b["acc"],
        "active_params": net.active_parameter_count(),
        "hidden_counts": net.hidden_counts(),
        "edge_counts": net.edge_counts(by_origin=True),
        "prune_origin": prune_origin_summary(orch),
        "structure_near_switch": near,
        "sparsity": sparsity_report(net),
        "length_hist": length_histogram(net),
        "val_a": out_a["val"],
        "val_b": out_b["val"],
    }
    _write_json(seed_dir / "summary.json", summary)
    _write_structure_csv(seed_dir / "structure_timeseries.csv", structure_all)
    return summary


def run_fixed_baseline(
    cfg: dict[str, Any],
    results_root: Path,
    seed: int,
    *,
    smoke: bool = False,
    hidden: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    h = hidden or tuple(mcfg.get("hidden", [256, 128, 64]))
    net = MLPReasoningNet(
        input_dim=int(mcfg.get("input_dim", 784)),
        hidden=h,
        num_classes=int(mcfg.get("num_classes", 10)),
    )
    opt = build_optimizer(
        net,
        tcfg.get("optimizer", "sgd"),
        lr=float(tcfg.get("lr", 0.01)),
        momentum=float(tcfg.get("momentum", 0.9)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=cfg.get("data", {}).get("root", "data/mnist"),
        batch_size=int(tcfg.get("batch_size", 128)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 0 if smoke else 2)),
        seed=seed,
        probe_size=128 if smoke else 1024,
    )
    epochs = int(tcfg.get("epochs", 20))
    max_steps = 5 if smoke else None
    history = []
    for ep in range(epochs):
        m = train_epoch(
            net,
            loaders["train_loader"],
            opt,
            device=device,
            max_steps=max_steps,
        )
        history.append({"epoch": ep, **m})
    test_m = evaluate(net, loaders["test_loader"], device=device, max_steps=max_steps)
    n_params = sum(p.numel() for p in net.parameters())
    seed_dir = results_root / "baseline" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "variant": "fixed_topology",
        "hidden": list(h),
        "test_acc": test_m["acc"],
        "test_loss": test_m["loss"],
        "params": n_params,
        "train_history": history,
    }
    _write_json(seed_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1a structural plasticity runner")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument(
        "--condition",
        choices=["stationary", "task_shift", "both"],
        default="stationary",
    )
    p.add_argument("--results-root", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--baseline", action="store_true", help="Run fixed-topology baseline")
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.smoke:
        cfg = _apply_smoke(cfg)

    seeds = (
        _parse_seeds(args.seeds)
        if args.seeds
        else list((cfg.get("experiment") or {}).get("seeds") or [0])
    )
    results_root = Path(
        args.results_root
        or (cfg.get("experiment") or {}).get("output_dir")
        or "results/phase1a"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "meta").mkdir(exist_ok=True)
    (results_root / "meta" / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[phase1a] seed={seed} smoke={args.smoke} baseline={args.baseline}", flush=True)
        if args.baseline:
            summaries.append(
                run_fixed_baseline(cfg, results_root, seed, smoke=args.smoke)
            )
            continue
        if args.condition in ("stationary", "both"):
            summaries.append(
                run_plastic_stationary(cfg, results_root, seed, smoke=args.smoke)
            )
        if args.condition in ("task_shift", "both"):
            summaries.append(
                run_plastic_task_shift(cfg, results_root, seed, smoke=args.smoke)
            )

    _write_json(results_root / "aggregate" / "summaries.json", summaries)
    print(f"[phase1a] done → {results_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
