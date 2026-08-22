#!/usr/bin/env python3
"""Fashion grow/prune 2×2 (full / grow_only / prune_only / frozen) under calibrated r*."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "fashion_mnist_calibration" / "grow_prune_2x2"
FASHION_R = {"h1": 0.71, "h2": 0.80, "h3": 0.91}

# Product defaults
RADIUS = 0.35
K_MIN = 11
TAU = 0.5
PRUNE_Z = -1.0
PRUNE_EDGE_FRAC = 0.05
MAX_NEW_NEURONS = 4
MAX_NEW_EDGES = 64
SEED_EDGES = 4


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fashion_loaders(seed: int):
    task = TaskContext(make_identity_transform())
    return build_mnist_loaders(
        task,
        root=str(_REPO / "data" / "fashion_mnist"),
        batch_size=128,
        num_workers=0,
        seed=seed,
        probe_size=1024,
        dataset="fashion_mnist",
    )


def make_cfg(arm: str) -> PlasticityConfig:
    """Map arm → PlasticityConfig via product knobs (no orchestrator fork)."""
    grow = arm in ("full", "grow_only")
    prune = arm in ("full", "prune_only")
    return PlasticityConfig(
        radius=RADIUS,
        target_rates=dict(FASHION_R),
        tau=TAU,
        threshold_kind="fixed",
        mu=0.999,
        # Prune off: fraction 0 skips edge prune; z cut ≪ any finite z skips neurons.
        prune_edge_fraction=PRUNE_EDGE_FRAC if prune else 0.0,
        prune_neuron_z=PRUNE_Z if prune else -1.0e9,
        # Grow off: zero caps.
        max_new_edges_per_layer=MAX_NEW_EDGES if grow else 0,
        max_new_neurons_per_layer=MAX_NEW_NEURONS if grow else 0,
        seed_edges_per_new_neuron=SEED_EDGES if grow else 0,
        newborn_grace_steps=3,
        newborn_ema_init="layer_target",
        newborn_birth_bias=0.5,
        newborn_birth_bias_steps=3,
        newborn_weight_init="gap_imprinted",
        newborn_parent_noise=0.0,
        newborn_copy_outgoing=True,
        newborn_parent_coord_jitter=0.05,
        newborn_lr_mult=1.0,
        neuron_growth_trigger="gated_novelty",
        novelty_k_min=K_MIN,
        novelty_rank_fraction=1.0,
    )


def swing_stats(accs: list[float]) -> dict[str, Any]:
    sw = [abs(accs[i] - accs[i - 1]) * 100 for i in range(1, len(accs))]
    if not sw:
        return {
            "abs_swings_pp": [],
            "sum_abs_swing_pp": 0.0,
            "max_abs_swing_pp": 0.0,
            "mean_abs_swing_pp": 0.0,
            "median_abs_swing_pp": 0.0,
            "n_swings_gt_5pp": 0,
            "n_swings_gt_10pp": 0,
            "acc_range_pp": 0.0,
        }
    return {
        "abs_swings_pp": sw,
        "sum_abs_swing_pp": float(sum(sw)),
        "max_abs_swing_pp": float(max(sw)),
        "mean_abs_swing_pp": float(np.mean(sw)),
        "median_abs_swing_pp": float(np.median(sw)),
        "n_swings_gt_5pp": int(sum(1 for s in sw if s > 5)),
        "n_swings_gt_10pp": int(sum(1 for s in sw if s > 10)),
        "acc_range_pp": float((max(accs) - min(accs)) * 100),
        "rms_swing_pp": float(np.sqrt(np.mean(np.square(sw)))),
        "late_half_mean_abs_pp": float(np.mean(sw[len(sw) // 2 :])),
        "early_half_mean_abs_pp": float(np.mean(sw[: max(1, len(sw) // 2)])),
    }


def dense_params(h1: int, h2: int, h3: int, c: int = 10, din: int = 784) -> int:
    return h1 * din + h2 * h1 + h3 * h2 + c * h3 + h1 + h2 + h3 + c


def run_arm(arm: str, seed: int, epochs: int) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    loaders = fashion_loaders(seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = make_cfg(arm)
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    global_step = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(epochs):
        metrics = train_epoch_with_signals(
            net,
            loaders["train_loader"],
            opt,
            signal,
            orch,
            device=device,
            global_step=global_step,
        )
        global_step = int(metrics["global_step"])
        # Always collect gated probe so under-floor is logged even if growth off
        # (frozen/prune_only still need a valid context if trigger is gated —
        # growth path no-ops via caps=0).
        gated = collect_gated_novelty_probe(
            net,
            loaders["probe_loader"],
            device=device,
            radius=cfg.radius,
            k_min=cfg.novelty_k_min,
        )
        scores = {k: v.detach().cpu() for k, v in signal.neuron_scores().items()}
        if "in" not in scores:
            scores["in"] = torch.zeros(784)
        slog = orch.step(
            net, scores, optimizer=opt, epoch=epoch, signal=signal, gated_novelty=gated
        )
        vm = evaluate(net, loaders["test_loader"], device=device)
        H = net.hidden_counts()
        active = int(net.active_parameter_count())
        edge_tot = sum(v["total"] for v in net.edge_counts(by_origin=True).values())
        row = {
            "epoch": epoch,
            "test_acc": float(vm["acc"]),
            "test_loss": float(vm["loss"]),
            "train_loss": float(metrics.get("loss", float("nan"))),
            "neurons_grown": int(slog.neurons_grown),
            "neurons_pruned": int(slog.neurons_pruned),
            "edges_pruned_competitive": int(slog.edges_pruned_competitive),
            "edges_grown": int(getattr(slog, "edges_grown", 0) or 0),
            "hidden_counts": H,
            "H_sum": int(sum(H.values())),
            "active_params": active,
            "edge_total": int(edge_tot),
            "fill_active_over_dense_H": active
            / max(1, dense_params(H["h1"], H["h2"], H["h3"])),
        }
        rows.append(row)
        print(
            f"  [{arm:10s} s{seed}] ep{epoch:02d} acc={vm['acc']:.4f} "
            f"loss={vm['loss']:.3f} g={row['neurons_grown']} p={row['neurons_pruned']} "
            f"e_pr={row['edges_pruned_competitive']} H={H} P={active}",
            flush=True,
        )

    accs = [r["test_acc"] for r in rows]
    sw = swing_stats(accs)
    return {
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "target_rates": dict(FASHION_R),
        "grow_enabled": arm in ("full", "grow_only"),
        "prune_enabled": arm in ("full", "prune_only"),
        "cfg_snapshot": {
            "prune_edge_fraction": cfg.prune_edge_fraction,
            "prune_neuron_z": cfg.prune_neuron_z,
            "max_new_neurons_per_layer": cfg.max_new_neurons_per_layer,
            "max_new_edges_per_layer": cfg.max_new_edges_per_layer,
        },
        "trajectory": rows,
        "acc_trajectory": accs,
        "swing": sw,
        "totals": {
            "neurons_grown": int(sum(r["neurons_grown"] for r in rows)),
            "neurons_pruned": int(sum(r["neurons_pruned"] for r in rows)),
            "edges_pruned_competitive": int(
                sum(r["edges_pruned_competitive"] for r in rows)
            ),
        },
        "final_acc": float(accs[-1]),
        "peak_acc": float(max(accs)),
        "final_active_params": int(rows[-1]["active_params"]),
        "final_hidden": rows[-1]["hidden_counts"],
        "final_fill": float(rows[-1]["fill_active_over_dense_H"]),
    }


def classify_isolation(by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map 2×2 swing pattern → growth / prune / interaction / neither."""
    # Use max abs swing + sum abs as co-primary; thrash bar: max>10 or sum>40
    def thrash(arm: str) -> bool:
        sw = by_arm[arm]["swing"]
        return sw["max_abs_swing_pp"] >= 10.0 or sw["sum_abs_swing_pp"] >= 40.0

    def mild(arm: str) -> bool:
        sw = by_arm[arm]["swing"]
        return sw["max_abs_swing_pp"] < 5.0 and sw["sum_abs_swing_pp"] < 20.0

    flags = {a: thrash(a) for a in by_arm}
    mild_f = {a: mild(a) for a in by_arm}
    full, grow, prune, frozen = (
        flags["full"],
        flags["grow_only"],
        flags["prune_only"],
        flags["frozen"],
    )

    # Pre-registered decision table
    if frozen and not mild_f["frozen"]:
        verdict = "CONTROL_FAILED"
        reading = (
            "Frozen arm itself thrashes — setup bug or non-structure noise; "
            "do not interpret grow/prune cells."
        )
    elif not full and not grow and not prune:
        verdict = "NO_ARM_THRASHES"
        reading = (
            "No arm exceeds thrash bar under this seed/horizon — thrash not "
            "reproduced; re-check seed or extend epochs."
        )
    elif grow and not prune:
        verdict = "GROWTH_SUFFICIENT"
        reading = (
            "Grow-only thrashes; prune-only does not. Continuous gap-imprinted "
            "insertion (rate/ caps) is sufficient for residual instability. "
            "Prune constants are not required for the thrash phenotype."
        )
    elif prune and not grow:
        verdict = "PRUNE_SUFFICIENT"
        reading = (
            "Prune-only thrashes; grow-only does not. Edge fraction and/or neuron "
            "z-cutoff under Fashion are the natural re-derivation targets."
        )
    elif grow and prune:
        # Both alone thrash
        if full:
            verdict = "GROWTH_AND_PRUNE_EACH_SUFFICIENT"
            reading = (
                "Either structure edit alone produces thrash. Both pathways are "
                "destabilizing on Fashion; product stack is double-loaded."
            )
        else:
            verdict = "EACH_ALONE_BUT_FULL_QUIET"
            reading = "Unexpected: alone thrash but full quiet — check rate interactions."
    elif full and not grow and not prune:
        verdict = "INTERACTION_ONLY"
        reading = (
            "Only full stack thrashes; grow-only and prune-only stay mild. "
            "Instability is an interaction of growth and pruning, not either alone."
        )
    elif full and grow and not prune:
        verdict = "GROWTH_SUFFICIENT"
        reading = (
            "Full and grow-only thrash; prune-only mild → growth is sufficient "
            "(full reproduces grow path)."
        )
    elif full and prune and not grow:
        verdict = "PRUNE_SUFFICIENT"
        reading = (
            "Full and prune-only thrash; grow-only mild → prune is sufficient."
        )
    else:
        verdict = "MIXED_PATTERN"
        reading = "Pattern does not match a clean cell of the pre-registered table."

    # Magnitude ranking for nuance
    rank = sorted(
        by_arm.keys(),
        key=lambda a: (
            by_arm[a]["swing"]["max_abs_swing_pp"],
            by_arm[a]["swing"]["sum_abs_swing_pp"],
        ),
        reverse=True,
    )
    return {
        "thrash_flags": flags,
        "mild_flags": mild_f,
        "thrash_bar": {
            "max_abs_swing_pp": 10.0,
            "sum_abs_swing_pp": 40.0,
            "mild_max": 5.0,
            "mild_sum": 20.0,
        },
        "verdict": verdict,
        "reading": reading,
        "rank_by_max_then_sum": rank,
        "next_step_hint": {
            "GROWTH_SUFFICIENT": (
                "Probe birth rate (max_new_neurons_per_layer), gap_imprint scale, "
                "or birth cadence before 20ep efficiency claims."
            ),
            "PRUNE_SUFFICIENT": (
                "Re-derive or ablate prune_edge_fraction and prune_neuron_z on Fashion."
            ),
            "INTERACTION_ONLY": (
                "Study grow→prune timing (grace, judgment of newborns) rather than "
                "either constant in isolation."
            ),
            "GROWTH_AND_PRUNE_EACH_SUFFICIENT": (
                "Both levers hurt; product stack needs both softened or gated on Fashion."
            ),
            "CONTROL_FAILED": "Debug frozen arm first.",
            "NO_ARM_THRASHES": "Reproduce thrash (seed/horizon) before isolation claims.",
            "MIXED_PATTERN": "Inspect per-arm trajectories qualitatively.",
            "EACH_ALONE_BUT_FULL_QUIET": "Unexpected; re-run.",
        }.get(verdict, ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument(
        "--arms",
        type=str,
        default="full,grow_only,prune_only,frozen",
        help="Comma-separated subset of arms",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("full", "grow_only", "prune_only", "frozen"):
            raise SystemExit(f"unknown arm {a!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {
        "protocol": {
            "dataset": "fashion_mnist",
            "target_rates": FASHION_R,
            "epochs": args.epochs,
            "seeds": seeds,
            "arms": arms,
            "stack_base": "fixed+gated_novelty+gap_imprinted",
            "train": "sgd lr=0.01 mom=0.9 wd=1e-4 batch=128",
        },
        "by_seed": {},
    }

    for seed in seeds:
        print(f"\n======== seed {seed} ========", flush=True)
        by_arm: dict[str, dict[str, Any]] = {}
        for arm in arms:
            print(f"\n=== arm={arm} ===", flush=True)
            res = run_arm(arm, seed=seed, epochs=args.epochs)
            by_arm[arm] = res
            path = OUT / f"seed_{seed}" / f"{arm}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, indent=2))
            print(
                f"  -> max_swing={res['swing']['max_abs_swing_pp']:.1f}pp "
                f"sum={res['swing']['sum_abs_swing_pp']:.1f} "
                f"n>5={res['swing']['n_swings_gt_5pp']} "
                f"final_acc={res['final_acc']:.4f} "
                f"grown={res['totals']['neurons_grown']} "
                f"pruned={res['totals']['neurons_pruned']}",
                flush=True,
            )

        isolation = classify_isolation(by_arm) if set(arms) >= {
            "full",
            "grow_only",
            "prune_only",
            "frozen",
        } else {"verdict": "INCOMPLETE_ARMS", "reading": "Need all four arms."}

        # Compact comparison table
        table = []
        for arm in arms:
            r = by_arm[arm]
            table.append(
                {
                    "arm": arm,
                    "max_abs_swing_pp": r["swing"]["max_abs_swing_pp"],
                    "sum_abs_swing_pp": r["swing"]["sum_abs_swing_pp"],
                    "mean_abs_swing_pp": r["swing"]["mean_abs_swing_pp"],
                    "n_swings_gt_5pp": r["swing"]["n_swings_gt_5pp"],
                    "n_swings_gt_10pp": r["swing"]["n_swings_gt_10pp"],
                    "acc_range_pp": r["swing"]["acc_range_pp"],
                    "final_acc": r["final_acc"],
                    "peak_acc": r["peak_acc"],
                    "neurons_grown": r["totals"]["neurons_grown"],
                    "neurons_pruned": r["totals"]["neurons_pruned"],
                    "edges_pruned": r["totals"]["edges_pruned_competitive"],
                    "final_P": r["final_active_params"],
                    "final_fill": r["final_fill"],
                    "acc_trajectory": r["acc_trajectory"],
                }
            )

        seed_block = {
            "seed": seed,
            "table": table,
            "isolation": isolation,
            "by_arm": {a: by_arm[a] for a in arms},
        }
        all_results["by_seed"][str(seed)] = seed_block
        (OUT / f"seed_{seed}" / "summary.json").write_text(
            json.dumps(
                {"seed": seed, "table": table, "isolation": isolation},
                indent=2,
            )
        )

    # Headline from primary seed 0 if present
    primary = all_results["by_seed"].get("0") or next(
        iter(all_results["by_seed"].values())
    )
    all_results["headline"] = {
        "seed": primary["seed"],
        "verdict": primary["isolation"].get("verdict"),
        "reading": primary["isolation"].get("reading"),
        "next_step_hint": primary["isolation"].get("next_step_hint"),
        "table": primary["table"],
    }
    (OUT / "summary.json").write_text(json.dumps(all_results, indent=2))

    # Markdown report
    h = all_results["headline"]
    lines = [
        "# Fashion grow/prune 2×2 isolation",
        "",
        f"**Verdict (seed {h['seed']}):** `{h['verdict']}`",
        "",
        h.get("reading") or "",
        "",
        f"**Next:** {h.get('next_step_hint') or '—'}",
        "",
        "## Protocol",
        "",
        f"- Fashion-MNIST, r\\*_ℓ = {FASHION_R}, fixed + gated_novelty + gap_imprinted base",
        f"- {args.epochs} epochs, SGD lr=0.01 mom=0.9 wd=1e-4 batch=128",
        f"- Seeds: {seeds}",
        "",
        "## Comparison table",
        "",
        "| Arm | max\\|Δ\\| pp | sum\\|Δ\\| | #Δ>5 | #Δ>10 | final acc | peak | grown | pruned | e_pruned | final P | fill |",
        "|-----|------------:|---------:|-----:|------:|----------:|-----:|------:|-------:|---------:|--------:|-----:|",
    ]
    for row in h["table"]:
        lines.append(
            f"| {row['arm']} | {row['max_abs_swing_pp']:.1f} | {row['sum_abs_swing_pp']:.1f} | "
            f"{row['n_swings_gt_5pp']} | {row['n_swings_gt_10pp']} | {row['final_acc']:.3f} | "
            f"{row['peak_acc']:.3f} | {row['neurons_grown']} | {row['neurons_pruned']} | "
            f"{row['edges_pruned']} | {row['final_P']} | {row['final_fill']:.3f} |"
        )
    lines += ["", "## Acc trajectories", ""]
    for row in h["table"]:
        traj = " → ".join(f"{a:.3f}" for a in row["acc_trajectory"])
        lines.append(f"- **{row['arm']}:** `{traj}`")
    lines += [
        "",
        "## Pre-registered thrash bar",
        "",
        "- thrash: max\\|Δ\\| ≥ 10pp **or** sum\\|Δ\\| ≥ 40pp",
        "- mild: max\\|Δ\\| < 5pp **and** sum\\|Δ\\| < 20pp",
        "",
        "Artifacts under this directory: `seed_*/{arm}.json`, `summary.json`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps(h, indent=2), flush=True)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
