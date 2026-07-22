#!/usr/bin/env python3
"""Fashion-MNIST dense slope scout + plastic Pareto (process lessons from MNIST).

Process (built in from the start — not reactive follow-ups):

1. **Wide dense grid** (not a naive port of MNIST tags) to find where Fashion-MNIST
   still has slope, before locking a comparison grid.
2. **Per-cell seed spread** reported immediately with the dense scout.
3. Plastic fixed + sliding 20-ep holds on Fashion-MNIST under **Fashion-calibrated
   r*_ℓ** (not MNIST's 0.77/1.02/1.28 — see fashion_mnist_calibration/).
4. Score cost-matched Pareto with δ=1.0 pp, both arms, mid-horizon caveat.

**Calibration required:** run ``scripts/run_fashion_mnist_calibration.py`` first
(or rely on locked rates below from that probe). Prior plastic runs that reused
MNIST target_rates are INVALID for product claims.

Usage:
  python scripts/run_fashion_mnist_pareto.py
  python scripts/run_fashion_mnist_pareto.py --skip-dense   # reuse dense scout
  python scripts/run_fashion_mnist_pareto.py --skip-plastic # score only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet  # noqa: E402
from neuronca.models.mlp import MLPReasoningNet  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate, train_epoch  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "fashion_mnist_pareto"
DENSE_ROOT = OUT / "dense_scout_grid"
# Recalibrated plastic outputs (Fashion r*); uncalibrated MNIST-r* runs live under plastic/
PLASTIC_ROOT = OUT / "plastic_calibrated"
DELTA = 0.01
NEAR_BAND = 0.005
EPOCHS = 20
SEEDS = (0, 1, 2)

# Locked from results/phase1a_diagnostics/fashion_mnist_calibration/target_rate_probe.json
# Do NOT use MNIST (0.77, 1.02, 1.28) on Fashion — h2/h3 are ~22–29% lower.
FASHION_TARGET_RATES = {"h1": 0.71, "h2": 0.80, "h3": 0.91}
# Product birth rate after RATE_CAP_RESOLVES_THRASH (cap=4 thrashes; cap=1 mild).
FASHION_MAX_NEW_NEURONS = 1


def _cfg(threshold_kind: str) -> PlasticityConfig:
    """Fashion-calibrated full stack (dataset-specific r*_ℓ + thrash-safe rate)."""
    return PlasticityConfig(
        radius=0.35,
        target_rates=dict(FASHION_TARGET_RATES),
        tau=0.5,
        threshold_kind=str(threshold_kind),
        mu=0.999,
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=FASHION_MAX_NEW_NEURONS,
        seed_edges_per_new_neuron=4,
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
        novelty_k_min=11,
        novelty_rank_fraction=1.0,
    )

# Wide scout: from very small through past MNIST plateau into high capacity.
# Not centered on MNIST plastic P (~88k) alone.
WIDE_SCOUT: list[tuple[str, tuple[int, int, int]]] = [
    ("xs_32", (32, 16, 8)),
    ("s_48", (48, 24, 12)),
    ("s_64", (64, 32, 16)),
    ("s_80", (80, 40, 20)),
    ("m_96", (96, 48, 24)),
    ("m_112", (112, 56, 28)),
    ("m_128", (128, 64, 32)),
    ("m_160", (160, 80, 40)),
    ("l_192", (192, 96, 48)),
    ("l_224", (224, 112, 56)),
    ("l_256", (256, 128, 64)),
    ("xl_320", (320, 160, 80)),
    ("xl_384", (384, 192, 96)),
    ("xxl_512", (512, 256, 128)),
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dense_params(h1: int, h2: int, h3: int, c: int = 10, din: int = 784) -> int:
    return h1 * din + h2 * h1 + h3 * h2 + c * h3 + h1 + h2 + h3 + c


def fashion_loaders(seed: int, batch_size: int = 128):
    task = TaskContext(make_identity_transform())
    return build_mnist_loaders(
        task,
        root=str(_REPO / "data" / "fashion_mnist"),
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
        probe_size=1024,
        dataset="fashion_mnist",
    )


def train_dense_cell(
    tag: str, hidden: tuple[int, int, int], seed: int, epochs: int
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    net = MLPReasoningNet(input_dim=784, hidden=hidden, num_classes=10).to(device)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    loaders = fashion_loaders(seed)
    history = []
    for ep in range(epochs):
        m = train_epoch(net, loaders["train_loader"], opt, device=device)
        history.append({"epoch": ep, **m})
    test_m = evaluate(net, loaders["test_loader"], device=device)
    n_params = int(sum(p.numel() for p in net.parameters()))
    return {
        "tag": tag,
        "seed": seed,
        "hidden": list(hidden),
        "params": n_params,
        "test_acc": float(test_m["acc"]),
        "test_loss": float(test_m["loss"]),
        "train_history": history,
    }


def run_dense_scout(epochs: int = EPOCHS) -> list[dict[str, Any]]:
    DENSE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for tag, hidden in WIDE_SCOUT:
        runs = []
        for seed in SEEDS:
            path = DENSE_ROOT / tag / f"seed_{seed}.json"
            if path.is_file():
                print(f"[dense skip] {tag} seed {seed}", flush=True)
                row = json.loads(path.read_text())
            else:
                print(f"[dense train] {tag} H={hidden} seed={seed}", flush=True)
                row = train_dense_cell(tag, hidden, seed, epochs)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            runs.append(row)
            print(f"  -> P={row['params']} acc={row['test_acc']:.4f}", flush=True)
        accs = [r["test_acc"] for r in runs]
        params = runs[0]["params"]
        rows.append(
            {
                "tag": tag,
                "hidden": list(hidden),
                "mean_params": float(params),
                "mean_test_acc": float(np.mean(accs)),
                "sample_std_pp": float(np.std(accs) * 100),
                "range_pp": float((max(accs) - min(accs)) * 100),
                "per_seed_test_acc": {str(r["seed"]): r["test_acc"] for r in runs},
            }
        )
        print(
            f"  mean {tag}: P={params} acc={rows[-1]['mean_test_acc']:.4f} "
            f"range={rows[-1]['range_pp']:.2f}pp std={rows[-1]['sample_std_pp']:.2f}pp",
            flush=True,
        )
    rows.sort(key=lambda r: r["mean_params"])
    return rows


def analyze_slope(dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Find where acc still rises with params (slope signal)."""
    # Consecutive log-param slope (pp per decade of params)
    slopes = []
    for i in range(len(dense_rows) - 1):
        a0, a1 = dense_rows[i], dense_rows[i + 1]
        dlog = math.log10(a1["mean_params"]) - math.log10(a0["mean_params"])
        dacc_pp = (a1["mean_test_acc"] - a0["mean_test_acc"]) * 100
        slopes.append(
            {
                "from": a0["tag"],
                "to": a1["tag"],
                "params_from": a0["mean_params"],
                "params_to": a1["mean_params"],
                "acc_from": a0["mean_test_acc"],
                "acc_to": a1["mean_test_acc"],
                "delta_acc_pp": dacc_pp,
                "pp_per_log10_params": dacc_pp / dlog if dlog else None,
            }
        )

    # "Has slope" if consecutive gain > max(0.3pp, 2× local seed range proxy)
    slope_edges = [
        s
        for s in slopes
        if s["delta_acc_pp"] > 0.3
    ]
    # Plateau: consecutive |delta| < 0.25pp and both means high-ish
    plateau_edges = [s for s in slopes if abs(s["delta_acc_pp"]) < 0.25]

    # Largest consecutive gains
    top_gains = sorted(slopes, key=lambda s: s["delta_acc_pp"], reverse=True)[:5]

    # Recommend comparison grid: cover from clear slope region through saturation
    # Pick cells at: min scout, points around steepest gains, mid, high
    mean_std = float(np.median([r["sample_std_pp"] for r in dense_rows]))
    mean_range = float(np.median([r["range_pp"] for r in dense_rows]))

    # Find first param where further +1 cell gains < 0.3pp for two consecutive steps
    sat_start = None
    for i in range(len(slopes) - 1):
        if slopes[i]["delta_acc_pp"] < 0.3 and slopes[i + 1]["delta_acc_pp"] < 0.3:
            sat_start = slopes[i]["params_from"]
            break

    return {
        "consecutive_slopes": slopes,
        "slope_edges_gt_0p3pp": slope_edges,
        "plateau_edges_lt_0p25pp": plateau_edges,
        "top_gains": top_gains,
        "seed_noise_summary": {
            "median_cell_std_pp": mean_std,
            "median_cell_range_pp": mean_range,
            "max_cell_range_pp": max(r["range_pp"] for r in dense_rows),
            "max_cell_std_pp": max(r["sample_std_pp"] for r in dense_rows),
        },
        "saturation_hint_params": sat_start,
        "reading": _slope_reading(dense_rows, slopes, sat_start, mean_range),
    }


def _slope_reading(rows, slopes, sat_start, median_range) -> list[str]:
    lines = []
    if not rows:
        return ["empty dense scout"]
    lines.append(
        f"Fashion dense acc spans {rows[0]['mean_test_acc']:.3f} (P={rows[0]['mean_params']:.0f}) "
        f"→ {rows[-1]['mean_test_acc']:.3f} (P={rows[-1]['mean_params']:.0f})."
    )
    lines.append(
        f"Dense seed noise: median range {median_range:.2f} pp "
        f"(compare MNIST dense median ~0.12 pp)."
    )
    gains = [s["delta_acc_pp"] for s in slopes]
    lines.append(
        f"Consecutive cell gains: min {min(gains):+.2f} / max {max(gains):+.2f} pp."
    )
    if sat_start:
        lines.append(
            f"Saturation hint: consecutive <0.3pp gains starting near P≈{sat_start:.0f}."
        )
    else:
        lines.append("No clear two-step plateau; slope may persist into high capacity.")
    # vs MNIST lesson
    lines.append(
        "Unlike MNIST (flat by ~81k at ~98%), Fashion should show a longer slope "
        "if means keep rising; lock comparison grid to cover rising region + plateau edge."
    )
    return lines


def pick_comparison_grid(dense_rows: list[dict[str, Any]], slope: dict[str, Any]) -> list[dict[str, Any]]:
    """Lock a subset spanning slope + plateau for plastic comparison."""
    # Always include ends + every other cell, plus cells around top gains
    tags_keep = set()
    tags_keep.add(dense_rows[0]["tag"])
    tags_keep.add(dense_rows[-1]["tag"])
    # every 2nd
    for i, r in enumerate(dense_rows):
        if i % 2 == 0:
            tags_keep.add(r["tag"])
    # endpoints of top 3 gains
    for g in slope["top_gains"][:3]:
        tags_keep.add(g["from"])
        tags_keep.add(g["to"])
    # ensure m_96, m_128, l_256 present if in scout (familiar anchors)
    for t in ("m_96", "m_128", "l_256", "s_64", "xl_320"):
        if any(r["tag"] == t for r in dense_rows):
            tags_keep.add(t)

    locked = [r for r in dense_rows if r["tag"] in tags_keep]
    locked.sort(key=lambda r: r["mean_params"])
    return locked


def run_plastic_arm(
    *,
    threshold_kind: str,
    seed: int,
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    loaders = fashion_loaders(seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = _cfg(threshold_kind)
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    global_step = 0
    structure_trace = []
    val_hist = []
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
        structure_trace.append(
            {
                "epoch": epoch,
                "neurons_grown": int(slog.neurons_grown),
                "neurons_pruned": int(slog.neurons_pruned),
                "hidden_counts": net.hidden_counts(),
                "edge_counts": net.edge_counts(by_origin=True),
            }
        )
        vm = evaluate(net, loaders["test_loader"], device=device)
        val_hist.append({"epoch": float(epoch), **vm})
        print(
            f"  [{threshold_kind} s{seed}] ep{epoch:02d} "
            f"grown={slog.neurons_grown} pruned={slog.neurons_pruned} "
            f"acc={vm['acc']:.4f} H={net.hidden_counts()}",
            flush=True,
        )
    active = int(net.active_parameter_count())
    h = net.hidden_counts()
    return {
        "threshold_kind": threshold_kind,
        "seed": seed,
        "epochs": epochs,
        "dataset": "fashion_mnist",
        "target_rates": dict(FASHION_TARGET_RATES),
        "calibration": "fashion_mnist_calibration/target_rate_probe.json",
        "hidden_counts_final": h,
        "H_sum": int(sum(h.values())),
        "active_params": active,
        "edge_counts_final": net.edge_counts(by_origin=True),
        "test_acc": float(val_hist[-1]["acc"]),
        "val_history": val_hist,
        "structure_trace": structure_trace,
    }


def active_params_from_counts(hidden: dict, edges: dict) -> int:
    n_w = sum(edges[k]["total"] for k in edges)
    n_b = hidden["h1"] + hidden["h2"] + hidden["h3"] + 10
    return n_w + n_b


def interpolate_ab(grid: list[dict[str, Any]], p_target: float) -> dict[str, Any]:
    ps = [r["mean_params"] for r in grid]
    accs = [r["mean_test_acc"] for r in grid]
    if p_target < ps[0] or p_target > ps[-1]:
        # clamp to nearest for scoring with annotation
        if p_target < ps[0]:
            return {
                "A_b": accs[0],
                "method": "clamp_low",
                "bracket": [grid[0]["tag"], grid[0]["tag"]],
                "warning": f"P_p={p_target} below grid min {ps[0]}",
            }
        return {
            "A_b": accs[-1],
            "method": "clamp_high",
            "bracket": [grid[-1]["tag"], grid[-1]["tag"]],
            "warning": f"P_p={p_target} above grid max {ps[-1]}",
        }
    for i in range(len(ps) - 1):
        if ps[i] <= p_target <= ps[i + 1]:
            log_p = math.log(p_target)
            t = (log_p - math.log(ps[i])) / (math.log(ps[i + 1]) - math.log(ps[i]))
            a_b = accs[i] + t * (accs[i + 1] - accs[i])
            return {
                "A_b": float(a_b),
                "method": "log_params_linear",
                "bracket": [grid[i]["tag"], grid[i + 1]["tag"]],
                "params_bracket": [ps[i], ps[i + 1]],
                "t": t,
            }
    raise RuntimeError("interp failed")


def score_arm(
    name: str, runs: list[dict[str, Any]], dense_grid: list[dict[str, Any]]
) -> dict[str, Any]:
    accs = [r["test_acc"] for r in runs]
    ps = [r["active_params"] for r in runs]
    a_p = float(np.mean(accs))
    p_p = float(np.mean(ps))
    interp = interpolate_ab(dense_grid, p_p)
    a_b = float(interp["A_b"])
    thr = a_b - DELTA
    margin = a_p - thr
    dominated = [
        {
            "tag": r["tag"],
            "mean_params": r["mean_params"],
            "mean_test_acc": r["mean_test_acc"],
            "acc_minus_plastic_pp": (r["mean_test_acc"] - a_p) * 100,
        }
        for r in dense_grid
        if r["mean_params"] <= p_p + 1 and r["mean_test_acc"] > a_p + DELTA
    ]
    return {
        "arm": name,
        "A_p": a_p,
        "P_p": p_p,
        "A_b_at_P_p": a_b,
        "threshold": thr,
        "margin_pp": margin * 100,
        "pass": a_p >= thr,
        "verdict": "PASS" if a_p >= thr else "FAIL_ACCURACY_SHORTFALL",
        "near_boundary_delta_uncertain": abs(margin) < NEAR_BAND,
        "interpolation": interp,
        "dominated_by_smaller_dense": dominated,
        "test_acc_range_pp": (max(accs) - min(accs)) * 100,
        "test_acc_std_pp": float(np.std(accs) * 100),
        "per_seed": [
            {
                "seed": r["seed"],
                "active_params": r["active_params"],
                "test_acc": r["test_acc"],
                "H_sum": r["H_sum"],
                "hidden": r["hidden_counts_final"],
            }
            for r in runs
        ],
    }


def plot_dense_scout(dense_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ps = [r["mean_params"] for r in dense_rows]
    accs = [r["mean_test_acc"] for r in dense_rows]
    ranges = [r["range_pp"] for r in dense_rows]
    ax = axes[0]
    ax.errorbar(
        ps,
        accs,
        yerr=[r / 200 for r in ranges],  # half-range as visual bar
        fmt="o-",
        color="#1565c0",
        capsize=3,
        lw=1.5,
    )
    for r in dense_rows:
        ax.annotate(r["tag"], (r["mean_params"], r["mean_test_acc"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Dense params")
    ax.set_ylabel("Test acc (mean ± half seed-range)")
    ax.set_title("Fashion-MNIST dense scout (find slope)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(dense_rows)), ranges, color="#90caf9", edgecolor="k", lw=0.3)
    ax.set_xticks(range(len(dense_rows)))
    ax.set_xticklabels([r["tag"] for r in dense_rows], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Seed range (pp)")
    ax.set_title("Dense per-cell seed spread (built-in)")
    ax.axhline(float(np.median(ranges)), color="gray", ls="--", lw=1, label="median")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    dense_rows: list[dict[str, Any]],
    fixed_score: dict,
    sliding_score: dict,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ps = [r["mean_params"] for r in dense_rows]
    accs = [r["mean_test_acc"] for r in dense_rows]
    ax.plot(ps, accs, "o-", color="#616161", label="dense scout/grid", lw=1.5)
    ax.scatter(
        [fixed_score["P_p"]],
        [fixed_score["A_p"]],
        s=140,
        c="#c62828",
        marker="*",
        zorder=5,
        label="plastic fixed",
    )
    ax.scatter(
        [sliding_score["P_p"]],
        [sliding_score["A_p"]],
        s=140,
        c="#1565c0",
        marker="*",
        zorder=5,
        label="plastic sliding",
    )
    for s in fixed_score["per_seed"]:
        ax.scatter([s["active_params"]], [s["test_acc"]], c="#ef9a9a", s=35, zorder=4)
    for s in sliding_score["per_seed"]:
        ax.scatter([s["active_params"]], [s["test_acc"]], c="#90caf9", s=35, marker="s", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        f"Fashion-MNIST Pareto @20ep — fixed {fixed_score['verdict']} | "
        f"sliding {sliding_score['verdict']}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--skip-plastic", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs)

    # --- 1) Wide dense scout + seed spreads ---
    if args.skip_dense and (OUT / "dense_scout_summary.json").is_file():
        dense_rows = json.loads((OUT / "dense_scout_summary.json").read_text())
        print("[dense] loaded existing scout", flush=True)
    else:
        dense_rows = run_dense_scout(epochs=epochs)
        (OUT / "dense_scout_summary.json").write_text(
            json.dumps(dense_rows, indent=2), encoding="utf-8"
        )

    slope = analyze_slope(dense_rows)
    (OUT / "dense_slope_analysis.json").write_text(
        json.dumps(slope, indent=2), encoding="utf-8"
    )
    # Seed-spread artifact first-class
    seed_spread = {
        "dataset": "fashion_mnist",
        "purpose": "Built-in dense seed-spread check (not a reactive follow-up).",
        "per_cell": dense_rows,
        "summary": slope["seed_noise_summary"],
        "slope_reading": slope["reading"],
    }
    (OUT / "dense_grid_seed_spread.json").write_text(
        json.dumps(seed_spread, indent=2), encoding="utf-8"
    )
    plot_dense_scout(dense_rows, OUT / "dense_scout_curve.png")

    print("\n=== DENSE SLOPE / SPREAD ===", flush=True)
    for line in slope["reading"]:
        print(" ", line, flush=True)
    print(
        " seed noise:",
        json.dumps(slope["seed_noise_summary"]),
        flush=True,
    )

    comparison_grid = pick_comparison_grid(dense_rows, slope)
    (OUT / "comparison_grid_locked.json").write_text(
        json.dumps(
            {
                "note": "Locked after dense scout; spans slope + high capacity.",
                "cells": comparison_grid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "\n[lock] comparison grid tags:",
        [c["tag"] for c in comparison_grid],
        flush=True,
    )

    # --- 2) Plastic arms ---
    plastic: dict[str, list[dict[str, Any]]] = {"fixed": [], "sliding": []}
    for kind in ("fixed", "sliding"):
        for seed in SEEDS:
            path = PLASTIC_ROOT / kind / f"seed_{seed}.json"
            if args.skip_plastic and path.is_file():
                print(f"[plastic skip] {kind} seed {seed}", flush=True)
                plastic[kind].append(json.loads(path.read_text()))
                continue
            if path.is_file() and not args.skip_plastic:
                # allow resume
                print(f"[plastic skip existing] {kind} seed {seed}", flush=True)
                plastic[kind].append(json.loads(path.read_text()))
                continue
            print(f"\n=== plastic {kind} seed {seed} ===", flush=True)
            row = run_plastic_arm(threshold_kind=kind, seed=seed, epochs=epochs)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            plastic[kind].append(row)

    # Use full dense scout for interp (wider than locked subset is fine)
    fixed_score = score_arm("fixed", plastic["fixed"], dense_rows)
    sliding_score = score_arm("sliding", plastic["sliding"], dense_rows)

    pf, af = fixed_score["P_p"], fixed_score["A_p"]
    ps, ass = sliding_score["P_p"], sliding_score["A_p"]
    if (ps <= pf and ass >= af) and (ps < pf or ass > af):
        svf = "SLIDING_DOMINATES_FIXED"
    elif (pf <= ps and af >= ass) and (pf < ps or af > ass):
        svf = "FIXED_DOMINATES_SLIDING"
    elif ps < pf and ass < af:
        svf = "TRADEOFF_SLIDING_CHEAPER_LOWER_ACC"
    else:
        svf = "TRADEOFF_OR_SIMILAR"

    both = fixed_score["pass"] and sliding_score["pass"]
    headline = (
        "FASHION_PARETO_BOTH_PASS"
        if both
        else (
            "FASHION_PARETO_BOTH_FAIL"
            if not fixed_score["pass"] and not sliding_score["pass"]
            else (
                "FASHION_PARETO_FIXED_PASS_SLIDING_FAIL"
                if fixed_score["pass"]
                else "FASHION_PARETO_SLIDING_PASS_FIXED_FAIL"
            )
        )
    )

    # Dominance vs seed noise
    for sc in (fixed_score, sliding_score):
        for d in sc["dominated_by_smaller_dense"]:
            cell = next(r for r in dense_rows if r["tag"] == d["tag"])
            d["dense_range_pp"] = cell["range_pp"]
            d["gap_over_dense_range"] = (
                d["acc_minus_plastic_pp"] / cell["range_pp"]
                if cell["range_pp"] > 0
                else None
            )

    report = {
        "dataset": "fashion_mnist",
        "headline": headline,
        "horizon_epochs": epochs,
        "horizon_caveat": (
            "20-ep snapshot; carry MNIST lesson that specialists may still be "
            "mid-decline — do not present as long-run architecture efficiency."
        ),
        "delta_pp": DELTA * 100,
        "dense_scout": dense_rows,
        "dense_slope_analysis": slope,
        "dense_seed_spread": seed_spread["summary"],
        "comparison_grid_locked": comparison_grid,
        "fixed": fixed_score,
        "sliding": sliding_score,
        "sliding_vs_fixed": {
            "label": svf,
            "delta_P": ps - pf,
            "delta_A_pp": (ass - af) * 100,
        },
        "process": {
            "wide_dense_scout_before_lock": True,
            "seed_spread_first_class": True,
            "both_plastic_arms": True,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    plot_pareto(dense_rows, fixed_score, sliding_score, OUT / "pareto_curve.png")

    lines = [
        f"# Fashion-MNIST Pareto — `{headline}`",
        "",
        "## Dense scout (slope + seed spread first)",
        "",
        f"Median dense seed range: **{slope['seed_noise_summary']['median_cell_range_pp']:.2f} pp** "
        f"(max {slope['seed_noise_summary']['max_cell_range_pp']:.2f} pp).",
        "",
    ]
    for r in dense_rows:
        lines.append(
            f"- `{r['tag']}` P={r['mean_params']:.0f} acc={r['mean_test_acc']:.4f} "
            f"range={r['range_pp']:.2f}pp"
        )
    lines += ["", "## Plastic vs cost-matched dense", ""]
    for sc in (fixed_score, sliding_score):
        lines.append(
            f"- **{sc['arm']}**: A_p={sc['A_p']:.4f} P_p={sc['P_p']:.0f} "
            f"A_b={sc['A_b_at_P_p']:.4f} margin={sc['margin_pp']:+.2f}pp "
            f"→ **{sc['verdict']}** near_boundary={sc['near_boundary_delta_uncertain']}"
        )
    lines += [
        "",
        f"**Sliding vs fixed:** `{svf}`",
        "",
        "Seed spreads: `dense_grid_seed_spread.json` (not a follow-up).",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "headline": headline,
        "dense_seed_noise": slope["seed_noise_summary"],
        "fixed": {
            "verdict": fixed_score["verdict"],
            "A_p": fixed_score["A_p"],
            "P_p": fixed_score["P_p"],
            "margin_pp": fixed_score["margin_pp"],
            "near": fixed_score["near_boundary_delta_uncertain"],
        },
        "sliding": {
            "verdict": sliding_score["verdict"],
            "A_p": sliding_score["A_p"],
            "P_p": sliding_score["P_p"],
            "margin_pp": sliding_score["margin_pp"],
            "near": sliding_score["near_boundary_delta_uncertain"],
        },
        "sliding_vs_fixed": svf,
    }, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0 if both else 1


if __name__ == "__main__":
    raise SystemExit(main())
