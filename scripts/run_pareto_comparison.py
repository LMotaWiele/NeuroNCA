#!/usr/bin/env python3
"""Run pre-registered Phase 1a Pareto comparison (fixed-topo grid + plastic arms).

Loads results/phase1a_diagnostics/pareto_comparison_prereg.json, trains the
shared fixed-topology size grid (seeds 0–2, 20 ep), scores:

  PARETO_ACC_AT_COST_FIXED
  PARETO_ACC_AT_COST_SLIDING

and annotates SLIDING_VS_FIXED_ON_PARETO / NEAR_BOUNDARY_DELTA_UNCERTAIN.

Usage:
  python scripts/run_pareto_comparison.py
  python scripts/run_pareto_comparison.py --skip-train   # score only if grid already run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.mlp import MLPReasoningNet  # noqa: E402
from neuronca.training.loop import evaluate, train_epoch  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402

PREREG = _REPO / "results" / "phase1a_diagnostics" / "pareto_comparison_prereg.json"
OUT = _REPO / "results" / "phase1a_diagnostics" / "pareto_comparison"
GRID_ROOT = OUT / "fixed_topo_grid"
DELTA = 0.01
NEAR_BAND = 0.005  # 0.5 pp


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_baseline_cell(
    *,
    tag: str,
    hidden: list[int],
    seed: int,
    epochs: int = 20,
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    h = tuple(int(x) for x in hidden)
    net = MLPReasoningNet(input_dim=784, hidden=h, num_classes=10).to(device)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=str(_REPO / "data" / "mnist"),
        batch_size=128,
        num_workers=0,
        seed=seed,
        probe_size=1024,
    )
    history = []
    for ep in range(epochs):
        m = train_epoch(net, loaders["train_loader"], opt, device=device)
        history.append({"epoch": ep, **m})
    test_m = evaluate(net, loaders["test_loader"], device=device)
    n_params = int(sum(p.numel() for p in net.parameters()))
    return {
        "tag": tag,
        "seed": seed,
        "hidden": list(h),
        "params": n_params,
        "test_acc": float(test_m["acc"]),
        "test_loss": float(test_m["loss"]),
        "train_history": history,
    }


def train_grid(prereg: dict[str, Any], *, epochs: int = 20) -> dict[str, Any]:
    cells = prereg["fixed_topology_size_grid"]["cells"]
    seeds = [0, 1, 2]
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        tag = cell["tag"]
        hidden = cell["hidden"]
        by_tag[tag] = []
        for seed in seeds:
            out_path = GRID_ROOT / tag / f"seed_{seed}.json"
            if out_path.is_file():
                print(f"[skip existing] {tag} seed {seed}", flush=True)
                row = json.loads(out_path.read_text())
            else:
                print(f"[train] {tag} H={hidden} seed={seed}", flush=True)
                row = run_baseline_cell(tag=tag, hidden=hidden, seed=seed, epochs=epochs)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            by_tag[tag].append(row)
            print(
                f"  -> params={row['params']} test_acc={row['test_acc']:.4f}",
                flush=True,
            )
        mean_acc = float(np.mean([r["test_acc"] for r in by_tag[tag]]))
        mean_p = float(np.mean([r["params"] for r in by_tag[tag]]))
        print(f"  mean {tag}: params={mean_p:.0f} acc={mean_acc:.4f}", flush=True)
    return by_tag


def load_grid_from_disk(prereg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for cell in prereg["fixed_topology_size_grid"]["cells"]:
        tag = cell["tag"]
        by_tag[tag] = []
        for seed in (0, 1, 2):
            p = GRID_ROOT / tag / f"seed_{seed}.json"
            if not p.is_file():
                raise FileNotFoundError(f"missing grid cell {p}")
            by_tag[tag].append(json.loads(p.read_text()))
    return by_tag


def grid_summary(by_tag: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for tag, runs in by_tag.items():
        accs = [r["test_acc"] for r in runs]
        params = [r["params"] for r in runs]
        rows.append(
            {
                "tag": tag,
                "hidden": runs[0]["hidden"],
                "mean_params": float(np.mean(params)),
                "mean_test_acc": float(np.mean(accs)),
                "test_acc_std": float(np.std(accs)),
                "test_acc_range_pp": (max(accs) - min(accs)) * 100.0,
                "per_seed": [
                    {"seed": r["seed"], "params": r["params"], "test_acc": r["test_acc"]}
                    for r in runs
                ],
            }
        )
    rows.sort(key=lambda r: r["mean_params"])
    return rows


def interpolate_baseline_acc(
    grid_rows: list[dict[str, Any]], p_target: float
) -> dict[str, Any]:
    """Linear interp of mean test acc in log(params)."""
    ps = [r["mean_params"] for r in grid_rows]
    accs = [r["mean_test_acc"] for r in grid_rows]
    if p_target < ps[0] or p_target > ps[-1]:
        raise ValueError(
            f"P_p={p_target} outside grid [{ps[0]}, {ps[-1]}]; extend grid before scoring"
        )
    # exact match
    for r in grid_rows:
        if abs(r["mean_params"] - p_target) < 1.0:
            return {
                "A_b": r["mean_test_acc"],
                "method": "exact_cell",
                "bracket": [r["tag"], r["tag"]],
                "params_bracket": [r["mean_params"], r["mean_params"]],
            }
    lo = hi = None
    for i in range(len(ps) - 1):
        if ps[i] <= p_target <= ps[i + 1]:
            lo, hi = i, i + 1
            break
    assert lo is not None and hi is not None
    log_p = math.log(p_target)
    log_lo = math.log(ps[lo])
    log_hi = math.log(ps[hi])
    t = (log_p - log_lo) / (log_hi - log_lo) if log_hi != log_lo else 0.0
    a_b = accs[lo] + t * (accs[hi] - accs[lo])
    return {
        "A_b": float(a_b),
        "method": "log_params_linear",
        "bracket": [grid_rows[lo]["tag"], grid_rows[hi]["tag"]],
        "params_bracket": [ps[lo], ps[hi]],
        "acc_bracket": [accs[lo], accs[hi]],
        "t": t,
    }


def score_arm(
    arm_name: str,
    arm: dict[str, Any],
    grid_rows: list[dict[str, Any]],
    *,
    delta: float = DELTA,
) -> dict[str, Any]:
    a_p = float(arm["mean_test_acc"])
    p_p = float(arm["mean_active_params"])
    interp = interpolate_baseline_acc(grid_rows, p_p)
    a_b = float(interp["A_b"])
    threshold = a_b - delta
    margin = a_p - threshold  # positive => pass
    margin_pp = margin * 100.0
    passed = a_p >= threshold
    near = abs(margin) < NEAR_BAND

    # DOMINATED_BY_SMALLER_DENSE
    dominated = []
    for r in grid_rows:
        if r["mean_params"] <= p_p + 1.0 and r["mean_test_acc"] > a_p + delta:
            dominated.append(
                {
                    "tag": r["tag"],
                    "mean_params": r["mean_params"],
                    "mean_test_acc": r["mean_test_acc"],
                    "acc_minus_plastic_pp": (r["mean_test_acc"] - a_p) * 100.0,
                }
            )

    vs_ge_p = [
        {
            "tag": r["tag"],
            "mean_params": r["mean_params"],
            "mean_test_acc": r["mean_test_acc"],
            "plastic_minus_cell_pp": (a_p - r["mean_test_acc"]) * 100.0,
        }
        for r in grid_rows
        if r["mean_params"] >= p_p - 1.0
    ]

    return {
        "arm": arm_name,
        "predicate": arm["predicate_id"],
        "A_p": a_p,
        "P_p": p_p,
        "A_b_at_P_p": a_b,
        "delta": delta,
        "threshold_A_b_minus_delta": threshold,
        "margin": margin,
        "margin_pp": margin_pp,
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL_ACCURACY_SHORTFALL",
        "near_boundary_delta_uncertain": near,
        "interpolation": interp,
        "dominated_by_smaller_dense": dominated,
        "vs_cells_params_ge_Pp": vs_ge_p,
        "per_seed_plastic": arm["per_seed"],
    }


def sliding_vs_fixed(fixed_score: dict, sliding_score: dict) -> dict[str, Any]:
    pf, af = fixed_score["P_p"], fixed_score["A_p"]
    ps, ass = sliding_score["P_p"], sliding_score["A_p"]
    # dominate: lower-or-equal P and higher-or-equal A, strict in at least one
    s_dom_f = (ps <= pf and ass >= af) and (ps < pf or ass > af)
    f_dom_s = (pf <= ps and af >= ass) and (pf < ps or af > ass)
    if s_dom_f:
        label = "SLIDING_DOMINATES_FIXED"
    elif f_dom_s:
        label = "FIXED_DOMINATES_SLIDING"
    elif ps < pf and ass < af:
        label = "TRADEOFF_SLIDING_CHEAPER_LOWER_ACC"
    elif ps > pf and ass > af:
        label = "TRADEOFF_SLIDING_COSTLIER_HIGHER_ACC"
    else:
        label = "TRADEOFF_OR_SIMILAR"
    return {
        "label": label,
        "fixed": {"P_p": pf, "A_p": af},
        "sliding": {"P_p": ps, "A_p": ass},
        "delta_P": ps - pf,
        "delta_A_pp": (ass - af) * 100.0,
        "note": (
            "Tests whether reduced early-cohort erosion under sliding shows up as "
            "better task-performance-per-parameter on the Pareto axes."
        ),
    }


def plot_pareto(
    grid_rows: list[dict[str, Any]],
    fixed_arm: dict[str, Any],
    sliding_arm: dict[str, Any],
    scores: dict[str, Any],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    gp = [r["mean_params"] for r in grid_rows]
    ga = [r["mean_test_acc"] for r in grid_rows]
    ax.plot(gp, ga, "o-", color="#616161", lw=1.5, markersize=7, label="fixed-topo dense grid")
    for r in grid_rows:
        ax.annotate(
            r["tag"],
            (r["mean_params"], r["mean_test_acc"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#424242",
        )

    # plastic means
    ax.scatter(
        [fixed_arm["mean_active_params"]],
        [fixed_arm["mean_test_acc"]],
        s=120,
        c="#c62828",
        marker="*",
        zorder=5,
        label=f"plastic fixed (P={fixed_arm['mean_active_params']:.0f})",
    )
    ax.scatter(
        [sliding_arm["mean_active_params"]],
        [sliding_arm["mean_test_acc"]],
        s=120,
        c="#1565c0",
        marker="*",
        zorder=5,
        label=f"plastic sliding (P={sliding_arm['mean_active_params']:.0f})",
    )
    # per-seed
    for s in fixed_arm["per_seed"]:
        ax.scatter(
            [s["active_params"]],
            [s["test_acc"]],
            s=40,
            c="#ef9a9a",
            marker="o",
            zorder=4,
            alpha=0.9,
        )
    for s in sliding_arm["per_seed"]:
        ax.scatter(
            [s["active_params"]],
            [s["test_acc"]],
            s=40,
            c="#90caf9",
            marker="s",
            zorder=4,
            alpha=0.9,
        )

    # threshold markers at each arm's P
    for name, color, sc in (
        ("fixed", "#c62828", scores["fixed"]),
        ("sliding", "#1565c0", scores["sliding"]),
    ):
        ax.axvline(sc["P_p"], color=color, ls=":", lw=1, alpha=0.5)
        thr = sc["threshold_A_b_minus_delta"]
        ax.plot(
            [sc["P_p"]],
            [thr],
            marker="_",
            markersize=18,
            color=color,
            mew=2,
            label=f"{name} bar A_b−δ={thr:.4f}",
        )

    ax.set_xlabel("Parameter count (plastic: active; baseline: dense)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Phase 1a Pareto @ 20ep — mid-decline snapshot\n"
        f"fixed {scores['fixed']['verdict']} | sliding {scores['sliding']['verdict']} | "
        f"{scores['sliding_vs_fixed']['label']}"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing grid JSON under pareto_comparison/fixed_topo_grid/",
    )
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        by_tag = load_grid_from_disk(prereg)
    else:
        by_tag = train_grid(prereg, epochs=int(args.epochs))

    grid_rows = grid_summary(by_tag)
    (OUT / "grid_summary.json").write_text(
        json.dumps(grid_rows, indent=2), encoding="utf-8"
    )

    fixed_arm = prereg["plastic_arms"]["fixed"]
    sliding_arm = prereg["plastic_arms"]["sliding"]
    fixed_score = score_arm("fixed", fixed_arm, grid_rows)
    sliding_score = score_arm("sliding", sliding_arm, grid_rows)
    svf = sliding_vs_fixed(fixed_score, sliding_score)

    overall = {
        "PARETO_ACC_AT_COST_FIXED": fixed_score["pass"],
        "PARETO_ACC_AT_COST_SLIDING": sliding_score["pass"],
        "both_pass": fixed_score["pass"] and sliding_score["pass"],
        "any_near_boundary": (
            fixed_score["near_boundary_delta_uncertain"]
            or sliding_score["near_boundary_delta_uncertain"]
        ),
        "sliding_vs_fixed": svf["label"],
    }
    if overall["both_pass"]:
        headline = "PARETO_BOTH_PASS"
    elif fixed_score["pass"] and not sliding_score["pass"]:
        headline = "PARETO_FIXED_PASS_SLIDING_FAIL"
    elif sliding_score["pass"] and not fixed_score["pass"]:
        headline = "PARETO_SLIDING_PASS_FIXED_FAIL"
    else:
        headline = "PARETO_BOTH_FAIL"

    report = {
        "headline": headline,
        "horizon_caveat": prereg["horizon_caveat"],
        "delta": DELTA,
        "near_boundary_band": NEAR_BAND,
        "overall": overall,
        "fixed": fixed_score,
        "sliding": sliding_score,
        "sliding_vs_fixed": svf,
        "grid": grid_rows,
        "prereg_path": str(PREREG),
        "writeup_reminders": [
            "20-ep accuracy is a mid-decline snapshot (specialists still eroding).",
            "If NEAR_BOUNDARY_DELTA_UNCERTAIN, do not overclaim clean pass/fail.",
            "δ was not retuned after seeing results.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    plot_pareto(
        grid_rows,
        fixed_arm,
        sliding_arm,
        {"fixed": fixed_score, "sliding": sliding_score, "sliding_vs_fixed": svf},
        OUT / "pareto_curve.png",
    )

    # README
    lines = [
        f"# Pareto comparison — `{headline}`",
        "",
        f"**Horizon:** 20 epochs (mid-decline snapshot; specialists still eroding).",
        f"**δ = {DELTA*100:.1f} pp** test acc. Near-boundary band ±{NEAR_BAND*100:.1f} pp.",
        "",
        "## Predicates",
        "",
        f"| Arm | A_p | P_p | A_b(P_p) | A_b−δ | margin (pp) | Verdict | Near boundary? |",
        f"|-----|----:|----:|---------:|------:|------------:|---------|----------------|",
    ]
    for sc in (fixed_score, sliding_score):
        lines.append(
            f"| {sc['arm']} | {sc['A_p']:.4f} | {sc['P_p']:.0f} | {sc['A_b_at_P_p']:.4f} | "
            f"{sc['threshold_A_b_minus_delta']:.4f} | {sc['margin_pp']:+.2f} | "
            f"**{sc['verdict']}** | {sc['near_boundary_delta_uncertain']} |"
        )
    lines += [
        "",
        f"**Sliding vs fixed:** `{svf['label']}` "
        f"(ΔP={svf['delta_P']:.0f}, ΔA={svf['delta_A_pp']:+.2f} pp)",
        "",
        "## Fixed-topo grid (mean over seeds)",
        "",
        "| Tag | H | params | test acc |",
        "|-----|---|-------:|---------:|",
    ]
    for r in grid_rows:
        lines.append(
            f"| {r['tag']} | {r['hidden']} | {r['mean_params']:.0f} | {r['mean_test_acc']:.4f} |"
        )
    lines += [
        "",
        f"Plot: `pareto_curve.png`",
        f"JSON: `summary.json`",
        "",
    ]
    if fixed_score["dominated_by_smaller_dense"] or sliding_score["dominated_by_smaller_dense"]:
        lines.append("## DOMINATED_BY_SMALLER_DENSE")
        lines.append("")
        for sc in (fixed_score, sliding_score):
            if sc["dominated_by_smaller_dense"]:
                lines.append(f"- **{sc['arm']}**: {sc['dominated_by_smaller_dense']}")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"headline": headline, "overall": overall, "fixed": {
        "verdict": fixed_score["verdict"], "margin_pp": fixed_score["margin_pp"],
        "A_p": fixed_score["A_p"], "A_b": fixed_score["A_b_at_P_p"], "near": fixed_score["near_boundary_delta_uncertain"],
    }, "sliding": {
        "verdict": sliding_score["verdict"], "margin_pp": sliding_score["margin_pp"],
        "A_p": sliding_score["A_p"], "A_b": sliding_score["A_b_at_P_p"], "near": sliding_score["near_boundary_delta_uncertain"],
    }, "sliding_vs_fixed": svf}, indent=2), flush=True)
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return 0 if overall["both_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
