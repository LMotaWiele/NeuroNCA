#!/usr/bin/env python3
"""Fashion rate-cap ablate: vary only ``max_new_neurons_per_layer`` (cap=1 closed thrash)."""

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

OUT = (
    _REPO
    / "results"
    / "phase1a_diagnostics"
    / "fashion_mnist_calibration"
    / "rate_cap_ablate"
)
FASHION_R = {"h1": 0.71, "h2": 0.80, "h3": 0.91}


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


def make_cfg(max_new: int, *, prune: bool = True) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates=dict(FASHION_R),
        tau=0.5,
        threshold_kind="fixed",
        mu=0.999,
        prune_edge_fraction=0.05 if prune else 0.0,
        prune_neuron_z=-1.0 if prune else -1.0e9,
        max_new_edges_per_layer=64 if max_new > 0 else 0,
        max_new_neurons_per_layer=int(max_new),
        seed_edges_per_new_neuron=4 if max_new > 0 else 0,
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


def swing_stats(accs: list[float]) -> dict[str, Any]:
    sw = [abs(accs[i] - accs[i - 1]) * 100 for i in range(1, len(accs))]
    if not sw:
        return {
            "sum_abs_swing_pp": 0.0,
            "max_abs_swing_pp": 0.0,
            "mean_abs_swing_pp": 0.0,
            "median_abs_swing_pp": 0.0,
            "n_swings_gt_5pp": 0,
            "n_swings_gt_10pp": 0,
            "acc_range_pp": 0.0,
            "abs_swings_pp": [],
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
    }


def run_cap(
    max_new: int,
    *,
    seed: int,
    epochs: int,
    prune: bool,
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
    cfg = make_cfg(max_new, prune=prune)
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
        row = {
            "epoch": epoch,
            "test_acc": float(vm["acc"]),
            "test_loss": float(vm["loss"]),
            "neurons_grown": int(slog.neurons_grown),
            "neurons_pruned": int(slog.neurons_pruned),
            "edges_pruned_competitive": int(slog.edges_pruned_competitive),
            "hidden_counts": H,
            "H_sum": int(sum(H.values())),
            "active_params": active,
        }
        rows.append(row)
        print(
            f"  [cap={max_new} prune={int(prune)} s{seed}] ep{epoch:02d} "
            f"acc={vm['acc']:.4f} g={row['neurons_grown']} p={row['neurons_pruned']} "
            f"Hsum={row['H_sum']} P={active}",
            flush=True,
        )

    accs = [r["test_acc"] for r in rows]
    sw = swing_stats(accs)
    thrash = sw["max_abs_swing_pp"] >= 10.0 or sw["sum_abs_swing_pp"] >= 40.0
    mild = sw["max_abs_swing_pp"] < 5.0 and sw["sum_abs_swing_pp"] < 20.0
    return {
        "max_new_neurons_per_layer": max_new,
        "max_births_per_epoch_ceiling": 3 * max_new,
        "prune_enabled": prune,
        "seed": seed,
        "epochs": epochs,
        "target_rates": dict(FASHION_R),
        "trajectory": rows,
        "acc_trajectory": accs,
        "swing": sw,
        "thrash": thrash,
        "mild": mild,
        "totals": {
            "neurons_grown": int(sum(r["neurons_grown"] for r in rows)),
            "neurons_pruned": int(sum(r["neurons_pruned"] for r in rows)),
            "edges_pruned": int(sum(r["edges_pruned_competitive"] for r in rows)),
            "mean_grown_per_epoch": float(
                np.mean([r["neurons_grown"] for r in rows])
            ),
        },
        "final_acc": float(accs[-1]),
        "peak_acc": float(max(accs)),
        "min_acc": float(min(accs)),
        "final_active_params": int(rows[-1]["active_params"]),
        "final_hidden": rows[-1]["hidden_counts"],
    }


def classify(by_cap: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered reading of rate-cap response."""
    caps = sorted(by_cap.keys(), reverse=True)
    product = by_cap.get(4)
    low = by_cap.get(1) or by_cap.get(min(c for c in caps if c > 0), None)
    zero = by_cap.get(0)

    def swing_key(c: int) -> tuple[float, float]:
        s = by_cap[c]["swing"]
        return (s["max_abs_swing_pp"], s["sum_abs_swing_pp"])

    # Monotonic: lower cap → lower max swing?
    ordered = sorted(caps)
    maxes = [by_cap[c]["swing"]["max_abs_swing_pp"] for c in ordered]
    sums = [by_cap[c]["swing"]["sum_abs_swing_pp"] for c in ordered]
    # Spearmans of cap vs max swing (expect positive if higher cap → more thrash)
    if len(ordered) >= 3:
        # ranks
        from math import isfinite

        def spearmans(x, y):
            xr = np.argsort(np.argsort(x)).astype(float)
            yr = np.argsort(np.argsort(y)).astype(float)
            if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
                return float("nan")
            return float(np.corrcoef(xr, yr)[0, 1])

        rho_max = spearmans(ordered, maxes)
        rho_sum = spearmans(ordered, sums)
    else:
        rho_max = rho_sum = float("nan")

    product_thrash = bool(product["thrash"]) if product else None
    low_mild = bool(low["mild"]) if low else None
    low_thrash = bool(low["thrash"]) if low else None
    zero_mild = bool(zero["mild"]) if zero else None

    if product and low and product_thrash and low_mild:
        verdict = "RATE_CAP_RESOLVES_THRASH"
        reading = (
            f"Product cap=4 thrashes (max={product['swing']['max_abs_swing_pp']:.1f}pp); "
            f"cap={low['max_new_neurons_per_layer']} is mild "
            f"(max={low['swing']['max_abs_swing_pp']:.1f}pp). "
            "Cadence/volume is causal for global instability; lower birth rate is "
            "the Fashion product fix for thrash."
        )
    elif product and low and product_thrash and not low_thrash and not low_mild:
        verdict = "RATE_CAP_REDUCES_NOT_RESOLVES"
        reading = (
            f"Lower cap reduces thrash "
            f"(max {product['swing']['max_abs_swing_pp']:.1f}→"
            f"{low['swing']['max_abs_swing_pp']:.1f}pp) but does not reach mild bar. "
            "Volume is a real lever; may need still lower rate or cadence change "
            "(every-k epochs) for product smoothness."
        )
    elif product and low and product_thrash and low_thrash:
        # check if still reduced
        if low["swing"]["max_abs_swing_pp"] < 0.7 * product["swing"]["max_abs_swing_pp"]:
            verdict = "RATE_CAP_REDUCES_NOT_RESOLVES"
            reading = (
                "Lower cap materially cuts swing magnitude but thrash bar still fails. "
                "Volume matters; not a complete fix at tested caps."
            )
        else:
            verdict = "RATE_CAP_NO_HELP"
            reading = (
                "Lowering max_new_neurons_per_layer does not meaningfully reduce thrash. "
                "Cadence volume is not the residual driver at these settings — reopen "
                "birth realization / imprint aggressiveness."
            )
    elif product and not product_thrash:
        verdict = "PRODUCT_ALREADY_MILD"
        reading = "Product cap did not thrash this seed/horizon; rate-cap story not testable here."
    else:
        verdict = "INCONCLUSIVE"
        reading = "Incomplete cap grid or unexpected pattern; see table."

    # Acc tradeoff: does lower rate sacrifice final/peak acc badly?
    acc_note = ""
    if product and low:
        d_final = low["final_acc"] - product["final_acc"]
        d_peak = low["peak_acc"] - product["peak_acc"]
        d_min = low["min_acc"] - product["min_acc"]
        acc_note = (
            f"cap={low['max_new_neurons_per_layer']} vs product: "
            f"Δfinal={d_final:+.3f}, Δpeak={d_peak:+.3f}, Δmin={d_min:+.3f} "
            f"(higher min = less floor crash)."
        )

    return {
        "verdict": verdict,
        "reading": reading,
        "acc_tradeoff_note": acc_note,
        "spearman_cap_vs_max_swing": rho_max,
        "spearman_cap_vs_sum_swing": rho_sum,
        "thrash_bar": {"max_pp": 10.0, "sum_pp": 40.0},
        "mild_bar": {"max_pp": 5.0, "sum_pp": 20.0},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caps", type=str, default="4,2,1,0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-prune-at-zero",
        action="store_true",
        help="When cap=0, also disable prune (pure frozen)",
    )
    args = ap.parse_args()
    caps = [int(x) for x in args.caps.split(",") if x.strip() != ""]
    OUT.mkdir(parents=True, exist_ok=True)

    by_cap: dict[int, dict[str, Any]] = {}
    for cap in caps:
        prune = True
        if cap == 0 and args.no_prune_at_zero:
            prune = False
        print(f"\n=== max_new_neurons_per_layer={cap} prune={prune} ===", flush=True)
        res = run_cap(cap, seed=args.seed, epochs=args.epochs, prune=prune)
        by_cap[cap] = res
        path = OUT / f"cap_{cap}_seed_{args.seed}.json"
        path.write_text(json.dumps(res, indent=2))
        print(
            f"  -> thrash={res['thrash']} mild={res['mild']} "
            f"max_swing={res['swing']['max_abs_swing_pp']:.1f} "
            f"sum={res['swing']['sum_abs_swing_pp']:.1f} "
            f"grown={res['totals']['neurons_grown']} "
            f"final={res['final_acc']:.3f} min={res['min_acc']:.3f}",
            flush=True,
        )

    isolation = classify(by_cap)
    table = []
    for cap in sorted(by_cap.keys(), reverse=True):
        r = by_cap[cap]
        table.append(
            {
                "cap": cap,
                "max_births_ep_ceiling": 3 * cap,
                "mean_grown_per_ep": r["totals"]["mean_grown_per_epoch"],
                "total_grown": r["totals"]["neurons_grown"],
                "max_abs_swing_pp": r["swing"]["max_abs_swing_pp"],
                "sum_abs_swing_pp": r["swing"]["sum_abs_swing_pp"],
                "n_swings_gt_5pp": r["swing"]["n_swings_gt_5pp"],
                "n_swings_gt_10pp": r["swing"]["n_swings_gt_10pp"],
                "thrash": r["thrash"],
                "mild": r["mild"],
                "final_acc": r["final_acc"],
                "peak_acc": r["peak_acc"],
                "min_acc": r["min_acc"],
                "final_P": r["final_active_params"],
                "acc_trajectory": r["acc_trajectory"],
            }
        )

    summary = {
        "protocol": {
            "dataset": "fashion_mnist",
            "target_rates": FASHION_R,
            "stack": "fixed+gated_novelty+gap_imprinted+competitive_prune",
            "varied": "max_new_neurons_per_layer only",
            "epochs": args.epochs,
            "seed": args.seed,
            "caps": caps,
        },
        "table": table,
        "isolation": isolation,
        "chapter_close": {
            "fashion_growth_story": (
                "Trigger tracks error; majority births help locally; minority backfire "
                "not crash-proximate; thrash = insertion volume. Rate-cap is the last "
                "Fashion-specific growth lever in this chain."
            ),
            "next_chapter": "adaptation-cost / task-shift under locked stack",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Fashion rate-cap ablate (final growth thrash lever)",
        "",
        f"**Verdict:** `{isolation['verdict']}`",
        "",
        isolation["reading"],
        "",
        isolation.get("acc_tradeoff_note") or "",
        "",
        f"Spearman(cap, max_swing)={isolation.get('spearman_cap_vs_max_swing')}; "
        f"Spearman(cap, sum_swing)={isolation.get('spearman_cap_vs_sum_swing')}",
        "",
        "## Protocol",
        "",
        f"Fashion r\\*_ℓ={FASHION_R}, full product stack, seed {args.seed}, "
        f"{args.epochs} ep. Only `max_new_neurons_per_layer` varied.",
        "",
        "Thrash bar: max\\|Δ\\|≥10pp or sum\\|Δ\\|≥40pp. Mild: max<5 and sum<20.",
        "",
        "## Results",
        "",
        "| Cap | ≤births/ep | mean grown/ep | total grown | max\\|Δ\\| | sum\\|Δ\\| | #Δ>5 | thrash | mild | final | peak | min |",
        "|----:|----------:|-------------:|------------:|---------:|---------:|-----:|:------:|:----:|------:|-----:|----:|",
    ]
    for row in table:
        lines.append(
            f"| {row['cap']} | {row['max_births_ep_ceiling']} | "
            f"{row['mean_grown_per_ep']:.1f} | {row['total_grown']} | "
            f"{row['max_abs_swing_pp']:.1f} | {row['sum_abs_swing_pp']:.1f} | "
            f"{row['n_swings_gt_5pp']} | {row['thrash']} | {row['mild']} | "
            f"{row['final_acc']:.3f} | {row['peak_acc']:.3f} | {row['min_acc']:.3f} |"
        )
    lines += ["", "## Acc trajectories", ""]
    for row in table:
        traj = " → ".join(f"{a:.3f}" for a in row["acc_trajectory"])
        lines.append(f"- **cap={row['cap']}:** `{traj}`")
    lines += [
        "",
        "## Chapter position",
        "",
        summary["chapter_close"]["fashion_growth_story"],
        "",
        f"**Next:** {summary['chapter_close']['next_chapter']}",
        "",
        "Artifacts: `cap_*_seed_*.json`, `summary.json`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps({"isolation": isolation, "table": table}, indent=2), flush=True)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
