#!/usr/bin/env python3
"""Self-calibrating constants battery (§4.1–4.6).

Stages (run in order; prereg bars in results/.../self_calibrate/prereg.json):

  mnist_equiv   — frozen vs self-cal on MNIST (implementation guard)
  fashion_adapt — Fashion with warm-start only (no ported MNIST r*)
  mu_sweep      — decay-rate grid per constant
  feedback      — coupled r*↔structure runaway check
  ablation      — single-constant-on + all-on on Fashion
  kmnist        — third dataset, zero manual constants

Usage:
  python scripts/run_self_calibrate_battery.py --stage mnist_equiv
  python scripts/run_self_calibrate_battery.py --stage all
"""

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

from neuronca.data.mnist import (  # noqa: E402
    DEFAULT_FASHION_ROOT,
    DEFAULT_KMNIST_ROOT,
    build_mnist_loaders,
)
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet, HIDDEN_LAYERS  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.probe_rates import probe_target_rates  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.plasticity.self_calibrate import (  # noqa: E402
    detect_rate_runaway,
    detect_stat_oscillation,
)
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "self_calibrate"
PREREG = json.loads((OUT / "prereg.json").read_text())
MNIST_R = PREREG["bars"]["mnist_frozen_r_star"]
FASHION_R = PREREG["bars"]["fashion_hand_probed_r_star"]
MU_GRID = PREREG["bars"]["mu_grid"]
TOL = float(PREREG["bars"]["mnist_equiv_r_star_tolerance_rel"])


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loaders(dataset: str, seed: int):
    task = TaskContext(make_identity_transform())
    root = "data/mnist"
    name = dataset
    if dataset in ("fashion", "fashion_mnist"):
        root = str(DEFAULT_FASHION_ROOT)
        name = "fashion_mnist"
    elif dataset in ("kmnist",):
        root = str(DEFAULT_KMNIST_ROOT)
        name = "kmnist"
    return build_mnist_loaders(
        task,
        root=root,
        batch_size=128,
        num_workers=0,
        seed=seed,
        probe_size=1024,
        dataset=name,
    )


def make_cfg(
    *,
    target_rates: dict[str, float] | None,
    self_cal: dict[str, bool],
    mu_r: float = 0.999,
    mu_p: float = 0.999,
    mu_s: float = 0.99,
    hybrid: int | None = None,
    max_new: int = 1,
    tau: float = 0.5,
) -> PlasticityConfig:
    rates = dict(target_rates) if target_rates else {}
    return PlasticityConfig(
        radius=0.35,
        target_rates=rates,
        tau=tau,
        threshold_kind="fixed",
        mu=0.999,
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=max_new,
        seed_edges_per_new_neuron=4,
        newborn_grace_steps=3,
        newborn_ema_init="layer_target",
        newborn_birth_bias=0.5,
        newborn_birth_bias_steps=3,
        newborn_weight_init="gap_imprinted",
        newborn_copy_outgoing=True,
        newborn_parent_coord_jitter=0.05,
        neuron_growth_trigger="gated_novelty",
        novelty_k_min=11,
        novelty_rank_fraction=1.0,
        self_calibrate=dict(self_cal),
        self_cal_mu_r_star=mu_r,
        self_cal_mu_prune=mu_p,
        self_cal_mu_straddle=mu_s,
        self_cal_hybrid_period=hybrid,
    )


def swing_stats(accs: list[float]) -> dict[str, Any]:
    sw = [abs(accs[i] - accs[i - 1]) * 100 for i in range(1, len(accs))]
    if not sw:
        return {"max_abs_swing_pp": 0.0, "sum_abs_swing_pp": 0.0, "n_gt_5": 0}
    return {
        "max_abs_swing_pp": float(max(sw)),
        "sum_abs_swing_pp": float(sum(sw)),
        "mean_abs_swing_pp": float(np.mean(sw)),
        "n_gt_5": int(sum(1 for s in sw if s > 5)),
        "thrash": max(sw) >= 10.0 or sum(sw) >= 40.0,
        "mild": max(sw) < 5.0 and sum(sw) < 20.0,
    }


def run_plastic(
    *,
    dataset: str,
    seed: int,
    epochs: int,
    cfg: PlasticityConfig,
    warm_start_probe: bool,
    label: str,
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    ld = loaders(dataset, seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    orch = PlasticityOrchestrator(config=cfg)

    probe_info: dict[str, Any] = {}
    if warm_start_probe or any(cfg.self_calibrate_flags().values()):
        # Always probe when self-cal on; optional when frozen for comparison
        need = cfg.self_calibrate_flags()
        if need["r_star"] or warm_start_probe:
            probe_info = probe_target_rates(
                net, ld["train_loader"], signal=signal, optimizer=opt, device=device
            )
            rates = probe_info["target_rates"]
            if need["r_star"] or need["prune_cutoff"] or need["straddle"]:
                orch.warm_start_self_cal(
                    target_rates=rates if need["r_star"] else cfg.resolved_target_rates(),
                    prune_z_cut=float(cfg.prune_neuron_z),
                )
            # If r_star self-cal, config already updated by warm_start
            if not need["r_star"] and not cfg.target_rates:
                # frozen path requested rates — should have been in cfg
                pass
        elif need["prune_cutoff"] or need["straddle"]:
            orch.warm_start_self_cal(
                target_rates=cfg.resolved_target_rates() if cfg.target_rates else MNIST_R,
                prune_z_cut=float(cfg.prune_neuron_z),
            )

    gstep = 0
    rows: list[dict[str, Any]] = []
    r_hist: list[dict[str, float]] = []
    for epoch in range(epochs):
        m = train_epoch_with_signals(
            net, ld["train_loader"], opt, signal, orch, device=device, global_step=gstep
        )
        gstep = int(m["global_step"])
        gated = collect_gated_novelty_probe(
            net,
            ld["probe_loader"],
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
        vm = evaluate(net, ld["test_loader"], device=device)
        live_r = dict(slog.r_star_live) if slog.r_star_live else dict(
            orch.config.resolved_target_rates()
        )
        r_hist.append(live_r)
        rows.append(
            {
                "epoch": epoch,
                "test_acc": float(vm["acc"]),
                "test_loss": float(vm["loss"]),
                "neurons_grown": int(slog.neurons_grown),
                "neurons_pruned": int(slog.neurons_pruned),
                "n_contraction_frac": float(slog.n_contraction_frac),
                "r_star_live": live_r,
                "prune_z_cut_live": slog.prune_z_cut_live,
                "hidden_counts": net.hidden_counts(),
            }
        )
        print(
            f"  [{label}] ep{epoch:02d} acc={vm['acc']:.4f} "
            f"g={slog.neurons_grown} p={slog.neurons_pruned} "
            f"cfrac={slog.n_contraction_frac:.3f} r*={live_r}",
            flush=True,
        )

    accs = [r["test_acc"] for r in rows]
    sw = swing_stats(accs)
    final_r = r_hist[-1] if r_hist else {}
    return {
        "label": label,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "self_calibrate": cfg.self_calibrate_flags(),
        "probe_rates": probe_info.get("target_rates"),
        "probe_rates_rounded": probe_info.get("target_rates_rounded"),
        "final_r_star": final_r,
        "r_star_history": r_hist,
        "trajectory": rows,
        "acc_trajectory": accs,
        "swing": sw,
        "final_acc": float(accs[-1]) if accs else float("nan"),
        "mean_contraction_frac": float(
            np.mean([r["n_contraction_frac"] for r in rows])
        ),
        "final_contraction_frac": float(rows[-1]["n_contraction_frac"]) if rows else float("nan"),
        "runaway": detect_rate_runaway(r_hist) if len(r_hist) > 1 else {},
        "r_star_oscillation": {
            h: detect_stat_oscillation([step[h] for step in r_hist if h in step])
            for h in HIDDEN_LAYERS
        }
        if r_hist
        else {},
    }


def within_tol(live: dict[str, float], ref: dict[str, float], tol: float) -> dict[str, Any]:
    ok = True
    per = {}
    for h in HIDDEN_LAYERS:
        a, b = float(live[h]), float(ref[h])
        rel = abs(a - b) / max(abs(b), 1e-8)
        per[h] = {"live": a, "ref": b, "rel_err": rel, "ok": rel <= tol}
        ok = ok and rel <= tol
    return {"pass": ok, "per_layer": per, "tol": tol}


def stage_mnist_equiv(epochs: int, seed: int) -> dict[str, Any]:
    print("=== §4.1 MNIST equivalence ===", flush=True)
    frozen = run_plastic(
        dataset="mnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates=dict(MNIST_R),
            self_cal={"r_star": False, "prune_cutoff": False, "straddle": False},
            max_new=4,
        ),
        warm_start_probe=False,
        label="mnist_frozen",
    )
    scal = run_plastic(
        dataset="mnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates=dict(MNIST_R),  # ignored after probe warm-start
            self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
            max_new=4,
        ),
        warm_start_probe=True,
        label="mnist_selfcal_r",
    )
    eq = within_tol(scal["final_r_star"], MNIST_R, TOL)
    # Also compare to frozen final rates (should be frozen MNIST_R)
    acc_delta = abs(scal["final_acc"] - frozen["final_acc"])
    verdict = "PASS" if eq["pass"] and acc_delta < 0.05 else "FAIL"
    if not eq["pass"]:
        # Implementation bug until fixed
        verdict = "FAIL_IMPLEMENTATION"
    out = {
        "stage": "mnist_equiv",
        "verdict": verdict,
        "equivalence": eq,
        "acc_frozen": frozen["final_acc"],
        "acc_selfcal": scal["final_acc"],
        "acc_delta": acc_delta,
        "frozen": frozen,
        "selfcal": scal,
    }
    (OUT / "mnist_equiv.json").write_text(json.dumps(out, indent=2))
    print(f"§4.1 verdict={verdict} eq={eq['pass']} acc_delta={acc_delta:.4f}", flush=True)
    return out


def stage_fashion_adapt(epochs: int, seed: int) -> dict[str, Any]:
    print("=== §4.2 Fashion auto-adaptation ===", flush=True)
    # Ported MNIST r* baseline (the failure mode)
    bad = run_plastic(
        dataset="fashion_mnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates=dict(MNIST_R),
            self_cal={"r_star": False, "prune_cutoff": False, "straddle": False},
            max_new=1,
        ),
        warm_start_probe=False,
        label="fashion_ported_mnist_r",
    )
    # Self-cal: no manual Fashion rates in cfg
    good = run_plastic(
        dataset="fashion_mnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates={},  # empty — warm-start only
            self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
            max_new=1,
        ),
        warm_start_probe=True,
        label="fashion_selfcal_r",
    )
    match = within_tol(good["final_r_star"], FASHION_R, TOL)
    # Also compare to probe-time rates (should stay near probe)
    probe_match = within_tol(
        good["final_r_star"], good.get("probe_rates") or FASHION_R, 0.15
    )
    cfrac_bad = bad["mean_contraction_frac"]
    cfrac_good = good["mean_contraction_frac"]
    # Pass: r* matches Fashion hand probe; contraction not chronically worse than self-cal mid
    verdict = (
        "PASS"
        if match["pass"] and cfrac_good < 0.55
        else "FAIL"
    )
    out = {
        "stage": "fashion_adapt",
        "verdict": verdict,
        "match_hand_probed": match,
        "match_own_probe": probe_match,
        "ported_mnist_r_mean_contraction_frac": cfrac_bad,
        "selfcal_mean_contraction_frac": cfrac_good,
        "final_r_star": good["final_r_star"],
        "hand_probed_ref": FASHION_R,
        "ported_mnist_ref": MNIST_R,
        "probe_at_start": good.get("probe_rates"),
        "ported": bad,
        "selfcal": good,
    }
    (OUT / "fashion_adapt.json").write_text(json.dumps(out, indent=2))
    print(
        f"§4.2 verdict={verdict} final_r*={good['final_r_star']} "
        f"cfrac_ported={cfrac_bad:.3f} cfrac_self={cfrac_good:.3f}",
        flush=True,
    )
    return out


def stage_mu_sweep(epochs: int, seed: int) -> dict[str, Any]:
    print("=== §4.3 μ sweep (r_star) ===", flush=True)
    rows = []
    for mu in MU_GRID:
        res = run_plastic(
            dataset="fashion_mnist",
            seed=seed,
            epochs=epochs,
            cfg=make_cfg(
                target_rates={},
                self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
                mu_r=float(mu),
                max_new=1,
            ),
            warm_start_probe=True,
            label=f"mu_r={mu}",
        )
        osc_any = any(
            bool(v.get("oscillating")) for v in res["r_star_oscillation"].values()
        )
        run = bool(res["runaway"].get("runaway"))
        rows.append(
            {
                "mu": mu,
                "swing": res["swing"],
                "final_r_star": res["final_r_star"],
                "oscillating": osc_any,
                "runaway": run,
                "final_acc": res["final_acc"],
                "mild": res["swing"].get("mild"),
                "thrash": res["swing"].get("thrash"),
            }
        )
        print(
            f"  mu={mu}: thrash={res['swing'].get('thrash')} mild={res['swing'].get('mild')} "
            f"osc={osc_any} runaway={run} max_sw={res['swing']['max_abs_swing_pp']:.1f}",
            flush=True,
        )
    # Lock: prefer mild, not oscillating, not runaway; among those highest mu (stable)
    candidates = [
        r for r in rows if r["mild"] and not r["oscillating"] and not r["runaway"]
    ]
    if not candidates:
        candidates = [r for r in rows if not r["thrash"] and not r["runaway"]]
    locked = max(candidates, key=lambda r: r["mu"]) if candidates else rows[-2]
    out = {
        "stage": "mu_sweep",
        "rows": rows,
        "locked_mu_r_star": locked["mu"],
        "lock_reason": "mild+no_osc+no_runaway; prefer higher mu",
        "verdict": "PASS" if candidates else "FAIL_NO_STABLE_MU",
    }
    (OUT / "mu_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"§4.3 locked mu_r_star={locked['mu']} verdict={out['verdict']}", flush=True)
    return out


def stage_feedback(epochs: int, seed: int, mu: float) -> dict[str, Any]:
    print("=== §4.4 feedback-loop stability ===", flush=True)
    results = {}
    for ds in ("mnist", "fashion_mnist"):
        res = run_plastic(
            dataset=ds,
            seed=seed,
            epochs=epochs,
            cfg=make_cfg(
                target_rates={} if ds != "mnist" else dict(MNIST_R),
                self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
                mu_r=mu,
                max_new=4 if ds == "mnist" else 1,
            ),
            warm_start_probe=True,
            label=f"feedback_{ds}",
        )
        results[ds] = res
    run_any = any(bool(results[d]["runaway"].get("runaway")) for d in results)
    hybrid = None
    if run_any:
        hybrid = int(PREREG["bars"]["hybrid_period_if_runaway"])
        # Re-run fashion with hybrid
        res_h = run_plastic(
            dataset="fashion_mnist",
            seed=seed,
            epochs=epochs,
            cfg=make_cfg(
                target_rates={},
                self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
                mu_r=mu,
                hybrid=hybrid,
                max_new=1,
            ),
            warm_start_probe=True,
            label="feedback_fashion_hybrid",
        )
        results["fashion_hybrid"] = res_h
        run_any = bool(res_h["runaway"].get("runaway"))
    verdict = "PASS" if not run_any else "FAIL_RUNAWAY"
    out = {
        "stage": "feedback",
        "verdict": verdict,
        "hybrid_period_engaged": hybrid,
        "results": results,
    }
    (OUT / "feedback.json").write_text(json.dumps(out, indent=2))
    print(f"§4.4 verdict={verdict} hybrid={hybrid}", flush=True)
    return out


def stage_ablation(epochs: int, seed: int, mu: float) -> dict[str, Any]:
    print("=== §4.5 ablation ===", flush=True)
    baseline = run_plastic(
        dataset="fashion_mnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates=dict(MNIST_R),
            self_cal={"r_star": False, "prune_cutoff": False, "straddle": False},
            max_new=1,
        ),
        warm_start_probe=False,
        label="abl_baseline_ported",
    )
    arms = {
        "r_star_only": {"r_star": True, "prune_cutoff": False, "straddle": False},
        "prune_only": {"r_star": False, "prune_cutoff": True, "straddle": False},
        "straddle_only": {"r_star": False, "prune_cutoff": False, "straddle": True},
        "all_on": {"r_star": True, "prune_cutoff": True, "straddle": True},
    }
    by_arm = {"baseline_ported": baseline}
    for name, flags in arms.items():
        # prune_only / straddle_only still need *some* r* — use hand Fashion for fair
        # isolation of prune/straddle; r_star_only uses probe.
        rates: dict[str, float] | None
        if flags["r_star"]:
            rates = {}
        else:
            rates = dict(FASHION_R)  # correct r* so prune/straddle isolation is clean
        by_arm[name] = run_plastic(
            dataset="fashion_mnist",
            seed=seed,
            epochs=epochs,
            cfg=make_cfg(
                target_rates=rates,
                self_cal=flags,
                mu_r=mu,
                max_new=1,
            ),
            warm_start_probe=bool(flags["r_star"] or flags["prune_cutoff"] or flags["straddle"]),
            label=f"abl_{name}",
        )
    # Attribution: r* recovery = match Fashion r* + lower contraction vs ported
    def recovers(arm: dict[str, Any]) -> bool:
        fr = arm.get("final_r_star") or {}
        if not fr:
            return False
        m = within_tol(fr, FASHION_R, TOL)
        return bool(m["pass"]) and arm["mean_contraction_frac"] < baseline[
            "mean_contraction_frac"
        ] - 0.05

    attribution = {
        name: {
            "recovers_fashion_r_star": recovers(by_arm[name])
            if "r_star" in name or name == "all_on"
            else False,
            "mean_contraction_frac": by_arm[name]["mean_contraction_frac"],
            "final_r_star": by_arm[name].get("final_r_star"),
            "swing": by_arm[name]["swing"],
            "final_acc": by_arm[name]["final_acc"],
        }
        for name in by_arm
    }
    # Required vs optional
    r_star_does = recovers(by_arm["r_star_only"])
    required = ["r_star"] if r_star_does else []
    optional = []
    if not recovers(by_arm["prune_only"]) and by_arm["prune_only"]["mean_contraction_frac"] >= by_arm[
        "r_star_only"
    ]["mean_contraction_frac"] - 0.02:
        optional.append("prune_cutoff")
    else:
        # prune alone with correct r* — measure thrash help only
        if by_arm["prune_only"]["swing"]["max_abs_swing_pp"] < by_arm["baseline_ported"][
            "swing"
        ]["max_abs_swing_pp"] - 2:
            required.append("prune_cutoff")
        else:
            optional.append("prune_cutoff")
    optional.append("straddle")  # already live formula; layer EMA is diagnostic

    out = {
        "stage": "ablation",
        "attribution": attribution,
        "required_conversions": required,
        "optional_conversions": optional,
        "rule": PREREG["ablation_rule"],
        "by_arm": {k: {kk: vv for kk, vv in v.items() if kk != "trajectory"} for k, v in by_arm.items()},
    }
    (OUT / "ablation.json").write_text(json.dumps(out, indent=2))
    print(f"§4.5 required={required} optional={optional}", flush=True)
    return out


def stage_kmnist(epochs: int, seed: int, mu: float) -> dict[str, Any]:
    print("=== §4.6 KMNIST generalization ===", flush=True)
    res = run_plastic(
        dataset="kmnist",
        seed=seed,
        epochs=epochs,
        cfg=make_cfg(
            target_rates={},
            self_cal={"r_star": True, "prune_cutoff": False, "straddle": False},
            mu_r=mu,
            max_new=1,
        ),
        warm_start_probe=True,
        label="kmnist_selfcal",
    )
    run = bool(res["runaway"].get("runaway"))
    thrash = bool(res["swing"].get("thrash"))
    rates_ok = all(v > 0.05 for v in res["final_r_star"].values())
    verdict = "PASS" if (not run and not thrash and rates_ok) else "FAIL"
    out = {
        "stage": "kmnist",
        "verdict": verdict,
        "final_r_star": res["final_r_star"],
        "probe_rates": res.get("probe_rates"),
        "swing": res["swing"],
        "runaway": res["runaway"],
        "final_acc": res["final_acc"],
        "result": {k: v for k, v in res.items() if k != "trajectory"},
    }
    (OUT / "kmnist.json").write_text(json.dumps(out, indent=2))
    print(
        f"§4.6 verdict={verdict} r*={res['final_r_star']} thrash={thrash} runaway={run}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "mnist_equiv",
            "fashion_adapt",
            "mu_sweep",
            "feedback",
            "ablation",
            "kmnist",
        ],
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mu-r-star",
        type=float,
        default=None,
        help="Override locked μ after sweep (default: read mu_sweep.json or 0.999)",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stages = (
        ["mnist_equiv", "mu_sweep", "feedback", "fashion_adapt", "ablation", "kmnist"]
        if args.stage == "all"
        else [args.stage]
    )

    mu = args.mu_r_star
    if mu is None:
        ms = OUT / "mu_sweep.json"
        if ms.exists():
            mu = float(json.loads(ms.read_text())["locked_mu_r_star"])
        else:
            mu = 0.999

    summary: dict[str, Any] = {"stages": {}, "locked_mu_r_star": mu}
    for st in stages:
        if st == "mnist_equiv":
            summary["stages"][st] = stage_mnist_equiv(args.epochs, args.seed)
        elif st == "fashion_adapt":
            summary["stages"][st] = stage_fashion_adapt(args.epochs, args.seed)
        elif st == "mu_sweep":
            summary["stages"][st] = stage_mu_sweep(args.epochs, args.seed)
            mu = float(summary["stages"][st]["locked_mu_r_star"])
            summary["locked_mu_r_star"] = mu
        elif st == "feedback":
            summary["stages"][st] = stage_feedback(args.epochs, args.seed, mu)
        elif st == "ablation":
            summary["stages"][st] = stage_ablation(args.epochs, args.seed, mu)
        elif st == "kmnist":
            summary["stages"][st] = stage_kmnist(args.epochs, args.seed, mu)

    # Compact headline without full trajectories
    headline = {
        "locked_mu_r_star": summary["locked_mu_r_star"],
        "verdicts": {
            k: v.get("verdict") for k, v in summary["stages"].items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(headline, indent=2))
    # Full dump without huge nested trajectories in top summary
    (OUT / "summary_full_meta.json").write_text(
        json.dumps(headline, indent=2)
    )
    print("\n=== BATTERY HEADLINE ===", flush=True)
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
