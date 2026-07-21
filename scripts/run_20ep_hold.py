#!/usr/bin/env python3
"""20-epoch hold of establish + stays-specialized under the locked full stack.

Official recipe (configs/phase1a_fixed.yaml knobs):
  fixed threshold + gated_novelty (k_min=11) + gap_imprinted
  (sep-power / open-fan / straddle) + competitive prune + grace=3.

Also records structure snapshots (seed 0 by default) and renders static +
animation for the full 20-epoch sequence.

Usage:
  python scripts/run_20ep_hold.py --seeds 0,1,2
  python scripts/run_20ep_hold.py --seeds 0 --no-viz

Artifacts:
  results/phase1a_diagnostics/gap_imprint_survival_20ep_hold/
    summary.json
    seed_*.json
    seed_0/structure_snapshots/
    visualizations/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet  # noqa: E402
from neuronca.plasticity.masks import BIRTH_KIND_NAMES, NEURON_ORIGIN_INIT  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.structure_snapshot import StructureSnapshotRecorder  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)
from run_gap_imprint_survival import (  # noqa: E402
    GATE_GOW,
    GATE_ONOFF,
    GRACE,
    STAY_GOW_DECAY,
    STAY_GOW_FINAL,
    STAY_ONOFF_DECAY,
    STAY_ONOFF_FINAL,
    UnitTrack,
    _cfg,
    _mean_rates,
    _passes_birth_gate,
    _passes_stays_specialized,
    _probe_posts,
    _profile,
)
from visualize_structure import render_animation, render_static  # noqa: E402

OUT_ROOT_FIXED = (
    _REPO / "results" / "phase1a_diagnostics" / "gap_imprint_survival_20ep_hold"
)
OUT_ROOT_SLIDING = (
    _REPO
    / "results"
    / "phase1a_diagnostics"
    / "gap_imprint_survival_20ep_hold_sliding"
)
EPOCHS_HOLD = 20


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one(
    *,
    seed: int,
    epochs: int,
    record_snapshots: bool,
    seed_dir: Path,
    threshold_kind: str = "fixed",
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=str(_REPO / "data" / "mnist"),
        batch_size=128,
        num_workers=0,
        seed=seed,
        probe_size=1024,
    )
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = _cfg(str(threshold_kind))
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, name="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)

    snap_rec: StructureSnapshotRecorder | None = None
    if record_snapshots:
        seed_dir.mkdir(parents=True, exist_ok=True)
        snap_rec = StructureSnapshotRecorder.from_config(
            seed_dir,
            total_epochs=epochs,
            record=True,
            cadence="every_epoch",
        )

    live: dict[tuple[str, int], UnitTrack] = {}
    finished: list[UnitTrack] = []
    structure_trace: list[dict[str, Any]] = []
    val_hist: list[dict[str, float]] = []
    global_step = 0

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

        posts = _probe_posts(net, loaders["probe_loader"], device)
        for key, tr in list(live.items()):
            layer, slot = key
            if not bool(net.active_mask(layer)[slot]):
                continue
            age = epoch - tr.birth_epoch
            prof = _profile(posts[layer][:, slot], tr.gap_mask)
            tr.profiles.append(
                {"epoch": epoch, "phase": "pre_structure", "age": age, **prof}
            )
            if snap_rec is not None:
                snap_rec.set_specialization(layer, slot, prof)

        slog = orch.step(
            net,
            scores,
            optimizer=opt,
            epoch=epoch,
            signal=signal,
            gated_novelty=gated,
        )

        prune_origins: Counter[str] = Counter()
        for prec in slog.neuron_prune_records:
            kind_code = int(prec.get("birth_kind", 0) or 0)
            origin_code = int(prec.get("neuron_origin", 0) or 0)
            name = BIRTH_KIND_NAMES.get(kind_code, "unknown")
            if kind_code == 0 and origin_code == NEURON_ORIGIN_INIT:
                name = "init"
            prune_origins[name] += 1

        for key, tr in list(live.items()):
            layer, slot = key
            if not bool(net.active_mask(layer)[slot]):
                tr.pruned_epoch = epoch
                finished.append(tr)
                del live[key]

        posts_after = _probe_posts(net, loaders["probe_loader"], device)
        grow_idx = {h: 0 for h in HIDDEN_LAYERS}
        for layer, slot in slog.neuron_grow_keys:
            ranked = (
                list(gated.by_layer[layer].ranked_triggers)
                if layer in gated.by_layer
                else []
            )
            gi = grow_idx[layer]
            grow_idx[layer] = gi + 1
            trigger = int(ranked[gi]) if gi < len(ranked) else None
            gmask = None
            if trigger is not None and (layer, trigger) in gated.gap_masks:
                gmask = gated.gap_masks[(layer, trigger)].bool().cpu().clone()
            if gmask is None:
                continue
            prof = _profile(posts_after[layer][:, int(slot)], gmask)
            tr = UnitTrack(
                layer=layer,
                slot=int(slot),
                trigger=trigger,
                birth_epoch=epoch,
                gap_mask=gmask,
            )
            tr.profiles.append(
                {"epoch": epoch, "phase": "birth", "age": 0, **prof}
            )
            live[(layer, int(slot))] = tr
            if snap_rec is not None:
                snap_rec.set_specialization(layer, int(slot), prof)

        if snap_rec is not None:
            snap_rec.maybe_record(epoch, net, signal=signal, structure_log=slog)

        structure_trace.append(
            {
                "epoch": epoch,
                "neurons_grown": int(slog.neurons_grown),
                "neurons_pruned": int(slog.neurons_pruned),
                "novelty_under_floor_total": int(slog.novelty_under_floor_total),
                "prune_origins": dict(prune_origins),
                "hidden_counts": net.hidden_counts(),
                "edge_counts": net.edge_counts(by_origin=True),
                "n_live_tracked": len(live),
            }
        )
        vm = evaluate(net, loaders["test_loader"], device=device)
        val_hist.append({"epoch": float(epoch), **vm})
        print(
            f"  ep{epoch:02d}: grown={slog.neurons_grown} pruned={slog.neurons_pruned} "
            f"live_track={len(live)} val_acc={vm['acc']:.3f} "
            f"origins={dict(prune_origins)}",
            flush=True,
        )

    for tr in live.values():
        tr.pruned_epoch = None
        finished.append(tr)

    step_now = int(net.cooldown.plasticity_step)
    unit_rows: list[dict[str, Any]] = []
    for tr in finished:
        bp = tr.birth_profile() or {}
        fp = tr.final_profile() or {}
        max_age = max((int(p.get("age", 0)) for p in tr.profiles), default=0)
        # Plasticity age for alive units (more precise for right-censoring)
        age_end: int | None = None
        if tr.pruned_epoch is None and bool(net.active_mask(tr.layer)[tr.slot]):
            bstep = int(net.birth_step(tr.layer)[tr.slot].item())
            age_end = step_now - bstep if bstep >= 0 else max_age
        else:
            age_end = (
                int(tr.pruned_epoch) - tr.birth_epoch
                if tr.pruned_epoch is not None
                else max_age
            )
        established = max_age >= GRACE or (age_end is not None and age_end >= GRACE)
        birth_gate = _passes_birth_gate(bp) if bp else False
        stays = (
            _passes_stays_specialized(bp, fp)
            if (established and birth_gate and bp and fp)
            else False
        )
        unit_rows.append(
            {
                "layer": tr.layer,
                "slot": tr.slot,
                "trigger": tr.trigger,
                "birth_epoch": tr.birth_epoch,
                "pruned_epoch": tr.pruned_epoch,
                "max_age": max_age,
                "age_end": age_end,
                "alive_end": tr.pruned_epoch is None,
                "established": established,
                "birth_gate": birth_gate,
                "stays_specialized": stays,
                "birth_profile": bp,
                "final_profile": fp,
                "trajectory": tr.profiles,
            }
        )

    n_births = len(unit_rows)
    n_gate = sum(1 for u in unit_rows if u["birth_gate"])
    n_est = sum(1 for u in unit_rows if u["established"])
    n_est_gate = sum(
        1 for u in unit_rows if u["established"] and u["birth_gate"]
    )
    n_stays = sum(
        1
        for u in unit_rows
        if u["established"] and u["birth_gate"] and u["stays_specialized"]
    )

    alive = [u for u in unit_rows if u["alive_end"]]
    alive_tested = [u for u in alive if int(u.get("age_end") or u["max_age"]) >= GRACE]
    alive_censored = [
        u for u in alive if int(u.get("age_end") or u["max_age"]) < GRACE
    ]
    alive_age_hist = Counter(
        int(u.get("age_end") or u["max_age"]) for u in alive
    )

    by_age_est: dict[int, list[dict[str, float]]] = {}
    for u in unit_rows:
        if not (u["established"] and u["birth_gate"]):
            continue
        for p in u["trajectory"]:
            age = int(p.get("age") or 0)
            by_age_est.setdefault(age, []).append(p)

    def mean_age_est(key: str) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for age, plist in sorted(by_age_est.items()):
            vals = [
                float(p[key])
                for p in plist
                if p.get(key) is not None and p[key] == p[key]
            ]
            out[str(age)] = sum(vals) / len(vals) if vals else None
        return out

    frac_est = n_est / n_births if n_births else 0.0
    frac_gate = n_gate / n_births if n_births else 0.0
    frac_stays_of_est_gate = n_stays / n_est_gate if n_est_gate else 0.0
    frac_stays_of_births = n_stays / n_births if n_births else 0.0

    if n_est_gate == 0:
        label = "ESTABLISH_FAIL"
    elif frac_stays_of_est_gate >= 0.5:
        label = "SPECIALIST_SURVIVES"
    else:
        label = "ESTABLISH_ONLY"

    first_prune_ep = next(
        (r["epoch"] for r in structure_trace if r["neurons_pruned"] > 0), None
    )
    first_wave = (
        structure_trace[first_prune_ep].get("prune_origins")
        if first_prune_ep is not None
        else {}
    )

    return {
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "threshold_kind": str(threshold_kind),
            "stack": "gated_novelty+gap_imprinted(sep_power,open_fan,straddle_B)",
            "hold": "20ep_establish_stays",
            "gates": {
                "birth_gow": GATE_GOW,
                "birth_on_off": GATE_ONOFF,
                "stay_gow_final": STAY_GOW_FINAL,
                "stay_gow_decay": STAY_GOW_DECAY,
                "stay_on_off_final": STAY_ONOFF_FINAL,
                "stay_on_off_decay": STAY_ONOFF_DECAY,
                "grace": GRACE,
            },
        },
        "counts": {
            "n_births": n_births,
            "n_birth_gate": n_gate,
            "n_established": n_est,
            "n_established_and_gated": n_est_gate,
            "n_stays_specialized": n_stays,
            "n_alive_end": len(alive),
            "n_alive_end_age_ge_grace_tested": len(alive_tested),
            "n_alive_end_age_lt_grace_right_censored": len(alive_censored),
        },
        "rates": {
            "frac_births_passing_birth_gate": frac_gate,
            "frac_established_of_births": frac_est,
            "frac_established_of_gated": (n_est_gate / n_gate if n_gate else 0.0),
            "frac_of_established_gated_passing_stays_specialized": frac_stays_of_est_gate,
            "frac_stays_of_all_births": frac_stays_of_births,
        },
        "right_censoring": {
            "alive_end": len(alive),
            "alive_tested_age_ge_grace": len(alive_tested),
            "alive_right_censored_age_lt_grace": len(alive_censored),
            "alive_age_histogram": dict(sorted(alive_age_hist.items())),
            "note": (
                "Do not quote alive_end as establishment; "
                "establishment = ever age≥grace."
            ),
        },
        "optional_label": label,
        "mean_trajectory_established_gated": {
            "gap_over_whole_by_age": mean_age_est("gap_over_whole"),
            "on_off_by_age": mean_age_est("on_off"),
        },
        "first_prune_wave": {"epoch": first_prune_ep, "origins": first_wave},
        "structure_trace": structure_trace,
        "val_history": val_hist,
        "hidden_counts_final": net.hidden_counts(),
        "edge_counts_final": net.edge_counts(by_origin=True),
        "val_acc_final": val_hist[-1]["acc"] if val_hist else None,
        "n_structure_snapshots": len(snap_rec.written) if snap_rec else 0,
        "units": unit_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=EPOCHS_HOLD)
    ap.add_argument(
        "--threshold",
        choices=["fixed", "sliding"],
        default="fixed",
        help="Homeostatic threshold policy (default fixed; sliding = Variant B retest)",
    )
    ap.add_argument(
        "--snapshot-seed",
        type=int,
        default=0,
        help="Seed for which to record structure snapshots (default 0; -1 = none)",
    )
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    epochs = int(args.epochs)
    threshold_kind = str(args.threshold)
    OUT_ROOT = OUT_ROOT_SLIDING if threshold_kind == "sliding" else OUT_ROOT_FIXED

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        print(
            f"\n=== 20-ep hold {threshold_kind} seed {seed} ===",
            flush=True,
        )
        seed_dir = OUT_ROOT / f"seed_{seed}"
        record = args.snapshot_seed == seed
        r = run_one(
            seed=seed,
            epochs=epochs,
            record_snapshots=record,
            seed_dir=seed_dir,
            threshold_kind=threshold_kind,
        )
        seed_results.append(r)
        # Drop full unit trajectories from disk if huge — keep summary fields.
        # Still write full seed file for audit (units included).
        (OUT_ROOT / f"seed_{seed}.json").write_text(
            json.dumps(r, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  births={r['counts']['n_births']} "
            f"gate={r['rates']['frac_births_passing_birth_gate']:.3f} "
            f"est={r['rates']['frac_established_of_births']:.3f} "
            f"stays|est_gate={r['rates']['frac_of_established_gated_passing_stays_specialized']:.3f} "
            f"alive={r['counts']['n_alive_end']} "
            f"(tested={r['counts']['n_alive_end_age_ge_grace_tested']}, "
            f"censored={r['counts']['n_alive_end_age_lt_grace_right_censored']}) "
            f"label={r['optional_label']} "
            f"acc={r['val_acc_final']}",
            flush=True,
        )

    mean_rates = _mean_rates(seed_results)
    # Compare to 6-ep primary fixed mean if present
    primary_6ep = None
    p6 = (
        _REPO
        / "results"
        / "phase1a_diagnostics"
        / "gap_imprint_survival_comparison.json"
    )
    if p6.is_file():
        try:
            payload = json.loads(p6.read_text())
            key = "sliding_mean" if threshold_kind == "sliding" else "fixed_mean"
            primary_6ep = payload.get(key)
        except json.JSONDecodeError:
            primary_6ep = None

    comparison_to_6ep: dict[str, Any] = {}
    if primary_6ep:
        for k, v20 in mean_rates.items():
            if k.startswith("mean_"):
                continue
            v6 = primary_6ep.get(k)
            if v20 is not None and v6 is not None:
                comparison_to_6ep[k] = {
                    "ep6": v6,
                    "ep20": v20,
                    "delta_pp": (float(v20) - float(v6)) * 100.0,
                }

    summary = {
        "protocol": {
            "threshold_kind": threshold_kind,
            "seeds": seeds,
            "epochs": epochs,
            "stack": "gated_novelty+gap_imprinted(sep_power,open_fan,straddle_B)",
            "hold": "20ep_establish_stays",
            "config": (
                "configs/phase1a_sliding.yaml knobs"
                if threshold_kind == "sliding"
                else "configs/phase1a_fixed.yaml knobs"
            ),
            "note": (
                "Full-horizon hold of establish + stays-specialized; "
                "compares to 6-ep primary mechanism rates. Sliding arm is the "
                "long-horizon retest of B after early-cohort erosion was found under fixed."
                if threshold_kind == "sliding"
                else (
                    "Full-horizon hold of establish + stays-specialized; "
                    "compares to 6-ep primary mechanism rates."
                )
            ),
        },
        "mean_rates": mean_rates,
        "comparison_to_6ep_primary_fixed": comparison_to_6ep,
        "per_seed": [
            {
                "seed": r["protocol"]["seed"],
                "counts": r["counts"],
                "rates": r["rates"],
                "right_censoring": r["right_censoring"],
                "optional_label": r["optional_label"],
                "val_acc_final": r["val_acc_final"],
                "first_prune_wave": r["first_prune_wave"],
                "mean_trajectory_established_gated": r[
                    "mean_trajectory_established_gated"
                ],
            }
            for r in seed_results
        ],
        "verdict": None,
    }

    # Hold reading vs 6-ep
    stays20 = mean_rates.get(
        "frac_of_established_gated_passing_stays_specialized"
    )
    est20 = mean_rates.get("frac_established_of_births")
    if stays20 is not None and est20 is not None and primary_6ep:
        stays6 = primary_6ep.get(
            "frac_of_established_gated_passing_stays_specialized", 0
        )
        est6 = primary_6ep.get("frac_established_of_births", 0)
        # Hold if within ~10 pp of 6-ep rates (mechanism stable)
        if abs(stays20 - stays6) <= 0.10 and abs(est20 - est6) <= 0.10 and stays20 >= 0.5:
            summary["verdict"] = "HOLD_CONFIRMS_6EP"
        elif stays20 >= 0.5 and est20 >= 0.3:
            summary["verdict"] = "HOLD_OK_RATES_SHIFTED"
        elif est20 < 0.1:
            summary["verdict"] = "HOLD_ESTABLISH_FAIL"
        else:
            summary["verdict"] = "HOLD_SEE_RATES"
    elif stays20 is not None and stays20 >= 0.5 and est20 and est20 >= 0.3:
        summary["verdict"] = "HOLD_OK"
    else:
        summary["verdict"] = "HOLD_SEE_RATES"

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== 20-ep HOLD SUMMARY ===", flush=True)
    print(json.dumps({"verdict": summary["verdict"], "mean_rates": mean_rates, "vs_6ep": comparison_to_6ep}, indent=2), flush=True)

    # Visualization for snapshot seed
    if not args.no_viz and args.snapshot_seed in seeds:
        viz_dir = OUT_ROOT / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        snap_root = OUT_ROOT / f"seed_{args.snapshot_seed}"
        print(f"\n=== Rendering structure viz from {snap_root} ===", flush=True)
        try:
            render_static(
                snap_root,
                viz_dir / "structure_static_late.png",
                tag="late" if False else f"epoch_{epochs - 1}",
            )
            # Also early / mid callouts
            mid_ep = epochs // 2 - 1  # 0-based mid
            early_ep = 0
            render_static(
                snap_root,
                viz_dir / "structure_static_early.png",
                epoch=early_ep,
            )
            render_static(
                snap_root,
                viz_dir / "structure_static_mid.png",
                epoch=mid_ep,
            )
            # Prefer a frame with prunes for death markers if available
            sr0 = next(
                r for r in seed_results if r["protocol"]["seed"] == args.snapshot_seed
            )
            prune_eps = [
                t["epoch"]
                for t in sr0["structure_trace"]
                if t["neurons_pruned"] > 0
            ]
            if prune_eps:
                # mid prune epoch for callout
                pe = prune_eps[min(len(prune_eps) // 2, len(prune_eps) - 1)]
                render_static(
                    snap_root,
                    viz_dir / f"structure_static_epoch{pe}_prunes.png",
                    epoch=pe,
                )
            render_animation(
                snap_root,
                viz_dir / "structure_animation_20ep.gif",
                fps=1.5,
            )
            print(f"wrote visualizations under {viz_dir}", flush=True)
        except Exception as exc:
            print(f"WARNING: visualization failed: {exc}", flush=True)
            summary["viz_error"] = str(exc)
            (OUT_ROOT / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )

    # Human README
    lines = [
        f"# 20-epoch hold — {summary['verdict']}",
        "",
        f"Stack: fixed + gated_novelty + gap_imprinted  |  seeds={seeds}  |  epochs={epochs}",
        "",
        "## Mean rates",
        "",
    ]
    for k, v in (mean_rates or {}).items():
        if v is not None and not str(k).startswith("mean_n"):
            lines.append(f"- `{k}`: **{v:.4f}**" if isinstance(v, float) else f"- `{k}`: {v}")
    if comparison_to_6ep:
        lines += ["", "## vs 6-ep primary (fixed mean)", ""]
        for k, d in comparison_to_6ep.items():
            lines.append(
                f"- `{k}`: 6ep={d['ep6']:.4f} → 20ep={d['ep20']:.4f} "
                f"(Δ {d['delta_pp']:+.1f} pp)"
            )
    lines += [
        "",
        "## Per seed",
        "",
    ]
    for ps in summary["per_seed"]:
        lines.append(
            f"- seed {ps['seed']}: est={ps['rates']['frac_established_of_births']:.3f} "
            f"stays|est∧gate={ps['rates']['frac_of_established_gated_passing_stays_specialized']:.3f} "
            f"alive={ps['counts']['n_alive_end']} "
            f"(tested={ps['counts']['n_alive_end_age_ge_grace_tested']}, "
            f"censored={ps['counts']['n_alive_end_age_lt_grace_right_censored']}) "
            f"label={ps['optional_label']}"
        )
    lines += [
        "",
        "## Visualizations",
        "",
        f"`visualizations/` from seed {args.snapshot_seed} (every-epoch snapshots).",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_ROOT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
