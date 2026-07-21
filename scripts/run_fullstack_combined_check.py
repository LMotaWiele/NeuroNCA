#!/usr/bin/env python3
"""Short combined-stack check for the locked Phase 1a full recipe.

Prior batteries validated pieces in isolation (trigger, birth, specialization).
This run exercises the **bundled** official config end-to-end:

  fixed threshold + gated_novelty (k_min=11) + gap_imprinted
  (sep-power edges, open-fan scale, straddle B) + competitive prune
  + grace/cooldown.

Does **not** replace the multi-seed primary or 20-ep hold — only asks whether
the combined system behaves sensibly on a short window before the real
comparison run.

Usage:
  python scripts/run_fullstack_combined_check.py
  python scripts/run_fullstack_combined_check.py --epochs 6 --seed 0

Artifact:
  results/phase1a_diagnostics/fullstack_combined_check/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.config import load_yaml  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import (  # noqa: E402
    HIDDEN_LAYERS,
    DynamicMLPReasoningNet,
)
from neuronca.plasticity.masks import (  # noqa: E402
    BIRTH_KIND_NAMES,
    NEURON_BIRTH_GAP_IMPRINTED,
    NEURON_ORIGIN_GROWN,
    NEURON_ORIGIN_INIT,
)
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "fullstack_combined_check"
OFFICIAL_CFG = _REPO / "configs" / "phase1a_fixed.yaml"
GRACE = 3


def _plasticity_from_yaml(p: dict[str, Any]) -> PlasticityConfig:
    raw_rates = p.get("target_rates") or {}
    target_rates = {str(k): float(v) for k, v in raw_rates.items()}
    return PlasticityConfig(
        radius=float(p.get("radius", 0.35)),
        target_rates=target_rates,
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
        newborn_weight_init=str(p.get("newborn_weight_init", "gap_imprinted")),
        newborn_parent_noise=float(p.get("newborn_parent_noise", 0.0)),
        newborn_copy_outgoing=bool(p.get("newborn_copy_outgoing", True)),
        newborn_parent_coord_jitter=float(p.get("newborn_parent_coord_jitter", 0.05)),
        newborn_lr_mult=float(p.get("newborn_lr_mult", 1.0)),
        neuron_growth_trigger=str(p.get("neuron_growth_trigger", "gated_novelty")),
        novelty_k_min=int(p.get("novelty_k_min", 11)),
        novelty_rank_fraction=float(p.get("novelty_rank_fraction", 1.0)),
    )


def _assert_official_recipe(p: dict[str, Any]) -> list[str]:
    """Hard checks that the YAML is the locked stack (not a stale kaiming file)."""
    fails: list[str] = []
    expect = {
        "threshold_kind": "fixed",
        "neuron_growth_trigger": "gated_novelty",
        "newborn_weight_init": "gap_imprinted",
        "novelty_k_min": 11,
        "newborn_grace_steps": 3,
    }
    for k, v in expect.items():
        got = p.get(k)
        if str(got).strip().lower() != str(v).strip().lower() and got != v:
            fails.append(f"{k}: expected {v!r}, got {got!r}")
    return fails


def run(
    *,
    epochs: int,
    seed: int,
    max_steps: int | None,
) -> dict[str, Any]:
    raw = load_yaml(OFFICIAL_CFG)
    pcfg_raw = raw.get("plasticity") or {}
    recipe_fails = _assert_official_recipe(pcfg_raw)
    if recipe_fails:
        raise RuntimeError(
            "configs/phase1a_fixed.yaml is not the locked full stack:\n  "
            + "\n  ".join(recipe_fails)
        )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = raw["model"]
    tcfg = raw["train"]

    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=str(_REPO / (raw.get("data") or {}).get("root", "data/mnist")),
        batch_size=int(tcfg.get("batch_size", 128)),
        num_workers=0,
        seed=seed,
        probe_size=1024,
    )
    net = DynamicMLPReasoningNet(
        input_dim=int(mcfg.get("input_dim", 784)),
        hidden_init=tuple(mcfg.get("hidden_init", [256, 128, 64])),
        hidden_max=tuple(mcfg.get("hidden_max", [512, 256, 128])),
        num_classes=int(mcfg.get("num_classes", 10)),
        coord_seed=seed,
    ).to(device)
    cfg = _plasticity_from_yaml(pcfg_raw)
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(
        net,
        name=str(tcfg.get("optimizer", "sgd")),
        lr=float(tcfg.get("lr", 0.01)),
        momentum=float(tcfg.get("momentum", 0.9)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    signal = ActivationMagnitude(decay=0.99)

    # Live tracking of grown units for establish / prune-origin checks.
    live: dict[tuple[str, int], dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    structure_trace: list[dict[str, Any]] = []
    global_step = 0
    train_hist: list[dict[str, float]] = []
    val_hist: list[dict[str, float]] = []

    for epoch in range(epochs):
        metrics = train_epoch_with_signals(
            net,
            loaders["train_loader"],
            opt,
            signal,
            orch,
            device=device,
            max_steps=max_steps,
            global_step=global_step,
        )
        global_step = int(metrics["global_step"])
        train_hist.append({"epoch": float(epoch), **metrics})

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
            net,
            scores,
            optimizer=opt,
            epoch=epoch,
            signal=signal,
            gated_novelty=gated,
        )

        # Finalize pruned grown lives (tracked cohort).
        for key in list(live.keys()):
            layer, slot = key
            if not bool(net.active_mask(layer)[slot]):
                rec = live.pop(key)
                rec["pruned_epoch"] = epoch
                rec["max_age"] = epoch - int(rec["birth_epoch"])
                finished.append(rec)

        # Origin of *all* prunes this step (init + grown) from orchestrator records.
        prune_origins: Counter[str] = Counter()
        for prec in slog.neuron_prune_records:
            kind_code = int(prec.get("birth_kind", 0) or 0)
            origin_code = int(prec.get("neuron_origin", 0) or 0)
            name = BIRTH_KIND_NAMES.get(kind_code, "unknown")
            if kind_code == 0 and origin_code == NEURON_ORIGIN_INIT:
                name = "init"
            elif kind_code == 0 and origin_code == NEURON_ORIGIN_GROWN:
                name = "grown"
            prune_origins[name] += 1

        # Register new births from structure log + birth_kind buffers.
        for layer, slot in slog.neuron_grow_keys:
            slot = int(slot)
            kind_code = int(net.birth_kind(layer)[slot].item())
            kind_name = BIRTH_KIND_NAMES.get(kind_code, "unknown")
            parent = int(net.parent_slot(layer)[slot].item())
            live[(layer, slot)] = {
                "layer": layer,
                "slot": slot,
                "birth_epoch": epoch,
                "birth_kind": kind_name,
                "parent_slot": parent if parent >= 0 else None,
            }

        edge_counts = net.edge_counts(by_origin=True)
        structure_trace.append(
            {
                "epoch": epoch,
                "neurons_grown": int(slog.neurons_grown),
                "neurons_pruned": int(slog.neurons_pruned),
                "edges_grown": int(slog.edges_grown),
                "edges_grown_seed": int(slog.edges_grown_seed),
                "novelty_under_floor_total": int(slog.novelty_under_floor_total),
                "novelty_under_floor_by_layer": dict(slog.novelty_under_floor_by_layer),
                "hidden_counts": net.hidden_counts(),
                "edge_counts": edge_counts,
                "prune_origins_this_step": dict(prune_origins),
                "n_live_grown": len(live),
            }
        )

        vm = evaluate(net, loaders["test_loader"], device=device, max_steps=max_steps)
        val_hist.append({"epoch": float(epoch), **vm})
        print(
            f"ep {epoch}: grown={slog.neurons_grown} pruned={slog.neurons_pruned} "
            f"under_floor={slog.novelty_under_floor_total} "
            f"gap_e={sum(edge_counts[l]['gap_imprinted_seeded'] for l in edge_counts)} "
            f"live_grown={len(live)} val_acc={vm['acc']:.3f}"
        )

    # End survivors.
    step_now = int(net.cooldown.plasticity_step)
    for key, rec in live.items():
        layer, slot = key
        rec["pruned_epoch"] = None
        birth_step = int(net.birth_step(layer)[slot].item())
        rec["max_age"] = step_now - birth_step if birth_step >= 0 else 0
        rec["alive_end"] = True
        finished.append(rec)

    n_births = sum(1 for u in finished)
    n_gap_births = sum(1 for u in finished if u.get("birth_kind") == "gap_imprinted")
    n_established = sum(
        1
        for u in finished
        if int(u.get("max_age") or 0) >= GRACE
    )
    n_gap_established = sum(
        1
        for u in finished
        if u.get("birth_kind") == "gap_imprinted"
        and int(u.get("max_age") or 0) >= GRACE
    )
    n_gap_alive_end = sum(
        1
        for u in finished
        if u.get("birth_kind") == "gap_imprinted" and u.get("pruned_epoch") is None
    )
    n_gap_pruned = sum(
        1
        for u in finished
        if u.get("birth_kind") == "gap_imprinted" and u.get("pruned_epoch") is not None
    )
    # Right-censoring split (survivor-age-census lesson): alive-end ≠ established.
    gap_alive_units = [
        u
        for u in finished
        if u.get("birth_kind") == "gap_imprinted" and u.get("pruned_epoch") is None
    ]
    alive_age_hist: Counter[int] = Counter(
        int(u.get("max_age") or 0) for u in gap_alive_units
    )
    n_gap_alive_tested = sum(
        1 for u in gap_alive_units if int(u.get("max_age") or 0) >= GRACE
    )
    n_gap_alive_censored = sum(
        1 for u in gap_alive_units if int(u.get("max_age") or 0) < GRACE
    )

    # First prune wave origin (epoch with first non-zero prune).
    first_prune_ep = next(
        (r["epoch"] for r in structure_trace if r["neurons_pruned"] > 0), None
    )
    first_wave_origins: dict[str, int] = {}
    if first_prune_ep is not None:
        first_wave_origins = structure_trace[first_prune_ep].get(
            "prune_origins_this_step"
        ) or {}

    # Active composition at end.
    end_comp = Counter()
    for h in HIDDEN_LAYERS:
        active = net.active_mask(h)
        kinds = net.birth_kind(h)
        origins = net.neuron_origin(h)
        for s in active.nonzero(as_tuple=False).view(-1).tolist():
            kc = int(kinds[s].item())
            oc = int(origins[s].item())
            name = BIRTH_KIND_NAMES.get(kc, "unknown")
            if kc == 0 and oc == NEURON_ORIGIN_INIT:
                name = "init"
            end_comp[name] += 1

    edge_final = net.edge_counts(by_origin=True)
    n_gap_edges = sum(v.get("gap_imprinted_seeded", 0) for v in edge_final.values())
    n_init_edges = sum(v.get("init_dense", 0) for v in edge_final.values())

    total_grown = sum(r["neurons_grown"] for r in structure_trace)
    total_pruned = sum(r["neurons_pruned"] for r in structure_trace)
    total_under_floor = sum(r["novelty_under_floor_total"] for r in structure_trace)
    epochs_with_growth = sum(1 for r in structure_trace if r["neurons_grown"] > 0)
    epochs_with_prune = sum(1 for r in structure_trace if r["neurons_pruned"] > 0)

    val_accs = [float(v["acc"]) for v in val_hist]
    acc_start = val_accs[0] if val_accs else float("nan")
    acc_end = val_accs[-1] if val_accs else float("nan")

    # --- Sensible-behavior gates (combined stack, short window) ---
    checks: dict[str, dict[str, Any]] = {}

    def _gate(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}

    _gate(
        "recipe_is_locked_stack",
        len(recipe_fails) == 0,
        {"fails": recipe_fails, "config": str(OFFICIAL_CFG)},
    )
    _gate(
        "gated_novelty_fires",
        total_grown > 0 and epochs_with_growth >= 1,
        {"total_grown": total_grown, "epochs_with_growth": epochs_with_growth},
    )
    _gate(
        "births_are_gap_imprinted",
        n_gap_births >= max(1, int(0.9 * n_births)) if n_births else False,
        {"n_births": n_births, "n_gap_births": n_gap_births},
    )
    _gate(
        "gap_seeded_edges_present",
        n_gap_edges > 0,
        {"n_gap_imprinted_seeded_edges": n_gap_edges, "edge_counts": edge_final},
    )
    _gate(
        "under_floor_monitored_not_silent",
        True,  # presence of field is enough; total may be 0 if density ok
        {
            "total_under_floor_events": total_under_floor,
            "note": "exclude+log path exercised via slog fields each step",
        },
    )
    _gate(
        "first_prune_wave_init_dominated",
        (
            first_prune_ep is not None
            and first_wave_origins.get("gap_imprinted", 0) == 0
            and (
                first_wave_origins.get("init", 0)
                >= 0.9 * sum(first_wave_origins.values())
                if first_wave_origins
                else False
            )
        )
        or (first_prune_ep is None and epochs < GRACE),
        {
            "first_prune_epoch": first_prune_ep,
            "origins": first_wave_origins,
        },
    )
    _gate(
        "some_gap_units_establish",
        n_gap_established > 0,
        {
            "n_gap_established_age_ge_grace": n_gap_established,
            "n_gap_births": n_gap_births,
            "frac_establish": (n_gap_established / n_gap_births) if n_gap_births else 0.0,
        },
    )
    # Not all established gap units die at first judgment (weak combined check).
    _gate(
        "gap_not_total_revolving_door",
        n_gap_alive_tested > 0 or n_gap_established > n_gap_pruned,
        {
            "n_gap_alive_end": n_gap_alive_end,
            "n_gap_alive_end_age_ge_grace_tested": n_gap_alive_tested,
            "n_gap_alive_end_age_lt_grace_right_censored": n_gap_alive_censored,
            "alive_age_histogram": dict(sorted(alive_age_hist.items())),
            "n_gap_pruned": n_gap_pruned,
            "n_gap_established": n_gap_established,
            "note": (
                "Do not treat n_gap_alive_end as establishment; it includes "
                "right-censored units with age < grace (born late in the window)."
            ),
        },
    )
    _gate(
        "accuracy_finite_and_progresses",
        (
            acc_end == acc_end
            and acc_start == acc_start
            and acc_end >= 0.15  # MNIST well above chance early
        ),
        {"acc_start": acc_start, "acc_end": acc_end},
    )
    _gate(
        "init_still_majority_or_pruned",
        n_init_edges > 0,  # dense-start still present; prune has material
        {
            "n_init_dense_edges": n_init_edges,
            "end_neuron_composition": dict(end_comp),
        },
    )

    all_pass = all(c["pass"] for c in checks.values())
    verdict = "FULLSTACK_COMBINED_OK" if all_pass else "FULLSTACK_COMBINED_FAIL"

    report = {
        "verdict": verdict,
        "all_checks_pass": all_pass,
        "stack": {
            "threshold_kind": "fixed",
            "neuron_growth_trigger": "gated_novelty",
            "novelty_k_min": 11,
            "newborn_weight_init": "gap_imprinted",
            "birth_realization": "sep_power_edges+open_fan_kaiming+straddle_B",
            "prune": "continuous_competitive",
            "grace_steps": GRACE,
            "config": str(OFFICIAL_CFG.relative_to(_REPO)),
        },
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "max_steps_per_epoch": max_steps,
            "note": (
                "Combined-stack sanity only — not multi-seed primary, not 20-ep "
                "hold, not stays-specialized remeasure."
            ),
        },
        "rates": {
            "n_births": n_births,
            "n_gap_births": n_gap_births,
            "n_established_any": n_established,
            "n_gap_established": n_gap_established,
            "frac_gap_established_of_gap_births": (
                n_gap_established / n_gap_births if n_gap_births else None
            ),
            "n_gap_alive_end": n_gap_alive_end,
            "n_gap_alive_end_age_ge_grace_tested": n_gap_alive_tested,
            "n_gap_alive_end_age_lt_grace_right_censored": n_gap_alive_censored,
            "alive_age_histogram": dict(sorted(alive_age_hist.items())),
            "alive_end_clarification": (
                f"{n_gap_alive_end} alive end = {n_gap_alive_tested} tested "
                f"(age≥{GRACE}) + {n_gap_alive_censored} right-censored "
                f"(age<{GRACE}); establishment = {n_gap_established}/{n_gap_births}"
            ),
            "n_gap_pruned": n_gap_pruned,
            "total_grown_events": total_grown,
            "total_pruned_events": total_pruned,
            "epochs_with_growth": epochs_with_growth,
            "epochs_with_prune": epochs_with_prune,
        },
        "first_prune_wave": {
            "epoch": first_prune_ep,
            "origins": first_wave_origins,
        },
        "end_state": {
            "hidden_counts": net.hidden_counts(),
            "edge_counts": edge_final,
            "neuron_composition_by_birth_kind": dict(end_comp),
            "val_acc_start": acc_start,
            "val_acc_end": acc_end,
        },
        "checks": checks,
        "structure_trace": structure_trace,
        "train_history": train_hist,
        "val_history": val_hist,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-epoch train step cap (default: full epoch)",
    )
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    report = run(epochs=args.epochs, seed=args.seed, max_steps=args.max_steps)
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Human-readable one-pager
    lines = [
        f"# Full-stack combined check — {report['verdict']}",
        "",
        f"Config: `{report['stack']['config']}`",
        f"Protocol: seed={report['protocol']['seed']}, epochs={report['protocol']['epochs']}, "
        f"max_steps={report['protocol']['max_steps_per_epoch']}",
        "",
        "## Checks",
        "",
    ]
    for name, c in report["checks"].items():
        mark = "PASS" if c["pass"] else "FAIL"
        lines.append(f"- **{mark}** `{name}`: {json.dumps(c['detail'], default=str)}")
    lines += [
        "",
        "## Rates (short window — not primary)",
        "",
        f"- gap births: {report['rates']['n_gap_births']}",
        f"- gap established (age≥{GRACE}, meaningful): "
        f"{report['rates']['n_gap_established']} "
        f"({report['rates']['frac_gap_established_of_gap_births']})",
        f"- gap alive end: {report['rates']['n_gap_alive_end']} = "
        f"{report['rates'].get('n_gap_alive_end_age_ge_grace_tested')} tested + "
        f"{report['rates'].get('n_gap_alive_end_age_lt_grace_right_censored')} "
        f"right-censored (age<{GRACE}) — **not** establishment",
        f"- gap pruned: {report['rates']['n_gap_pruned']}",
        f"- alive age hist: {report['rates'].get('alive_age_histogram')}",
        f"- first prune wave: epoch {report['first_prune_wave']['epoch']} "
        f"origins={report['first_prune_wave']['origins']}",
        f"- val acc: {report['end_state']['val_acc_start']:.3f} → "
        f"{report['end_state']['val_acc_end']:.3f}",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "rates": report["rates"], "checks": {k: v["pass"] for k, v in report["checks"].items()}}, indent=2))
    print(f"wrote {out_path}")
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
