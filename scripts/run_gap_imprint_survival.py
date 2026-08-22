#!/usr/bin/env python3
"""Gap-imprint specialist survival (gated novelty + gap_imprinted; fixed vs sliding)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet  # noqa: E402
from neuronca.plasticity.masks import NEURON_ORIGIN_GROWN  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

# Pre-registered bars
GATE_GOW = 1.3
GATE_ONOFF = 0.2
STAY_GOW_FINAL = 1.2
STAY_GOW_DECAY = 0.4
STAY_ONOFF_FINAL = 0.15
STAY_ONOFF_DECAY = 0.15
GRACE = 3
EPOCHS = 6
LIN_IN = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}
ACT_KEY = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(threshold_kind: str) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates={"h1": 0.77, "h2": 1.02, "h3": 1.28},
        tau=0.5,
        threshold_kind=str(threshold_kind),
        mu=0.999,
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=4,
        seed_edges_per_new_neuron=4,
        newborn_grace_steps=GRACE,
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


@torch.no_grad()
def _probe_posts(
    net: DynamicMLPReasoningNet,
    loader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Full probe post-ReLU acts per layer [B, H]."""
    bufs: dict[str, list[torch.Tensor]] = {h: [] for h in HIDDEN_LAYERS}
    net.eval()
    for batch in loader:
        x = batch[0].to(device)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        net(x)
        acts = net.last_activations
        assert acts is not None
        for h in HIDDEN_LAYERS:
            bufs[h].append(acts[ACT_KEY[h]].detach().cpu())
    return {h: torch.cat(bufs[h], dim=0) for h in HIDDEN_LAYERS}


def _profile(post: torch.Tensor, gap_mask: torch.Tensor) -> dict[str, float]:
    """post [B] abs acts; gap_mask [B] bool."""
    a = post.detach().float().abs().reshape(-1)
    m = gap_mask.bool().reshape(-1)
    n = min(a.numel(), m.numel())
    a, m = a[:n], m[:n]
    gap, ng = m, ~m
    whole = float(a.mean().item()) + 1e-8
    pg = float(a[gap].mean().item()) if gap.any() else float("nan")
    pn = float(a[ng].mean().item()) if ng.any() else float("nan")
    fg = float((a[gap] > 0).float().mean().item()) if gap.any() else float("nan")
    fn = float((a[ng] > 0).float().mean().item()) if ng.any() else float("nan")
    return {
        "mean_post_whole": float(a.mean().item()),
        "mean_post_gap": pg,
        "mean_post_nongap": pn,
        "gap_over_whole": pg / whole if pg == pg else float("nan"),
        "frac_gap_on": fg,
        "frac_nongap_on": fn,
        "on_off": (fg - fn) if fg == fg and fn == fn else float("nan"),
    }


def _passes_birth_gate(p: dict[str, float]) -> bool:
    return (
        p.get("gap_over_whole", 0) >= GATE_GOW
        and p.get("on_off", -1) >= GATE_ONOFF
    )


def _passes_stays_specialized(birth: dict[str, float], final: dict[str, float]) -> bool:
    g0 = birth.get("gap_over_whole", float("nan"))
    o0 = birth.get("on_off", float("nan"))
    gf = final.get("gap_over_whole", float("nan"))
    of = final.get("on_off", float("nan"))
    if any(x != x for x in (g0, o0, gf, of)):
        return False
    if gf < STAY_GOW_FINAL or (g0 - gf) >= STAY_GOW_DECAY:
        return False
    if of < STAY_ONOFF_FINAL or (o0 - of) >= STAY_ONOFF_DECAY:
        return False
    return True


@dataclass
class UnitTrack:
    layer: str
    slot: int
    trigger: int | None
    birth_epoch: int
    gap_mask: torch.Tensor
    profiles: list[dict[str, Any]] = field(default_factory=list)
    pruned_epoch: int | None = None

    def birth_profile(self) -> dict[str, float] | None:
        for p in self.profiles:
            if p.get("phase") == "birth":
                return p
        return self.profiles[0] if self.profiles else None

    def final_profile(self) -> dict[str, float] | None:
        return self.profiles[-1] if self.profiles else None


def run_one(
    *,
    seed: int,
    threshold_kind: str,
    epochs: int = EPOCHS,
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
    cfg = _cfg(threshold_kind)
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, name="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)

    live: dict[tuple[str, int], UnitTrack] = {}
    finished: list[UnitTrack] = []
    structure_trace: list[dict[str, Any]] = []
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

        # Pre-structure profiles for live units
        posts = _probe_posts(net, loaders["probe_loader"], device)
        for key, tr in list(live.items()):
            layer, slot = key
            if not bool(net.active_mask(layer)[slot]):
                continue
            age = epoch - tr.birth_epoch
            prof = _profile(posts[layer][:, slot], tr.gap_mask)
            tr.profiles.append(
                {
                    "epoch": epoch,
                    "phase": "pre_structure",
                    "age": age,
                    **prof,
                }
            )

        slog = orch.step(
            net,
            scores,
            optimizer=opt,
            epoch=epoch,
            signal=signal,
            gated_novelty=gated,
        )
        structure_trace.append(
            {
                "epoch": epoch,
                "neurons_grown": slog.neurons_grown,
                "neurons_pruned": slog.neurons_pruned,
                "novelty_under_floor_total": slog.novelty_under_floor_total,
                "novelty_under_floor_by_layer": dict(slog.novelty_under_floor_by_layer),
            }
        )

        # Finalize pruned
        for key, tr in list(live.items()):
            layer, slot = key
            if not bool(net.active_mask(layer)[slot]):
                tr.pruned_epoch = epoch
                finished.append(tr)
                del live[key]

        # New births
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
                # Cannot track without gap mask
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
                {
                    "epoch": epoch,
                    "phase": "birth",
                    "age": 0,
                    **prof,
                }
            )
            live[(layer, int(slot))] = tr

    # End survivors
    for tr in live.values():
        tr.pruned_epoch = None
        finished.append(tr)

    # Summarize units
    step_now = int(net.cooldown.plasticity_step)
    unit_rows: list[dict[str, Any]] = []
    for tr in finished:
        origin_ok = True
        if tr.pruned_epoch is None:
            # still alive — check origin
            if not bool(net.active_mask(tr.layer)[tr.slot]):
                origin_ok = False
            else:
                origin_ok = (
                    int(net.neuron_origin(tr.layer)[tr.slot].item())
                    == NEURON_ORIGIN_GROWN
                )
                birth_step = int(net.birth_step(tr.layer)[tr.slot].item())
                age_end = step_now - birth_step
        else:
            age_end = tr.pruned_epoch - tr.birth_epoch  # type: ignore[operator]

        bp = tr.birth_profile() or {}
        # last profile while alive
        fp = tr.final_profile() or {}
        max_age = max((int(p.get("age", 0)) for p in tr.profiles), default=0)
        # Established if they ever reached grace age while tracked
        established = max_age >= GRACE

        birth_gate = _passes_birth_gate(bp) if bp else False
        stays = False
        if established and birth_gate and bp and fp:
            # final profile among those with age >= grace
            aged = [p for p in tr.profiles if int(p.get("age", 0)) >= GRACE]
            final_est = aged[-1] if aged else fp
            stays = _passes_stays_specialized(bp, final_est)
            fp = final_est

        unit_rows.append(
            {
                "layer": tr.layer,
                "slot": tr.slot,
                "trigger": tr.trigger,
                "birth_epoch": tr.birth_epoch,
                "pruned_epoch": tr.pruned_epoch,
                "max_age": max_age,
                "established": established,
                "birth_gate": birth_gate,
                "stays_specialized": stays,
                "birth": {
                    k: bp.get(k)
                    for k in (
                        "gap_over_whole",
                        "on_off",
                        "frac_gap_on",
                        "frac_nongap_on",
                        "mean_post_whole",
                    )
                },
                "final": {
                    k: fp.get(k)
                    for k in (
                        "gap_over_whole",
                        "on_off",
                        "frac_gap_on",
                        "frac_nongap_on",
                        "mean_post_whole",
                        "age",
                    )
                },
                "trajectory": [
                    {
                        "epoch": p.get("epoch"),
                        "age": p.get("age"),
                        "phase": p.get("phase"),
                        "gap_over_whole": p.get("gap_over_whole"),
                        "on_off": p.get("on_off"),
                        "frac_gap_on": p.get("frac_gap_on"),
                        "frac_nongap_on": p.get("frac_nongap_on"),
                    }
                    for p in tr.profiles
                ],
            }
        )

    n_births = len(unit_rows)
    n_gate = sum(1 for u in unit_rows if u["birth_gate"])
    n_est = sum(1 for u in unit_rows if u["established"])
    n_est_gate = sum(1 for u in unit_rows if u["established"] and u["birth_gate"])
    n_stays = sum(
        1 for u in unit_rows if u["established"] and u["birth_gate"] and u["stays_specialized"]
    )
    # Also stays among all established (even if failed birth gate — secondary)
    n_stays_all_est = sum(
        1
        for u in unit_rows
        if u["established"]
        and u["birth_gate"]
        and u["stays_specialized"]
    )

    # Mean trajectories by age for gated births
    by_age: dict[int, list[dict[str, float]]] = {}
    for u in unit_rows:
        if not u["birth_gate"]:
            continue
        for p in u["trajectory"]:
            age = int(p.get("age") or 0)
            by_age.setdefault(age, []).append(p)

    def mean_age(key: str) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for age, plist in sorted(by_age.items()):
            vals = [
                float(p[key])
                for p in plist
                if p.get(key) is not None and p[key] == p[key]
            ]
            out[str(age)] = sum(vals) / len(vals) if vals else None
        return out

    # Mean trajectories for established+gated only
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

    # Optional convenience label
    if n_est_gate == 0:
        label = "ESTABLISH_FAIL"
    elif frac_stays_of_est_gate >= 0.5:
        label = "SPECIALIST_SURVIVES"
    elif n_est_gate > 0:
        label = "ESTABLISH_ONLY"
    else:
        label = "ESTABLISH_FAIL"

    return {
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "threshold_kind": threshold_kind,
            "stack": "gated_novelty+gap_imprinted(sep_power,open_fan,straddle_B)",
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
        },
        "rates": {
            "frac_births_passing_birth_gate": frac_gate,
            "frac_established_of_births": frac_est,
            "frac_established_of_gated": (
                n_est_gate / n_gate if n_gate else 0.0
            ),
            "frac_of_established_gated_passing_stays_specialized": frac_stays_of_est_gate,
            "frac_stays_of_all_births": frac_stays_of_births,
        },
        "optional_label": label,
        "mean_trajectory_gated_births": {
            "gap_over_whole_by_age": mean_age("gap_over_whole"),
            "on_off_by_age": mean_age("on_off"),
            "frac_gap_on_by_age": mean_age("frac_gap_on"),
            "frac_nongap_on_by_age": mean_age("frac_nongap_on"),
        },
        "mean_trajectory_established_gated": {
            "gap_over_whole_by_age": mean_age_est("gap_over_whole"),
            "on_off_by_age": mean_age_est("on_off"),
            "frac_gap_on_by_age": mean_age_est("frac_gap_on"),
            "frac_nongap_on_by_age": mean_age_est("frac_nongap_on"),
        },
        "structure_trace": structure_trace,
        "hidden_counts_final": net.hidden_counts(),
        "units": unit_rows,
    }


def _mean_rates(seed_results: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = [
        "frac_births_passing_birth_gate",
        "frac_established_of_births",
        "frac_established_of_gated",
        "frac_of_established_gated_passing_stays_specialized",
        "frac_stays_of_all_births",
    ]
    out: dict[str, float | None] = {}
    for k in keys:
        vals = [float(r["rates"][k]) for r in seed_results if k in r["rates"]]
        out[k] = sum(vals) / len(vals) if vals else None
    # mean counts
    for ck in (
        "n_births",
        "n_birth_gate",
        "n_established",
        "n_established_and_gated",
        "n_stays_specialized",
    ):
        vals = [float(r["counts"][ck]) for r in seed_results]
        out[f"mean_{ck}"] = sum(vals) / len(vals) if vals else None
    return out


def aggregate_arm(threshold_kind: str, seeds: list[int]) -> dict[str, Any]:
    out_dir = (
        _REPO
        / "results"
        / "phase1a_diagnostics"
        / f"gap_imprint_survival_{threshold_kind}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in seeds:
        print(f"\n=== {threshold_kind} seed {seed} ===", flush=True)
        r = run_one(seed=seed, threshold_kind=threshold_kind, epochs=EPOCHS)
        seed_results.append(r)
        (out_dir / f"seed_{seed}.json").write_text(
            json.dumps(r, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  births={r['counts']['n_births']} gate={r['rates']['frac_births_passing_birth_gate']:.2f} "
            f"est={r['rates']['frac_established_of_births']:.2f} "
            f"stays|est_gate={r['rates']['frac_of_established_gated_passing_stays_specialized']:.2f} "
            f"label={r['optional_label']}",
            flush=True,
        )

    summary = {
        "protocol": {
            "threshold_kind": threshold_kind,
            "seeds": seeds,
            "epochs": EPOCHS,
            "stack": "gated_novelty+gap_imprinted(sep_power,open_fan,straddle_B)",
            "primary_phase1a": True,
            "note": (
                "Primary Phase 1a specialization comparison; supersedes early "
                "τ-sweep fixed-vs-sliding as the headline A/B result."
            ),
        },
        "mean_rates": _mean_rates(seed_results),
        "per_seed": [
            {
                "seed": r["protocol"]["seed"],
                "counts": r["counts"],
                "rates": r["rates"],
                "optional_label": r["optional_label"],
                "mean_trajectory_established_gated": r[
                    "mean_trajectory_established_gated"
                ],
            }
            for r in seed_results
        ],
        "seed_results": seed_results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def compare(fixed: dict[str, Any], sliding: dict[str, Any]) -> dict[str, Any]:
    fr = fixed["mean_rates"]
    sr = sliding["mean_rates"]
    keys = [
        "frac_births_passing_birth_gate",
        "frac_established_of_births",
        "frac_established_of_gated",
        "frac_of_established_gated_passing_stays_specialized",
        "frac_stays_of_all_births",
    ]
    deltas = {}
    for k in keys:
        a, b = fr.get(k), sr.get(k)
        if a is not None and b is not None:
            deltas[k] = {
                "fixed": a,
                "sliding": b,
                "sliding_minus_fixed_pp": (b - a) * 100.0,
            }

    stays_f = fr.get("frac_of_established_gated_passing_stays_specialized") or 0.0
    stays_s = sr.get("frac_of_established_gated_passing_stays_specialized") or 0.0
    est_f = fr.get("frac_established_of_births") or 0.0
    est_s = sr.get("frac_established_of_births") or 0.0

    # Primary reading from continuous rates
    if est_f < 0.05 and est_s < 0.05:
        reading = "ESTABLISH_FAIL_BOTH"
    elif stays_s - stays_f >= 0.15 or (stays_f < 0.1 and stays_s >= 0.25):
        reading = "B_HELPS_SPECIALIST"
    elif stays_f < 0.2 and stays_s < 0.2 and (est_f >= 0.05 or est_s >= 0.05):
        reading = "ESTABLISH_ONLY_BOTH"
    elif abs(stays_s - stays_f) < 0.1:
        reading = "B_NO_HELP_SIMILAR"
    else:
        reading = "MIXED_SEE_RATES"

    return {
        "primary_phase1a_comparison": True,
        "supersedes": "early_tau_sweep_fixed_vs_sliding",
        "deltas_pp": deltas,
        "reading": reading,
        "headline_rates": {
            "fixed": fr,
            "sliding": sr,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", choices=["fixed", "sliding"], default=None)
    p.add_argument("--both", action="store_true")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    epochs = int(args.epochs)

    def _agg(kind: str, seeds_: list[int]) -> dict[str, Any]:
        out_dir = (
            _REPO
            / "results"
            / "phase1a_diagnostics"
            / f"gap_imprint_survival_{kind}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_results = []
        for seed in seeds_:
            print(f"\n=== {kind} seed {seed} ===", flush=True)
            r = run_one(seed=seed, threshold_kind=kind, epochs=epochs)
            seed_results.append(r)
            (out_dir / f"seed_{seed}.json").write_text(
                json.dumps(r, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"  births={r['counts']['n_births']} "
                f"gate={r['rates']['frac_births_passing_birth_gate']:.2f} "
                f"est={r['rates']['frac_established_of_births']:.2f} "
                f"stays|est_gate={r['rates']['frac_of_established_gated_passing_stays_specialized']:.2f} "
                f"label={r['optional_label']}",
                flush=True,
            )
        summary = {
            "protocol": {
                "threshold_kind": kind,
                "seeds": seeds_,
                "epochs": epochs,
                "stack": "gated_novelty+gap_imprinted(sep_power,open_fan,straddle_B)",
                "primary_phase1a": True,
            },
            "mean_rates": _mean_rates(seed_results),
            "per_seed": [
                {
                    "seed": r["protocol"]["seed"],
                    "counts": r["counts"],
                    "rates": r["rates"],
                    "optional_label": r["optional_label"],
                    "mean_trajectory_established_gated": r[
                        "mean_trajectory_established_gated"
                    ],
                }
                for r in seed_results
            ],
            "seed_results": seed_results,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        print(f"Wrote {out_dir / 'summary.json'}", flush=True)
        return summary

    if args.both or args.threshold is None:
        fixed = _agg("fixed", seeds)
        sliding = _agg("sliding", seeds)
        comp = compare(fixed, sliding)
        out = (
            _REPO
            / "results"
            / "phase1a_diagnostics"
            / "gap_imprint_survival_comparison.json"
        )
        payload = {
            "comparison": comp,
            "fixed_mean": fixed["mean_rates"],
            "sliding_mean": sliding["mean_rates"],
            "fixed_per_seed": fixed["per_seed"],
            "sliding_per_seed": sliding["per_seed"],
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("\n=== COMPARISON ===")
        print(json.dumps(payload, indent=2, default=str)[:4000])
        print("Wrote", out)
    else:
        _agg(args.threshold, seeds)


if __name__ == "__main__":
    main()
