#!/usr/bin/env python3
"""Fashion grow_only: trigger locus vs post-birth local CE on the gap samples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet, HIDDEN_LAYERS  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import (  # noqa: E402
    PlasticityConfig,
    neighbor_mask_within_radius,
)
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
    / "growth_locus_validation"
)
FASHION_R = {"h1": 0.71, "h2": 0.80, "h3": 0.91}
RADIUS = 0.35
K_MIN = 11
TOP_Q = 0.25  # MNIST mistarget used top-quartile of locus signal
CORR_POS = 0.05  # genuine-demand corr bar (MNIST)
# Outcome: CE change on birth gap samples (absolute CE points)
HELP_EPS = 0.01
WORSEN_EPS = 0.01
FOLLOW_HORIZONS = (1, 2, 3)


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


def grow_only_cfg() -> PlasticityConfig:
    return PlasticityConfig(
        radius=RADIUS,
        target_rates=dict(FASHION_R),
        tau=0.5,
        threshold_kind="fixed",
        mu=0.999,
        prune_edge_fraction=0.0,
        prune_neuron_z=-1.0e9,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=4,
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
        novelty_k_min=K_MIN,
        novelty_rank_fraction=1.0,
    )


def _pearson(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    n = min(x.numel(), y.numel())
    if n < 8:
        return float("nan")
    x, y = x[:n], y[:n]
    x = x - x.mean()
    y = y - y.mean()
    den = float(x.std(unbiased=False).item() * y.std(unbiased=False).item())
    if den < 1e-12:
        return 0.0
    return float((x * y).mean().item() / den)


def _probe_posts_and_ce(
    net: DynamicMLPReasoningNet,
    loader,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Full probe: posts per layer [B,H], CE [B], logits unused."""
    act_key = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}
    bufs: dict[str, list[Tensor]] = {h: [] for h in HIDDEN_LAYERS}
    ce_chunks: list[Tensor] = []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            if y.ndim > 1:
                y = y.view(-1)
            logits = net(x)
            ce = F.cross_entropy(logits, y, reduction="none")
            ce_chunks.append(ce.detach().cpu())
            acts = net.last_activations
            assert acts is not None
            for h in HIDDEN_LAYERS:
                bufs[h].append(acts[act_key[h]].detach().cpu())
    posts = {h: torch.cat(bufs[h], dim=0) for h in HIDDEN_LAYERS}
    ce = torch.cat(ce_chunks, dim=0)
    return posts, ce, ce  # third unused placeholder


def locus_metrics_at_trigger(
    *,
    posts: Tensor,
    ce: Tensor,
    coords: Tensor,
    active: Tensor,
    trigger: int,
    gap_mask: Tensor,
    radius: float = RADIUS,
) -> dict[str, float]:
    """MNIST-style mistarget metrics; locus signal = novelty (−nbr_max)."""
    abs_act = posts.detach().float().abs()  # [B, H]
    ce = ce.detach().float().reshape(-1)
    B = min(abs_act.shape[0], ce.numel())
    abs_act, ce = abs_act[:B], ce[:B]
    gap = gap_mask.bool().reshape(-1)[:B]
    if gap.numel() < B:
        # pad false
        g = torch.zeros(B, dtype=torch.bool)
        g[: gap.numel()] = gap
        gap = g

    # Neighbor max for trigger (same definition as gated novelty)
    nbr = neighbor_mask_within_radius(coords.cpu().float(), active.cpu().bool(), radius)
    n_mask = nbr[int(trigger)]
    nbr_idx = n_mask.nonzero(as_tuple=False).view(-1)
    if nbr_idx.numel() == 0:
        nbr_max = torch.zeros(B)
    else:
        nbr_max = abs_act[:, nbr_idx].max(dim=1).values
    novelty = -nbr_max  # high novelty = low nbr coverage
    self_act = abs_act[:, int(trigger)]

    corr_nov_ce = _pearson(novelty, ce)
    corr_self_ce = _pearson(self_act, ce)

    # Top-quartile of locus signal (novelty) — MNIST top_q=0.25 of act
    k = max(1, int(B * TOP_Q))
    top_idx = torch.argsort(novelty, descending=True)[:k]
    ce_all = float(ce.mean().item())
    ce_top_nov = float(ce[top_idx].mean().item())
    # Also top-quartile self-act (act>r* era metric)
    top_self = torch.argsort(self_act, descending=True)[:k]
    ce_top_self = float(ce[top_self].mean().item())

    ce_gap = float(ce[gap].mean().item()) if bool(gap.any()) else float("nan")
    ce_nongap = float(ce[~gap].mean().item()) if bool((~gap).any()) else float("nan")
    local_ce_excess = ce_gap - ce_all if ce_gap == ce_gap else float("nan")

    err = (ce > 0).float()  # soft; use high-CE as proxy if needed
    # hard error rate needs labels — use CE>median as rough err proxy for corr only
    err_proxy = (ce > ce.median()).float()
    corr_nov_err = _pearson(novelty, err_proxy)

    # Entropy not available without full softmax per sample without re-forward;
    # skip or approximate from CE magnitude.
    return {
        "mean_nbr_max": float(nbr_max.mean().item()),
        "novelty_mean": float(novelty.mean().item()),
        "mean_self_act": float(self_act.mean().item()),
        "corr_novelty_ce": corr_nov_ce,
        "corr_self_act_ce": corr_self_ce,
        "corr_novelty_err_proxy": corr_nov_err,
        "ce_all": ce_all,
        "ce_gap": ce_gap,
        "ce_nongap": ce_nongap,
        "local_ce_excess": local_ce_excess,
        "ce_top_novelty": ce_top_nov,
        "ce_top_novelty_minus_all": ce_top_nov - ce_all,
        "ce_top_self_minus_all": ce_top_self - ce_all,
        "n_gap": int(gap.sum().item()),
        "n_probe": int(B),
        "n_neighbors": int(nbr_idx.numel()),
    }


def classify_demand(m: dict[str, float]) -> str:
    """Genuine demand vs mistarget (gated-novelty locus = novelty)."""
    corr = m["corr_novelty_ce"]
    ce_top_m = m["ce_top_novelty_minus_all"]
    excess = m["local_ce_excess"]
    # MNIST genuine: corr>0.05 & CE_top>CE_all
    # For gated novelty, local_ce_excess>0 is also required at trigger time
    genuine = (
        corr == corr
        and corr > CORR_POS
        and ce_top_m > 0
        and excess == excess
        and excess > 0
    )
    # Mistarget: gate-passed but novelty not error-associated
    not_error = corr != corr or corr <= CORR_POS or ce_top_m <= 0
    if genuine:
        return "genuine_demand"
    if excess == excess and excess > 0 and not_error:
        return "mistarget_gate_not_corr"
    if excess == excess and excess <= 0:
        return "no_ce_excess"
    return "unclear"


@dataclass
class BirthTrack:
    layer: str
    child_slot: int
    trigger: int
    birth_epoch: int
    gap_mask: Tensor  # [B] bool on probe
    at_birth: dict[str, float] = field(default_factory=dict)
    demand_class: str = ""
    # horizon h → ce_gap, ce_all, delta_ce_gap
    followups: dict[int, dict[str, float]] = field(default_factory=dict)


def _agg(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) == r.get(k)]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"median_{k}"] = float(np.median(vals)) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = _device()
    loaders = fashion_loaders(args.seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=args.seed,
    ).to(device)
    cfg = grow_only_cfg()
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    global_step = 0

    live: list[BirthTrack] = []
    acc_hist: list[float] = []

    for epoch in range(args.epochs):
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

        # Probe posts+CE for mistarget + follow-ups (same loader order as gated)
        posts, ce, _ = _probe_posts_and_ce(net, loaders["probe_loader"], device)

        # Update follow-ups for existing births before new ones this epoch
        for tr in live:
            age = epoch - tr.birth_epoch
            if age in FOLLOW_HORIZONS and age not in tr.followups:
                g = tr.gap_mask.bool()
                n = min(ce.numel(), g.numel())
                g, ce_e = g[:n], ce[:n]
                if bool(g.any()):
                    ce_gap = float(ce_e[g].mean().item())
                else:
                    ce_gap = float("nan")
                ce_all = float(ce_e.mean().item())
                birth_gap = tr.at_birth.get("ce_gap", float("nan"))
                tr.followups[age] = {
                    "epoch": float(epoch),
                    "ce_gap": ce_gap,
                    "ce_all": ce_all,
                    "delta_ce_gap": (
                        ce_gap - birth_gap
                        if ce_gap == ce_gap and birth_gap == birth_gap
                        else float("nan")
                    ),
                    "delta_ce_all": ce_all - tr.at_birth.get("ce_all", ce_all),
                    "rel_delta_ce_gap": (
                        (ce_gap - birth_gap) / max(abs(birth_gap), 1e-6)
                        if ce_gap == ce_gap and birth_gap == birth_gap
                        else float("nan")
                    ),
                }

        slog = orch.step(
            net, scores, optimizer=opt, epoch=epoch, signal=signal, gated_novelty=gated
        )

        # New births this step
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
            if trigger is None or (layer, trigger) not in gated.gap_masks:
                continue
            gmask = gated.gap_masks[(layer, trigger)].bool().cpu().clone()
            coords = net.coords(layer).detach().cpu()
            active = net.active_mask(layer).detach().cpu()
            # Use posts before grow for trigger metrics (trigger is pre-existing)
            # posts were taken before orch.step — correct for trigger at decision time
            m = locus_metrics_at_trigger(
                posts=posts[layer],
                ce=ce,
                coords=coords,
                active=active,
                trigger=trigger,
                gap_mask=gmask,
            )
            # EMA percentile of trigger among active (optional context)
            s_layer = scores.get(layer, scores.get({"h1": "fc1", "h2": "fc2", "h3": "fc3"}[layer]))
            if s_layer is not None:
                s = s_layer.detach().float().cpu()
                a = active.bool()
                n = min(s.numel(), a.numel())
                s, a = s[:n], a[:n]
                if bool(a.any()) and trigger < n and bool(a[trigger]):
                    active_s = s[a]
                    rank = float((active_s < s[trigger]).float().mean().item())
                    m["ema_percentile"] = rank
                    m["ema_act"] = float(s[trigger].item())
                    rstar = FASHION_R[layer]
                    m["ema_over_rstar"] = m["ema_act"] / rstar if rstar else float("nan")
            dclass = classify_demand(m)
            live.append(
                BirthTrack(
                    layer=layer,
                    child_slot=int(slot),
                    trigger=int(trigger),
                    birth_epoch=epoch,
                    gap_mask=gmask,
                    at_birth=m,
                    demand_class=dclass,
                )
            )

        vm = evaluate(net, loaders["test_loader"], device=device)
        acc_hist.append(float(vm["acc"]))
        print(
            f"[grow_only s{args.seed}] ep{epoch:02d} acc={vm['acc']:.4f} "
            f"grown={slog.neurons_grown} tracked={len(live)}",
            flush=True,
        )

    # End-of-run: fill any remaining follow-ups at last epoch for horizons not hit
    posts_f, ce_f, _ = _probe_posts_and_ce(net, loaders["probe_loader"], device)
    for tr in live:
        age = (args.epochs - 1) - tr.birth_epoch
        for h in FOLLOW_HORIZONS:
            if h <= age and h not in tr.followups:
                g = tr.gap_mask.bool()
                n = min(ce_f.numel(), g.numel())
                g, ce_e = g[:n], ce_f[:n]
                ce_gap = float(ce_e[g].mean().item()) if bool(g.any()) else float("nan")
                ce_all = float(ce_e.mean().item())
                birth_gap = tr.at_birth.get("ce_gap", float("nan"))
                tr.followups[h] = {
                    "epoch": float(tr.birth_epoch + h),
                    "ce_gap": ce_gap,
                    "ce_all": ce_all,
                    "delta_ce_gap": (
                        ce_gap - birth_gap
                        if ce_gap == ce_gap and birth_gap == birth_gap
                        else float("nan")
                    ),
                    "delta_ce_all": ce_all - tr.at_birth.get("ce_all", ce_all),
                    "rel_delta_ce_gap": (
                        (ce_gap - birth_gap) / max(abs(birth_gap), 1e-6)
                        if ce_gap == ce_gap and birth_gap == birth_gap
                        else float("nan")
                    ),
                    "note": "backfilled_at_run_end",
                }

    # --- Aggregate mistarget (MNIST methodology) ---
    birth_rows: list[dict[str, Any]] = []
    for tr in live:
        row = {
            "layer": tr.layer,
            "child": tr.child_slot,
            "trigger": tr.trigger,
            "birth_epoch": tr.birth_epoch,
            "demand_class": tr.demand_class,
            **tr.at_birth,
            "followups": {str(k): v for k, v in tr.followups.items()},
        }
        # Outcome at each horizon
        for h in FOLLOW_HORIZONS:
            fu = tr.followups.get(h)
            if not fu:
                row[f"outcome_h{h}"] = "no_data"
                row[f"delta_ce_gap_h{h}"] = float("nan")
                continue
            d = fu["delta_ce_gap"]
            row[f"delta_ce_gap_h{h}"] = d
            if d != d:
                row[f"outcome_h{h}"] = "nan"
            elif d < -HELP_EPS:
                row[f"outcome_h{h}"] = "helped"
            elif d > WORSEN_EPS:
                row[f"outcome_h{h}"] = "worsened"
            else:
                row[f"outcome_h{h}"] = "flat"
        birth_rows.append(row)

    n = len(birth_rows)
    corr_nov = [r["corr_novelty_ce"] for r in birth_rows if r["corr_novelty_ce"] == r["corr_novelty_ce"]]
    corr_self = [r["corr_self_act_ce"] for r in birth_rows if r["corr_self_act_ce"] == r["corr_self_act_ce"]]
    ce_top_m = [r["ce_top_novelty_minus_all"] for r in birth_rows]
    excess = [r["local_ce_excess"] for r in birth_rows]

    demand_counts = {}
    for r in birth_rows:
        demand_counts[r["demand_class"]] = demand_counts.get(r["demand_class"], 0) + 1

    mistarget = {
        "protocol": {
            "dataset": "fashion_mnist",
            "arm": "grow_only",
            "trigger": "gated_novelty",
            "locus_signal": "novelty = -nbr_max (not self-act; product trigger)",
            "also_reported": "corr(self_act, CE) for MNIST act>r* comparison",
            "top_q": TOP_Q,
            "corr_bar": CORR_POS,
            "epochs": args.epochs,
            "seed": args.seed,
            "target_rates": FASHION_R,
            "mnist_reference": {
                "verdict": "MISTARGET_HOT_BUT_FINE",
                "mean_corr_act_ce": -0.009,
                "frac_corr_gt_0.05": 0.0,
                "frac_genuine_demand": 0.0,
                "frac_mistarget": 0.72,
            },
        },
        "n_births": n,
        "mean_corr_novelty_ce": float(np.mean(corr_nov)) if corr_nov else float("nan"),
        "mean_corr_self_act_ce": float(np.mean(corr_self)) if corr_self else float("nan"),
        "frac_corr_novelty_ce_gt_0.05": (
            float(np.mean([c > CORR_POS for c in corr_nov])) if corr_nov else float("nan")
        ),
        "frac_corr_self_act_ce_gt_0.05": (
            float(np.mean([c > CORR_POS for c in corr_self])) if corr_self else float("nan")
        ),
        "mean_ce_top_novelty_minus_all": float(np.mean(ce_top_m)) if ce_top_m else float("nan"),
        "mean_local_ce_excess": float(np.mean(excess)) if excess else float("nan"),
        "frac_ce_top_novelty_gt_all": (
            float(np.mean([x > 0 for x in ce_top_m])) if ce_top_m else float("nan")
        ),
        "frac_local_ce_excess_gt_0": (
            float(np.mean([x > 0 for x in excess])) if excess else float("nan")
        ),
        "demand_class_counts": demand_counts,
        "frac_genuine_demand": demand_counts.get("genuine_demand", 0) / max(n, 1),
        "frac_mistarget_gate_not_corr": demand_counts.get("mistarget_gate_not_corr", 0)
        / max(n, 1),
    }

    # Mistarget verdict (adapted for gated novelty)
    if mistarget["frac_genuine_demand"] >= 0.5 and mistarget["frac_corr_novelty_ce_gt_0.05"] >= 0.5:
        mistarget["verdict"] = "TRIGGER_TRACKS_ERROR"
        mistarget["interpretation"] = (
            "Gated-novelty locus co-locates with higher CE (corr and top-quartile). "
            "Unlike MNIST act>r* mistarget, Fashion growth sites look error-adjacent at fire time."
        )
    elif mistarget["frac_genuine_demand"] < 0.2 and mistarget["mean_corr_novelty_ce"] <= 0.02:
        mistarget["verdict"] = "MISTARGET_OR_WEAK"
        mistarget["interpretation"] = (
            "Growth events do not show strong novelty–CE association under the same "
            "bars as MNIST genuine-demand; gate may pass on weak excess without stable corr."
        )
    else:
        mistarget["verdict"] = "MIXED_TRIGGER_SIGNAL"
        mistarget["interpretation"] = (
            "Partial error association; not MNIST-style pure mistarget, not clean track."
        )

    # --- Outcome aggregation ---
    outcome: dict[str, Any] = {
        "protocol": {
            "locus": "birth gap_mask samples (same probe indices)",
            "metric": "mean CE on gap_mask",
            "help_if": f"delta_ce_gap < -{HELP_EPS}",
            "worsen_if": f"delta_ce_gap > +{WORSEN_EPS}",
            "horizons": list(FOLLOW_HORIZONS),
            "note": (
                "Negative delta = local CE fell after birth (growth helped locus). "
                "Does not require network-wide acc improvement."
            ),
        },
        "by_horizon": {},
    }
    for h in FOLLOW_HORIZONS:
        deltas = [
            r[f"delta_ce_gap_h{h}"]
            for r in birth_rows
            if r.get(f"delta_ce_gap_h{h}") == r.get(f"delta_ce_gap_h{h}")
        ]
        labels = [r[f"outcome_h{h}"] for r in birth_rows if r.get(f"outcome_h{h}") not in ("no_data",)]
        n_h = len(deltas)
        if n_h == 0:
            outcome["by_horizon"][str(h)] = {"n": 0}
            continue
        n_help = sum(1 for x in labels if x == "helped")
        n_flat = sum(1 for x in labels if x == "flat")
        n_wors = sum(1 for x in labels if x == "worsened")
        outcome["by_horizon"][str(h)] = {
            "n": n_h,
            "mean_delta_ce_gap": float(np.mean(deltas)),
            "median_delta_ce_gap": float(np.median(deltas)),
            "frac_helped": n_help / n_h,
            "frac_flat": n_flat / n_h,
            "frac_worsened": n_wors / n_h,
            "p25_delta": float(np.percentile(deltas, 25)),
            "p75_delta": float(np.percentile(deltas, 75)),
        }

    # Primary horizon +2 (if available) else +1
    primary_h = 2 if "2" in outcome["by_horizon"] and outcome["by_horizon"]["2"].get("n", 0) > 0 else 1
    oh = outcome["by_horizon"].get(str(primary_h), {})
    if oh.get("n", 0) == 0:
        outcome["verdict"] = "NO_OUTCOME_DATA"
        outcome["interpretation"] = "No follow-up CE measurements."
    elif oh["frac_helped"] >= 0.5 and oh["mean_delta_ce_gap"] < -HELP_EPS:
        outcome["verdict"] = "GROWTH_HELPS_LOCUS"
        outcome["interpretation"] = (
            f"At horizon +{primary_h}, majority of births reduce mean CE on their "
            "gap samples; growth is not pure thrash at the locus level."
        )
    elif oh["frac_worsened"] >= 0.5 and oh["mean_delta_ce_gap"] > WORSEN_EPS:
        outcome["verdict"] = "GROWTH_HURTS_OR_NEUTRAL_LOCUS"
        outcome["interpretation"] = (
            f"At horizon +{primary_h}, local CE on birth gap samples rises for most "
            "events — growth is not repairing the locus error it fired on; consistent "
            "with thrash from unhelpful insertion."
        )
    elif oh["frac_flat"] + oh.get("frac_worsened", 0) >= 0.6:
        outcome["verdict"] = "GROWTH_NO_LOCAL_HELP"
        outcome["interpretation"] = (
            f"At horizon +{primary_h}, local CE is mostly flat or worse — trigger may "
            "fire on error-adjacent samples but births do not reduce that local error."
        )
    else:
        outcome["verdict"] = "MIXED_LOCAL_OUTCOME"
        outcome["interpretation"] = (
            f"Horizon +{primary_h}: mixed help/hurt; see frac_helped="
            f"{oh.get('frac_helped')} mean_delta={oh.get('mean_delta_ce_gap')}."
        )
    outcome["primary_horizon"] = primary_h

    # Cross-tab: genuine demand vs helped
    cross = []
    for r in birth_rows:
        cross.append(
            {
                "demand_class": r["demand_class"],
                "outcome_h2": r.get("outcome_h2", r.get("outcome_h1")),
                "delta_h2": r.get("delta_ce_gap_h2", r.get("delta_ce_gap_h1")),
            }
        )
    cross_summary: dict[str, Any] = {}
    for dclass in sorted(set(c["demand_class"] for c in cross)):
        sub = [c for c in cross if c["demand_class"] == dclass]
        outs = [c["outcome_h2"] for c in sub if c["outcome_h2"] not in ("no_data", None)]
        if not outs:
            continue
        cross_summary[dclass] = {
            "n": len(outs),
            "frac_helped": sum(1 for o in outs if o == "helped") / len(outs),
            "frac_worsened": sum(1 for o in outs if o == "worsened") / len(outs),
            "frac_flat": sum(1 for o in outs if o == "flat") / len(outs),
        }

    # Combined product reading
    if (
        mistarget["verdict"] == "TRIGGER_TRACKS_ERROR"
        and outcome["verdict"] == "GROWTH_HELPS_LOCUS"
    ):
        combined = "TRIGGER_OK_AND_HELPS"
        combined_read = (
            "Trigger is error-adjacent and births reduce local CE — thrash may be "
            "collateral (rate too high / global disruption) not mistarget. Next: "
            "lower birth rate/cap, not redesign need signal."
        )
    elif (
        mistarget["verdict"] == "TRIGGER_TRACKS_ERROR"
        and outcome["verdict"] in ("GROWTH_NO_LOCAL_HELP", "GROWTH_HURTS_OR_NEUTRAL_LOCUS")
    ):
        combined = "TRIGGER_OK_BUT_BIRTH_FAILS"
        combined_read = (
            "Trigger finds real local error but gap_imprinted insertion does not "
            "fix it (or worsens). Next: birth realization / rate, not novelty gate."
        )
    elif mistarget["verdict"] in ("MISTARGET_OR_WEAK",) and outcome["verdict"] in (
        "GROWTH_NO_LOCAL_HELP",
        "GROWTH_HURTS_OR_NEUTRAL_LOCUS",
        "MIXED_LOCAL_OUTCOME",
    ):
        combined = "MISTARGET_AND_NO_HELP"
        combined_read = (
            "Weak error association and no local CE repair — same root-cause class "
            "as MNIST mistarget: growth is not demand-driven help."
        )
    else:
        combined = f"{mistarget['verdict']}__{outcome['verdict']}"
        combined_read = (
            f"Mistarget={mistarget['verdict']}; outcome={outcome['verdict']}. "
            "See tables for detail."
        )

    summary = {
        "dataset": "fashion_mnist",
        "arm": "grow_only",
        "seed": args.seed,
        "epochs": args.epochs,
        "n_births": n,
        "acc_trajectory": acc_hist,
        "mistarget": mistarget,
        "outcome": outcome,
        "cross_demand_x_outcome": cross_summary,
        "combined_verdict": combined,
        "combined_reading": combined_read,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "birth_rows.json").write_text(
        json.dumps({"n": n, "rows": birth_rows}, indent=2)
    )

    # Markdown
    lines = [
        "# Fashion grow_only: trigger mistarget + local-CE outcome",
        "",
        f"**Combined verdict:** `{combined}`",
        "",
        combined_read,
        "",
        f"n_births={n}, seed={args.seed}, epochs={args.epochs}, arm=grow_only, Fashion r*",
        "",
        "## 1. Trigger mistarget (MNIST methodology, gated-novelty locus)",
        "",
        "Locus signal = **novelty (−nbr_max)** — the product ranking signal — not self-act.",
        "Self-act corr reported for comparison to MNIST act>r* era.",
        "",
        "| Metric | Fashion grow_only | MNIST mistarget (est. parents) |",
        "|--------|------------------:|-------------------------------:|",
        f"| mean corr(novelty, CE) | **{mistarget['mean_corr_novelty_ce']:.3f}** | n/a (was act) |",
        f"| mean corr(self_act, CE) | {mistarget['mean_corr_self_act_ce']:.3f} | **−0.009** |",
        f"| frac corr(novelty,CE)>0.05 | **{mistarget['frac_corr_novelty_ce_gt_0.05']:.0%}** | 0% (act) |",
        f"| mean CE_top_nov − CE_all | **{mistarget['mean_ce_top_novelty_minus_all']:.3f}** | −0.019 (act top) |",
        f"| mean local CE excess (gap) | **{mistarget['mean_local_ce_excess']:.3f}** | — |",
        f"| frac genuine demand | **{mistarget['frac_genuine_demand']:.0%}** | **0%** |",
        f"| frac mistarget (gate, not corr) | **{mistarget['frac_mistarget_gate_not_corr']:.0%}** | 72% |",
        "",
        f"**Mistarget verdict:** `{mistarget['verdict']}`",
        "",
        mistarget["interpretation"],
        "",
        f"Demand class counts: `{demand_counts}`",
        "",
        "## 2. Post-birth local CE outcome (new — Fashion headroom)",
        "",
        "For each birth: mean CE on **fixed birth gap_mask** samples at +1/+2/+3 epochs.",
        f"Helped if ΔCE_gap < −{HELP_EPS}; worsened if ΔCE_gap > +{WORSEN_EPS}.",
        "",
        "| Horizon | n | mean ΔCE_gap | median | frac helped | frac flat | frac worsened |",
        "|--------:|--:|-------------:|-------:|------------:|----------:|--------------:|",
    ]
    for h in FOLLOW_HORIZONS:
        oh = outcome["by_horizon"].get(str(h), {})
        if oh.get("n", 0) == 0:
            lines.append(f"| +{h} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| +{h} | {oh['n']} | **{oh['mean_delta_ce_gap']:+.3f}** | "
            f"{oh['median_delta_ce_gap']:+.3f} | {oh['frac_helped']:.0%} | "
            f"{oh['frac_flat']:.0%} | {oh['frac_worsened']:.0%} |"
        )
    lines += [
        "",
        f"**Outcome verdict (primary h=+{primary_h}):** `{outcome['verdict']}`",
        "",
        outcome["interpretation"],
        "",
        "## 3. Cross-tab demand × outcome (primary horizon)",
        "",
        "| Demand class | n | frac helped | frac flat | frac worsened |",
        "|--------------|--:|------------:|----------:|--------------:|",
    ]
    for k, v in cross_summary.items():
        lines.append(
            f"| {k} | {v['n']} | {v['frac_helped']:.0%} | {v['frac_flat']:.0%} | "
            f"{v['frac_worsened']:.0%} |"
        )
    lines += [
        "",
        "## Acc trajectory (grow_only, this run)",
        "",
        f"`{' → '.join(f'{a:.3f}' for a in acc_hist)}`",
        "",
        "Artifacts: `summary.json`, `birth_rows.json`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2)[:4000], flush=True)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
