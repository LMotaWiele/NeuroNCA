#!/usr/bin/env python3
"""Fashion-MNIST r* probe, straddle check, and growth-fire gate (do not reuse MNIST rates)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet, HIDDEN_LAYERS  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    _dummy_broadcast,
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "fashion_mnist_calibration"
LIN_FOR = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}
# MNIST locked reference (do not reuse for Fashion product claims).
MNIST_RSTAR = {"h1": 0.77, "h2": 1.02, "h3": 1.28}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_epoch_signals_only(
    net: DynamicMLPReasoningNet,
    loader,
    opt,
    signal: ActivationMagnitude,
    device: torch.device,
    global_step: int = 0,
) -> dict[str, float]:
    """One epoch: CE + SGD + ActivationMagnitude EMA, no plasticity step."""
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    net.train()
    total_loss = 0.0
    total_n = 0
    steps = 0
    step = global_step
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.ndim > 1:
            y = y.view(-1)
        opt.zero_grad(set_to_none=True)
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        net.zero_inactive_grads()
        opt.step()
        acts = net.last_activations
        assert acts is not None
        signal.update(
            activations=acts,
            layer_shapes=net.layer_shapes(),
            broadcast=_dummy_broadcast(int(y.size(0)), int(net.num_classes), step),
            step=step,
        )
        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs
        steps += 1
        step += 1
    return {
        "loss": total_loss / max(total_n, 1),
        "n": float(total_n),
        "steps": float(steps),
        "global_step": float(step),
    }


def _active_ema_stats(
    signal: ActivationMagnitude, net: DynamicMLPReasoningNet
) -> dict[str, dict[str, float]]:
    scores = signal.neuron_scores()
    out: dict[str, dict[str, float]] = {}
    for h, lin in LIN_FOR.items():
        # neuron scores keyed by linear post name for ActivationMagnitude
        # DynamicMLP uses act keys; neuron buffers are by linear layer name.
        key = lin  # fc1/fc2/fc3
        if key not in scores:
            # try hidden-layer naming
            key = h
        s = scores[key].detach().float().cpu()
        mask = net.active_mask(h).detach().cpu().bool()
        n_cap = min(s.numel(), mask.numel())
        s, mask = s[:n_cap], mask[:n_cap]
        active = s[mask]
        if active.numel() == 0:
            out[lin] = {"n_active": 0}
            continue
        a = active.numpy()
        out[lin] = {
            "n_active": int(active.numel()),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "frac_gt_0.1": float(np.mean(a > 0.1)),
            "frac_gt_0.05": float(np.mean(a > 0.05)),
            "frac_lt_0.01": float(np.mean(a < 0.01)),
        }
    return out


def run_target_rate_probe(seed: int = 0) -> dict[str, Any]:
    """MNIST protocol: 1 full train epoch, ActivationMagnitude EMA means → r*_ℓ."""
    set_seed(seed)
    device = _device()
    loaders = fashion_loaders(seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    metrics = train_epoch_signals_only(
        net, loaders["train_loader"], opt, signal, device
    )
    layers = _active_ema_stats(signal, net)
    # Lock r* = layer mean EMA (same statistic as MNIST lock).
    r_star = {
        "h1": round(float(layers["fc1"]["mean"]), 2),
        "h2": round(float(layers["fc2"]["mean"]), 2),
        "h3": round(float(layers["fc3"]["mean"]), 2),
    }
    # Raw batch mean |act| at end of epoch (sanity).
    net.eval()
    raw_sums: dict[str, list[float]] = {h: [] for h in HIDDEN_LAYERS}
    with torch.no_grad():
        for batch in loaders["probe_loader"]:
            x = batch[0].to(device)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            net(x)
            acts = net.last_activations
            assert acts is not None
            for h in HIDDEN_LAYERS:
                a = acts[{"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}[h]]
                mask = net.active_mask(h)
                active = a[:, mask].abs().mean(dim=0)
                raw_sums[h].extend(active.detach().cpu().tolist())
    raw_batch = {
        f"act_{LIN_FOR[h]}": {
            "mean": float(np.mean(raw_sums[h])),
            "median": float(np.median(raw_sums[h])),
            "p90": float(np.percentile(raw_sums[h], 90)),
        }
        for h in HIDDEN_LAYERS
        if raw_sums[h]
    }
    # Pressure under MNIST r* (miscalibration check).
    pressure_vs_mnist: dict[str, Any] = {}
    scores = signal.neuron_scores()
    for h, lin in LIN_FOR.items():
        key = lin if lin in scores else h
        s = scores[key].detach().float().cpu()
        mask = net.active_mask(h).detach().cpu().bool()
        n = min(s.numel(), mask.numel())
        s, mask = s[:n], mask[:n]
        active = s[mask]
        r_m = MNIST_RSTAR[h]
        r_f = r_star[h]
        pressure_vs_mnist[h] = {
            "mnist_r_star": r_m,
            "fashion_r_star": r_f,
            "ratio_fashion_over_mnist": r_f / r_m if r_m else float("nan"),
            "frac_above_mnist_r": float((active > r_m).float().mean().item()),
            "frac_below_mnist_r": float((active < r_m).float().mean().item()),
            "frac_above_fashion_r": float((active > r_f).float().mean().item()),
            "frac_below_fashion_r": float((active < r_f).float().mean().item()),
            "mean_ema": float(active.mean().item()),
        }

    result = {
        "dataset": "fashion_mnist",
        "protocol": (
            "DynamicMLP Phase-0 init sizes, 1 full Fashion-MNIST train epoch, "
            "ActivationMagnitude EMA (λ=0.99), seed 0. Active-only postsynaptic "
            "neuron scores. r*_ℓ = layer mean EMA (same lock rule as MNIST)."
        ),
        "seed": seed,
        "steps": int(metrics["steps"]),
        "mean_loss": float(metrics["loss"]),
        "layers": layers,
        "r_star_locked": r_star,
        "mnist_r_star_reference": MNIST_RSTAR,
        "pressure_vs_mnist_r_star": pressure_vs_mnist,
        "raw_batch_mean_abs": raw_batch,
        "verdict": {
            "reuse_mnist_r_star": "INVALID"
            if any(
                abs(r_star[h] - MNIST_RSTAR[h]) / MNIST_RSTAR[h] > 0.05
                for h in HIDDEN_LAYERS
            )
            else "CLOSE_ENOUGH",
            "note": (
                "If Fashion means differ from MNIST by >5%, locked MNIST r*_ℓ "
                "must not be used for Fashion plastic runs."
            ),
        },
    }
    return result


def _cfg(target_rates: dict[str, float], threshold_kind: str = "fixed") -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates=dict(target_rates),
        tau=0.5,
        threshold_kind=str(threshold_kind),
        mu=0.999,
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
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
        novelty_k_min=11,
        novelty_rank_fraction=1.0,
    )


def _unit_pre_biasfree(
    net: DynamicMLPReasoningNet,
    layer: str,
    slot: int,
    pre_acts: Tensor,
) -> Tensor:
    """Per-sample pre-activation W_eff · x without Parameter bias."""
    lin = LIN_FOR[layer]
    mod = net.linear_module(lin)
    w = mod.weight[slot].detach().float().cpu()
    em = net.edge_mask(lin)[slot].detach().cpu().bool()
    w_eff = w * em.float()
    pre = pre_acts.detach().float().cpu()
    return (pre * w_eff.unsqueeze(0)).sum(dim=1)


def _classify_separation(
    pre_gap: float,
    pre_nongap: float,
    post_gap: float,
    post_nongap: float,
    frac_pre_gt0: float,
) -> str:
    if pre_gap > 0 and pre_nongap < 0:
        return "SEPARATES_OK"
    if pre_gap <= 0 and pre_nongap <= 0 and frac_pre_gt0 < 0.05:
        return "RELU_DEAD_ZONE"
    if pre_gap <= 0 and pre_nongap > 0:
        return "INVERTED"
    return "PARTIAL"


def run_straddle_and_growth_gate(
    target_rates: dict[str, float],
    *,
    seed: int = 0,
    epochs: int = 6,
) -> dict[str, Any]:
    """Live births under Fashion r*: operating-point stats + growth-fire gate."""
    set_seed(seed)
    device = _device()
    loaders = fashion_loaders(seed)
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = _cfg(target_rates, "fixed")
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    global_step = 0

    birth_rows: list[dict[str, Any]] = []
    structure_trace: list[dict[str, Any]] = []
    val_hist: list[dict[str, Any]] = []
    steps_with_growth = 0
    total_grown = 0
    total_pruned = 0

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
        n_grown = int(slog.neurons_grown)
        n_pruned = int(slog.neurons_pruned)
        total_grown += n_grown
        total_pruned += n_pruned
        if n_grown > 0:
            steps_with_growth += 1

        # Match grow keys → trigger via ranked_triggers order (same as survival script).
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
            if layer not in gated.pre_acts:
                continue
            gap_mask = gated.gap_masks[(layer, trigger)]
            pre = gated.pre_acts[layer]
            lin = LIN_FOR[layer]
            mod = net.linear_module(lin)
            slot = int(slot)

            unit_pre0 = _unit_pre_biasfree(net, layer, slot, pre)
            m = gap_mask.bool().cpu().reshape(-1)
            n = min(unit_pre0.numel(), m.numel())
            unit_pre0, m = unit_pre0[:n], m[:n]
            if not bool(m.any()) or not bool((~m).any()):
                continue
            mean_gap0 = float(unit_pre0[m].mean().item())
            mean_nongap0 = float(unit_pre0[~m].mean().item())
            mean_whole0 = float(unit_pre0.mean().item())
            B_straddle = -0.5 * (mean_gap0 + mean_nongap0)
            B_gap0 = -mean_gap0 + 1e-3
            B_center = -mean_whole0
            B_actual = (
                float(mod.bias[slot].detach().cpu().item())
                if mod.bias is not None
                else 0.0
            )
            # Post-straddle (with actual Parameter bias)
            unit_pre_b = unit_pre0 + B_actual
            mean_gap_b = float(unit_pre_b[m].mean().item())
            mean_nongap_b = float(unit_pre_b[~m].mean().item())
            frac_pre_gt0 = float((unit_pre_b > 0).float().mean().item())
            post_gap = float(torch.relu(unit_pre_b[m]).mean().item())
            post_nongap = float(torch.relu(unit_pre_b[~m]).mean().item())
            post_whole = float(torch.relu(unit_pre_b).mean().item()) + 1e-8
            frac_gap_on = float((unit_pre_b[m] > 0).float().mean().item())
            frac_nongap_on = float((unit_pre_b[~m] > 0).float().mean().item())
            delta0 = mean_gap0 - mean_nongap0
            mode = _classify_separation(
                mean_gap_b, mean_nongap_b, post_gap, post_nongap, frac_pre_gt0
            )
            birth_rows.append(
                {
                    "epoch": epoch,
                    "layer": layer,
                    "slot": slot,
                    "trigger": trigger,
                    "bias_free": {
                        "mean_pre_whole": mean_whole0,
                        "mean_pre_gap": mean_gap0,
                        "mean_pre_nongap": mean_nongap0,
                        "delta_gap_minus_nongap": delta0,
                    },
                    "derived_B": {
                        "B_straddle": B_straddle,
                        "B_gap_zero": B_gap0,
                        "B_center_whole": B_center,
                        "B_actual_param": B_actual,
                        "abs_err_straddle": abs(B_actual - B_straddle),
                    },
                    "after_bias": {
                        "mean_pre_gap": mean_gap_b,
                        "mean_pre_nongap": mean_nongap_b,
                        "frac_pre_gt0": frac_pre_gt0,
                        "post_gap": post_gap,
                        "post_nongap": post_nongap,
                        "gap_over_whole_post": post_gap / post_whole,
                        "frac_gap_on": frac_gap_on,
                        "frac_nongap_on": frac_nongap_on,
                        "on_off": frac_gap_on - frac_nongap_on,
                    },
                    "failure_mode": mode,
                }
            )

        structure_trace.append(
            {
                "epoch": epoch,
                "neurons_grown": n_grown,
                "neurons_pruned": n_pruned,
                "hidden_counts": net.hidden_counts(),
                "under_floor": int(getattr(slog, "under_floor_excluded", 0) or 0),
            }
        )
        vm = evaluate(net, loaders["test_loader"], device=device)
        val_hist.append({"epoch": float(epoch), **vm})
        print(
            f"  [gate s{seed}] ep{epoch:02d} grown={n_grown} pruned={n_pruned} "
            f"acc={vm['acc']:.4f} H={net.hidden_counts()} births_logged={len(birth_rows)}",
            flush=True,
        )

    # Aggregate operating point
    n = len(birth_rows)
    if n > 0:
        mean_gap0 = float(np.mean([r["bias_free"]["mean_pre_gap"] for r in birth_rows]))
        mean_nongap0 = float(
            np.mean([r["bias_free"]["mean_pre_nongap"] for r in birth_rows])
        )
        mean_B_straddle = float(
            np.mean([r["derived_B"]["B_straddle"] for r in birth_rows])
        )
        mean_B_actual = float(
            np.mean([r["derived_B"]["B_actual_param"] for r in birth_rows])
        )
        modes = [r["failure_mode"] for r in birth_rows]
        mode_counts = {m: modes.count(m) for m in sorted(set(modes))}
        frac_separates = modes.count("SEPARATES_OK") / n
        frac_dead = modes.count("RELU_DEAD_ZONE") / n
        mean_gow = float(
            np.mean([r["after_bias"]["gap_over_whole_post"] for r in birth_rows])
        )
        mean_onoff = float(np.mean([r["after_bias"]["on_off"] for r in birth_rows]))
        # Birth-gate clearance (MNIST bars: gow≥1.3, on_off≥0.2)
        frac_gow = float(
            np.mean(
                [r["after_bias"]["gap_over_whole_post"] >= 1.3 for r in birth_rows]
            )
        )
        frac_onoff = float(
            np.mean([r["after_bias"]["on_off"] >= 0.2 for r in birth_rows])
        )
        frac_both = float(
            np.mean(
                [
                    r["after_bias"]["gap_over_whole_post"] >= 1.3
                    and r["after_bias"]["on_off"] >= 0.2
                    for r in birth_rows
                ]
            )
        )
        operating = {
            "n_births_measured": n,
            "bias_free_means": {
                "mean_pre_whole": float(
                    np.mean([r["bias_free"]["mean_pre_whole"] for r in birth_rows])
                ),
                "mean_pre_gap": mean_gap0,
                "mean_pre_nongap": mean_nongap0,
                "mean_delta": mean_gap0 - mean_nongap0,
                "mnist_reference_gap_nongap": (-5.0, -11.4),
                "note": (
                    "MNIST bias-free sat near mean_gap≈−5 / mean_nongap≈−11; "
                    "Fashion numbers here are the re-derived operating point."
                ),
            },
            "derived": {
                "B_straddle_mean": mean_B_straddle,
                "B_straddle_median": float(
                    np.median([r["derived_B"]["B_straddle"] for r in birth_rows])
                ),
                "B_actual_mean": mean_B_actual,
                "formula": "B = -(mean_pre_gap + mean_pre_nongap)/2  (per unit, from probe)",
                "expected_mean_gap_after": 0.5 * (mean_gap0 - mean_nongap0),
                "expected_mean_nongap_after": -0.5 * (mean_gap0 - mean_nongap0),
            },
            "after_straddle": {
                "mean_pre_gap": float(
                    np.mean([r["after_bias"]["mean_pre_gap"] for r in birth_rows])
                ),
                "mean_pre_nongap": float(
                    np.mean([r["after_bias"]["mean_pre_nongap"] for r in birth_rows])
                ),
                "mean_gap_over_whole_post": mean_gow,
                "mean_on_off": mean_onoff,
                "frac_separates_ok": frac_separates,
                "frac_relu_dead_zone": frac_dead,
                "failure_mode_counts": mode_counts,
                "birth_gate_frac_gow_ge_1.3": frac_gow,
                "birth_gate_frac_onoff_ge_0.2": frac_onoff,
                "birth_gate_frac_both": frac_both,
            },
            "verdict": (
                "STRADDLE_OK"
                if frac_separates >= 0.9 and frac_dead <= 0.05
                else "STRADDLE_DEGRADED"
                if frac_separates >= 0.5
                else "STRADDLE_FAIL"
            ),
        }
    else:
        operating = {
            "n_births_measured": 0,
            "verdict": "NO_BIRTHS",
            "note": "No gap_imprinted births measured; cannot validate straddle.",
        }

    # Growth-fire gate (MNIST criterion: edges+neurons grown >=5 and ≥1 step with growth;
    # for gated_novelty we require growth every epoch-ish and non-zero establish path).
    gate_pass = (
        total_grown >= 5
        and steps_with_growth >= max(1, epochs // 2)
        and n > 0
        and operating.get("verdict") in ("STRADDLE_OK", "STRADDLE_DEGRADED")
    )
    # Acc stability: max swing between consecutive epochs
    accs = [v["acc"] for v in val_hist]
    swings = [abs(accs[i] - accs[i - 1]) for i in range(1, len(accs))] if len(accs) > 1 else []
    max_swing = float(max(swings)) if swings else 0.0

    growth_gate = {
        "protocol": f"{epochs} epochs Fashion-MNIST, seed {seed}, Fashion r*_ℓ, fixed+gated_novelty+gap_imprinted",
        "target_rates": target_rates,
        "gate_criterion": (
            "neurons_grown_total >= 5 AND steps_with_growth >= epochs/2 "
            "AND at least one birth with straddle measurement"
        ),
        "neurons_grown_total": total_grown,
        "neurons_pruned_total": total_pruned,
        "steps_with_any_growth": steps_with_growth,
        "epochs": epochs,
        "growth_fire_gate_pass": bool(gate_pass),
        "final_hidden": net.hidden_counts(),
        "final_test_acc": float(accs[-1]) if accs else float("nan"),
        "acc_trajectory": accs,
        "max_consecutive_acc_swing_pp": max_swing * 100.0,
        "unstable_swings_flag": max_swing > 0.08,
        "structure_trace": structure_trace,
        "val_history": val_hist,
    }

    return {
        "dataset": "fashion_mnist",
        "seed": seed,
        "target_rates_used": target_rates,
        "operating_point": operating,
        "growth_fire_gate": growth_gate,
        "birth_rows": birth_rows,
    }


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs-gate", type=int, default=6)
    ap.add_argument(
        "--skip-probe",
        action="store_true",
        help="Reuse target_rate_probe.json if present",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Target-rate probe ----
    probe_path = OUT / "target_rate_probe.json"
    if args.skip_probe and probe_path.exists():
        print(f"Reusing {probe_path}", flush=True)
        probe = json.loads(probe_path.read_text())
    else:
        print("=== Fashion-MNIST target-rate probe (1 epoch) ===", flush=True)
        probe = run_target_rate_probe(seed=args.seed)
        probe_path.write_text(json.dumps(probe, indent=2))
        print(json.dumps(probe["r_star_locked"], indent=2), flush=True)
        print(
            f"vs MNIST: {json.dumps(probe['pressure_vs_mnist_r_star'], indent=2)}",
            flush=True,
        )
        print(f"Wrote {probe_path}", flush=True)

    r_star = probe["r_star_locked"]
    print(f"\nLocked Fashion r*_ℓ = {r_star}", flush=True)

    # ---- 2+3. Straddle operating point + growth-fire gate ----
    print(
        f"\n=== Straddle re-measure + growth-fire gate ({args.epochs_gate} ep) ===",
        flush=True,
    )
    gate = run_straddle_and_growth_gate(
        r_star, seed=args.seed, epochs=args.epochs_gate
    )
    # Drop bulky birth_rows from top-level summary; keep in separate file
    birth_rows = gate.pop("birth_rows")
    (OUT / "birth_operating_points.json").write_text(
        json.dumps({"rows": birth_rows, "n": len(birth_rows)}, indent=2)
    )
    (OUT / "straddle_and_growth_gate.json").write_text(json.dumps(gate, indent=2))

    # ---- Summary ----
    op = gate["operating_point"]
    gg = gate["growth_fire_gate"]
    summary = {
        "dataset": "fashion_mnist",
        "status": "CALIBRATION_COMPLETE",
        "r_star_fashion": r_star,
        "r_star_mnist_was": MNIST_RSTAR,
        "r_star_reuse_verdict": probe["verdict"]["reuse_mnist_r_star"],
        "straddle_verdict": op.get("verdict"),
        "growth_fire_gate_pass": gg["growth_fire_gate_pass"],
        "unstable_swings_flag": gg["unstable_swings_flag"],
        "max_consecutive_acc_swing_pp": gg["max_consecutive_acc_swing_pp"],
        "final_test_acc_6ep": gg["final_test_acc"],
        "n_births_measured": op.get("n_births_measured", 0),
        "bias_free_means": op.get("bias_free_means"),
        "after_straddle": op.get("after_straddle"),
        "prior_fashion_pareto": {
            "verdict_was": "FASHION_PARETO_BOTH_FAIL",
            "status_now": (
                "INVALID_UNTIL_RECALIBRATED_RERUN"
                if probe["verdict"]["reuse_mnist_r_star"] == "INVALID"
                else "R_STAR_CLOSE_RECHECK_OPTIONAL"
            ),
            "reason": (
                "Plastic Fashion Pareto used MNIST-locked r*_ℓ and was not "
                "re-probed; unstable mid-training swings are consistent with "
                "miscalibrated homeostatic targets, not a clean mechanism ceiling."
            ),
        },
        "next_steps": [
            "If r*_ℓ differs from MNIST: write configs/phase1a_fashion.yaml with Fashion rates",
            "Re-run plastic fixed+sliding 20-ep under Fashion r* before any Pareto claim",
            "Only then re-score FASHION_PARETO_*",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Human README
    lines = [
        "# Fashion-MNIST calibration (re-probe of MNIST sequence)",
        "",
        "**Status:** calibration complete — prior `FASHION_PARETO_BOTH_FAIL` is **not** a valid mechanism-efficiency finding until plastic arms re-run under these constants.",
        "",
        "## 1. Target-rate probe",
        "",
        f"Protocol: 1 epoch Fashion-MNIST, ActivationMagnitude EMA λ=0.99, seed {args.seed}.",
        "",
        "| Layer | Fashion r*_ℓ (mean EMA) | MNIST r*_ℓ (was) | ratio |",
        "|-------|------------------------:|-----------------:|------:|",
    ]
    for h in HIDDEN_LAYERS:
        fr = r_star[h]
        mr = MNIST_RSTAR[h]
        lines.append(f"| {h} | **{fr:.2f}** | {mr:.2f} | {fr/mr:.2f}× |")
    lines += [
        "",
        f"**Reuse MNIST r*:** `{probe['verdict']['reuse_mnist_r_star']}`",
        "",
        "## 2. Straddle operating point",
        "",
        f"Verdict: **`{op.get('verdict')}`** (n={op.get('n_births_measured', 0)} births)",
        "",
    ]
    if op.get("bias_free_means"):
        bf = op["bias_free_means"]
        der = op["derived"]
        af = op["after_straddle"]
        lines += [
            "Bias-free pre (after sep-power + open-fan, B=0):",
            "",
            f"- mean_pre_gap = **{bf['mean_pre_gap']:.2f}** (MNIST ref ≈ −5.0)",
            f"- mean_pre_nongap = **{bf['mean_pre_nongap']:.2f}** (MNIST ref ≈ −11.4)",
            f"- mean Δ = **{bf['mean_delta']:.2f}**",
            f"- B_straddle mean = **{der['B_straddle_mean']:.2f}** (formula unchanged, per-unit)",
            "",
            "After orchestrator straddle B:",
            "",
            f"- mean pre gap / nongap = **{af['mean_pre_gap']:.2f} / {af['mean_pre_nongap']:.2f}**",
            f"- SEPARATES_OK frac = **{af['frac_separates_ok']:.0%}**",
            f"- RELU_DEAD_ZONE frac = **{af['frac_relu_dead_zone']:.0%}**",
            f"- birth-gate both (gow≥1.3 ∧ on_off≥0.2) = **{af['birth_gate_frac_both']:.0%}**",
            "",
        ]
    lines += [
        "## 3. Growth-fire gate",
        "",
        f"- pass: **`{gg['growth_fire_gate_pass']}`**",
        f"- grown / pruned: {gg['neurons_grown_total']} / {gg['neurons_pruned_total']}",
        f"- steps with growth: {gg['steps_with_any_growth']}/{gg['epochs']}",
        f"- final test acc: {gg['final_test_acc']:.4f}",
        f"- max consecutive acc swing: {gg['max_consecutive_acc_swing_pp']:.1f} pp",
        f"- unstable_swings_flag: **{gg['unstable_swings_flag']}**",
        "",
        "## Prior Pareto",
        "",
        f"`{summary['prior_fashion_pareto']['status_now']}` — {summary['prior_fashion_pareto']['reason']}",
        "",
        "## Artifacts",
        "",
        "- `target_rate_probe.json`",
        "- `straddle_and_growth_gate.json`",
        "- `birth_operating_points.json`",
        "- `summary.json`",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
