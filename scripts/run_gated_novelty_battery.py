#!/usr/bin/env python3
"""Gated-novelty + gap-imprint battery (dead-ReLU, specialist, establishment)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import (  # noqa: E402
    HIDDEN_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.masks import NEURON_ORIGIN_GROWN  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import (  # noqa: E402
    PlasticityConfig,
    competitive_neuron_zscores,
    gap_imprint_direction,
    population_signals_from_activation,
    scale_direction_to_kaiming_relu,
    select_neurons_to_prune,
)
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)
from neuronca.training.optim import build_optimizer  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(
    weight_init: str = "gap_imprinted",
    *,
    threshold_kind: str = "fixed",
    mu: float = 0.999,
) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates={"h1": 0.77, "h2": 1.02, "h3": 1.28},
        tau=0.5,
        threshold_kind=str(threshold_kind),
        mu=float(mu),
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=4,
        seed_edges_per_new_neuron=4,
        newborn_grace_steps=3,
        newborn_ema_init="layer_target",
        newborn_birth_bias=0.5,
        newborn_birth_bias_steps=3,
        newborn_weight_init=str(weight_init),
        newborn_parent_noise=0.1,
        newborn_copy_outgoing=True,
        newborn_parent_coord_jitter=0.05,
        newborn_lr_mult=1.0,
        neuron_growth_trigger="gated_novelty",
        novelty_k_min=11,
        novelty_rank_fraction=1.0,
    )


@torch.no_grad()
def _specialist_act_profile(
    acts: torch.Tensor,
    gap_mask: torch.Tensor,
) -> dict[str, float]:
    """Compare unit activation on own gap samples vs whole probe.

    ``acts``: [B] post-act for one unit. ``gap_mask``: [B] bool birth gap samples.
    """
    a = acts.detach().float().abs().cpu().reshape(-1)
    m = gap_mask.bool().cpu().reshape(-1)
    if m.numel() != a.numel():
        # Probe length drift — truncate to min
        n = min(int(m.numel()), int(a.numel()))
        a, m = a[:n], m[:n]
    whole = float(a.mean().item()) if a.numel() else 0.0
    if bool(m.any()):
        gap = float(a[m].mean().item())
        gap_p90 = float(torch.quantile(a[m], 0.9).item()) if m.sum() >= 4 else gap
    else:
        gap, gap_p90 = float("nan"), float("nan")
    whole_p90 = float(torch.quantile(a, 0.9).item()) if a.numel() >= 4 else whole
    non_gap = float(a[~m].mean().item()) if bool((~m).any()) else float("nan")
    ratio = gap / whole if whole > 1e-8 else float("inf")
    return {
        "mean_act_whole": whole,
        "mean_act_gap": gap,
        "mean_act_non_gap": non_gap,
        "p90_act_whole": whole_p90,
        "p90_act_gap": gap_p90,
        "gap_over_whole": ratio,
        "frac_gap_samples": float(m.float().mean().item()) if m.numel() else 0.0,
        "n_gap": int(m.sum().item()) if m.numel() else 0,
    }


def _concat_weights(net: DynamicMLPReasoningNet, layer: str, slot: int) -> torch.Tensor:
    """Incoming row + outgoing col (where present) for cosine."""
    parts: list[torch.Tensor] = []
    for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
        mod = net.linear_module(lin)
        if post == layer:
            parts.append(mod.weight[slot].detach().float().cpu().reshape(-1))
        if pre == layer:
            parts.append(mod.weight[:, slot].detach().float().cpu().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.norm().clamp(min=1e-8)
    bn = b.norm().clamp(min=1e-8)
    return float((a @ b) / (an * bn))


@torch.no_grad()
def _probe_batches(
    model: DynamicMLPReasoningNet,
    loader,
    device: torch.device,
    max_batches: int = 8,
):
    model.eval()
    xs, ys, acts_h, logits_list = [], [], {h: [] for h in HIDDEN_LAYERS}, []
    key = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        if y.ndim > 1:
            y = y.view(-1)
        logits = model(x)
        acts = model.last_activations
        assert acts is not None
        xs.append(x.cpu())
        ys.append(y.cpu())
        logits_list.append(logits.detach().cpu())
        for h in HIDDEN_LAYERS:
            acts_h[h].append(acts[key[h]].detach().cpu())
    if not xs:
        return None
    return {
        "x": torch.cat(xs, 0),
        "y": torch.cat(ys, 0),
        "logits": torch.cat(logits_list, 0),
        "acts": {h: torch.cat(acts_h[h], 0) for h in HIDDEN_LAYERS},
    }


@torch.no_grad()
def _pre_post_relu_for_slot(
    model: DynamicMLPReasoningNet,
    layer: str,
    slot: int,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-ReLU and post-ReLU for one hidden slot on batch x."""
    device = next(model.parameters()).device
    x = x.to(device)
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    act_in = x
    # Manual forward to expose pre-acts (mirrors DynamicMLP.forward).
    pre1 = F.linear(
        act_in, model.effective_weight("fc1"), model.effective_bias("fc1")
    )
    h1 = F.relu(pre1)
    pre2 = F.linear(h1, model.effective_weight("fc2"), model.effective_bias("fc2"))
    h2 = F.relu(pre2)
    pre3 = F.linear(h2, model.effective_weight("fc3"), model.effective_bias("fc3"))
    h3 = F.relu(pre3)
    if layer == "h1":
        return pre1[:, slot].detach().cpu(), h1[:, slot].detach().cpu()
    if layer == "h2":
        return pre2[:, slot].detach().cpu(), h2[:, slot].detach().cpu()
    if layer == "h3":
        return pre3[:, slot].detach().cpu(), h3[:, slot].detach().cpu()
    raise ValueError(layer)


def _dead_relu_stats(pre: torch.Tensor, post: torch.Tensor) -> dict[str, float]:
    frac_post0 = float((post == 0).float().mean().item())
    mean_post = float(post.abs().mean().item())
    mean_pre = float(pre.mean().item())
    if frac_post0 >= 0.99:
        kind = "dead_relu"
    elif frac_post0 >= 0.5:
        kind = "mostly_dead"
    else:
        kind = "alive_attenuated" if mean_post > 0 else "dead_relu"
    return {
        "frac_post_zero": frac_post0,
        "mean_post": mean_post,
        "mean_pre": mean_pre,
        "kind": kind,
    }


@torch.no_grad()
def _ablate_neuron_delta_l(
    model: DynamicMLPReasoningNet,
    layer: str,
    slot: int,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
) -> float:
    """ΔL⁺ = max(0, L_abl − L_base) on probe (optional sample subset)."""
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    if sample_mask is not None:
        m = sample_mask.to(device)
        x, y = x[m], y[m]
    if x.numel() == 0:
        return 0.0
    model.eval()
    logits = model(x)
    l_base = float(F.cross_entropy(logits, y).item())

    # Zero incoming row + bias + outgoing cols (Phase 0–style).
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
        mod = model.linear_module(lin)
        if post == layer:
            saved.append((mod.weight, mod.weight[slot].detach().clone()))
            mod.weight[slot] = 0
            if mod.bias is not None:
                saved.append((mod.bias, mod.bias[slot : slot + 1].detach().clone()))
                mod.bias[slot] = 0
        if pre == layer:
            saved.append((mod.weight, mod.weight[:, slot].detach().clone()))
            mod.weight[:, slot] = 0
    logits_a = model(x)
    l_abl = float(F.cross_entropy(logits_a, y).item())
    # Restore
    # Re-apply saved carefully: weight rows/cols and bias entries.
    # Simpler: re-load from clones we took — but we only stored values.
    # Restore by writing back in reverse order is wrong for overlapping.
    # Use explicit restore from second pass structure:
    # We need a better restore. Re-run with stored full tensors.
    del logits_a
    # Restore from saved list — each entry is (param_tensor, value_slice_or_col).
    # Because we overwrote, restore the last saved for each unique view.
    # Cleaner approach: keep full weight clones per modified module.
    # Rebuild restore from a dedicated map:
    # (Handled below with a dedicated path — see call site that uses full snapshot.)
    return max(0.0, l_abl - l_base)


@torch.no_grad()
def _ablate_with_restore(
    model: DynamicMLPReasoningNet,
    layer: str,
    slot: int,
    x: torch.Tensor,
    y: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> float:
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    if sample_mask is not None:
        m = sample_mask.bool()
        x, y = x[m], y[m]
    if y.numel() == 0:
        return 0.0
    snaps: dict[str, dict[str, torch.Tensor]] = {}
    for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
        if post != layer and pre != layer:
            continue
        mod = model.linear_module(lin)
        snaps[lin] = {
            "w": mod.weight.detach().clone(),
            "b": mod.bias.detach().clone() if mod.bias is not None else None,  # type: ignore
        }
    logits = model(x)
    l_base = float(F.cross_entropy(logits, y).item())
    for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
        mod = model.linear_module(lin)
        if post == layer:
            mod.weight[slot] = 0
            if mod.bias is not None:
                mod.bias[slot] = 0
        if pre == layer:
            mod.weight[:, slot] = 0
    logits_a = model(x)
    l_abl = float(F.cross_entropy(logits_a, y).item())
    for lin, snap in snaps.items():
        mod = model.linear_module(lin)
        mod.weight.copy_(snap["w"])
        if snap["b"] is not None and mod.bias is not None:
            mod.bias.copy_(snap["b"])
    return max(0.0, l_abl - l_base)


def run_battery(
    *,
    seed: int = 0,
    epochs: int = 6,
    out_dir: Path | None = None,
    weight_init: str = "gap_imprinted",
    threshold_kind: str = "fixed",
    mu: float = 0.999,
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    out_dir = out_dir or (
        _REPO / "results" / "phase1a_diagnostics" / "gated_novelty_battery"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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
        input_dim=784,
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = _cfg(weight_init=weight_init, threshold_kind=threshold_kind, mu=mu)
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, name="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)

    # Birth records: (epoch, layer, child, parent)
    births: list[dict[str, Any]] = []
    dead_relu_birth: list[dict[str, Any]] = []
    cos_traces: dict[tuple[str, int, int], list[float]] = {}
    under_floor_trace: list[dict[str, Any]] = []
    prune_recheck: list[dict[str, Any]] = []
    # Live gap-imprinted specialists: key (layer, child) → tracking dict
    live_specialists: dict[tuple[str, int], dict[str, Any]] = {}
    specialist_profiles: list[dict[str, Any]] = []

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

        gated_ctx = collect_gated_novelty_probe(
            net,
            loaders["probe_loader"],
            device=device,
            radius=cfg.radius,
            k_min=cfg.novelty_k_min,
        )
        # Snapshot active set before grow for parent tracking.
        pre_active = {h: net.active_mask(h).cpu().clone() for h in HIDDEN_LAYERS}
        scores = signal.neuron_scores()
        # Move scores to cpu if needed
        scores = {k: v.detach().cpu() for k, v in scores.items()}
        if "in" not in scores:
            scores["in"] = torch.zeros(784)

        # Full probe for specialist profiles (same loader as gated novelty).
        probe_full = _probe_batches(
            net, loaders["probe_loader"], device, max_batches=64
        )

        # --- Specialist profile BEFORE structure step (units still alive) ---
        # Also capture profiles for units about to be pruned this step.
        if probe_full is not None:
            for key, tr in list(live_specialists.items()):
                layer, child = key
                if not bool(net.active_mask(layer)[child]):
                    continue
                acts = probe_full["acts"][layer][:, child]
                gmask = tr["gap_mask"]
                prof = _specialist_act_profile(acts, gmask)
                # Judgment-relevant EMA score
                fc = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[layer]
                ema = float(scores[fc][child].item()) if child < scores[fc].numel() else float("nan")
                r_star = float(cfg.resolved_target_rates()[layer])
                row = {
                    "epoch": epoch,
                    "phase": "pre_structure",
                    "layer": layer,
                    "child": child,
                    "age": epoch - int(tr["birth_epoch"]),
                    "ema_act": ema,
                    "r_star": r_star,
                    "ema_minus_r_star": ema - r_star,
                    **prof,
                }
                tr["profiles"].append(row)

        # Own-subset vs whole-probe re-check on units the live prune rule
        # would remove this step (pre-structure, same signals as orch).
        probe_pre = probe_full
        if probe_pre is not None and epoch >= 1:
            pop = population_signals_from_activation(net, scores)
            policy = orch.ensure_policy(net)
            hidden_signals = {h: pop[h] for h in HIDDEN_LAYERS}
            active_m = {h: net.active_mask(h).cpu() for h in HIDDEN_LAYERS}
            policy.update(hidden_signals, active_m)
            y = probe_pre["y"]
            x = probe_pre["x"]
            acts = probe_pre["acts"]
            for h in HIDDEN_LAYERS:
                z = competitive_neuron_zscores(net, h, pop)
                to_prune = select_neurons_to_prune(
                    net,
                    h,
                    z,
                    policy,
                    cfg.prune_neuron_z,
                    grace_steps=cfg.newborn_grace_steps,
                )[:12]
                for slot in to_prune:
                    a = acts[h][:, slot]
                    class_means = []
                    for c in range(10):
                        m = y == c
                        class_means.append(
                            float(a[m].mean().item()) if bool(m.any()) else -1e9
                        )
                    pref = int(max(range(10), key=lambda c: class_means[c]))
                    own_mask = y == pref
                    dl_whole = _ablate_with_restore(net, h, slot, x, y, None)
                    dl_own = _ablate_with_restore(net, h, slot, x, y, own_mask)
                    prune_recheck.append(
                        {
                            "epoch": epoch,
                            "layer": h,
                            "slot": int(slot),
                            "preferred_class": pref,
                            "delta_l_whole": dl_whole,
                            "delta_l_own_subset": dl_own,
                            "own_gt_whole": dl_own > dl_whole + 1e-6,
                            "own_over_whole": (
                                dl_own / dl_whole if dl_whole > 1e-8 else None
                            ),
                        }
                    )
                    # If this is a tracked specialist, mark last profile as at_prune
                    key = (h, int(slot))
                    if key in live_specialists and live_specialists[key]["profiles"]:
                        last = dict(live_specialists[key]["profiles"][-1])
                        last["phase"] = "at_prune_candidate"
                        last["would_prune"] = True
                        live_specialists[key]["profiles"].append(last)

        slog = orch.step(
            net,
            scores,
            optimizer=opt,
            epoch=epoch,
            signal=signal,
            gated_novelty=gated_ctx,
        )

        # Finalize specialists that died this step
        for key, tr in list(live_specialists.items()):
            layer, child = key
            if not bool(net.active_mask(layer)[child]):
                tr["pruned_epoch"] = epoch
                tr["survived"] = False
                specialist_profiles.append(tr)
                del live_specialists[key]
        under_floor_trace.append(
            {
                "epoch": epoch,
                "under_floor_total": slog.novelty_under_floor_total,
                "under_floor_by_layer": dict(slog.novelty_under_floor_by_layer),
                "under_floor_frac_by_layer": dict(
                    slog.novelty_under_floor_frac_by_layer
                ),
                "candidates_by_layer": dict(slog.novelty_candidates_by_layer),
                "neurons_grown": slog.neurons_grown,
                "neurons_pruned": slog.neurons_pruned,
            }
        )

        # Newborns this step. Grow order matches ranked trigger order (rate-cap).
        # Fresh probe after grow for birth specialist profiles.
        probe = _probe_batches(net, loaders["probe_loader"], device, max_batches=64)
        grow_idx_by_layer: dict[str, int] = {h: 0 for h in HIDDEN_LAYERS}
        for layer, slot in slog.neuron_grow_keys:
            child_w = _concat_weights(net, layer, slot)
            lin_in = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[layer]
            child_in = net.linear_module(lin_in).weight[slot].detach().float().cpu()

            # Official trigger: ith growth this layer ↔ ith ranked candidate.
            ranked = []
            if layer in gated_ctx.by_layer:
                ranked = list(gated_ctx.by_layer[layer].ranked_triggers)
            gi = grow_idx_by_layer[layer]
            grow_idx_by_layer[layer] = gi + 1
            trigger = int(ranked[gi]) if gi < len(ranked) else None

            # Cosine to trigger unit (clone test) — only if trigger still active.
            cos_trigger = None
            if trigger is not None and bool(pre_active[layer][trigger]):
                cos_trigger = _cosine(
                    child_w, _concat_weights(net, layer, int(trigger))
                )

            # Cosine to gap imprint direction for this trigger.
            cos_gap = None
            gap_dir_norm = None
            gmask = None
            if (
                trigger is not None
                and (layer, int(trigger)) in gated_ctx.gap_masks
                and layer in gated_ctx.pre_acts
            ):
                gmask = gated_ctx.gap_masks[(layer, int(trigger))].clone()
                pre_a = gated_ctx.pre_acts[layer]
                try:
                    gdir = scale_direction_to_kaiming_relu(
                        gap_imprint_direction(
                            pre_a, gmask, gated_ctx.ce, subtract_global_mean=True
                        )
                    )
                    cos_gap = _cosine(child_in, gdir)
                    gap_dir_norm = float(gdir.norm().item())
                except ValueError:
                    pass

            rec = {
                "epoch": epoch,
                "layer": layer,
                "child": int(slot),
                "trigger": trigger,
                "locus": trigger,
                "locus_child_cos_birth": cos_trigger,
                "gap_dir_child_cos_birth": cos_gap,
                "gap_dir_norm": gap_dir_norm,
                "weight_init": cfg.newborn_weight_init,
            }
            births.append(rec)
            if trigger is not None and cos_trigger is not None:
                cos_traces[(layer, int(trigger), int(slot))] = [float(cos_trigger)]

            if probe is not None:
                pre, post = _pre_post_relu_for_slot(
                    net, layer, int(slot), probe["x"][:256]
                )
                st = _dead_relu_stats(pre, post)
                dead_relu_birth.append({**rec, **st})

            # Register specialist tracker with birth gap mask + birth profile.
            if gmask is not None and probe is not None:
                acts_c = probe["acts"][layer][:, int(slot)]
                # Align mask length to this probe if needed
                if gmask.numel() != acts_c.shape[0]:
                    # Recompute gap mask length from gated_ctx post acts length
                    # vs current probe — use min overlap
                    n = min(int(gmask.numel()), int(acts_c.shape[0]))
                    gmask_use = gmask[:n]
                    acts_use = acts_c[:n]
                else:
                    gmask_use = gmask
                    acts_use = acts_c
                birth_prof = _specialist_act_profile(acts_use, gmask_use)
                fc = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[layer]
                # EMA just seeded to r* on grow — read signal buffer if present
                ema = float("nan")
                neuron = getattr(signal, "_neuron", None)
                if isinstance(neuron, dict) and fc in neuron:
                    buf = neuron[fc]
                    if int(slot) < buf.shape[0]:
                        ema = float(buf[int(slot)].item())
                live_specialists[(layer, int(slot))] = {
                    "layer": layer,
                    "child": int(slot),
                    "trigger": trigger,
                    "birth_epoch": epoch,
                    "gap_mask": gmask_use.cpu().clone(),
                    "cos_gap_birth": cos_gap,
                    "cos_trigger_birth": cos_trigger,
                    "profiles": [
                        {
                            "epoch": epoch,
                            "phase": "at_birth",
                            "layer": layer,
                            "child": int(slot),
                            "age": 0,
                            "ema_act": ema,
                            "r_star": float(cfg.resolved_target_rates()[layer]),
                            "ema_minus_r_star": (
                                ema - float(cfg.resolved_target_rates()[layer])
                                if ema == ema
                                else float("nan")
                            ),
                            **birth_prof,
                        }
                    ],
                }

        # Age cos traces for surviving pairs
        for key, series in list(cos_traces.items()):
            layer, parent, child = key
            if not bool(net.active_mask(layer)[child]):
                continue
            if not bool(net.active_mask(layer)[parent]):
                continue
            series.append(
                _cosine(
                    _concat_weights(net, layer, child),
                    _concat_weights(net, layer, parent),
                )
            )

    # Survivors still alive at end
    for key, tr in list(live_specialists.items()):
        tr["pruned_epoch"] = None
        tr["survived"] = True
        specialist_profiles.append(tr)
    live_specialists.clear()

    # --- Establishment + redundancy ---
    grace = cfg.newborn_grace_steps
    step_now = int(net.cooldown.plasticity_step)
    established = []
    for rec in births:
        layer, child = rec["layer"], rec["child"]
        if not bool(net.active_mask(layer)[child]):
            continue
        birth_step = int(net.birth_step(layer)[child].item())
        age = step_now - birth_step
        if age < grace:
            continue
        origin = int(net.neuron_origin(layer)[child].item())
        if origin != NEURON_ORIGIN_GROWN:
            continue
        locus = rec.get("locus")
        cos_series = (
            cos_traces.get((layer, locus, child), []) if locus is not None else []
        )
        established.append(
            {
                **rec,
                "age": age,
                "cos_locus_birth": cos_series[0] if cos_series else rec.get(
                    "locus_child_cos_birth"
                ),
                "cos_locus_final": cos_series[-1] if cos_series else None,
                "cos_locus_decay": (
                    (cos_series[0] - cos_series[-1]) if len(cos_series) >= 2 else None
                ),
                "cos_gap_birth": rec.get("gap_dir_child_cos_birth"),
            }
        )

    # Summaries
    n_birth = len(dead_relu_birth)
    kinds = {"dead_relu": 0, "mostly_dead": 0, "alive_attenuated": 0}
    for r in dead_relu_birth:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1

    cos_finals = [
        e["cos_locus_final"]
        for e in established
        if e.get("cos_locus_final") is not None
    ]
    cos_gap_births = [
        r["gap_dir_child_cos_birth"]
        for r in births
        if r.get("gap_dir_child_cos_birth") is not None
    ]
    cos_locus_births = [
        r["locus_child_cos_birth"]
        for r in births
        if r.get("locus_child_cos_birth") is not None
    ]
    # --- Specialist profile aggregation (gap vs whole before death) ---
    # Use last pre_structure / at_prune profile per unit; also birth.
    def _safe_mean(vals: list[float]) -> float | None:
        vals = [v for v in vals if v == v and v not in (float("inf"), float("-inf"))]
        return sum(vals) / len(vals) if vals else None

    birth_rows: list[dict[str, Any]] = []
    last_pre_death: list[dict[str, Any]] = []
    for tr in specialist_profiles:
        profs = tr.get("profiles") or []
        # strip non-serializable gap_mask for rows later
        birth_p = next((p for p in profs if p.get("phase") == "at_birth"), None)
        # last profile before death (prefer at_prune_candidate, else last pre_structure)
        last_p = None
        for p in reversed(profs):
            if p.get("phase") in ("at_prune_candidate", "pre_structure"):
                last_p = p
                break
        if birth_p is not None:
            birth_rows.append(
                {
                    "layer": tr["layer"],
                    "child": tr["child"],
                    "trigger": tr.get("trigger"),
                    "survived": tr.get("survived"),
                    "pruned_epoch": tr.get("pruned_epoch"),
                    **{k: birth_p[k] for k in birth_p if k not in ("layer", "child")},
                }
            )
        if last_p is not None:
            last_pre_death.append(
                {
                    "layer": tr["layer"],
                    "child": tr["child"],
                    "survived": tr.get("survived"),
                    "pruned_epoch": tr.get("pruned_epoch"),
                    "n_profiles": len(profs),
                    **{k: last_p[k] for k in last_p if k not in ("layer", "child")},
                }
            )

    # Verdict helpers
    def _frac(pred, rows):
        if not rows:
            return None
        return sum(1 for r in rows if pred(r)) / len(rows)

    specialist_summary = {
        "n_tracked": len(specialist_profiles),
        "n_pruned": sum(1 for t in specialist_profiles if not t.get("survived")),
        "n_survived_end": sum(1 for t in specialist_profiles if t.get("survived")),
        "at_birth": {
            "n": len(birth_rows),
            "mean_act_whole": _safe_mean([r["mean_act_whole"] for r in birth_rows]),
            "mean_act_gap": _safe_mean([r["mean_act_gap"] for r in birth_rows]),
            "mean_act_non_gap": _safe_mean(
                [r["mean_act_non_gap"] for r in birth_rows]
            ),
            "mean_gap_over_whole": _safe_mean(
                [r["gap_over_whole"] for r in birth_rows]
            ),
            "frac_gap_over_whole_ge_1.5": _frac(
                lambda r: r.get("gap_over_whole", 0) >= 1.5, birth_rows
            ),
            "frac_gap_over_whole_ge_2.0": _frac(
                lambda r: r.get("gap_over_whole", 0) >= 2.0, birth_rows
            ),
            "frac_gap_gt_whole": _frac(
                lambda r: r.get("mean_act_gap", 0) > r.get("mean_act_whole", 0) + 1e-6,
                birth_rows,
            ),
        },
        "last_pre_death_or_end": {
            "n": len(last_pre_death),
            "mean_act_whole": _safe_mean(
                [r["mean_act_whole"] for r in last_pre_death]
            ),
            "mean_act_gap": _safe_mean([r["mean_act_gap"] for r in last_pre_death]),
            "mean_act_non_gap": _safe_mean(
                [r["mean_act_non_gap"] for r in last_pre_death]
            ),
            "mean_gap_over_whole": _safe_mean(
                [r["gap_over_whole"] for r in last_pre_death]
            ),
            "mean_ema_act": _safe_mean([r.get("ema_act", float("nan")) for r in last_pre_death]),
            "mean_ema_minus_r_star": _safe_mean(
                [r.get("ema_minus_r_star", float("nan")) for r in last_pre_death]
            ),
            "frac_gap_over_whole_ge_1.5": _frac(
                lambda r: r.get("gap_over_whole", 0) >= 1.5, last_pre_death
            ),
            "frac_gap_over_whole_ge_2.0": _frac(
                lambda r: r.get("gap_over_whole", 0) >= 2.0, last_pre_death
            ),
            "frac_gap_gt_whole": _frac(
                lambda r: r.get("mean_act_gap", 0) > r.get("mean_act_whole", 0) + 1e-6,
                last_pre_death,
            ),
            # Specialist signature: gap high relative to whole, but whole low vs r*
            "frac_specialist_signature": _frac(
                lambda r: (
                    r.get("gap_over_whole", 0) >= 1.5
                    and r.get("mean_act_whole", 1e9) < 0.5 * r.get("r_star", 1.0)
                ),
                last_pre_death,
            ),
            # Uniformly weak: gap also near-zero / not above whole
            "frac_uniformly_weak": _frac(
                lambda r: (
                    r.get("mean_act_gap", 1) < 0.1
                    and r.get("gap_over_whole", 1) < 1.2
                ),
                last_pre_death,
            ),
        },
        "rows_birth": birth_rows[:40],
        "rows_last": last_pre_death[:40],
    }
    # Verdict string
    sig = specialist_summary["last_pre_death_or_end"].get("frac_specialist_signature")
    weak = specialist_summary["last_pre_death_or_end"].get("frac_uniformly_weak")
    if sig is not None and weak is not None:
        if sig >= 0.5 and (weak is None or weak < 0.3):
            specialist_verdict = "SPECIALIST_KILLED_BY_AVERAGE"
        elif weak is not None and weak >= 0.5:
            specialist_verdict = "UNIFORMLY_WEAK_IMPRINT"
        else:
            specialist_verdict = "MIXED_OR_INCONCLUSIVE"
    else:
        specialist_verdict = "NO_DATA"
    specialist_summary["verdict"] = specialist_verdict

    summary: dict[str, Any] = {
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "neuron_growth_trigger": "gated_novelty",
            "newborn_weight_init": cfg.newborn_weight_init,
            "threshold_kind": cfg.threshold_kind,
            "mu": cfg.mu,
            "novelty_k_min": cfg.novelty_k_min,
            "under_floor_policy": "exclude_log_monitor",
            "note": (
                "Gated novelty ∧ CE gate ∧ k_min=11; under-floor excluded+logged. "
                "parent_perturbed = clone of trigger unit (NOT gap content). "
                "gap_imprinted = residual pre-act on CE-excess samples. "
                "Specialist profile: act on birth-gap samples vs whole probe."
            ),
        },
        "specialist_profile": specialist_summary,
        "growth_activity": {
            "total_births": len(births),
            "total_established_age_ge_grace": len(established),
            "under_floor_trace": under_floor_trace,
            "mean_under_floor_frac_h3": (
                sum(
                    t["under_floor_frac_by_layer"].get("h3", 0.0)
                    for t in under_floor_trace
                )
                / max(len(under_floor_trace), 1)
            ),
            "final_under_floor_by_layer": (
                under_floor_trace[-1]["under_floor_by_layer"]
                if under_floor_trace
                else {}
            ),
            "neurons_grown_total": sum(t["neurons_grown"] for t in under_floor_trace),
            "neurons_pruned_total": sum(
                t["neurons_pruned"] for t in under_floor_trace
            ),
        },
        "dead_relu_at_birth": {
            "n": n_birth,
            "frac_dead_relu": kinds.get("dead_relu", 0) / max(n_birth, 1),
            "frac_mostly_dead": kinds.get("mostly_dead", 0) / max(n_birth, 1),
            "frac_alive_attenuated": kinds.get("alive_attenuated", 0)
            / max(n_birth, 1),
            "mean_frac_post_zero": (
                sum(r["frac_post_zero"] for r in dead_relu_birth) / max(n_birth, 1)
            ),
            "mean_post": (
                sum(r["mean_post"] for r in dead_relu_birth) / max(n_birth, 1)
            ),
            "kinds": kinds,
        },
        "redundancy_divergence": {
            "n_established": len(established),
            "mean_cos_locus_birth_all": (
                sum(cos_locus_births) / max(len(cos_locus_births), 1)
                if cos_locus_births
                else None
            ),
            "mean_cos_gap_birth_all": (
                sum(cos_gap_births) / max(len(cos_gap_births), 1)
                if cos_gap_births
                else None
            ),
            "mean_cos_locus_birth_established": (
                sum(e.get("cos_locus_birth") or 0 for e in established)
                / max(len(established), 1)
                if established
                else None
            ),
            "mean_cos_locus_final": (
                sum(cos_finals) / max(len(cos_finals), 1) if cos_finals else None
            ),
            "frac_cos_locus_final_ge_0.9": (
                sum(1 for c in cos_finals if c >= 0.9) / max(len(cos_finals), 1)
                if cos_finals
                else None
            ),
            "frac_cos_locus_decay_ge_0.1": (
                sum(
                    1
                    for e in established
                    if e.get("cos_locus_decay") is not None
                    and e["cos_locus_decay"] >= 0.1
                )
                / max(len(established), 1)
                if established
                else None
            ),
            "mean_cos_gap_birth_established": (
                sum(
                    e["cos_gap_birth"]
                    for e in established
                    if e.get("cos_gap_birth") is not None
                )
                / max(
                    sum(1 for e in established if e.get("cos_gap_birth") is not None),
                    1,
                )
                if established
                else None
            ),
            "pairs": established[:48],
        },
        "own_subset_vs_whole_prune_recheck": {
            "n_prune_candidates": len(prune_recheck),
            "mean_delta_l_whole": (
                sum(r["delta_l_whole"] for r in prune_recheck)
                / max(len(prune_recheck), 1)
                if prune_recheck
                else None
            ),
            "mean_delta_l_own": (
                sum(r["delta_l_own_subset"] for r in prune_recheck)
                / max(len(prune_recheck), 1)
                if prune_recheck
                else None
            ),
            "frac_own_gt_whole": (
                sum(1 for r in prune_recheck if r["own_gt_whole"])
                / max(len(prune_recheck), 1)
                if prune_recheck
                else None
            ),
            "frac_own_delta_ge_1e-3": (
                sum(1 for r in prune_recheck if r["delta_l_own_subset"] >= 1e-3)
                / max(len(prune_recheck), 1)
                if prune_recheck
                else None
            ),
            "rows": prune_recheck[:40],
        },
        "hidden_counts_final": net.hidden_counts(),
    }

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "redundancy_divergence"}, indent=2, default=str)[:4000])
    print("\nWrote", path)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument(
        "--weight-init",
        type=str,
        default="gap_imprinted",
        choices=["gap_imprinted", "parent_perturbed", "kaiming"],
    )
    p.add_argument(
        "--threshold",
        type=str,
        default="fixed",
        choices=["fixed", "sliding"],
        help="Variant A fixed shared r* vs Variant B sliding own-history θ",
    )
    p.add_argument("--mu", type=float, default=0.999)
    p.add_argument(
        "--out",
        type=str,
        default="",
    )
    args = p.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        tag = f"{args.weight_init}_{args.threshold}"
        out = (
            _REPO
            / "results"
            / "phase1a_diagnostics"
            / f"gated_novelty_battery_{tag}"
        )
    run_battery(
        seed=args.seed,
        epochs=args.epochs,
        out_dir=out,
        weight_init=args.weight_init,
        threshold_kind=args.threshold,
        mu=args.mu,
    )


if __name__ == "__main__":
    main()
