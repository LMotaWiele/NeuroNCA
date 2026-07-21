"""Phase 1a structure diagnostics: sparsity, length hist, stability helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuronca.models.dynamic_mlp import (
    LINEAR_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.coords import pairwise_distances
from neuronca.plasticity.masks import (
    EDGE_ORIGIN_GAP_IMPRINT,
    EDGE_ORIGIN_INCENTIVE,
    EDGE_ORIGIN_INIT,
)
from neuronca.plasticity.orchestrator import PlasticityOrchestrator, StructureEventLog


def connection_lengths(
    net: DynamicMLPReasoningNet,
    *,
    origin: int | None = None,
) -> Tensor:
    """Euclidean lengths of active edges; optionally filter by origin code."""
    lengths: list[Tensor] = []
    for layer, (pre, post) in _LINEAR_ENDPOINTS.items():
        em = net.edge_mask(layer).cpu()
        eo = net.edge_origin(layer).cpu()
        if origin is not None:
            em = em & (eo == origin)
        if not bool(em.any()):
            continue
        pre_c = net.coords(pre).cpu().float()
        post_c = net.coords(post).cpu().float()
        dist = pairwise_distances(post_c, pre_c)
        lengths.append(dist[em])
    if not lengths:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(lengths, dim=0)


def length_histogram(
    net: DynamicMLPReasoningNet,
    bins: int = 20,
    max_r: float | None = None,
) -> dict[str, Any]:
    """Histograms for all / init_dense / incentive_grown edge lengths."""
    out: dict[str, Any] = {}
    for key, origin in (
        ("all", None),
        ("init_dense", EDGE_ORIGIN_INIT),
        ("incentive_grown", EDGE_ORIGIN_INCENTIVE),
        ("gap_imprinted_seeded", EDGE_ORIGIN_GAP_IMPRINT),
    ):
        lens = connection_lengths(net, origin=origin)
        if max_r is None:
            hi = float(lens.max().item()) if lens.numel() else 1.0
            hi = max(hi, 1e-6)
        else:
            hi = float(max_r)
        if lens.numel() == 0:
            hist = torch.zeros(bins)
            edges = torch.linspace(0, hi, bins + 1)
        else:
            hist = torch.histc(lens, bins=bins, min=0.0, max=hi)
            edges = torch.linspace(0, hi, bins + 1)
        out[key] = {
            "counts": hist.tolist(),
            "bin_edges": edges.tolist(),
            "n": int(lens.numel()),
            "mean": float(lens.mean().item()) if lens.numel() else float("nan"),
            "max": float(lens.max().item()) if lens.numel() else float("nan"),
        }
    return out


def sparsity_report(net: DynamicMLPReasoningNet) -> dict[str, Any]:
    """Connection density vs fully-connected active baseline, overall and by origin."""
    report: dict[str, Any] = {}
    for layer, (pre, post) in _LINEAR_ENDPOINTS.items():
        n_pre = int(net.population_active(pre).sum())
        n_post = int(net.population_active(post).sum())
        dense = max(n_pre * n_post, 1)
        em = net.edge_mask(layer)
        eo = net.edge_origin(layer)
        pre_a = net.population_active(pre)
        post_a = net.population_active(post)
        used = em & post_a.unsqueeze(1) & pre_a.unsqueeze(0)
        n_init = int(((eo == EDGE_ORIGIN_INIT) & used).sum())
        n_grown = int(((eo == EDGE_ORIGIN_INCENTIVE) & used).sum())
        n_gap = int(((eo == EDGE_ORIGIN_GAP_IMPRINT) & used).sum())
        n_tot = int(used.sum())
        report[layer] = {
            "n_pre_active": n_pre,
            "n_post_active": n_post,
            "dense_baseline": dense,
            "n_edges": n_tot,
            "density": n_tot / dense,
            "n_init_dense": n_init,
            "n_incentive_grown": n_grown,
            "n_gap_imprinted_seeded": n_gap,
            "density_init_among_active": n_init / dense,
            "density_grown_among_active": n_grown / dense,
        }
    return report


def prune_origin_summary(orch: PlasticityOrchestrator) -> dict[str, Any]:
    """Cumulative prune rates by origin for results write-up."""
    return {
        "cumulative_pruned_init_dense": orch.cumulative_pruned_init,
        "cumulative_pruned_incentive_grown": orch.cumulative_pruned_incentive,
        "cumulative_grown_edges": orch.cumulative_grown_edges,
    }


def thrash_events(
    history: list[StructureEventLog],
    *,
    window: int = 3,
) -> dict[str, Any]:
    """Count grow↔prune flip-flops on the same key within ``window`` consecutive epochs.

    Pre-registered bar: annotate if neuron or edge thrash events exceed N=10
    (docs/phase1a_preregistration.md stability bar item 4).

    A thrash event is recorded when, for a given key, the ordered sequence of
    actions over a sliding window of epochs contains both a grow and a prune
    (in either order) — i.e. the same identity was both added and removed within
    the window.
    """
    if not history:
        return {
            "neuron_thrash_events": 0,
            "edge_thrash_events": 0,
            "thrash_events_total": 0,
            "exceeds_bar_n10": False,
        }

    # Per epoch: sets of prune/grow keys
    neuron_events: list[dict[str, set]] = []
    edge_events: list[dict[str, set]] = []
    for ev in history:
        neuron_events.append(
            {
                "prune": set(ev.neuron_prune_keys),
                "grow": set(ev.neuron_grow_keys),
            }
        )
        edge_events.append(
            {
                "prune": set(ev.edge_prune_keys),
                "grow": set(ev.edge_grow_keys),
            }
        )

    def _count_thrash(seq: list[dict[str, set]]) -> int:
        n = 0
        # For each key that ever appears, scan timeline for flip within window.
        keys: set = set()
        for s in seq:
            keys |= s["prune"] | s["grow"]
        for key in keys:
            # Timeline of actions: list of (epoch_idx, 'p'|'g')
            actions: list[tuple[int, str]] = []
            for i, s in enumerate(seq):
                if key in s["prune"]:
                    actions.append((i, "p"))
                if key in s["grow"]:
                    actions.append((i, "g"))
            # Sliding pairs of opposite actions within window epochs
            for a in range(len(actions)):
                for b in range(a + 1, len(actions)):
                    ei, ti = actions[a]
                    ej, tj = actions[b]
                    if ej - ei >= window:
                        break
                    if ti != tj:
                        n += 1
                        # Count each flip once per ordered pair of opposite events
        return n

    n_thrash = _count_thrash(neuron_events)
    e_thrash = _count_thrash(edge_events)
    total = n_thrash + e_thrash
    return {
        "neuron_thrash_events": n_thrash,
        "edge_thrash_events": e_thrash,
        "thrash_events_total": total,
        "exceeds_bar_n10": total > 10,
        "window_epochs": window,
    }


def stability_check(
    history: list[StructureEventLog],
    *,
    max_hidden: int,
    max_cap_hit_epochs: int = 2,
    init_hidden_sum: int = 256 + 128 + 64,
    growth_factor: float = 1.5,
    thrash_bar: int = 10,
) -> dict[str, Any]:
    """Evaluate pre-registered stability bars on a run history."""
    if not history:
        return {"stable": False, "reason": "empty_history"}
    final = history[-1]
    total_h = sum(final.hidden_counts.values())
    cap_hit_epochs = sum(1 for ev in history if any(ev.cap_hits.values()))
    same_step = sum(ev.same_step_free_reuse_blocked for ev in history)
    bound_limit = int(growth_factor * init_hidden_sum)
    bounded = total_h <= max(max_hidden, bound_limit)
    thrash = thrash_events(history)
    # same_step blocks are expected when guard fires; nonzero is OK (blocked).
    stable = (
        bounded
        and cap_hit_epochs <= max_cap_hit_epochs
        and not thrash["exceeds_bar_n10"]
    )
    return {
        "stable": stable,
        "final_hidden_total": total_h,
        "cap_hit_epochs": cap_hit_epochs,
        "cooldown_blocks_total": same_step,
        "bounded_vs_1_5x_init": total_h <= bound_limit,
        "bound_limit": bound_limit,
        "thrash": thrash,
        "passes_cap_bar": cap_hit_epochs <= max_cap_hit_epochs,
        "passes_thrash_bar": not thrash["exceeds_bar_n10"],
        "passes_bound_bar": total_h <= bound_limit,
    }


def history_to_rows(history: list[StructureEventLog]) -> list[dict[str, Any]]:
    return [ev.as_dict() for ev in history]
