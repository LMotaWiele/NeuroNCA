#!/usr/bin/env python3
"""Demo: record and render one seed of the product stack (does not touch primaries).

Frames are post-train + post-orchestrator; ``epoch=0`` is after the first
plasticity step, not init topology.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.plasticity.structure_snapshot import (  # noqa: E402
    StructureSnapshotRecorder,
)
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)
from visualize_structure import render_animation, render_static  # noqa: E402

OUT = _REPO / "results" / "phase1a_diagnostics" / "visualizations" / "example_seed0"
# 10 ep so grace (3) prunes actually fire and death markers get exercised.
EPOCHS = 10
SEED = 0
# Short train steps for wall-clock only — does NOT inflate growth caps.
MAX_STEPS = 40
# Production defaults (same as phase1a_fixed / survival battery).
MAX_NEW_NEURONS_PER_LAYER = 4


def _cfg() -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates={"h1": 0.77, "h2": 1.02, "h3": 1.28},
        tau=0.5,
        threshold_kind="fixed",
        mu=0.999,
        prune_edge_fraction=0.05,
        prune_neuron_z=-1.0,
        max_new_edges_per_layer=64,
        max_new_neurons_per_layer=MAX_NEW_NEURONS_PER_LAYER,
        seed_edges_per_new_neuron=4,
        edge_eligibility_mode="neither_contraction",
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


def _profile_gap(
    posts: torch.Tensor, gap_mask: torch.Tensor
) -> dict[str, float]:
    a = posts.detach().float().abs().reshape(-1)
    m = gap_mask.bool().reshape(-1)
    n = min(a.numel(), m.numel())
    a, m = a[:n], m[:n]
    whole = float(a.mean().item()) + 1e-8
    pg = float(a[m].mean().item()) if m.any() else float("nan")
    fg = float((a[m] > 0).float().mean().item()) if m.any() else float("nan")
    fn = float((a[~m] > 0).float().mean().item()) if (~m).any() else float("nan")
    return {
        "gap_over_whole": pg / whole if pg == pg else float("nan"),
        "on_off": (fg - fn) if fg == fg and fn == fn else float("nan"),
    }


@torch.no_grad()
def _probe_posts(net, loader, device) -> dict[str, torch.Tensor]:
    act_key = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}
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
            bufs[h].append(acts[act_key[h]].detach().cpu())
    return {h: torch.cat(bufs[h], dim=0) for h in HIDDEN_LAYERS}


def main() -> int:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    task = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        task,
        root=str(_REPO / "data" / "mnist"),
        batch_size=128,
        num_workers=0,
        seed=SEED,
        probe_size=256,
    )
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=SEED,
    ).to(device)
    cfg = _cfg()
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, name="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)
    rec = StructureSnapshotRecorder.from_config(
        OUT,
        total_epochs=EPOCHS,
        record=True,
        cadence="every_epoch",
    )
    assert rec is not None

    live_gap: dict[tuple[str, int], torch.Tensor] = {}
    global_step = 0
    structure_trace: list[dict[str, int]] = []
    for epoch in range(EPOCHS):
        metrics = train_epoch_with_signals(
            net,
            loaders["train_loader"],
            opt,
            signal,
            orch,
            device=device,
            max_steps=MAX_STEPS,
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
                "neurons_grown": int(slog.neurons_grown),
                "neurons_pruned": int(slog.neurons_pruned),
                "n_prune_records": len(slog.neuron_prune_records),
            }
        )
        for layer, slot in list(live_gap.keys()):
            if not bool(net.active_mask(layer)[slot]):
                del live_gap[(layer, slot)]
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
            if trigger is not None and (layer, trigger) in gated.gap_masks:
                live_gap[(layer, int(slot))] = (
                    gated.gap_masks[(layer, trigger)].bool().cpu().clone()
                )

        if live_gap:
            posts = _probe_posts(net, loaders["probe_loader"], device)
            for (layer, slot), gmask in live_gap.items():
                if not bool(net.active_mask(layer)[slot]):
                    continue
                prof = _profile_gap(posts[layer][:, slot], gmask)
                rec.set_specialization(layer, slot, prof)

        path = rec.maybe_record(epoch, net, signal=signal, structure_log=slog)
        print(
            f"epoch {epoch}: grown={slog.neurons_grown} pruned={slog.neurons_pruned} "
            f"snapshot={path.name if path else None} live_gap={len(live_gap)} "
            f"prune_recs={len(slog.neuron_prune_records)}"
        )

    total_grown = sum(r["neurons_grown"] for r in structure_trace)
    total_pruned = sum(r["neurons_pruned"] for r in structure_trace)
    epochs_with_prune = sum(1 for r in structure_trace if r["neurons_pruned"] > 0)

    summary = {
        "seed": SEED,
        "epochs": EPOCHS,
        "max_steps_per_epoch": MAX_STEPS,
        "stack": "gated_novelty+gap_imprinted",
        "growth_caps": {
            "max_new_neurons_per_layer": MAX_NEW_NEURONS_PER_LAYER,
            "max_births_per_epoch_all_layers": MAX_NEW_NEURONS_PER_LAYER * 3,
            "inflated_for_demo": False,
            "note": (
                "Production caps (same as phase1a_fixed / survival battery). "
                "MAX_STEPS shortens train only; does not change structure caps."
            ),
        },
        "snapshot_semantics": {
            "epoch_index": (
                "Snapshot at epoch k is state AFTER epoch-k training and the "
                "orchestrator step for that epoch. epoch=0 is post first plasticity "
                "step, not init-only topology."
            ),
            "cadence": "every_epoch",
        },
        "structure_trace": structure_trace,
        "total_neurons_grown": total_grown,
        "total_neurons_pruned": total_pruned,
        "epochs_with_prune": epochs_with_prune,
        "hidden_counts_final": net.hidden_counts(),
        "edge_counts_final": net.edge_counts(by_origin=True),
        "n_snapshots": len(rec.written),
        "n_live_gap_tracked_end": len(live_gap),
        "snapshot_files": [str(p) for p in rec.written],
        "edge_render_policy": (
            "max_init_edges=500 per subplot; gap_imprinted_seeded and "
            "incentive_grown always drawn in full (never subsampled)."
        ),
        "resolution": {
            "static_dpi": 350,
            "anim_dpi": 250,
            "note": "2.5× prior defaults (140 / 100)",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if total_pruned == 0:
        print(
            "WARNING: zero prunes over the run — death-marker path still unexercised"
        )

    static_path = OUT / "structure_static.png"
    anim_path = OUT / "structure_animation.gif"
    # Prefer a late frame that has prune markers if any epoch pruned.
    prune_epochs = [r["epoch"] for r in structure_trace if r["neurons_pruned"] > 0]
    static_tag = f"epoch_{prune_epochs[-1]}" if prune_epochs else f"epoch_{EPOCHS - 1}"
    render_static(OUT, static_path, tag=static_tag)
    render_animation(OUT, anim_path, fps=1.2)
    print(f"static: {static_path} (tag={static_tag})")
    print(f"animation: {anim_path}")
    print(
        f"validation: total_grown={total_grown} total_pruned={total_pruned} "
        f"epochs_with_prune={epochs_with_prune}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
