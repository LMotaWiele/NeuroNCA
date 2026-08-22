#!/usr/bin/env python3
"""A/B: ``neighbor_metric`` grid vs feature. Prereg: ``docs/feature_neighbors_preregistration.md``."""

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
sys.path.insert(0, str(_REPO / "scripts"))

from neuronca.data.mnist import build_mnist_loaders  # noqa: E402
from neuronca.data.tasks import TaskContext, make_identity_transform  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import (  # noqa: E402
    HIDDEN_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.rules import (  # noqa: E402
    PlasticityConfig,
    competitive_neuron_zscores,
)
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "feature_neighbors_ab"

# Pre-registered margins (§5)
REDUNDANCY_MARGIN = 0.25  # feature mean own ΔL ≤ (1 - margin) * grid
ACC_MEAN_TOL_PP = 1.0
ACC_SEED_TOL_PP = 1.5
THRASH_MULT = 1.5
SWING_SLACK_PP = 2.0
CHURN_BOUND = 0.5


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(metric: str) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        neighbor_metric=str(metric),
        neighbor_similarity="cosine",
        neighbor_top_k=None,  # auto-match mean grid |N|
        neighbor_churn_bound=CHURN_BOUND,
        neighbor_signature_ema_mu=0.0,
        target_rates={"h1": 0.77, "h2": 1.02, "h3": 1.28},
        tau=0.5,
        threshold_kind="fixed",
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


@torch.no_grad()
def _ablate_own_subset_delta_l(
    model: DynamicMLPReasoningNet,
    layer: str,
    slot: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    """Own-subset + whole-set ablation ΔL⁺ on probe (preferred class)."""
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    if y.numel() == 0:
        return {"delta_l_whole": 0.0, "delta_l_own_subset": 0.0, "preferred_class": -1}

    model.eval()
    # Preferred class = argmax mean activation on probe
    acts = model.last_activations
    if acts is None:
        model(x)
        acts = model.last_activations
    assert acts is not None
    key = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}[layer]
    a = acts[key][:, slot].detach().float().abs()
    # Recompute acts if shape mismatch (batch size)
    if a.numel() != y.numel():
        model(x)
        acts = model.last_activations
        assert acts is not None
        a = acts[key][:, slot].detach().float().abs().cpu()
        y_cpu = y.cpu()
    else:
        a = a.cpu()
        y_cpu = y.cpu()

    class_means = []
    for c in range(int(y_cpu.max().item()) + 1 if y_cpu.numel() else 10):
        m = y_cpu == c
        class_means.append(float(a[m].mean().item()) if bool(m.any()) else -1e9)
    if not class_means:
        pref = 0
    else:
        pref = int(max(range(len(class_means)), key=lambda c: class_means[c]))
    own_mask = y_cpu == pref

    def _delta(sample_mask: torch.Tensor | None) -> float:
        xs, ys = x, y
        if sample_mask is not None:
            m = sample_mask.to(device)
            if m.numel() != ys.numel():
                m = m.cpu()
                ys_c = ys.cpu()
                xs_c = xs.cpu()
                xs, ys = xs_c[m].to(device), ys_c[m].to(device)
            else:
                xs, ys = xs[m], ys[m]
        if ys.numel() == 0:
            return 0.0
        snaps: dict[str, dict[str, torch.Tensor | None]] = {}
        for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
            if post != layer and pre != layer:
                continue
            mod = model.linear_module(lin)
            snaps[lin] = {
                "w": mod.weight.detach().clone(),
                "b": mod.bias.detach().clone() if mod.bias is not None else None,
            }
        logits = model(xs)
        l_base = float(F.cross_entropy(logits, ys).item())
        for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
            mod = model.linear_module(lin)
            if post == layer:
                mod.weight.data[slot] = 0
                if mod.bias is not None:
                    mod.bias.data[slot] = 0
            if pre == layer:
                mod.weight.data[:, slot] = 0
        logits_a = model(xs)
        l_abl = float(F.cross_entropy(logits_a, ys).item())
        for lin, snap in snaps.items():
            mod = model.linear_module(lin)
            mod.weight.data.copy_(snap["w"])  # type: ignore[arg-type]
            if snap["b"] is not None and mod.bias is not None:
                mod.bias.data.copy_(snap["b"])
        return max(0.0, l_abl - l_base)

    return {
        "delta_l_whole": _delta(None),
        "delta_l_own_subset": _delta(own_mask),
        "preferred_class": float(pref),
    }


def _max_consecutive_swing(accs: list[float]) -> float:
    if len(accs) < 2:
        return 0.0
    return max(abs(accs[i] - accs[i - 1]) for i in range(1, len(accs)))


def run_one(
    *,
    seed: int,
    metric: str,
    epochs: int,
    out_dir: Path,
    smoke: bool = False,
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
        probe_size=256 if smoke else 1024,
    )
    net = DynamicMLPReasoningNet(
        hidden_init=(256, 128, 64),
        hidden_max=(512, 256, 128),
        num_classes=10,
        coord_seed=seed,
    ).to(device)
    cfg = _cfg(metric)
    if smoke:
        cfg.max_new_neurons_per_layer = 2
    orch = PlasticityOrchestrator(config=cfg)
    opt = build_optimizer(net, name="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
    signal = ActivationMagnitude(decay=0.99)

    probe_loader = loaders["probe_loader"]
    val_hist: list[dict[str, float]] = []
    structure_trace: list[dict[str, Any]] = []
    churn_trace: list[dict[str, Any]] = []
    prune_ablations: list[dict[str, Any]] = []
    global_step = 0

    print(f"\n=== seed={seed} metric={metric} epochs={epochs} ===", flush=True)

    for epoch in range(epochs):
        metrics = train_epoch_with_signals(
            net,
            loaders["train_loader"],
            opt,
            signal,
            orch,
            device=device,
            global_step=global_step,
            max_steps=8 if smoke else None,
        )
        global_step = int(metrics["global_step"])

        # Probe + neighbor refresh + gated novelty (uses orch neighbor graph)
        gated = collect_gated_novelty_probe(
            net,
            probe_loader,
            device=device,
            max_batches=2 if smoke else None,
            radius=float(cfg.radius),
            k_min=int(cfg.novelty_k_min),
            orch=orch,
        )
        # Rebuild probe x,y for ablation (preferred-class own-subset ΔL)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        net.eval()
        n_b = 0
        for batch in probe_loader:
            if smoke and n_b >= 2:
                break
            x, y = batch[0], batch[1]
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            xs.append(x)
            ys.append(y.view(-1) if y.ndim > 1 else y)
            n_b += 1
        x_probe = torch.cat(xs, dim=0)
        y_probe = torch.cat(ys, dim=0)

        # Pre-step: measure own-subset ΔL of units that will be pruned
        scores = signal.neuron_scores()
        signals = orch._signals(net, scores)
        policy = orch.ensure_policy(net)
        active = {h: net.active_mask(h).cpu() for h in HIDDEN_LAYERS}
        policy.update(
            {"h1": signals["h1"], "h2": signals["h2"], "h3": signals["h3"]},
            active,
        )
        # Ensure probe forward for preferred-class acts
        with torch.no_grad():
            net(x_probe.to(device)[: min(256, x_probe.size(0))])

        # Match orchestrator.step: begin_plasticity_step() increments the step
        # *before* prune selection, so grace uses step+1 relative to now.
        step_after_begin = int(net.cooldown.plasticity_step) + 1
        grace = int(cfg.newborn_grace_steps)
        z_cut = float(cfg.prune_neuron_z)
        for h in HIDDEN_LAYERS:
            nmask = orch.neighbors.get_mask(h)
            z = competitive_neuron_zscores(net, h, signals, neighbor_mask=nmask)
            a = net.active_mask(h).cpu()
            contr = policy.contraction_pressure(h).cpu()
            birth = net.birth_step(h).cpu()
            age_ok = (birth >= 0) & ((step_after_begin - birth) >= grace)
            cand = a & contr & (z < z_cut) & age_ok
            to_prune = cand.nonzero(as_tuple=False).view(-1).tolist()
            for slot in to_prune:
                ab = _ablate_own_subset_delta_l(
                    net,
                    h,
                    int(slot),
                    x_probe[: min(1024, x_probe.size(0))],
                    y_probe[: min(1024, y_probe.size(0))],
                )
                prune_ablations.append(
                    {
                        "epoch": epoch,
                        "layer": h,
                        "slot": int(slot),
                        **ab,
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
        sd = slog.as_dict()
        structure_trace.append(sd)
        if orch.neighbor_refresh_log:
            churn_trace.append(
                {"epoch": epoch, **orch.neighbor_refresh_log[-1]}
            )

        vm = evaluate(net, loaders["test_loader"], device=device)
        val_hist.append({"epoch": float(epoch), **vm})
        print(
            f"  ep{epoch:02d}: grown={slog.neurons_grown} pruned={slog.neurons_pruned} "
            f"acc={vm['acc']:.4f} churn={sd.get('neighbor_mean_churn', 0):.3f} "
            f"unstable={sd.get('neighbor_unstable', False)}",
            flush=True,
        )

    own_dls = [r["delta_l_own_subset"] for r in prune_ablations]
    whole_dls = [r["delta_l_whole"] for r in prune_ablations]
    accs = [float(v["acc"]) for v in val_hist]
    thrash = sum(
        int(s.get("neurons_grown", 0)) + int(s.get("neurons_pruned", 0))
        for s in structure_trace
    )
    # Post-first-refresh churn (ignore epoch 0 empty prev)
    churns = [c["mean_churn"] for c in churn_trace if c.get("mean_churn") is not None]
    churns_post0 = churns[1:] if len(churns) > 1 else churns
    mean_churn = float(sum(churns_post0) / len(churns_post0)) if churns_post0 else 0.0
    max_churn = float(max(churns_post0)) if churns_post0 else 0.0
    unstable_any = any(bool(c.get("unstable")) for c in churn_trace)
    # After stability response, final epochs should be bounded
    late_churn = churns_post0[-3:] if len(churns_post0) >= 3 else churns_post0
    late_mean_churn = float(sum(late_churn) / len(late_churn)) if late_churn else 0.0
    churn_bounded = late_mean_churn <= CHURN_BOUND

    result = {
        "seed": seed,
        "metric": metric,
        "epochs": epochs,
        "n_pruned_measured": len(prune_ablations),
        "mean_delta_l_own_subset": (
            float(sum(own_dls) / len(own_dls)) if own_dls else float("nan")
        ),
        "mean_delta_l_whole": (
            float(sum(whole_dls) / len(whole_dls)) if whole_dls else float("nan")
        ),
        "final_acc": accs[-1] if accs else float("nan"),
        "acc_history": accs,
        "max_consecutive_swing": _max_consecutive_swing(accs),
        "thrash_total": thrash,
        "mean_neighbor_churn": mean_churn,
        "max_neighbor_churn": max_churn,
        "late_mean_neighbor_churn": late_mean_churn,
        "neighbor_unstable_any": unstable_any,
        "churn_bounded": churn_bounded,
        "matched_k_last": (
            churn_trace[-1].get("matched_k_by_layer") if churn_trace else {}
        ),
        "hidden_counts_final": net.hidden_counts(),
        "prune_ablations": prune_ablations,
        "churn_trace": churn_trace,
        "structure_trace": structure_trace,
        "val_history": val_hist,
        "similarity": "cosine",
        "refresh_n": 1,
        "churn_bound": CHURN_BOUND,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{seed}.json"
    # Slim write: drop full structure if huge — keep summary fields + ablations
    slim = {k: v for k, v in result.items() if k not in ("structure_trace",)}
    slim["structure_summary"] = [
        {
            "epoch": s.get("epoch"),
            "neurons_grown": s.get("neurons_grown"),
            "neurons_pruned": s.get("neurons_pruned"),
            "neighbor_mean_churn": s.get("neighbor_mean_churn"),
            "neighbor_matched_k_by_layer": s.get("neighbor_matched_k_by_layer"),
            "neighbor_unstable": s.get("neighbor_unstable"),
            "val_acc": val_hist[i]["acc"] if i < len(val_hist) else None,
        }
        for i, s in enumerate(structure_trace)
    ]
    path.write_text(json.dumps(slim, indent=2) + "\n")
    print(
        f"  done: final_acc={result['final_acc']:.4f} "
        f"mean_own_dL={result['mean_delta_l_own_subset']:.5f} "
        f"n_prune={result['n_pruned_measured']} thrash={thrash} "
        f"churn_bounded={churn_bounded}",
        flush=True,
    )
    return result


def _seed_primary(grid: dict[str, Any], feat: dict[str, Any]) -> str:
    g = grid["mean_delta_l_own_subset"]
    f = feat["mean_delta_l_own_subset"]
    if g != g or f != f:  # NaN
        return "INCONCLUSIVE"
    if g <= 1e-12 and f <= 1e-12:
        return "TIE_BOTH_NEAR_ZERO"
    # Feature better if lower own-subset impact
    if f <= g * (1.0 - REDUNDANCY_MARGIN):
        return "FEATURE_LOWER"
    if f >= g * (1.0 + REDUNDANCY_MARGIN):
        return "FEATURE_HIGHER"
    if f < g:
        return "FEATURE_DIRECTIONALLY_LOWER"
    if f > g:
        return "FEATURE_DIRECTIONALLY_HIGHER"
    return "TIE"


def verdict(by_arm: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    grid_seeds = by_arm.get("grid", {})
    feat_seeds = by_arm.get("feature", {})
    common = sorted(set(grid_seeds) & set(feat_seeds))
    if not common:
        return {"label": "INCONCLUSIVE", "reason": "no paired seeds"}

    primary_labels = []
    rows = []
    for s in common:
        g, f = grid_seeds[s], feat_seeds[s]
        pl = _seed_primary(g, f)
        primary_labels.append(pl)
        rows.append(
            {
                "seed": s,
                "primary": pl,
                "own_dL_grid": g["mean_delta_l_own_subset"],
                "own_dL_feature": f["mean_delta_l_own_subset"],
                "acc_grid": g["final_acc"],
                "acc_feature": f["final_acc"],
                "thrash_grid": g["thrash_total"],
                "thrash_feature": f["thrash_total"],
                "swing_grid": g["max_consecutive_swing"],
                "swing_feature": f["max_consecutive_swing"],
                "churn_bounded_feature": f["churn_bounded"],
                "mean_churn_feature": f["mean_neighbor_churn"],
            }
        )

    n = len(common)
    n_lower = sum(1 for p in primary_labels if p == "FEATURE_LOWER")
    n_higher = sum(1 for p in primary_labels if p == "FEATURE_HIGHER")
    n_dir_lower = sum(
        1
        for p in primary_labels
        if p in ("FEATURE_LOWER", "FEATURE_DIRECTIONALLY_LOWER")
    )
    n_dir_higher = sum(
        1
        for p in primary_labels
        if p in ("FEATURE_HIGHER", "FEATURE_DIRECTIONALLY_HIGHER")
    )

    primary_win = n_lower == n and n >= 1
    primary_loss = n_higher == n and n >= 1
    # Also allow directional-only if all 3 directionally lower but margin miss → not win
    primary_tie = not primary_win and not primary_loss

    # Secondary
    acc_g = [grid_seeds[s]["final_acc"] for s in common]
    acc_f = [feat_seeds[s]["final_acc"] for s in common]
    mean_g = sum(acc_g) / len(acc_g)
    mean_f = sum(acc_f) / len(acc_f)
    mean_gap_pp = (mean_f - mean_g) * 100.0
    seed_gaps_pp = [(acc_f[i] - acc_g[i]) * 100.0 for i in range(n)]
    acc_ok = abs(mean_gap_pp) <= ACC_MEAN_TOL_PP and all(
        gap >= -ACC_SEED_TOL_PP for gap in seed_gaps_pp
    )

    thrash_g = sum(grid_seeds[s]["thrash_total"] for s in common) / n
    thrash_f = sum(feat_seeds[s]["thrash_total"] for s in common) / n
    thrash_ok = thrash_f <= thrash_g * THRASH_MULT + 1e-9

    swing_g = sum(grid_seeds[s]["max_consecutive_swing"] for s in common) / n
    swing_f = sum(feat_seeds[s]["max_consecutive_swing"] for s in common) / n
    swing_ok = swing_f <= swing_g + SWING_SLACK_PP / 100.0 + 1e-9

    secondary_ok = acc_ok and thrash_ok and swing_ok

    churn_ok = all(feat_seeds[s]["churn_bounded"] for s in common)

    if not churn_ok:
        label = "FEATURE_NEIGHBORS_WORSE"
        reason = "NEIGHBORHOOD_UNSTABLE: late churn not bounded after stability response"
    elif primary_loss or not secondary_ok:
        label = "FEATURE_NEIGHBORS_WORSE"
        reason = (
            "primary_loss"
            if primary_loss
            else f"secondary_regression acc_ok={acc_ok} thrash_ok={thrash_ok} swing_ok={swing_ok}"
        )
    elif primary_win and secondary_ok:
        label = "FEATURE_NEIGHBORS_BETTER"
        reason = "primary ≥25% lower own-subset ΔL 3/3 + no regression"
    else:
        label = "FEATURE_NEIGHBORS_NEUTRAL"
        reason = (
            f"primary tie/partial (n_lower={n_lower} n_dir_lower={n_dir_lower}) "
            f"secondary_ok={secondary_ok}"
        )

    return {
        "label": label,
        "reason": reason,
        "primary_win": primary_win,
        "primary_loss": primary_loss,
        "primary_tie": primary_tie,
        "n_seeds": n,
        "n_feature_lower_margin": n_lower,
        "n_feature_higher_margin": n_higher,
        "n_feature_dir_lower": n_dir_lower,
        "n_feature_dir_higher": n_dir_higher,
        "secondary_ok": secondary_ok,
        "acc_ok": acc_ok,
        "thrash_ok": thrash_ok,
        "swing_ok": swing_ok,
        "churn_ok": churn_ok,
        "mean_acc_grid": mean_g,
        "mean_acc_feature": mean_f,
        "mean_acc_gap_pp": mean_gap_pp,
        "mean_thrash_grid": thrash_g,
        "mean_thrash_feature": thrash_f,
        "mean_swing_grid": swing_g,
        "mean_swing_feature": swing_f,
        "by_seed": rows,
        "proceed_to_experiment_2": label
        in ("FEATURE_NEIGHBORS_BETTER", "FEATURE_NEIGHBORS_NEUTRAL"),
    }


def write_results_md(summary: dict[str, Any], path: Path) -> None:
    v = summary["verdict"]
    lines = [
        "# Feature-space neighborhoods — Experiment 1 results",
        "",
        f"**Prereg:** `docs/feature_neighbors_preregistration.md`",
        f"**Headline: `{v['label']}`**",
        f"**Proceed to Experiment 2?** **{'YES' if v.get('proceed_to_experiment_2') else 'NO'}**",
        "",
        v.get("reason", ""),
        "",
        "## Primary: pruned-unit own-subset ablation ΔL",
        "",
        "| seed | own ΔL grid | own ΔL feature | primary | acc grid | acc feature | thrash G | thrash F | churn ok |",
        "|-----:|------------:|---------------:|---------|---------:|------------:|---------:|---------:|:--------:|",
    ]
    for r in v.get("by_seed", []):
        lines.append(
            f"| {r['seed']} | {r['own_dL_grid']:.5f} | {r['own_dL_feature']:.5f} | "
            f"{r['primary']} | {r['acc_grid']:.4f} | {r['acc_feature']:.4f} | "
            f"{r['thrash_grid']} | {r['thrash_feature']} | {r['churn_bounded_feature']} |"
        )
    lines += [
        "",
        "## Secondary / stability",
        "",
        f"- mean acc grid={v.get('mean_acc_grid'):.4f} feature={v.get('mean_acc_feature'):.4f} "
        f"gap={v.get('mean_acc_gap_pp'):.2f} pp (ok={v.get('acc_ok')})",
        f"- thrash grid={v.get('mean_thrash_grid'):.1f} feature={v.get('mean_thrash_feature'):.1f} "
        f"(ok={v.get('thrash_ok')})",
        f"- swing grid={v.get('mean_swing_grid'):.4f} feature={v.get('mean_swing_feature'):.4f} "
        f"(ok={v.get('swing_ok')})",
        f"- feature neighbor-churn bounded={v.get('churn_ok')}",
        "",
        "## Locked protocol",
        "",
        "- Signature: probe activation **cosine**",
        "- Matched k: round(mean grid |N| under R) per layer per refresh",
        "- Refresh N=1 epoch; churn bound=0.5; stability response=EMA μ=0.5",
        "- Redundancy margin: feature own-subset ΔL ≥25% lower, 3/3 directional",
        "",
        f"Artifacts under `{path.parent}`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--arms", type=str, default="grid,feature")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    epochs = 3 if args.smoke else int(args.epochs)

    OUT.mkdir(parents=True, exist_ok=True)
    prereg_note = {
        "prereg": "docs/feature_neighbors_preregistration.md",
        "redundancy_margin": REDUNDANCY_MARGIN,
        "acc_mean_tol_pp": ACC_MEAN_TOL_PP,
        "churn_bound": CHURN_BOUND,
        "epochs": epochs,
        "seeds": seeds,
        "arms": arms,
        "smoke": bool(args.smoke),
    }
    (OUT / "prereg.json").write_text(json.dumps(prereg_note, indent=2) + "\n")

    by_arm: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in arms}
    for metric in arms:
        arm_dir = OUT / metric
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            r = run_one(
                seed=seed,
                metric=metric,
                epochs=epochs,
                out_dir=arm_dir,
                smoke=bool(args.smoke),
            )
            by_arm[metric][seed] = r

    v = verdict(by_arm)
    summary = {
        "prereg": prereg_note,
        "verdict": v,
        "by_arm_seed": {
            m: {
                str(s): {
                    k: r[k]
                    for k in (
                        "final_acc",
                        "mean_delta_l_own_subset",
                        "mean_delta_l_whole",
                        "n_pruned_measured",
                        "thrash_total",
                        "max_consecutive_swing",
                        "mean_neighbor_churn",
                        "late_mean_neighbor_churn",
                        "churn_bounded",
                        "neighbor_unstable_any",
                        "matched_k_last",
                    )
                    if k in r
                }
                for s, r in seeds_map.items()
            }
            for m, seeds_map in by_arm.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_results_md(summary, OUT / "RESULTS.md")
    print("\n======== VERDICT ========", flush=True)
    print(f"  {v['label']}", flush=True)
    print(f"  {v['reason']}", flush=True)
    print(f"  proceed_to_experiment_2={v.get('proceed_to_experiment_2')}", flush=True)
    print(f"Wrote {OUT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
