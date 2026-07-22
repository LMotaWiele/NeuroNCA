#!/usr/bin/env python3
"""Adaptation-cost runner (digit-split primary; frozen-continue baseline).

Pre-registration: docs/adaptation_cost_preregistration.md
                  results/phase1a_diagnostics/adaptation_cost/prereg.json

Primary claim setup:
  - Shift: digit-split (B2); B1 optional secondary
  - Baseline: frozen_continue (weights keep training; no reset)
  - Cost: wall-clock to t_star_sustained (epochs secondary)
  - Recovery: t_star_sustained on B val acc (not first-crossing)
  - Trough: min post-switch B acc

Usage:
  python scripts/run_adaptation_cost.py --smoke
  python scripts/run_adaptation_cost.py --seeds 0 --epochs-a 10 --epochs-b 15
  python scripts/run_adaptation_cost.py --shift digit_split --arms plastic,fixed_continue
  python scripts/run_adaptation_cost.py --shift b1 --arms plastic,fixed_continue  # secondary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.data.mnist import (  # noqa: E402
    TaskAwareDataset,
    _mnist_targets,
    build_mnist_datasets,
)
from neuronca.data.tasks import (  # noqa: E402
    TaskContext,
    indices_in_digit_range,
    make_digit_split_transform,
    make_identity_transform,
    make_label_permutation_transform,
)
from neuronca.eval.time_lag import first_crossing, first_sustained_crossing  # noqa: E402
from neuronca.io.seed import set_seed  # noqa: E402
from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet  # noqa: E402
from neuronca.models.mlp import MLPReasoningNet  # noqa: E402
from neuronca.plasticity.orchestrator import PlasticityOrchestrator  # noqa: E402
from neuronca.plasticity.probe_rates import probe_target_rates  # noqa: E402
from neuronca.plasticity.rules import PlasticityConfig  # noqa: E402
from neuronca.signals.activation import ActivationMagnitude  # noqa: E402
from neuronca.training.loop import evaluate, train_epoch  # noqa: E402
from neuronca.training.optim import build_optimizer  # noqa: E402
from neuronca.training.plastic_loop import (  # noqa: E402
    collect_gated_novelty_probe,
    train_epoch_with_signals,
)

OUT = _REPO / "results" / "phase1a_diagnostics" / "adaptation_cost"
PREREG = json.loads((OUT / "prereg.json").read_text())

ACC_REC_DIGIT = float(PREREG["recovery"]["digit_split_acc_rec_bar"])
DELTA_T = float(PREREG["compute_tie_margin"])
DELTA_TROUGH = float(PREREG["trough"]["delta_trough_max_for_plastic_win"])
FINETUNE_LR_MULT = float(PREREG["finetune_lr_mult"])


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(
    ds,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_digit_split_loaders(seed: int, batch_size: int = 128) -> dict[str, Any]:
    """Task A: digits 0–4; Task B: 5–9 remapped to 0–4. Shared TaskContext for B."""
    tmp = TaskContext(make_identity_transform())
    bundles = build_mnist_datasets(
        tmp, root="data/mnist", probe_size=1024, seed=seed, download=True
    )
    train_full = bundles["train_full"]
    targets = _mnist_targets(train_full)
    # Official test for A/B eval
    from torchvision import datasets, transforms
    from neuronca.data.mnist import default_image_transform

    test_full = datasets.MNIST(
        root="data/mnist",
        train=False,
        download=True,
        transform=default_image_transform(dataset="mnist"),
    )
    test_targets = _mnist_targets(test_full)

    def filt(idx: torch.Tensor, lo: int, hi: int) -> list[int]:
        t = targets[idx.long()]
        m = (t >= lo) & (t <= hi)
        return idx[m].long().tolist()

    train_idx = bundles["train_indices"]
    probe_idx = bundles["probe_indices"]
    a_train = filt(train_idx, 0, 4)
    a_probe = filt(probe_idx, 0, 4)
    b_train = filt(train_idx, 5, 9)
    b_probe = filt(probe_idx, 5, 9)
    a_test = indices_in_digit_range(test_targets, 0, 4).tolist()
    b_test = indices_in_digit_range(test_targets, 5, 9).tolist()

    ctx_a = TaskContext(make_identity_transform())
    ctx_b = TaskContext(make_identity_transform())  # flipped at switch

    def wrap(full, indices, ctx):
        return TaskAwareDataset(Subset(full, indices), ctx, return_index=False)

    return {
        "train_a": _loader(wrap(train_full, a_train, ctx_a), batch_size=batch_size, shuffle=True),
        "probe_a": _loader(wrap(train_full, a_probe, ctx_a), batch_size=batch_size, shuffle=False),
        "test_a": _loader(wrap(test_full, a_test, ctx_a), batch_size=batch_size, shuffle=False),
        "train_b": _loader(wrap(train_full, b_train, ctx_b), batch_size=batch_size, shuffle=True),
        "probe_b": _loader(wrap(train_full, b_probe, ctx_b), batch_size=batch_size, shuffle=False),
        "test_b": _loader(wrap(test_full, b_test, ctx_b), batch_size=batch_size, shuffle=False),
        "ctx_b": ctx_b,
        "transform_b": make_digit_split_transform(5, 9),
        "num_classes": 5,
        "shift": "digit_split",
    }


def build_b1_loaders(seed: int, batch_size: int = 128) -> dict[str, Any]:
    """Full MNIST; B = label permutation on shared loaders via TaskContext flip."""
    ctx = TaskContext(make_identity_transform())
    ld = __import__("neuronca.data.mnist", fromlist=["build_mnist_loaders"]).build_mnist_loaders(
        ctx,
        root="data/mnist",
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
        probe_size=1024,
        dataset="mnist",
    )
    return {
        "train_a": ld["train_loader"],
        "probe_a": ld["probe_loader"],
        "test_a": ld["test_loader"],
        "train_b": ld["train_loader"],
        "probe_b": ld["probe_loader"],
        "test_b": ld["test_loader"],
        "ctx_b": ctx,
        "transform_b": make_label_permutation_transform(10, seed=123),
        "num_classes": 10,
        "shift": "label_permutation_b1",
        "b1_artifact_warning": PREREG["b1_artifact_warning"],
    }


def plastic_cfg(*, max_new: int = 4) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates={},  # warm-start
        tau=0.5,
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
        self_calibrate={"r_star": True, "prune_cutoff": False, "straddle": False},
        self_cal_mu_r_star=0.999,
    )


def recovery_metrics(
    times: list[float],
    accs: list[float],
    *,
    acc_rec_bar: float,
) -> dict[str, Any]:
    t_abs = first_crossing(times, accs, threshold=acc_rec_bar, max_t=max(times) if times else 0)
    t_sus = first_sustained_crossing(
        times, accs, threshold=acc_rec_bar, max_t=max(times) if times else 0
    )
    trough = float(min(accs)) if accs else float("nan")
    return {
        "acc_rec_bar": acc_rec_bar,
        "t_star_abs": float(t_abs),
        "t_star_sustained": float(t_sus),
        "recovered_abs": bool(np.isfinite(t_abs)),
        "recovered_sustained": bool(np.isfinite(t_sus)),
        "acc_trough": trough,
        "trough_depth_vs_bar": float(acc_rec_bar - trough) if np.isfinite(trough) else float("nan"),
        "acc_end": float(accs[-1]) if accs else float("nan"),
        "acc_switch": float(accs[0]) if accs else float("nan"),
    }


def cost_to_sustained(
    traj: list[dict[str, Any]],
    t_star_sustained: float,
) -> dict[str, Any]:
    """Map sustained recovery epoch index to wall-clock / steps from B start."""
    if not np.isfinite(t_star_sustained):
        # full B cost
        if not traj:
            return {
                "wall_clock_s": float("nan"),
                "train_steps": 0,
                "epochs": float("inf"),
                "recovered": False,
            }
        return {
            "wall_clock_s": float(traj[-1]["cum_wall_s"]),
            "train_steps": int(traj[-1]["cum_steps"]),
            "epochs": float(traj[-1]["t_epoch"]),
            "recovered": False,
        }
    # t=0 is switch eval (0 train cost); t=k means after k B epochs
    t = float(t_star_sustained)
    if t <= 0:
        return {
            "wall_clock_s": 0.0,
            "train_steps": 0,
            "epochs": 0.0,
            "recovered": True,
        }
    # find first traj row with t_epoch >= t
    for row in traj:
        if row["t_epoch"] + 1e-9 >= t and row["phase"] == "B":
            return {
                "wall_clock_s": float(row["cum_wall_s"]),
                "train_steps": int(row["cum_steps"]),
                "epochs": float(row["t_epoch"]),
                "recovered": True,
            }
    last = traj[-1]
    return {
        "wall_clock_s": float(last["cum_wall_s"]),
        "train_steps": int(last["cum_steps"]),
        "epochs": float(last["t_epoch"]),
        "recovered": True,
    }


def run_arm(
    *,
    arm: str,
    shift: str,
    seed: int,
    epochs_a: int,
    epochs_b: int,
    batch_size: int = 128,
    max_new: int = 4,
) -> dict[str, Any]:
    """One arm: phase A → switch → phase B with frozen-continue / finetune / scratch."""
    set_seed(seed)
    device = _device()
    if shift in ("digit_split", "b2"):
        data = build_digit_split_loaders(seed, batch_size=batch_size)
        acc_rec_mode = "absolute"
    elif shift in ("b1", "label_permutation", "label_permutation_b1"):
        data = build_b1_loaders(seed, batch_size=batch_size)
        acc_rec_mode = "relative_pre"
    else:
        raise ValueError(f"unknown shift {shift}")

    n_classes = int(data["num_classes"])
    is_plastic = arm == "plastic"
    is_scratch = arm == "fixed_scratch"
    is_finetune = arm == "fixed_finetune"

    if is_plastic:
        # Probe r* on a throwaway net so phase-A epoch count matches fixed-topo.
        probe_net = DynamicMLPReasoningNet(
            hidden_init=(256, 128, 64),
            hidden_max=(512, 256, 128),
            num_classes=n_classes,
            coord_seed=seed,
        ).to(device)
        probe_opt = build_optimizer(
            probe_net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        probe = probe_target_rates(
            probe_net, data["train_a"], optimizer=probe_opt, device=device
        )
        del probe_net, probe_opt

        net = DynamicMLPReasoningNet(
            hidden_init=(256, 128, 64),
            hidden_max=(512, 256, 128),
            num_classes=n_classes,
            coord_seed=seed,
        ).to(device)
        cfg = plastic_cfg(max_new=max_new)
        orch = PlasticityOrchestrator(config=cfg)
        signal = ActivationMagnitude(decay=0.99)
        opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
        orch.warm_start_self_cal(target_rates=probe["target_rates"])
    else:
        net = MLPReasoningNet(
            input_dim=784, hidden=(256, 128, 64), num_classes=n_classes
        ).to(device)
        orch = None
        signal = None
        opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
        probe = None

    # ----- Phase A -----
    t_a0 = time.perf_counter()
    gstep = 0
    hist_a: list[dict[str, Any]] = []
    for ep in range(epochs_a):
        if is_plastic:
            assert orch is not None and signal is not None
            m = train_epoch_with_signals(
                net,  # type: ignore[arg-type]
                data["train_a"],
                opt,
                signal,
                orch,
                device=device,
                global_step=gstep,
            )
            gstep = int(m["global_step"])
            gated = collect_gated_novelty_probe(
                net,  # type: ignore[arg-type]
                data["probe_a"],
                device=device,
                radius=orch.config.radius,
                k_min=orch.config.novelty_k_min,
            )
            scores = {k: v.detach().cpu() for k, v in signal.neuron_scores().items()}
            if "in" not in scores:
                scores["in"] = torch.zeros(784)
            slog = orch.step(
                net,  # type: ignore[arg-type]
                scores,
                optimizer=opt,
                epoch=ep,
                signal=signal,
                gated_novelty=gated,
            )
            grown, pruned = int(slog.neurons_grown), int(slog.neurons_pruned)
        else:
            m = train_epoch(net, data["train_a"], opt, device=device)
            grown, pruned = 0, 0
            gstep += int(m.get("steps", 0))
        va = evaluate(net, data["test_a"], device=device)
        hist_a.append(
            {
                "epoch": ep,
                "phase": "A",
                "test_acc": float(va["acc"]),
                "neurons_grown": grown,
                "neurons_pruned": pruned,
            }
        )
        print(
            f"  [{arm} s{seed} A] ep{ep:02d} acc={va['acc']:.4f} g={grown} p={pruned}",
            flush=True,
        )
    wall_a = time.perf_counter() - t_a0
    acc_pre = float(hist_a[-1]["test_acc"]) if hist_a else float("nan")

    # ----- Switch -----
    data["ctx_b"].set_transform(data["transform_b"])
    if is_scratch:
        # Deprioritized baseline: reinit + fresh optimizer; train B only
        net = MLPReasoningNet(
            input_dim=784, hidden=(256, 128, 64), num_classes=n_classes
        ).to(device)
        opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
        is_plastic = False
        orch = None
        signal = None
    elif is_finetune:
        # Stretch: lower LR, keep weights + rebuild SGD state at low lr
        opt = build_optimizer(
            net, "sgd", lr=0.01 * FINETUNE_LR_MULT, momentum=0.9, weight_decay=1e-4
        )

    # Immediate post-switch eval on B (t=0)
    vb0 = evaluate(net, data["test_b"], device=device)
    times = [0.0]
    accs_b = [float(vb0["acc"])]
    traj_b: list[dict[str, Any]] = [
        {
            "t_epoch": 0.0,
            "phase": "B",
            "test_acc": float(vb0["acc"]),
            "cum_wall_s": 0.0,
            "cum_steps": 0,
            "neurons_grown": 0,
            "neurons_pruned": 0,
            "note": "switch_eval_before_B_train",
        }
    ]
    print(
        f"  [{arm} s{seed} switch] B_acc={vb0['acc']:.4f} (pre_A_acc={acc_pre:.4f})",
        flush=True,
    )

    # ----- Phase B -----
    t_b0 = time.perf_counter()
    cum_steps = 0
    tot_grown = tot_pruned = 0
    for ep in range(epochs_b):
        ep_t0 = time.perf_counter()
        if is_plastic:
            assert orch is not None and signal is not None
            m = train_epoch_with_signals(
                net,  # type: ignore[arg-type]
                data["train_b"],
                opt,
                signal,
                orch,
                device=device,
                global_step=gstep,
            )
            gstep = int(m["global_step"])
            steps = int(m.get("steps", 0))
            gated = collect_gated_novelty_probe(
                net,  # type: ignore[arg-type]
                data["probe_b"],
                device=device,
                radius=orch.config.radius,
                k_min=orch.config.novelty_k_min,
            )
            scores = {k: v.detach().cpu() for k, v in signal.neuron_scores().items()}
            if "in" not in scores:
                scores["in"] = torch.zeros(784)
            slog = orch.step(
                net,  # type: ignore[arg-type]
                scores,
                optimizer=opt,
                epoch=epochs_a + ep,
                signal=signal,
                gated_novelty=gated,
            )
            grown, pruned = int(slog.neurons_grown), int(slog.neurons_pruned)
        else:
            m = train_epoch(net, data["train_b"], opt, device=device)
            steps = int(m.get("steps", 0))
            grown, pruned = 0, 0
        cum_steps += steps
        tot_grown += grown
        tot_pruned += pruned
        vb = evaluate(net, data["test_b"], device=device)
        cum_wall = time.perf_counter() - t_b0
        t_ep = float(ep + 1)  # after this many B epochs
        times.append(t_ep)
        accs_b.append(float(vb["acc"]))
        traj_b.append(
            {
                "t_epoch": t_ep,
                "phase": "B",
                "test_acc": float(vb["acc"]),
                "cum_wall_s": float(cum_wall),
                "cum_steps": int(cum_steps),
                "epoch_wall_s": float(time.perf_counter() - ep_t0),
                "neurons_grown": grown,
                "neurons_pruned": pruned,
            }
        )
        print(
            f"  [{arm} s{seed} B] ep{ep:02d} acc={vb['acc']:.4f} "
            f"g={grown} p={pruned} wall={cum_wall:.1f}s",
            flush=True,
        )
    wall_b = time.perf_counter() - t_b0

    if acc_rec_mode == "relative_pre":
        acc_rec_bar = 0.90 * acc_pre
    else:
        acc_rec_bar = ACC_REC_DIGIT

    rec = recovery_metrics(times, accs_b, acc_rec_bar=acc_rec_bar)
    cost = cost_to_sustained(traj_b, rec["t_star_sustained"])

    out: dict[str, Any] = {
        "arm": arm,
        "shift": data["shift"],
        "seed": seed,
        "epochs_a": epochs_a,
        "epochs_b": epochs_b,
        "num_classes": n_classes,
        "acc_pre_task_a": acc_pre,
        "acc_rec_bar": acc_rec_bar,
        "acc_rec_mode": acc_rec_mode,
        "recovery": rec,
        "cost_to_sustained": cost,
        "wall_clock_phase_a_s": float(wall_a),
        "wall_clock_phase_b_s": float(wall_b),
        "train_steps_phase_b": int(cum_steps),
        "structure_b": {"neurons_grown": tot_grown, "neurons_pruned": tot_pruned},
        "trajectory_a": hist_a,
        "trajectory_b": traj_b,
        "b1_artifact_warning": data.get("b1_artifact_warning"),
        "probe_r_star": None if probe is None else probe.get("target_rates"),
        "baseline_protocol": {
            "frozen_continue": "weights+opt continue; same lr",
            "fixed_finetune": f"weights continue; lr *= {FINETUNE_LR_MULT}",
            "fixed_scratch": "reinit at switch; train B only",
            "plastic": "weights continue; structure plastic; self-cal r*",
        }.get(arm, arm),
    }
    if is_plastic and isinstance(net, DynamicMLPReasoningNet):
        out["hidden_end"] = net.hidden_counts()
        out["active_params_end"] = int(net.active_parameter_count())
    return out


def classify_seed(
    plastic: dict[str, Any], fixed: dict[str, Any]
) -> dict[str, Any]:
    pr = plastic["recovery"]["recovered_sustained"]
    fr = fixed["recovery"]["recovered_sustained"]
    if not pr and not fr:
        label = "BOTH_FAIL_RECOVER"
    elif fr and not pr:
        label = "FIXED_ONLY"
    elif pr and not fr:
        label = "PLASTIC_ONLY"
    else:
        tw_p = plastic["cost_to_sustained"]["wall_clock_s"]
        tw_f = fixed["cost_to_sustained"]["wall_clock_s"]
        if tw_p < tw_f * (1.0 - DELTA_T):
            sub = "PLASTIC_FASTER_COMPUTE"
        elif tw_f < tw_p * (1.0 - DELTA_T):
            sub = "FIXED_FASTER_COMPUTE"
        else:
            sub = "COMPUTE_TIE"
        # trough guard
        tp = plastic["recovery"]["acc_trough"]
        tf = fixed["recovery"]["acc_trough"]
        trough_ok = (tf - tp) <= DELTA_TROUGH
        label = f"BOTH_RECOVER_{sub}"
        if sub == "PLASTIC_FASTER_COMPUTE" and not trough_ok:
            label = "BOTH_RECOVER_PLASTIC_FASTER_BUT_TROUGH_FAIL"
    return {
        "label": label,
        "plastic_recovered_sustained": pr,
        "fixed_recovered_sustained": fr,
        "T_w_plastic": plastic["cost_to_sustained"]["wall_clock_s"],
        "T_w_fixed": fixed["cost_to_sustained"]["wall_clock_s"],
        "acc_trough_plastic": plastic["recovery"]["acc_trough"],
        "acc_trough_fixed": fixed["recovery"]["acc_trough"],
        "t_star_sustained_plastic": plastic["recovery"]["t_star_sustained"],
        "t_star_sustained_fixed": fixed["recovery"]["t_star_sustained"],
    }


def multi_seed_headline(seed_labels: list[str]) -> str:
    def count(pred) -> int:
        return sum(1 for s in seed_labels if pred(s))

    n = len(seed_labels)
    if n == 0:
        return "ADAPT_COST_INCONCLUSIVE"
    need = max(1, (2 * n + 2) // 3)  # ≥2/3 rounded up for n=3 → 2
    if count(lambda s: s == "PLASTIC_ONLY" or s == "BOTH_RECOVER_PLASTIC_FASTER_COMPUTE") >= need:
        return "ADAPT_COST_PLASTIC_WINS"
    if count(lambda s: s == "FIXED_ONLY" or s == "BOTH_RECOVER_FIXED_FASTER_COMPUTE") >= need:
        return "ADAPT_COST_FIXED_WINS"
    if count(lambda s: "TROUGH_FAIL" in s) >= need:
        return "ADAPT_COST_TRADEOFF"
    if count(lambda s: s == "BOTH_FAIL_RECOVER") >= need:
        return "ADAPT_COST_BOTH_FAIL"
    if count(lambda s: s.startswith("BOTH_RECOVER")) >= need:
        return "ADAPT_COST_TRADEOFF"
    return "ADAPT_COST_INCONCLUSIVE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shift", type=str, default="digit_split",
                    choices=["digit_split", "b2", "b1", "label_permutation"])
    ap.add_argument("--arms", type=str, default="plastic,fixed_continue")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--epochs-a", type=int, default=None)
    ap.add_argument("--epochs-b", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new", type=int, default=4)
    args = ap.parse_args()

    if args.smoke:
        epochs_a = PREREG["protocol"]["smoke_epochs_a"]
        epochs_b = PREREG["protocol"]["smoke_epochs_b"]
        seeds = [0]
    else:
        epochs_a = args.epochs_a or PREREG["protocol"]["epochs_a"]
        epochs_b = args.epochs_b or PREREG["protocol"]["epochs_b"]
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    shift = args.shift
    shift_dir = "digit_split" if shift in ("digit_split", "b2") else "b1_label_perm"
    root = OUT / shift_dir
    root.mkdir(parents=True, exist_ok=True)

    by_seed: dict[str, Any] = {}
    for seed in seeds:
        print(f"\n======== seed {seed} shift={shift} ========", flush=True)
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for arm in arms:
            print(f"\n--- arm={arm} ---", flush=True)
            res = run_arm(
                arm=arm,
                shift=shift,
                seed=seed,
                epochs_a=epochs_a,
                epochs_b=epochs_b,
                max_new=args.max_new,
            )
            results[arm] = res
            (seed_dir / f"{arm}.json").write_text(json.dumps(res, indent=2))
            print(
                f"  -> sustained={res['recovery']['recovered_sustained']} "
                f"t*={res['recovery']['t_star_sustained']} "
                f"trough={res['recovery']['acc_trough']:.3f} "
                f"T_w={res['cost_to_sustained']['wall_clock_s']:.1f}s "
                f"end={res['recovery']['acc_end']:.3f}",
                flush=True,
            )

        seed_block: dict[str, Any] = {"seed": seed, "arms": results}
        if "plastic" in results and "fixed_continue" in results:
            seed_block["classification"] = classify_seed(
                results["plastic"], results["fixed_continue"]
            )
            print(f"  class={seed_block['classification']['label']}", flush=True)
        by_seed[str(seed)] = seed_block
        (seed_dir / "summary.json").write_text(json.dumps(seed_block, indent=2, default=str))

    labels = [
        by_seed[str(s)]["classification"]["label"]
        for s in seeds
        if "classification" in by_seed[str(s)]
    ]
    headline = multi_seed_headline(labels) if labels else "ADAPT_COST_INCONCLUSIVE"
    summary = {
        "shift": shift_dir,
        "prereg": "docs/adaptation_cost_preregistration.md",
        "epochs_a": epochs_a,
        "epochs_b": epochs_b,
        "seeds": seeds,
        "arms": arms,
        "headline": headline,
        "seed_classifications": labels,
        "by_seed": {
            k: {
                "classification": v.get("classification"),
                "plastic_recovery": (v.get("arms") or {}).get("plastic", {}).get("recovery"),
                "fixed_recovery": (v.get("arms") or {}).get("fixed_continue", {}).get("recovery"),
                "plastic_cost": (v.get("arms") or {}).get("plastic", {}).get("cost_to_sustained"),
                "fixed_cost": (v.get("arms") or {}).get("fixed_continue", {}).get("cost_to_sustained"),
            }
            for k, v in by_seed.items()
        },
        "notes": [
            "Primary baseline: frozen_continue (weights continue, no reset).",
            "Primary cost: wall-clock to t_star_sustained.",
            "Primary recovery: t_star_sustained (not first-crossing).",
            "Primary shift: digit_split; B1 flagged for frozen-weight artifact.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"# Adaptation-cost — `{shift_dir}`",
        "",
        f"**Headline:** `{headline}`",
        "",
        f"Seeds {seeds}; A={epochs_a} ep, B={epochs_b} ep.",
        "",
        "See `docs/adaptation_cost_preregistration.md`.",
        "",
        "| Seed | Class | T_w plastic | T_w fixed | trough P | trough F | t*_sus P | t*_sus F |",
        "|-----:|-------|------------:|----------:|---------:|---------:|---------:|---------:|",
    ]
    for s in seeds:
        b = by_seed[str(s)]
        c = b.get("classification") or {}
        lines.append(
            f"| {s} | {c.get('label', '—')} | "
            f"{c.get('T_w_plastic', float('nan')):.1f} | "
            f"{c.get('T_w_fixed', float('nan')):.1f} | "
            f"{c.get('acc_trough_plastic', float('nan')):.3f} | "
            f"{c.get('acc_trough_fixed', float('nan')):.3f} | "
            f"{c.get('t_star_sustained_plastic', float('nan'))} | "
            f"{c.get('t_star_sustained_fixed', float('nan'))} |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps(summary, indent=2)[:2500], flush=True)
    print(f"\nWrote {root}/", flush=True)


if __name__ == "__main__":
    main()
