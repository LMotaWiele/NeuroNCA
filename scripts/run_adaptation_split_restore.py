#!/usr/bin/env python3
"""Split→restore retention: A (0–4) → B (5–9) → restore A. Prereg: docs/adaptation_cost_split_restore_prereg.md."""

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
    default_image_transform,
)
from neuronca.data.tasks import (  # noqa: E402
    TaskContext,
    indices_in_digit_range,
    make_digit_split_transform,
    make_identity_transform,
)
from neuronca.eval.time_lag import first_sustained_crossing  # noqa: E402
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
from torchvision import datasets  # noqa: E402

OUT = _REPO / "results" / "phase1a_diagnostics" / "adaptation_cost" / "split_restore"
DELTA_RET = 0.05
ACC_B_MIN = 0.80  # both arms must learn B
TOO_EASY_RET = 0.85
CATASTROPHIC_RET = 0.25


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ds, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_loaders(seed: int, batch_size: int = 128) -> dict[str, Any]:
    tmp = TaskContext(make_identity_transform())
    bundles = build_mnist_datasets(
        tmp, root="data/mnist", probe_size=1024, seed=seed, download=True
    )
    train_full = bundles["train_full"]
    targets = _mnist_targets(train_full)
    test_full = datasets.MNIST(
        root="data/mnist",
        train=False,
        download=True,
        transform=default_image_transform(dataset="mnist"),
    )
    test_targets = _mnist_targets(test_full)

    def filt(idx: torch.Tensor, lo: int, hi: int) -> list[int]:
        t = targets[idx.long()]
        return idx[(t >= lo) & (t <= hi)].long().tolist()

    train_idx = bundles["train_indices"]
    probe_idx = bundles["probe_indices"]
    a_train, a_probe = filt(train_idx, 0, 4), filt(probe_idx, 0, 4)
    b_train, b_probe = filt(train_idx, 5, 9), filt(probe_idx, 5, 9)
    a_test = indices_in_digit_range(test_targets, 0, 4).tolist()
    b_test = indices_in_digit_range(test_targets, 5, 9).tolist()

    ctx_a = TaskContext(make_identity_transform())
    ctx_b = TaskContext(make_identity_transform())

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
        "transform_a": make_identity_transform(),
    }


def plastic_cfg(max_new: int = 4) -> PlasticityConfig:
    return PlasticityConfig(
        radius=0.35,
        target_rates={},
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


def train_phase(
    *,
    net,
    opt,
    train_loader,
    probe_loader,
    test_loader,
    epochs: int,
    device,
    phase: str,
    is_plastic: bool,
    orch=None,
    signal=None,
    gstep: int = 0,
    dual_eval_a=None,
    dual_eval_b=None,
) -> tuple[list[dict[str, Any]], int]:
    """Train one phase; return trajectory rows and updated global_step."""
    rows: list[dict[str, Any]] = []
    for ep in range(epochs):
        if is_plastic:
            assert orch is not None and signal is not None
            m = train_epoch_with_signals(
                net, train_loader, opt, signal, orch, device=device, global_step=gstep
            )
            gstep = int(m["global_step"])
            gated = collect_gated_novelty_probe(
                net,
                probe_loader,
                device=device,
                radius=orch.config.radius,
                k_min=orch.config.novelty_k_min,
            )
            scores = {k: v.detach().cpu() for k, v in signal.neuron_scores().items()}
            if "in" not in scores:
                scores["in"] = torch.zeros(784)
            slog = orch.step(
                net, scores, optimizer=opt, epoch=ep, signal=signal, gated_novelty=gated
            )
            grown, pruned = int(slog.neurons_grown), int(slog.neurons_pruned)
        else:
            train_epoch(net, train_loader, opt, device=device)
            grown = pruned = 0
        va = evaluate(net, test_loader, device=device)
        row: dict[str, Any] = {
            "phase": phase,
            "epoch": ep,
            "test_acc": float(va["acc"]),
            "neurons_grown": grown,
            "neurons_pruned": pruned,
        }
        if dual_eval_a is not None:
            row["acc_A"] = float(evaluate(net, dual_eval_a, device=device)["acc"])
        if dual_eval_b is not None:
            row["acc_B"] = float(evaluate(net, dual_eval_b, device=device)["acc"])
        rows.append(row)
        extra = ""
        if "acc_A" in row:
            extra += f" A={row['acc_A']:.4f}"
        if "acc_B" in row:
            extra += f" B={row['acc_B']:.4f}"
        print(
            f"  [{phase}] ep{ep:02d} acc={va['acc']:.4f} g={grown} p={pruned}{extra}",
            flush=True,
        )
    return rows, gstep


def run_arm(
    arm: str,
    seed: int,
    *,
    epochs_a: int,
    epochs_b: int,
    epochs_c: int,
    max_new: int = 4,
    finetune: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    device = _device()
    data = build_loaders(seed)
    is_plastic = arm == "plastic"

    if is_plastic:
        probe_net = DynamicMLPReasoningNet(
            hidden_init=(256, 128, 64),
            hidden_max=(512, 256, 128),
            num_classes=5,
            coord_seed=seed,
        ).to(device)
        probe = probe_target_rates(
            probe_net,
            data["train_a"],
            optimizer=build_optimizer(
                probe_net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4
            ),
            device=device,
        )
        del probe_net
        net: torch.nn.Module = DynamicMLPReasoningNet(
            hidden_init=(256, 128, 64),
            hidden_max=(512, 256, 128),
            num_classes=5,
            coord_seed=seed,
        ).to(device)
        orch = PlasticityOrchestrator(config=plastic_cfg(max_new=max_new))
        signal = ActivationMagnitude(decay=0.99)
        opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
        orch.warm_start_self_cal(target_rates=probe["target_rates"])
        probe_rates = probe["target_rates"]
    else:
        net = MLPReasoningNet(
            input_dim=784, hidden=(256, 128, 64), num_classes=5
        ).to(device)
        orch = None
        signal = None
        opt = build_optimizer(net, "sgd", lr=0.01, momentum=0.9, weight_decay=1e-4)
        probe_rates = None

    # ----- A: learn 0–4 -----
    print(f"\n=== {arm} s{seed} phase A (0–4) ===", flush=True)
    hist_a, gstep = train_phase(
        net=net,
        opt=opt,
        train_loader=data["train_a"],
        probe_loader=data["probe_a"],
        test_loader=data["test_a"],
        epochs=epochs_a,
        device=device,
        phase="A",
        is_plastic=is_plastic,
        orch=orch,
        signal=signal,
        gstep=0,
    )
    acc_A_end = float(hist_a[-1]["test_acc"])

    # ----- B: learn 5–9 -----
    data["ctx_b"].set_transform(data["transform_b"])
    if finetune and not is_plastic:
        opt = build_optimizer(
            net, "sgd", lr=0.01 * 0.1, momentum=0.9, weight_decay=1e-4
        )
    print(f"\n=== {arm} s{seed} phase B (5–9) ===", flush=True)
    t_b0 = time.perf_counter()
    hist_b, gstep = train_phase(
        net=net,
        opt=opt,
        train_loader=data["train_b"],
        probe_loader=data["probe_b"],
        test_loader=data["test_b"],
        epochs=epochs_b,
        device=device,
        phase="B",
        is_plastic=is_plastic,
        orch=orch,
        signal=signal,
        gstep=gstep,
        dual_eval_a=data["test_a"],  # track forgetting of A during B
        dual_eval_b=None,
    )
    wall_b = time.perf_counter() - t_b0
    acc_B_end = float(hist_b[-1]["test_acc"])
    # Retention: A accuracy after B (identity transform on A loaders)
    acc_A_after_B = float(evaluate(net, data["test_a"], device=device)["acc"])
    forgetting = acc_A_end - acc_A_after_B

    # ----- C: restore 0–4 -----
    print(f"\n=== {arm} s{seed} phase C (restore 0–4) ===", flush=True)
    # Ensure A identity for C training (ctx_a loaders already identity)
    t_c0 = time.perf_counter()
    # switch eval on A before C train
    acc_A_c0 = float(evaluate(net, data["test_a"], device=device)["acc"])
    times_c = [0.0]
    accs_c = [acc_A_c0]
    traj_c = [
        {
            "t_epoch": 0.0,
            "acc_A": acc_A_c0,
            "cum_wall_s": 0.0,
            "note": "C_start_before_train",
        }
    ]
    cum_steps = 0
    for ep in range(epochs_c):
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
            cum_steps += int(m.get("steps", 0))
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
            cum_steps += int(m.get("steps", 0))
            grown = pruned = 0
        acc_A = float(evaluate(net, data["test_a"], device=device)["acc"])
        cum_wall = time.perf_counter() - t_c0
        t_ep = float(ep + 1)
        times_c.append(t_ep)
        accs_c.append(acc_A)
        traj_c.append(
            {
                "t_epoch": t_ep,
                "acc_A": acc_A,
                "cum_wall_s": cum_wall,
                "cum_steps": cum_steps,
                "neurons_grown": grown,
                "neurons_pruned": pruned,
            }
        )
        print(
            f"  [C] ep{ep:02d} A={acc_A:.4f} g={grown} p={pruned} wall={cum_wall:.1f}s",
            flush=True,
        )
    wall_c = time.perf_counter() - t_c0

    acc_rec_bar = 0.90 * acc_A_end
    t_sus = first_sustained_crossing(
        times_c, accs_c, threshold=acc_rec_bar, max_t=max(times_c)
    )
    # cost to sustained on C
    if not np.isfinite(t_sus):
        t_w_c = wall_c
        rec_c = False
    elif t_sus <= 0:
        t_w_c = 0.0
        rec_c = True
    else:
        rec_c = True
        t_w_c = wall_c
        for row in traj_c:
            if row["t_epoch"] + 1e-9 >= t_sus:
                t_w_c = float(row["cum_wall_s"])
                break

    # A trajectory during B (forgetting curve)
    forget_curve = [
        {"epoch": r["epoch"], "acc_A": r.get("acc_A"), "acc_B": r["test_acc"]}
        for r in hist_b
    ]

    return {
        "arm": arm,
        "seed": seed,
        "acc_A_end": acc_A_end,
        "acc_B_end": acc_B_end,
        "acc_A_after_B": acc_A_after_B,
        "forgetting": forgetting,
        "retained_acc": acc_A_after_B,
        "b_learned": acc_B_end >= ACC_B_MIN,
        "restore": {
            "acc_rec_bar": acc_rec_bar,
            "t_star_sustained": float(t_sus),
            "recovered_sustained": rec_c,
            "T_w_C": float(t_w_c),
            "acc_A_trough_C": float(min(accs_c)),
            "acc_A_end_C": float(accs_c[-1]),
            "wall_clock_C_s": float(wall_c),
        },
        "wall_clock_B_s": float(wall_b),
        "history_a": hist_a,
        "history_b": hist_b,
        "forget_curve_during_B": forget_curve,
        "trajectory_c": traj_c,
        "probe_r_star": probe_rates,
    }


def discriminating(retained_p: float, retained_f: float) -> str:
    if retained_p >= TOO_EASY_RET and retained_f >= TOO_EASY_RET:
        return "TASK_TOO_EASY_FOR_RETENTION"
    if retained_p <= CATASTROPHIC_RET and retained_f <= CATASTROPHIC_RET:
        return "BOTH_CATASTROPHIC_FORGET"
    if abs(retained_p - retained_f) < DELTA_RET:
        # still discriminating if one side of 0.5/0.35 split
        if (retained_p > 0.5 and retained_f < 0.35) or (
            retained_f > 0.5 and retained_p < 0.35
        ):
            return "DISCRIMINATING"
        return "RETENTION_TIE_BAND"
    return "DISCRIMINATING"


def classify_seed(p: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    gate = discriminating(p["retained_acc"], f["retained_acc"])
    if gate in ("TASK_TOO_EASY_FOR_RETENTION", "BOTH_CATASTROPHIC_FORGET"):
        label = gate
    elif not p["b_learned"] or not f["b_learned"]:
        label = "INVALID_B_NOT_LEARNED"
    elif p["retained_acc"] >= f["retained_acc"] + DELTA_RET:
        label = "RETENTION_PLASTIC_WINS"
    elif f["retained_acc"] >= p["retained_acc"] + DELTA_RET:
        label = "RETENTION_FIXED_WINS"
    else:
        label = "RETENTION_TIE"
    return {
        "label": label,
        "gate": gate,
        "retained_plastic": p["retained_acc"],
        "retained_fixed": f["retained_acc"],
        "forgetting_plastic": p["forgetting"],
        "forgetting_fixed": f["forgetting"],
        "acc_B_end_plastic": p["acc_B_end"],
        "acc_B_end_fixed": f["acc_B_end"],
        "T_w_C_plastic": p["restore"]["T_w_C"],
        "T_w_C_fixed": f["restore"]["T_w_C"],
    }


def multi_seed_headline(labels: list[str]) -> str:
    n = len(labels)
    if n == 0:
        return "RETENTION_INCONCLUSIVE"
    need = max(1, (2 * n + 2) // 3)
    if sum(1 for s in labels if s == "RETENTION_PLASTIC_WINS") >= need:
        return "RETENTION_PLASTIC_WINS"
    if sum(1 for s in labels if s == "RETENTION_FIXED_WINS") >= need:
        return "RETENTION_FIXED_WINS"
    if sum(
        1
        for s in labels
        if s in ("TASK_TOO_EASY_FOR_RETENTION", "BOTH_CATASTROPHIC_FORGET")
    ) >= need:
        return "RETENTION_NON_DISCRIMINATING"
    if sum(1 for s in labels if s == "RETENTION_TIE") >= need:
        return "RETENTION_TIE"
    return "RETENTION_TIE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--epochs-a", type=int, default=None)
    ap.add_argument("--epochs-b", type=int, default=None)
    ap.add_argument("--epochs-c", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new", type=int, default=4)
    args = ap.parse_args()

    if args.smoke:
        ea, eb, ec, seeds = 2, 3, 3, [0]
    else:
        ea = args.epochs_a or 10
        eb = args.epochs_b or 15
        ec = args.epochs_c or 15
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    by_seed: dict[str, Any] = {}

    for seed in seeds:
        print(f"\n######## SEED {seed} split→restore ########", flush=True)
        seed_dir = OUT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for arm in ("plastic", "fixed_continue"):
            print(f"\n----- {arm} -----", flush=True)
            res = run_arm(
                arm,
                seed,
                epochs_a=ea,
                epochs_b=eb,
                epochs_c=ec,
                max_new=args.max_new,
            )
            results[arm] = res
            (seed_dir / f"{arm}.json").write_text(json.dumps(res, indent=2))
            print(
                f"  -> A_end={res['acc_A_end']:.3f} B_end={res['acc_B_end']:.3f} "
                f"A_after_B={res['acc_A_after_B']:.3f} forget={res['forgetting']:.3f} "
                f"C_t*={res['restore']['t_star_sustained']} T_w_C={res['restore']['T_w_C']:.1f}s",
                flush=True,
            )
        cls = classify_seed(results["plastic"], results["fixed_continue"])
        print(f"  class={cls['label']} gate={cls['gate']}", flush=True)
        by_seed[str(seed)] = {"seed": seed, "arms": results, "classification": cls}
        (seed_dir / "summary.json").write_text(
            json.dumps(by_seed[str(seed)], indent=2, default=str)
        )

    labels = [by_seed[str(s)]["classification"]["label"] for s in seeds]
    headline = multi_seed_headline(labels)
    summary = {
        "protocol": "A=0-4, B=5-9, C=restore 0-4",
        "prereg": "docs/adaptation_cost_split_restore_prereg.md",
        "epochs": {"A": ea, "B": eb, "C": ec},
        "seeds": seeds,
        "headline": headline,
        "seed_classifications": labels,
        "stop_rule": (
            "This is the single harder adaptation cell. Outcome is final for "
            "retention thesis on these digit tasks."
        ),
        "by_seed": {
            k: {
                "classification": v["classification"],
                "plastic_retained": v["arms"]["plastic"]["retained_acc"],
                "fixed_retained": v["arms"]["fixed_continue"]["retained_acc"],
                "plastic_A_end": v["arms"]["plastic"]["acc_A_end"],
                "fixed_A_end": v["arms"]["fixed_continue"]["acc_A_end"],
                "plastic_B_end": v["arms"]["plastic"]["acc_B_end"],
                "fixed_B_end": v["arms"]["fixed_continue"]["acc_B_end"],
            }
            for k, v in by_seed.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Split→restore retention",
        "",
        f"**Headline:** `{headline}`",
        "",
        f"Seeds {seeds}; A={ea}, B={eb}, C={ec}.",
        "",
        "| Seed | Class | retained P | retained F | forget P | forget F | B_end P | B_end F |",
        "|-----:|-------|-----------:|-----------:|---------:|---------:|--------:|--------:|",
    ]
    for s in seeds:
        c = by_seed[str(s)]["classification"]
        lines.append(
            f"| {s} | {c['label']} | {c['retained_plastic']:.3f} | "
            f"{c['retained_fixed']:.3f} | {c['forgetting_plastic']:.3f} | "
            f"{c['forgetting_fixed']:.3f} | {c['acc_B_end_plastic']:.3f} | "
            f"{c['acc_B_end_fixed']:.3f} |"
        )
    lines += [
        "",
        "Primary metric: `retained_acc` = task-A accuracy after phase B.",
        "See `docs/adaptation_cost_split_restore_prereg.md`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
