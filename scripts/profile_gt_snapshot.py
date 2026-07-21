#!/usr/bin/env python3
"""Profile one full-GT snapshot to locate time (before any rewrite).

Measures a *real* ``InstrumentedTrainer.snapshot`` with the same GT suite used
in production stationary coarse tags (``grad_abs`` + ``ablation_relu``).

Outputs
-------
- Wall-clock sectional timers (train warm-up, freeze, grad GT, ablation GT,
  full snapshot, isolation assert, GEVB alignment).
- cProfile top cumulative / tottime functions.
- ``torch.profiler`` CPU table (and CUDA if available).
- Micro-benchmarks: bare ``mean_probe_loss``, hook register/remove loop,
  extrapolated neuron/edge ablation counts.

Usage
-----
::

    python scripts/profile_gt_snapshot.py
    python scripts/profile_gt_snapshot.py --device cuda --probe-size 1024
    python scripts/profile_gt_snapshot.py --skip-full-ablation  # grad-only micro

Does **not** change production code or thresholds.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# repo root on path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from neuronca.broadcast import BroadcastErrorEncoder
from neuronca.data.mnist import build_mnist_loaders
from neuronca.data.tasks import TaskContext, make_identity_transform
from neuronca.importance import AblationImportance, GradientSaliency
from neuronca.importance.ablation import load_edge_ablation_mask
from neuronca.importance.base import mean_probe_loss
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.signals import (
    ActivationMagnitude,
    GEVBVectorEligibility,
    HebbianResidualEnergyCorr,
    ThreeFactorScalarEligibility,
)
from neuronca.signals.gevb_alignment import compute_gevb_alignment
from neuronca.training.callbacks import InstrumentedTrainer, capture_signal_traces
from neuronca.training.optim import build_optimizer


def _sec(t0: float) -> float:
    return time.perf_counter() - t0


def section(name: str, fn, reps: int = 1) -> float:
    # warm-up once
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(_sec(t0))
    mean = sum(times) / len(times)
    print(f"  {name:48s}  {mean:8.3f}s  (n={reps})")
    return mean


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--probe-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=50, help="warm-up train steps")
    p.add_argument(
        "--edge-sample",
        type=int,
        default=512,
        help="edge ablation sample per layer (production default 512)",
    )
    p.add_argument(
        "--skip-full-ablation",
        action="store_true",
        help="skip full ablation compute (only micro + grad GT + structure)",
    )
    p.add_argument(
        "--out-dir",
        default="results/phase0_profile",
        help="directory for profiler artifacts",
    )
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device={device} probe_size={args.probe_size} edge_sample={args.edge_sample}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data / model matching production scale ---
    ctx = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        ctx,
        root="data/mnist",
        probe_size=args.probe_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=0,
        download=True,
    )
    probe_loader = loaders["probe_loader"]
    train_loader = loaders["train_loader"]

    model = MLPReasoningNet().to(device)
    opt = build_optimizer(model)
    signals = [
        ActivationMagnitude(decay=0.99),
        HebbianResidualEnergyCorr(decay=0.99),
        ThreeFactorScalarEligibility(decay=0.99),
        GEVBVectorEligibility(
            decay=0.99,
            experiment_seed=args.seed,
            num_classes=10,
            layer_shapes=model.layer_shapes(),
        ),
    ]
    trainer = InstrumentedTrainer(
        model,
        opt,
        signals,
        encoder=BroadcastErrorEncoder(),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    # Warm signals / weights with real train path
    t0 = time.perf_counter()
    trainer.train_steps(train_loader, max_steps=args.train_steps)
    print(f"warm-up train_steps({args.train_steps}): {_sec(t0):.3f}s")

    # Edge mask: reuse primary seed-0 mask if shapes match, else build
    mask_path = Path(f"results/phase0/meta/edge_ablation_mask_seed{args.seed}.pt")
    if mask_path.is_file():
        edge_mask = load_edge_ablation_mask(mask_path)
        print(f"loaded edge mask from {mask_path}")
    else:
        from neuronca.importance.ablation import build_edge_ablation_mask

        edge_mask = build_edge_ablation_mask(
            model, seed=args.seed, sample_per_layer=args.edge_sample
        )
        print("built fresh edge mask")

    n_neuron = sum(
        model.get_submodule(n).out_features for n in ("fc1", "fc2", "fc3")
    )
    n_edge = sum(int(m.sum().item()) for m in edge_mask.values())
    print(f"ablation units: neurons={n_neuron}  masked_edges={n_edge}  total_units={n_neuron+n_edge}")

    criterion = nn.CrossEntropyLoss()
    gt_grad = GradientSaliency()
    gt_abl = AblationImportance(
        edge_ablation_sample=args.edge_sample,
        seed=args.seed,
        neuron_ablation="all_hidden",
    )
    gt_full = [gt_grad, gt_abl]
    gt_grad_only = [gt_grad]

    # =====================================================================
    # 1) Micro-benchmarks (locate per-call cost)
    # =====================================================================
    print("\n=== Micro-benchmarks ===")

    def once_probe_loss():
        return mean_probe_loss(model, probe_loader, criterion, device=torch.device(device))

    t_probe = section("mean_probe_loss (1 full probe pass)", once_probe_loss, reps=5)

    # Hook register/remove only (no forward)
    mod = model.fc1

    def hook_reg_loop(n: int = 448):
        for j in range(n):
            h = mod.register_forward_hook(lambda m, i, o, jj=j: o)
            h.remove()

    t_hooks = section("hook register+remove x448 (no forward)", lambda: hook_reg_loop(448), reps=3)

    # One neuron ablation unit = hook + full probe
    def one_neuron_ablation_unit():
        h = mod.register_forward_hook(
            lambda m, i, o: torch.cat([o[:, :1] * 0, o[:, 1:]], dim=1)
            if o.shape[1] > 1
            else o * 0
        )
        try:
            mean_probe_loss(model, probe_loader, criterion, device=torch.device(device))
        finally:
            h.remove()

    t_one_neuron = section("1 neuron ablation unit (hook+probe)", one_neuron_ablation_unit, reps=3)

    # One edge ablation unit
    def one_edge_unit():
        with torch.no_grad():
            orig = model.fc1.weight[0, 0].item()
            model.fc1.weight[0, 0] = 0.0
        try:
            mean_probe_loss(model, probe_loader, criterion, device=torch.device(device))
        finally:
            with torch.no_grad():
                model.fc1.weight[0, 0] = orig

    t_one_edge = section("1 edge ablation unit (w=0 + probe)", one_edge_unit, reps=3)

    # Extrapolations
    print("\n=== Extrapolated ablation cost (from micro) ===")
    print(f"  neurons only ({n_neuron} x probe):     {n_neuron * t_one_neuron:8.1f}s")
    print(f"  edges only   ({n_edge} x probe):     {n_edge * t_one_edge:8.1f}s")
    print(f"  neurons+edges:                       {(n_neuron + n_edge) * t_probe:8.1f}s  (if probe dominates)")
    print(f"  hook overhead only x{n_neuron}:              {t_hooks * (n_neuron / 448):8.3f}s")

    # =====================================================================
    # 2) Sectional wall-clock (production path pieces)
    # =====================================================================
    print("\n=== Sectional wall-clock (production path) ===")

    def freeze():
        trainer.freeze_local_scores()

    def run_grad():
        with trainer.signals_disabled():
            gt_grad.compute(model, probe_loader, criterion)

    def run_abl():
        with trainer.signals_disabled():
            gt_abl.compute(
                model, probe_loader, criterion, edge_sample_mask=edge_mask
            )

    def run_snap_grad_only():
        trainer.snapshot(
            gt_grad_only, probe_loader, edge_sample_mask=edge_mask, check_isolation=False
        )

    def run_snap_full():
        trainer.snapshot(
            gt_full, probe_loader, edge_sample_mask=edge_mask, check_isolation=False
        )

    def run_snap_iso():
        trainer.snapshot(
            gt_full, probe_loader, edge_sample_mask=edge_mask, check_isolation=True
        )

    def run_iso_only():
        pre = [capture_signal_traces(s) for s in signals]
        post = [capture_signal_traces(s) for s in signals]
        # equality check only
        for a, b in zip(pre, post):
            for k in a:
                if torch.is_tensor(a[k]) and torch.is_tensor(b[k]):
                    assert torch.equal(a[k], b[k])

    def run_align():
        from neuronca.signals.gevb_eligibility import GEVBVectorEligibility as G

        gevb = next(s for s in signals if isinstance(s, G))
        compute_gevb_alignment(
            model, gevb, probe_loader, device=device, max_batches=4
        )

    t_freeze = section("freeze_local_scores", freeze, reps=3)
    t_grad = section("GT-1 GradientSaliency.compute", run_grad, reps=3)
    t_iso = section("isolation capture+assert only", run_iso_only, reps=3)
    t_align = section("GEVB FA alignment (4 batches)", run_align, reps=2)
    t_snap_g = section("snapshot(grad_abs only)", run_snap_grad_only, reps=2)

    results = {
        "device": device,
        "probe_size": args.probe_size,
        "batch_size": args.batch_size,
        "n_neuron": n_neuron,
        "n_edge": n_edge,
        "micro": {
            "mean_probe_loss_s": t_probe,
            "hook_reg_x448_s": t_hooks,
            "one_neuron_unit_s": t_one_neuron,
            "one_edge_unit_s": t_one_edge,
            "extrapolated_neuron_s": n_neuron * t_one_neuron,
            "extrapolated_edge_s": n_edge * t_one_edge,
            "extrapolated_all_probe_dominant_s": (n_neuron + n_edge) * t_probe,
        },
        "sections": {
            "freeze_s": t_freeze,
            "grad_saliency_s": t_grad,
            "isolation_assert_s": t_iso,
            "gevb_alignment_s": t_align,
            "snapshot_grad_only_s": t_snap_g,
        },
    }

    t_abl = None
    t_snap_full = None
    t_snap_iso = None
    if not args.skip_full_ablation:
        print("\n=== Full ablation / full snapshot (expensive; 1–2 reps) ===")
        t_abl = section("GT-2 AblationImportance.compute (FULL)", run_abl, reps=1)
        t_snap_full = section("snapshot(grad+ablation) NO iso", run_snap_full, reps=1)
        t_snap_iso = section("snapshot(grad+ablation) WITH iso", run_snap_iso, reps=1)
        results["sections"]["ablation_full_s"] = t_abl
        results["sections"]["snapshot_full_no_iso_s"] = t_snap_full
        results["sections"]["snapshot_full_with_iso_s"] = t_snap_iso
        results["sections"]["iso_overhead_on_full_s"] = t_snap_iso - t_snap_full
        results["fraction_ablation_of_full"] = t_abl / t_snap_full if t_snap_full else None
        results["fraction_probe_of_one_unit"] = t_probe / t_one_neuron if t_one_neuron else None
    else:
        print("\n(skipping full ablation compute; --skip-full-ablation)")

    # =====================================================================
    # 3) cProfile on full snapshot (or grad-only if skip)
    # =====================================================================
    print("\n=== cProfile (cumulative top 30) ===")
    pr = cProfile.Profile()
    pr.enable()
    if args.skip_full_ablation:
        run_snap_grad_only()
    else:
        run_snap_full()
    pr.disable()
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    stats.print_stats(40)
    cprof_text = s.getvalue()
    print(cprof_text)
    (out_dir / "cprofile_cumulative.txt").write_text(cprof_text)

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(30)
    tottime_text = s2.getvalue()
    print("=== cProfile (tottime top 30) ===")
    print(tottime_text)
    (out_dir / "cprofile_tottime.txt").write_text(tottime_text)

    # Focused name counts for the three hypotheses
    print("=== cProfile name hits (hypothesis targets) ===")
    st = pstats.Stats(pr)
    targets = (
        "mean_probe_loss",
        "register_forward_hook",
        "remove",
        "_ablate_neurons",
        "_ablate_edges",
        "compute",
        "forward",
        "snapshot",
        "signals_disabled",
    )
    # pstats doesn't have easy filter API; scan stats
    hits = []
    for (filename, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if any(t in func or t in filename for t in targets):
            hits.append((ct, tt, nc, func, filename.split("/")[-1], line))
    hits.sort(reverse=True)
    for ct, tt, nc, func, fn, line in hits[:25]:
        print(f"  cum={ct:8.3f}s  tot={tt:8.3f}s  ncalls={nc:6d}  {fn}:{line} {func}")

    # =====================================================================
    # 4) torch.profiler
    # =====================================================================
    print("\n=== torch.profiler (CPU" + ("+CUDA" if device == "cuda" else "") + ") ===")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiled = run_snap_grad_only if args.skip_full_ablation else run_snap_full
    # one warm
    profiled()
    if device == "cuda":
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        profiled()
        if device == "cuda":
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    print(table)
    (out_dir / "torch_profiler.txt").write_text(table)

    # Also group by "aten::" vs python
    key_avgs = prof.key_averages()
    cpu_self = {}
    for e in key_avgs:
        key = e.key
        bucket = "aten" if key.startswith("aten::") else (
            "autograd" if "autograd" in key or "Backward" in key else "other"
        )
        cpu_self[bucket] = cpu_self.get(bucket, 0.0) + e.self_cpu_time_total
    total_us = sum(cpu_self.values()) or 1.0
    print("torch.profiler self_cpu bucket fractions:")
    for b, us in sorted(cpu_self.items(), key=lambda x: -x[1]):
        print(f"  {b:12s}  {us/1000:10.2f} ms  ({100*us/total_us:.1f}%)")
    results["torch_profiler_cpu_buckets_us"] = cpu_self

    # =====================================================================
    # Summary verdict for the three hypotheses
    # =====================================================================
    print("\n=== Hypothesis summary ===")
    if t_abl is not None:
        print(
            f"Full ablation wall: {t_abl:.1f}s | "
            f"full snapshot: {t_snap_full:.1f}s | "
            f"ablation share: {100*t_abl/t_snap_full:.1f}%"
        )
    print(
        f"Per-unit: probe={t_probe*1000:.1f}ms  "
        f"neuron_unit={t_one_neuron*1000:.1f}ms  "
        f"edge_unit={t_one_edge*1000:.1f}ms  "
        f"hooks_x448={t_hooks*1000:.2f}ms"
    )
    probe_frac_neuron = t_probe / t_one_neuron if t_one_neuron > 0 else float("nan")
    print(
        f"Probe fraction of one neuron-ablation unit: {100*probe_frac_neuron:.1f}% "
        f"(if ~100%, Python/hook overhead is negligible vs forward)"
    )
    hook_per = t_hooks / 448
    print(
        f"Hook reg/remove per call: {hook_per*1e6:.1f} µs  |  "
        f"as fraction of one probe: {100*hook_per/t_probe:.4f}%"
    )
    print(
        f"Isolation assert: {t_iso*1000:.1f}ms | "
        f"GEVB alignment: {t_align*1000:.1f}ms | "
        f"grad GT: {t_grad*1000:.1f}ms"
    )

    (out_dir / "profile_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nWrote artifacts under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
