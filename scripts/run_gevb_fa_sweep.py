#!/usr/bin/env python3
"""GEVB-RF FA diagnostic sweep (Phase 0 refinement).

Grid: lr ∈ {0.001, 0.01, 0.05} × B_ℓ init_scale ∈ {0.1, 1.0, 10.0} of default
std (= 1/sqrt(C)), 3 seeds, stationary 20 epochs. Secondary: epochs=40 at
defaults only.

Pre-registered interpretation: see module docstring at bottom / results README.
Does not modify frozen B1/B3 archives or threshold files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neuronca.eval.gevb_alignment_verdict import (  # noqa: E402
    ALIGNMENT_CONVERGED,
    ALIGNMENT_INSUFFICIENT_DATA,
    ALIGNMENT_PARTIAL,
    ALIGNMENT_STALLED,
    GevbAlignmentThresholds,
    collect_alignment_from_results,
    decide_alignment_status,
)
from neuronca.experiments.stationary import run_stationary  # noqa: E402
from neuronca.io.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered sweep constants (do not retune after seeing results)
# ---------------------------------------------------------------------------

LRS = (0.001, 0.01, 0.05)
B_SCALES = (0.1, 1.0, 10.0)
SEEDS = (0, 1, 2)
DEFAULT_LR = 0.01
DEFAULT_B_SCALE = 1.0
DEFAULT_EPOCHS = 20
DOUBLE_EPOCHS = 40
SANITY_ACC_FLOOR = 0.90
STATIONARY_RHO_MEAN = 0.60
STATIONARY_RHO_FLOOR = 0.50
STATIONARY_RHO_FLOOR_MIN_COUNT = 2
GEVB_NAME = "eligibility_gevb_rf"

CELL_INCONCLUSIVE = "INCONCLUSIVE"
CELL_CONVERGED = "CONVERGED"
CELL_STALLED = "STALLED"
CELL_PARTIAL = "PARTIAL"

OVERALL_SALVAGEABLE = "Salvageable"
OVERALL_NOT_SALVAGEABLE = "Not salvageable at this architecture/task scale"
OVERALL_INCONCLUSIVE = "Inconclusive"


def _deep_set(cfg: dict[str, Any], *path: str, value: Any) -> None:
    cur: dict[str, Any] = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _cell_id(lr: float, b_scale: float, epochs: int) -> str:
    def fmt(x: float) -> str:
        s = f"{x:g}"
        return s.replace(".", "p")

    return f"lr{fmt(lr)}_bscale{fmt(b_scale)}_ep{int(epochs)}"


def _build_cell_config(
    base: Mapping[str, Any],
    *,
    lr: float,
    b_scale: float,
    epochs: int,
    results_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base))
    _deep_set(cfg, "train", "lr", value=float(lr))
    _deep_set(cfg, "train", "epochs", value=int(epochs))
    _deep_set(cfg, "broadcast", "gevb_feedback", "init_scale", value=float(b_scale))
    _deep_set(cfg, "broadcast", "gevb_feedback", "init", value="gaussian_scaled")
    _deep_set(cfg, "experiment", "conditions", value=["stationary"])
    _deep_set(cfg, "experiment", "seeds", value=list(SEEDS))
    _deep_set(cfg, "experiment", "output_dir", value=str(results_root))
    # Stationary only — no task switch for FA late diagnostic.
    _deep_set(cfg, "tasks", "shift_type", value=None)
    return cfg


def _gevb_rho_from_summary(summary: Mapping[str, Any]) -> float:
    late = summary.get("late_edge_grad_abs") or {}
    per = late.get("per_candidate") or {}
    v = per.get(GEVB_NAME)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _run_cell(
    base_cfg: Mapping[str, Any],
    *,
    lr: float,
    b_scale: float,
    epochs: int,
    sweep_root: Path,
) -> dict[str, Any]:
    cid = _cell_id(lr, b_scale, epochs)
    results_root = sweep_root / "cells" / cid
    results_root.mkdir(parents=True, exist_ok=True)
    cfg = _build_cell_config(
        base_cfg, lr=lr, b_scale=b_scale, epochs=epochs, results_root=results_root
    )

    print(f"[gevb_fa_sweep] START {cid}", flush=True)
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"[gevb_fa_sweep]   seed={seed} lr={lr} b_scale={b_scale} ep={epochs}", flush=True)
        summary = run_stationary(cfg, results_root, seed=int(seed), smoke=False)
        acc = summary.get("final_train_acc")
        loss = summary.get("final_train_loss")
        try:
            acc_f = float(acc) if acc is not None else float("nan")
        except (TypeError, ValueError):
            acc_f = float("nan")
        rho = _gevb_rho_from_summary(summary)
        seed_rows.append(
            {
                "seed": int(seed),
                "final_train_acc": acc_f,
                "final_train_loss": loss,
                "gevb_rho_late_edge_grad_abs": rho,
                "sanity_pass": bool(math.isfinite(acc_f) and acc_f >= SANITY_ACC_FLOOR),
            }
        )

    late_by_seed, early_by_seed = collect_alignment_from_results(
        results_root, list(SEEDS), condition="stationary"
    )
    # Per-seed FA layer means at mid too (for tables)
    mid_by_seed: dict[int, float] = {}
    for seed in SEEDS:
        path = results_root / "stationary" / f"seed_{int(seed)}" / "gevb_alignment.csv"
        from neuronca.eval.gevb_alignment_verdict import load_seed_tag_layer_means

        mid_by_seed[int(seed)] = load_seed_tag_layer_means(path, tag="mid")

    thr = GevbAlignmentThresholds()
    fa_verdict = decide_alignment_status(late_by_seed, early_by_seed, thresholds=thr)

    n_sane = sum(1 for r in seed_rows if r["sanity_pass"])
    all_sane = n_sane == len(SEEDS)
    any_sane = n_sane > 0

    # FA status restricted to sanity-passing seeds when some fail
    if not all_sane and any_sane:
        late_sane = {
            int(r["seed"]): late_by_seed[int(r["seed"])]
            for r in seed_rows
            if r["sanity_pass"] and int(r["seed"]) in late_by_seed
        }
        early_sane = {
            int(r["seed"]): early_by_seed.get(int(r["seed"]), float("nan"))
            for r in seed_rows
            if r["sanity_pass"]
        }
        fa_verdict_sane = decide_alignment_status(
            late_sane, early_sane, thresholds=thr
        )
    else:
        fa_verdict_sane = fa_verdict

    gevb_rhos = [
        float(r["gevb_rho_late_edge_grad_abs"])
        for r in seed_rows
        if r["sanity_pass"] and math.isfinite(float(r["gevb_rho_late_edge_grad_abs"]))
    ]
    mean_rho = float(sum(gevb_rhos) / len(gevb_rhos)) if gevb_rhos else float("nan")
    n_above_floor = sum(1 for x in gevb_rhos if x >= STATIONARY_RHO_FLOOR)
    rho_pass = (
        math.isfinite(mean_rho)
        and mean_rho >= STATIONARY_RHO_MEAN
        and n_above_floor >= STATIONARY_RHO_FLOOR_MIN_COUNT
    )

    # Cell-level label per Section 3
    if not any_sane:
        cell_label = CELL_INCONCLUSIVE
        cell_reason = "all seeds failed training sanity floor"
    elif not all_sane:
        # Partial training success: still score FA on sane seeds; flag partial
        cell_label = CELL_INCONCLUSIVE
        cell_reason = (
            f"only {n_sane}/{len(SEEDS)} seeds passed sanity floor; "
            f"FA_on_sane={fa_verdict_sane.alignment_status}"
        )
    else:
        st = fa_verdict.alignment_status
        if st == ALIGNMENT_CONVERGED:
            cell_label = CELL_CONVERGED
            cell_reason = "FA CONVERGED on all sanity-passing seeds"
        elif st == ALIGNMENT_STALLED:
            cell_label = CELL_STALLED
            cell_reason = "FA STALLED"
        elif st == ALIGNMENT_PARTIAL:
            cell_label = CELL_PARTIAL
            cell_reason = "FA PARTIAL"
        else:
            cell_label = CELL_INCONCLUSIVE
            cell_reason = f"FA {st}"

    salvage_point = (
        all_sane
        and fa_verdict.alignment_status == ALIGNMENT_CONVERGED
        and rho_pass
    )

    cell = {
        "cell_id": cid,
        "lr": float(lr),
        "b_init_scale": float(b_scale),
        "epochs": int(epochs),
        "results_root": str(results_root),
        "seeds": seed_rows,
        "n_sane_seeds": n_sane,
        "all_sane": all_sane,
        "sanity_acc_floor": SANITY_ACC_FLOOR,
        "fa_alignment_status": fa_verdict.alignment_status,
        "fa_mean_late_deg": fa_verdict.mean_late_fa_deg,
        "fa_mean_early_deg": fa_verdict.mean_early_fa_deg,
        "fa_mean_improvement_deg": fa_verdict.mean_improvement_deg,
        "fa_seeds_below_floor": fa_verdict.seeds_below_floor,
        "fa_per_seed_late": {str(k): v for k, v in fa_verdict.per_seed_late_layer_mean.items()},
        "fa_per_seed_early": {str(k): early_by_seed.get(int(k), float("nan")) for k in SEEDS},
        "fa_per_seed_mid": {str(k): mid_by_seed.get(int(k), float("nan")) for k in SEEDS},
        "gevb_rho_mean": mean_rho,
        "gevb_rho_n_above_floor": n_above_floor,
        "gevb_rho_pass_stationary_bar": rho_pass,
        "cell_label": cell_label,
        "cell_reason": cell_reason,
        "salvage_point": salvage_point,
    }
    print(
        f"[gevb_fa_sweep] DONE {cid} label={cell_label} FA={fa_verdict.alignment_status} "
        f"mean_late_FA={fa_verdict.mean_late_fa_deg:.2f}° mean_ρ={mean_rho:.4f} "
        f"sane={n_sane}/{len(SEEDS)} salvage={salvage_point}",
        flush=True,
    )
    return cell


def _overall_verdict(cells: list[dict[str, Any]]) -> dict[str, Any]:
    salvage = [c for c in cells if c.get("salvage_point")]
    valid = [c for c in cells if c.get("all_sane")]
    any_sane = [c for c in cells if c.get("n_sane_seeds", 0) > 0]
    converged_valid = [
        c for c in valid if c.get("fa_alignment_status") == ALIGNMENT_CONVERGED
    ]

    if salvage:
        overall = OVERALL_SALVAGEABLE
        detail = (
            "At least one sanity-passing grid point has FA CONVERGED and "
            "GEVB-RF ρ clears the stationary bar. "
            f"Salvage points: {[c['cell_id'] for c in salvage]}"
        )
    elif not any_sane:
        overall = OVERALL_INCONCLUSIVE
        detail = "No seed/configuration passed the training sanity floor."
    elif not valid:
        # some seeds sane but no full cell all_sane
        if any(
            c.get("fa_alignment_status") == ALIGNMENT_CONVERGED
            for c in any_sane
        ):
            overall = OVERALL_INCONCLUSIVE
            detail = (
                "Hints of FA convergence only appear on cells that are not fully "
                "sanity-floor-passing; no complete valid CONVERGED+ρ cell."
            )
        else:
            overall = OVERALL_INCONCLUSIVE
            detail = "No fully sanity-passing cells; cannot claim salvage or architecture failure."
    elif not converged_valid:
        # All valid cells stalled or partial
        if all(
            c.get("fa_alignment_status") in (ALIGNMENT_STALLED, ALIGNMENT_PARTIAL, ALIGNMENT_INSUFFICIENT_DATA)
            for c in valid
        ):
            overall = OVERALL_NOT_SALVAGEABLE
            detail = (
                "No sanity-floor-passing grid point (including epoch-doubled arm if present) "
                "shows FA CONVERGED. GEVB-RF failure is not a single-default hyperparameter accident "
                "within this grid."
            )
        else:
            overall = OVERALL_INCONCLUSIVE
            detail = "Valid cells present but no CONVERGED FA."
    else:
        # FA converged somewhere but ρ bar failed → not fully salvageable for Phase 0 candidate
        overall = OVERALL_INCONCLUSIVE
        detail = (
            f"FA CONVERGED at {[c['cell_id'] for c in converged_valid]} but GEVB-RF ρ "
            "did not clear the stationary bar at those points (or only partial sanity). "
            "Not Salvageable under Section 3 dual requirement (FA + ρ)."
        )
        # Spec: Salvageable requires FA CONVERGED AND ρ bar. So FA alone is not Salvageable.
        # If no salvage points, and all valid are not CONVERGED → Not salvageable.
        # If some CONVERGED without ρ → Inconclusive vs Not salvageable?
        # Spec: "Salvageable: at least one valid grid point shows CONVERGED alignment *and* GEVB-RF ρ clears"
        # "Not salvageable: no grid point converges" — meaning FA converges
        # So FA converges but ρ fails → not Salvageable; not "no grid point converges" either → Inconclusive is OK
        # Or we could say Not salvageable for candidate utility. Spec's Not salvageable is about FA convergence.
        if not salvage and not converged_valid:
            overall = OVERALL_NOT_SALVAGEABLE

    # Re-apply cleanly
    if salvage:
        overall = OVERALL_SALVAGEABLE
        detail = (
            "At least one valid grid point: FA CONVERGED and GEVB-RF ρ passes stationary bar. "
            f"Points: {[c['cell_id'] for c in salvage]}"
        )
    elif valid and not converged_valid:
        overall = OVERALL_NOT_SALVAGEABLE
        detail = (
            "No sanity-floor-passing combination shows FA CONVERGED "
            f"(statuses: {sorted({c['fa_alignment_status'] for c in valid})})."
        )
    elif valid and converged_valid and not salvage:
        overall = OVERALL_INCONCLUSIVE
        detail = (
            "FA CONVERGED at some valid points but GEVB-RF ρ did not clear the stationary bar "
            f"there: {[c['cell_id'] for c in converged_valid]}."
        )
    elif not valid:
        overall = OVERALL_INCONCLUSIVE
        detail = "No fully sanity-floor-passing cells to interpret."

    return {
        "overall_verdict": overall,
        "detail": detail,
        "salvage_points": [c["cell_id"] for c in salvage],
        "converged_fa_valid_cells": [c["cell_id"] for c in converged_valid],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("results/phase0_gevb_fa_sweep"),
        help="Root for all sweep cells and aggregate outputs",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stationary.yaml"),
        help="Base config (merged with defaults)",
    )
    p.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip 3×3 grid (only epoch-doubled arm)",
    )
    p.add_argument(
        "--skip-double-epochs",
        action="store_true",
        help="Skip 40-epoch secondary arm",
    )
    args = p.parse_args(argv)

    sweep_root = Path(args.sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config, defaults_path=Path("configs/default.yaml"), smoke=False)
    cells: list[dict[str, Any]] = []

    if not args.skip_grid:
        for lr in LRS:
            for b_scale in B_SCALES:
                cell = _run_cell(
                    base,
                    lr=lr,
                    b_scale=b_scale,
                    epochs=DEFAULT_EPOCHS,
                    sweep_root=sweep_root,
                )
                cells.append(cell)
                # Persist incremental
                (sweep_root / "cells_partial.json").write_text(
                    json.dumps(cells, indent=2) + "\n", encoding="utf-8"
                )

    double_cell = None
    if not args.skip_double_epochs:
        double_cell = _run_cell(
            base,
            lr=DEFAULT_LR,
            b_scale=DEFAULT_B_SCALE,
            epochs=DOUBLE_EPOCHS,
            sweep_root=sweep_root,
        )
        cells.append(double_cell)

    # Grid table (20-ep only)
    grid_cells = [c for c in cells if int(c["epochs"]) == DEFAULT_EPOCHS]
    table: dict[str, dict[str, str]] = {}
    for lr in LRS:
        row: dict[str, str] = {}
        for b in B_SCALES:
            match = next(
                (
                    c
                    for c in grid_cells
                    if abs(c["lr"] - lr) < 1e-12 and abs(c["b_init_scale"] - b) < 1e-12
                ),
                None,
            )
            row[str(b)] = match["cell_label"] if match else "MISSING"
        table[str(lr)] = row

    overall = _overall_verdict(cells)

    summary = {
        "sweep": "gevb_fa_diagnostic_phase0_refinement",
        "pre_registered": {
            "lrs": list(LRS),
            "b_init_scales": list(B_SCALES),
            "seeds": list(SEEDS),
            "default_epochs": DEFAULT_EPOCHS,
            "double_epochs": DOUBLE_EPOCHS,
            "sanity_acc_floor": SANITY_ACC_FLOOR,
            "stationary_rho_mean": STATIONARY_RHO_MEAN,
            "stationary_rho_floor": STATIONARY_RHO_FLOOR,
            "fa_thresholds": "configs/default.yaml eval.thresholds gevb_fa_*",
        },
        "grid_table_lr_x_bscale": table,
        "cells": cells,
        "double_epoch_arm": double_cell,
        "overall": overall,
    }

    out_json = sweep_root / "sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Human-readable markdown
    lines = [
        "# GEVB-RF FA diagnostic sweep",
        "",
        f"**Overall verdict:** {overall['overall_verdict']}",
        "",
        overall["detail"],
        "",
        "## Grid (20 epochs): cell = FA/sanity label",
        "",
        "| LR \\\\ B_scale | " + " | ".join(str(b) for b in B_SCALES) + " |",
        "|---|" + "|".join(["---"] * len(B_SCALES)) + "|",
    ]
    for lr in LRS:
        row = table[str(lr)]
        lines.append(
            "| " + str(lr) + " | " + " | ".join(row[str(b)] for b in B_SCALES) + " |"
        )
    lines.extend(
        [
            "",
            "## Per-cell metrics",
            "",
            "| cell_id | lr | B_scale | ep | sane | FA status | mean_late_FA° | mean GEVB ρ | ρ_pass | salvage |",
            "|---|---:|---:|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for c in cells:
        lines.append(
            f"| {c['cell_id']} | {c['lr']} | {c['b_init_scale']} | {c['epochs']} | "
            f"{c['n_sane_seeds']}/{len(SEEDS)} | {c['fa_alignment_status']} | "
            f"{c['fa_mean_late_deg']:.2f} | {c['gevb_rho_mean']:.4f} | "
            f"{c['gevb_rho_pass_stationary_bar']} | {c['salvage_point']} |"
        )
    if double_cell:
        lines.extend(
            [
                "",
                "## Secondary arm (40 epochs, default LR × B_scale)",
                "",
                f"- cell: `{double_cell['cell_id']}`",
                f"- FA: {double_cell['fa_alignment_status']} "
                f"(mean late {double_cell['fa_mean_late_deg']:.2f}°)",
                f"- GEVB ρ mean: {double_cell['gevb_rho_mean']:.4f}",
                f"- label: {double_cell['cell_label']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Sanity floor",
            "",
            f"`final_train_acc` ≥ {SANITY_ACC_FLOOR}; failing cells are INCONCLUSIVE.",
            "",
            "Artifacts: `cells/<cell_id>/`, `sweep_summary.json`.",
            "",
        ]
    )
    (sweep_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Flat CSV for plotting
    csv_path = sweep_root / "cells_flat.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "lr",
                "b_init_scale",
                "epochs",
                "seed",
                "final_train_acc",
                "sanity_pass",
                "gevb_rho",
                "fa_late_mean",
                "fa_early_mean",
                "fa_status",
                "cell_label",
                "salvage_point",
            ],
        )
        w.writeheader()
        for c in cells:
            for s in c["seeds"]:
                w.writerow(
                    {
                        "cell_id": c["cell_id"],
                        "lr": c["lr"],
                        "b_init_scale": c["b_init_scale"],
                        "epochs": c["epochs"],
                        "seed": s["seed"],
                        "final_train_acc": s["final_train_acc"],
                        "sanity_pass": s["sanity_pass"],
                        "gevb_rho": s["gevb_rho_late_edge_grad_abs"],
                        "fa_late_mean": c["fa_per_seed_late"].get(str(s["seed"])),
                        "fa_early_mean": c["fa_per_seed_early"].get(str(s["seed"])),
                        "fa_status": c["fa_alignment_status"],
                        "cell_label": c["cell_label"],
                        "salvage_point": c["salvage_point"],
                    }
                )

    print(f"\n=== OVERALL: {overall['overall_verdict']} ===", flush=True)
    print(overall["detail"], flush=True)
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
