#!/usr/bin/env python3
"""Phase 0 multi-seed orchestration CLI.

Runs stationary and/or task_shift across seeds, then writes
``{results_root}/aggregate/verdict.json`` **only** via
:func:`neuronca.eval.verdict.aggregate_verdict` and
:func:`neuronca.io.artifacts.write_verdict_json`.

Examples
--------
::

    python scripts/run_phase0.py --config configs/default.yaml
    python scripts/run_phase0.py --config configs/task_shift.yaml --condition task_shift
    python scripts/run_phase0.py --config configs/default.yaml --smoke
    python scripts/run_phase0.py --config configs/default.yaml --seeds 0,1,2 \\
        --results-root results/phase0 --condition both
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ensure repo-root imports work when invoked as ``python scripts/run_phase0.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neuronca.eval.verdict import (  # noqa: E402
    SeedLagCurve,
    VerdictResult,
    VerdictThresholds,
    aggregate_verdict,
    mean_rho_per_candidate,
    select_best_local,
)
from neuronca.experiments.stationary import run_stationary  # noqa: E402
from neuronca.experiments.task_shift import run_task_shift  # noqa: E402
from neuronca.io.artifacts import ensure_results_tree, write_verdict_json  # noqa: E402
from neuronca.io.config import extract_thresholds, load_config  # noqa: E402
from neuronca.io.ledger import load_existing_ledger  # noqa: E402

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_RESULTS_ROOT = "results/phase0"
DEFAULT_SEEDS = (0, 1, 2)
GT_MEASURE = "grad_abs"
STATIONARY_LATE_TAG = "late"
LAG_GT_MEASURE = "grad_abs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated seed list (e.g. ``\"0,1,2\"``)."""
    parts = [p.strip() for p in str(value).split(",")]
    seeds = [int(p) for p in parts if p]
    if not seeds:
        raise argparse.ArgumentTypeError("seeds list must not be empty")
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run Phase 0 multi-seed experiments (stationary / task_shift) and "
            "write aggregate/verdict.json via eval.verdict + write_verdict_json."
        ),
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--condition",
        choices=("stationary", "task_shift", "both"),
        default="both",
        help="Which condition(s) to run (default: both)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Apply CI smoke overrides (1 epoch, probe 128, seeds [0], short lag)",
    )
    p.add_argument(
        "--seeds",
        type=parse_seeds,
        default=None,
        help="Comma-separated seeds (default: from config / smoke, else 0,1,2)",
    )
    p.add_argument(
        "--results-root",
        default=None,
        help=f"Experiment output root (default: config experiment.output_dir or {DEFAULT_RESULTS_ROOT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config/CLI and print plan only; do not train or write verdict",
    )
    return p


def load_cli_config(config_path: str | Path, *, smoke: bool) -> dict[str, Any]:
    """Load resolved config; merge named overrides onto ``default.yaml`` when needed."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    if path.name in ("default.yaml", "defaults.yaml"):
        return load_config(defaults_path=path, smoke=smoke)
    defaults = path.parent / "default.yaml"
    if defaults.is_file():
        return load_config(path, defaults_path=defaults, smoke=smoke)
    return load_config(path, smoke=smoke)


def thresholds_from_config(config: Mapping[str, Any]) -> VerdictThresholds:
    """Map config eval thresholds → :class:`VerdictThresholds`."""
    thr = extract_thresholds(config)
    eval_cfg = dict(config.get("eval") or {})
    return VerdictThresholds(
        stationary_edge_spearman=float(thr.get("stationary_edge_spearman", 0.60)),
        stationary_seed_floor=float(thr.get("stationary_seed_floor", 0.50)),
        stationary_seed_floor_min_count=int(thr.get("stationary_seed_floor_min_count", 2)),
        shift_recovery_spearman=float(thr.get("shift_recovery_spearman", 0.60)),
        shift_recovery_max_steps=int(
            thr.get("shift_recovery_max_steps")
            or eval_cfg.get("lag_window_steps")
            or 500
        ),
        shift_recovery_min_seed_count=int(thr.get("shift_recovery_min_seed_count", 2)),
        fail_rho_ceiling=float(thr.get("fail_rho_ceiling", 0.30)),
        gt_shift_disagreement_ceiling=float(
            thr.get("gt_shift_disagreement_ceiling")
            or eval_cfg.get("gt_shift_disagreement_ceiling")
            or 0.90
        ),
    )


# ---------------------------------------------------------------------------
# Collectors (disk + in-memory summaries → aggregate_verdict inputs)
# ---------------------------------------------------------------------------


def _finite_or_nan(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def seed_rho_from_stationary_summary(summary: Mapping[str, Any]) -> dict[str, float]:
    """candidate → edge Spearman vs grad_abs at stationary ``late`` for one seed."""
    late = summary.get("late_edge_grad_abs") or {}
    per = late.get("per_candidate") or {}
    out: dict[str, float] = {}
    for cand, rho in per.items():
        if rho is None:
            continue
        f = _finite_or_nan(rho)
        if math.isfinite(f):
            out[str(cand)] = f
    return out


def seed_rho_from_correlations_csv(
    path: Path,
    *,
    snapshot_tag: str = STATIONARY_LATE_TAG,
    gt_measure: str = GT_MEASURE,
) -> dict[str, float]:
    """Fallback: parse overall edge Spearman rows from correlations.csv."""
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("snapshot_tag") != snapshot_tag
                or row.get("gt_measure") != gt_measure
                or row.get("granularity") != "edge"
            ):
                continue
            is_overall = str(row.get("is_overall", "")).lower() in ("1", "true", "yes")
            if not is_overall and row.get("layer") != "overall":
                continue
            name = str(row.get("local_signal") or "")
            if not name:
                continue
            rho = _finite_or_nan(row.get("spearman"))
            if math.isfinite(rho):
                out[name] = rho
    return out


def collect_seed_rho(
    results_root: Path,
    seeds: Sequence[int],
    *,
    stationary_summaries: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, dict[int, float]]:
    """candidate → {seed → ρ} at stationary late, edge vs grad_abs."""
    seed_rho: dict[str, dict[int, float]] = {}
    summaries = stationary_summaries or {}
    for seed in seeds:
        per_cand: dict[str, float] = {}
        if int(seed) in summaries:
            per_cand = seed_rho_from_stationary_summary(summaries[int(seed)])
        if not per_cand:
            corr = (
                results_root
                / "stationary"
                / f"seed_{int(seed)}"
                / "correlations.csv"
            )
            per_cand = seed_rho_from_correlations_csv(corr)
        for cand, rho in per_cand.items():
            seed_rho.setdefault(cand, {})[int(seed)] = float(rho)
    return seed_rho


def load_seed_lag_curve(
    results_root: Path,
    seed: int,
    local_signal: str,
    *,
    gt_measure: str = LAG_GT_MEASURE,
    summary: Mapping[str, Any] | None = None,
) -> SeedLagCurve | None:
    """Load lag series for ``local_signal`` vs GT from ``time_lag.csv``."""
    lag_path = results_root / "task_shift" / f"seed_{int(seed)}" / "time_lag.csv"
    if not lag_path.is_file():
        return None

    points: list[tuple[float, float]] = []
    rho_pre = float("nan")
    with lag_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("local_signal") != local_signal:
                continue
            if row.get("gt_measure") != gt_measure:
                continue
            t = _finite_or_nan(row.get("steps_after_switch"))
            rho = _finite_or_nan(row.get("spearman"))
            if math.isfinite(t):
                points.append((t, rho if math.isfinite(rho) else float("nan")))
            pre = row.get("rho_pre_switch")
            if pre not in (None, ""):
                pre_f = _finite_or_nan(pre)
                if math.isfinite(pre_f):
                    rho_pre = pre_f

    if not points and summary:
        # No dense lag rows — still record rho_pre from summary when available.
        pre_map = summary.get("rho_pre_switch") or {}
        if local_signal in pre_map and pre_map[local_signal] is not None:
            rho_pre = _finite_or_nan(pre_map[local_signal])

    if not points and not math.isfinite(rho_pre):
        return None

    points.sort(key=lambda p: p[0])
    # Deduplicate times (keep last)
    by_t: dict[float, float] = {}
    for t, r in points:
        by_t[t] = r
    times = sorted(by_t.keys())
    rhos = [by_t[t] for t in times]
    return SeedLagCurve(
        seed=int(seed),
        times=list(times),
        rhos=list(rhos),
        rho_pre_switch=rho_pre,
    )


def collect_lag_curves(
    results_root: Path,
    seeds: Sequence[int],
    best_local: str,
    *,
    task_shift_summaries: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, SeedLagCurve]:
    """seed → lag curve for the frozen global ``best_local`` id."""
    summaries = task_shift_summaries or {}
    out: dict[int, SeedLagCurve] = {}
    for seed in seeds:
        curve = load_seed_lag_curve(
            results_root,
            int(seed),
            best_local,
            summary=summaries.get(int(seed)),
        )
        if curve is not None:
            out[int(seed)] = curve
    return out


def collect_rho_shift(
    results_root: Path,
    seeds: Sequence[int],
    *,
    task_shift_summaries: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, float]:
    """seed → gt_shift_disagreement (Spearman I_pre vs I_switch_b0)."""
    summaries = task_shift_summaries or {}
    out: dict[int, float] = {}
    for seed in seeds:
        s = int(seed)
        rho: float | None = None
        if s in summaries:
            raw = summaries[s].get("gt_shift_disagreement")
            if raw is not None:
                f = _finite_or_nan(raw)
                if math.isfinite(f):
                    rho = f
        if rho is None:
            # Disk fallback: summary.json
            path = results_root / "task_shift" / f"seed_{s}" / "summary.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                raw = data.get("gt_shift_disagreement")
                if raw is not None:
                    f = _finite_or_nan(raw)
                    if math.isfinite(f):
                        rho = f
        if rho is not None:
            out[s] = rho
    return out


def resolve_experiment_id(
    results_root: Path,
    *,
    summaries: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Prefer ledger meta, then any run summary, else placeholder."""
    ledger = load_existing_ledger(results_root)
    if ledger is not None and ledger.experiment_id:
        return ledger.experiment_id
    for s in summaries:
        eid = s.get("experiment_id")
        if eid:
            return str(eid)
    meta_file = results_root / "meta" / "experiment_id.txt"
    if meta_file.is_file():
        text = meta_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "unknown"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def conditions_to_run(condition: str) -> tuple[bool, bool]:
    """Return ``(run_stationary, run_task_shift)``."""
    if condition == "stationary":
        return True, False
    if condition == "task_shift":
        return False, True
    if condition == "both":
        return True, True
    raise ValueError(f"unknown condition {condition!r}")


def aggregate_phase0_verdict(
    results_root: Path,
    seeds: Sequence[int],
    *,
    stationary_summaries: Mapping[int, Mapping[str, Any]] | None = None,
    task_shift_summaries: Mapping[int, Mapping[str, Any]] | None = None,
    thresholds: VerdictThresholds | None = None,
    experiment_id: str | None = None,
) -> tuple[VerdictResult, Path]:
    """Build multi-seed verdict **only** through verdict.py + write_verdict_json.

    Returns
    -------
    (VerdictResult, path)
        Aggregated result and path to ``aggregate/verdict.json``.
    """
    results_root = Path(results_root)
    ensure_results_tree(results_root)

    seed_rho = collect_seed_rho(
        results_root,
        seeds,
        stationary_summaries=stationary_summaries,
    )
    mean_rho = mean_rho_per_candidate(seed_rho)
    best = select_best_local(mean_rho)

    lag_curves: dict[int, SeedLagCurve] | None = None
    if best is not None:
        curves = collect_lag_curves(
            results_root,
            seeds,
            best,
            task_shift_summaries=task_shift_summaries,
        )
        if curves:
            lag_curves = curves

    rho_shift = collect_rho_shift(
        results_root,
        seeds,
        task_shift_summaries=task_shift_summaries,
    )

    result = aggregate_verdict(
        seed_rho,
        lag_curves=lag_curves,
        rho_shift_by_seed=rho_shift or None,
        thresholds=thresholds,
    )

    all_summaries: list[Mapping[str, Any]] = []
    if stationary_summaries:
        all_summaries.extend(stationary_summaries.values())
    if task_shift_summaries:
        all_summaries.extend(task_shift_summaries.values())
    eid = experiment_id or resolve_experiment_id(results_root, summaries=all_summaries)

    verdict_path = results_root / "aggregate" / "verdict.json"
    write_verdict_json(verdict_path, result, experiment_id=eid, overwrite=True)
    return result, verdict_path


def run_phase0(
    *,
    config: Mapping[str, Any],
    results_root: str | Path,
    seeds: Sequence[int],
    condition: str = "both",
    smoke: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run multi-seed conditions and write ``aggregate/verdict.json``.

    Parameters
    ----------
    config
        Resolved runtime config (smoke already merged if desired).
    results_root
        Experiment root (e.g. ``results/phase0``).
    seeds
        Seeds to run (e.g. ``[0, 1, 2]``).
    condition
        ``stationary`` | ``task_shift`` | ``both``.
    smoke
        Forwarded to runners (re-applies base smoke block when True).
    dry_run
        If True, skip training and aggregation; return plan only.
    """
    results_root = Path(results_root)
    cfg = dict(config)
    exp = dict(cfg.get("experiment") or {})
    exp["seeds"] = [int(s) for s in seeds]
    exp["output_dir"] = str(results_root)
    cfg["experiment"] = exp

    run_stat, run_shift = conditions_to_run(condition)
    plan = {
        "results_root": str(results_root),
        "seeds": [int(s) for s in seeds],
        "condition": condition,
        "run_stationary": run_stat,
        "run_task_shift": run_shift,
        "smoke": bool(smoke),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return plan

    ensure_results_tree(results_root)
    stationary_summaries: dict[int, dict[str, Any]] = {}
    task_shift_summaries: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        s = int(seed)
        if run_stat:
            print(f"[phase0] stationary seed={s} smoke={smoke}", flush=True)
            stationary_summaries[s] = run_stationary(
                cfg, results_root, seed=s, smoke=smoke
            )
        if run_shift:
            print(f"[phase0] task_shift seed={s} smoke={smoke}", flush=True)
            task_shift_summaries[s] = run_task_shift(
                cfg, results_root, seed=s, smoke=smoke
            )

    thr = thresholds_from_config(cfg)
    result, verdict_path = aggregate_phase0_verdict(
        results_root,
        seeds,
        stationary_summaries=stationary_summaries or None,
        task_shift_summaries=task_shift_summaries or None,
        thresholds=thr,
    )

    print(
        f"[phase0] verdict={result.verdict} phase1_go={result.phase1_go} "
        f"best_local={result.best_local} → {verdict_path}",
        flush=True,
    )
    return {
        **plan,
        "verdict": result.verdict,
        "phase1_go": result.phase1_go,
        "best_local": result.best_local,
        "stationary_pass": result.stationary_pass,
        "recovered": result.recovered,
        "weak_shift": result.weak_shift,
        "catastrophic_fail": result.catastrophic_fail,
        "verdict_path": str(verdict_path),
        "experiment_id": resolve_experiment_id(
            results_root,
            summaries=[*stationary_summaries.values(), *task_shift_summaries.values()],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_cli_config(args.config, smoke=args.smoke)

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        exp_seeds = (cfg.get("experiment") or {}).get("seeds")
        if isinstance(exp_seeds, list) and exp_seeds:
            seeds = [int(s) for s in exp_seeds]
        else:
            seeds = list(DEFAULT_SEEDS)

    if args.results_root is not None:
        results_root = Path(args.results_root)
    else:
        out = (cfg.get("experiment") or {}).get("output_dir") or DEFAULT_RESULTS_ROOT
        results_root = Path(out)

    summary = run_phase0(
        config=cfg,
        results_root=results_root,
        seeds=seeds,
        condition=args.condition,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Phase 0 complete. See {summary.get('verdict_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
