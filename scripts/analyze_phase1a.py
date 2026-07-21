#!/usr/bin/env python3
"""Aggregate Phase 1a diagnostics from a results root (origin splits included)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    report: dict[str, Any] = {"root": str(root), "stationary": [], "task_shift": [], "baseline": []}

    for cond in ("stationary", "task_shift", "baseline"):
        d = root / cond
        if not d.is_dir():
            continue
        for seed_dir in sorted(d.glob("seed_*")):
            summary_path = seed_dir / "summary.json"
            if not summary_path.is_file():
                continue
            s = _load(summary_path)
            report[cond].append(s)

    # Cross-seed stationary means when present.
    if report["stationary"]:
        accs = [s.get("test_acc") for s in report["stationary"] if s.get("test_acc") is not None]
        params = [s.get("active_params") for s in report["stationary"] if s.get("active_params") is not None]
        report["stationary_mean_acc"] = sum(accs) / len(accs) if accs else None
        report["stationary_mean_active_params"] = sum(params) / len(params) if params else None
        # Prune-by-origin pooled
        pi = sum(
            (s.get("prune_origin") or {}).get("cumulative_pruned_init_dense", 0)
            for s in report["stationary"]
        )
        pg = sum(
            (s.get("prune_origin") or {}).get("cumulative_pruned_incentive_grown", 0)
            for s in report["stationary"]
        )
        report["pooled_pruned_init_dense"] = pi
        report["pooled_pruned_incentive_grown"] = pg

    out = root / "aggregate" / "phase1a_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, required=True)
    args = p.parse_args(argv)
    report = analyze_root(Path(args.results_root))
    print(json.dumps({k: report[k] for k in report if k != "stationary"}, indent=2, default=str)[:2000])
    print(f"wrote {args.results_root}/aggregate/phase1a_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
