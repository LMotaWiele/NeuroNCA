#!/usr/bin/env python3
"""Analyze t_star_abs vs t_star_sustained on frozen B1/B3 time_lag.csv files.

Pure offline analysis — does not re-run experiments or change verdict gates.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.eval.time_lag import analyze_lag_curve  # noqa: E402

HIST = _REPO / "results" / "historical" / "pre_patch_rng_isolation"
EXPERIMENTS = {
    "B1_label_permutation": HIST / "phase0_B1_label_permutation",
    "B3_pixel_permutation": HIST / "phase0_B3_pixel_permutation",
}
SEEDS = (0, 1, 2)
THRESHOLD = 0.60
MAX_T = 500.0
OUT_DIR = _REPO / "results" / "phase0_diagnostics"
GT = "grad_abs"


def _load_lag_curves(path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(local, gt) → sorted list of (steps_after_switch, spearman)."""
    curves: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["steps_after_switch"])
                rho = float(row["spearman"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(t):
                continue
            key = (str(row["local_signal"]), str(row["gt_measure"]))
            # Deduplicate by t (CSV repeats summary cols on every row)
            curves[key].append((t, rho if math.isfinite(rho) else float("nan")))
    # unique by t (last write wins)
    out: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for key, pts in curves.items():
        by_t: dict[float, float] = {}
        for t, r in pts:
            by_t[t] = r
        out[key] = sorted(by_t.items(), key=lambda p: p[0])
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    disagreements: list[dict] = []

    for exp_name, root in EXPERIMENTS.items():
        for seed in SEEDS:
            lag_path = root / "task_shift" / f"seed_{seed}" / "time_lag.csv"
            if not lag_path.is_file():
                print(f"missing {lag_path}", flush=True)
                continue
            curves = _load_lag_curves(lag_path)
            for (local, gt), pts in sorted(curves.items()):
                if gt != GT:
                    continue
                times = [p[0] for p in pts]
                rhos = [p[1] for p in pts]
                rec = analyze_lag_curve(
                    times, rhos, threshold=THRESHOLD, max_t=MAX_T
                )
                row = {
                    "experiment": exp_name,
                    "seed": seed,
                    "local_signal": local,
                    "gt_measure": gt,
                    "t_star_abs": rec.t_star_abs if math.isfinite(rec.t_star_abs) else None,
                    "t_star_sustained": (
                        rec.t_star_sustained
                        if math.isfinite(rec.t_star_sustained)
                        else None
                    ),
                    "recovered_abs": rec.recovered_abs,
                    "recovered_sustained": rec.recovered_sustained,
                    "rho_at_window_end": rec.rho_at_window_end,
                    "rho_max_in_window": rec.rho_max_in_window,
                    "disagree": (
                        rec.recovered_abs != rec.recovered_sustained
                        or (
                            math.isfinite(rec.t_star_abs)
                            and math.isfinite(rec.t_star_sustained)
                            and abs(rec.t_star_abs - rec.t_star_sustained) > 1e-9
                        )
                        or (
                            math.isfinite(rec.t_star_abs)
                            and not math.isfinite(rec.t_star_sustained)
                        )
                    ),
                }
                rows.append(row)
                if row["disagree"]:
                    disagreements.append(row)

    # Write CSV
    csv_path = OUT_DIR / "t_star_abs_vs_sustained.csv"
    cols = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown report
    lines = [
        "# Sustained recovery diagnostic (`t_star_abs` vs `t_star_sustained`)",
        "",
        "Offline analysis of frozen B1/B3 `time_lag.csv` (edge Spearman vs "
        f"`{GT}`, threshold={THRESHOLD}, window={int(MAX_T)}).",
        "",
        "`t_star_abs`: first lag t with ρ ≥ threshold (existing primary).",
        "`t_star_sustained`: first t after which **all** subsequent logged ρ stay "
        "≥ threshold through t=500 (diagnostic only; not a pass/fail gate).",
        "",
        "## Full table",
        "",
        "| exp | seed | candidate | t*_abs | t*_sustained | rec_abs | rec_sust | disagree |",
        "|-----|-----:|-----------|-------:|-------------:|--------:|---------:|----------|",
    ]
    for r in rows:
        ta = "inf" if r["t_star_abs"] is None else str(int(r["t_star_abs"]))
        ts = "inf" if r["t_star_sustained"] is None else str(int(r["t_star_sustained"]))
        lines.append(
            f"| {r['experiment']} | {r['seed']} | {r['local_signal']} | "
            f"{ta} | {ts} | {r['recovered_abs']} | {r['recovered_sustained']} | "
            f"{'**yes**' if r['disagree'] else 'no'} |"
        )
    lines.extend(["", "## Disagreements (call-outs)", ""])
    if not disagreements:
        lines.append("_None._")
    else:
        lines.append(
            "| exp | seed | candidate | t*_abs | t*_sustained | note |"
        )
        lines.append("|-----|-----:|-----------|-------:|-------------:|------|")
        for r in disagreements:
            ta = "inf" if r["t_star_abs"] is None else str(int(r["t_star_abs"]))
            ts = "inf" if r["t_star_sustained"] is None else str(int(r["t_star_sustained"]))
            note = (
                "first-crossing then collapse (sustained later or never)"
                if r["recovered_abs"] and not r["recovered_sustained"]
                else (
                    "sustained later than first crossing"
                    if r["recovered_abs"] and r["recovered_sustained"]
                    else "other"
                )
            )
            lines.append(
                f"| {r['experiment']} | {r['seed']} | `{r['local_signal']}` | "
                f"{ta} | {ts} | {note} |"
            )
    lines.extend(
        [
            "",
            "## Expected pattern",
            "",
            "`activation` under B1 (and often B3) is expected to disagree: "
            "`t_star_abs=0` from frozen-weight coincidence, then collapse, "
            "with sustained recovery much later or never.",
            "",
            f"CSV: `{csv_path.relative_to(_REPO)}`",
            "",
        ]
    )
    md_path = OUT_DIR / "sustained_recovery_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"disagreements: {len(disagreements)} / {len(rows)}")
    for r in disagreements:
        print(
            f"  {r['experiment']} seed{r['seed']} {r['local_signal']}: "
            f"abs={r['t_star_abs']} sust={r['t_star_sustained']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
