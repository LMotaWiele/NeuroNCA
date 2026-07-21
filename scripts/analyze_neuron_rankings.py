#!/usr/bin/env python3
"""Extract neuron-level overall rankings from existing correlations.csv files."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUT_DIR = _REPO / "results" / "phase0_diagnostics"

EXPERIMENTS = {
    "B1_label_permutation_stationary": {
        "root": _REPO
        / "results/historical/pre_patch_rng_isolation/phase0_B1_label_permutation",
        "condition": "stationary",
        "tag": "late",
    },
    "B3_pixel_permutation_stationary": {
        "root": _REPO
        / "results/historical/pre_patch_rng_isolation/phase0_B3_pixel_permutation",
        "condition": "stationary",
        "tag": "late",
    },
    "truncated_6ep_postpatch_stationary": {
        "root": _REPO / "results/phase0_truncated_epochs",
        "condition": "stationary",
        "tag": "late",
    },
}
SEEDS = (0, 1, 2)
GT_MEASURES = ("grad_abs", "ablation_relu")
CANDIDATES = (
    "activation",
    "hebbian_corr",
    "eligibility_residual_energy",
    "eligibility_gevb_rf",
)


def _mean_rho(
    root: Path,
    *,
    condition: str,
    tag: str,
    granularity: str,
    gt: str,
) -> dict[str, float]:
    """Mean overall Spearman across seeds for each local_signal."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for seed in SEEDS:
        path = root / condition / f"seed_{seed}" / "correlations.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("snapshot_tag")) != tag:
                    continue
                if str(row.get("granularity")) != granularity:
                    continue
                if str(row.get("gt_measure")) != gt:
                    continue
                # overall rows: is_overall true or layer == overall
                is_ov = str(row.get("is_overall", "")).lower() in ("true", "1", "yes")
                if not is_ov and str(row.get("layer")) != "overall":
                    continue
                local = str(row.get("local_signal") or "")
                if local not in CANDIDATES:
                    continue
                try:
                    rho = float(row["spearman"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(rho):
                    continue
                sums[local] += rho
                counts[local] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def _rank(means: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(means.items(), key=lambda kv: -kv[1])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"experiments": {}}
    lines = [
        "# Neuron-level vs edge-level candidate rankings",
        "",
        "Source: existing `correlations.csv` (`is_overall` / layer=overall), "
        "mean Spearman across seeds 0–2, snapshot `late`, stationary condition.",
        "",
        "No new training runs. Disagreements in candidate **order** are flagged.",
        "",
    ]

    for exp_name, meta in EXPERIMENTS.items():
        root = Path(meta["root"])
        if not root.is_dir():
            lines.append(f"## {exp_name}\n\n_Missing root {root}_\n")
            continue
        exp_block: dict = {}
        lines.append(f"## {exp_name}\n")
        for gt in GT_MEASURES:
            edge = _mean_rho(
                root,
                condition=meta["condition"],
                tag=meta["tag"],
                granularity="edge",
                gt=gt,
            )
            neuron = _mean_rho(
                root,
                condition=meta["condition"],
                tag=meta["tag"],
                granularity="neuron",
                gt=gt,
            )
            re, rn = _rank(edge), _rank(neuron)
            order_e = [n for n, _ in re]
            order_n = [n for n, _ in rn]
            agree = order_e == order_n
            exp_block[gt] = {
                "edge_mean_rho": edge,
                "neuron_mean_rho": neuron,
                "edge_order": order_e,
                "neuron_order": order_n,
                "order_agree": agree,
            }
            flag = " **ORDER DISAGREE**" if not agree else ""
            lines.append(f"### GT = `{gt}`{flag}\n")
            lines.append(
                "| rank | edge candidate | edge ρ | neuron candidate | neuron ρ |"
            )
            lines.append("|-----:|----------------|-------:|------------------|---------:|")
            n_max = max(len(re), len(rn))
            for i in range(n_max):
                ce, re_rho = re[i] if i < len(re) else ("", float("nan"))
                cn, rn_rho = rn[i] if i < len(rn) else ("", float("nan"))
                lines.append(
                    f"| {i+1} | {ce} | {re_rho:.4f} | {cn} | {rn_rho:.4f} |"
                )
            lines.append("")
            lines.append(f"- edge order: `{order_e}`")
            lines.append(f"- neuron order: `{order_n}`")
            lines.append(f"- order agree: **{agree}**")
            lines.append("")
        report["experiments"][exp_name] = exp_block

    # Cross-flag
    any_disagree = any(
        not v.get("order_agree", True)
        for exp in report["experiments"].values()
        for v in exp.values()
    )
    lines.insert(
        5,
        f"**Any edge/neuron order disagreement:** {'YES — see flags below' if any_disagree else 'no'}",
    )
    lines.insert(6, "")

    md = OUT_DIR / "neuron_vs_edge_rankings.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "neuron_vs_edge_rankings.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {md}")
    print(f"any_disagree={any_disagree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
