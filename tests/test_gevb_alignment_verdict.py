"""Pre-registered GEVB FA alignment verdict rules."""

from __future__ import annotations

import csv
from pathlib import Path

from neuronca.eval.gevb_alignment_verdict import (
    ALIGNMENT_CONVERGED,
    ALIGNMENT_INSUFFICIENT_DATA,
    ALIGNMENT_PARTIAL,
    ALIGNMENT_STALLED,
    GevbAlignmentThresholds,
    decide_alignment_status,
    aggregate_gevb_alignment_verdict,
)


def test_converged_requires_mean_and_seed_floor():
    thr = GevbAlignmentThresholds()
    # three seeds all well below 60° / 70°
    late = {0: 55.0, 1: 50.0, 2: 58.0}
    early = {0: 90.0, 1: 88.0, 2: 92.0}
    v = decide_alignment_status(late, early, thresholds=thr)
    assert v.alignment_status == ALIGNMENT_CONVERGED
    assert v.mean_late_fa_deg < 60.0
    assert v.seeds_below_floor == 3


def test_stalled_high_mean():
    thr = GevbAlignmentThresholds()
    late = {0: 88.0, 1: 90.0, 2: 85.0}
    early = {0: 90.0, 1: 91.0, 2: 89.0}
    v = decide_alignment_status(late, early, thresholds=thr)
    assert v.alignment_status == ALIGNMENT_STALLED
    assert v.mean_late_fa_deg >= 75.0


def test_stalled_flat_improvement():
    thr = GevbAlignmentThresholds()
    # late mean ~72 (between 60 and 75) but improvement < 10°
    late = {0: 72.0, 1: 71.0, 2: 73.0}
    early = {0: 78.0, 1: 76.0, 2: 77.0}  # improvement ~5°
    v = decide_alignment_status(late, early, thresholds=thr)
    assert v.alignment_status == ALIGNMENT_STALLED


def test_partial_band():
    thr = GevbAlignmentThresholds()
    # mean ~67°, good improvement, all seeds ≤70 floor
    late = {0: 65.0, 1: 68.0, 2: 66.0}
    early = {0: 90.0, 1: 88.0, 2: 92.0}
    v = decide_alignment_status(late, early, thresholds=thr)
    assert v.alignment_status == ALIGNMENT_PARTIAL


def test_smoke_single_seed_insufficient_or_stalled():
    thr = GevbAlignmentThresholds()
    late = {0: 89.0}
    early = {0: 90.0}
    v = decide_alignment_status(late, early, thresholds=thr)
    assert v.alignment_status in (ALIGNMENT_STALLED, ALIGNMENT_INSUFFICIENT_DATA)
    assert v.n_seeds == 1


def test_thresholds_frozen_in_verdict_payload():
    thr = GevbAlignmentThresholds(
        gevb_fa_converged_mean_deg=60.0,
        gevb_fa_stalled_mean_deg=75.0,
    )
    v = decide_alignment_status({0: 50.0, 1: 52.0, 2: 55.0}, thresholds=thr)
    assert v.thresholds["gevb_fa_converged_mean_deg"] == 60.0
    assert "Do not extend epoch budgets" in v.process_rules[0]


def test_aggregate_from_csv(tmp_path: Path):
    # Write two seeds of alignment CSVs
    for seed, late_ang in [(0, 50.0), (1, 55.0), (2, 52.0)]:
        run = tmp_path / "stationary" / f"seed_{seed}"
        run.mkdir(parents=True)
        path = run / "gevb_alignment.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "seed",
                    "step",
                    "epoch",
                    "snapshot_tag",
                    "layer",
                    "fa_signal_angle_deg",
                    "subspace_principal_angle_deg",
                    "n_samples",
                ],
            )
            w.writeheader()
            for tag, base in [("early", 90.0), ("late", late_ang)]:
                for layer in ("fc1", "fc2", "fc3", "fc_out"):
                    w.writerow(
                        {
                            "seed": seed,
                            "step": 100 if tag == "early" else 9000,
                            "epoch": 1 if tag == "early" else 20,
                            "snapshot_tag": tag,
                            "layer": layer,
                            "fa_signal_angle_deg": base,
                            "subspace_principal_angle_deg": 40.0,
                            "n_samples": 128,
                        }
                    )
    v = aggregate_gevb_alignment_verdict(
        tmp_path, [0, 1, 2], thresholds=GevbAlignmentThresholds()
    )
    assert v.alignment_status == ALIGNMENT_CONVERGED
    assert v.mean_late_fa_deg < 60.0
