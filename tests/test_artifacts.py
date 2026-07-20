"""Tests for results tree, correlation CSV helpers, and verdict.json schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neuronca.eval.verdict import VerdictResult, aggregate_verdict
from neuronca.io.artifacts import (
    CORRELATION_CSV_COLUMNS,
    PHASE0_SUBDIRS,
    correlation_rows_from_result,
    ensure_results_tree,
    seed_run_dir,
    write_correlations_csv,
    write_verdict_json,
)


def test_ensure_results_tree(tmp_path: Path):
    root = ensure_results_tree(tmp_path / "phase0")
    assert root.is_dir()
    for name in PHASE0_SUBDIRS:
        assert (root / name).is_dir()
    # idempotent
    ensure_results_tree(root)


def test_seed_run_dir(tmp_path: Path):
    d = seed_run_dir(tmp_path / "phase0", "stationary", 1)
    assert d == tmp_path / "phase0" / "stationary" / "seed_1"
    assert d.is_dir()


def test_write_correlations_csv(tmp_path: Path):
    rows = correlation_rows_from_result(
        per_layer={
            "fc1": {"spearman": 0.7, "pearson": 0.65, "n_pairs": 100},
            "fc2": {"spearman": 0.5, "pearson": 0.4, "n_pairs": 80},
        },
        overall_spearman=0.6,
        overall_pearson=0.55,
        overall_n_pairs=180,
        seed=0,
        condition="stationary",
        snapshot_tag="late",
        step=1000,
        local_signal="activation",
        gt_measure="grad_abs",
        granularity="edge",
        mask_restricted=False,
    )
    path = tmp_path / "correlations.csv"
    write_correlations_csv(path, rows)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        got = list(reader)
    assert len(got) == 3  # 2 layers + overall
    assert reader.fieldnames is not None or True
    # re-read fieldnames
    with path.open(encoding="utf-8", newline="") as f:
        fieldnames = next(csv.reader(f))
    for col in CORRELATION_CSV_COLUMNS:
        assert col in fieldnames
    overall = [r for r in got if r["layer"] == "overall"][0]
    assert overall["is_overall"] in ("True", "true", True) or overall["is_overall"] == "True"
    assert float(overall["spearman"]) == 0.6

    # append
    write_correlations_csv(
        path,
        [
            {
                "seed": 1,
                "condition": "stationary",
                "snapshot_tag": "late",
                "step": 1000,
                "local_signal": "hebbian_corr",
                "gt_measure": "grad_abs",
                "layer": "overall",
                "granularity": "edge",
                "spearman": 0.8,
                "pearson": 0.7,
                "n_pairs": 180,
                "mask_restricted": False,
                "is_overall": True,
            }
        ],
        append=True,
    )
    with path.open(encoding="utf-8", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == 4


def test_write_verdict_json_schema(tmp_path: Path):
    seed_rho = {
        "eligibility_gevb_rf": {0: 0.70, 1: 0.72, 2: 0.68},
        "activation": {0: 0.20, 1: 0.22, 2: 0.18},
    }
    result = aggregate_verdict(seed_rho, rho_shift_by_seed={0: 0.4, 1: 0.5, 2: 0.45})
    path = tmp_path / "aggregate" / "verdict.json"
    write_verdict_json(path, result, experiment_id="2026-07-20_deadbeef")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "experiment_id",
        "verdict",
        "phase1_go",
        "best_local",
        "stationary_pass",
        "recovered",
        "weak_shift",
        "catastrophic_fail",
        "thresholds",
        "stationary",
        "task_shift",
        "interpretation_notes",
    ):
        assert key in data
    assert data["experiment_id"] == "2026-07-20_deadbeef"
    assert data["best_local"] == "eligibility_gevb_rf"
    assert data["stationary"]["best_local"] == "eligibility_gevb_rf"
    assert "0" in data["stationary"]["per_seed_rho"]
    assert data["phase1_go"] is False  # no recovery lag → not PASS
    assert isinstance(data["thresholds"], dict)


def test_verdict_payload_from_mapping(tmp_path: Path):
    result = VerdictResult(
        verdict="FAIL",
        phase1_go=False,
        best_local=None,
        stationary_pass=False,
        recovered=False,
        weak_shift=False,
        catastrophic_fail=True,
        mean_rho_by_candidate={},
        seed_rho_by_candidate={},
        recovery_by_seed={},
        recovery_rho_mean=float("nan"),
        rho_shift_mean=float("nan"),
        rho_shift_by_seed={},
        thresholds={"fail_rho_ceiling": 0.3},
        notes=["empty"],
    )
    path = write_verdict_json(tmp_path / "v.json", result, experiment_id="x")
    data = json.loads(path.read_text())
    assert data["verdict"] == "FAIL"
    assert data["catastrophic_fail"] is True


def test_verdict_json_portable_null_not_nan(tmp_path: Path):
    """Non-finite floats become null; file has no NaN/Infinity tokens (Issue 5)."""
    result = VerdictResult(
        verdict="FAIL",
        phase1_go=False,
        best_local="activation",
        stationary_pass=False,
        recovered=False,
        weak_shift=False,
        catastrophic_fail=True,
        mean_rho_by_candidate={"activation": float("nan")},
        seed_rho_by_candidate={},
        recovery_by_seed={},
        recovery_rho_mean=float("nan"),
        rho_shift_mean=float("nan"),
        rho_shift_by_seed={},
        thresholds={"fail_rho_ceiling": 0.3},
        notes=[],
    )
    path = write_verdict_json(tmp_path / "v.json", result, experiment_id="e1")
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    data = json.loads(raw)  # strict RFC-friendly enough for stdlib
    assert data["recovery_rho_mean"] is None
    assert data["rho_shift_mean"] is None
    assert data["mean_rho_by_candidate"]["activation"] is None
    # best_rho_mean falls back to mean map → also null when nan
    assert data["stationary"]["best_rho_mean"] is None


def test_best_rho_mean_fallback_when_seeds_empty():
    """When per-seed map empty, use mean_rho_by_candidate (Issue 6)."""
    from neuronca.io.artifacts import verdict_payload

    result = {
        "verdict": "FAIL",
        "phase1_go": False,
        "best_local": "activation",
        "stationary_pass": False,
        "recovered": False,
        "weak_shift": False,
        "catastrophic_fail": False,
        "mean_rho_by_candidate": {"activation": 0.55},
        "seed_rho_by_candidate": {"activation": {}},
        "recovery_by_seed": {},
        "recovery_rho_mean": float("nan"),
        "rho_shift_mean": float("nan"),
        "rho_shift_by_seed": {},
        "thresholds": {"stationary_seed_floor": 0.5},
        "notes": [],
    }
    payload = verdict_payload(result, experiment_id="e")
    assert payload["stationary"]["best_rho_mean"] == pytest.approx(0.55)


def test_extra_cannot_clobber_required_fields():
    from neuronca.io.artifacts import verdict_payload

    result = VerdictResult(
        verdict="FAIL",
        phase1_go=False,
        best_local=None,
        stationary_pass=False,
        recovered=False,
        weak_shift=False,
        catastrophic_fail=True,
        mean_rho_by_candidate={},
        seed_rho_by_candidate={},
        recovery_by_seed={},
        recovery_rho_mean=float("nan"),
        rho_shift_mean=float("nan"),
        rho_shift_by_seed={},
        thresholds={},
        notes=[],
    )
    with pytest.raises(ValueError, match="clobber"):
        verdict_payload(result, experiment_id="e", extra={"verdict": "PASS"})


def test_write_verdict_overwrite_false(tmp_path: Path):
    result = VerdictResult(
        verdict="FAIL",
        phase1_go=False,
        best_local=None,
        stationary_pass=False,
        recovered=False,
        weak_shift=False,
        catastrophic_fail=True,
        mean_rho_by_candidate={},
        seed_rho_by_candidate={},
        recovery_by_seed={},
        recovery_rho_mean=float("nan"),
        rho_shift_mean=float("nan"),
        rho_shift_by_seed={},
        thresholds={},
        notes=[],
    )
    path = tmp_path / "verdict.json"
    write_verdict_json(path, result, experiment_id="e")
    with pytest.raises(FileExistsError):
        write_verdict_json(path, result, experiment_id="e", overwrite=False)
