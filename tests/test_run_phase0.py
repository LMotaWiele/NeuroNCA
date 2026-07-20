"""CLI / aggregation tests for scripts/run_phase0.py (dry path + mocked runs)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import helpers from the script module
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import run_phase0 as rp  # noqa: E402
from neuronca.eval.verdict import aggregate_verdict  # noqa: E402
from neuronca.io.artifacts import write_verdict_json  # noqa: E402


def test_parse_seeds():
    assert rp.parse_seeds("0,1,2") == [0, 1, 2]
    assert rp.parse_seeds(" 0 , 2 ") == [0, 2]
    with pytest.raises(Exception):
        rp.parse_seeds("")


def test_dry_run_cli(tmp_path: Path):
    """--dry-run prints plan without training."""
    out = rp.main(
        [
            "--config",
            str(_REPO / "configs" / "default.yaml"),
            "--smoke",
            "--seeds",
            "0,1",
            "--results-root",
            str(tmp_path / "phase0"),
            "--condition",
            "both",
            "--dry-run",
        ]
    )
    assert out == 0


def test_dry_run_subprocess():
    """Smoke CLI dry path via subprocess (no training)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "run_phase0.py"),
            "--config",
            str(_REPO / "configs" / "default.yaml"),
            "--smoke",
            "--seeds",
            "0",
            "--condition",
            "stationary",
            "--dry-run",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["smoke"] is True
    assert payload["seeds"] == [0]
    assert payload["run_stationary"] is True
    assert payload["run_task_shift"] is False


def _fake_stationary(config, results_root, seed, smoke=False):
    results_root = Path(results_root)
    run_dir = results_root / "stationary" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # correlations.csv with late overall edge rows
    rows = []
    for cand, rho in (
        ("eligibility_gevb_rf", 0.70 + 0.01 * seed),
        ("activation", 0.20),
    ):
        rows.append(
            {
                "seed": seed,
                "condition": "stationary",
                "snapshot_tag": "late",
                "step": 10,
                "local_signal": cand,
                "gt_measure": "grad_abs",
                "layer": "overall",
                "granularity": "edge",
                "spearman": rho,
                "pearson": 0.0,
                "n_pairs": 100,
                "mask_restricted": False,
                "is_overall": True,
            }
        )
    corr = run_dir / "correlations.csv"
    with corr.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "seed": seed,
        "condition": "stationary",
        "smoke": smoke,
        "experiment_id": "2026-07-21_testhash",
        "late_edge_grad_abs": {
            "best_local": "eligibility_gevb_rf",
            "best_rho": 0.70 + 0.01 * seed,
            "per_candidate": {
                "eligibility_gevb_rf": 0.70 + 0.01 * seed,
                "activation": 0.20,
            },
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # minimal ledger meta for experiment_id
    meta = results_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "experiment_id.txt").write_text("2026-07-21_testhash\n", encoding="utf-8")
    return summary


def _fake_task_shift(config, results_root, seed, smoke=False):
    results_root = Path(results_root)
    run_dir = results_root / "task_shift" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # lag curve that recovers at t=25 for best_local
    lag_rows = []
    for t, rho in ((0, 0.35), (25, 0.65), (50, 0.70)):
        lag_rows.append(
            {
                "steps_after_switch": t,
                "seed": seed,
                "local_signal": "eligibility_gevb_rf",
                "gt_measure": "grad_abs",
                "spearman": rho,
                "recovered_abs": True,
                "recovered_rel": False,
                "t_star_abs": 25,
                "t_star_rel": "",
                "rho_pre_switch": 0.72,
                "rho_rec": 0.65,
                "rho_max_in_window": 0.70,
            }
        )
    lag_path = run_dir / "time_lag.csv"
    with lag_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lag_rows[0].keys()))
        w.writeheader()
        w.writerows(lag_rows)
    summary = {
        "seed": seed,
        "condition": "task_shift",
        "smoke": smoke,
        "experiment_id": "2026-07-21_testhash",
        "gt_shift_disagreement": 0.40,
        "weak_shift": False,
        "rho_pre_switch": {"eligibility_gevb_rf": 0.72},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def test_run_phase0_mocked_aggregate(tmp_path: Path):
    """Multi-seed orchestration writes verdict.json only via verdict + writer."""
    results_root = tmp_path / "phase0"
    cfg = {
        "experiment": {"seeds": [0, 1, 2], "output_dir": str(results_root)},
        "eval": {
            "thresholds": {
                "stationary_edge_spearman": 0.60,
                "stationary_seed_floor": 0.50,
                "stationary_seed_floor_min_count": 2,
                "shift_recovery_spearman": 0.60,
                "shift_recovery_max_steps": 500,
                "shift_recovery_min_seed_count": 2,
                "fail_rho_ceiling": 0.30,
            },
            "gt_shift_disagreement_ceiling": 0.90,
        },
    }

    with (
        patch.object(rp, "run_stationary", side_effect=_fake_stationary),
        patch.object(rp, "run_task_shift", side_effect=_fake_task_shift),
    ):
        summary = rp.run_phase0(
            config=cfg,
            results_root=results_root,
            seeds=[0, 1, 2],
            condition="both",
            smoke=False,
        )

    assert summary["verdict"] == "PASS"
    assert summary["phase1_go"] is True
    assert summary["best_local"] == "eligibility_gevb_rf"
    verdict_path = Path(summary["verdict_path"])
    assert verdict_path.is_file()
    assert verdict_path == results_root / "aggregate" / "verdict.json"
    data = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["phase1_go"] is True
    assert data["experiment_id"] == "2026-07-21_testhash"
    assert data["best_local"] == "eligibility_gevb_rf"
    assert "stationary" in data
    assert "task_shift" in data


def test_aggregate_only_path_uses_verdict_module(tmp_path: Path):
    """aggregate_phase0_verdict delegates to aggregate_verdict + write_verdict_json."""
    results_root = tmp_path / "phase0"
    for seed in (0, 1, 2):
        _fake_stationary({}, results_root, seed)
        _fake_task_shift({}, results_root, seed)

    with (
        patch(
            "run_phase0.aggregate_verdict", wraps=aggregate_verdict
        ) as mock_agg,
        patch(
            "run_phase0.write_verdict_json", wraps=write_verdict_json
        ) as mock_write,
    ):
        result, path = rp.aggregate_phase0_verdict(
            results_root,
            [0, 1, 2],
            experiment_id="2026-07-21_testhash",
        )
    assert mock_agg.called
    assert mock_write.called
    assert result.verdict == "PASS"
    assert path.is_file()


def test_reproduce_sh_is_executable():
    path = _REPO / "scripts" / "reproduce.sh"
    assert path.is_file()
    mode = path.stat().st_mode
    assert mode & 0o111, "reproduce.sh must be executable"
    text = path.read_text(encoding="utf-8")
    assert "run_phase0.py" in text
    assert "--smoke" in text
    assert "set -euo pipefail" in text
