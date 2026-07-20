"""Tests for experiment ledger freeze, content hash, and overwrite guards."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from neuronca.io.config import load_config
from neuronca.io.ledger import (
    CODE_HASH_FILE,
    CONFIG_HASH_FILE,
    CONFIG_RESOLVED_FILE,
    EXPERIMENT_ID_FILE,
    THRESHOLDS_FILE,
    LedgerConflictError,
    content_sha256,
    freeze_ledger,
    load_existing_ledger,
    make_experiment_id,
    signals_broadcast_code_sha256,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PACKAGE = REPO / "src" / "neuronca"


@pytest.fixture
def resolved_cfg():
    return load_config(defaults_path=CONFIGS / "default.yaml")


def test_content_hash_changes_with_config(resolved_cfg):
    h1 = content_sha256(resolved_cfg)
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)

    cfg2 = dict(resolved_cfg)
    cfg2["train"] = dict(resolved_cfg["train"])
    cfg2["train"]["lr"] = 0.123456
    h2 = content_sha256(cfg2)
    assert h1 != h2

    # Same content → same hash (key order independent via canonical YAML)
    cfg3 = load_config(defaults_path=CONFIGS / "default.yaml")
    assert content_sha256(cfg3) == h1


def test_experiment_id_format(resolved_cfg):
    exp = make_experiment_id(resolved_cfg, when=date(2026, 7, 20))
    h = content_sha256(resolved_cfg)
    assert exp == f"2026-07-20_{h[:8]}"


def test_freeze_ledger_writes_meta(tmp_path: Path, resolved_cfg):
    out = tmp_path / "phase0"
    meta = freeze_ledger(
        out,
        resolved_cfg,
        when=date(2026, 7, 20),
        package_dir=PACKAGE,
    )
    assert meta.meta_dir == out / "meta"
    assert meta.experiment_id.startswith("2026-07-20_")
    assert (meta.meta_dir / EXPERIMENT_ID_FILE).read_text(encoding="utf-8").strip() == (
        meta.experiment_id
    )
    assert (meta.meta_dir / CONFIG_HASH_FILE).read_text(encoding="utf-8").strip() == (
        meta.content_sha256
    )
    assert (meta.meta_dir / CODE_HASH_FILE).read_text(encoding="utf-8").strip() == (
        meta.code_sha256
    )
    resolved = yaml.safe_load((meta.meta_dir / CONFIG_RESOLVED_FILE).read_text())
    assert resolved["experiment"]["id"] == meta.experiment_id
    thresholds = json.loads((meta.meta_dir / THRESHOLDS_FILE).read_text())
    assert thresholds["stationary_edge_spearman"] == 0.60
    assert "gt_shift_disagreement_ceiling" in thresholds


def test_freeze_idempotent(tmp_path: Path, resolved_cfg):
    out = tmp_path / "phase0"
    m1 = freeze_ledger(out, resolved_cfg, when=date(2026, 7, 20), package_dir=PACKAGE)
    m2 = freeze_ledger(out, resolved_cfg, when=date(2026, 7, 21), package_dir=PACKAGE)
    # Same config → same frozen id (date of first freeze kept)
    assert m2.experiment_id == m1.experiment_id
    assert m2.content_sha256 == m1.content_sha256
    existing = load_existing_ledger(out)
    assert existing is not None
    assert existing.experiment_id == m1.experiment_id


def test_freeze_refuses_config_change(tmp_path: Path, resolved_cfg):
    out = tmp_path / "phase0"
    freeze_ledger(out, resolved_cfg, when=date(2026, 7, 20), package_dir=PACKAGE)
    changed = dict(resolved_cfg)
    changed["train"] = dict(resolved_cfg["train"])
    changed["train"]["epochs"] = 99
    with pytest.raises(LedgerConflictError, match="config content hash"):
        freeze_ledger(out, changed, when=date(2026, 7, 20), package_dir=PACKAGE)


def test_freeze_refuses_code_hash_change(tmp_path: Path, resolved_cfg):
    out = tmp_path / "phase0"
    # Fake package dir with signals/broadcast so we can mutate code after freeze
    pkg = tmp_path / "neuronca_pkg"
    (pkg / "signals").mkdir(parents=True)
    (pkg / "broadcast").mkdir(parents=True)
    (pkg / "signals" / "__init__.py").write_text("# v1\n", encoding="utf-8")
    (pkg / "broadcast" / "__init__.py").write_text("# v1\n", encoding="utf-8")

    freeze_ledger(out, resolved_cfg, when=date(2026, 7, 20), package_dir=pkg)
    (pkg / "signals" / "__init__.py").write_text("# v2 redefined\n", encoding="utf-8")
    with pytest.raises(LedgerConflictError, match="signals/broadcast code hash"):
        freeze_ledger(out, resolved_cfg, when=date(2026, 7, 20), package_dir=pkg)


def test_force_overwrites(tmp_path: Path, resolved_cfg):
    out = tmp_path / "phase0"
    freeze_ledger(out, resolved_cfg, when=date(2026, 7, 20), package_dir=PACKAGE)
    changed = dict(resolved_cfg)
    changed["seed"] = 42
    m = freeze_ledger(
        out, changed, when=date(2026, 7, 20), package_dir=PACKAGE, force=True
    )
    assert m.content_sha256 == content_sha256(changed)


def test_signals_broadcast_code_hash_stable():
    h1 = signals_broadcast_code_sha256(PACKAGE)
    h2 = signals_broadcast_code_sha256(PACKAGE)
    assert h1 == h2
    assert len(h1) == 64
