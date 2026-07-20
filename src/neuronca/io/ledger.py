"""Experiment ledger: freeze experiment_id, config content hash, thresholds.

Scientific integrity rules (Phase 0 design):

1. At primary launch, write under ``{output_dir}/meta/``:
   - ``experiment_id.txt`` = ``UTC_date + "_" + first8(sha256(canonical_yaml(config)))``
   - ``config.resolved.yaml``
   - ``config.content_sha256.txt``  (full SHA-256 of canonical resolved YAML)
   - ``thresholds.json``
   - ``signals_broadcast_code_sha256.txt`` (hash of ``signals/*`` + ``broadcast/*``)

2. **Content hash** of the resolved config is the simple identity key: same config
   bytes ⇒ same ``experiment_id`` suffix. Documented here so runners do not invent
   alternate hashing schemes.

3. Freeze is **idempotent** when meta already matches (same config hash and same
   signals/broadcast code hash).

4. Freeze **refuses to overwrite** a primary ledger when:
   - the signals/broadcast source hash differs (code change mid-run / after primary), or
   - the resolved-config content hash differs (would change experiment_id).

   Callers should archive under ``results/phase0_runs/<experiment_id>/`` instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from neuronca.io.config import canonical_yaml, extract_thresholds

_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # .../src/neuronca
META_DIRNAME = "meta"

EXPERIMENT_ID_FILE = "experiment_id.txt"
CONFIG_RESOLVED_FILE = "config.resolved.yaml"
CONFIG_HASH_FILE = "config.content_sha256.txt"
THRESHOLDS_FILE = "thresholds.json"
CODE_HASH_FILE = "signals_broadcast_code_sha256.txt"

# Modules whose source changes invalidate a frozen primary run.
_CODE_HASH_SUBPACKAGES = ("signals", "broadcast")


class LedgerConflictError(RuntimeError):
    """Raised when freeze would overwrite a primary ledger with different identity."""


@dataclass(frozen=True)
class LedgerMeta:
    """Frozen ledger identity written under ``meta/``."""

    experiment_id: str
    content_sha256: str
    code_sha256: str
    meta_dir: Path
    thresholds: dict[str, Any]


def content_sha256(config: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonical YAML of ``config``.

    This is the documented content hash for resolved configs: serialize with
    :func:`neuronca.io.config.canonical_yaml` (sorted keys), then SHA-256.
    """
    payload = canonical_yaml(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_id_from_hash(content_hash: str, *, when: date | datetime | None = None) -> str:
    """Build ``YYYY-MM-DD_<first8>`` from a content SHA-256 hex digest."""
    if when is None:
        when = datetime.now(timezone.utc).date()
    elif isinstance(when, datetime):
        when = when.astimezone(timezone.utc).date()
    day = when.isoformat()
    return f"{day}_{content_hash[:8]}"


def make_experiment_id(
    config: Mapping[str, Any],
    *,
    when: date | datetime | None = None,
) -> str:
    """``UTC_date + '_' + first8(sha256(canonical resolved config))``."""
    return experiment_id_from_hash(content_sha256(config), when=when)


def signals_broadcast_code_sha256(
    package_dir: str | Path | None = None,
) -> str:
    """Content hash of all ``*.py`` files under ``signals/`` and ``broadcast/``.

    Files are hashed in sorted path order; each contribution is
    ``relative_path + NUL + file_bytes``. Used to refuse overwriting a primary
    ledger after signal/broadcast code changes.
    """
    root = Path(package_dir) if package_dir is not None else _PACKAGE_DIR
    h = hashlib.sha256()
    paths: list[Path] = []
    for sub in _CODE_HASH_SUBPACKAGES:
        d = root / sub
        if not d.is_dir():
            continue
        paths.extend(sorted(d.rglob("*.py")))
    for path in paths:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def meta_dir_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / META_DIRNAME


def load_existing_ledger(output_dir: str | Path) -> LedgerMeta | None:
    """Return existing primary ledger meta if ``experiment_id.txt`` is present."""
    meta = meta_dir_for(output_dir)
    exp = _read_text(meta / EXPERIMENT_ID_FILE)
    if exp is None:
        return None
    content = _read_text(meta / CONFIG_HASH_FILE) or ""
    code = _read_text(meta / CODE_HASH_FILE) or ""
    thr_path = meta / THRESHOLDS_FILE
    thresholds: dict[str, Any] = {}
    if thr_path.is_file():
        thresholds = json.loads(thr_path.read_text(encoding="utf-8"))
    return LedgerMeta(
        experiment_id=exp,
        content_sha256=content,
        code_sha256=code,
        meta_dir=meta,
        thresholds=thresholds,
    )


def freeze_ledger(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    when: date | datetime | None = None,
    package_dir: str | Path | None = None,
    force: bool = False,
) -> LedgerMeta:
    """Freeze experiment identity and thresholds under ``{output_dir}/meta/``.

    Parameters
    ----------
    output_dir
        Experiment root (e.g. ``results/phase0``).
    config
        Resolved runtime config (already smoke-merged / device-resolved).
    when
        Optional date override for the experiment_id prefix (tests).
    package_dir
        ``src/neuronca`` root for signals/broadcast hashing.
    force
        If True, overwrite even when hashes differ (use only for deliberate
        re-runs; not for primary scientific ledger discipline).

    Raises
    ------
    LedgerConflictError
        When an existing primary ledger has a different config content hash or
        a different signals/broadcast code hash and ``force`` is False.
    """
    output_dir = Path(output_dir)
    meta = meta_dir_for(output_dir)
    cfg_hash = content_sha256(config)
    code_hash = signals_broadcast_code_sha256(package_dir)
    exp_id = experiment_id_from_hash(cfg_hash, when=when)
    thresholds = extract_thresholds(config)

    existing = load_existing_ledger(output_dir)
    if existing is not None and not force:
        same_config = existing.content_sha256 == cfg_hash
        same_code = (not existing.code_sha256) or existing.code_sha256 == code_hash
        # Prefer full identity match for idempotency (date prefix may differ if
        # only content hash stored previously without re-reading date).
        if same_config and same_code:
            # Ensure all files exist; rewrite missing ones only if content matches.
            _ensure_ledger_files(
                meta,
                config=config,
                experiment_id=existing.experiment_id,
                content_sha256=cfg_hash,
                code_sha256=code_hash,
                thresholds=existing.thresholds or thresholds,
            )
            return LedgerMeta(
                experiment_id=existing.experiment_id,
                content_sha256=cfg_hash,
                code_sha256=code_hash,
                meta_dir=meta,
                thresholds=existing.thresholds or thresholds,
            )
        reasons: list[str] = []
        if not same_config:
            reasons.append(
                f"config content hash changed "
                f"(existing={existing.content_sha256[:12]}… new={cfg_hash[:12]}…)"
            )
        if not same_code:
            reasons.append(
                f"signals/broadcast code hash changed "
                f"(existing={existing.code_sha256[:12]}… new={code_hash[:12]}…); "
                "do not overwrite primary ledger after signal code changes"
            )
        raise LedgerConflictError(
            "refusing to overwrite primary experiment ledger: "
            + "; ".join(reasons)
            + f". Archive under results/phase0_runs/{exp_id}/ or pass force=True."
        )

    # Allow force or fresh write; if force and existing id date differs, use new id.
    _ensure_ledger_files(
        meta,
        config=config,
        experiment_id=exp_id,
        content_sha256=cfg_hash,
        code_sha256=code_hash,
        thresholds=thresholds,
    )
    # Stamp experiment.id into a copy note only via resolved yaml (config is caller-owned).
    return LedgerMeta(
        experiment_id=exp_id,
        content_sha256=cfg_hash,
        code_sha256=code_hash,
        meta_dir=meta,
        thresholds=thresholds,
    )


def _ensure_ledger_files(
    meta: Path,
    *,
    config: Mapping[str, Any],
    experiment_id: str,
    content_sha256: str,
    code_sha256: str,
    thresholds: Mapping[str, Any],
) -> None:
    meta.mkdir(parents=True, exist_ok=True)
    _write_text(meta / EXPERIMENT_ID_FILE, experiment_id)
    resolved_path = meta / CONFIG_RESOLVED_FILE
    # Inject frozen experiment id into the archived resolved config copy.
    archived = dict(config)
    exp = dict(archived.get("experiment") or {})
    exp["id"] = experiment_id
    archived["experiment"] = exp
    resolved_path.write_text(canonical_yaml(archived), encoding="utf-8")
    _write_text(meta / CONFIG_HASH_FILE, content_sha256)
    _write_text(meta / CODE_HASH_FILE, code_sha256)
    thr_path = meta / THRESHOLDS_FILE
    thr_path.write_text(
        json.dumps(dict(thresholds), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_resolved_config(output_dir: str | Path) -> dict[str, Any]:
    """Load ``meta/config.resolved.yaml`` for a frozen run."""
    path = meta_dir_for(output_dir) / CONFIG_RESOLVED_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no frozen resolved config at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"resolved config is not a mapping: {path}")
    return data
