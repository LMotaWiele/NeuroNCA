"""Experiment ledger: freeze experiment_id, config content hash, thresholds.

Scientific integrity rules (Phase 0 design):

1. At primary launch, write under ``{output_dir}/meta/``:
   - ``experiment_id.txt`` = ``UTC_date + "_" + first8(sha256(canonical_yaml(config)))``
   - ``config.resolved.yaml``
   - ``config.content_sha256.txt``  (full SHA-256 of **pre-stamp** resolved config)
   - ``thresholds.json``
   - ``signals_broadcast_code_sha256.txt`` (hash of ``signals/*`` + ``broadcast/*``)

2. **Content hash** of the resolved config is the simple identity key: same config
   bytes ⇒ same ``experiment_id`` suffix. Hash is of the **runtime** config with
   ``experiment.id`` forced to ``null`` (ledger-owned stamp is applied only when
   writing ``config.resolved.yaml``). Re-hashing the stamped YAML file will
   **not** match ``config.content_sha256.txt``; verify with the pre-stamp form
   (see header comment written into ``config.resolved.yaml``).

3. Freeze is **idempotent** when meta already matches (same config hash, same
   signals/broadcast code hash, and on-disk ``experiment_id`` suffix matches
   ``content_sha256[:8]``). Thresholds are always re-derived from the passed
   config (disk thresholds are not trusted).

4. Freeze **refuses to overwrite** a primary ledger when:
   - the signals/broadcast source hash differs **or is missing on disk**, or
   - the resolved-config content hash differs, or
   - the on-disk experiment_id suffix disagrees with the content hash.

   Callers should archive under ``results/phase0_runs/<experiment_id>/`` instead.
"""

from __future__ import annotations

import copy
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

_RESOLVED_HEADER = (
    "# NeuroNCA resolved config (ledger archive).\n"
    "# content_sha256 in config.content_sha256.txt is sha256 of the *pre-stamp*\n"
    "# canonical YAML (experiment.id forced to null). The experiment.id below is\n"
    "# ledger-owned and intentionally excluded from that hash.\n"
)


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


def _config_for_hash(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy config with experiment.id forced to null for stable content hashing."""
    cfg = copy.deepcopy(dict(config))
    exp = dict(cfg.get("experiment") or {})
    exp["id"] = None
    cfg["experiment"] = exp
    return cfg


def content_sha256(config: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonical pre-stamp resolved config.

    Serializes with :func:`neuronca.io.config.canonical_yaml` (sorted keys) after
    forcing ``experiment.id`` to ``null`` so the ledger stamp cannot circularly
    affect the identity. This is the documented content hash for Phase 0.
    """
    payload = canonical_yaml(_config_for_hash(config)).encode("utf-8")
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
    """``UTC_date + '_' + first8(sha256(pre-stamp canonical resolved config))``."""
    return experiment_id_from_hash(content_sha256(config), when=when)


def experiment_id_suffix_matches(experiment_id: str, content_hash: str) -> bool:
    """True if ``experiment_id`` ends with ``_<first8(content_hash)>``."""
    expected = f"_{content_hash[:8]}"
    return experiment_id.endswith(expected)


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
        Optional date override for the experiment_id prefix (tests / first freeze).
    package_dir
        ``src/neuronca`` root for signals/broadcast hashing.
    force
        If True, overwrite even when hashes differ (use only for deliberate
        re-runs; not for primary scientific ledger discipline).

    Raises
    ------
    LedgerConflictError
        When an existing primary ledger has a different config content hash,
        a different or missing signals/broadcast code hash, or a tampered
        experiment_id suffix, and ``force`` is False.
    """
    output_dir = Path(output_dir)
    meta = meta_dir_for(output_dir)
    cfg_hash = content_sha256(config)
    code_hash = signals_broadcast_code_sha256(package_dir)
    exp_id = experiment_id_from_hash(cfg_hash, when=when)
    thresholds = extract_thresholds(config)

    existing = load_existing_ledger(output_dir)
    if existing is not None and not force:
        same_config = bool(existing.content_sha256) and existing.content_sha256 == cfg_hash
        # Exact match required: missing/empty on-disk code hash is a conflict.
        same_code = bool(existing.code_sha256) and existing.code_sha256 == code_hash
        id_ok = experiment_id_suffix_matches(existing.experiment_id, cfg_hash)

        if same_config and same_code and id_ok:
            # Heal meta from trusted inputs: keep first-freeze experiment_id
            # (date prefix), re-derive thresholds from config, rewrite files.
            _ensure_ledger_files(
                meta,
                config=config,
                experiment_id=existing.experiment_id,
                content_sha256=cfg_hash,
                code_sha256=code_hash,
                thresholds=thresholds,
            )
            return LedgerMeta(
                experiment_id=existing.experiment_id,
                content_sha256=cfg_hash,
                code_sha256=code_hash,
                meta_dir=meta,
                thresholds=thresholds,
            )

        reasons: list[str] = []
        if not same_config:
            reasons.append(
                f"config content hash changed "
                f"(existing={existing.content_sha256[:12] or '<missing>'}… "
                f"new={cfg_hash[:12]}…)"
            )
        if not same_code:
            if not existing.code_sha256:
                reasons.append(
                    "signals/broadcast code hash missing on disk; "
                    "refusing to overwrite primary ledger without recorded code hash"
                )
            else:
                reasons.append(
                    f"signals/broadcast code hash changed "
                    f"(existing={existing.code_sha256[:12]}… new={code_hash[:12]}…); "
                    "do not overwrite primary ledger after signal code changes"
                )
        if same_config and same_code and not id_ok:
            reasons.append(
                f"experiment_id suffix does not match content hash "
                f"(id={existing.experiment_id!r}, expected suffix _{cfg_hash[:8]})"
            )
        raise LedgerConflictError(
            "refusing to overwrite primary experiment ledger: "
            + "; ".join(reasons)
            + f". Archive under results/phase0_runs/{exp_id}/ or pass force=True."
        )

    # Fresh write or force overwrite.
    _ensure_ledger_files(
        meta,
        config=config,
        experiment_id=exp_id,
        content_sha256=cfg_hash,
        code_sha256=code_hash,
        thresholds=thresholds,
    )
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
    archived = copy.deepcopy(dict(config))
    exp = dict(archived.get("experiment") or {})
    exp["id"] = experiment_id
    archived["experiment"] = exp
    body = canonical_yaml(archived)
    resolved_path.write_text(_RESOLVED_HEADER + body, encoding="utf-8")
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
