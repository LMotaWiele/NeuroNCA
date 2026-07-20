"""YAML config loading, deep-merge, smoke overrides, and device resolution.

Resolved config is the effective runtime dict after merging:

    default.yaml  ⊕  override YAML(s)  ⊕  optional smoke block  ⊕  optional kwargs

The top-level ``smoke:`` block is an override source only and is stripped from
the returned resolved config so content hashes reflect runtime values only.

When ``smoke=True``, the smoke block is taken from the **base** defaults file
(before override nullification). Overrides may not wipe CI smoke by setting
``smoke: null``. If no usable smoke mapping is found, :class:`SmokeConfigError`
is raised rather than silently running a full-budget config.

Device resolution: ``device: auto`` becomes ``cuda`` if available else ``cpu``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
import yaml

# Repo-root-relative defaults when no explicit path is given.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]  # .../src/neuronca/io -> repo
DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "configs" / "default.yaml"
CONFIGS_DIR = _PACKAGE_ROOT / "configs"


class SmokeConfigError(ValueError):
    """Raised when ``smoke=True`` but no usable smoke override block is available."""


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Nested dicts are merged; all other values (including lists) are replaced.
    ``None`` override values replace the base value.
    """
    out: dict[str, Any] = copy.deepcopy(dict(base))
    if not override:
        return out
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``. Empty files yield ``{}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def resolve_device(device: str | None = "auto") -> str:
    """Map ``auto`` → ``cuda`` if available else ``cpu``; pass through other values."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _apply_device(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    return cfg


def _usable_smoke_block(block: Any) -> dict[str, Any] | None:
    """Return a non-empty smoke mapping, or None if missing / null / empty."""
    if not isinstance(block, Mapping) or not block:
        return None
    return dict(block)


def load_config(
    *override_paths: str | Path,
    defaults_path: str | Path | None = None,
    smoke: bool = False,
    overrides: Mapping[str, Any] | None = None,
    resolve_dev: bool = True,
) -> dict[str, Any]:
    """Load and deep-merge configs into a single resolved dict.

    Parameters
    ----------
    override_paths
        Optional YAML files merged left-to-right on top of defaults
        (e.g. ``configs/stationary.yaml``).
    defaults_path
        Base YAML (default: ``configs/default.yaml`` relative to repo root).
    smoke
        If True, deep-merge the base file's top-level ``smoke:`` block so CI can
        shrink epochs / probe size / seeds. The block is read from the base
        defaults **before** override files run, so ``smoke: null`` in an override
        cannot disable smoke. Raises :class:`SmokeConfigError` if no usable
        smoke mapping exists on the base file.
    overrides
        Optional in-memory mapping merged last (CLI flags, etc.). The top-level
        ``smoke`` key in overrides is ignored (stripped); use ``smoke=True`` to
        apply smoke, not an in-memory smoke dict via overrides.
    resolve_dev
        If True (default), replace ``device: auto`` with a concrete device.

    Returns
    -------
    dict
        Resolved runtime config. The ``smoke`` key is always removed so that
        content hashes reflect applied values only.
    """
    base_path = Path(defaults_path) if defaults_path is not None else DEFAULT_CONFIG_PATH
    base_raw = load_yaml(base_path)
    # Capture smoke from the base file only (before overrides can nullify it).
    base_smoke = _usable_smoke_block(base_raw.get("smoke"))

    cfg = copy.deepcopy(base_raw)
    for path in override_paths:
        cfg = deep_merge(cfg, load_yaml(path))

    # Always strip smoke from the merged tree (override nulls / partial maps).
    cfg.pop("smoke", None)

    if smoke:
        if base_smoke is None:
            raise SmokeConfigError(
                f"smoke=True but no usable smoke: block found in base config "
                f"{base_path}. Define a non-empty top-level smoke: mapping on "
                f"the defaults file; overrides must not be relied on to supply it."
            )
        cfg = deep_merge(cfg, base_smoke)

    if overrides:
        # Do not allow overrides to re-introduce a smoke block into the hash.
        ov = dict(overrides)
        ov.pop("smoke", None)
        cfg = deep_merge(cfg, ov)
        cfg.pop("smoke", None)

    if resolve_dev:
        cfg = _apply_device(cfg)

    return cfg


def load_config_named(
    name: str,
    *,
    smoke: bool = False,
    overrides: Mapping[str, Any] | None = None,
    configs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience: load ``configs/{name}.yaml`` on top of ``default.yaml``.

    ``name`` may be ``\"default\"`` (no extra override), ``\"stationary\"``,
    or ``\"task_shift\"``.
    """
    root = Path(configs_dir) if configs_dir is not None else CONFIGS_DIR
    defaults = root / "default.yaml"
    if name in ("", "default", "defaults"):
        return load_config(defaults_path=defaults, smoke=smoke, overrides=overrides)
    override = root / f"{name}.yaml"
    return load_config(
        override,
        defaults_path=defaults,
        smoke=smoke,
        overrides=overrides,
    )


def extract_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Pull evaluation thresholds (and weak-shift ceiling) for ledger freeze."""
    eval_cfg = dict(config.get("eval") or {})
    thr = dict(eval_cfg.get("thresholds") or {})
    # Single source of truth for weak_shift lives at eval level
    if "gt_shift_disagreement_ceiling" in eval_cfg:
        thr.setdefault(
            "gt_shift_disagreement_ceiling",
            eval_cfg["gt_shift_disagreement_ceiling"],
        )
    return thr


def canonical_yaml(config: Mapping[str, Any]) -> str:
    """Stable YAML serialization for content hashing (sorted keys, unicode)."""
    return yaml.safe_dump(
        dict(config),
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
    )
