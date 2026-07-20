"""Results tree helpers, correlation CSV writers, and verdict.json schema.

Artifact layout (Phase 0)::

    results/phase0/
    ├── meta/           # ledger (see ledger.py)
    ├── stationary/
    ├── task_shift/
    ├── figures/
    ├── aggregate/      # verdict.json, tables, mean_std_correlations.csv
    ├── sensitivity/
    └── docs_snapshot/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neuronca.eval.verdict import VerdictResult

DEFAULT_RESULTS_ROOT = Path("results/phase0")

PHASE0_SUBDIRS: tuple[str, ...] = (
    "meta",
    "stationary",
    "task_shift",
    "figures",
    "aggregate",
    "sensitivity",
    "docs_snapshot",
)

# correlations.csv column order (design + runner fields). Extra keys are appended.
CORRELATION_CSV_COLUMNS: tuple[str, ...] = (
    "seed",
    "condition",
    "snapshot_tag",
    "step",
    "local_signal",
    "gt_measure",
    "layer",
    "granularity",
    "spearman",
    "pearson",
    "n_pairs",
    "mask_restricted",
    "is_overall",
)

# Core fields expected in aggregate/verdict.json (design example + VerdictResult).
VERDICT_JSON_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "verdict",
    "phase1_go",
    "best_local",
    "stationary_pass",
    "recovered",
    "weak_shift",
    "catastrophic_fail",
    "thresholds",
)

# recovery_rho_definition: window-end wording (config-driven max steps, not hard-coded 500).
RECOVERY_RHO_DEFINITION = "first_crossing_or_rho_at_window_end"


def ensure_results_tree(root: str | Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Create the Phase 0 results directory tree; return the root path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in PHASE0_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def seed_run_dir(
    root: str | Path,
    condition: str,
    seed: int,
) -> Path:
    """``{root}/{condition}/seed_{k}/`` (created)."""
    path = Path(root) / condition / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {c: None for c in CORRELATION_CSV_COLUMNS}
    for row in rows:
        for key in row:
            if key not in seen:
                seen[key] = None
    # Keep preferred columns first even if missing from rows
    return list(seen.keys())


def write_correlations_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
) -> Path:
    """Write (or append) correlation rows to CSV with stable column order.

    Each row is a mapping; recommended keys match :data:`CORRELATION_CSV_COLUMNS`.
    ``layer`` may be a layer name or ``\"overall\"`` with ``is_overall=True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialised = [dict(r) for r in rows]
    fieldnames = _ordered_fieldnames(materialised)
    write_header = True
    mode = "w"
    if append and path.is_file() and path.stat().st_size > 0:
        mode = "a"
        write_header = False
        # Prefer existing header order when appending
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
            if existing:
                # Extend with any new keys
                for k in fieldnames:
                    if k not in existing:
                        existing.append(k)
                fieldnames = existing

    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in materialised:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def correlation_rows_from_result(
    *,
    per_layer: Mapping[str, Mapping[str, Any]] | None = None,
    overall_spearman: float | None = None,
    overall_pearson: float | None = None,
    overall_n_pairs: int | None = None,
    seed: int | None = None,
    condition: str | None = None,
    snapshot_tag: str | None = None,
    step: int | None = None,
    local_signal: str | None = None,
    gt_measure: str | None = None,
    granularity: str = "edge",
    mask_restricted: bool = False,
) -> list[dict[str, Any]]:
    """Build CSV row dicts from per-layer metrics + optional overall.

    ``per_layer`` values may be mappings with spearman/pearson/n_pairs or objects
    with those attributes (e.g. ``LayerCorrelation``).
    """
    rows: list[dict[str, Any]] = []
    base = {
        "seed": seed if seed is not None else "",
        "condition": condition or "",
        "snapshot_tag": snapshot_tag or "",
        "step": step if step is not None else "",
        "local_signal": local_signal or "",
        "gt_measure": gt_measure or "",
        "granularity": granularity,
        "mask_restricted": bool(mask_restricted),
    }
    if per_layer:
        for layer, metrics in per_layer.items():
            if hasattr(metrics, "spearman"):
                sp = float(metrics.spearman)
                pe = float(metrics.pearson)
                n = int(metrics.n_pairs)
            else:
                m = dict(metrics)  # type: ignore[arg-type]
                sp = float(m.get("spearman", float("nan")))
                pe = float(m.get("pearson", float("nan")))
                n = int(m.get("n_pairs", 0))
            rows.append(
                {
                    **base,
                    "layer": layer,
                    "spearman": sp,
                    "pearson": pe,
                    "n_pairs": n,
                    "is_overall": False,
                }
            )
    if overall_spearman is not None or overall_pearson is not None:
        rows.append(
            {
                **base,
                "layer": "overall",
                "spearman": (
                    float(overall_spearman) if overall_spearman is not None else float("nan")
                ),
                "pearson": (
                    float(overall_pearson) if overall_pearson is not None else float("nan")
                ),
                "n_pairs": int(overall_n_pairs or 0),
                "is_overall": True,
            }
        )
    return rows


def _is_nonfinite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, np.floating, np.integer)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return False
        return not math.isfinite(f)
    return False


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to None for portable JSON."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if _is_nonfinite_number(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def verdict_payload(
    result: VerdictResult | Mapping[str, Any],
    *,
    experiment_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable verdict dict with required schema fields.

    Flattens :class:`~neuronca.eval.verdict.VerdictResult` and always includes
    ``experiment_id``. Nested ``stationary`` / ``task_shift`` blocks are added
    when recoverable from the result for design-doc compatibility.

    Non-finite floats become ``null``. Reserved schema keys cannot be clobbered
    via ``extra`` (those keys are applied under ``extra`` only if reserved, else
    rejected — see below).
    """
    if isinstance(result, VerdictResult):
        data = result.to_dict()
    else:
        data = dict(result)

    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "verdict": data.get("verdict"),
        "phase1_go": bool(data.get("phase1_go", False)),
        "best_local": data.get("best_local"),
        "stationary_pass": bool(data.get("stationary_pass", False)),
        "recovered": bool(data.get("recovered", False)),
        "weak_shift": bool(data.get("weak_shift", False)),
        "catastrophic_fail": bool(data.get("catastrophic_fail", False)),
        "thresholds": dict(data.get("thresholds") or {}),
        "mean_rho_by_candidate": dict(data.get("mean_rho_by_candidate") or {}),
        "seed_rho_by_candidate": _stringify_seed_keys(
            data.get("seed_rho_by_candidate") or {}
        ),
        "recovery_by_seed": _stringify_seed_keys(data.get("recovery_by_seed") or {}),
        "recovery_rho_mean": data.get("recovery_rho_mean"),
        "rho_shift_mean": data.get("rho_shift_mean"),
        "rho_shift_by_seed": _stringify_seed_keys(data.get("rho_shift_by_seed") or {}),
        "notes": list(data.get("notes") or data.get("interpretation_notes") or []),
    }

    # Nested summary blocks (design example shape)
    best = payload["best_local"]
    mean_rho = payload["mean_rho_by_candidate"]
    seed_rho = payload["seed_rho_by_candidate"]
    best_seed: dict[str, float] = {}
    if best and best in seed_rho:
        for k, v in seed_rho[best].items():
            fv = _finite_or_none(v)
            if fv is not None:
                best_seed[str(k)] = fv
    floor = float(payload["thresholds"].get("stationary_seed_floor", 0.50))
    seeds_above = sum(1 for v in best_seed.values() if v >= floor)
    best_vals = list(best_seed.values())
    if best_vals:
        best_rho_mean: float | None = float(sum(best_vals) / len(best_vals))
        if len(best_vals) > 1:
            best_rho_std: float | None = float(np.std(best_vals, ddof=0))
        else:
            best_rho_std = 0.0
    else:
        # Fall back to candidate mean map when per-seed map is empty.
        fallback = mean_rho.get(best) if best else None
        best_rho_mean = _finite_or_none(fallback)
        best_rho_std = None

    payload["stationary"] = {
        "best_local": best,
        "best_rho_mean": best_rho_mean,
        "best_rho_std": best_rho_std,
        "per_seed_rho": best_seed,
        "seeds_above_floor": seeds_above,
    }
    rec_by_seed = payload["recovery_by_seed"]
    seeds_recovered = sum(
        1
        for rec in rec_by_seed.values()
        if isinstance(rec, Mapping) and rec.get("recovered_abs")
    )
    # recovery_steps_mean: mean finite t_star_abs across seeds (optional design field)
    t_stars = []
    for rec in rec_by_seed.values():
        if isinstance(rec, Mapping):
            t = _finite_or_none(rec.get("t_star_abs"))
            if t is not None:
                t_stars.append(t)
    recovery_steps_mean = float(sum(t_stars) / len(t_stars)) if t_stars else None

    payload["task_shift"] = {
        "best_local": best,
        "recovery_rho_mean": payload["recovery_rho_mean"],
        "recovery_rho_definition": RECOVERY_RHO_DEFINITION,
        "recovery_steps_mean": recovery_steps_mean,
        "seeds_recovered": seeds_recovered,
        "recovered": payload["recovered"],
        "gt_shift_disagreement_mean": payload["rho_shift_mean"],
    }
    payload["interpretation_notes"] = payload["notes"]

    if extra:
        reserved = set(VERDICT_JSON_REQUIRED_FIELDS) | {
            "stationary",
            "task_shift",
            "interpretation_notes",
            "notes",
            "mean_rho_by_candidate",
            "seed_rho_by_candidate",
            "recovery_by_seed",
            "recovery_rho_mean",
            "rho_shift_mean",
            "rho_shift_by_seed",
        }
        clobber = sorted(k for k in extra if k in reserved)
        if clobber:
            raise ValueError(
                f"extra= cannot clobber reserved verdict fields: {clobber}. "
                "Put non-schema annotations under a non-reserved key."
            )
        payload.update(dict(extra))

    return _json_safe(payload)


def _stringify_seed_keys(obj: Any) -> Any:
    """JSON-friendly: convert int seed keys to strings in nested dicts."""
    if not isinstance(obj, Mapping):
        return obj
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = str(k)
        if isinstance(v, Mapping):
            # candidate → {seed → rho} or seed → recovery dict
            inner: dict[str, Any] = {}
            for ik, iv in v.items():
                inner[str(ik)] = iv
            out[key] = inner
        else:
            out[key] = v
    return out


def write_verdict_json(
    path: str | Path,
    result: VerdictResult | Mapping[str, Any],
    *,
    experiment_id: str,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write portable ``verdict.json`` with required schema fields.

    Non-finite floats are written as JSON ``null`` (``allow_nan=False``).

    Parameters
    ----------
    overwrite
        If False and ``path`` already exists, raise :class:`FileExistsError`.
        Use ``overwrite=False`` for primary aggregate paths when sensitivity
        runs must not clobber ``aggregate/verdict.json``.
    """
    path = Path(path)
    if path.is_file() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing verdict at {path}; "
            "pass overwrite=True only for deliberate primary rewrites"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = verdict_payload(result, experiment_id=experiment_id, extra=extra)
    missing = [f for f in VERDICT_JSON_REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"verdict payload missing required fields: {missing}")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure to ``path`` (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path
