"""Pre-registered multi-seed GEVB FA-alignment verdict (frozen before primary run).

This is **not** soft guidance. Thresholds live in ``configs/default.yaml`` under
``eval.thresholds`` and are frozen into ``meta/thresholds.json`` by the ledger
with the same scientific integrity rules as Spearman ρ bars.

Primary Phase 0 ρ verdict (PASS / PARTIAL_PASS / FAIL) is **unchanged**. This
module only annotates whether fixed-random feedback was *aligned enough* at
the stationary late snapshot for GEVB-RF Spearman to be interpretable as a
gradient proxy rather than "B_ℓ still orthogonal to ∂L/∂z".

--------------------------------------------------------------------
Pre-registered decision rule (do not retune after seeing primary results)
--------------------------------------------------------------------

Per seed, at stationary snapshot tag ``late`` (or the last recorded stationary
alignment tag if smoke collapses early=mid=late):

    layer_mean_fa_s = mean over layers of fa_signal_angle_deg
                      (finite values only; require ≥ min_layers)

Aggregate:

    mean_late_fa = mean_s layer_mean_fa_s

**CONVERGED** iff:
  1. mean_late_fa ≤ gevb_fa_converged_mean_deg  (default **60°**)
  2. ≥ gevb_fa_seed_floor_min_count seeds have
        layer_mean_fa_s ≤ gevb_fa_seed_floor_deg  (default **70°**, min count **2**)

**STALLED** iff:
  1. mean_late_fa ≥ gevb_fa_stalled_mean_deg  (default **75°**)
  **OR**
  2. mean improvement early→late < gevb_fa_min_improvement_deg (default **10°**)
     AND mean_late_fa ≥ gevb_fa_seed_floor_deg (still high at late)

**PARTIAL** otherwise (60° < mean_late_fa < 75°, or floors not met).

Trajectory notes (early→late) are logged for plots only; they do **not**
override CONVERGED/STALLED once late bars are applied, except the STALLED
improvement clause above which uses a **pre-registered** minimum improvement.

--------------------------------------------------------------------
Interpretation matrix (GEVB ρ × alignment) — annotation only
--------------------------------------------------------------------

| alignment_status | GEVB late edge ρ vs grad_abs | Allowed interpretation |
|------------------|-----------------------------|-------------------------|
| CONVERGED        | high / best_local           | FA working as theory suggests |
| CONVERGED        | low / last                  | Serious: aligned teaching signal still poor proxy |
| STALLED          | low / last                  | Expected under FA lag; candidate not validated *at this budget* |
| STALLED          | high                        | Suspicious; investigate before trusting |
| PARTIAL          | anything                    | Inconclusive for FA; do not claim either success or architecture failure |

**Hard process rules (pre-registered):**
1. Do **not** extend epoch budgets, retune B_ℓ scale, or re-draw feedback seeds
   based on trajectories observed in the primary run.
2. "Maybe it just needs more training" is **only** legitimate as a **separate**
   follow-up experiment id with its own frozen ledger, using the follow-up
   criteria below — never as a live decision while staring at primary numbers.
3. Primary Phase 0 ρ PASS/FAIL is independent; alignment annotations never flip
   ``phase1_go`` by themselves.

--------------------------------------------------------------------
Follow-up experiment (optional, separate experiment_id)
--------------------------------------------------------------------

Only if primary alignment_status ∈ {STALLED, PARTIAL} **and** a follow-up is
explicitly launched (new ledger / new output dir):

- Config change allowed: double stationary epochs (40) and task-shift
  epochs_a/b (30/30); **no** change to B_ℓ init, residual definition, or
  alignment angle thresholds.
- Success criterion (pre-registered): same CONVERGED rule on the follow-up
  late snapshot (mean_late_fa ≤ 60° and seed floor).
- Failure: still STALLED or PARTIAL → treat GEVB-RF as not validated for this
  architecture at the expanded budget; do not iterate further under Phase 0.

Smoke runs (1 seed, 1 epoch) are expected to report STALLED / insufficient
seeds; they must not be used to retune these thresholds.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Frozen defaults (must match configs/default.yaml eval.thresholds)
# ---------------------------------------------------------------------------

DEFAULT_FA_CONVERGED_MEAN_DEG = 60.0
DEFAULT_FA_STALLED_MEAN_DEG = 75.0
DEFAULT_FA_SEED_FLOOR_DEG = 70.0
DEFAULT_FA_SEED_FLOOR_MIN_COUNT = 2
DEFAULT_FA_MIN_IMPROVEMENT_DEG = 10.0
DEFAULT_FA_MIN_LAYERS = 3  # of 4 (fc1,fc2,fc3,fc_out); tolerate one nan layer
DEFAULT_LATE_TAG = "late"
DEFAULT_EARLY_TAG = "early"

ALIGNMENT_CONVERGED = "CONVERGED"
ALIGNMENT_STALLED = "STALLED"
ALIGNMENT_PARTIAL = "PARTIAL"
ALIGNMENT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

ALIGNMENT_STATUS_VALUES = (
    ALIGNMENT_CONVERGED,
    ALIGNMENT_STALLED,
    ALIGNMENT_PARTIAL,
    ALIGNMENT_INSUFFICIENT_DATA,
)


@dataclass(frozen=True)
class GevbAlignmentThresholds:
    """Pre-registered FA alignment bars (frozen at ledger time)."""

    gevb_fa_converged_mean_deg: float = DEFAULT_FA_CONVERGED_MEAN_DEG
    gevb_fa_stalled_mean_deg: float = DEFAULT_FA_STALLED_MEAN_DEG
    gevb_fa_seed_floor_deg: float = DEFAULT_FA_SEED_FLOOR_DEG
    gevb_fa_seed_floor_min_count: int = DEFAULT_FA_SEED_FLOOR_MIN_COUNT
    gevb_fa_min_improvement_deg: float = DEFAULT_FA_MIN_IMPROVEMENT_DEG
    gevb_fa_min_layers: int = DEFAULT_FA_MIN_LAYERS
    gevb_fa_late_tag: str = DEFAULT_LATE_TAG
    gevb_fa_early_tag: str = DEFAULT_EARLY_TAG

    @classmethod
    def from_mapping(cls, thr: Mapping[str, Any] | None) -> GevbAlignmentThresholds:
        thr = dict(thr or {})
        return cls(
            gevb_fa_converged_mean_deg=float(
                thr.get("gevb_fa_converged_mean_deg", DEFAULT_FA_CONVERGED_MEAN_DEG)
            ),
            gevb_fa_stalled_mean_deg=float(
                thr.get("gevb_fa_stalled_mean_deg", DEFAULT_FA_STALLED_MEAN_DEG)
            ),
            gevb_fa_seed_floor_deg=float(
                thr.get("gevb_fa_seed_floor_deg", DEFAULT_FA_SEED_FLOOR_DEG)
            ),
            gevb_fa_seed_floor_min_count=int(
                thr.get(
                    "gevb_fa_seed_floor_min_count", DEFAULT_FA_SEED_FLOOR_MIN_COUNT
                )
            ),
            gevb_fa_min_improvement_deg=float(
                thr.get(
                    "gevb_fa_min_improvement_deg", DEFAULT_FA_MIN_IMPROVEMENT_DEG
                )
            ),
            gevb_fa_min_layers=int(
                thr.get("gevb_fa_min_layers", DEFAULT_FA_MIN_LAYERS)
            ),
            gevb_fa_late_tag=str(thr.get("gevb_fa_late_tag", DEFAULT_LATE_TAG)),
            gevb_fa_early_tag=str(thr.get("gevb_fa_early_tag", DEFAULT_EARLY_TAG)),
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass
class GevbAlignmentVerdict:
    alignment_status: str
    mean_late_fa_deg: float
    mean_early_fa_deg: float
    mean_improvement_deg: float
    seeds_below_floor: int
    n_seeds: int
    per_seed_late_layer_mean: dict[int, float]
    per_seed_early_layer_mean: dict[int, float]
    thresholds: dict[str, float | int | str]
    notes: list[str] = field(default_factory=list)
    # Process commitments recorded in every verdict artifact
    process_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROCESS_RULES: tuple[str, ...] = (
    "Do not extend epoch budgets or retune B_ℓ / residual definition based on "
    "primary-run FA trajectories.",
    "CONVERGED / STALLED / PARTIAL use only pre-registered angle thresholds "
    "frozen in meta/thresholds.json.",
    "'Maybe more training' requires a separate follow-up experiment_id with "
    "doubled epochs and the same CONVERGED bar — not a live re-decision.",
    "This annotation does not flip phase1_go; primary ρ verdict is independent.",
)


def layer_mean_fa_angle(
    layer_angles: Mapping[str, float],
    *,
    min_layers: int = DEFAULT_FA_MIN_LAYERS,
) -> float:
    """Mean FA angle over layers; nan if fewer than min_layers finite values."""
    vals = [float(v) for v in layer_angles.values() if math.isfinite(float(v))]
    if len(vals) < int(min_layers):
        return float("nan")
    return float(np.mean(vals))


def load_seed_tag_layer_means(
    alignment_csv: Path,
    *,
    tag: str,
    min_layers: int = DEFAULT_FA_MIN_LAYERS,
) -> float:
    """Read gevb_alignment.csv and return layer-mean FA angle at ``tag``."""
    if not alignment_csv.is_file():
        return float("nan")
    by_layer: dict[str, float] = {}
    with alignment_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("snapshot_tag")) != tag:
                continue
            layer = str(row.get("layer") or "")
            raw = row.get("fa_signal_angle_deg")
            if not layer or raw in (None, ""):
                continue
            try:
                ang = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ang):
                by_layer[layer] = ang
    return layer_mean_fa_angle(by_layer, min_layers=min_layers)


def collect_alignment_from_results(
    results_root: Path,
    seeds: Sequence[int],
    *,
    late_tag: str = DEFAULT_LATE_TAG,
    early_tag: str = DEFAULT_EARLY_TAG,
    min_layers: int = DEFAULT_FA_MIN_LAYERS,
    condition: str = "stationary",
) -> tuple[dict[int, float], dict[int, float]]:
    """seed → late layer-mean FA; seed → early layer-mean FA."""
    late: dict[int, float] = {}
    early: dict[int, float] = {}
    for seed in seeds:
        path = (
            Path(results_root)
            / condition
            / f"seed_{int(seed)}"
            / "gevb_alignment.csv"
        )
        late[int(seed)] = load_seed_tag_layer_means(
            path, tag=late_tag, min_layers=min_layers
        )
        early[int(seed)] = load_seed_tag_layer_means(
            path, tag=early_tag, min_layers=min_layers
        )
        # Smoke collapse: early=mid=late may only write one tag (often early).
        # If late is missing but early exists, treat early as the only late point
        # for smoke interpretability — full runs write true late.
        if not math.isfinite(late[int(seed)]) and math.isfinite(early[int(seed)]):
            # Prefer last tag present in file
            late[int(seed)] = _fallback_last_tag_mean(path, min_layers=min_layers)
    return late, early


def _fallback_last_tag_mean(path: Path, *, min_layers: int) -> float:
    if not path.is_file():
        return float("nan")
    # Group by step, take max step
    by_step: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(float(row.get("step", -1)))
            except (TypeError, ValueError):
                continue
            layer = str(row.get("layer") or "")
            try:
                ang = float(row.get("fa_signal_angle_deg"))
            except (TypeError, ValueError):
                continue
            if layer and math.isfinite(ang):
                by_step.setdefault(step, {})[layer] = ang
    if not by_step:
        return float("nan")
    last = max(by_step.keys())
    return layer_mean_fa_angle(by_step[last], min_layers=min_layers)


def decide_alignment_status(
    late_by_seed: Mapping[int, float],
    early_by_seed: Mapping[int, float] | None = None,
    *,
    thresholds: GevbAlignmentThresholds | None = None,
) -> GevbAlignmentVerdict:
    """Apply pre-registered CONVERGED / STALLED / PARTIAL rules."""
    thr = thresholds or GevbAlignmentThresholds()
    early_by_seed = dict(early_by_seed or {})

    late_finite = {
        int(s): float(v)
        for s, v in late_by_seed.items()
        if math.isfinite(float(v))
    }
    n_seeds = len(late_finite)
    notes: list[str] = []

    if n_seeds == 0:
        return GevbAlignmentVerdict(
            alignment_status=ALIGNMENT_INSUFFICIENT_DATA,
            mean_late_fa_deg=float("nan"),
            mean_early_fa_deg=float("nan"),
            mean_improvement_deg=float("nan"),
            seeds_below_floor=0,
            n_seeds=0,
            per_seed_late_layer_mean={int(s): float(v) for s, v in late_by_seed.items()},
            per_seed_early_layer_mean={
                int(s): float(v) for s, v in early_by_seed.items()
            },
            thresholds=thr.to_dict(),
            notes=["no finite late FA layer-means found"],
            process_rules=list(_PROCESS_RULES),
        )

    mean_late = float(np.mean(list(late_finite.values())))
    seeds_below = sum(
        1 for v in late_finite.values() if v <= thr.gevb_fa_seed_floor_deg
    )

    early_finite = {
        int(s): float(v)
        for s, v in early_by_seed.items()
        if math.isfinite(float(v))
    }
    # Pair seeds that have both early and late
    paired_imp: list[float] = []
    for s, late_v in late_finite.items():
        if s in early_finite:
            paired_imp.append(early_finite[s] - late_v)
    mean_early = (
        float(np.mean(list(early_finite.values()))) if early_finite else float("nan")
    )
    mean_imp = float(np.mean(paired_imp)) if paired_imp else float("nan")

    # CONVERGED
    converged = (
        mean_late <= thr.gevb_fa_converged_mean_deg
        and seeds_below >= thr.gevb_fa_seed_floor_min_count
    )

    # STALLED
    stalled_high = mean_late >= thr.gevb_fa_stalled_mean_deg
    stalled_flat = (
        math.isfinite(mean_imp)
        and mean_imp < thr.gevb_fa_min_improvement_deg
        and mean_late >= thr.gevb_fa_seed_floor_deg
    )
    stalled = stalled_high or stalled_flat

    # Insufficient seeds for floor (e.g. smoke n=1): cannot CONVERGED
    if n_seeds < thr.gevb_fa_seed_floor_min_count:
        notes.append(
            f"only {n_seeds} seed(s) with finite late FA; "
            f"need ≥{thr.gevb_fa_seed_floor_min_count} for CONVERGED floor "
            f"(smoke runs are expected to be INSUFFICIENT_DATA or STALLED)."
        )
        if converged:
            # demote: mean looks good but seed count fails formal rule
            converged = False
            notes.append(
                "mean_late_fa would pass converged bar but seed floor count fails."
            )

    if converged and not stalled:
        status = ALIGNMENT_CONVERGED
    elif stalled and not converged:
        status = ALIGNMENT_STALLED
        if stalled_high:
            notes.append(
                f"mean_late_fa={mean_late:.1f}° ≥ stalled bar "
                f"{thr.gevb_fa_stalled_mean_deg}°."
            )
        if stalled_flat:
            notes.append(
                f"early→late improvement={mean_imp:.1f}° < "
                f"{thr.gevb_fa_min_improvement_deg}° with late still high."
            )
    elif converged and stalled:
        # Pathological: both true → prefer STALLED (conservative)
        status = ALIGNMENT_STALLED
        notes.append(
            "both CONVERGED and STALLED predicates true; conservative STALLED."
        )
    elif n_seeds < thr.gevb_fa_seed_floor_min_count and mean_late >= thr.gevb_fa_stalled_mean_deg:
        status = ALIGNMENT_STALLED
    elif n_seeds < thr.gevb_fa_seed_floor_min_count:
        status = ALIGNMENT_INSUFFICIENT_DATA
    else:
        status = ALIGNMENT_PARTIAL
        notes.append(
            f"mean_late_fa={mean_late:.1f}° between converged "
            f"({thr.gevb_fa_converged_mean_deg}°) and stalled "
            f"({thr.gevb_fa_stalled_mean_deg}°) bars, or seed floor not met."
        )

    notes.append(
        "Follow-up (doubled epochs, same bars) only if status is STALLED or "
        "PARTIAL and launched as a new experiment_id — never mid-primary."
    )

    return GevbAlignmentVerdict(
        alignment_status=status,
        mean_late_fa_deg=mean_late,
        mean_early_fa_deg=mean_early,
        mean_improvement_deg=mean_imp,
        seeds_below_floor=seeds_below,
        n_seeds=n_seeds,
        per_seed_late_layer_mean={int(s): float(v) for s, v in late_by_seed.items()},
        per_seed_early_layer_mean={
            int(s): float(v) for s, v in early_by_seed.items()
        },
        thresholds=thr.to_dict(),
        notes=notes,
        process_rules=list(_PROCESS_RULES),
    )


def aggregate_gevb_alignment_verdict(
    results_root: Path | str,
    seeds: Sequence[int],
    *,
    thresholds: Mapping[str, Any] | GevbAlignmentThresholds | None = None,
    condition: str = "stationary",
) -> GevbAlignmentVerdict:
    """Load per-seed gevb_alignment.csv and apply pre-registered rules."""
    if isinstance(thresholds, GevbAlignmentThresholds):
        thr = thresholds
    else:
        thr = GevbAlignmentThresholds.from_mapping(thresholds)

    late, early = collect_alignment_from_results(
        Path(results_root),
        seeds,
        late_tag=thr.gevb_fa_late_tag,
        early_tag=thr.gevb_fa_early_tag,
        min_layers=thr.gevb_fa_min_layers,
        condition=condition,
    )
    return decide_alignment_status(late, early, thresholds=thr)
