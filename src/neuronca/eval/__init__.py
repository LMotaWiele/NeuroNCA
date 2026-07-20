"""Correlations, time-lag analysis, schedules, multi-seed verdict."""

from neuronca.eval.correlation import (
    CorrelationResult,
    LayerCorrelation,
    correlate_layers,
    pearson_r,
    spearman_rho,
)
from neuronca.eval.schedule import (
    EvalSchedule,
    SnapshotSpec,
    lag_snapshot_tags,
    lag_times,
    stationary_schedule,
    stationary_snapshot_tags,
    switch_snapshot_tags,
    task_shift_schedule,
)
from neuronca.eval.time_lag import (
    LagRecoveryResult,
    analyze_lag_curve,
    compute_rho_rec,
    first_crossing,
)
from neuronca.eval.gevb_alignment_verdict import (
    ALIGNMENT_CONVERGED,
    ALIGNMENT_INSUFFICIENT_DATA,
    ALIGNMENT_PARTIAL,
    ALIGNMENT_STALLED,
    GevbAlignmentThresholds,
    GevbAlignmentVerdict,
    aggregate_gevb_alignment_verdict,
    decide_alignment_status,
)
from neuronca.eval.verdict import (
    SeedLagCurve,
    VerdictResult,
    VerdictThresholds,
    aggregate_verdict,
    decide_verdict,
    select_best_local,
)

__all__ = [
    "ALIGNMENT_CONVERGED",
    "ALIGNMENT_INSUFFICIENT_DATA",
    "ALIGNMENT_PARTIAL",
    "ALIGNMENT_STALLED",
    "GevbAlignmentThresholds",
    "GevbAlignmentVerdict",
    "aggregate_gevb_alignment_verdict",
    "decide_alignment_status",
    "CorrelationResult",
    "LayerCorrelation",
    "correlate_layers",
    "pearson_r",
    "spearman_rho",
    "EvalSchedule",
    "SnapshotSpec",
    "lag_snapshot_tags",
    "lag_times",
    "stationary_schedule",
    "stationary_snapshot_tags",
    "switch_snapshot_tags",
    "task_shift_schedule",
    "LagRecoveryResult",
    "analyze_lag_curve",
    "compute_rho_rec",
    "first_crossing",
    "SeedLagCurve",
    "VerdictResult",
    "VerdictThresholds",
    "aggregate_verdict",
    "decide_verdict",
    "select_best_local",
]
