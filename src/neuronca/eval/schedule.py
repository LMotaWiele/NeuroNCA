"""Evaluation snapshot schedules for stationary and task-shift conditions.

Produces tags for:
- stationary: early / mid / late (and optional step_frac)
- task-shift switch boundary: late_a, switch_b0
- dense lag: every lag_stride for lag_window_steps after first B optimizer step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SnapshotKind = Literal[
    "stationary",
    "switch",
    "lag",
    "coarse",
    "epoch_end",
    "step_frac",
]


@dataclass(frozen=True)
class SnapshotSpec:
    """A single evaluation snapshot request."""

    tag: str
    kind: SnapshotKind
    # Optional timing metadata (filled by runners; schedule may leave None)
    epoch: int | None = None
    step: int | None = None
    steps_after_switch: int | None = None
    # GT suite hint: "full" = GT-1+GT-2; "gt1" = lag dense
    gt_suite: Literal["full", "gt1"] = "full"
    active_transform: Literal["A", "B", "none"] | None = None
    optimizer: bool = False  # whether an optimizer step is allowed at this point


@dataclass
class EvalSchedule:
    """Collection of snapshot specs for one condition."""

    condition: str
    snapshots: list[SnapshotSpec] = field(default_factory=list)

    def tags(self) -> list[str]:
        return [s.tag for s in self.snapshots]

    def by_kind(self, kind: SnapshotKind) -> list[SnapshotSpec]:
        return [s for s in self.snapshots if s.kind == kind]


def stationary_snapshot_tags(
    epochs: int = 20,
    *,
    include_step_frac: bool = False,
    total_steps: int | None = None,
) -> list[SnapshotSpec]:
    """Stationary condition snapshots: early / mid / late.

    | Tag   | When                        |
    |-------|-----------------------------|
    | early | end of epoch 1              |
    | mid   | end of epoch epochs // 2    |
    | late  | end of final epoch          |

    Optionally add step_frac tags at {0.1, 0.25, 0.5, 0.75, 1.0} × T when
    ``include_step_frac`` and ``total_steps`` are set.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    early_epoch = 1
    mid_epoch = max(1, epochs // 2)
    late_epoch = epochs

    specs: list[SnapshotSpec] = [
        SnapshotSpec(
            tag="early",
            kind="stationary",
            epoch=early_epoch,
            gt_suite="full",
            active_transform="A",
        ),
        SnapshotSpec(
            tag="mid",
            kind="stationary",
            epoch=mid_epoch,
            gt_suite="full",
            active_transform="A",
        ),
        SnapshotSpec(
            tag="late",
            kind="stationary",
            epoch=late_epoch,
            gt_suite="full",
            active_transform="A",
        ),
    ]

    if include_step_frac:
        if total_steps is None or total_steps < 1:
            raise ValueError("total_steps required when include_step_frac=True")
        fracs = (0.1, 0.25, 0.5, 0.75, 1.0)
        covered_epochs = {early_epoch, mid_epoch, late_epoch}
        for f in fracs:
            step = int(round(f * total_steps))
            tag = f"step_frac_{f:g}"
            # Avoid pure duplicates of late when f=1.0 lands on T; still emit
            # step_frac tags as distinct schedule entries per design.
            specs.append(
                SnapshotSpec(
                    tag=tag,
                    kind="step_frac",
                    step=step,
                    gt_suite="full",
                    active_transform="A",
                )
            )
        _ = covered_epochs  # reserved for future de-duplication

    return specs


def switch_snapshot_tags() -> list[SnapshotSpec]:
    """Ordered switch boundary tags: late_a then switch_b0.

    late_a / pre_switch_a: transform A, full GT, no optimizer (I_pre).
    switch_b0: transform B, same weights, no optimizer, full GT (I_immediate_post).
    """
    return [
        SnapshotSpec(
            tag="late_a",
            kind="switch",
            gt_suite="full",
            active_transform="A",
            optimizer=False,
        ),
        SnapshotSpec(
            tag="switch_b0",
            kind="switch",
            gt_suite="full",
            active_transform="B",
            optimizer=False,
        ),
    ]


def lag_snapshot_tags(
    lag_window_steps: int = 500,
    lag_stride: int = 25,
    *,
    include_t0: bool = True,
) -> list[SnapshotSpec]:
    """Dense lag tags every ``lag_stride`` for ``lag_window_steps``.

    t = 0 is the first GT-1 snapshot **after** the first B optimizer step.
    Subsequent points at t = lag_stride, 2*lag_stride, ..., lag_window_steps.
    ``switch_b0`` is not a lag point.
    """
    if lag_stride < 1:
        raise ValueError("lag_stride must be >= 1")
    if lag_window_steps < 0:
        raise ValueError("lag_window_steps must be >= 0")

    times: list[int] = []
    if include_t0:
        times.append(0)
    t = lag_stride
    while t <= lag_window_steps:
        times.append(t)
        t += lag_stride
    # Ensure window end is included even if not a multiple of stride
    if lag_window_steps not in times and lag_window_steps > 0:
        times.append(lag_window_steps)
    times = sorted(set(times))

    return [
        SnapshotSpec(
            tag=f"lag_t{t}",
            kind="lag",
            steps_after_switch=t,
            step=t,
            gt_suite="gt1",
            active_transform="B",
            optimizer=True,
        )
        for t in times
    ]


def lag_times(
    lag_window_steps: int = 500,
    lag_stride: int = 25,
    *,
    include_t0: bool = True,
) -> list[int]:
    """Integer lag times corresponding to :func:`lag_snapshot_tags`."""
    return [
        int(s.steps_after_switch)  # type: ignore[arg-type]
        for s in lag_snapshot_tags(
            lag_window_steps, lag_stride, include_t0=include_t0
        )
    ]


def task_shift_schedule(
    epochs_a: int = 15,
    epochs_b: int = 15,
    *,
    lag_window_steps: int = 500,
    lag_stride: int = 25,
    include_a_early_mid: bool = True,
) -> EvalSchedule:
    """Full task-shift evaluation schedule (A phase + switch + lag + B coarse)."""
    snaps: list[SnapshotSpec] = []

    if include_a_early_mid and epochs_a >= 1:
        snaps.append(
            SnapshotSpec(
                tag="early",
                kind="stationary",
                epoch=1,
                gt_suite="full",
                active_transform="A",
            )
        )
        if epochs_a >= 2:
            snaps.append(
                SnapshotSpec(
                    tag="mid",
                    kind="stationary",
                    epoch=max(1, epochs_a // 2),
                    gt_suite="full",
                    active_transform="A",
                )
            )

    snaps.extend(switch_snapshot_tags())
    snaps.extend(lag_snapshot_tags(lag_window_steps, lag_stride))

    # Coarse B-phase: mid-B epoch, late-B
    if epochs_b >= 1:
        mid_b = max(1, epochs_b // 2)
        snaps.append(
            SnapshotSpec(
                tag="mid_b",
                kind="coarse",
                epoch=mid_b,
                gt_suite="full",
                active_transform="B",
                optimizer=True,
            )
        )
        snaps.append(
            SnapshotSpec(
                tag="late_b",
                kind="coarse",
                epoch=epochs_b,
                gt_suite="full",
                active_transform="B",
                optimizer=True,
            )
        )

    return EvalSchedule(condition="task_shift", snapshots=snaps)


def stationary_schedule(
    epochs: int = 20,
    *,
    include_step_frac: bool = False,
    total_steps: int | None = None,
) -> EvalSchedule:
    """Stationary condition schedule wrapper."""
    return EvalSchedule(
        condition="stationary",
        snapshots=stationary_snapshot_tags(
            epochs,
            include_step_frac=include_step_frac,
            total_steps=total_steps,
        ),
    )
