"""Production snapshot-isolation guard.

``signals_disabled()`` is the safety mechanism (always on during GT). The bitwise
trace assertion is a **live verification** of that mechanism on the real run.

Benchmarks on full MNIST-scale MLP + 4 candidates (CPU):
  isolation capture+assert ≈ 3–4 ms
  train step ≈ 3 ms
  GT grad_abs snapshot ≈ 3 ms
  ~300 snapshots × 3.3 ms ≈ **1 s** total per seed

Cheap relative to a multi-hour Phase 0 run — default mode is therefore ``always``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IsolationMode = Literal["always", "first", "off"]

VALID_ISOLATION_MODES: tuple[IsolationMode, ...] = ("always", "first", "off")


def parse_isolation_mode(value: object, *, default: IsolationMode = "always") -> IsolationMode:
    """Parse config value into an isolation mode."""
    if value is None:
        return default
    if isinstance(value, bool):
        # Backward-compat: true → always, false → off
        return "always" if value else "off"
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return "always"
    if text in ("false", "0", "no", "off"):
        return "off"
    if text in VALID_ISOLATION_MODES:
        return text  # type: ignore[return-value]
    raise ValueError(
        f"unknown check_isolation mode {value!r}; expected one of {VALID_ISOLATION_MODES}"
    )


@dataclass
class IsolationGuard:
    """Decides whether to run the bitwise isolation assert on each snapshot.

    Modes
    -----
    always
        Assert on every snapshot (recommended default for reported runs).
    first
        Assert only on the first snapshot of this guard's lifetime (per seed /
        runner instance) — canary with near-zero cost.
    off
        Never assert (mechanism still uses ``signals_disabled``; no live check).
    """

    mode: IsolationMode = "always"
    _checked: bool = False

    def should_check(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "always":
            return True
        # first
        return not self._checked

    def note_checked(self) -> None:
        """Call after a snapshot that actually ran the assert (or attempted it)."""
        self._checked = True

    def for_snapshot(self) -> bool:
        """Return whether to check this snapshot and advance the canary state.

        When mode is ``first``, the first call returns True and subsequent calls
        return False. When ``always``, every call returns True.
        """
        do = self.should_check()
        if do:
            self.note_checked()
        return do
