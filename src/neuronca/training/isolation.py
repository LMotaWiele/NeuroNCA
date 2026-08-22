"""Live bitwise check that GT did not mutate signal traces.

``signals_disabled()`` is the safety mechanism; this guard is the verification.
Default ``always`` — cheap vs a full Phase 0 run. See ``docs/design/invariants.md``.
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
    """When to assert traces across a snapshot: ``always``, ``first``, or ``off``."""

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
        """Whether to assert this snapshot (advances ``first``-mode canary)."""
        do = self.should_check()
        if do:
            self.note_checked()
        return do
