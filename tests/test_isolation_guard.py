"""IsolationGuard modes and config parsing."""

from neuronca.training.isolation import IsolationGuard, parse_isolation_mode


def test_parse_isolation_mode_aliases():
    assert parse_isolation_mode("always") == "always"
    assert parse_isolation_mode("first") == "first"
    assert parse_isolation_mode("off") == "off"
    assert parse_isolation_mode(True) == "always"
    assert parse_isolation_mode(False) == "off"
    assert parse_isolation_mode(None) == "always"


def test_guard_always_checks_every_time():
    g = IsolationGuard(mode="always")
    assert g.for_snapshot() is True
    assert g.for_snapshot() is True
    assert g.for_snapshot() is True


def test_guard_first_is_canary():
    g = IsolationGuard(mode="first")
    assert g.for_snapshot() is True
    assert g.for_snapshot() is False
    assert g.for_snapshot() is False


def test_guard_off():
    g = IsolationGuard(mode="off")
    assert g.for_snapshot() is False
    assert g.for_snapshot() is False
