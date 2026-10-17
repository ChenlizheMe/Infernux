"""Unified error-reporting policy: visibility, counting and reset."""

import pytest

from infernux.debug import Debug, DebugConsole, LogType


@pytest.fixture
def console(monkeypatch):
    previous = DebugConsole._instance
    DebugConsole._instance = None
    instance = DebugConsole.instance()
    try:
        yield instance
    finally:
        DebugConsole._instance = previous


def _caught(message: str = "boom") -> ValueError:
    try:
        raise ValueError(message)
    except ValueError as exc:
        return exc


def test_script_error_keeps_the_first_traceback(console):
    Debug.report_script_error(_caught(), "Probe.update")

    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].log_type is LogType.EXCEPTION
    assert "[Script @ Probe.update]" in entries[0].message
    assert "ValueError: boom" in entries[0].message
    assert "ValueError: boom" in entries[0].stack_trace


def test_script_error_counts_repeats_and_reports_at_ten(console):
    for _ in range(10):
        Debug.report_script_error(_caught(), "Probe.update")

    entries = console.get_entries()
    assert len(entries) == 2
    assert "(repeated 10 times)" in entries[1].message
    # The full stack was already published with the first occurrence.
    assert entries[1].stack_trace == ""

    Debug.report_script_error(_caught(), "Probe.update")
    assert len(console.get_entries()) == 2


def test_recovered_failures_are_visible_without_being_errors(console):
    Debug.log_suppressed("AssetWatcher.poll", _caught("gone"))

    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].log_type is LogType.WARNING
    assert "[Recovered @ AssetWatcher.poll]" in entries[0].message
    assert "ValueError: gone" in entries[0].stack_trace


def test_report_sites_are_counted_independently(console):
    Debug.report_internal_error(_caught(), "Panel:A")
    Debug.report_internal_error(_caught(), "Panel:B")

    assert len(console.get_entries()) == 2


def test_clearing_the_console_restarts_counting(console):
    Debug.report_script_error(_caught(), "Probe.update")
    Debug.report_script_error(_caught(), "Probe.update")
    assert len(console.get_entries()) == 1

    Debug.clear_console()
    Debug.report_script_error(_caught(), "Probe.update")

    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].stack_trace != ""
