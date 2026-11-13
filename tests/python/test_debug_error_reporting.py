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


def test_script_error_keeps_the_first_traceback(console, tmp_path):
    Debug.report_script_error(_caught(), "Probe.update")

    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].log_type is LogType.EXCEPTION
    assert "[Script @ Probe.update]" in entries[0].message
    assert "ValueError: boom" in entries[0].message
    assert "ValueError: boom" in entries[0].stack_trace
    assert entries[0].source_file == __file__

    console._project_root = str(tmp_path)
    script = str(tmp_path / "Assets" / "Scripts" / "Combat.py")
    namespace = {}
    source = "import random\ndef update():\n    fire()\ndef fire():\n    random.choice(())\n"
    exec(compile(source, script, "exec"), namespace)
    try:
        namespace["update"]()
    except IndexError as exc:
        Debug.report_script_error(exc, "Combat.update")

    entry = console.get_entries()[-1]
    assert entry.source_file == script and entry.source_line == 5
    assert "random.py" in entry.stack_trace
    assert "IndexError: Cannot choose from an empty sequence" in entry.message


def test_script_error_updates_one_problem_and_keeps_its_traceback(console):
    for _ in range(10):
        Debug.report_script_error(_caught(), "Probe.update")

    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].occurrences == 10
    assert "ValueError: boom" in entries[0].stack_trace

    Debug.report_script_error(_caught(), "Probe.update")
    assert len(console.get_entries()) == 1
    assert entries[0].occurrences == 11
    for message in ("different error", "boom"):
        Debug.report_script_error(_caught(message), "Probe.update")
        assert len(console.get_entries()) == 1
        assert f"ValueError: {message}" in entries[0].message
        assert f"ValueError: {message}" in entries[0].stack_trace
    assert entries[0].occurrences == 13


def test_diagnostic_owners_resolve_independently_and_preserve_history(console):
    from datetime import datetime
    from infernux.debug import LogEntry

    source = "Assets/共同源.py"
    Debug.log_error("user history", source_file=source)
    console.log(LogEntry("bad interface", LogType.ERROR, datetime.now(), stack_trace="native details",
                         source_file=source, diagnostic_owner="shader", diagnostic_key="one"))
    Debug.set_diagnostic("script.compile", "two", "bad syntax", source_file=source)
    Debug.set_diagnostic("shader", "one", "new interface error", source_file=source)
    assert console.error_count == 3
    assert len(console.get_entries()) == 3
    assert Debug.resolve_diagnostic("shader", "one")
    assert not Debug.resolve_diagnostic("shader", "one")
    assert console.error_count == 2
    assert console.resolve_source_diagnostics(source, "script.compile") == 1
    assert [entry.message for entry in console.get_entries()] == ["user history"]


def test_active_diagnostics_survive_history_retention_and_can_be_resolved_before_attach(console):
    console._max_entries = 2
    Debug.set_diagnostic("asset", "one", "bad material", source_file="One.mat")
    for index in range(5):
        Debug.log(f"history {index}")
    assert console.error_count == 1
    assert Debug.resolve_diagnostic("asset", "one")
    assert all(not entry.diagnostic_owner for entry in console._pending_native_entries)


def test_callback_recovery_is_instance_and_method_specific(console):
    class Context:
        pass
    first, second = Context(), Context()
    Debug.report_script_error(_caught(), "Probe.update", first)
    Debug.report_script_error(_caught(), "Probe.fixed_update", first)
    Debug.report_script_error(_caught(), "Probe.update", second)
    assert console.error_count == 3
    assert Debug.resolve_script_error("Probe.update", first)
    assert console.error_count == 2
    Debug.report_script_error(_caught("new error"), "Probe.update", first)
    latest = console.get_entries()[-1]
    assert latest.occurrences == 1 and "new error" in latest.stack_trace


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
