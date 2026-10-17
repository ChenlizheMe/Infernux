"""Native log stream, owned diagnostics, and Python Debug bridge."""

import sys
from pathlib import Path

import pytest

from infernux.debug import Debug, DebugConsole, LogEntry, LogType
from infernux.lib import ConsolePanel, LogLevel, inflog_internal


@pytest.fixture
def native_console():
    console = DebugConsole()
    panel = ConsolePanel()
    console.set_native_console(panel)
    yield panel
    console.set_native_console(None)
    console.clear()
    DebugConsole._instance = None


def test_log_stream_counts_filters_and_clear():
    panel = ConsolePanel()
    assert (panel.get_info_count(), panel.get_warning_count(), panel.get_error_count()) == (0, 0, 0)
    assert panel.show_info and panel.show_warnings and panel.show_errors
    assert not panel.collapse and panel.clear_on_play and not panel.error_pause
    panel.show_info = False
    assert not panel.show_info
    panel.log_from_python(LogLevel.Info, "hello")
    panel.log_from_python(LogLevel.Warn, "careful")
    panel.log_from_python(LogLevel.Error, "failure")
    assert (panel.get_info_count(), panel.get_warning_count(), panel.get_error_count()) == (1, 1, 1)
    panel.clear()
    assert (panel.get_info_count(), panel.get_warning_count(), panel.get_error_count()) == (0, 0, 0)


def test_diagnostic_removal_preserves_other_sources_owners_and_user_history(tmp_path):
    panel = ConsolePanel()
    source, other = str(tmp_path / "pipeline.py"), str(tmp_path / "other.py")
    panel.log_from_python(LogLevel.Error, "old error", source_file=source)
    panel.log_from_python(LogLevel.Warn, "old warning", source_file=source)
    panel.log_from_python(LogLevel.Error, "other error", source_file=other)
    assert panel.remove_entries_from_source(source) == 2
    assert panel.get_error_count() == 1 and panel.get_warning_count() == 0
    assert [entry["message"] for entry in panel._get_visible_log_snapshot(10)] == ["other error"]

    panel.log_from_python(LogLevel.Error, "[RenderStack:1] rejected", source_file=source)
    panel.log_from_python(LogLevel.Error, "[RenderStack:2] rejected", source_file=source)
    panel.log_from_python(LogLevel.Info, "user message", source_file=source)
    entries = panel._get_visible_log_snapshot(10)
    owned = next(entry for entry in entries if entry["message"] == "[RenderStack:1] rejected")
    panel.set_selection_snapshot(owned["uid"])
    panel.log_from_python(LogLevel.Error, "[RenderStack:1] pending", source_file=source)
    assert panel.remove_entries_from_source(source, "[RenderStack:1] ") == 2
    assert panel.get_error_count() == 2 and panel.get_info_count() == 1
    assert not panel.has_selected_entry()
    assert [entry["message"] for entry in panel._get_visible_log_snapshot(10)] == [
        "other error", "[RenderStack:2] rejected", "user message",
    ]


def test_large_pending_log_counts_are_incremental():
    panel = ConsolePanel()
    for index in range(20000):
        panel.log_from_python(LogLevel.Info, f"entry {index}")
    panel.log_from_python(LogLevel.Warn, "warning")
    panel.log_from_python(LogLevel.Error, "error")
    assert (panel.get_info_count(), panel.get_warning_count(), panel.get_error_count()) == (20000, 1, 1)


def test_debug_bridge_preserves_levels_source_location_and_internal_visibility(native_console):
    source_line = sys._getframe().f_lineno + 1
    Debug.log("native source location")
    Debug.log_warning("bridge warning")
    Debug.log_error("bridge error")
    try:
        raise ValueError("test exception")
    except ValueError as error:
        Debug.log_exception(error)
    Debug.log_internal("hidden Python internal")
    inflog_internal("hidden native internal")
    assert (
        native_console.get_info_count(), native_console.get_warning_count(),
        native_console.get_error_count(),
    ) == (1, 1, 2)
    entry = next(entry for entry in native_console._get_visible_log_snapshot(10)
                 if entry["message"] == "native source location")
    assert Path(entry["source_file"]).resolve() == Path(__file__).resolve()
    assert entry["source_line"] == source_line


def test_disconnected_debug_bridge_does_not_write_to_old_panel():
    console = DebugConsole()
    panel = ConsolePanel()
    try:
        console.set_native_console(panel)
        Debug.log("before disconnect")
        console.set_native_console(None)
        Debug.log("after disconnect")
        assert panel.get_info_count() == 1
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = None


def test_component_log_selects_owning_object_and_retains_diagnostic(scene, native_console):
    from infernux.engine._bootstrap_panels import BootstrapPanelsMixin
    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import EditorInteractionCore, SelectionDomain, SelectionService, SelectionTarget

    class ConsoleOwner(BootstrapPanelsMixin, BootstrapSelectionMixin):
        pass

    previous_selection = SelectionService._instance
    core = EditorInteractionCore()
    owner = object.__new__(ConsoleOwner)
    owner.interaction_core = core
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    core.panels.register_selection_authority("console", (SelectionDomain.DIAGNOSTIC_ENTRY,))
    core.navigation.register(SelectionDomain.SCENE_OBJECT, lambda _target, _request: True)
    core.selection.add_listener(lambda _change: native_console.set_selection_snapshot(0))
    game_object = scene.create_game_object("Logged object")
    component = game_object.add_component("BoxCollider")
    try:
        Debug.log_error("component failure", component)
        entry = native_console._get_visible_log_snapshot(1)[0]
        uid = entry["uid"]
        object_id = game_object.id
        assert native_console.get_entry_context_object_id(uid) == object_id
        owner._on_console_selection_changed(native_console, uid, False)
        assert core.selection.snapshot.primary == SelectionTarget.scene_object(object_id)
        assert core.selection.snapshot.owner_id == "hierarchy"
        assert native_console._selected_uid == uid

        Debug.log("object observation", game_object)
        object_uid = native_console._get_visible_log_snapshot(1)[0]["uid"]
        assert native_console.get_entry_context_object_id(object_uid) == object_id
        scene.destroy_game_object(game_object)
        scene.process_pending_destroys()
        assert native_console.get_entry_context_object_id(uid) == 0
        Debug.log("retired component observation", component)
        Debug.log("retired object observation", game_object)
        assert all(native_console.get_entry_context_object_id(entry["uid"]) == 0
                   for entry in native_console._get_visible_log_snapshot(2))
        owner._on_console_selection_changed(native_console, uid, False)
        assert core.selection.snapshot.domain is SelectionDomain.DIAGNOSTIC_ENTRY
    finally:
        core.shutdown()
        SelectionService._instance = previous_selection


def test_log_display_is_relative_and_trace_remains_structured(tmp_path, capsys):
    from datetime import datetime

    previous = DebugConsole._instance
    console = DebugConsole()
    panel = ConsolePanel(str(tmp_path))
    console.set_native_console(panel)
    source = str(tmp_path / "Assets" / "Scripts" / "Player.py")
    try:
        console.log(LogEntry("ValueError: failed", LogType.EXCEPTION, datetime.now(),
                             stack_trace="Traceback original diagnostic", source_file=source, source_line=35))
        expected = "[Error] (Assets/Scripts/Player.py, line 35) ValueError: failed"
        assert capsys.readouterr().err.strip() == expected
        assert panel._get_status_snapshot()[0] == expected
        snapshot = panel._get_visible_log_snapshot(1)[0]
        assert snapshot["stack_trace"] == "Traceback original diagnostic"
        assert snapshot["source_file"] == source
        assert snapshot["message"] == "ValueError: failed"
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = previous
