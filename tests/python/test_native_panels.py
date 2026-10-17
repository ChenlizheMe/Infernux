"""Native panel lifecycle and callbacks actually dispatched by native code."""

import pytest

from infernux.lib import (
    ConsolePanel,
    EditorPanel,
    HierarchyPanel,
    InspectorPanel,
    LogLevel,
    MenuBarPanel,
    ProjectPanel,
    ToolbarPanel,
)


def test_native_panels_share_window_lifecycle_and_command_bridge():
    for panel, window_id in (
        (ConsolePanel(), "console"),
        (ToolbarPanel(), "toolbar"),
        (HierarchyPanel(), "hierarchy"),
        (InspectorPanel(), "inspector"),
        (ProjectPanel(), "project"),
    ):
        assert isinstance(panel, EditorPanel)
        assert panel.get_window_id() == window_id
        assert panel.is_open()
        panel.set_open(False)
        assert not panel.is_open()
        panel.set_open(True)
        assert panel.is_open()
        calls = []
        panel.execute_command = lambda command, source, argument: calls.append(
            (command, source, argument)
        ) or True
        panel.can_execute_command = lambda command, argument: bool(command and not argument)
        assert panel.execute_command("test.command", "api", "")
        assert panel.can_execute_command("test.command", "")
        assert not panel.can_execute_command("", "")
        assert calls == [("test.command", "api", "")]


def test_console_selection_intent_and_silent_projection_remain_distinct():
    panel = ConsolePanel()
    selected, focused = [], []
    panel.on_selection_changed = lambda uid, history: selected.append((uid, history))
    panel.on_request_focus = lambda: focused.append(True)
    panel.log_from_python(LogLevel.Info, "first")
    panel.log_from_python(LogLevel.Warn, "second")
    panel.log_from_python(LogLevel.Error, "latest")
    message, level, info, warning, error, uid = panel._get_status_snapshot()
    assert (message, level, info, warning, error) == ("[Error] latest", "error", 1, 1, 1)
    assert uid > 0

    panel.select_entry(uid)
    assert selected == [(uid, True)]
    assert focused == [True]
    assert panel._selected_uid == 0
    panel.set_selection_snapshot(uid)
    assert panel._selected_uid == uid
    assert panel._selected_uids == [uid]
    assert panel.has_selected_entry()
    panel.set_selection_snapshot(0)
    assert panel._selected_uid == 0
    assert selected == [(uid, True)]
    panel.select_latest_entry()
    assert focused == [True, True]


def test_console_clear_resets_counts_and_selection_without_reusing_ids():
    panel = ConsolePanel()
    paused, selected = [], []
    panel.error_pause = True
    panel.on_error_pause = lambda: paused.append(True)
    panel.on_selection_changed = lambda uid, history: selected.append((uid, history))
    panel.log_from_python(LogLevel.Error, "failure")
    uid = panel._get_status_snapshot()[-1]
    assert paused == [True]
    panel.set_selection_snapshot(uid)
    revision = panel._revision
    panel.clear()
    assert panel._revision > revision
    assert panel._get_status_snapshot() == ("", "info", 0, 0, 0, 0)
    assert panel._selected_uids == [] and not panel.has_selected_entry()
    assert selected == [(0, False)]
    panel.log_from_python(LogLevel.Info, "after clear")
    assert panel._get_status_snapshot()[-1] > uid

    assert panel.has_view_option("collapse")
    panel.set_view_option("collapse", True)
    panel.set_search_query("validation")
    assert panel.get_view_option("collapse")
    assert panel.get_search_query() == "validation"
    panel.set_detail_height(123.0)
    assert panel.get_detail_height() == pytest.approx(123.0)
    panel.set_detail_height(1.0)
    assert panel.get_detail_height() == pytest.approx(40.0)


def test_toolbar_camera_update_is_published_to_engine():
    panel = ToolbarPanel()
    defaults = panel.get_camera_settings()
    assert (defaults["orthographic"], defaults["fov"], defaults["move_speed"]) == (
        False, 60.0, 5.0,
    )
    applied = {}
    panel.apply_camera_to_engine = applied.update
    settings = {
        "orthographic": True, "fov": 75.0, "orthographic_size": 12.0,
        "rotation_speed": 0.1, "pan_speed": 2.0, "zoom_speed": 1.5,
        "move_speed": 10.0, "move_speed_boost": 5.0,
    }
    panel.set_camera_settings(settings)
    assert panel.get_camera_settings() == pytest.approx(settings)
    assert applied == pytest.approx(settings)


def test_menu_resolves_declared_translation_and_plugin_literal():
    panel = MenuBarPanel()
    declared = {"menu.extensions": "扩展"}
    panel.translate = lambda key: declared.get(key, key)
    panel.has_translation = lambda key: key in declared
    assert panel.resolve_menu_label("menu.extensions", "Extensions") == "扩展"
    assert panel.resolve_menu_label("menu.extensions_example_plugin", "Example Plugin") == "Example Plugin"


def test_hierarchy_native_selection_routes_to_authoritative_selection():
    panel = HierarchyPanel()
    selected = set()
    panel.select_id = selected.add
    panel.get_primary = lambda: max(selected) if selected else 0
    panel.selection_count = lambda: len(selected)
    panel.get_selected_ids = lambda: list(selected)
    panel.is_selection_empty = lambda: not selected
    assert panel.is_selection_empty()
    panel.set_selected_object_by_id(42)
    assert selected == {42}
    assert panel.get_primary() == 42 and panel.selection_count() == 1
