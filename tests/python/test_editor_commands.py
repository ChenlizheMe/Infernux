from __future__ import annotations

from types import SimpleNamespace


from infernux.engine._bootstrap_wiring import BootstrapWiringMixin
from infernux.engine.interaction import (
    ActionOrigin,
    CommandSource,
    CommandStatus,
    EditorCommand,
    EditorCommandRegistry,
    FocusService,
    SelectionService,
    SelectionDomain,
    SelectionTarget,
    EditorInteractionCore,
    KeyChord,
    ShortcutEvent,
    ShortcutRouteStatus,
)
from infernux.engine.ui.animtimeline_editor_panel import (
    _TIMELINE_PANEL_INTERACTION,
)
from infernux.engine.ui.core_panel_interactions import (
    console_panel_interaction,
    ui_editor_panel_interaction,
)


def test_fresh_editor_bootstrap_registers_builtin_menus_and_panel_commands(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    source = str(Path(__file__).resolve().parents[2] / "python")
    result = subprocess.run(
        [sys.executable, "-", source, str(tmp_path)],
        input="""
import sys
sys.path.insert(0, sys.argv[1])
from infernux.engine.bootstrap import EditorBootstrap
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.ui.window_manager import WindowManager

bootstrap = EditorBootstrap(sys.argv[2])
bootstrap.interaction_core = EditorInteractionCore()
bootstrap.window_manager = WindowManager(None, bootstrap.interaction_core.panels, lambda *_: None)
bootstrap._register_window_types()
bootstrap._register_core_editor_commands(bootstrap.window_manager, None)
expected = {
    "animclip2d_editor": "Animation", "animfsm_editor": "Animation",
    "animtimeline_editor": "Animation", "particle_graph_editor": "Rendering",
    "plugins": "Extensions", "build_settings": "", "environment_settings": "",
    "inxpackage_import": "", "preferences": "", "physics_settings": "",
    "toolbar": "", "console": "Window", "game_view": "Window",
    "hierarchy": "Window", "history": "Window", "inspector": "Window",
    "project": "Window", "scene_view": "Window", "tag_layer_settings": "Window",
    "ui_editor": "Window",
}
actual = {key: info.menu_path for key, info in bootstrap.window_manager.get_registered_types().items()}
assert actual == expected, actual
assert bootstrap.interaction_core.commands.get("animclip2d.fill_sequence") is not None
bootstrap.interaction_core.shutdown()
""",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _SceneCommandStub:
    def __init__(self, panel_getter):
        self._panel_getter = panel_getter

    @staticmethod
    def has_selection(context):
        return bool(context.selection.domain is SelectionDomain.SCENE_OBJECT and context.selection.targets)

    def can_copy(self, context):
        return self.has_selection(context) and self._panel_getter() is not None

    def copy(self, _context, *, cut):
        panel = self._panel_getter()
        return bool(panel and panel.copy_selected(cut))

    def can_paste(self, _context):
        panel = self._panel_getter()
        return bool(panel and panel.has_clipboard_data())

    def paste(self, _context):
        panel = self._panel_getter()
        return bool(panel and panel.paste_clipboard())

    def duplicate(self, _context):
        panel = self._panel_getter()
        return bool(panel and panel.duplicate_selected())

    def delete(self, _context):
        panel = self._panel_getter()
        if panel is None:
            return False
        panel.delete_selected_objects()
        return True

    def can_external_drop(self, reference, parent_id=0, is_guid=False):
        del is_guid
        return bool(reference and parent_id >= 0 and self._panel_getter() is not None)

    def instantiate_prefab(self, reference, parent_id=0, is_guid=False):
        panel = self._panel_getter()
        return bool(panel and panel.instantiate_prefab(reference, parent_id, is_guid))

    def create_model(self, reference, parent_id=0, is_guid=False):
        panel = self._panel_getter()
        return bool(panel and panel.create_model(reference, parent_id, is_guid))

    def rename(self, object_id, new_name):
        panel = self._panel_getter()
        return bool(panel and panel.rename_object(object_id, new_name))

    def move_hierarchy(self, object_ids, mode, target_id, after, destination_world_id=0):
        panel = self._panel_getter()
        return bool(panel and panel.move_hierarchy(object_ids, mode, target_id, after, destination_world_id))


class _PrefabCommandStub:
    def __init__(self, calls):
        self.calls = calls

    def can_execute(self, action, *, object_id=0, path=""):
        if action == "exit":
            return True
        return object_id == 42 and bool(action) or bool(path)

    def create_from_object(self, object_id, current_path="", *, origin=None):
        self.calls.append(("prefab", object_id, "save_as"))
        return "C:/Project/Assets/Test.prefab"

    def locate(self, *, object_id=0, path="", record_history=True):
        self.calls.append(("prefab", object_id, "select"))
        return True

    def open(self, *, object_id=0, path="", origin=None):
        self.calls.append(("prefab", object_id, "open"))
        return True

    def apply(self, object_id, *, origin=None):
        self.calls.append(("prefab", object_id, "apply"))
        return True

    def revert(self, object_id, *, origin=None):
        self.calls.append(("prefab", object_id, "revert"))
        return True

    def unpack(self, object_id, *, origin=None):
        self.calls.append(("prefab", object_id, "unpack"))
        return True

    def exit(self, *, origin=None):
        self.calls.append(("prefab", 0, "exit"))
        return True

    def shutdown(self):
        pass


def _registry():
    focus = FocusService()
    selection = SelectionService()
    return EditorCommandRegistry(focus=focus, selection=selection), focus, selection


def _bind_panel(core, type_id, descriptor, instance, *, view_id=None):
    core.panels.register_type(type_id, descriptor, replace=True)
    core.panels.bind_view(view_id or type_id, type_id, instance)


def test_shortcut_router_retains_the_latest_authoritative_route_result():
    core = EditorInteractionCore()
    core.focus.activate_panel("project", view_id="project", record_history=False)

    first = core.shortcuts.route(ShortcutEvent(KeyChord.parse("F12")))
    second = core.shortcuts.route(ShortcutEvent(KeyChord.parse("F12"), text_input_active=True))

    assert first.status is ShortcutRouteStatus.NO_MATCH
    assert second.status is ShortcutRouteStatus.NO_MATCH
    assert core.shortcuts.route_revision == 2
    assert core.shortcuts.last_event == ShortcutEvent(KeyChord.parse("F12"), text_input_active=True)
    assert core.shortcuts.last_result is second

    core.shutdown()


def test_command_disabled_noop_failure_and_missing_results_are_structured():
    registry, _focus, _selection = _registry()
    registry.register(EditorCommand("disabled", lambda _context: None, can_execute=lambda _context: False))
    registry.register(EditorCommand("noop", lambda _context: False))
    registry.register(EditorCommand("failure", lambda _context: 1 / 0))

    assert registry.execute("disabled").status is CommandStatus.DISABLED
    assert registry.execute("noop").status is CommandStatus.NO_OP
    assert registry.execute("failure").status is CommandStatus.FAILED
    assert registry.execute("missing").status is CommandStatus.NOT_FOUND


def test_command_registry_wraps_context_changes_in_one_user_action():
    from infernux.engine.interaction import EditorContextSnapshot
    from infernux.engine.undo import UndoManager

    registry, focus, selection = _registry()
    manager = UndoManager()
    manager.set_context_hooks(
        lambda: EditorContextSnapshot(focus.snapshot, selection.snapshot),
        lambda context, _phase: (
            focus.apply_snapshot(context.focus, record_history=False),
            selection.apply_snapshot(context.selection, record_history=False),
        ),
    )
    focus.activate_panel("scene_view", record_history=False)
    selection.select(
        SelectionTarget.scene_object(7),
        owner_id="scene_view",
        record_history=False,
    )

    def select_asset(_context):
        focus.activate_panel("project")
        selection.select(
            SelectionTarget.asset("Assets/Test.mat"),
            owner_id="project",
        )
        return True

    registry.register(EditorCommand("project.select", select_asset, "Select Asset"))
    assert registry.execute("project.select").accepted

    assert len(manager.action_journal.entries) == 1
    assert manager.undo_description == "Select Asset"
    manager.undo()
    assert focus.snapshot.active_panel_id == "scene_view"
    assert selection.snapshot.primary == SelectionTarget.scene_object(7)


def test_window_toggle_uses_the_user_navigation_path_for_opening():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    calls = []
    open_windows = set()

    def open_from_user(target_id, *, reason="window_open_command"):
        calls.append(("open_from_user", target_id, reason))
        open_windows.add(target_id)
        return object()

    def close_window(target_id):
        calls.append(("close", target_id))
        open_windows.discard(target_id)

    windows = SimpleNamespace(
        get_window_instance=lambda _panel_id: None,
        get_registered_types=lambda: {"preferences": object()},
        open_window_from_user=open_from_user,
        close_window=close_window,
        is_window_open=lambda target_id: target_id in open_windows,
        reset_layout=lambda: None,
    )
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    try:
        BootstrapWiringMixin._register_core_editor_commands(
            bootstrap,
            windows,
            SimpleNamespace(),
        )
        registry = bootstrap.interaction_core.commands
        context = registry.context(
            CommandSource.MENU,
            {"target_id": "preferences"},
        )

        assert registry.can_execute("window.toggle", context)
        assert not registry.is_checked("window.toggle", context)
        assert registry.execute(
            "window.toggle",
            source=CommandSource.MENU,
            payload={"target_id": "preferences"},
        ).accepted
        assert registry.is_checked("window.toggle", context)
        assert calls == [("open_from_user", "preferences", "window_toggle_command")]

        assert registry.execute(
            "window.toggle",
            source=CommandSource.MENU,
            payload={"target_id": "preferences"},
        ).accepted
        assert not registry.is_checked("window.toggle", context)
        assert calls[-1] == ("close", "preferences")
    finally:
        bootstrap.interaction_core.shutdown()


def test_failed_project_document_open_does_not_publish_window_navigation():
    from infernux.engine.interaction import (
        DocumentOpenResult,
        DocumentOpenStatus,
    )

    class BootstrapHarness(BootstrapWiringMixin):
        pass

    calls = []
    windows = SimpleNamespace(
        get_window_instance=lambda _panel_id: None,
        get_registered_types=lambda: {},
        open_window_from_user=lambda target_id, **_kwargs: calls.append(target_id),
        is_window_open=lambda _target_id: False,
        reset_layout=lambda: None,
    )
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(
        _play_mode_manager=None,
        get_asset_database=lambda: None,
    )
    bootstrap.interaction_core.document_open.open_resource = lambda *_args, **_kwargs: DocumentOpenResult(
        DocumentOpenStatus.FAILED,
        message="rejected",
    )

    try:
        BootstrapWiringMixin._register_core_editor_commands(
            bootstrap,
            windows,
            SimpleNamespace(),
        )
        result = bootstrap.interaction_core.commands.execute(
            "project.open_document",
            source=CommandSource.POINTER,
            payload={
                "path": "C:/Project/Assets/Broken.animfsm",
                "document_kind": "animation_fsm",
            },
        )

        assert result.status is CommandStatus.NO_OP
        assert result.value is False
        assert calls == []
    finally:
        bootstrap.interaction_core.shutdown()


def test_play_command_reveals_game_view_through_window_manager_only():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    class PlayModeStub:
        def __init__(self):
            self.is_playing = False
            self.is_edit_mode = True
            self.calls = []

        def enter_play_mode(self):
            self.calls.append("enter")
            self.is_playing = True
            self.is_edit_mode = False
            return True

        def exit_play_mode(self):
            self.calls.append("exit")
            self.is_playing = False
            self.is_edit_mode = True
            return True

    play_mode = PlayModeStub()
    window_calls = []
    windows = SimpleNamespace(
        get_window_instance=lambda _panel_id: None,
        get_registered_types=lambda: {"game_view": object()},
        open_window_from_user=lambda target_id, *, reason: window_calls.append((target_id, reason)) or object(),
        is_window_open=lambda _target_id: True,
        reset_layout=lambda: None,
    )
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(_play_mode_manager=play_mode)

    BootstrapWiringMixin._register_core_editor_commands(
        bootstrap,
        windows,
        SimpleNamespace(new_scene=lambda: None),
    )

    registry = bootstrap.interaction_core.commands
    assert registry.execute("play.toggle").accepted
    assert play_mode.calls == ["enter"]
    assert window_calls == [("game_view", "play_mode_enter")]

    assert registry.execute("play.toggle").accepted
    assert play_mode.calls == ["enter", "exit"]
    assert window_calls == [("game_view", "play_mode_enter")]

    # Failed restoration has stopped simulation but still owns a Play snapshot.
    play_mode.is_edit_mode = False
    assert registry.execute("play.toggle").accepted
    assert play_mode.calls == ["enter", "exit", "exit"]
    assert window_calls == [("game_view", "play_mode_enter")]


def test_ui_editor_uses_global_scene_commands_and_selection_clear():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    calls = []
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    bootstrap.hierarchy = SimpleNamespace(
        copy_selected=lambda cut: calls.append(("copy", cut)) or True,
        paste_clipboard=lambda: calls.append("paste") or True,
        has_clipboard_data=lambda: True,
        delete_selected_objects=lambda: calls.append("delete"),
        begin_rename_object=lambda _object_id: None,
    )
    scene_commands = _SceneCommandStub(lambda: bootstrap.hierarchy)
    creation_service = SimpleNamespace(
        can_create=lambda kind, parent_id=0: kind.startswith("ui.") and parent_id >= 0,
        create=lambda kind, **kwargs: calls.append(("create_ui", kind, kwargs)) or {"id": 99},
    )
    bootstrap.ui_editor = SimpleNamespace(
        can_nudge_selected=lambda: False,
        command_nudge_selected=lambda _dx, _dy: False,
    )
    _bind_panel(
        bootstrap.interaction_core,
        "ui_editor",
        ui_editor_panel_interaction(
            scene_commands,
            creation_service=creation_service,
        ),
        bootstrap.ui_editor,
    )
    windows = SimpleNamespace(
        get_window_instance=lambda _panel_id: None,
        get_registered_types=lambda: {},
        reset_layout=lambda: None,
    )
    BootstrapWiringMixin._register_core_editor_commands(
        bootstrap,
        windows,
        SimpleNamespace(),
    )
    core = bootstrap.interaction_core
    core.focus.activate_panel("ui_editor", view_id="ui_editor")
    core.selection.select(
        SelectionTarget.scene_object(42),
        owner_id="ui_editor",
        record_history=False,
    )

    copied = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Ctrl+C")))
    deleted = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Delete")))
    deselected = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Escape")))

    assert copied.status is ShortcutRouteStatus.EXECUTED
    assert deleted.status is ShortcutRouteStatus.EXECUTED
    assert deselected.status is ShortcutRouteStatus.EXECUTED
    assert calls == [("copy", False), "delete"]
    assert core.selection.snapshot.targets == ()

    assert core.commands.execute(
        "scene.create_object",
        source=CommandSource.POINTER,
        payload={"kind": "ui.text", "parent_id": 12},
    ).accepted
    assert calls[-1] == (
        "create_ui",
        "ui.text",
        {
            "parent_id": 12,
            "selection_owner_id": "ui_editor",
            "selection_reason": "ui_editor_create_text",
        },
    )

    core.shutdown()


def test_asset_rename_command_uses_project_asset_authority_and_command_origin():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    calls = []
    service = bootstrap.interaction_core.project_assets
    service.can_rename = lambda source, name: source == "C:/Project/Assets/Old.mat" and name == "New.mat"
    service.rename = lambda source, name, *, origin: calls.append((source, name, origin)) or "C:/Project/Assets/New.mat"
    windows = SimpleNamespace(get_registered_types=lambda: {}, reset_layout=lambda: None)
    BootstrapWiringMixin._register_core_editor_commands(
        bootstrap,
        windows,
        SimpleNamespace(),
    )

    result = bootstrap.interaction_core.commands.execute(
        "asset.rename",
        source=CommandSource.AUTOMATION,
        payload={
            "source_path": "C:/Project/Assets/Old.mat",
            "new_name": "New.mat",
        },
    )

    assert result.accepted
    assert calls == [
        (
            "C:/Project/Assets/Old.mat",
            "New.mat",
            ActionOrigin.AUTOMATION,
        )
    ]


def test_timeline_toolbar_and_shortcuts_share_command_handlers():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    calls = []
    panel = SimpleNamespace(
        command_new_timeline=lambda: calls.append("new_timeline") or True,
        command_toggle_playback=lambda: calls.append("play_pause") or True,
        command_stop_playback=lambda: calls.append("stop") or True,
        command_set_loop_preview=lambda value: calls.append(("loop_preview", value)) or True,
        command_add_keyframe=lambda: calls.append("add_keyframe") or True,
        command_delete_selected_keyframe=lambda: calls.append("delete_keyframe") or True,
        can_delete_selected_keyframe=lambda: True,
    )
    windows = SimpleNamespace(
        get_window_instance=lambda panel_id: panel if panel_id == "animtimeline_editor" else None,
        get_registered_types=lambda: {},
        reset_layout=lambda: None,
    )
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.interaction_core.panels.register_type(
        "animtimeline_editor",
        _TIMELINE_PANEL_INTERACTION,
    )
    bootstrap.interaction_core.panels.bind_view(
        "animtimeline_editor",
        "animtimeline_editor",
        panel,
    )
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    bootstrap.hierarchy = None
    bootstrap.project_panel = None
    BootstrapWiringMixin._register_core_editor_commands(
        bootstrap,
        windows,
        SimpleNamespace(),
    )

    core = bootstrap.interaction_core
    core.focus.activate_panel(
        "animtimeline_editor",
        view_id="animtimeline_editor",
        document_id="timeline:test",
    )
    core.selection.select(
        SelectionTarget.timeline_element(
            "timeline:test",
            "keyframe:test",
            sub_kind="keyframe",
        ),
        owner_id="animtimeline_editor",
    )

    assert core.commands.execute("timeline.new", source=CommandSource.TOOLBAR).accepted
    assert core.commands.execute("timeline.add_keyframe", source=CommandSource.TOOLBAR).accepted
    assert core.commands.execute("timeline.stop", source=CommandSource.TOOLBAR).accepted
    assert core.shortcuts.route(ShortcutEvent(KeyChord.parse("Space"))).status is ShortcutRouteStatus.EXECUTED
    assert core.shortcuts.route(ShortcutEvent(KeyChord.parse("Delete"))).status is ShortcutRouteStatus.EXECUTED

    assert calls == [
        "new_timeline",
        "add_keyframe",
        "stop",
        "play_pause",
        "delete_keyframe",
    ]

    blocked = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Space"), text_input_active=True))
    assert blocked.status is ShortcutRouteStatus.BLOCKED


def test_console_copy_shortcut_routes_through_the_focused_panel_adapter():
    class BootstrapHarness(BootstrapWiringMixin):
        pass

    calls = []
    counts = {"info": 1, "warning": 0, "error": 0}
    options = {
        "show_info": True,
        "show_warnings": True,
        "show_errors": True,
        "collapse": False,
        "clear_on_play": True,
        "error_pause": False,
        "follow": True,
    }
    view = {"search": "", "detail_height": 90.0}

    def _clear():
        calls.append("clear")
        counts.update(info=0, warning=0, error=0)

    panel = SimpleNamespace(
        has_selected_entry=lambda: True,
        copy_selected_entry=lambda: calls.append("copy_log") or True,
        clear=_clear,
        get_info_count=lambda: counts["info"],
        get_warning_count=lambda: counts["warning"],
        get_error_count=lambda: counts["error"],
        has_view_option=lambda option: option in options,
        get_view_option=lambda option: options[option],
        set_view_option=lambda option, enabled: options.__setitem__(option, bool(enabled)),
        get_search_query=lambda: view["search"],
        set_search_query=lambda value: view.__setitem__("search", str(value)),
        get_detail_height=lambda: view["detail_height"],
        set_detail_height=lambda value: view.__setitem__("detail_height", float(value)),
        request_search_focus=lambda: calls.append("focus_search"),
    )
    bootstrap = BootstrapHarness()
    bootstrap.interaction_core = EditorInteractionCore()
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    bootstrap.hierarchy = None
    bootstrap.project_panel = None
    _bind_panel(
        bootstrap.interaction_core,
        "console",
        console_panel_interaction(),
        panel,
    )
    windows = SimpleNamespace(
        get_window_instance=lambda _panel_id: None,
        get_registered_types=lambda: {},
        reset_layout=lambda: None,
    )
    BootstrapWiringMixin._register_core_editor_commands(
        bootstrap,
        windows,
        SimpleNamespace(),
    )
    core = bootstrap.interaction_core
    core.focus.activate_panel("console", view_id="console")
    core.selection.select(
        SelectionTarget.diagnostic_entry("console", "log:42"),
        owner_id="console",
    )

    result = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Ctrl+C")))

    assert result.status is ShortcutRouteStatus.EXECUTED
    assert calls == ["copy_log"]

    result = core.shortcuts.route(ShortcutEvent(KeyChord.parse("Ctrl+F"), game_view_captured=True))
    assert result.status is ShortcutRouteStatus.EXECUTED
    assert calls == ["copy_log", "focus_search"]

    result = core.commands.execute("console.clear", source=CommandSource.POINTER)

    assert result.accepted
    assert calls == ["copy_log", "focus_search", "clear"]
    assert not core.commands.can_execute("console.clear", core.commands.context(CommandSource.POINTER))


def test_native_shortcut_event_blocks_editor_commands_while_game_view_owns_input():
    from infernux.engine._bootstrap_wiring import _native_shortcut_event
    from infernux.input import Input

    previous = Input.is_game_focused()
    try:
        Input.set_game_focused(True)
        captured = _native_shortcut_event("Left", False, False)
        assert captured.game_view_captured is True

        Input.set_game_focused(False)
        editor = _native_shortcut_event("Left", False, False)
        assert editor.game_view_captured is False
    finally:
        Input.set_game_focused(previous)
