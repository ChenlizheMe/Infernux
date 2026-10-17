from __future__ import annotations

from types import SimpleNamespace


def _bind_toolbar_scene_view(core) -> None:
    from infernux.engine.interaction import PanelInteractionDescriptor
    from infernux.engine.ui.core_panel_interactions import toolbar_panel_interaction

    core.panels.register_type("toolbar", toolbar_panel_interaction())
    core.panels.register_type("scene_view", PanelInteractionDescriptor())
    core.panels.bind_view("scene_view", "scene_view", object())


def test_view_state_command_is_undoable_without_dirtying_documents():
    from infernux.engine.interaction import ViewCommandService
    from infernux.engine.undo import UndoManager

    previous_manager = UndoManager._instance
    previous_service = ViewCommandService._instance
    manager = UndoManager()
    service = ViewCommandService()
    state = {"tool": 0}
    try:
        assert service.set_value(
            0,
            2,
            lambda value: state.__setitem__("tool", value),
            description="Select Rotate Tool",
        )
        assert state["tool"] == 2
        entry = manager.action_journal.applied_entries()[0]
        assert entry.action.marks_dirty is False
        manager.undo()
        assert state["tool"] == 0
        manager.redo()
        assert state["tool"] == 2
    finally:
        service.shutdown()
        ViewCommandService._instance = previous_service
        UndoManager._instance = previous_manager


def test_view_state_still_applies_when_history_is_temporarily_disabled():
    from infernux.engine.interaction import ViewCommandService
    from infernux.engine.undo import UndoManager

    previous_manager = UndoManager._instance
    previous_service = ViewCommandService._instance
    manager = UndoManager()
    manager.enabled = False
    service = ViewCommandService()
    state = {"path": "Assets"}
    try:
        assert service.set_value(
            "Assets",
            "Assets/Nested",
            lambda value: state.__setitem__("path", value),
            description="Navigate Project",
        )
        assert state["path"] == "Assets/Nested"
        assert manager.action_journal.entries == ()
    finally:
        service.shutdown()
        ViewCommandService._instance = previous_service
        UndoManager._instance = previous_manager


def test_toolbar_camera_drag_commits_one_non_dirty_view_command():
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager

    previous_manager = UndoManager._instance
    previous_core = EditorInteractionCore._instance
    manager = UndoManager()
    core = EditorInteractionCore()
    camera = SimpleNamespace(
        orthographic=False,
        fov=60.0,
        orthographic_size=5.0,
        rotation_speed=0.15,
        pan_speed=1.0,
        zoom_speed=1.0,
        move_speed=5.0,
        move_speed_boost=3.0,
    )
    native = SimpleNamespace(
        is_show_grid=lambda: True,
        set_show_grid=lambda _value: None,
    )
    engine = SimpleNamespace(
        _play_mode_manager=None,
        editor_camera=camera,
        get_native_engine=lambda: native,
    )
    toolbar = SimpleNamespace(
        get_camera_settings=lambda: {},
    )
    bootstrap = object.__new__(EditorBootstrap)
    bootstrap.interaction_core = core
    try:
        _bind_toolbar_scene_view(core)
        core.focus.activate_panel(
            "particle_graph_editor",
            view_id="particle_graph_editor",
            record_history=False,
        )
        bootstrap._wire_toolbar_callbacks_on(toolbar, engine)
        initial = toolbar.sync_camera_from_engine()
        changed = dict(initial, move_speed=12.0)

        toolbar.begin_camera_edit("toolbar.move_speed", initial)
        toolbar.apply_camera_to_engine(changed)
        toolbar.end_camera_edit("toolbar.move_speed", changed)

        assert camera.move_speed == 12.0
        entries = manager.action_journal.applied_entries()
        assert len(entries) == 1
        assert entries[0].action.description == "Change Scene Camera Move Speed"
        assert entries[0].action.marks_dirty is False
        assert entries[0].before_context.focus.active_view_id == "scene_view"
        assert entries[0].after_context.focus.active_view_id == "scene_view"
        assert core.focus.snapshot.active_view_id == "particle_graph_editor"

        manager.undo()
        assert camera.move_speed == 5.0
        manager.redo()
        assert camera.move_speed == 12.0
    finally:
        core.shutdown()
        EditorInteractionCore._instance = previous_core
        UndoManager._instance = previous_manager


def test_shared_node_graph_center_view_is_undoable_without_document_dirty():
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui.node_graph_editor_panel import NodeGraphEditorPanel
    from infernux.engine.undo import UndoManager

    previous_manager = UndoManager._instance
    previous_core = EditorInteractionCore._instance
    manager = UndoManager()
    core = EditorInteractionCore()
    view = SimpleNamespace(
        pan_x=15.0,
        pan_y=-7.0,
        zoom=0.5,
        graph=SimpleNamespace(nodes=[object()]),
    )

    def center_on_nodes():
        view.pan_x = 120.0
        view.pan_y = 80.0
        view.zoom = 1.0

    view.center_on_nodes = center_on_nodes
    panel = NodeGraphEditorPanel.__new__(NodeGraphEditorPanel)
    panel._view = view
    try:
        assert panel.command_graph_center_view()
        assert (view.pan_x, view.pan_y, view.zoom) == (120.0, 80.0, 1.0)
        entry = manager.action_journal.applied_entries()[0]
        assert entry.action.description == "Center Node Graph View"
        assert entry.action.marks_dirty is False

        manager.undo()
        assert (view.pan_x, view.pan_y, view.zoom) == (15.0, -7.0, 0.5)
        manager.redo()
        assert (view.pan_x, view.pan_y, view.zoom) == (120.0, 80.0, 1.0)
    finally:
        core.shutdown()
        EditorInteractionCore._instance = previous_core
        UndoManager._instance = previous_manager


def test_timeline_scrub_and_preview_camera_use_non_dirty_view_history():
    from infernux.core.animation_timeline import AnimationTimeline
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui.animtimeline_editor_panel import AnimTimelineEditorPanel
    from infernux.engine.undo import UndoManager

    previous_manager = UndoManager._instance
    previous_core = EditorInteractionCore._instance
    manager = UndoManager()
    core = EditorInteractionCore()
    panel = AnimTimelineEditorPanel.__new__(AnimTimelineEditorPanel)
    panel._timeline = AnimationTimeline(name="View Test", duration=2.0)
    panel._playhead = 1.25
    panel._playhead_scrub_before = 0.25
    panel._cam_yaw = 0.8
    panel._cam_pitch = -0.2
    panel._cam_dist = 9.0
    panel._preview_orbit_before = (-0.6, 0.5, 6.0)
    panel._preview_zoom_before = None
    try:
        assert panel._commit_playhead_scrub()
        assert panel._commit_preview_view_gesture("orbit")
        entries = manager.action_journal.applied_entries()
        assert [entry.action.description for entry in entries] == [
            "Scrub Timeline Playhead",
            "Orbit Timeline Preview",
        ]
        assert all(entry.action.marks_dirty is False for entry in entries)

        manager.undo()
        assert (panel._cam_yaw, panel._cam_pitch, panel._cam_dist) == (
            -0.6,
            0.5,
            6.0,
        )
        manager.undo()
        assert panel._playhead == 0.25
        manager.redo()
        assert panel._playhead == 1.25
        manager.redo()
        assert (panel._cam_yaw, panel._cam_pitch, panel._cam_dist) == (
            0.8,
            -0.2,
            9.0,
        )
    finally:
        core.shutdown()
        EditorInteractionCore._instance = previous_core
        UndoManager._instance = previous_manager
