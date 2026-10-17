from infernux.engine.interaction import (
    AssetMutation,
    AssetMutationKind,
    FocusService,
    InputContext,
    SelectionDomain,
    SelectionService,
    SelectionSnapshot,
    SelectionTarget,
)
import pytest


def test_selection_snapshot_rejects_mixed_domains():
    with pytest.raises(ValueError, match="cannot mix"):
        SelectionSnapshot.create(
            (
                SelectionTarget.scene_object(7),
                SelectionTarget.asset("Assets/Test.mat"),
            ),
            owner_id="invalid",
        )

    with pytest.raises(ValueError, match="requires an owner"):
        SelectionSnapshot.create(
            (SelectionTarget.scene_object(7),),
            owner_id="",
        )


def test_selection_reconcile_preserves_valid_order_and_repairs_primary_anchor():
    service = SelectionService()
    first = SelectionTarget.asset_subresource("Assets/sheet.png", "1" * 32, sub_kind="sprite_frame")
    second = SelectionTarget.asset_subresource("Assets/sheet.png", "2" * 32, sub_kind="sprite_frame")
    third = SelectionTarget.asset_subresource("Assets/sheet.png", "3" * 32, sub_kind="sprite_frame")
    service.replace(
        (first, second, third),
        owner_id="inspector",
        primary=second,
        anchor=second,
        record_history=False,
    )

    assert service.reconcile(
        lambda target: target != second,
        reason="test_reconcile",
        record_history=False,
    )
    assert service.snapshot.targets == (first, third)
    assert service.snapshot.primary == third
    assert service.snapshot.anchor == first


def test_asset_refresh_reconciles_sprite_selection_without_inspector_visibility(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from infernux.engine import _bootstrap_selection as selection_bootstrap
    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.core import asset_types

    previous = SelectionService._instance
    service = SelectionService()
    asset_path = tmp_path / "sheet.png"
    asset_path.write_bytes(b"texture")
    asset_guid = "sheet-guid"
    monkeypatch.setattr(
        selection_bootstrap,
        "_selection_asset_database",
        lambda: SimpleNamespace(
            get_guid_from_path=lambda _path: asset_guid,
            get_path_from_guid=lambda _guid: str(asset_path),
        ),
    )
    stale = SelectionTarget.asset_subresource(
        asset_guid,
        "1" * 32,
        sub_kind="sprite_frame",
    )
    service.select(stale, owner_id="inspector", record_history=False)
    monkeypatch.setattr(
        asset_types,
        "read_texture_import_settings",
        lambda _path: SimpleNamespace(
            texture_type=asset_types.TextureType.SPRITE,
            sprite_frames=(SimpleNamespace(stable_id="2" * 32),),
        ),
    )
    try:
        BootstrapSelectionMixin()._on_asset_selection_source_changed(
            AssetMutation(
                AssetMutationKind.MODIFIED,
                str(asset_path),
                guid=asset_guid,
            )
        )
        assert service.snapshot.primary == SelectionTarget.asset(asset_guid)
    finally:
        SelectionService._instance = previous


def test_timeline_editor_projects_stable_keyframe_selection():
    from infernux.core.animation_timeline import TimelineKeyframe
    from infernux.engine.ui.animtimeline_editor_panel import AnimTimelineEditorPanel

    previous = SelectionService._instance
    service = SelectionService()
    panel = AnimTimelineEditorPanel()
    key = TimelineKeyframe(time=0.5)
    panel._timeline.keyframes.append(key)
    try:
        panel.on_enable()
        panel._select_key(key)

        target = service.snapshot.primary
        assert target == SelectionTarget.timeline_element(
            panel.document_id,
            key.stable_id,
            sub_kind="keyframe",
        )
        assert panel._current_sel_key() is key

        service.select(SelectionTarget.asset("Assets/Test.mat"), owner_id="project")
        assert panel._current_sel_key() is None

        service.select(target, owner_id=panel.window_id, record_history=False)
        assert panel._current_sel_key() is key
        panel._timeline.keyframes.remove(key)
        assert panel._current_sel_key() is None
        assert service.snapshot.is_empty
    finally:
        panel.on_disable()
        SelectionService._instance = previous


def test_focus_history_keeps_keyboard_focus_and_replaced_dock_tab_separate():
    from types import SimpleNamespace

    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import (
        EditorActionJournal,
        EditorContextSnapshot,
        WindowLocator,
    )
    from infernux.engine.undo import UndoManager

    focus = FocusService()
    focus.activate_panel("console", view_id="console", record_history=False)
    selection = SelectionService()
    SelectionService.install(selection)
    journal = EditorActionJournal()
    manager = UndoManager(journal)

    def capture_context(**overrides):
        snapshot = overrides.get("focus", focus.snapshot)
        view_id = snapshot.active_view_id or snapshot.active_panel_id
        return EditorContextSnapshot(
            snapshot,
            overrides.get("selection", selection.snapshot),
            window=WindowLocator(view_id, view_id) if view_id else None,
        )

    class _WindowManager:
        @staticmethod
        def locate_window(window_id):
            return WindowLocator(window_id, window_id)

    bootstrap = BootstrapSelectionMixin()
    bootstrap.interaction_core = SimpleNamespace(
        focus=focus,
        capture_context=capture_context,
    )
    bootstrap.undo_manager = manager
    bootstrap.window_manager = _WindowManager()
    focus.add_change_listener(bootstrap._on_global_focus_changed)

    assert focus.activate_panel(
        "particle_graph_editor",
        view_id="particle_graph_editor",
        record_history=True,
        presentation_before_view_id="scene_view",
    )

    entry = journal.peek_undo()
    assert entry is not None
    assert entry.before_context.focus.active_view_id == "console"
    assert entry.before_context.window.window_id == "scene_view"
    assert entry.after_context.focus.active_view_id == "particle_graph_editor"
    assert entry.after_context.window.window_id == "particle_graph_editor"


def test_context_restore_publishes_focus_intent_before_native_window_poll():
    from types import SimpleNamespace

    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import (
        ContextRestoreStatus,
        EditorContextSnapshot,
        WindowLocator,
    )

    focus = FocusService()
    focus.activate_panel("project", record_history=False)
    target_focus = type(focus.snapshot)(
        active_panel_id="particle_graph_editor",
        active_view_id="particle_graph_editor",
    )
    observed_focus = []

    class _WindowManager:
        @staticmethod
        def is_window_content_visible(_window_id):
            return False

        @staticmethod
        def restore_window(_locator):
            observed_focus.append(focus.snapshot)
            return ContextRestoreStatus.PENDING

    bootstrap = BootstrapSelectionMixin()
    bootstrap.interaction_core = SimpleNamespace(
        focus=focus,
        document_open=SimpleNamespace(),
    )
    bootstrap.window_manager = _WindowManager()

    status = bootstrap._restore_editor_context(
        EditorContextSnapshot(
            focus=target_focus,
            selection=SelectionService.instance().snapshot,
            window=WindowLocator("particle_graph_editor", "particle_graph_editor"),
        ),
        "undo_complete",
    )

    assert status is ContextRestoreStatus.PENDING
    assert observed_focus == [target_focus]
    assert focus.snapshot == target_focus


def test_project_selection_intent_does_not_forge_panel_focus(monkeypatch):
    from types import SimpleNamespace

    from infernux.engine import _bootstrap_selection as selection_bootstrap
    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin

    monkeypatch.setattr(
        selection_bootstrap,
        "_selection_asset_database",
        lambda: SimpleNamespace(
            get_guid_from_path=lambda _path: "smoke-guid",
            get_path_from_guid=lambda _guid: "Assets/Smoke.particlegraph",
        ),
    )

    focus = FocusService()
    selection = SelectionService()
    SelectionService.install(selection)
    bootstrap = BootstrapSelectionMixin()
    bootstrap.interaction_core = SimpleNamespace(focus=focus)

    bootstrap._on_project_selection_changed(
        ("Assets/Smoke.particlegraph",),
        "Assets/Smoke.particlegraph",
    )

    assert focus.snapshot.active_panel_id == ""
    assert focus.snapshot.active_view_id == ""
    assert selection.snapshot.owner_id == "project"
    assert selection.snapshot.primary == SelectionTarget.asset("smoke-guid")


def test_native_panel_focus_loss_keeps_last_editor_command_context():
    focus = FocusService()
    changes = []
    focus.add_change_listener(changes.append)

    assert focus.observe_panel_focus("project", True, view_id="project")
    assert focus.snapshot.active_panel_id == "project"
    assert not focus.observe_panel_focus("project", False, view_id="project")
    assert focus.snapshot.active_panel_id == "project"

    assert focus.observe_panel_focus("hierarchy", True, view_id="hierarchy")
    assert focus.snapshot.active_panel_id == "hierarchy"
    assert [change.record_history for change in changes] == [False, False]


def test_project_click_records_focus_then_selection_as_distinct_user_actions():
    from types import SimpleNamespace

    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import EditorContextSnapshot
    from infernux.engine.undo import UndoManager

    focus = FocusService()
    selection = SelectionService()
    SelectionService.install(selection)
    focus.activate_panel("scene_view", record_history=False)
    selection.select(
        SelectionTarget.scene_object(42),
        owner_id="scene_view",
        record_history=False,
    )
    manager = UndoManager()
    manager.set_context_hooks(
        lambda: EditorContextSnapshot(focus.snapshot, selection.snapshot),
        lambda context, _phase: (
            focus.apply_snapshot(context.focus, record_history=False),
            selection.apply_snapshot(context.selection, record_history=False),
        ),
    )
    bootstrap = BootstrapSelectionMixin()
    bootstrap.interaction_core = SimpleNamespace(
        focus=focus,
        capture_context=lambda **overrides: EditorContextSnapshot(
            overrides.get("focus", focus.snapshot),
            overrides.get("selection", selection.snapshot),
        ),
    )
    bootstrap._present_selection_snapshot = lambda _snapshot, *, preserve_console_selection=False: None
    focus.add_change_listener(bootstrap._on_global_focus_changed)
    selection.add_listener(bootstrap._on_global_selection_changed)

    # Native EditorPanelFocusPublisher owns the first operation. The Project
    # selection adapter publishes only the second operation.
    focus.activate_panel(
        "project",
        view_id="project",
        reason="pointer_panel_activation",
        record_history=True,
    )
    bootstrap._on_project_selection_changed(
        ("Assets/Smoke.particlegraph",),
        "Assets/Smoke.particlegraph",
    )

    assert len(manager.action_journal.entries) == 2
    assert manager.undo_description == "Change Selection"
    manager.undo()
    assert focus.snapshot.active_panel_id == "project"
    assert selection.snapshot.primary == SelectionTarget.scene_object(42)
    manager.undo()
    assert focus.snapshot.active_panel_id == "scene_view"
    assert selection.snapshot.primary == SelectionTarget.scene_object(42)


def test_input_context_stack_honors_priority_and_modal_barrier():
    focus = FocusService()
    stack = focus.input_contexts
    stack.push(InputContext("global", "editor", priority=0))
    stack.push(InputContext("hierarchy", "hierarchy", priority=10))
    stack.push(InputContext("rename", "hierarchy", priority=20, blocks_lower=True))

    assert [context.context_id for context in stack.ordered()] == ["rename"]

    stack.remove("rename")
    assert [context.context_id for context in stack.ordered()] == [
        "hierarchy",
        "global",
    ]


@pytest.mark.parametrize(
    "sub_kind,token", [("submesh", "::submesh:"), ("submaterial", "::submat:"), ("subanimation", "::subanim:")]
)
def test_bootstrap_projects_subresources_and_all_component_owners(monkeypatch, sub_kind, token):
    from types import SimpleNamespace

    import infernux.lib as native
    from infernux.engine import _bootstrap_selection as selection_bootstrap
    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin

    objects = {value: SimpleNamespace(id=value) for value in (41, 42)}

    class _Scene:
        world_id = 73
        structure_version = 1
        temporal_discontinuity_revision = 0

        @staticmethod
        def find_by_id(object_id):
            return objects.get(object_id)

    class _SceneManager:
        @staticmethod
        def instance():
            return _SceneManager()

        @staticmethod
        def get_active_scene():
            return _Scene()

    monkeypatch.setattr(native, "SceneManager", _SceneManager)

    project_calls = []
    monkeypatch.setattr(
        selection_bootstrap,
        "_selection_asset_database",
        lambda: SimpleNamespace(
            get_guid_from_path=lambda _path: "robot-guid",
            get_path_from_guid=lambda _guid: "Assets/Robot.fbx",
        ),
    )
    inspector_calls = []
    component_projection = []
    outlines = []
    bootstrap = BootstrapSelectionMixin()
    bootstrap.project_panel = SimpleNamespace(
        set_selected_files=lambda paths, primary, notify: project_calls.append((list(paths), primary, notify)),
        clear_selection=lambda notify: project_calls.append(("clear", notify)),
    )
    bootstrap.inspector_panel = SimpleNamespace(
        set_selected_object_id=lambda object_id: inspector_calls.append(object_id),
        clear_selected_object=lambda: None,
        set_selected_component_ids=lambda ids: component_projection.append(list(ids)),
        clear_selected_components=lambda: component_projection.append([]),
    )
    bootstrap._inspector_set_selected_file = inspector_calls.append
    bootstrap._set_outline = lambda primary, selected: outlines.append((primary, list(selected)))
    bootstrap.event_bus = SimpleNamespace(emit=lambda *_args: None)

    subresource = SelectionSnapshot.create(
        (SelectionTarget.asset_subresource("robot-guid", "body", sub_kind=sub_kind),),
        owner_id="project",
    )
    bootstrap._present_selection_snapshot(subresource)
    asset_path = "Assets/Robot.fbx" + token + "body"
    assert project_calls == [([asset_path], asset_path, False)]
    assert inspector_calls == [asset_path]

    project_calls.clear()
    inspector_calls.clear()
    components = SelectionSnapshot.create(
        (
            SelectionTarget.component(41, 1),
            SelectionTarget.component(42, 2),
        ),
        owner_id="inspector",
        primary=SelectionTarget.component(42, 2),
    )
    bootstrap._present_selection_snapshot(components)
    assert project_calls == [("clear", False)]
    assert inspector_calls == [42]
    assert outlines[-1] == (42, [41, 42])
    assert component_projection[-1] == [1, 2]


def test_ui_editor_projects_directly_from_typed_selection(monkeypatch):
    from types import SimpleNamespace

    import infernux.lib as native
    from infernux.engine._bootstrap_wiring import BootstrapWiringMixin

    selected_object = SimpleNamespace(id=42)

    class _Scene:
        @staticmethod
        def find_by_id(object_id):
            return selected_object if object_id == 42 else None

    class _SceneManager:
        @staticmethod
        def instance():
            return _SceneManager()

        @staticmethod
        def get_active_scene():
            return _Scene()

    monkeypatch.setattr(native, "SceneManager", _SceneManager)

    projected = []
    ui_editor = SimpleNamespace(
        project_global_selection=projected.append,
    )
    bootstrap = BootstrapWiringMixin()
    bootstrap.ui_editor = ui_editor

    service = SelectionService()
    bootstrap._wire_ui_editor()
    assert projected == [None]

    service.select(
        SelectionTarget.scene_object(42),
        owner_id="scene_view",
        record_history=False,
    )
    assert projected[-1] is selected_object

    service.select(
        SelectionTarget.asset("Assets/Test.mat"),
        owner_id="project",
        record_history=False,
    )
    assert projected[-1] is None


def test_scene_selection_range_toggle_cross_view_and_asset_domain_switch():
    service = SelectionService()
    service.set_ordered_scene_objects("hierarchy", [1, 2, 3, 4, 5])
    anchor = SelectionTarget.scene_object(2)
    service.select(anchor, owner_id="scene_view")
    service.range_select_scene_object(5, owner_id="hierarchy")
    assert service.scene_object_ids() == (2, 3, 4, 5)
    assert service.snapshot.anchor == anchor
    assert service.snapshot.primary == SelectionTarget.scene_object(5)
    assert service.snapshot.owner_id == "hierarchy"

    service.toggle_scene_object(3, owner_id="scene_view")
    assert service.scene_object_ids() == (2, 4, 5)
    first = SelectionTarget.scene_object(1)
    service.toggle(first, owner_id="hierarchy")
    assert service.snapshot.targets == tuple(SelectionTarget.scene_object(value) for value in (2, 4, 5, 1))
    assert service.snapshot.anchor == anchor and service.snapshot.primary == first

    asset = SelectionTarget.asset("Assets/Test.mat")
    service.select(asset, owner_id="project")
    assert service.snapshot.domain is SelectionDomain.ASSET
    assert service.snapshot.targets == (asset,) and service.snapshot.primary == asset


def test_focus_request_acknowledgement_child_context_and_history_replay():
    focus = FocusService()
    changes = []
    focus.add_change_listener(changes.append)
    assert focus.request_panel_focus("project")
    assert focus.consume_panel_focus_request("project")
    assert not focus.consume_panel_focus_request("project")
    assert focus.activate_panel("project", child_context_id="project.search", reason="project_click")
    assert focus.snapshot.active_panel_id == "project"
    assert focus.snapshot.child_context_id == "project.search"
    assert not focus.activate_panel("project")
    assert focus.snapshot.child_context_id == "project.search"
    assert focus.apply_snapshot(FocusService().snapshot, reason="undo", record_history=False)
    assert [change.reason for change in changes] == ["project_click", "undo"]
    assert [change.record_history for change in changes] == [True, False]
