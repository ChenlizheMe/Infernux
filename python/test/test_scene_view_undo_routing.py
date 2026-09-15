from __future__ import annotations

from types import SimpleNamespace

from Infernux.engine.ui.scene_view_panel import SceneViewPanel
from Infernux.engine.ui._scene_view_gizmo import SceneViewGizmoMixin
from Infernux.engine.ui.scene_view_panel import (
    TOOL_RECT,
    TOOL_ROTATE,
    TOOL_TRANSLATE,
)
from Infernux.engine.interaction import (
    ContinuousEditService,
    FocusService,
    SelectionService,
    TransientInteractionService,
    ViewCommandService,
)
from Infernux.engine.undo import UndoManager
from Infernux.lib import Vector3


class _EditorCameraStub:
    def __init__(self):
        self.position = Vector3(0.0, 0.0, -10.0)
        self.rotation = (0.0, 0.0)
        self.focus_distance = 10.0
        self.fov = 60.0

    def restore_state(
        self,
        px,
        py,
        pz,
        _fx,
        _fy,
        _fz,
        distance,
        yaw,
        pitch,
    ):
        self.position = Vector3(px, py, pz)
        self.focus_distance = float(distance)
        self.rotation = (float(yaw), float(pitch))


def _install_gizmo_interaction_services():
    previous = (
        UndoManager.instance(),
        FocusService._instance,
        ContinuousEditService._instance,
        TransientInteractionService._instance,
    )
    manager = UndoManager()
    focus = FocusService()
    focus.activate_panel(
        "scene_view",
        view_id="scene_view",
        record_history=False,
    )
    ContinuousEditService()
    transients = TransientInteractionService(focus)
    return previous, manager, transients


def _restore_gizmo_interaction_services(previous):
    (
        UndoManager._instance,
        FocusService._instance,
        ContinuousEditService._instance,
        TransientInteractionService._instance,
    ) = previous


def _begin_test_gizmo_drag(panel, owner):
    panel._is_gizmo_dragging = True
    panel._gizmo_drag_obj_id = int(owner.id)
    panel._gizmo_drag_items = {
        int(owner.id): panel._snapshot_gizmo_object(owner),
    }
    panel._begin_gizmo_drag_transaction(TOOL_TRANSLATE)


def test_scene_view_has_no_private_structural_shortcut_handler():
    assert not hasattr(SceneViewPanel, "_handle_object_clipboard_shortcuts")


def test_frame_selected_camera_state_is_undoable_without_dirtying_scene():
    previous_manager = UndoManager.instance()
    previous_view_commands = ViewCommandService.instance()
    manager = UndoManager()
    ViewCommandService()
    camera = _EditorCameraStub()
    panel = SceneViewPanel(engine=SimpleNamespace(editor_camera=camera))
    panel._compute_object_bounds = lambda _obj: ((5.0, 2.0, 1.0), 2.0)
    try:
        assert panel.fly_to_object(SimpleNamespace(id=17))
        assert manager.undo_description == "Frame Selected"
        entry = manager.action_journal.peek_undo()
        assert entry is not None
        assert not entry.action.marks_dirty

        panel._tick_fly_to(1.0)
        assert tuple(camera.position) != (0.0, 0.0, -10.0)

        manager.undo()
        panel._tick_fly_to(1.0)
        assert tuple(camera.position) == (0.0, 0.0, -10.0)

        manager.redo()
        panel._tick_fly_to(1.0)
        assert tuple(camera.position) != (0.0, 0.0, -10.0)
    finally:
        UndoManager._instance = previous_manager
        ViewCommandService._instance = previous_view_commands


def test_gizmo_drag_orders_selection_primary_first(monkeypatch):
    import Infernux.lib._Infernux as native

    old = SelectionService._instance
    selection = SelectionService()
    try:
        selection.replace_scene_objects(
            [10, 20], owner_id="scene_view", record_history=False
        )
        objects = {
            10: SimpleNamespace(id=10),
            20: SimpleNamespace(id=20),
        }
        scene = SimpleNamespace(find_by_id=lambda object_id: objects.get(object_id))
        manager = SimpleNamespace(
            find_runtime_object_by_id=lambda object_id: objects.get(object_id)
        )
        monkeypatch.setattr(
            native,
            "SceneManager",
            SimpleNamespace(instance=lambda: manager),
        )
        mixin = object.__new__(SceneViewGizmoMixin)

        result = mixin._get_gizmo_drag_objects(
            scene, selection.primary_scene_object_id()
        )

        assert [obj.id for obj in result] == [20, 10]
    finally:
        SelectionService._instance = old


def test_gizmo_drag_uses_only_topmost_selected_hierarchy_roots(scene):
    previous_selection = SelectionService._instance
    selection = SelectionService()
    SelectionService.install(selection)
    try:
        parent = scene.create_game_object("Selected Parent")
        child = scene.create_game_object("Selected Child")
        sibling = scene.create_game_object("Selected Sibling")
        child.set_parent(parent, True)
        selection.replace_scene_objects(
            [int(parent.id), int(child.id), int(sibling.id)],
            owner_id="scene_view",
            record_history=False,
        )
        panel = SceneViewPanel(engine=None)

        objects = panel._get_gizmo_drag_objects(
            scene, selection.primary_scene_object_id(),
        )

        assert [int(obj.id) for obj in objects] == [int(sibling.id), int(parent.id)]
        assert selection.scene_object_ids() == (
            int(parent.id), int(child.id), int(sibling.id),
        )
    finally:
        SelectionService._instance = previous_selection


def test_rect_gizmo_can_begin_on_linked_prefab_child(scene):
    previous_selection = SelectionService._instance
    previous_services, _manager, _transients = _install_gizmo_interaction_services()
    selection = SelectionService()
    SelectionService.install(selection)
    try:
        parent = scene.create_game_object("Prefab Root")
        child = scene.create_game_object("Prefab Child")
        child.set_parent(parent, True)
        parent.prefab_guid = "prefab-guid"
        parent.prefab_root = True
        child.prefab_guid = "prefab-guid"
        child.prefab_root = False
        selection.replace_scene_objects(
            [int(child.id)], owner_id="scene_view", record_history=False,
        )
        panel = SceneViewPanel(engine=None)
        panel._plane_hit_coords = lambda *_args: (0.0, 0.0)
        native = SimpleNamespace(
            get_editor_rect_frame=lambda: {
                "center": (0.0, 0.0, 0.0),
                "half_size": (0.5, 0.5),
                "axis_indices": (0, 1),
                "axis_u": (1.0, 0.0, 0.0),
                "axis_v": (0.0, 1.0, 0.0),
            },
        )
        engine = SimpleNamespace(get_native_engine=lambda: native)
        ctx = SimpleNamespace(is_key_down=lambda _key: False)

        assert panel._start_gizmo_drag(
            engine, 15, ctx, 0.0, 0.0, 100.0, 100.0, TOOL_RECT,
        )
        assert list(panel._gizmo_drag_items) == [int(child.id)]
        panel._interrupt_gizmo_drag(commit=False)
    finally:
        SelectionService._instance = previous_selection
        _restore_gizmo_interaction_services(previous_services)


def test_particle_system_owner_gizmo_drag_records_direct_transform_undo(scene):
    from Infernux.components import ParticleSystem

    previous_manager = UndoManager.instance()
    manager = UndoManager()
    try:
        owner = scene.create_game_object("Particle Gizmo Undo")
        owner.add_py_component(ParticleSystem())
        panel = SceneViewPanel(engine=None)
        panel._gizmo_drag_obj_id = int(owner.id)
        panel._gizmo_drag_items = {
            int(owner.id): panel._snapshot_gizmo_object(owner),
        }

        owner.transform.position = Vector3(3.0, 2.0, 1.0)
        panel._record_gizmo_undo(TOOL_TRANSLATE)

        assert manager.undo_description == "Translate"
        manager.undo()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        manager.redo()
        assert tuple(owner.transform.position) == (3.0, 2.0, 1.0)
    finally:
        UndoManager._instance = previous_manager


def test_rect_gizmo_records_position_and_scale_as_one_undo(scene):
    previous_manager = UndoManager.instance()
    manager = UndoManager()
    try:
        owner = scene.create_game_object("Rect Gizmo Undo")
        panel = SceneViewPanel(engine=None)
        panel._gizmo_drag_obj_id = int(owner.id)
        panel._gizmo_drag_items = {
            int(owner.id): panel._snapshot_gizmo_object(owner),
        }

        owner.transform.position = Vector3(3.0, 2.0, 1.0)
        owner.transform.local_scale = Vector3(2.0, 4.0, 1.0)
        panel._record_gizmo_undo(TOOL_RECT)

        assert manager.undo_description == "Rect"
        manager.undo()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        assert tuple(owner.transform.local_scale) == (1.0, 1.0, 1.0)
        manager.redo()
        assert tuple(owner.transform.position) == (3.0, 2.0, 1.0)
        assert tuple(owner.transform.local_scale) == (2.0, 4.0, 1.0)
    finally:
        UndoManager._instance = previous_manager


def test_rect_gizmo_records_multiple_objects_as_one_undo(scene):
    previous_manager = UndoManager.instance()
    manager = UndoManager()
    try:
        first = scene.create_game_object("Rect Multi First")
        second = scene.create_game_object("Rect Multi Second")
        first.transform.position = Vector3(-2.0, 0.0, 0.0)
        second.transform.position = Vector3(2.0, 0.0, 0.0)
        panel = SceneViewPanel(engine=None)
        panel._gizmo_drag_obj_id = int(first.id)
        panel._gizmo_drag_items = {
            int(first.id): panel._snapshot_gizmo_object(first),
            int(second.id): panel._snapshot_gizmo_object(second),
        }

        first.transform.position = Vector3(-1.5, 1.0, 0.0)
        second.transform.position = Vector3(2.5, 1.0, 0.0)
        first.transform.local_scale = Vector3(-2.0, 3.0, 1.0)
        second.transform.local_scale = Vector3(2.0, 3.0, 1.0)
        panel._record_gizmo_undo(TOOL_RECT)

        assert manager.undo_description == "Rect"
        manager.undo()
        assert tuple(first.transform.position) == (-2.0, 0.0, 0.0)
        assert tuple(second.transform.position) == (2.0, 0.0, 0.0)
        assert tuple(first.transform.local_scale) == (1.0, 1.0, 1.0)
        assert tuple(second.transform.local_scale) == (1.0, 1.0, 1.0)
        manager.redo()
        assert tuple(first.transform.local_scale) == (-2.0, 3.0, 1.0)
        assert tuple(second.transform.local_scale) == (2.0, 3.0, 1.0)
    finally:
        UndoManager._instance = previous_manager


def test_rect_gizmo_records_world_ui_layout_instead_of_transform_scale(scene):
    from Infernux.ui import UIFrame
    from Infernux.ui.enums import UILayoutSizing

    previous_manager = UndoManager.instance()
    manager = UndoManager()
    try:
        owner = scene.create_game_object("World UI Rect Undo")
        component = UIFrame()
        component.width = 240.0
        component.height = 120.0
        component.width_sizing = UILayoutSizing.Hug
        owner.add_py_component(component)
        panel = SceneViewPanel(engine=None)
        panel._gizmo_drag_obj_id = int(owner.id)
        panel._gizmo_drag_items = {
            int(owner.id): panel._snapshot_gizmo_object(owner),
        }

        component.width = 360.0
        component.height = 180.0
        component.width_sizing = UILayoutSizing.Fixed
        owner.transform.position = Vector3(1.0, 2.0, 3.0)
        panel._record_gizmo_undo(TOOL_RECT)

        assert manager.undo_description == "Rect"
        manager.undo()
        assert component.width == 240.0
        assert component.height == 120.0
        assert component.width_sizing == UILayoutSizing.Hug
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        assert tuple(owner.transform.local_scale) == (1.0, 1.0, 1.0)
        manager.redo()
        assert component.width == 360.0
        assert component.height == 180.0
        assert component.width_sizing == UILayoutSizing.Fixed
        assert tuple(owner.transform.position) == (1.0, 2.0, 3.0)
        assert tuple(owner.transform.local_scale) == (1.0, 1.0, 1.0)
    finally:
        UndoManager._instance = previous_manager


def test_world_ui_rect_frame_uses_layout_size_and_engine_scale(scene):
    from Infernux.engine.ui.ui_rect_manipulation import resolve_world_ui_frame
    from Infernux.ui import UIFrame

    owner = scene.create_game_object("World UI Bounds")
    owner.transform.position = Vector3(2.0, 3.0, 4.0)
    component = UIFrame()
    component.width = 200.0
    component.height = 100.0
    owner.add_py_component(component)

    frame = resolve_world_ui_frame(component)

    assert frame is not None
    assert frame["object_id"] == int(owner.id)
    assert frame["center"] == (2.0, 3.0, 4.0)
    assert frame["axis_u"] == (1.0, 0.0, 0.0)
    assert frame["axis_v"] == (0.0, -1.0, 0.0)
    assert frame["half_size"] == (1.0, 0.5)


def test_gizmo_transform_history_preserves_pointer_down_selection(scene):
    from Infernux.engine.interaction import ContextRestoreStatus, EditorContextSnapshot

    previous_manager = UndoManager.instance()
    previous_selection = SelectionService._instance
    manager = UndoManager()
    selection = SelectionService()
    SelectionService.install(selection)
    def restore_context(context, _phase):
        selection.apply_snapshot(
            context.selection,
            reason="test_gizmo_restore",
            record_history=False,
        )
        return ContextRestoreStatus.READY

    manager.set_context_hooks(
        lambda: EditorContextSnapshot(selection=selection.snapshot),
        restore_context,
    )
    try:
        owner = scene.create_game_object("Selected Gizmo Target")
        selection.replace_scene_objects(
            [int(owner.id)],
            owner_id="hierarchy",
            record_history=False,
        )
        panel = SceneViewPanel(engine=None)
        panel._gizmo_drag_obj_id = int(owner.id)
        panel._gizmo_drag_items = {
            int(owner.id): panel._snapshot_gizmo_object(owner),
        }
        panel._gizmo_drag_selection_snapshot = selection.snapshot

        owner.transform.position = Vector3(4.0, 0.0, 0.0)
        selection.clear(reason="late_frame_selection_noise", record_history=False)
        panel._record_gizmo_undo(TOOL_TRANSLATE)

        entry = manager.action_journal.peek_undo()
        assert entry is not None
        assert entry.before_context.selection.primary.scene_object_id() == int(owner.id)
        assert entry.after_context.selection.primary.scene_object_id() == int(owner.id)

        manager.undo()
        assert selection.primary_scene_object_id() == int(owner.id)
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    finally:
        UndoManager._instance = previous_manager
        SelectionService._instance = previous_selection


def test_gizmo_tool_switch_commits_live_transform_once(scene):
    from Infernux.engine.ui.inspector_snapshot import (
        InspectorSnapshotService,
        InspectorTarget,
    )

    previous, manager, _transients = _install_gizmo_interaction_services()
    try:
        owner = scene.create_game_object("Tool Switch Gizmo")
        snapshots = InspectorSnapshotService.instance()
        snapshots.reset_for_tests()
        target = InspectorTarget.scene_object(int(owner.id))
        before = snapshots.snapshot(target)
        panel = SceneViewPanel(engine=None)
        _begin_test_gizmo_drag(panel, owner)
        owner.transform.position = Vector3(5.0, 1.0, 0.0)
        panel._update_gizmo_drag_transaction()

        assert snapshots.snapshot(target).value_revision > before.value_revision

        panel._set_tool_mode(TOOL_ROTATE)

        assert not panel._is_gizmo_dragging
        assert manager.undo_description == "Translate"
        manager.undo()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        manager.redo()
        assert tuple(owner.transform.position) == (5.0, 1.0, 0.0)
    finally:
        _restore_gizmo_interaction_services(previous)


def test_gizmo_focus_loss_commits_through_transient_owner(scene):
    previous, manager, transients = _install_gizmo_interaction_services()
    try:
        owner = scene.create_game_object("Focus Loss Gizmo")
        panel = SceneViewPanel(engine=None)
        _begin_test_gizmo_drag(panel, owner)
        owner.transform.position = Vector3(2.0, 3.0, 4.0)
        panel._update_gizmo_drag_transaction()

        assert transients.cancel_owner("scene_view") == 1

        assert not panel._is_gizmo_dragging
        assert manager.undo_description == "Translate"
        manager.undo()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    finally:
        _restore_gizmo_interaction_services(previous)


def test_gizmo_panel_close_commits_live_transform(scene):
    previous, manager, _transients = _install_gizmo_interaction_services()
    try:
        owner = scene.create_game_object("Panel Close Gizmo")
        panel = SceneViewPanel(engine=None)
        _begin_test_gizmo_drag(panel, owner)
        owner.transform.position = Vector3(7.0, 0.0, 0.0)
        panel._update_gizmo_drag_transaction()

        panel.on_disable()

        assert not panel._is_gizmo_dragging
        assert manager.undo_description == "Translate"
        manager.undo()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    finally:
        _restore_gizmo_interaction_services(previous)


def test_gizmo_interruption_rolls_back_when_history_rejects_commit(scene):
    previous, manager, _transients = _install_gizmo_interaction_services()
    try:
        owner = scene.create_game_object("Rejected Gizmo")
        panel = SceneViewPanel(engine=None)
        _begin_test_gizmo_drag(panel, owner)
        owner.transform.position = Vector3(9.0, 0.0, 0.0)
        panel._update_gizmo_drag_transaction()
        manager.enabled = False

        panel._set_tool_mode(TOOL_ROTATE)

        assert not panel._is_gizmo_dragging
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        assert manager.undo_description == ""
    finally:
        _restore_gizmo_interaction_services(previous)
