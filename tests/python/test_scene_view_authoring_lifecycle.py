"""Scene View gestures and ownership using native objects and camera rays.

The tests call the production gesture/lifecycle entry points. They do not
replace picking with a claim about GUI hit testing or particle GPU output.
"""
from types import SimpleNamespace

import pytest

from infernux.components import ParticleSystem
from infernux.components.builtin import Camera
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.ui.core_panel_interactions import scene_view_panel_interaction
from infernux.engine.ui.scene_view_panel import SceneViewPanel, TOOL_SCALE
from infernux.engine.undo import UndoManager
from infernux.lib import InputManager, SceneManager, Vector3


class _NativeViewportHost:
    def __init__(self, engine):
        self.native = engine

    def get_native_engine(self):
        return self.native

    def __getattr__(self, name):
        return getattr(self.native, name)


class _CameraRays:
    def __init__(self, camera):
        self.camera = camera
        self.editor_camera = SimpleNamespace(position=camera.transform.position)

    def screen_to_world_ray(self, x, y, width, height):
        origin, direction = self.camera.screen_point_to_ray(x, y, width, height)
        return (*origin, *direction)


@pytest.fixture
def viewport(engine, scene):
    previous_undo = UndoManager.instance()
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    panel = SceneViewPanel(engine=_NativeViewportHost(engine))
    core.panels.register_type('scene_view', scene_view_panel_interaction(core.scene_objects))
    core.panels.bind_view('scene_view', 'scene_view', panel)
    core.focus.activate_panel('scene_view', view_id='scene_view', record_history=False)
    panel.on_enable()
    try:
        yield panel, core, undo, scene
    finally:
        panel.on_disable()
        core.shutdown()
        UndoManager._instance = previous_undo


def _begin_scale(viewport, scale, space, handle):
    panel, core, _, scene = viewport
    owner = scene.create_game_object('Scaled Object')
    owner.transform.local_scale = Vector3(*scale)
    camera_owner = scene.create_game_object('Ray Camera')
    camera_owner.transform.position = Vector3(0.0, 0.0, -10.0)
    rays = _CameraRays(camera_owner.add_component(Camera))
    panel._coord_space = space
    panel._gizmo_tool_mode = TOOL_SCALE
    core.selection.select_scene_object(owner.id, owner_id='scene_view', record_history=False)
    assert panel._start_gizmo_drag(rays, handle, SimpleNamespace(is_key_down=lambda _key: False),
                                  60.0, 40.0, 100.0, 100.0, TOOL_SCALE)
    return owner, rays


@pytest.mark.parametrize('space,handle', [(1, 1), (0, 1), (1, 4), (0, 4), (1, 7)])
@pytest.mark.parametrize('scale', [(2.0, 3.0, 4.0), (-2.0, 3.0, -4.0), (-0.0001, 0.0002, -0.0003)])
def test_scale_identity_preserves_authored_values(viewport, space, handle, scale):
    panel, _, _, _ = viewport
    owner, rays = _begin_scale(viewport, scale, space, handle)
    before = tuple(owner.transform.local_scale)
    panel._drag_scale(rays, 60.0, 40.0, 100.0, 100.0)
    actual = tuple(owner.transform.local_scale)
    panel._finish_gizmo_drag(TOOL_SCALE, commit=False)
    assert tuple(owner.transform.local_scale) == pytest.approx(before)
    assert actual == pytest.approx(before)


@pytest.mark.parametrize('space,handle', [(1, 1), (0, 1), (1, 4), (0, 4), (1, 7)])
def test_scale_preserves_mirroring_and_undo_redo(viewport, space, handle):
    panel, _, undo, _ = viewport
    owner, rays = _begin_scale(viewport, (-2.0, 3.0, -4.0), space, handle)
    before = tuple(owner.transform.local_scale)
    panel._drag_scale(rays, 70.0, 30.0, 100.0, 100.0)
    changed = tuple(owner.transform.local_scale)
    panel._update_gizmo_drag_transaction()
    panel._finish_gizmo_drag(TOOL_SCALE, commit=True)
    assert changed != pytest.approx(before)
    assert changed[0] < 0 and changed[1] > 0 and changed[2] < 0
    undo.undo()
    assert tuple(owner.transform.local_scale) == pytest.approx(before)
    undo.redo()
    assert tuple(owner.transform.local_scale) == pytest.approx(changed)


@pytest.mark.parametrize('transition', ['hidden', 'visible_focus_loss', 'disable'])
def test_camera_capture_ends_when_scene_view_retires(viewport, transition):
    panel, core, _, _ = viewport
    context = SimpleNamespace(get_global_mouse_pos_x=lambda: 150.0, get_global_mouse_pos_y=lambda: 250.0)
    panel._is_camera_dragging = True
    panel._was_right_down = panel._was_middle_down = True
    panel._begin_camera_capture(context)
    assert InputManager.instance().is_editor_mouse_capture_active
    assert core.focus.snapshot.capture_owner_id == 'scene_view.camera'
    core.focus.activate_panel('game_view', view_id='game_view', record_history=False)
    if transition == 'hidden':
        panel._on_not_visible(context)
        panel._on_not_visible(context)
    elif transition == 'visible_focus_loss':
        panel._on_visible_pre(context)
    else:
        panel.on_disable()
    assert not panel._camera_capture_active
    assert not InputManager.instance().is_editor_mouse_capture_active
    assert not core.focus.snapshot.capture_owner_id
    assert not panel._is_camera_dragging
    assert not panel._was_right_down and not panel._was_middle_down


@pytest.mark.parametrize('owner_state', ['active', 'additive', 'inactive'])
def test_particle_preview_resolves_selected_loaded_scene(viewport, owner_state):
    panel, core, _, active = viewport
    manager = SceneManager.instance()
    other = manager.create_scene('Additive Preview')
    manager.set_active_scene(active)
    owner_scene = other if owner_state == 'additive' else active
    owner = owner_scene.create_game_object('Selected Emitter')
    component = owner.add_component(ParticleSystem)
    if owner_state == 'inactive':
        owner.active = False
    core.selection.select_scene_object(owner.id, owner_id='scene_view', record_history=False)
    assert manager.get_active_scene().world_id == active.world_id
    if owner_state == 'inactive':
        assert panel._particle_preview_component is None
        return
    assert panel._particle_preview_component is component
    assert panel._particle_preview_object.id == owner.id
    if owner_state == 'additive':
        manager.set_active_scene(other)
        panel._restore_particle_preview_selection()
        assert panel._particle_preview_component is component
        manager.set_active_scene(active)
        panel._restore_particle_preview_selection()
        assert panel._particle_preview_component is component
        manager.unload_scene(other)
        panel._restore_particle_preview_selection()
        assert panel._particle_preview_component is None
    else:
        owner.active = False
        panel._tick_particle_preview(0.0)
        assert panel._particle_preview_component is None
