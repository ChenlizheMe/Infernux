"""Real renderer wrappers and retired scene graphs at editor panel boundaries."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("renderer_type", ["MeshRenderer", "SkinnedMeshRenderer", "SpriteRenderer", "LineRenderer"])
def test_inspector_collects_each_supported_renderer(scene, renderer_type):
    from infernux.engine.bootstrap_inspector._materials import _collect_material_renderers, _rebuild_material_entries

    owner = scene.create_game_object(renderer_type)
    renderer = owner.add_component(renderer_type)
    item = SimpleNamespace(is_native=True, type_name=renderer_type, component_id=renderer.component_id)
    collected, signature = _collect_material_renderers([item], {renderer.component_id: renderer}, owner)
    assert len(collected) == len(signature) == 1
    entries = _rebuild_material_entries(collected)
    if renderer_type == "SpriteRenderer":
        assert collected[0][3] == ()
        assert entries and entries[0]["label"] == "Element 0"


@pytest.mark.parametrize("replacement", ["same_world", "unload"])
def test_game_view_cancellation_discards_retired_pointer_targets(scene, replacement):
    from infernux.engine.ui.game_view_panel import GameViewPanel
    from infernux.lib import SceneManager
    from infernux.ui.ui_canvas import UICanvas
    from infernux.ui.ui_event_data import PointerType
    from infernux.ui.ui_event_system import _PointerState

    owner = scene.create_game_object("Captured runtime object")
    canvas = owner.add_component(UICanvas)
    panel = GameViewPanel()
    panel._synchronize_input_scene(scene)
    panel._mouse_event_dispatcher._hover_object = owner
    panel._mouse_event_dispatcher._pressed_object = owner
    panel._ui_event_processor._pointers[(PointerType.Mouse, -1)] = _PointerState(
        pointer_type=PointerType.Mouse, hover_canvas=canvas,
    )
    if replacement == "same_world":
        assert scene._commit_document(scene.serialize_document())
    else:
        manager = SceneManager.instance()
        fresh = manager.create_scene("Restored author scene")
        manager.set_active_scene(fresh)
        manager.unload_scene(scene)
    panel._reset_pointer_input()
    assert panel._mouse_event_dispatcher._hover_object is None
    assert panel._mouse_event_dispatcher._pressed_object is None
    assert not panel._ui_event_processor._pointers


def test_runtime_material_preview_does_not_create_an_asset_identity(tmp_path, monkeypatch):
    from infernux.core.material import Material
    from infernux.engine import project_context
    from infernux.engine.ui import inspector_material, inspector_support

    monkeypatch.setattr(project_context, 'get_project_root', lambda: str(tmp_path))
    material = Material.create_unlit().native
    material.name = 'SpriteUnlit_Default_60'
    panel = SimpleNamespace(_inline_material_cache={})
    before = material.serialize_document()
    for _ in range(2):
        assert inspector_support.ensure_material_file_path(material) == ''
        state = inspector_material._build_inline_state(panel, material)
        assert not state.document_id and state.resource_controller is None
    assert not material.file_path and not material.guid
    assert material.serialize_document() == before
    assert not list(tmp_path.iterdir())


def test_runtime_inline_material_shader_refresh_is_read_only(monkeypatch):
    from infernux.core.material import Material
    from infernux.engine.ui import inspector_material as ui

    material = Material.create_unlit().native
    panel = SimpleNamespace(_inline_material_cache={})
    state = ui._build_inline_state(panel, material)
    readonly = []
    for name in ('_native_mat', '_cached_data', '_shader_cache'):
        monkeypatch.setattr(ui, name, getattr(ui, name))
    monkeypatch.setattr(ui, '_sync_shader_annotations', lambda *args: (True, True))
    monkeypatch.setattr(ui, '_render_material_top',
                        lambda ctx, panel, state, data, locked, *args:
                        (readonly.append(locked) or (False, False, False, '', False, False)))
    monkeypatch.setattr(ui, '_render_virtualized_material_block', lambda *args: (False, '', False))
    monkeypatch.setattr(ui, '_flush_deferred_undo', lambda *args, **kwargs: None)
    def reject_edit(*args, **kwargs):
        raise AssertionError('Runtime material refresh entered asset authoring')
    monkeypatch.setattr(ui, '_apply_material_changes', reject_edit)
    context = SimpleNamespace(separator=lambda: None, is_any_item_active=lambda: False,
                              want_text_input=lambda: False)
    ui._render_material_body_impl(context, panel, state)
    assert readonly == [True]
