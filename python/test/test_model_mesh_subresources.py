"""Model mesh children share source geometry and survive editor reference workflows."""
import json

import numpy as np
import pytest

from test_model_hierarchy_instances import hierarchy_asset
from test_model_material_defaults import imported_model
from Infernux.lib import AssetRegistry
from Infernux.lib._Infernux import make_model_mesh_reference, split_model_mesh_reference


@pytest.mark.parametrize('path', [['Assembly', 'Upper'], ['中文/节点', 'x::submesh: !\\y']])
def test_mesh_reference_preserves_source_names(path):
    reference = make_model_mesh_reference('Assets/模型.blend', path)
    assert split_model_mesh_reference(reference) == ('Assets/模型.blend', path)
    assert split_model_mesh_reference('Assets/plain.fbx') == ('Assets/plain.fbx', [])


@pytest.mark.parametrize('suffix', ['z0', '0', '5b5d'])
def test_bad_mesh_reference_rejected(suffix):
    with pytest.raises(ValueError):
        split_model_mesh_reference('Assets/model.fbx::submesh:' + suffix)


def test_mesh_children_manifest_local_preview_and_scene_roundtrip(scene, hierarchy_asset):
    database, source, guid = hierarchy_asset
    mesh = AssetRegistry.instance().load_mesh(str(source))
    manifest = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    assert {entry['name'] for entry in manifest} == {'Upper', 'Lower'}
    for entry in manifest:
        local = mesh.create_model_node_copy(entry['path'])
        assert local.name == entry['name']
        assert local.vertex_count == 3
        assert local.index_count == 3
        assert local.submesh_count == 1
        np.testing.assert_allclose(local.get_vertex_data()['positions'], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        reference = make_model_mesh_reference(guid, entry['path'])
        obj = scene.create_from_model(reference)
        assert obj.name == entry['name']
        assert obj.get_children() == []
        renderer = obj.get_component('MeshRenderer')
        doc = renderer.serialize_document()
        assert doc['meshAssetGuid'] == guid
        assert doc['modelNodePath'] == entry['path']
        assert len(renderer.get_submesh_infos()) == 1
        assert renderer.deserialize_document(doc)
        assert renderer.serialize_document()['modelNodePath'] == entry['path']


def test_node_material_inspector_does_not_keep_previous_node(imported_model):
    from types import SimpleNamespace
    from Infernux.engine.bootstrap_inspector._materials import _collect_material_renderers, _rebuild_material_entries
    renderer, _, _, _, _ = imported_model
    obj = renderer.game_object
    items = [SimpleNamespace(is_native=True, type_name='MeshRenderer', component_id=renderer.component_id)]
    native = {renderer.component_id: renderer}
    guid = renderer.mesh_asset_guid
    renderer.set_model_mesh(guid, ['Assembly', 'Empty pivot', 'Upper'])
    first, first_signature = _collect_material_renderers(items, native, obj)
    first_entries = _rebuild_material_entries(first)
    renderer.set_model_mesh(guid, ['Assembly', 'Empty pivot', 'Lower'])
    second, second_signature = _collect_material_renderers(items, native, obj)
    assert first_signature != second_signature
    assert first_entries[0]['label'] == 'Red (Slot 0)'
    assert _rebuild_material_entries(second)[0]['label'] == 'Green (Slot 0)'


def test_node_drop_and_assignment_use_global_undo(scene, hierarchy_asset, monkeypatch, engine):
    from Infernux.core.assets import AssetManager
    from Infernux.engine.undo import UndoManager
    from Infernux.engine.interaction import ClipboardService, SelectionService, SceneObjectCommandService
    from Infernux.engine.ui._inspector_extra_renderers import (
        _guid_and_path_from_model_payload, _mesh_additional_picker_items, _mesh_display_name,
    )
    from Infernux.engine.interaction.components import ComponentCommandService

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    path = ['Assembly', 'Empty pivot', 'Upper']
    virtual = make_model_mesh_reference(str(source), path)
    assert _guid_and_path_from_model_payload(virtual) == (guid, virtual)
    picks = _mesh_additional_picker_items('Hierarchy.gltf/Upper')
    assert any(item[1]['guid'] == guid and split_model_mesh_reference(item[1]['path_hint'])[1] == path
               for item in picks)
    previous = UndoManager._instance
    manager = UndoManager()
    UndoManager._instance = manager
    try:
        service = SceneObjectCommandService(SelectionService(), ClipboardService())
        obj = service.create_model_object(virtual)
        assert obj.name == 'Upper'
        object_id = obj.id
        manager.undo()
        assert scene.find_by_id(object_id) is None
        manager.redo()
        obj = scene.find_by_id(object_id)
        renderer = obj.get_component('MeshRenderer')
        assert _mesh_display_name(renderer) == 'Upper'
        before = renderer.serialize_document()
        commands = ComponentCommandService()
        from Infernux.engine.interaction.object_fields import AssetReferenceFieldModel
        from Infernux.engine.ui._inspector_extra_renderers import _assign_model_mesh, _mesh_reference_value
        field = AssetReferenceFieldModel(field_id='test.mesh', display_text='Upper', type_hint='Mesh',
                                        reference_value=_mesh_reference_value(renderer),
                                        on_assign=lambda value: _assign_model_mesh(renderer, value))
        field.dispatch_drop(make_model_mesh_reference(str(source), ['Assembly', 'Empty pivot', 'Lower']))
        assert _mesh_display_name(renderer) == 'Lower'
        manager.undo()
        assert renderer.serialize_document() == before
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_missing_mesh_does_not_create_or_modify_object(scene, hierarchy_asset):
    _, _, guid = hierarchy_asset
    before = scene.serialize_document()
    with pytest.raises(ValueError, match='no longer exists'):
        scene.create_from_model(make_model_mesh_reference(guid, ['Missing']))
    assert scene.serialize_document() == before


def test_project_selection_roundtrip(hierarchy_asset):
    from Infernux.engine._bootstrap_selection import _project_selection_target, _project_path_for_target
    _, source, _ = hierarchy_asset
    reference = make_model_mesh_reference(str(source), ['Assembly', 'Empty pivot', 'Upper'])
    target = _project_selection_target(reference)
    assert target.sub_kind == 'submesh'
    assert _project_path_for_target(target) == reference
