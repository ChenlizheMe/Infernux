"""Model mesh children share source geometry and survive editor reference workflows."""
import base64
import json
import struct

import numpy as np
import pytest

from test_model_hierarchy_instances import descendants, hierarchy_asset
from test_model_material_defaults import imported_model
from Infernux.lib import AssetRegistry, Vector3
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
        assert entry["subresource_id"]
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


def test_mesh_subresource_ids_survive_source_node_reorder(scene, hierarchy_asset, engine, monkeypatch):
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    before = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    ids = {tuple(item['path']): item['subresource_id'] for item in before}
    document = json.loads(source.read_text())
    document['nodes'][1]['children'].reverse()
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    after = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    assert {tuple(item['path']): item['subresource_id'] for item in after} == ids


def test_mesh_subresource_id_survives_unique_source_node_rename(scene, hierarchy_asset, engine, monkeypatch):
    """A DCC leaf rename updates the path without replacing its asset identity."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    before = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    upper = next(item for item in before if item['name'] == 'Upper')
    document = json.loads(source.read_text())
    document['nodes'][2]['name'] = 'Renamed Upper'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    after = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    renamed = next(item for item in after if item['name'] == 'Renamed Upper')
    assert renamed['subresource_id'] == upper['subresource_id']


def test_mesh_subresource_id_survives_unique_parent_rename(scene, hierarchy_asset, engine, monkeypatch):
    """Renaming a DCC pivot must not orphan a child mesh reference."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    document = json.loads(source.read_text())
    # Make the candidate unique first; the normal fixture intentionally has
    # two identical children, for which a parent rename must not guess.
    document['nodes'][1]['children'] = [2]
    document['nodes'].pop(3)
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    before = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    upper = next(item for item in before if item['name'] == 'Upper')
    document['nodes'][0]['name'] = 'Renamed Assembly'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    after = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    renamed = next(item for item in after if item['name'] == 'Upper')
    assert renamed['path'] == ['Renamed Assembly', 'Empty pivot', 'Upper']
    assert renamed['subresource_id'] == upper['subresource_id']


def test_mesh_subresource_id_survives_rename_with_artwork_edits(
        scene, hierarchy_asset, engine, monkeypatch):
    """Renaming while editing vertices/materials still addresses one source object."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)

    # Make the topology candidate unique and publish its current topology
    # identity before the compound edit.
    document = json.loads(source.read_text())
    document['nodes'][1]['children'] = [2]
    document['nodes'].pop(3)
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    before = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))[0]
    assert before['topology_key']
    root = scene.create_from_model(guid)
    live = descendants(root)['Upper']
    live_id = live.id
    live.transform.local_position = Vector3(7, 8, 9)

    document = json.loads(source.read_text())
    document['nodes'][2]['name'] = 'Renamed And Edited'
    encoded = document['buffers'][0]['uri'].split(',', 1)[1]
    payload = bytearray(base64.b64decode(encoded))
    struct.pack_into('<f', payload, 12, 2.0)
    document['buffers'][0]['uri'] = (
        'data:application/octet-stream;base64,'
        + base64.b64encode(payload).decode()
    )
    document['accessors'][0]['max'] = [2, 1, 0]
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)

    after = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))[0]
    assert after['path'][-1] == 'Renamed And Edited'
    assert after['geometry_key'] != before['geometry_key']
    assert after['topology_key'] == before['topology_key']
    assert after['subresource_id'] == before['subresource_id']
    renamed = descendants(root)['Renamed And Edited']
    assert renamed.id == live_id
    assert tuple(renamed.transform.local_position) == (7, 8, 9)
    assert renamed.get_component('MeshRenderer').model_subresource_id == before['subresource_id']


def test_mesh_renderer_cold_load_resolves_stable_identity_after_cross_parent_move(
        scene, hierarchy_asset, engine, monkeypatch):
    """A persisted node path is only a hint once a stable mesh identity exists."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)

    # Make Upper unambiguous before capturing its persisted reference.  The
    # stock fixture deliberately contains a geometry-identical sibling.
    source_document = json.loads(source.read_text())
    source_document['nodes'][1]['children'] = [2]
    source_document['nodes'].pop(3)
    source.write_text(json.dumps(source_document))
    assert AssetManager.reimport_asset(str(source), database=database)

    root = scene.create_from_model(guid)
    renderer = descendants(root)['Upper'].get_component('MeshRenderer')._require_cpp_component()
    old_document = renderer.serialize_document()
    old_path = old_document['modelNodePath']
    stable_id = old_document['modelSubresourceId']
    assert old_path == ['Assembly', 'Empty pivot', 'Upper']
    assert stable_id

    # The DCC moves the same mesh across a parent boundary.  Reimport retains
    # its stable identity while changing the authoritative path.
    source_document = json.loads(source.read_text())
    source_document['nodes'][0]['children'] = [1, 2]
    source_document['nodes'][1]['children'] = []
    source.write_text(json.dumps(source_document))
    assert AssetManager.reimport_asset(str(source), database=database)
    manifest = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    current = next(item for item in manifest if item['subresource_id'] == stable_id)
    current_path = current['path']
    assert current_path == ['Assembly', 'Upper']

    # ID + stale path resolves to the current imported node before strict node
    # validation, which is the cold scene-load contract.
    assert renderer.deserialize_document(old_document)
    assert renderer.serialize_document()['modelNodePath'] == current_path

    # Once present, the stable identity is authoritative: a valid path cannot
    # rescue a missing identity.
    missing_identity = dict(old_document)
    missing_identity['modelNodePath'] = current_path
    missing_identity['modelSubresourceId'] = 'missing-stable-model-mesh-id'
    assert renderer.deserialize_document(missing_identity) is False

    # Path-only references remain strict current references.  They neither
    # perform identity migration nor accept a stale path.
    path_only_current = dict(old_document)
    path_only_current.pop('modelSubresourceId')
    path_only_current['modelNodePath'] = current_path
    assert renderer.deserialize_document(path_only_current)
    path_only_stale = dict(path_only_current)
    path_only_stale['modelNodePath'] = old_path
    assert renderer.deserialize_document(path_only_stale) is False

    # Corrupt/ambiguous identity manifests are rejected instead of selecting
    # an arbitrary candidate or falling back to the serialized path.
    metadata = database.get_meta_by_guid(guid)
    meta_document = metadata.serialize_document()
    duplicate_manifest = list(manifest)
    duplicate = dict(current)
    duplicate['path'] = ['Assembly', 'Duplicate identity']
    duplicate_manifest.append(duplicate)
    meta_document['metadata']['model_meshes']['value'] = json.dumps(duplicate_manifest, separators=(',', ':'))
    metadata.deserialize_document(meta_document)
    refreshed_manifest = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    assert sum(item['subresource_id'] == stable_id for item in refreshed_manifest) == 2
    duplicate_identity = dict(old_document)
    duplicate_identity['modelNodePath'] = current_path
    assert renderer.deserialize_document(duplicate_identity) is False


def test_identical_node_geometry_has_same_signature(hierarchy_asset):
    database, _, guid = hierarchy_asset
    manifest = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    assert manifest[0]['geometry_key'] == manifest[1]['geometry_key']
    assert manifest[0]['identity_key'] == manifest[1]['identity_key']


@pytest.mark.parametrize('keep_original', [True, False])
def test_added_duplicate_cannot_steal_original_identity(hierarchy_asset, engine, monkeypatch, keep_original):
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    document = json.loads(source.read_text())
    document['nodes'][1]['children'] = [2]
    document['nodes'].pop()
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    before = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))[0]
    document['nodes'].append({'name': 'Added First', 'mesh': 0})
    document['nodes'][1]['children'] = [3, 2]
    if not keep_original:
        document['nodes'][2]['name'] = 'Renamed'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    after = json.loads(database.get_meta_by_guid(guid).get_string('model_meshes'))
    identifiers = [entry['subresource_id'] for entry in after]
    assert len(set(identifiers)) == len(identifiers)
    if keep_original:
        original = next(entry for entry in after if entry['name'] == 'Upper')
        assert original['subresource_id'] == before['subresource_id']
    else:
        # Two equally plausible rename candidates: neither inherits the old ID.
        assert before['subresource_id'] not in identifiers


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


def test_project_selection_roundtrip(hierarchy_asset, monkeypatch):
    from Infernux.engine._bootstrap_selection import _project_selection_target, _project_path_for_target
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    reference = make_model_mesh_reference(str(source), ['Assembly', 'Empty pivot', 'Upper'])
    target = _project_selection_target(reference)
    assert target.sub_kind == 'submesh'
    assert target.document_id == guid
    resolved_source, resolved_nodes = split_model_mesh_reference(
        _project_path_for_target(target)
    )
    assert resolved_source == database.get_path_from_guid(guid)
    assert resolved_nodes == ['Assembly', 'Empty pivot', 'Upper']
