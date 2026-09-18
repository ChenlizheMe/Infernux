"""Imported local geometry and authoring hierarchy share one transform chain."""
import json
import base64
import struct
from pathlib import Path

import numpy as np
import pytest

import infernux as inx
from Infernux.lib import Physics, Vector3


@pytest.fixture
def hierarchy_asset(engine, tmp_path):
    database = engine.get_asset_database()
    source = Path(database.assets_root) / tmp_path.name / "Hierarchy.gltf"
    source.parent.mkdir(parents=True)
    original = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_hierarchy.gltf"
    source.write_bytes(original.read_bytes())
    result = database.import_asset(str(source))
    assert result, result.error
    yield database, source, result.guid
    database.delete_asset(str(source))
    source.unlink(missing_ok=True)
    Path(str(source) + '.meta').unlink(missing_ok=True)


def world_matrix(obj):
    return np.asarray(obj.transform.local_to_world_matrix()).reshape((4, 4), order='F')


def descendants(root):
    result = {}
    def visit(obj):
        for child in obj.get_children():
            result[child.name] = child
            visit(child)
    visit(root)
    return result


def test_model_hierarchy_preserves_empty_pivots_local_geometry_and_roundtrip(scene, hierarchy_asset):
    _, _, guid = hierarchy_asset
    mesh = inx.Mesh.load_guid(guid)
    root = scene.create_from_model(guid, 'Imported Assembly')
    objects = descendants(root)
    root_document = root.serialize_document()
    assert root_document['model_source'] == {'guid': guid, 'path': []}
    assert objects['Upper'].serialize_document()['model_source']['path'][-1] == 'Upper'
    assert objects['Empty pivot'].get_parent().name == 'Assembly'
    assert objects['Upper'].get_parent().name == 'Empty pivot'
    assert objects['Lower'].get_parent().name == 'Empty pivot'
    assert objects['Empty pivot'].get_component('MeshRenderer') is None
    matrices = []
    for node in mesh.model_nodes:
        local = np.asarray(node['local_matrix'])
        matrices.append(local if node['parent_index'] < 0 else matrices[node['parent_index']] @ local)
        np.testing.assert_allclose(world_matrix(objects[node['name']]), matrices[-1], atol=2e-5)
        if node['node_group'] < 0:
            continue
        renderer = objects[node['name']].get_component('MeshRenderer')._require_cpp_component()
        assert renderer.serialize_document()['modelNodePath'][-1] == node['name']
        assert renderer.serialize_document()['meshAssetGuid'] == guid
        local_positions = np.asarray(renderer.get_positions())
        for index in range(mesh.submesh_count):
            sub = mesh.get_submesh(index)
            if sub['node_group'] != node['node_group']:
                continue
            start, count = sub['vertex_start'], sub['vertex_count']
            points = local_positions[start:start+count]
            transformed = (matrices[-1] @ np.column_stack([points, np.ones(count)]).T).T[:, :3]
            np.testing.assert_allclose(transformed, mesh.vertex_buffer['positions'][start:start+count], atol=2e-5)
            np.testing.assert_allclose(renderer.get_world_bounds(),
                                      np.r_[transformed.min(axis=0), transformed.max(axis=0)], atol=2e-5)
    document = scene.serialize_document()
    assert scene._commit_document(document)
    restored = next(o for o in scene.get_root_objects() if o.name == 'Imported Assembly')
    assert len(descendants(restored)) == len(objects)
    for name, obj in descendants(restored).items():
        renderer = obj.get_component('MeshRenderer')
        if renderer:
            assert renderer.serialize_document()['modelNodePath'][-1] == name


def test_model_node_collision_and_parent_edits_use_local_geometry(scene, hierarchy_asset):
    _, _, guid = hierarchy_asset
    root = scene.create_from_model(guid)
    objects = descendants(root)
    upper = objects['Upper']
    collider = upper.add_component('MeshCollider')
    collider.convex = False
    Physics.sync_transforms()
    assert not collider.shape_error
    # Imported triangle under a mirrored, nonuniformly scaled parent.
    before = world_matrix(upper)
    point = (before @ np.array([.2, .2, 0, 1]))[:3]
    hit = Physics.raycast(Vector3(*(point + [0, 0, 5])), Vector3(0, 0, -1), 10)
    assert hit is not None
    root.transform.position = Vector3(20, 0, 0)
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(*(point + [0, 0, 5])), Vector3(0, 0, -1), 10) is None
    assert Physics.raycast(Vector3(*(point + [20, 0, 5])), Vector3(0, 0, -1), 10) is not None


def test_non_trs_source_is_rejected_without_partial_scene(scene, hierarchy_asset):
    database, source, _ = hierarchy_asset
    document = json.loads(source.read_text())
    document['nodes'][0].pop('translation')
    document['nodes'][0]['matrix'] = [1, 0, 0, 0, .5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    source.write_text(json.dumps(document))
    result = database.import_asset(str(source))
    assert result, result.error
    before = scene.serialize_document()
    with pytest.raises(ValueError, match='shear'):
        scene.create_from_model(result.guid)
    assert scene.serialize_document() == before


def test_model_creation_is_one_undoable_hierarchy(scene, hierarchy_asset):
    from Infernux.engine.undo import UndoManager
    from Infernux.engine.interaction import ClipboardService, SelectionService, SceneObjectCommandService
    _, _, guid = hierarchy_asset
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        service = SceneObjectCommandService(SelectionService(), ClipboardService())
        root = service.create_model_object(guid, is_guid=True, name='Undoable model')
        assert root is not None
        count = len(descendants(root))
        assert count >= 4
        assert len(manager.action_journal.entries) == 1
        manager.undo()
        assert all(o.name != 'Undoable model' for o in scene.get_root_objects())
        manager.redo()
        restored = next(o for o in scene.get_root_objects() if o.name == 'Undoable model')
        assert len(descendants(restored)) == count
        renderer = descendants(restored)['Upper'].get_component('MeshRenderer')
        assert renderer.serialize_document()['modelNodePath'][-1] == 'Upper'
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_replacing_hierarchy_mesh_clears_local_view(scene, hierarchy_asset):
    _, _, guid = hierarchy_asset
    root = scene.create_from_model(guid)
    renderer = descendants(root)['Upper'].get_component('MeshRenderer')._require_cpp_component()
    saved = renderer.serialize_document()
    triangle = inx.Mesh.from_data(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                                 np.array([0, 1, 2], np.uint32))
    try:
        renderer.set_mesh_asset_guid(triangle.guid)
        assert 'modelNodePath' not in renderer.serialize_document()
        assert renderer.deserialize_document(saved)
        assert renderer.serialize_document()['modelNodePath'][-1] == 'Upper'
        renderer.clear_mesh_asset()
        assert 'modelNodePath' not in renderer.serialize_document()
    finally:
        triangle.destroy()


def test_geometry_reimport_keeps_instance_edits_and_updates_local_stream(scene, hierarchy_asset, engine, monkeypatch):
    from Infernux.core.assets import AssetManager
    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    root = scene.create_from_model(guid)
    objects = descendants(root)
    upper = objects['Upper']
    upper.transform.local_position = Vector3(7, 8, 9)
    renderer = upper.get_component('MeshRenderer')._require_cpp_component()
    before = np.asarray(renderer.get_positions())
    source_document = json.loads(source.read_text())
    encoded = source_document['buffers'][0]['uri'].split(',', 1)[1]
    payload = bytearray(base64.b64decode(encoded))
    struct.pack_into('<f', payload, 12, 2.0)  # second vertex x: 1 -> 2
    source_document['buffers'][0]['uri'] = 'data:application/octet-stream;base64,' + base64.b64encode(payload).decode()
    source_document['accessors'][0]['max'] = [2, 1, 0]
    source.write_text(json.dumps(source_document))
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert result.guid == guid
    assert not np.array_equal(renderer.get_positions(), before)
    assert renderer.serialize_document()['modelNodePath'][-1] == 'Upper'
    assert upper.transform.local_position.x == 7
    assert upper.get_parent().name == 'Empty pivot'


def test_external_model_source_reconciles_instances_without_overwriting_transforms(
    scene, hierarchy_asset, engine, monkeypatch
):
    """Every live instance follows source add/remove while author transforms win."""
    from Infernux.core.assets import AssetManager
    from Infernux.debug import Debug

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    source_warnings = []
    monkeypatch.setattr(Debug, 'log_warning', lambda message, context=None: source_warnings.append(str(message)))
    first = scene.create_from_model(guid, 'First Assembly')
    second = scene.create_from_model(guid, 'Second Assembly')
    first.transform.local_position = Vector3(10, 0, 0)
    second.transform.local_position = Vector3(-10, 0, 0)
    first_upper = descendants(first)['Upper']
    second_upper = descendants(second)['Upper']
    first_upper.name = 'Artist Renamed Upper'
    first_upper.transform.local_position = Vector3(7, 8, 9)
    second_upper.transform.local_scale = Vector3(2, 3, 4)

    document = json.loads(source.read_text())
    document['nodes'][0]['children'].append(len(document['nodes']))
    document['nodes'].append({
        'name': 'Added', 'translation': [0, 2, 0], 'mesh': 0,
    })
    source.write_text(json.dumps(document))
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert 'Added' in descendants(first)
    assert 'Added' in descendants(second)
    assert first.transform.local_position.x == 10
    assert second.transform.local_position.x == -10
    assert tuple(first_upper.transform.local_position) == (7, 8, 9)
    assert 'Artist Renamed Upper' in descendants(first)
    assert tuple(second_upper.transform.local_scale) == (2, 3, 4)
    document['nodes'][0]['children'].remove(document['nodes'][0]['children'][-1])
    document['nodes'].pop()
    source.write_text(json.dumps(document))
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert 'Added' not in descendants(first)
    assert 'Added' not in descendants(second)
    assert len(source_warnings) == 2
    assert all("source path 'Assembly/Added'" in message and guid in message for message in source_warnings)
    assert tuple(first_upper.transform.local_position) == (7, 8, 9)
    assert tuple(second_upper.transform.local_scale) == (2, 3, 4)


def test_removed_source_node_retains_author_owned_components(scene, hierarchy_asset, engine, monkeypatch):
    """Removing imported geometry must not delete an authored component host."""
    from Infernux.core.assets import AssetManager
    from Infernux.debug import Debug

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    warnings = []
    monkeypatch.setattr(Debug, 'log_warning', lambda message, context=None: warnings.append(str(message)))
    root = scene.create_from_model(guid, 'Author-owned Assembly')
    target = descendants(root)['Upper']
    authored_collider = target.add_component('BoxCollider')

    document = json.loads(source.read_text())
    document['nodes'][0]['children'].append(len(document['nodes']))
    document['nodes'].append({'name': 'Retained', 'translation': [0, 2, 0], 'mesh': 0})
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    retained = descendants(root)['Retained']
    retained.add_component('BoxCollider')

    document['nodes'][0]['children'].remove(document['nodes'][0]['children'][-1])
    document['nodes'].pop()
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    retained = descendants(root)['Retained']
    assert retained.get_component('MeshRenderer') is None
    assert retained.get_component('BoxCollider') is not None
    assert retained.serialize_document().get('model_source') in (None, {'guid': '', 'path': []})
    assert any("Retained" in message and guid in message for message in warnings)
    assert authored_collider is not None


def test_external_source_node_motion_updates_new_instances_only(scene, hierarchy_asset, engine, monkeypatch):
    """DCC node motion is source metadata; existing scene placement remains authored."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    existing = scene.create_from_model(guid, 'Authored Assembly')
    authored_upper = descendants(existing)['Upper']
    authored_upper.transform.local_position = Vector3(21, 22, 23)

    document = json.loads(source.read_text())
    document['nodes'][2]['translation'] = [9, 8, 7]
    source.write_text(json.dumps(document))
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error

    # Existing scene edits are authoritative and are never replaced by source TRS.
    assert tuple(authored_upper.transform.local_position) == (21, 22, 23)
    # A later instance receives the new source transform, proving the reimport
    # updated the model source rather than merely suppressing the change.
    fresh = scene.create_from_model(guid, 'Fresh Assembly')
    fresh_upper = descendants(fresh)['Upper']
    assert tuple(round(float(v), 5) for v in fresh_upper.transform.local_position) == (9, 8, 7)


def test_source_reorder_keeps_node_binding_and_source_rename_reconciles_instance(scene, hierarchy_asset, engine, monkeypatch):
    from Infernux.core.assets import AssetManager
    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    root = scene.create_from_model(guid)
    upper = descendants(root)['Upper'].get_component('MeshRenderer')._require_cpp_component()
    path = upper.serialize_document()['modelNodePath']
    before = upper.serialize_document()['nodeGroup']
    document = json.loads(source.read_text())
    document['nodes'][1]['children'].reverse()
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    assert upper.serialize_document()['nodeGroup'] != before
    assert upper.serialize_document()['modelNodePath'] == path
    document['nodes'][2]['name'] = 'Renamed Upper'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    objects = descendants(root)
    assert 'Upper' not in objects
    assert 'Renamed Upper' in objects
    renamed = objects['Renamed Upper'].get_component('MeshRenderer')
    assert renamed.model_node_path[-1] == 'Renamed Upper'
    document['nodes'][2]['name'] = 'Upper'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    assert 'Upper' in descendants(root)


def test_source_parent_rename_keeps_authored_instance_pose(scene, hierarchy_asset, engine, monkeypatch):
    """A DCC pivot rename updates the binding, not the author's placement."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    # A parent rename with multiple identical meshes is ambiguous. Use one
    # mesh here, rather than accidentally matching by merged-buffer offsets.
    document = json.loads(source.read_text())
    document['nodes'][1]['children'] = [2]
    document['nodes'].pop()
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    root = scene.create_from_model(guid)
    upper = descendants(root)['Upper']
    authored_position = Vector3(21, 22, 23)
    upper.transform.local_position = authored_position
    upper_object_id = upper.id

    document = json.loads(source.read_text())
    document['nodes'][0]['name'] = 'Renamed Assembly'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)

    objects = descendants(root)
    assert objects['Renamed Assembly'].get_parent() is root
    renamed = objects['Renamed Assembly']
    renamed_upper = next(child for child in renamed.get_children() if child.name == 'Empty pivot')
    renamed_upper = next(child for child in renamed_upper.get_children() if child.name == 'Upper')
    assert renamed_upper.id == upper_object_id
    assert tuple(renamed_upper.transform.local_position) == tuple(authored_position)
    assert renamed_upper.get_component('MeshRenderer').model_node_path == [
        'Renamed Assembly', 'Empty pivot', 'Upper'
    ]


def test_source_cross_parent_move_rehomes_imported_edge_but_keeps_authored_pose(
    scene, hierarchy_asset, engine, monkeypatch
):
    """A DCC parent change updates imported structure, not scene-authored TRS."""
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    root = scene.create_from_model(guid)
    upper = descendants(root)['Upper']
    upper_id = upper.id
    upper.transform.local_position = Vector3(21, 22, 23)

    document = json.loads(source.read_text())
    # First make the identical geometry unique so a later cross-parent move
    # has a deterministic stable subresource match rather than guessing.
    document['nodes'][1]['children'] = [2]
    document['nodes'].pop(3)
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)

    document['nodes'][0]['children'] = [1, 2]
    document['nodes'][1]['children'] = []
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)

    objects = descendants(root)
    assert objects['Upper'].id == upper_id
    assert objects['Upper'].get_parent().name == 'Assembly'
    assert tuple(objects['Upper'].transform.local_position) == (21, 22, 23)
    assert objects['Upper'].get_component('MeshRenderer').model_node_path == ['Assembly', 'Upper']


def test_source_rename_does_not_reset_unrelated_authored_parent(scene, hierarchy_asset, engine, monkeypatch):
    from Infernux.core.assets import AssetManager

    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    root = scene.create_from_model(guid)
    objects = descendants(root)
    upper, lower = objects['Upper'], objects['Lower']
    lower.set_parent(objects['Assembly'], world_position_stays=True)
    before = world_matrix(lower).copy()
    parent_id = lower.get_parent().id
    upper_id = upper.id
    document = json.loads(source.read_text())
    document['nodes'][2]['name'] = 'Renamed Upper'
    source.write_text(json.dumps(document))
    assert AssetManager.reimport_asset(str(source), database=database)
    assert descendants(root)['Renamed Upper'].id == upper_id
    assert lower.get_parent().id == parent_id
    np.testing.assert_allclose(world_matrix(lower), before, atol=2e-5)


def test_ambiguous_source_node_paths_fail_before_creating_objects(scene, hierarchy_asset):
    database, source, _ = hierarchy_asset
    document = json.loads(source.read_text())
    document['nodes'][3]['name'] = 'Upper'
    source.write_text(json.dumps(document))
    result = database.import_asset(str(source))
    assert result, result.error
    before = scene.serialize_document()
    with pytest.raises(ValueError, match='ambiguous'):
        scene.create_from_model(result.guid)
    assert scene.serialize_document() == before
