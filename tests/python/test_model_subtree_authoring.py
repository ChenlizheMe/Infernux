"""DCC subtree removal must preserve author-owned hosts and their transform chain."""
import json

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
from infernux.engine.undo import SetPropertyCommand, UndoManager
from infernux.lib import Vector3
from test_model_hierarchy_instances import hierarchy_asset, descendants, world_matrix


@pytest.fixture
def model_history():
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        yield manager
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.mark.parametrize("route", ["live", "cold_document"])
@pytest.mark.parametrize("prefab", [False, True])
@pytest.mark.parametrize("authored_on", ["none", "parent", "descendant", "authored_child", "python_descendant"])
def test_removed_source_subtree_preserves_host_and_ancestors(scene, hierarchy_asset, engine, monkeypatch,
                                                           tmp_path, model_history, route, prefab, authored_on):
    database, source, guid = hierarchy_asset
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    root = scene.create_from_model(guid, 'Authored Assembly')
    prefab_guid = None
    if prefab:
        from infernux.engine.prefab_manager import instantiate_prefab, save_prefab
        prefab_path = source.parent / 'Assembly.prefab'
        assert save_prefab(root, str(prefab_path))
        imported = database.import_asset(str(prefab_path))
        assert imported, imported.error
        prefab_guid = imported.guid
        scene.destroy_game_object(root)
        scene.process_pending_destroys()
        root = instantiate_prefab(file_path=str(prefab_path), guid=prefab_guid, scene=scene)
        assert root is not None
        root.name = 'Authored Assembly'
    objects = descendants(root)
    pivot, upper = (objects[name] for name in ('Empty pivot', 'Upper'))
    target = pivot if authored_on == 'parent' else upper
    if authored_on == 'authored_child':
        target = scene.create_game_object('Handmade puzzle trigger')
        target.set_parent(upper, world_position_stays=False)
    component = None
    python_component = authored_on == 'python_descendant'
    if authored_on != 'none':
        if python_component:
            from infernux.components.component_identity import bind_asset_script_guid
            from infernux.components.script_loader import load_component_class_from_file
            script = source.parent / 'PuzzleMarker.py'
            script.write_text('from infernux import InxComponent, serialized_field\n'
                              'class PuzzleMarker(InxComponent):\n'
                              '    progress: float = serialized_field(default=1.0)\n', encoding='utf-8')
            imported = database.import_asset(str(script))
            assert imported, imported.error
            cls = load_component_class_from_file(str(script), 'PuzzleMarker')
            bind_asset_script_guid(cls, imported.guid)
            component = target.add_py_component(cls())
            property_name, old_value, new_value = 'progress', 1., 7.5
        else:
            component = target.add_component('BoxCollider')
            property_name, old_value, new_value = 'size', Vector3(*component.size), Vector3(2, 3, 4)
        assert model_history.execute(SetPropertyCommand(component, property_name, old_value, new_value), raise_errors=True)
        target.name = 'Authored trigger'
        target.transform.local_position = Vector3(7, 8, 9)
        target.transform.local_scale = Vector3(2, 3, 4)
    target_id = target.id
    component_id = component.component_id if component else None
    transform_id = target.transform.component_id
    original_matrix = world_matrix(target).copy()
    # Capturing before reimport models a collaborator opening an old Scene
    # against the newly pulled model, not a post-repair serialization only.
    cold_document = scene.serialize_document()
    target_guid = cold_document['authoring_identity']['objects'][str(target_id)]
    component_guid = (cold_document['authoring_identity']['components'][str(component_id)]
                      if component else None)
    source_document = json.loads(source.read_text(encoding='utf-8'))
    source_document['nodes'][0]['children'] = [len(source_document['nodes'])]
    source_document['nodes'].append({'name': 'Surviving geometry', 'mesh': 0})
    source.write_text(json.dumps(source_document), encoding='utf-8')
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    if route == 'cold_document':
        assert SceneDocumentTransaction(scene, document=cold_document, asset_database=database).run_to_completion()

    def verify(same_runtime_ids=True):
        restored_root = scene.find('Authored Assembly')
        if prefab:
            assert restored_root.prefab_guid == prefab_guid
            assert restored_root.is_prefab_instance
        host = scene.find('Upper' if authored_on == 'none' else 'Authored trigger')
        if authored_on == 'none':
            assert host is None
            assert scene.find('Empty pivot') is None
        else:
            assert host is not None, 'Reimport destroyed an authored component host'
            identities = scene.serialize_document()['authoring_identity']
            assert identities['objects'][str(host.id)] == target_guid
            assert host.name == 'Authored trigger'
            retained = host.get_py_components()[0] if python_component else host.get_component('BoxCollider')
            assert identities['components'][str(retained.component_id)] == component_guid
            if same_runtime_ids:
                assert host.id == target_id
                assert host.transform.component_id == transform_id
                assert retained.component_id == component_id
            if python_component:
                assert retained.progress == 7.5
            else:
                assert tuple(retained.size) == (2, 3, 4)
            np.testing.assert_allclose(world_matrix(host), original_matrix, atol=1e-5)
            assert scene.find('Authored trigger' if authored_on == 'parent' else 'Empty pivot') is not None
            for name in ('Empty pivot', 'Upper', 'Authored trigger'):
                survivor = scene.find(name)
                if survivor:
                    assert not survivor._model_source_guid
                    assert survivor.get_component('MeshRenderer') is None
                    assert survivor.get_component('MeshCollider') is None
        assert scene.find('Lower') is None, 'Unowned generated sibling was retained'
        assert scene.find('Surviving geometry').get_component('MeshRenderer') is not None

    verify()
    if component is not None:
        model_history.undo()
        host = scene.find('Authored trigger')
        retained = host.get_py_components()[0] if python_component else host.get_component('BoxCollider')
        value = getattr(retained, property_name)
        assert (value == old_value if python_component else tuple(value) == tuple(old_value))
        model_history.redo()
        verify()
    # Serialize to actual local bytes and reopen through the normal Scene
    # transaction. This verifies persistence, not the Editor Save UI command.
    saved_path = tmp_path / 'reconciled.scene'
    assert scene.save_to_file(str(saved_path))
    saved = json.loads(saved_path.read_text(encoding='utf-8'))
    assert saved['identity_format'] == 'guid-v1'
    assert SceneDocumentTransaction(scene, path=str(saved_path), asset_database=database).run_to_completion()
    verify(same_runtime_ids=False)
    assert scene.serialize_asset_document() == saved
