"""Prefab authoring accepts the current model source identity contract."""
import copy
import json

import pytest

from infernux.engine.prefab_manager import instantiate_prefab, save_prefab, save_prefab_document
from test_model_hierarchy_instances import descendants, hierarchy_asset


def test_model_prefab_retains_root_and_child_source_identities(scene, hierarchy_asset):
    database, source, guid = hierarchy_asset
    root = scene.create_from_model(guid, 'Model')
    path = source.parent / 'Model.prefab'
    assert save_prefab(root, str(path))
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['root_object']['model_source'] == {'guid': guid, 'path': []}
    imported = database.import_asset(str(path))
    assert imported, imported.error
    instance = instantiate_prefab(file_path=str(path), guid=imported.guid, scene=scene)
    assert instance is not None
    assert instance._model_source_guid == guid and instance._model_source_path == []
    original_nodes, new_nodes = descendants(root), descendants(instance)
    assert new_nodes.keys() == original_nodes.keys()
    for name, obj in new_nodes.items():
        assert obj._model_source_guid == guid
        assert obj._model_source_path == original_nodes[name]._model_source_path
    assert instance.prefab_guid == imported.guid


@pytest.mark.parametrize('source', [None, {}, {'guid': '', 'path': []}, {'guid': 1, 'path': []},
                                   {'guid': 'model', 'path': 'Node'}, {'guid': 'model', 'path': ['']},
                                   {'guid': 'model', 'path': [1]}, {'guid': 'model', 'path': [], 'extra': 1}])
def test_invalid_model_binding_cannot_overwrite_prefab(scene, tmp_path, source):
    root = scene.create_game_object('Keep')
    path = tmp_path / 'Keep.prefab'
    assert save_prefab(root, str(path))
    before = path.read_bytes()
    invalid = copy.deepcopy(json.loads(before))
    invalid['root_object']['model_source'] = source
    assert not save_prefab_document(invalid, str(path))
    assert path.read_bytes() == before
