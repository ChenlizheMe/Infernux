"""Scene file boundaries keep GUIDs stable and reject implicit numeric formats."""
import json

import pytest

from infernux.lib import SceneManager
from infernux.engine.scene_authoring import decode_scene_document, encode_scene_document, encode_runtime_scene_artifact
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction


def test_scene_asset_roundtrip_is_independent_of_runtime_ids(scene, tmp_path):
    original = scene.create_game_object('Source')
    original.add_component('Light')
    scene_path = tmp_path / 'Source.scene'
    assert scene.save_to_file(str(scene_path))
    before = scene_path.read_bytes()
    document = json.loads(before)
    assert document['identity_format'] == 'guid-v1'
    assert not {'nextObjectId', 'nextComponentId', 'authoring_identity'}.intersection(document)
    target = SceneManager.instance().create_scene('Loaded')
    transaction = SceneDocumentTransaction(target, path=str(scene_path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    assert target.find('Source').id != original.id
    assert target.serialize_asset_document() == document
    assert scene_path.read_bytes() == before
    assert target.save_to_file(str(scene_path))
    assert scene_path.read_bytes() == before


def test_guid_file_does_not_persist_unreferenced_tombstones(scene):
    removed = scene.create_game_object('Removed')
    live = scene.create_game_object('Live')
    snapshot = scene.serialize_document()
    removed_guid = snapshot['authoring_identity']['objects'][str(removed.id)]
    live_guid = snapshot['authoring_identity']['objects'][str(live.id)]
    scene.destroy_game_object(removed)
    scene.process_pending_destroys()
    asset = scene.serialize_asset_document()
    assert removed_guid not in json.dumps(asset)
    assert live_guid == asset['objects'][0]['id']


@pytest.mark.parametrize('kind', ['unmarked_numeric', 'unknown', 'guid_with_runtime_table'])
def test_bad_file_format_rejects_without_replacing_live_world(scene, tmp_path, kind):
    owner = scene.create_game_object('KeepLive')
    before = scene.serialize_document()
    if kind == 'unmarked_numeric':
        document = before
    else:
        document = scene.serialize_asset_document()
        if kind == 'unknown':
            document['identity_format'] = 'not-a-schema'
        else:
            document['authoring_identity'] = before['authoring_identity']
    path = tmp_path / 'Invalid.scene'
    path.write_text(json.dumps(document), encoding='utf-8')
    transaction = SceneDocumentTransaction(scene, path=str(path))
    assert not transaction.run_to_completion(raise_on_failure=False)
    assert scene.find('KeepLive') is owner
    assert scene.serialize_document() == before


def test_cooked_scene_contract_has_deterministic_bytes_and_loads(scene, tmp_path):
    source = scene.create_game_object('PlayerObject')
    source.add_component('Light')
    asset = scene.serialize_asset_document()
    first = encode_runtime_scene_artifact(decode_scene_document(asset))
    second = encode_runtime_scene_artifact(decode_scene_document(asset))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first['identity_format'] == 'runtime-v1'
    assert not {'authoring_identity', 'nextObjectId', 'nextComponentId'}.intersection(first)
    target = SceneManager.instance().create_scene('PlayerWorld')
    path = tmp_path / 'Cooked.scene'
    path.write_text(json.dumps(first), encoding='utf-8')
    transaction = SceneDocumentTransaction(target, path=str(path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    assert target.find('PlayerObject').get_component('Light') is not None
    assert target.find('PlayerObject').id != source.id


def test_suspended_snapshot_encoding_retains_identity(scene):
    scene.create_game_object('Suspended').add_component('Light')
    snapshot = scene.serialize_document()
    asset = encode_scene_document(snapshot)
    assert asset == scene.serialize_asset_document()
    assert encode_scene_document(decode_scene_document(asset)) == asset
