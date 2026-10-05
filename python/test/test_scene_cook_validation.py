"""Scene cook and asynchronous Player reads share structural acceptance rules."""
import json

import pytest

from infernux.engine.game_builder import GameBuilder
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
from infernux.engine.scene_authoring import (
    decode_scene_document,
    encode_runtime_scene_artifact,
)
from infernux.lib import SceneManager


def _scene_document(scene, defect):
    owner = scene.create_game_object("CookWitness")
    owner.add_component("Camera")
    document = scene.serialize_asset_document()
    camera = document["objects"][0]["components"][0]
    if defect == "dangling_camera":
        document["mainCameraComponentId"] = "123456789abcdef0123456789abcdef0"
    elif defect == "wrong_type_camera":
        document["mainCameraComponentId"] = document["objects"][0]["transform"]["component_id"]
    elif defect == "invalid_camera":
        camera["data"]["fov"] = "wide"
    elif defect == "valid_camera":
        document["mainCameraComponentId"] = camera["component_id"]
    elif defect == "no_camera":
        document["objects"] = []
        document.pop("mainCameraComponentId", None)
    else:
        raise AssertionError(defect)
    return document


@pytest.mark.parametrize("defect", ["dangling_camera", "wrong_type_camera", "invalid_camera"])
def test_scene_artifact_rejects_documents_the_player_reader_rejects(scene, defect):
    document = decode_scene_document(_scene_document(scene, defect))
    before = scene.serialize_document()
    with pytest.raises(ValueError):
        encode_runtime_scene_artifact(document)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("defect", ["dangling_camera", "wrong_type_camera", "invalid_camera"])
def test_cook_rejects_invalid_unopened_scene_before_publishing_artifact(scene, tmp_path, defect):
    authored = _scene_document(scene, defect)
    assets = tmp_path / "Assets"
    assets.mkdir()
    path = assets / "Unopened.scene"
    source = json.dumps(authored).encode("utf-8")
    path.write_bytes(source)
    before = scene.serialize_document()
    builder = GameBuilder.__new__(GameBuilder)
    builder.project_path = str(tmp_path)
    builder._cooked_asset_entries = {"scene-guid": {"normalized_path": str(path)}}
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data = tmp_path / "Data"
    with pytest.raises(ValueError):
        builder._stage_library_runtime_documents(str(data))
    assert not (data / "Library/Artifacts/Document/scene-guid.scene").exists()
    assert not builder._runtime_artifact_bindings
    assert path.read_bytes() == source
    assert scene.serialize_document() == before


@pytest.mark.parametrize("kind", ["valid_camera", "no_camera"])
def test_valid_cooked_scene_loads_with_identical_contract(scene, tmp_path, kind):
    document = decode_scene_document(_scene_document(scene, kind))
    cooked = encode_runtime_scene_artifact(document)
    path = tmp_path / "Cooked.scene"
    path.write_text(json.dumps(cooked), encoding="utf-8")
    target = SceneManager.instance().create_scene("CookedWitness")
    transaction = SceneDocumentTransaction(target, path=str(path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    assert (target.main_camera is not None) is (kind == "valid_camera")
