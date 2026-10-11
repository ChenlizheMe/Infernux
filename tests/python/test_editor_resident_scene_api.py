"""Editor Scene APIs must finish authoring work while native owners are live."""
import pytest

from infernux.engine.interaction import DocumentRegistry
from infernux.engine.scene_manager import SceneFileManager
from infernux.lib import SceneManager as NativeSceneManager
from infernux.scene import SceneManager


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("playing", [False, True])
def test_public_unload_retires_editor_document_before_native_scene(
    scene, monkeypatch, registered, active, playing,
):
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(SceneManager, "_runtime_scene_service", None)
    files = SceneFileManager()
    native = NativeSceneManager.instance()
    remaining_world = scene.world_id
    expected_active = native.get_scene_at(0) if active else scene
    departing = native.create_scene("PublicUnloadTarget")
    retiring_world = departing.world_id
    departing.create_game_object("RetirementSnapshotWitness")
    if registered:
        document_id = files.register_loaded_scene(departing, "", dirty=True)
        locator = DocumentRegistry.instance().locate(document_id)
        expected_document = departing.serialize_document()
    if active:
        SceneManager.set_active_scene(departing)
    if playing:
        native.play()

    SceneManager.unload_scene(departing)

    assert native.get_scene_by_world_id(retiring_world) is None
    assert native.get_scene_by_world_id(remaining_world) is scene
    assert native.get_active_scene() is expected_active
    assert retiring_world not in files._loaded_scene_documents
    if registered:
        snapshot = files._scene_restore_snapshots[locator.stable_id]
        assert snapshot.document["objects"] == expected_document["objects"]
        assert snapshot.document["name"] == "PublicUnloadTarget"
        assert snapshot.revision == 1
        assert snapshot.saved_revision == 0
        assert DocumentRegistry.instance().get(document_id) is None
