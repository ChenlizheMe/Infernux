from __future__ import annotations

from types import SimpleNamespace

import pytest

from Infernux.engine.scene_manager import SceneFileManager, _LoadedSceneDocument
from Infernux.engine.undo import AdditiveSceneResidencyCommand


class _SceneAssetDatabase:
    def __init__(self, guid: str, current_path: str, *accepted_paths: str):
        self.guid = guid
        self.current_path = current_path
        self.accepted_paths = {str(path) for path in accepted_paths}
        self.accepted_paths.add(current_path)

    def get_guid_from_path(self, path):
        return self.guid if str(path) in self.accepted_paths else ""

    def get_path_from_guid(self, guid):
        return self.current_path if str(guid) == self.guid else ""


def test_single_replace_retires_discarded_session_document_views(tmp_path):
    from Infernux.engine.interaction import DocumentRegistry

    class Database:
        def __init__(self):
            self.paths_by_guid = {}
            self.guids_by_path = {}

        def register(self, path, guid):
            self.paths_by_guid[str(guid)] = str(path)
            self.guids_by_path[str(path)] = str(guid)

        def get_guid_from_path(self, path):
            return self.guids_by_path.get(str(path), "")

        def get_path_from_guid(self, guid):
            return self.paths_by_guid.get(str(guid), "")

    previous_registry = DocumentRegistry._instance
    previous_manager = SceneFileManager._instance
    try:
        registry = DocumentRegistry()
        manager = SceneFileManager()
        database = Database()
        initial_path = str(tmp_path / "Initial.scene")
        replacement_path = str(tmp_path / "Replacement.scene")
        database.register(initial_path, "initial-scene-guid")
        database.register(replacement_path, "replacement-scene-guid")
        manager.set_asset_database(database)

        initial = manager._replace_scene_document(
            kind="scene",
            resource_path=initial_path,
            title="Untitled Scene",
            dirty=True,
        )
        for view_id in ("scene_view", "game_view", "ui_editor"):
            registry.attach_view(initial.document_id, view_id)

        replacement = manager._replace_scene_document(
            kind="scene",
            resource_path=replacement_path,
            title="Replacement",
            dirty=False,
        )

        assert registry.get(initial.document_id) is None
        assert all(
            registry.document_for_view(view_id) is replacement
            for view_id in ("scene_view", "game_view", "ui_editor")
        )
        assert registry.dirty_documents() == ()
    finally:
        SceneFileManager._instance = previous_manager
        DocumentRegistry._instance = previous_registry


def test_additive_path_entry_resolves_guid_before_residency_dedup(tmp_path, monkeypatch):
    assets = tmp_path / "Project" / "Assets"
    assets.mkdir(parents=True)
    old_path = assets / "Old.scene"
    current_path = assets / "Moved.scene"
    current_path.write_text("{}", encoding="utf-8")

    manager = SceneFileManager()
    manager.set_asset_database(
        _SceneAssetDatabase(
            "scene-guid",
            str(current_path.resolve()),
            str(old_path.resolve()),
        )
    )
    resident = SimpleNamespace(world_id=71)
    manager._loaded_scene_documents[71] = _LoadedSceneDocument(
        scene=resident,
        asset_guid="scene-guid",
        resource_path=str(current_path.resolve()),
        document_id="scene-document",
    )
    activated = []
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    monkeypatch.setattr(manager, "_is_under_assets", lambda _path: True)
    monkeypatch.setattr(
        manager,
        "activate_loaded_scene",
        lambda scene: activated.append(scene) or True,
    )

    assert manager.open_scene_additive(str(old_path)) is True
    assert activated == [resident]
    assert manager._deferred_additive_guid is None


def test_path_unload_entry_resolves_to_resident_guid(tmp_path, monkeypatch):
    assets = tmp_path / "Project" / "Assets"
    assets.mkdir(parents=True)
    old_path = assets / "Old.scene"
    current_path = assets / "Moved.scene"
    current_path.write_text("{}", encoding="utf-8")

    manager = SceneFileManager()
    manager.set_asset_database(
        _SceneAssetDatabase(
            "scene-guid",
            str(current_path.resolve()),
            str(old_path.resolve()),
        )
    )
    resident = SimpleNamespace(world_id=72)
    manager._loaded_scene_documents[72] = _LoadedSceneDocument(
        scene=resident,
        asset_guid="scene-guid",
        resource_path=str(current_path.resolve()),
        document_id="scene-document",
    )
    requested = []
    monkeypatch.setattr(
        manager,
        "request_unload_scene",
        lambda scene: requested.append(scene) or True,
    )

    assert manager.request_unload_scene_path(str(old_path)) is True
    assert requested == [resident]


def test_additive_queue_retains_guid_instead_of_input_path(tmp_path, monkeypatch):
    assets = tmp_path / "Project" / "Assets"
    assets.mkdir(parents=True)
    scene_path = assets / "Queued.scene"
    scene_path.write_text("{}", encoding="utf-8")

    manager = SceneFileManager()
    manager.set_asset_database(
        _SceneAssetDatabase("queued-guid", str(scene_path.resolve()))
    )
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    monkeypatch.setattr(manager, "_is_under_assets", lambda _path: True)

    assert manager.open_scene_additive(str(scene_path), record_history=False) is True
    assert manager._deferred_additive_guid == "queued-guid"
    assert not hasattr(manager, "_deferred_additive_path")


def test_scene_move_updates_io_path_without_changing_residency_identity(tmp_path):
    destination = str((tmp_path / "Moved.scene").resolve())
    manager = SceneFileManager()
    resident = SimpleNamespace(world_id=73)
    manager._loaded_scene_documents[73] = _LoadedSceneDocument(
        scene=resident,
        asset_guid="scene-guid",
        resource_path=str((tmp_path / "Old.scene").resolve()),
        document_id="scene-document",
    )

    manager.resource_moved(
        document_id="scene-document",
        source_path=str(tmp_path / "Old.scene"),
        destination_path=destination,
        guid="scene-guid",
    )

    binding = manager._loaded_scene_documents[73]
    assert binding.asset_guid == "scene-guid"
    assert binding.resource_path == destination


def test_additive_residency_history_replays_guid_not_path():
    calls = []

    class _SceneFiles:
        def open_scene_additive_guid(self, guid, *, record_history):
            calls.append(("open", guid, record_history))
            return True

        def request_unload_scene_guid(self, guid):
            calls.append(("unload", guid))
            return True

    command = AdditiveSceneResidencyCommand(
        _SceneFiles(),
        "scene-guid",
        "Open Scene Additively",
    )

    command.execute()
    command.undo()
    command.redo()

    assert calls == [
        ("open", "scene-guid", False),
        ("unload", "scene-guid"),
        ("open", "scene-guid", False),
    ]


@pytest.mark.parametrize("native_active_moved_early", [False, True])
def test_additive_undo_rebinds_active_document_without_placeholder(
    scene, tmp_path, monkeypatch, native_active_moved_early
):
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.lib import SceneManager

    native = SceneManager.instance()
    primary_path = tmp_path / "Primary.scene"
    additive_path = tmp_path / "Additive.scene"
    primary_path.write_text("{}", encoding="utf-8")
    additive_path.write_text("{}", encoding="utf-8")
    database = _SceneAssetDatabase(
        "primary-guid", str(primary_path.resolve())
    )
    original_lookup = database.get_guid_from_path
    original_resolve = database.get_path_from_guid
    database.get_guid_from_path = lambda path: (
        "additive-guid" if str(path) == str(additive_path.resolve())
        else original_lookup(path)
    )
    database.get_path_from_guid = lambda guid: (
        str(additive_path.resolve()) if guid == "additive-guid"
        else original_resolve(guid)
    )
    manager = SceneFileManager()
    manager.set_asset_database(database)
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    primary_id = manager._replace_scene_document(
        kind="scene", resource_path=str(primary_path), title="Primary", dirty=False
    ).document_id
    additive = native.create_scene("Additive")
    additive_world_id = additive.world_id
    additive_id = manager.register_loaded_scene(additive, str(additive_path))
    assert manager.activate_loaded_scene(additive)
    registry = DocumentRegistry.instance()
    registry.attach_view(additive_id, "scene_view")
    observed = []

    def on_scene_changed():
        registry.detach_view("scene_view")
        registry.attach_view(manager.document_id, "scene_view")
        observed.append(registry.require(manager.document_id).document_id)

    manager.set_on_scene_changed(on_scene_changed)
    worlds_before = {native.get_scene_at(i).world_id for i in range(native.scene_count)}
    if native_active_moved_early:
        native.set_active_scene(scene)

    AdditiveSceneResidencyCommand(
        manager, "additive-guid", "Open Scene Additively"
    ).undo()
    manager.poll_deferred_load()

    assert native.get_active_scene().world_id == scene.world_id
    assert manager.document_id == primary_id
    assert observed[-1] == primary_id
    assert additive_world_id not in {
        native.get_scene_at(i).world_id for i in range(native.scene_count)
    }
    assert {native.get_scene_at(i).world_id for i in range(native.scene_count)} == (
        worlds_before - {additive_world_id}
    )
    assert registry.get(additive_id) is None


def test_play_stop_restores_additive_scene_and_editor_document(scene, tmp_path):
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.engine.play_mode import PlayModeManager
    from Infernux.lib import SceneManager

    native = SceneManager.instance()
    primary = tmp_path / "Primary.scene"
    additive_path = tmp_path / "Additive.scene"
    primary.write_text("{}", encoding="utf-8")
    additive_path.write_text("{}", encoding="utf-8")

    class Database:
        def get_guid_from_path(self, path):
            return {
                str(primary.resolve()): "primary-guid",
                str(additive_path.resolve()): "additive-guid",
            }.get(str(path), "")

        def get_path_from_guid(self, guid):
            return {
                "primary-guid": str(primary.resolve()),
                "additive-guid": str(additive_path.resolve()),
            }.get(str(guid), "")

    database = Database()
    files = SceneFileManager()
    files.set_asset_database(database)
    primary_id = files._replace_scene_document(
        kind="scene", resource_path=str(primary), title="Primary", dirty=False
    ).document_id
    additive = native.create_scene("Additive")
    additive_id = files.register_loaded_scene(additive, str(additive_path))
    files.activate_loaded_scene(additive)
    play = PlayModeManager()
    play.set_asset_database(database)
    play._save_scene_state()

    native.unload_scene(additive)
    runtime = native.create_scene("Runtime")
    native.set_active_scene(runtime)
    runtime_world_id = int(runtime.world_id)
    assert play._restore_loaded_scenes_after_play()

    restored = files.scene_for_document(additive_id)
    assert restored is not None
    # The runtime Scene is destroyed by restoration.  Retain its scalar ID
    # before the transition rather than dereferencing a retired pybind wrapper.
    assert restored.world_id != runtime_world_id
    assert files.document_id == additive_id
    assert native.get_active_scene().world_id == restored.world_id
    assert files.scene_for_document(primary_id).world_id == scene.world_id
    assert DocumentRegistry.instance().require(additive_id).key.identity == "additive-guid"
    assert files._binding_for_document(additive_id).asset_guid == "additive-guid"
