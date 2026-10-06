"""Stop must publish Edit only after the authored Scene snapshot is restored."""
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.engine import project_context
from infernux.engine.deferred_task import DeferredTaskRunner
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.play_mode import PlayModeManager, PlayModeState
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager
from infernux.renderstack.render_effect import RenderEffect


@pytest.fixture
def playing_author(engine, scene, tmp_path, monkeypatch):
    from infernux.components import script_loader

    # This fixture composes a fresh editor project on the shared native Engine;
    # script failures belonging to another test project must not leak into it.
    monkeypatch.setattr(script_loader, "_script_errors", {})
    monkeypatch.setattr(script_loader, "_script_error_revision", 0)
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(UndoManager, "_instance", None)
    runner = DeferredTaskRunner()
    monkeypatch.setattr(DeferredTaskRunner, "_instance", runner)
    previous_root = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    files = SceneFileManager()
    files.set_asset_database(database)
    core.project_assets.configure(database.project_root, database)
    obj = scene.create_game_object("AuthorDoor")
    object_id = obj.id
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    path = folder / "Author.scene"
    path.write_text(scene.serialize_asset(), encoding="utf-8")
    assert database.import_asset(str(path)).succeeded
    files.register_loaded_scene(scene, str(path))
    assert files.activate_loaded_scene(scene)
    manager = PlayModeManager()
    manager.set_asset_database(database)
    manager._native_engine = engine
    native = SceneManager.instance()
    try:
        assert manager.enter_play_mode()
        runner.tick()
        assert manager.state is PlayModeState.PLAYING and native.is_playing()
        scene.find_by_id(object_id).name = "RuntimeDoor"
        yield SimpleNamespace(manager=manager, runner=runner, files=files, scene=scene,
                              native=native, object_id=object_id, path=path,
                              original_bytes=path.read_bytes())
    finally:
        runner.cancel()
        if native.is_playing():
            native.stop()
        AssetManager._end_play_data_asset_isolation()
        Material._suppress_auto_save = False
        RenderEffect._suppress_auto_save = False
        engine.set_play_mode_rendering(False)
        core.shutdown()
        project_context.set_project_root(previous_root)


@pytest.mark.parametrize("paused", [False, True])
def test_stop_keeps_authoring_locked_until_restore_finishes(playing_author, paused):
    state = playing_author
    if paused:
        assert state.manager.pause()
        assert state.manager.step_frame()
        assert state.manager.pending_step_requests == 1
    completed = []
    assert state.manager.exit_play_mode(on_complete=completed.append)
    assert not state.native.is_playing()
    assert not state.manager.is_edit_mode
    assert completed == []
    assert state.manager.pending_step_requests == 0
    assert Material._suppress_auto_save and RenderEffect._suppress_auto_save
    assert not state.files.save_current_scene()
    assert state.path.read_bytes() == state.original_bytes
    state.runner.tick()
    assert completed == [True]
    assert state.manager.is_edit_mode
    assert state.scene.find_by_id(state.object_id).name == "AuthorDoor"
    assert not Material._suppress_auto_save and not RenderEffect._suppress_auto_save


@pytest.mark.parametrize("failure", ["rejected", "raised", "notification"])
def test_failed_stop_keeps_snapshot_and_requires_explicit_recovery(playing_author, monkeypatch, failure):
    state = playing_author
    backups = state.manager._scene_backups
    rebuild = state.manager._rebuild_scene
    notify = state.manager._notify_state_change

    def reject(scene, *args, **kwargs):
        if scene is state.scene:
            if failure == "raised":
                raise RuntimeError("injected scene restoration failure")
            return False
        return rebuild(scene, *args, **kwargs)

    def reject_notification(old, new):
        if new is PlayModeState.EDIT:
            raise RuntimeError("injected restored-state notification failure")
        return notify(old, new)

    if failure == "notification":
        monkeypatch.setattr(state.manager, "_notify_state_change", reject_notification)
    else:
        monkeypatch.setattr(state.manager, "_rebuild_scene", reject)
    completed = []
    assert state.manager.exit_play_mode(on_complete=completed.append)
    state.runner.tick()
    assert completed == [False]
    assert not state.manager.is_edit_mode and not state.native.is_playing()
    assert not state.runner.is_busy
    assert state.manager._scene_backups is backups
    assert Material._suppress_auto_save and RenderEffect._suppress_auto_save
    assert not state.files.save_current_scene()
    assert not state.manager.enter_play_mode()
    assert not state.manager.resume()
    assert not state.manager.step_frame()
    elapsed = state.manager.total_play_time
    state.manager.tick(3.0)
    assert state.manager.total_play_time == elapsed
    assert state.path.read_bytes() == state.original_bytes
    monkeypatch.setattr(state.manager, "_rebuild_scene", rebuild)
    monkeypatch.setattr(state.manager, "_notify_state_change", notify)
    assert state.manager.exit_play_mode(on_complete=completed.append)
    state.runner.tick()
    assert completed == [False, True]
    assert state.manager.is_edit_mode and not state.native.is_playing()
    assert state.scene.find_by_id(state.object_id).name == "AuthorDoor"
    assert state.path.read_bytes() == state.original_bytes
    assert not Material._suppress_auto_save and not RenderEffect._suppress_auto_save


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("operation", ["open", "new", "confirmed_open", "confirmed_new"])
def test_editor_scene_replacement_cannot_retire_play_document(playing_author, paused, operation):
    state = playing_author
    from infernux.engine.interaction import DocumentRegistry

    registry = DocumentRegistry.instance()
    document = registry.require(state.files.document_id)
    revisions = (document.revision, document.saved_revision)
    if paused:
        assert state.manager.pause()
    if operation == "open":
        assert not state.files.open_scene(str(state.path))
    elif operation == "new":
        state.files.new_scene()
    elif operation == "confirmed_open":
        state.files._begin_deferred_open(str(state.path))
    else:
        state.files._begin_deferred_new()
    assert not state.files.is_loading
    state.files.poll_deferred_load()
    assert registry.require(document.document_id) is document
    assert state.files.document_id == document.document_id
    assert state.scene.find_by_id(state.object_id).name == "RuntimeDoor"
    for _ in range(3):
        assert state.manager.exit_play_mode()
        state.runner.tick()
        assert state.manager.is_edit_mode
        assert state.scene.find_by_id(state.object_id).name == "AuthorDoor"
        assert registry.require(document.document_id) is document
        assert (document.revision, document.saved_revision) == revisions
        assert state.path.read_bytes() == state.original_bytes
        assert state.manager.enter_play_mode()
        state.runner.tick()
        assert state.manager.is_playing


def test_pending_play_request_locks_editor_scene_replacement(playing_author):
    state = playing_author
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert state.manager.enter_play_mode()
    assert state.manager.is_edit_mode and state.runner.is_busy
    assert not state.files.open_scene(str(state.path))
    state.files.new_scene()
    assert not state.files.is_loading
    state.runner.tick()
    assert state.manager.is_playing
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert state.manager.is_edit_mode


def test_pending_editor_scene_load_blocks_play_snapshot(playing_author):
    state = playing_author
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert state.files.open_scene(str(state.path))
    assert state.files.is_loading
    assert not state.manager.enter_play_mode()
    assert state.manager.is_edit_mode and not state.runner.is_busy
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state.files.poll_deferred_load()
        if not state.files.is_loading:
            break
        time.sleep(0.005)
    assert not state.files.is_loading
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert state.manager.is_playing
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert state.manager.is_edit_mode
