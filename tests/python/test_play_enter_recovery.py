"""Failed Play preparation restores authored Scenes before reopening authoring."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.engine import project_context, startup_warmup
from infernux.engine.deferred_task import DeferredTaskRunner
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.play_mode import PlayModeManager, PlayModeState
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager
from infernux.renderstack.render_effect import RenderEffect


@pytest.fixture
def authoring_project(engine, scene, tmp_path, monkeypatch):
    from infernux.components import script_loader

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
    obj = scene.create_game_object("AuthorEnterWitness")
    object_id = obj.id
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    path = folder / "Enter.scene"
    path.write_text(scene.serialize_asset(), encoding="utf-8")
    assert database.import_asset(str(path)).succeeded
    files.register_loaded_scene(scene, str(path))
    assert files.activate_loaded_scene(scene)
    manager = PlayModeManager()
    manager.set_asset_database(database)
    manager._native_engine = engine
    native = SceneManager.instance()
    try:
        yield SimpleNamespace(manager=manager, runner=runner, files=files, scene=scene,
                              native=native, object_id=object_id, path=path,
                              original_bytes=path.read_bytes())
    finally:
        runner.cancel()
        native.stop()
        AssetManager._end_play_data_asset_isolation()
        Material._suppress_auto_save = False
        RenderEffect._suppress_auto_save = False
        engine.set_play_mode_rendering(False)
        core.shutdown()
        project_context.set_project_root(previous_root)


def assert_authoring_restored(state):
    assert state.manager.is_edit_mode and not state.native.is_playing()
    assert not state.runner.is_busy
    assert not Material._suppress_auto_save and not RenderEffect._suppress_auto_save
    assert AssetManager._play_data_asset_cache is None
    assert state.scene.find_by_id(state.object_id).name == "AuthorEnterWitness"
    assert state.path.read_bytes() == state.original_bytes


@pytest.mark.parametrize("phase", [
    "snapshot", "sprite", "isolation", "enter_notification", "domain", "domain_rejected",
    "project_warmup", "native_before", "native_after", "component_warmup", "play_notification",
])
def test_failed_enter_restores_authoring_and_can_enter_again(authoring_project, monkeypatch, phase):
    state = authoring_project
    from infernux.components.builtin.sprite_renderer import SpriteRenderer
    from infernux.debug import Debug

    errors = []
    monkeypatch.setattr(Debug, "log_error", lambda message, *a, **kw: errors.append(str(message)))

    def fail(*args, **kwargs):
        # Snapshot capture fails before runtime ownership and must not mutate.
        if phase != "snapshot":
            state.scene.find_by_id(state.object_id).name = "PartialRuntimeMutation"
        raise RuntimeError(f"injected {phase} failure")

    with monkeypatch.context() as fault:
        if phase == "snapshot":
            fault.setattr(state.manager, "_save_scene_state", fail)
        elif phase == "sprite":
            initialize = SpriteRenderer.init_all_in_scene

            def initialize_once(scene):
                fault.setattr(SpriteRenderer, "init_all_in_scene", initialize)
                fail()

            fault.setattr(SpriteRenderer, "init_all_in_scene", initialize_once)
        elif phase == "isolation":
            begin = AssetManager._begin_play_data_asset_isolation

            def begin_then_fail():
                begin()
                fail()

            fault.setattr(AssetManager, "_begin_play_data_asset_isolation", begin_then_fail)
        elif phase in ("enter_notification", "play_notification"):
            notify = state.manager._notify_state_change

            def notify_then_fail(old, new):
                notify(old, new)
                if new.name == ("ENTERING" if phase == "enter_notification" else "PLAYING"):
                    fail()

            fault.setattr(state.manager, "_notify_state_change", notify_then_fail)
        elif phase in ("domain", "domain_rejected"):
            prepare = state.manager._prepare_loaded_scenes_for_play

            def prepare_then_fail():
                prepare()
                if phase == "domain_rejected":
                    return False
                fail()

            fault.setattr(state.manager, "_prepare_loaded_scenes_for_play", prepare_then_fail)
        elif phase == "project_warmup":
            fault.setattr(startup_warmup, "run_project_script_warmups", fail)
        elif phase == "component_warmup":
            fault.setattr(startup_warmup, "run_component_warmups", fail)
        else:
            class NativeStartFault:
                def __getattr__(self, name):
                    return getattr(state.native, name)

                def play(self):
                    if phase == "native_after":
                        state.native.play()
                    fail()

            fault.setattr(state.manager, "_get_scene_manager", lambda: NativeStartFault())
        assert state.manager.enter_play_mode()
        state.runner.tick()
        assert errors, "The failed preparation must be reported"
        assert state.manager.is_edit_mode, errors
        assert_authoring_restored(state)

    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert state.manager.is_playing and state.native.is_playing()
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert_authoring_restored(state)


def test_enter_publishes_playing_only_after_native_start(authoring_project):
    state = authoring_project
    observed = []

    def observe(event):
        observed.append((event.new_state.name, state.native.is_playing()))
        if event.new_state.name == "ENTERING":
            assert not state.manager.is_edit_mode
            assert not state.files.save_current_scene()

    state.manager.add_state_change_listener(observe)
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert observed == [("ENTERING", False), ("PLAYING", True)]


@pytest.mark.parametrize("restore_failure", ["rejected", "raised", "notification"])
def test_failed_enter_with_failed_rollback_retains_snapshot(authoring_project, monkeypatch, restore_failure):
    state = authoring_project

    def fail(**kwargs):
        raise RuntimeError("injected warmup failure")

    with monkeypatch.context() as fault:
        fault.setattr(startup_warmup, "run_project_script_warmups", fail)
        def reject_restore(*args):
            if restore_failure == "raised":
                raise RuntimeError("injected authoring recovery failure")
            return False

        if restore_failure == "notification":
            notify = state.manager._notify_state_change

            def reject_edit(old, new):
                if new is PlayModeState.EDIT:
                    raise RuntimeError("injected authoring notification failure")
                notify(old, new)

            fault.setattr(state.manager, "_notify_state_change", reject_edit)
        else:
            fault.setattr(state.manager, "_restore_loaded_scenes_after_play", reject_restore)
        assert state.manager.enter_play_mode()
        state.runner.tick()
        assert state.manager.state is PlayModeState.RECOVERY_REQUIRED
        assert not state.native.is_playing() and not state.runner.is_busy
        assert state.manager._scene_backups
        assert Material._suppress_auto_save and RenderEffect._suppress_auto_save
        assert not state.files.save_current_scene()
        assert state.path.read_bytes() == state.original_bytes
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert_authoring_restored(state)


def test_not_ready_warmup_is_not_a_failed_play(authoring_project, monkeypatch):
    state = authoring_project
    calls = []

    class NotReady:
        def _infernux_startup_warmup(self):
            calls.append(True)
            return False

    warmup = startup_warmup.run_component_warmups

    def invoke_actual_hook(*args, **kwargs):
        return warmup([NotReady()], scope="editor-play", project_path=kwargs["project_path"])

    monkeypatch.setattr(startup_warmup, "run_component_warmups", invoke_actual_hook)
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert calls and state.manager.is_playing and state.native.is_playing()
    assert state.manager.exit_play_mode()
    state.runner.tick()
    assert_authoring_restored(state)


def test_failed_second_scene_preparation_restores_every_resident(authoring_project, monkeypatch):
    state = authoring_project
    from infernux.engine.interaction import DocumentRegistry

    secondary = state.native.create_scene("SecondEnterScene")
    secondary.create_game_object("SecondAuthorWitness")
    path = state.path.parent / "Second.scene"
    path.write_text(secondary.serialize_asset(), encoding="utf-8")
    database = AssetManager._asset_database
    assert database.import_asset(str(path)).succeeded
    state.files.register_loaded_scene(secondary, str(path))
    registry = DocumentRegistry.instance()
    document_id = state.files.document_id_for_scene(secondary)
    registry.mark_changed(document_id)
    document = registry.require(document_id)
    revision = (document.revision, document.saved_revision)
    authored = secondary.serialize_asset()
    assert state.files.activate_loaded_scene(state.scene)
    prepare = state.manager._prepare_scene_for_play
    created = []

    def prepare_then_fail(scene, snapshot):
        result = prepare(scene, snapshot)
        if scene is secondary:
            state.scene.find_by_id(state.object_id).name = "PartialPrimary"
            secondary.find("SecondAuthorWitness").name = "PartialSecondary"
            created.append(state.native.create_scene("RuntimeSceneDuringEnter").world_id)
            raise RuntimeError("second resident preparation failed after mutation")
        return result

    monkeypatch.setattr(state.manager, "_prepare_scene_for_play", prepare_then_fail)
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert_authoring_restored(state)
    assert secondary.serialize_asset() == authored
    assert (document.revision, document.saved_revision) == revision
    assert all(state.native.get_scene_by_world_id(world_id) is None for world_id in created)
    assert state.native.get_active_scene() is state.scene
