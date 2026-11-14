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
from infernux.lib import SceneManager, _Infernux
from infernux.renderstack.render_effect import RenderEffect


@pytest.fixture
def play_host(engine, monkeypatch):
    """Attach the production Python owner to CI's existing native Engine."""
    import infernux.application as application_state
    from infernux.application import Application
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler
    from infernux.engine.engine import Engine
    from infernux.lib import RuntimeMode
    from infernux.renderstack.render_stack import RenderStack

    host = Engine.__new__(Engine)
    host._engine = engine
    host._mode = RuntimeMode.Graphical
    host._application_role = "editor"
    host._play_mode_manager = None
    host._player_runtime = None
    host._world_runtimes = {}
    host._runtime_world_session = None
    host._authoring_scheduler = RuntimeExecutionScheduler(name="pytest-play-recovery", native_bridge=True)
    host._initialize_work_runtime(engine.frame_work, accepting=True)
    native = SceneManager.instance()
    with monkeypatch.context() as globals_state:
        for name in ("_engine_ref", "_runtime_kind", "_exit_code", "_player_install_root"):
            globals_state.setattr(application_state, name, getattr(application_state, name))
        globals_state.setattr(RenderStack, "_scene_instances", dict(RenderStack._scene_instances))
        Application._bind_engine(host, "editor")
        try:
            host._install_pre_scene_time_callback()
            yield host
        finally:
            try:
                if host._runtime_world_session is not None:
                    host._runtime_world_session.close()
                host._shutdown_work_runtime()
                assert native._collect_retired_runtime_worlds()
                assert host._runtime_world_session is None
                assert not host._world_runtimes
            finally:
                engine._set_runtime_frame_callbacks(None, None)
                native._set_runtime_world_retirement_callbacks(None, None)
                native.clear_runtime_lifecycle_callbacks()
                host._authoring_scheduler.end_native_frame()
                host._authoring_scheduler.unbind_native_bridge()
                host._authoring_scheduler.clear()
                host._runtime_scene_manager = None
                Application._unbind_engine(host)
                host._engine = None


@pytest.fixture
def authoring_project(engine, scene, tmp_path, monkeypatch, play_host):
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
    play_host._play_mode_manager = manager
    native = SceneManager.instance()
    try:
        yield SimpleNamespace(manager=manager, runner=runner, files=files, scene=scene,
                              native=native, host=play_host, obj=obj, object_id=object_id, path=path,
                              runtime_domains=[],
                              original_bytes=path.read_bytes())
    finally:
        runner.cancel()
        native.stop()
        if manager._runtime_world_session is not None:
            manager._runtime_world_session.close()
            manager._runtime_world_session = None
        Material._suppress_auto_save = False
        RenderEffect._suppress_auto_save = False
        engine.set_play_mode_rendering(False)
        core.shutdown()
        project_context.set_project_root(previous_root)


def assert_authoring_restored(state):
    assert state.manager.is_edit_mode and not state.native.is_playing()
    assert not state.runner.is_busy
    assert not Material._suppress_auto_save and not RenderEffect._suppress_auto_save
    assert AssetManager._runtime_data_asset_cache() is None
    assert state.host._runtime_world_session is None
    assert not state.host._world_runtimes
    assert not state.native._runtime_world_ids
    assert all(not domain.data_assets for domain in state.runtime_domains)
    assert state.native.get_scene_by_world_id(state.scene.world_id) is state.scene
    assert state.scene.find_by_id(state.object_id) is state.obj
    assert state.obj.name == "AuthorEnterWitness"
    assert state.path.read_bytes() == state.original_bytes


def remember_runtime_domains(state):
    for domain in state.host._world_runtimes.values():
        if all(domain is not known for known in state.runtime_domains):
            state.runtime_domains.append(domain)


def mutate_runtime_clone(state):
    """Failure may leave runtime changes, while the retained author graph stays intact."""
    remember_runtime_domains(state)
    identities = state.native._runtime_world_ids
    assert identities
    identity = int(identities[0])
    with _Infernux._RuntimeWorldScope(identity):
        runtime_scene = state.native.get_scene(state.scene.name)
        assert runtime_scene is not None and runtime_scene is not state.scene
        runtime_obj = runtime_scene.find("AuthorEnterWitness")
        assert runtime_obj is not None and runtime_obj is not state.obj
        runtime_obj.name = "PartialRuntimeMutation"


@pytest.mark.parametrize("phase", [
    "snapshot", "sprite", "isolation", "enter_notification", "domain", "domain_rejected",
    "project_warmup", "native_before", "native_after", "component_warmup", "play_notification",
])
def test_failed_enter_restores_authoring_and_can_enter_again(authoring_project, monkeypatch, phase):
    state = authoring_project
    from infernux.components.builtin.sprite_renderer import SpriteRenderer
    from infernux.debug import Debug, DebugConsole

    Debug.clear_console()

    def fail(*args, **kwargs):
        if phase in ("snapshot", "sprite", "enter_notification", "isolation"):
            # These boundaries precede graph materialization; never mutate authoring.
            remember_runtime_domains(state)
            assert not state.native._runtime_world_ids
        else:
            mutate_runtime_clone(state)
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
            from infernux.engine import component_restore

            def isolated_domain_then_fail(*args, **kwargs):
                from infernux.engine.runtime_worlds import current_world_runtime

                domain = current_world_runtime()
                assert domain is not None
                assert state.host._world_runtimes[domain.identity] is domain
                assert AssetManager._runtime_data_asset_cache() is domain.data_assets
                domain.data_assets["recovery-witness"] = object()
                fail()

            # The real realm exists before the runtime component graph is prepared.
            fault.setattr(component_restore, "_prepare_python_component_records", isolated_domain_then_fail)
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
                    mutate_runtime_clone(state)
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
        errors = DebugConsole.instance().get_filtered_entries(show_logs=False, show_warnings=False)
        expected = ("Play scripting-domain preparation was rejected" if phase == "domain_rejected"
                    else f"injected {phase} failure")
        assert any(expected in entry.message for entry in errors), "The failed preparation must be reported"
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


@pytest.mark.parametrize("role", ["editor", "player", "headless"])
def test_play_progress_window_is_editor_only(authoring_project, monkeypatch, role):
    from infernux import application
    from infernux.engine.ui import busy_progress

    events = []

    class Progress:
        def __init__(self):
            events.append("open")

        def update(self, value, message):
            events.append((value, message))

        def close(self):
            events.append("close")

    monkeypatch.setattr(application, "_runtime_kind", role)
    monkeypatch.setattr(busy_progress, "BusyProgress", Progress)
    state = authoring_project

    def capture_preparation():
        events.append("prepare")
        raise RuntimeError("Stop at the progress presentation boundary")

    monkeypatch.setattr(state.manager, "_save_scene_state", capture_preparation)
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert state.manager.is_edit_mode and not state.native.is_playing()
    if role == "editor":
        assert events[0] == "open" and events[-1] == "close"
        assert any(isinstance(event, tuple) for event in events)
    else:
        assert events == ["prepare"]


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
    secondary_obj = secondary.create_game_object("SecondAuthorWitness")
    path = state.path.parent / "Second.scene"
    path.write_text(secondary.serialize_asset(), encoding="utf-8")
    original_bytes = path.read_bytes()
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
    from infernux.engine import component_restore

    prepare = component_restore.publish_prepared_scene_python_components
    created = []
    runtime_worlds = []

    def prepare_then_fail(runtime_scene, *args, **kwargs):
        result = prepare(runtime_scene, *args, **kwargs)
        if runtime_scene.name == secondary.name:
            assert runtime_scene is not secondary
            assert int(state.native._current_runtime_world_id) != int(state.native._authoring_runtime_world_id)
            primary = state.native.get_scene(state.scene.name)
            assert primary is not state.scene
            primary.find("AuthorEnterWitness").name = "PartialPrimary"
            runtime_scene.find("SecondAuthorWitness").name = "PartialSecondary"
            created.append(state.native.create_scene("RuntimeSceneDuringEnter").world_id)
            runtime_worlds.append(int(state.native._current_runtime_world_id))
            remember_runtime_domains(state)
            raise RuntimeError("second resident preparation failed after mutation")
        return result

    monkeypatch.setattr(component_restore, "publish_prepared_scene_python_components", prepare_then_fail)
    assert state.manager.enter_play_mode()
    state.runner.tick()
    assert_authoring_restored(state)
    assert created and runtime_worlds
    assert state.native.get_scene_by_world_id(secondary.world_id) is secondary
    assert secondary.find("SecondAuthorWitness") is secondary_obj
    assert secondary.serialize_asset() == authored
    assert path.read_bytes() == original_bytes
    assert (document.revision, document.saved_revision) == revision
    assert all(state.native.get_scene_by_world_id(world_id) is None for world_id in created)
    assert state.native.get_active_scene() is state.scene
