"""Tests for infernux.engine.play_mode — PlayModeState, PlayModeEvent, PlayModeManager."""

from contextlib import contextmanager
import sys
import time
from types import SimpleNamespace

import pytest

from infernux.components import InxComponent
from infernux.components.component_identity import bind_asset_script_guid
from infernux.components.missing_script import MissingScript
from infernux.core.asset_ref import MaterialRef
from infernux.engine.play_mode import PlayModeState, PlayModeEvent, PlayModeManager


class _FakeRuntimeGameObject:
    def __init__(self, object_id: int, name: str):
        self.id = object_id
        self.name = name
        self.transform = object()


class _FakeScriptGameObject:
    def __init__(self, object_id: int, component):
        self.id = object_id
        self.components = [component]
        self.remove_calls = 0
        self.add_calls = 0
        self.replace_calls = 0

    def get_py_components(self):
        return list(self.components)

    def remove_py_component(self, component):
        self.remove_calls += 1
        self.components.remove(component)

    def add_py_component(self, component):
        self.add_calls += 1
        self.components.append(component)

    def replace_py_component(self, old_component, new_component):
        self.replace_calls += 1
        index = self.components.index(old_component)
        new_component._component_id = old_component.component_id
        self.components[index] = new_component
        return new_component


class _FakeScriptScene:
    def __init__(self, game_object):
        self.game_object = game_object

    def get_all_objects(self):
        return [self.game_object]

    def find_by_id(self, object_id):
        return self.game_object if object_id == self.game_object.id else None


class _FakeMultiScriptScene:
    def __init__(self, game_objects):
        self.game_objects = list(game_objects)

    def get_all_objects(self):
        return list(self.game_objects)

    def find_by_id(self, object_id):
        return next((obj for obj in self.game_objects if obj.id == object_id), None)


class _FailingRecoveryGameObject(_FakeScriptGameObject):
    def __init__(self, object_id: int, component):
        super().__init__(object_id, component)
        self.fail_recovery = False

    def replace_py_component(self, old_component, new_component):
        if (
            self.fail_recovery
            and isinstance(old_component, MissingScript)
            and not isinstance(new_component, MissingScript)
        ):
            raise RuntimeError("simulated MissingScript replacement failure")
        return super().replace_py_component(old_component, new_component)


class _FakeScriptSceneManager:
    # These are source-graph transaction units, not managed Play sessions.
    _runtime_world_ids = ()

    def __init__(self, scene):
        self.scene = scene

    @property
    def _authoring_runtime_world_id(self):
        from infernux.lib import SceneManager
        return SceneManager.instance()._authoring_runtime_world_id

    def get_active_scene(self):
        return self.scene


class _FakePausedGroupSceneManager(_FakeScriptSceneManager):
    """Explicit two-member unit protocol; this is not real-World evidence."""
    def __init__(self):
        super().__init__(None)
        self._runtime_world_ids = (101, 202)
        self._scope_id = self._authoring_runtime_world_id
        self.admissions = []
        self.play_calls = []
        self.worlds = {
            101: SimpleNamespace(handle=SimpleNamespace(id=101, state='ready', destroy_requested=False),
                epoch=7, clock=SimpleNamespace(frame_count=0, unscaled_delta_time=0.0),
                target=10, stepping=False, playing=True, paused=True, pending_delta=None),
            202: SimpleNamespace(handle=SimpleNamespace(id=202, state='ready', destroy_requested=False),
                epoch=9, clock=SimpleNamespace(frame_count=4, unscaled_delta_time=.03),
                target=20, stepping=False, playing=True, paused=True, pending_delta=None),
        }

    @contextmanager
    def scope(self, identity):
        assert identity in self.worlds or identity == self._authoring_runtime_world_id
        previous, self._scope_id = self._scope_id, identity
        try:
            yield
        finally:
            self._scope_id = previous

    def _get_runtime_world_handle(self, identity):
        return self.worlds[identity].handle

    @property
    def simulation_epoch(self):
        return self.worlds[self._scope_id].epoch

    @property
    def _frame_time(self):
        return self.worlds[self._scope_id].clock

    @property
    def _step_permission_index(self):
        return self.worlds[self._scope_id].target

    def is_playing(self):
        return self.worlds[self._scope_id].playing

    def is_paused(self):
        return self.worlds[self._scope_id].paused

    def is_stepping(self):
        return self.worlds[self._scope_id].stepping

    def _request_runtime_steps(self, requests):
        batch = tuple(requests)
        assert {identity for identity, _epoch, _delta in batch} == set(self.worlds)
        for identity, epoch, delta in batch:
            state = self.worlds[identity]
            assert epoch == state.epoch and state.playing and state.paused and not state.stepping
            assert delta >= 0
        self.admissions.append(batch)
        for identity, _epoch, delta in batch:
            state = self.worlds[identity]
            state.target += 1
            state.stepping = True
            state.pending_delta = delta

    def complete(self, identity):
        state = self.worlds[identity]
        assert state.stepping
        state.clock.frame_count += 1
        state.clock.unscaled_delta_time = state.pending_delta
        state.pending_delta = None
        state.stepping = False

    def play(self):
        assert self._scope_id == self._authoring_runtime_world_id
        self.play_calls.append(self._scope_id)
        for state in self.worlds.values():
            state.paused = False


# ══════════════════════════════════════════════════════════════════════
# PlayModeState enum
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# PlayModeEvent
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# PlayModeManager
# ══════════════════════════════════════════════════════════════════════


class TestPlayModeManager:
    def test_paused_step_sequence_records_completed_commands(self, monkeypatch):
        from infernux.lib import _Infernux

        mgr = PlayModeManager()
        scene_manager = _FakePausedGroupSceneManager()
        monkeypatch.setattr(_Infernux, '_RuntimeWorldScope', scene_manager.scope)
        mgr._state = PlayModeState.PAUSED
        mgr._get_scene_manager = lambda: scene_manager

        assert mgr.step_frame()
        assert mgr.step_frame()
        assert mgr.step_sequence == 0
        assert scene_manager.admissions == []
        mgr.process_pending_step()
        assert mgr.step_sequence == 0  # Batch admission is not completion.
        assert len(scene_manager.admissions) == 1
        mgr.process_pending_step()
        assert mgr.step_sequence == 0
        assert len(scene_manager.admissions) == 1
        scene_manager.complete(101)
        mgr.process_pending_step()
        assert mgr.step_sequence == 0  # One finished member is not the group.
        assert len(scene_manager.admissions) == 1
        scene_manager.complete(202)
        mgr.process_pending_step()
        assert mgr.step_sequence == 1
        assert len(scene_manager.admissions) == 2
        for identity in scene_manager._runtime_world_ids:
            scene_manager.complete(identity)
        mgr.process_pending_step()
        assert mgr.step_sequence == 2

        assert mgr.step_frame()
        mgr.process_pending_step()
        mgr.process_pending_step()
        assert mgr.step_sequence == 2
        assert len(scene_manager.admissions) == 3
        for identity in scene_manager._runtime_world_ids:
            scene_manager.complete(identity)
        mgr.process_pending_step()
        assert mgr.step_sequence == 3
        for batch in scene_manager.admissions:
            assert batch == ((101, 7, pytest.approx(1.0/60.0)), (202, 9, pytest.approx(.03)))
        assert scene_manager.worlds[101].clock.frame_count == 3
        assert scene_manager.worlds[202].clock.frame_count == 7
        mgr.process_pending_step()
        assert mgr.step_sequence == 3
        assert len(scene_manager.admissions) == 3

    def test_paused_step_runs_after_authoring_transaction(self, monkeypatch):
        from infernux.lib import _Infernux
        from infernux.engine.runtime_change_journal import RuntimeChangeJournal

        journal = RuntimeChangeJournal()
        mgr = PlayModeManager()
        mgr._state = PlayModeState.PAUSED
        scene_manager = _FakePausedGroupSceneManager()
        monkeypatch.setattr(_Infernux, '_RuntimeWorldScope', scene_manager.scope)
        admit = scene_manager._request_runtime_steps

        def request_steps(members):
            journal.flush()  # Fails if admission was moved inside the transaction.
            admit(members)

        scene_manager._request_runtime_steps = request_steps
        mgr._get_scene_manager = lambda: scene_manager
        with journal.transaction():
            assert mgr.step_frame()
            assert mgr.step_sequence == 0
            assert scene_manager.admissions == []
        assert scene_manager.admissions == []
        mgr.process_pending_step()
        assert len(scene_manager.admissions) == 1
        assert mgr.step_sequence == 0
        scene_manager.complete(101)
        mgr.process_pending_step()
        assert mgr.step_sequence == 0
        scene_manager.complete(202)
        mgr.process_pending_step()
        assert mgr.step_sequence == 1
        assert len(scene_manager.admissions) == 1

    def test_pending_step_is_cancelled_on_resume(self, monkeypatch):
        from infernux.lib import _Infernux

        mgr = PlayModeManager()
        mgr._state = PlayModeState.PAUSED
        scene_manager = _FakePausedGroupSceneManager()
        monkeypatch.setattr(_Infernux, '_RuntimeWorldScope', scene_manager.scope)
        mgr._get_scene_manager = lambda: scene_manager
        assert mgr.step_frame()
        assert mgr.resume()
        assert scene_manager.play_calls == [scene_manager._authoring_runtime_world_id]
        mgr._state = PlayModeState.PAUSED
        mgr.process_pending_step()
        assert mgr.step_sequence == 0
        assert scene_manager.admissions == []

    def test_restore_scene_path_reasserts_authored_scene_persistence(self, monkeypatch, tmp_path):
        from infernux.engine.scene_manager import SceneFileManager

        authored = str(tmp_path / "racetrack.scene")
        runtime = str(tmp_path / "results.scene")
        manager = SceneFileManager()
        manager._current_scene_path = runtime
        restored_cameras = []
        remembered_paths = []
        scene_changed = []
        manager._restore_camera_state = restored_cameras.append
        manager._remember_last_scene = remembered_paths.append
        manager._on_scene_changed = lambda: scene_changed.append(True)

        mgr = PlayModeManager()
        mgr._scene_path_backup = authored
        mgr._scene_document_id_backup = manager.document_id
        from infernux.engine.interaction import DocumentRegistry

        document = DocumentRegistry.instance().require(manager.document_id)
        DocumentRegistry.instance().mark_changed(document.document_id)
        mgr._scene_revision_backup = document.revision
        mgr._scene_saved_revision_backup = document.saved_revision
        mgr._scene_document_state_backup = document.state
        DocumentRegistry.instance().restore_saved_revision(document.document_id)
        mgr._restore_scene_file_path()

        assert manager.current_scene_path == authored
        assert manager.is_dirty is True
        assert restored_cameras == [authored]
        assert remembered_paths == [authored]
        assert scene_changed == [True]

    def test_restore_unsaved_scene_preserves_document_identity_and_revision(self):
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.scene_manager import SceneFileManager

        manager = SceneFileManager()
        original_document_id = manager.document_id
        document = DocumentRegistry.instance().require(original_document_id)
        DocumentRegistry.instance().mark_changed(document.document_id)
        original_revision = document.revision
        original_saved_revision = document.saved_revision
        manager._current_scene_path = "runtime.scene"
        DocumentRegistry.instance().restore_saved_revision(document.document_id)
        scene_changed = []
        remembered_paths = []
        manager._on_scene_changed = lambda: scene_changed.append(True)
        manager._remember_last_scene = remembered_paths.append

        mgr = PlayModeManager()
        mgr._scene_path_backup = None
        mgr._scene_document_id_backup = original_document_id
        mgr._scene_revision_backup = original_revision
        mgr._scene_saved_revision_backup = original_saved_revision
        mgr._scene_document_state_backup = document.state
        mgr._restore_scene_file_path()

        assert manager.current_scene_path is None
        assert manager.document_id == original_document_id
        assert manager.is_dirty is True
        assert scene_changed == [True]
        assert remembered_paths == []

    def test_exit_play_mode_restores_document_state_from_current_snapshot(self, monkeypatch, scene):
        from infernux.engine.deferred_task import DeferredTaskRunner
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.scene_manager import SceneFileManager
        from infernux.input import Input
        from infernux.lib import SceneManager

        native = SceneManager.instance()

        class _SceneManager:
            def __init__(self):
                self.stop_calls = 0

            @property
            def _authoring_runtime_world_id(self):
                return native._authoring_runtime_world_id

            def get_scene_by_world_id(self, world_id):
                return native.get_scene_by_world_id(world_id)

            def stop(self):
                self.stop_calls += 1

        class _Runner:
            is_busy = False

            def __init__(self):
                self.steps = []

            def submit(self, _name, steps, on_done=None):
                self.steps = list(steps)
                self.on_done = on_done
                return True

        scene_manager = _SceneManager()
        runner = _Runner()
        monkeypatch.setattr(DeferredTaskRunner, "_instance", runner)
        focus_calls = []
        lock_calls = []
        monkeypatch.setattr(Input, "set_game_focused", focus_calls.append)
        monkeypatch.setattr(Input, "set_cursor_locked", lock_calls.append)

        monkeypatch.setattr(SceneFileManager, "_instance", None)
        files = SceneFileManager()
        document_id = files.register_loaded_scene(scene, "", dirty=True)
        assert files.activate_loaded_scene(scene)
        registry = DocumentRegistry.instance()
        document = registry.require(document_id)
        registry.mark_changed(document_id)
        authored_state = document.revision, document.saved_revision, document.state
        manager = PlayModeManager()
        manager._save_scene_state()
        snapshot = manager._scene_backup
        # Play can change the active document's UI state, but Stop must bind
        # the retained authored Scene and restore its exact revision pair.
        registry.mark_changed(document_id)
        files._current_scene_path = "runtime.scene"
        manager._state = PlayModeState.PLAYING
        manager._get_scene_manager = lambda: scene_manager
        # No runtime loader exists in this Stop/document-restoration unit.
        # Keep its cancellation boundary observable without inventing an Engine.
        cancelled_loads = []
        monkeypatch.setattr(manager, "_cancel_runtime_scene_loads", lambda: cancelled_loads.append(True))
        manager._rebuild_active_scene = lambda *args, **kwargs: pytest.fail(
            "Stop must reactivate the retained authored graph")
        notifications = []
        manager._notify_state_change = lambda old, new: notifications.append((old, new))

        assert manager.exit_play_mode() is True
        assert manager.state is PlayModeState.RESTORING
        assert scene_manager.stop_calls == 1
        assert cancelled_loads == [True]
        assert focus_calls == [False]
        assert lock_calls == [False]

        _, _, restore = runner.steps[0]
        restore()

        assert manager.state is PlayModeState.EDIT

        assert manager._scene_backup is snapshot
        assert native.get_active_scene() is scene
        assert files.document_id == document_id
        assert files.document_id_for_scene(scene) == document_id
        assert files.current_scene_path is None
        assert (document.revision, document.saved_revision, document.state) == authored_state
        assert files.is_dirty is True
        assert notifications == [(PlayModeState.PLAYING, PlayModeState.EDIT)]

    def test_deleted_script_becomes_missing_and_recovers_with_identity(self, tmp_path, monkeypatch):
        script_guid = "1" * 32

        class ScriptProbe(InxComponent):
            _uses_component_data_store = False
            speed: float = 1.0
            material: MaterialRef = MaterialRef()

        bind_asset_script_guid(ScriptProbe, script_guid)
        component = ScriptProbe()
        component._script_guid = script_guid
        component.speed = 7.5
        component.material = MaterialRef(guid="material-guid", path_hint="Assets/Test.mat")
        component.enabled = False
        component.execution_order = 23
        component._call_awake()
        component._call_start()
        original_id = component.component_id

        game_object = _FakeScriptGameObject(91, component)
        scene = _FakeScriptScene(game_object)
        manager = PlayModeManager()
        monkeypatch.setattr(
            manager,
            "_get_scene_manager",
            lambda: _FakeScriptSceneManager(scene),
        )

        script = tmp_path / "script_probe.py"
        assert manager.mark_components_missing_for_script(script_guid, str(script)) == 1
        missing = game_object.components[0]
        assert isinstance(missing, MissingScript)
        assert missing.component_id == original_id
        assert missing._script_guid == script_guid
        assert missing._serialize_fields_document()["speed"] == 7.5
        assert missing._serialize_fields_document()["material"]["guid"] == "material-guid"
        assert missing._enabled is False
        assert missing._awake_called is True
        assert missing._has_started is True
        assert missing._execution_order == 23
        assert game_object.replace_calls == 1
        assert game_object.remove_calls == 0
        assert game_object.add_calls == 0

        saved_missing = manager._serialize_py_component(missing)
        missing._preserved_fields["speed"] = 999.0
        manager._apply_py_component_state(missing, saved_missing)
        assert missing._serialize_fields_document()["speed"] == 7.5
        assert missing._serialize_fields_document()["material"]["guid"] == "material-guid"
        assert missing.component_id == original_id

        script.write_text(
            "from infernux.components import InxComponent\n"
            "from infernux.core.asset_ref import MaterialRef\n"
            "CALLS = []\n"
            "class ScriptProbe(InxComponent):\n"
            "    _uses_component_data_store = False\n"
            "    speed: float = 1.0\n"
            "    material: MaterialRef = MaterialRef()\n"
            "    def awake(self): CALLS.append('awake')\n"
            "    def start(self): CALLS.append('start')\n",
            encoding="utf-8",
        )

        class _AssetDatabase:
            def get_guid_from_path(self, path):
                assert path == str(script.resolve())
                return script_guid

        manager.set_asset_database(_AssetDatabase())
        from infernux.components._component_lifecycle import RuntimeExecutionScheduler
        from infernux.engine.play_mode import ScriptReloadBatchInput
        from infernux.engine.runtime_dispatch import current_runtime_epoch

        batch = manager.prepare_script_reload_batch((ScriptReloadBatchInput(str(script), script_guid),))
        before_epoch = current_runtime_epoch()
        scheduler = RuntimeExecutionScheduler()
        frame = scheduler.begin_frame()
        try:
            with pytest.raises(RuntimeError, match="safe point"):
                batch.commit()
            assert not batch.committed and not batch.rolled_back
            assert game_object.components[0] is missing
            assert current_runtime_epoch() is before_epoch
        finally:
            frame.close()
        assert manager.commit_script_reload_batch(batch).success
        manager.finalize_script_reload_batch(batch)

        restored = game_object.components[0]
        assert not isinstance(restored, MissingScript)
        assert restored.component_id == original_id
        assert restored._script_guid == script_guid
        assert restored._script_path == str(script.resolve())
        assert restored.speed == 7.5
        from infernux.components.fields import get_raw_field_value

        restored_material = get_raw_field_value(restored, "material")
        assert restored_material.guid == "material-guid"
        assert restored_material.path_hint == ""
        assert restored._enabled is False
        assert restored._awake_called is True
        assert restored._has_started is True
        assert restored._execution_order == 23
        restored._call_awake()
        restored._call_start()
        assert sys.modules[restored.__class__.__module__].CALLS == []
        assert game_object.replace_calls == 2
        assert game_object.remove_calls == 0
        assert game_object.add_calls == 0

    def test_missing_script_recovery_failure_rolls_back_scene_registry_and_module(
        self,
        tmp_path,
        monkeypatch,
    ):
        from infernux.components.registry import snapshot_component_registry_state
        from infernux.engine.project_context import get_script_module_name

        script_guid = "2" * 32

        class AtomicRecoveryProbe(InxComponent):
            _uses_component_data_store = False
            speed: float = 1.0

        bind_asset_script_guid(AtomicRecoveryProbe, script_guid)
        first = AtomicRecoveryProbe()
        second = AtomicRecoveryProbe()
        for index, component in enumerate((first, second), start=1):
            component._script_guid = script_guid
            component.speed = index * 4.5
            component.enabled = index == 1
            component._call_awake()
            component._call_start()

        first_object = _FailingRecoveryGameObject(101, first)
        second_object = _FailingRecoveryGameObject(102, second)
        scene = _FakeMultiScriptScene((first_object, second_object))
        manager = PlayModeManager()
        monkeypatch.setattr(
            manager,
            "_get_scene_manager",
            lambda: _FakeScriptSceneManager(scene),
        )

        script = tmp_path / "atomic_recovery_probe.py"
        assert manager.mark_components_missing_for_script(script_guid, str(script)) == 2
        first_missing = first_object.components[0]
        second_missing = second_object.components[0]
        before_registry = snapshot_component_registry_state()
        module_name = get_script_module_name(str(script.resolve()))
        before_module = sys.modules.get(module_name)

        script.write_text(
            "from infernux.components import InxComponent\n"
            "class AtomicRecoveryProbe(InxComponent):\n"
            "    _uses_component_data_store = False\n"
            "    speed: float = 1.0\n",
            encoding="utf-8",
        )

        class _AssetDatabase:
            def get_guid_from_path(self, path):
                assert path == str(script.resolve())
                return script_guid

        manager.set_asset_database(_AssetDatabase())
        second_object.fail_recovery = True
        outcome = manager.reload_components_from_script_result(str(script))

        assert outcome.success is False
        assert "simulated MissingScript replacement failure" in outcome.error
        assert first_object.components[0] is first_missing
        assert second_object.components[0] is second_missing
        assert first_missing._serialize_fields_document()["speed"] == 4.5
        assert second_missing._serialize_fields_document()["speed"] == 9.0
        assert first_missing._awake_called is True
        assert first_missing._has_started is True
        assert second_missing._awake_called is True
        assert second_missing._has_started is True
        assert snapshot_component_registry_state() == before_registry
        if before_module is None:
            assert module_name not in sys.modules
        else:
            assert sys.modules[module_name] is before_module

    def test_prepare_play_scene_propagates_primary_transaction_failure(self, monkeypatch):
        class _Scene:
            pass

        class _SceneManager:
            @staticmethod
            def get_active_scene():
                return _Scene()

        mgr = PlayModeManager()
        mgr._get_scene_manager = lambda: _SceneManager()
        rebuilds = []
        mgr._rebuild_active_scene = lambda *args, **kwargs: rebuilds.append((args, kwargs))

        from infernux.engine import component_restore

        def reject(*_args, **_kwargs):
            raise RuntimeError("component transaction rejected")

        monkeypatch.setattr(
            component_restore,
            "replace_scene_python_components_for_play",
            reject,
        )

        with pytest.raises(RuntimeError, match="component transaction rejected"):
            mgr._prepare_active_scene_for_play({"objects": []})

        assert rebuilds == []

    def test_rebuild_scene_failure_preserves_runtime_hidden_ids(self):
        mgr = PlayModeManager()
        mgr._runtime_hidden_object_ids = {1, 2, 3}

        assert not mgr._rebuild_active_scene(None, for_play=False)
        assert mgr._runtime_hidden_object_ids == {1, 2, 3}

    def test_rejected_document_preserves_runtime_hidden_ids(self, monkeypatch):
        class _RejectingScene:
            def _commit_document(self, snapshot):
                return False

        class _FakeSceneManager:
            def get_active_scene(self):
                return _RejectingScene()

        mgr = PlayModeManager()
        mgr._runtime_hidden_object_ids = {77}
        monkeypatch.setattr(mgr, "_get_scene_manager", lambda: _FakeSceneManager())

        assert not mgr._rebuild_active_scene({"invalid": True}, for_play=False)
        assert mgr._runtime_hidden_object_ids == {77}

    def test_rebuild_scene_enters_play_after_transaction(self, monkeypatch):
        class _FakeCommitToken:
            def __init__(self, scene, previous_document):
                self._scene = scene
                self._previous_document = previous_document
                self.is_active = True
                self.object_id_remap = {}
                self.component_id_remap = {}

            def rollback(self):
                if not self.is_active:
                    return False
                self._scene.document = self._previous_document
                self.is_active = False
                return True

            def finalize(self):
                self.is_active = False

        class _FakeScene:
            @property
            def _runtime_world_id(self):
                from infernux.lib import SceneManager
                return SceneManager.instance()._authoring_runtime_world_id

            def __init__(self):
                self.world_id = 1
                self.playing = None
                self.document = {
                    "name": "FakeLiveScene",
                    "isPlaying": False,
                    "objects": [],
                }

            def serialize_document(self):
                return dict(self.document)

            def get_all_objects(self):
                return []

            def find_objects_with_component(self, _type_name):
                return []

            def _commit_document_retaining_world(self, snapshot):
                token = _FakeCommitToken(self, dict(self.document))
                self.document = dict(snapshot)
                return token

            def set_playing(self, playing):
                self.playing = playing

        class _FakeSceneManager:
            def __init__(self, scene):
                self._scene = scene

            def get_active_scene(self):
                return self._scene

        mgr = PlayModeManager()
        scene = _FakeScene()
        scene_manager = _FakeSceneManager(scene)
        monkeypatch.setattr(mgr, "_get_scene_manager", lambda: scene_manager)
        from infernux.engine import component_restore

        monkeypatch.setattr(
            component_restore,
            "preflight_scene_python_components",
            lambda snapshot, asset_database=None, **_kwargs: component_restore.PreparedPythonComponentGraph([]),
        )
        monkeypatch.setattr(
            component_restore,
            "publish_prepared_scene_python_components",
            lambda scene, prepared, clear_registries=True, object_id_map=None, component_id_map=None: (
                prepared.consume()
            ),
        )

        snapshot = {
            "name": "PlayModeRebuild",
            "isPlaying": False,
            "objects": [],
        }
        assert mgr._rebuild_active_scene(snapshot, for_play=True)
        assert scene.playing is True
