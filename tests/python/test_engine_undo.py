"""Undo command workflows and independent transaction/lifetime failures."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types

import pytest

import infernux.engine.undo as _undo_mod
import infernux.engine.undo._helpers as _helpers_mod
import infernux.engine.undo._property_commands as _property_mod
import infernux.engine.undo._recreate as _recreate_mod
import infernux.engine.undo._structural_commands as _structural_mod
from infernux.engine.interaction import (
    ContextRestoreStatus,
    EditorContextSnapshot,
    SelectionService,
    SelectionSnapshot,
    SelectionTarget,
)
from infernux.engine.undo import (
    BuiltinPropertyCommand,
    CompoundCommand,
    CreateGameObjectCommand,
    DeleteGameObjectCommand,
    DeleteGameObjectsCommand,
    GenericComponentCommand,
    GlobalSelectionCommand,
    MaterialDocumentCommand,
    PrefabModeCommand,
    PrefabUnpackCommand,
    RemoveComponentsCommand,
    RenderStackFieldCommand,
    SetPropertyCommand,
    UndoCommand,
    UndoManager,
)


def _patch_undo_modules(monkeypatch, attr: str, value):
    for mod in (_undo_mod, _helpers_mod, _property_mod, _structural_mod):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, value)


def _patch_world_undo(monkeypatch, scene):
    """Bind undo fixtures to the same World-wide lookup used by the engine."""

    def _find(object_id):
        obj = scene.find_by_id(object_id) if scene is not None else None
        if obj is not None and getattr(obj, "scene", None) is None:
            obj.scene = scene
        return obj

    _patch_undo_modules(monkeypatch, "_get_active_scene", lambda: scene)
    _patch_undo_modules(monkeypatch, "_find_runtime_object", _find)
    _patch_undo_modules(monkeypatch, "_get_scene_by_world_id", lambda _world_id: scene)


@contextmanager
def _override_recreate_game_object(fn):
    orig_root = _undo_mod._recreate_game_object_from_document
    orig_sub = _recreate_mod._recreate_game_object_from_document
    _undo_mod._recreate_game_object_from_document = fn
    _recreate_mod._recreate_game_object_from_document = fn
    try:
        yield
    finally:
        _undo_mod._recreate_game_object_from_document = orig_root
        _recreate_mod._recreate_game_object_from_document = orig_sub


# ── Helpers ──


class _Obj:
    """Simple mutable target for SetPropertyCommand."""

    def __init__(self):
        self.x = 0
        self.y = 0


class _FakeComp:
    """Fake native component with typed document serialization."""

    def __init__(self):
        self.component_id = 42
        self.type_name = "FakeComp"
        self._document = {}

    def serialize_document(self):
        return dict(self._document)

    def deserialize_document(self, document):
        self._document = dict(document)
        return True


class _FakeStack:
    """Fake RenderStack with invalidate_graph() for RenderStackFieldCommand."""

    def __init__(self, pipeline=None):
        self.pipeline_class_name = ""
        self.pipeline = pipeline
        self.invalidated = 0
        self.synced = 0

    def set_pipeline_parameter(self, field_name, value, **_kwargs):
        setattr(self.pipeline, field_name, value)
        self.synced += 1
        self.invalidate_graph()

    def invalidate_graph(self):
        self.invalidated += 1


@pytest.fixture(autouse=True)
def _reset_undo_manager():
    """Ensure a fresh UndoManager for every test; tear down afterwards."""
    old = UndoManager._instance
    mgr = UndoManager()
    # Prevent dirty-sync from importing SceneFileManager / PlayModeManager
    yield mgr
    UndoManager._instance = old


# ══════════════════════════════════════════════════════════════════════
# _snapshot_value
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# SetPropertyCommand
# ══════════════════════════════════════════════════════════════════════


class TestSetPropertyCommand:
    def test_automatic_history_merge_keeps_first_snapshot_and_last_authored_value(self, _reset_undo_manager):
        manager, target = _reset_undo_manager, _Obj()
        first = SetPropertyCommand(target, "x", 0, 1)
        assert manager.execute(first)
        second = SetPropertyCommand(target, "x", 1, 5)
        second.timestamp = first.timestamp + 0.1
        assert first.can_merge(second)
        assert manager.execute(second)
        assert target.x == 5 and len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert target.x == 0
        manager.redo()
        assert target.x == 5

    def test_native_backed_undo_fails_closed_when_target_missing(self, monkeypatch):
        class _NativeTarget:
            def __init__(self):
                self.x = 99
                self.game_object_id = 42
                self.type_name = "Rigidbody"

        target = _NativeTarget()
        cmd = SetPropertyCommand(target, "x", 1, 2)
        _patch_undo_modules(monkeypatch, "_get_active_scene", lambda: None)
        with pytest.raises(RuntimeError, match="History target is unavailable"):
            cmd.undo()
        assert target.x == 99


# ══════════════════════════════════════════════════════════════════════
# GenericComponentCommand
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# BuiltinPropertyCommand
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# MaterialDocumentCommand
# ══════════════════════════════════════════════════════════════════════


class TestMaterialDocumentCommand:
    def test_undo_publishes_live_material_without_durable_invalidation(
        self,
        monkeypatch,
    ):
        publications = []

        class _AsyncMaterial(_FakeComp):
            file_path = "Assets/Materials/Undo.mat"

            def save(self):
                return True

        class _AssetManagerProbe:
            @classmethod
            def note_asset_edit(cls, path, **kwargs):
                publications.append((path, kwargs))

            @classmethod
            def on_material_saved(cls, _path):
                raise AssertionError("Undo/Redo must not publish durable completion before IO finishes")

        assets_module = types.ModuleType("infernux.core.assets")
        assets_module.AssetManager = _AssetManagerProbe
        monkeypatch.setitem(sys.modules, "infernux.core.assets", assets_module)

        material = _AsyncMaterial()
        command = MaterialDocumentCommand(
            material,
            {"color": "A"},
            {"color": "B"},
        )

        command.execute()
        command.undo()

        assert [entry[0] for entry in publications] == [
            material.file_path,
            material.file_path,
        ]
        assert '"color":"B"' in publications[0][1]["material_json"]
        assert '"color":"A"' in publications[1][1]["material_json"]


# ══════════════════════════════════════════════════════════════════════
# CompoundCommand
# ══════════════════════════════════════════════════════════════════════


class TestCompoundCommand:
    def test_compound_add_component_undo_reverses_auto_created(self):
        """Simulate adding a component with an automatically-created dependency.

        A CompoundCommand [AddDependency, AddMain] should remove the dependent
        component first and recreate the dependency first.
        """
        log: list = []

        class FakeAddCmd(UndoCommand):
            def __init__(self, tag):
                super().__init__(f"Add {tag}")
                self.tag = tag

            def execute(self):
                log.append(("add", self.tag))

            def undo(self):
                log.append(("remove", self.tag))

            def redo(self):
                log.append(("re-add", self.tag))

        # AddComponentTransactionCommand stores dependencies first, main last.
        compound = CompoundCommand([FakeAddCmd("Dependency"), FakeAddCmd("Main")], "Add Main")

        # Undo should reverse: remove Main, then Dependency
        compound.undo()
        assert log == [("remove", "Main"), ("remove", "Dependency")]

        log.clear()
        # Redo should replay in order: add Dependency, then Main
        compound.redo()
        assert log == [("re-add", "Dependency"), ("re-add", "Main")]

    def test_component_batch_removal_owns_selection_transition(self):
        previous = SelectionService._instance
        service = SelectionService()
        before = SelectionSnapshot.create(
            (
                SelectionTarget.component(41, 101),
                SelectionTarget.component(42, 102),
            ),
            owner_id="inspector",
            primary=SelectionTarget.component(42, 102),
        )
        after = SelectionSnapshot.create(
            (
                SelectionTarget.scene_object(41),
                SelectionTarget.scene_object(42),
            ),
            owner_id="inspector",
            primary=SelectionTarget.scene_object(42),
        )
        service.apply_snapshot(before, record_history=False)
        calls = []

        class RemoveStub(UndoCommand):
            def __init__(self, component_id):
                super().__init__(f"Remove {component_id}")
                self.component_id = component_id

            def execute(self):
                calls.append(("remove", self.component_id))

            def undo(self):
                calls.append(("restore", self.component_id))

        command = RemoveComponentsCommand(
            [RemoveStub(101), RemoveStub(102)],
            before,
            after,
        )
        try:
            command.execute()
            assert calls == [("remove", 101), ("remove", 102)]
            assert service.snapshot == after

            command.undo()
            assert calls[-2:] == [("restore", 102), ("restore", 101)]
            assert service.snapshot == before

            command.redo()
            assert calls[-2:] == [("remove", 101), ("remove", 102)]
            assert service.snapshot == after
        finally:
            SelectionService._instance = previous


# ══════════════════════════════════════════════════════════════════════
# UndoCommand base
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# GlobalSelectionCommand
# ══════════════════════════════════════════════════════════════════════


class TestPrefabModeCommand:
    def test_redo_resolves_the_current_path_from_stable_guid(self, monkeypatch):
        calls = []
        paths = {"prefab-guid": "Assets/BeforeMove.prefab"}

        class _FakeSceneManager:
            _asset_database = types.SimpleNamespace(get_path_from_guid=lambda guid: paths.get(guid, ""))

            def open_prefab_mode(self, path, preserve_undo_history=False):
                calls.append(("open", path, preserve_undo_history))
                return True

            def _do_exit_prefab_mode(self, preserve_undo_history=False):
                calls.append(("exit", "", preserve_undo_history))
                return True

        fake_sfm = _FakeSceneManager()
        scene_manager_mod = types.ModuleType("infernux.engine.scene_manager")

        class _SceneFileManager:
            @staticmethod
            def instance():
                return fake_sfm

        scene_manager_mod.SceneFileManager = _SceneFileManager
        monkeypatch.setitem(sys.modules, "infernux.engine.scene_manager", scene_manager_mod)

        command = PrefabModeCommand("prefab-guid", enter_mode=True)
        command.execute()
        command.undo()
        paths["prefab-guid"] = "Assets/AfterMove.prefab"
        command.redo()

        assert calls == [
            ("open", "Assets/BeforeMove.prefab", True),
            ("exit", "", True),
            ("open", "Assets/AfterMove.prefab", True),
        ]

    def test_rejected_transition_raises_and_cannot_be_recorded(self, monkeypatch):
        class _FakeSceneManager:
            _asset_database = types.SimpleNamespace(
                get_path_from_guid=lambda guid: "Assets/test.prefab" if guid == "prefab-guid" else ""
            )

            @staticmethod
            def open_prefab_mode(_path, preserve_undo_history=False):
                del preserve_undo_history
                return False

        scene_manager_mod = types.ModuleType("infernux.engine.scene_manager")

        class _SceneFileManager:
            @staticmethod
            def instance():
                return _FakeSceneManager()

        scene_manager_mod.SceneFileManager = _SceneFileManager
        monkeypatch.setitem(sys.modules, "infernux.engine.scene_manager", scene_manager_mod)

        command = PrefabModeCommand("prefab-guid", enter_mode=True)
        with pytest.raises(RuntimeError, match="Enter Prefab Mode was rejected"):
            command.execute()


# ══════════════════════════════════════════════════════════════════════
# RenderStackFieldCommand
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# UndoManager — core operations
# ══════════════════════════════════════════════════════════════════════


class TestUndoManager:
    def test_execute_reports_failure_without_recording(self, _reset_undo_manager):
        mgr = _reset_undo_manager

        class FailingCommand(UndoCommand):
            def execute(self):
                raise RuntimeError("expected test failure")

            def undo(self):
                pass

        assert mgr.execute(FailingCommand("fail")) is False
        assert not mgr.can_undo
        with mgr.suppress():
            assert mgr.is_executing
        assert not mgr.is_executing

    def test_property_recording_suppression_keeps_structural_actions_replayable(self, _reset_undo_manager):
        manager, target = _reset_undo_manager, _Obj()

        class StructuralCommand(UndoCommand):
            _is_property_edit = False

            def execute(self):
                target.y = 42

            def undo(self):
                target.y = 0

        with manager.suppress_property_recording():
            assert manager.execute(SetPropertyCommand(target, "x", 0, 99))
            assert target.x == 99 and not manager.can_undo
            assert manager.execute(StructuralCommand("Create object"))
            assert target.y == 42 and manager.can_undo
        manager.undo()
        assert target.y == 0 and target.x == 99


# ══════════════════════════════════════════════════════════════════════
# UndoManager — save points belong to DocumentRegistry
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Typed scene-object selection operations
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Integration: Selection undo via UndoManager
# ══════════════════════════════════════════════════════════════════════


class TestSelectionUndoIntegration:
    @pytest.fixture(autouse=True)
    def _fresh_selection(self):
        old = SelectionService._instance
        self.sel = SelectionService()
        yield
        SelectionService._instance = old

    @staticmethod
    def _install_recording_listener():
        from infernux.engine import _bootstrap_selection as module

        selection = module.BootstrapSelectionMixin()
        selection.window_manager = None
        selection._present_selection_snapshot = lambda _snapshot, *, preserve_console_selection=False: None
        selection._prev_selection_snapshot = SelectionService.instance().snapshot
        SelectionService.instance().add_listener(selection._on_global_selection_changed)
        manager = UndoManager.instance()
        manager.set_context_hooks(
            lambda: EditorContextSnapshot(
                selection=SelectionService.instance().snapshot,
            ),
            lambda context, _phase: (
                SelectionService.instance().apply_snapshot(
                    context.selection,
                    reason="test_restore",
                    record_history=False,
                ),
                ContextRestoreStatus.READY,
            )[1],
        )
        return selection

    def test_chained_typed_selection_uses_one_global_cursor(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        self._install_recording_listener()
        service = SelectionService.instance()
        service.select(
            SelectionTarget.scene_object(1),
            owner_id="hierarchy",
            record_history=True,
        )
        service.select(
            SelectionTarget.scene_object(2),
            owner_id="scene_view",
            record_history=True,
        )
        assert len(mgr.action_journal.entries) == 2

        mgr.undo()
        assert service.snapshot.primary == SelectionTarget.scene_object(1)

        mgr.undo()
        assert service.snapshot == SelectionSnapshot()

        mgr.redo()
        assert service.snapshot.primary == SelectionTarget.scene_object(1)


# ══════════════════════════════════════════════════════════════════════
# Structural command selection context
# ══════════════════════════════════════════════════════════════════════

# The commands call _get_active_scene() which uses SceneManager (native).
# We patch that helper to return None so __init__ doesn't blow up, then
# poke internal fields to simulate the state we want to test.

_undo_mod_ref = _undo_mod  # keep reference for patching


class TestStructuralCommandSelectionContext:
    """Structural commands use the global action context for selection."""

    @pytest.fixture(autouse=True)
    def _patch_scene(self, monkeypatch):
        _patch_undo_modules(monkeypatch, "_get_active_scene", lambda: None)
        _patch_undo_modules(monkeypatch, "_bump_inspector_structure", lambda: None)
        _patch_undo_modules(monkeypatch, "_notify_gizmos_scene_changed", lambda: None)

    @pytest.fixture(autouse=True)
    def _fresh_sel(self):
        old = SelectionService._instance
        self.sel = SelectionService()
        yield
        SelectionService._instance = old

    def test_delete_is_one_action_and_context_restores_selection(
        self,
        monkeypatch,
        _reset_undo_manager,
    ):
        class _Transform:
            @staticmethod
            def get_sibling_index():
                return 0

        class _Object:
            id = 42
            transform = _Transform()

            @staticmethod
            def get_parent():
                return None

        class _Scene:
            def __init__(self):
                self.object = _Object()

            def find_by_id(self, object_id):
                return self.object if self.object and object_id == 42 else None

        scene = _Scene()
        _patch_world_undo(monkeypatch, scene)
        monkeypatch.setattr(_structural_mod, "_snapshot_object", lambda _obj: {"id": 42})
        monkeypatch.setattr(
            _structural_mod,
            "_destroy_game_object_immediately",
            lambda _scene, _obj: setattr(scene, "object", None),
        )

        service = SelectionService.instance()
        selected = SelectionSnapshot.create(
            (SelectionTarget.scene_object(42),),
            owner_id="hierarchy",
        )
        service.apply_snapshot(selected, record_history=False)
        manager = _reset_undo_manager
        restore_phases = []

        def _restore_context(context, phase):
            if phase == "undo_complete":
                assert scene.object is not None
            restore_phases.append(phase)
            service.apply_snapshot(context.selection, record_history=False)

        manager.set_context_hooks(
            lambda: EditorContextSnapshot(selection=service.snapshot),
            _restore_context,
        )

        assert manager.execute(DeleteGameObjectCommand(42, "Delete"))
        assert service.snapshot == SelectionSnapshot()
        assert len(manager.action_journal.entries) == 1

        def restore_object(*_args, **_kwargs):
            scene.object = _Object()
            return scene.object

        with _override_recreate_game_object(restore_object):
            manager.undo()
        assert service.snapshot == selected
        assert scene.object is not None
        assert len(manager.action_journal.entries) == 1
        assert restore_phases == ["prepare_undo", "undo_complete"]

        manager.redo()
        assert service.snapshot == SelectionSnapshot()
        assert scene.object is None
        assert len(manager.action_journal.entries) == 1

    def test_failed_delete_undo_keeps_cursor_and_selection(
        self,
        monkeypatch,
        _reset_undo_manager,
    ):
        class _Transform:
            @staticmethod
            def get_sibling_index():
                return 0

        class _Object:
            id = 42
            transform = _Transform()

            @staticmethod
            def get_parent():
                return None

        class _Scene:
            def __init__(self):
                self.object = _Object()

            def find_by_id(self, object_id):
                return self.object if self.object and object_id == 42 else None

        scene = _Scene()
        _patch_world_undo(monkeypatch, scene)
        monkeypatch.setattr(_structural_mod, "_snapshot_object", lambda _obj: {"id": 42})
        monkeypatch.setattr(
            _structural_mod,
            "_destroy_game_object_immediately",
            lambda _scene, _obj: setattr(scene, "object", None),
        )
        service = SelectionService.instance()
        selected = SelectionSnapshot.create((SelectionTarget.scene_object(42),), owner_id="hierarchy")
        service.apply_snapshot(selected, record_history=False)
        manager = _reset_undo_manager
        manager.set_context_hooks(
            lambda: EditorContextSnapshot(selection=service.snapshot),
            lambda context, _phase: service.apply_snapshot(context.selection, record_history=False),
        )

        assert manager.execute(DeleteGameObjectCommand(42, "Delete"))
        assert service.snapshot.is_empty
        assert manager.can_undo

        with _override_recreate_game_object(lambda *_args, **_kwargs: None):
            manager.undo()

        assert manager.can_undo
        assert not manager.can_redo
        assert service.snapshot.is_empty
        assert scene.object is None


class TestDeleteGameObjectsCommand:
    class _Transform:
        def __init__(self, sibling_index):
            self._sibling_index = sibling_index

        def get_sibling_index(self):
            return self._sibling_index

    class _Object:
        def __init__(self, object_id, sibling_index, parent=None):
            self.id = object_id
            self.transform = TestDeleteGameObjectsCommand._Transform(sibling_index)
            self._parent = parent

        def get_parent(self):
            return self._parent

    class _Scene:
        def __init__(self, objects):
            self.objects = {obj.id: obj for obj in objects}

        def find_by_id(self, object_id):
            return self.objects.get(object_id)

    def test_filters_selected_descendants_and_restores_sibling_order(self, monkeypatch):
        first = self._Object(10, 0)
        parent = self._Object(20, 1)
        child = self._Object(21, 0, parent)
        scene = self._Scene([first, parent, child])
        destroyed = []
        restored = []

        _patch_world_undo(monkeypatch, scene)
        monkeypatch.setattr(_structural_mod, "_snapshot_object", lambda obj: {"id": obj.id})
        monkeypatch.setattr(
            _structural_mod,
            "_destroy_game_object_immediately",
            lambda _scene, obj: destroyed.append(obj.id),
        )
        monkeypatch.setattr(_structural_mod, "_bump_inspector_structure", lambda: None)
        monkeypatch.setattr(_structural_mod, "_notify_gizmos_scene_changed", lambda: None)

        command = DeleteGameObjectsCommand([20, 21, 10])
        assert [entry["object_id"] for entry in command._entries] == [20, 10]

        command.execute()
        assert destroyed == [20, 10]

        def restore_object(document, parent_id, sibling_index, *, scene=None):
            assert scene is not None
            restored.append((document["id"], parent_id, sibling_index))
            return self._Object(document["id"], sibling_index)

        batches = []

        def restore_batch(entries):
            batches.append(entries)
            return [
                restore_object(document, parent_id, sibling_index, scene=owner)
                for document, parent_id, sibling_index, owner in entries
            ]

        monkeypatch.setattr(_recreate_mod, "_recreate_game_objects_from_documents", restore_batch)
        command.undo()
        assert len(batches) == 1
        assert restored == [(10, None, 0), (20, None, 1)]


class TestImmediateDestroyHelpers:
    @pytest.fixture(autouse=True)
    def _patch_editor_side_effects(self, monkeypatch):
        _patch_undo_modules(monkeypatch, "_bump_inspector_structure", lambda: None)
        _patch_undo_modules(monkeypatch, "_notify_gizmos_scene_changed", lambda: None)


def test_property_snapshot_is_owned_across_execute_undo_redo():
    obj = _Obj()
    before = {"items": [[1]], "label": "before", "optional": None}
    after = {"items": [[2]], "label": "after", "optional": None}
    command = SetPropertyCommand(obj, "x", before, after)
    assert command.marks_dirty and "x" in command.description
    before["items"][0].append(99)
    after["items"][0].append(99)
    command.execute()
    assert obj.x == {"items": [[2]], "label": "after", "optional": None}
    assert obj.x is not after
    command.undo()
    assert obj.x == {"items": [[1]], "label": "before", "optional": None}
    command.redo()
    assert obj.x == {"items": [[2]], "label": "after", "optional": None}


def test_generic_component_document_restore_owns_its_snapshots():
    component = _FakeComp()
    before, after = {"value": [1]}, {"value": [2]}
    command = GenericComponentCommand(component, before, after)
    before["value"].append(99)
    after["value"].append(99)
    command.execute()
    assert component._document == {"value": [2]}
    command.undo()
    assert component._document == {"value": [1]}
    command.redo()
    assert component._document == {"value": [2]}


def test_material_document_history_owns_snapshots_without_scene_dirtying():
    material = _FakeComp()
    material.guid = "test-guid"
    before, after = {"properties": {"x": [1]}}, {"properties": {"x": [2]}}
    command = MaterialDocumentCommand(material, before, after)
    assert command.marks_dirty is False
    before["properties"]["x"].append(99)
    after["properties"]["x"].append(99)
    command.execute()
    assert material._document == {"properties": {"x": [2]}}
    command.undo()
    assert material._document == {"properties": {"x": [1]}}
    command.redo()
    assert material._document == {"properties": {"x": [2]}}


def test_compound_property_edit_replays_in_forward_and_reverse_order():
    first, second = _Obj(), _Obj()
    calls = []

    class LoggedProperty(SetPropertyCommand):
        def execute(self):
            super().execute()
            calls.append(("execute", self.description))

        def undo(self):
            super().undo()
            calls.append(("undo", self.description))

    command = CompoundCommand(
        [
            LoggedProperty(first, "x", 0, 10, "First"),
            LoggedProperty(second, "x", 0, 20, "Second"),
        ]
    )
    assert command.supports_redo
    command.execute()
    assert (first.x, second.x) == (10, 20)
    assert calls == [("execute", "First"), ("execute", "Second")]
    calls.clear()
    command.undo()
    assert (first.x, second.x) == (0, 0)
    assert calls == [("undo", "Second"), ("undo", "First")]
    command.redo()
    assert (first.x, second.x) == (10, 20)


def test_renderstack_continuous_edit_merges_and_republishes_each_replay():
    target = _Obj()
    stack = _FakeStack(target)
    first = RenderStackFieldCommand(stack, target, "x", 0, 1)
    first.execute()
    second = RenderStackFieldCommand(stack, target, "x", 1, 5)
    second.timestamp = first.timestamp + 0.1
    second.execute()
    assert target.x == 5 and stack.invalidated == 2
    assert first.can_merge(second)
    first.merge(second)
    first.undo()
    assert target.x == 0 and stack.invalidated == 3
    first.redo()
    assert target.x == 5 and stack.invalidated == 4


def test_history_walks_distinct_edits_then_discards_a_redo_branch(_reset_undo_manager):
    manager, target = _reset_undo_manager, _Obj()
    changed = []
    manager.set_on_state_changed(lambda: changed.append(True))
    assert not manager.can_undo and not manager.can_redo
    first = SetPropertyCommand(target, "x", 0, 10, "First")
    first.timestamp = 10
    assert manager.execute(first)
    assert target.x == 10 and manager.undo_description == "First" and len(changed) == 1
    second = SetPropertyCommand(target, "x", 10, 20, "Second")
    second.timestamp = 20
    assert manager.execute(second)
    third = SetPropertyCommand(target, "x", 20, 30, "Third")
    third.timestamp = 30
    assert manager.execute(third)
    assert target.x == 30 and len(changed) == 3
    manager.undo()
    assert target.x == 20 and manager.can_redo and len(changed) == 4
    manager.undo()
    assert target.x == 10
    manager.redo()
    assert target.x == 20
    manager.undo()
    assert manager.execute(SetPropertyCommand(target, "y", 0, 20, "Move Y"))
    assert target.x == 10 and target.y == 20
    assert not manager.can_redo and manager.undo_description == "Move Y"
    manager.clear()
    assert not manager.can_undo and not manager.can_redo
