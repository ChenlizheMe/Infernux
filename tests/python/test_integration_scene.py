"""Integration tests — Scene management and GameObject hierarchy (real engine)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading as threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.components import (
    FieldType,
    InxComponent,
    SerializableObject,
    list_field,
    serialized_field,
)
from infernux.components._cds_bridge import get_class_info
from infernux.components.decorators import disallow_multiple, require_component
from infernux.components.ref_wrappers import ComponentRef, GameObjectRef
from infernux.engine.component_restore import (
    PythonComponentRestoreError as PythonComponentRestoreError,
    clone_game_object_transactionally,
    deserialize_scene_document_transactionally as deserialize_scene_document_transactionally,
    deserialize_game_object_document_transactionally,
    instantiate_game_object_document_transactionally as instantiate_game_object_document_transactionally,
    preflight_scene_python_components as preflight_scene_python_components,
    replace_scene_python_components_for_play as replace_scene_python_components_for_play,
)
from infernux.engine.scene_document_transaction import (
    SceneDocumentTransaction as SceneDocumentTransaction,
    SceneDocumentTransactionState as SceneDocumentTransactionState,
)
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.scene_authoring import encode_scene_document, encode_runtime_scene_artifact
from infernux.engine.prefab_manager import (
    PrefabDocumentError as PrefabDocumentError,
    _strip_prefab_runtime_fields as _strip_prefab_runtime_fields,
)
from infernux.instantiate import Instantiate
from infernux.lib import (
    AssetRegistry,
    GameObject,
    InxMaterial as InxMaterial,
    PrimitiveType,
    SceneManager,
    Vector3,
    quatf,
)


@pytest.fixture
def editor_history():
    from infernux.engine.undo import UndoManager

    previous = UndoManager._instance
    manager = UndoManager()
    try:
        yield manager
    finally:
        manager.clear()
        UndoManager._instance = previous


class _ExplodingSerializationComponent(InxComponent):
    def _serialize_fields_document(self) -> dict:
        raise RuntimeError("intentional serialization failure")


class _StrictSceneComponent(InxComponent):
    value: int = 7


class _ReplacementLifecycleSource(InxComponent):
    value: int = 13
    destroy_calls = 0

    def on_destroy(self):
        type(self).destroy_calls += 1


class _ReplacementLifecycleTarget(InxComponent):
    value: int = 0


class _PlayLifecycleResetComponent(InxComponent):
    awake_calls = 0
    start_calls = 0
    destroy_calls = 0
    active_instance = None

    def awake(self):
        type(self).awake_calls += 1
        if type(self).active_instance not in (None, self):
            raise RuntimeError("edit-domain singleton registration leaked into Play")
        type(self).active_instance = self
        self._runtime_token = "initialized-in-awake"

    def start(self):
        type(self).start_calls += 1

    def on_destroy(self):
        type(self).destroy_calls += 1
        if type(self).active_instance is self:
            type(self).active_instance = None


class _AdditiveSceneComponent(InxComponent):
    value: int = 7
    label: str = "default"


class _ExplodingAfterDeserializeComponent(InxComponent):
    value: int = 11

    def on_after_deserialize(self):
        raise RuntimeError("intentional on_after_deserialize failure")


class _PublishBarrierComponent(InxComponent):
    value: int = 0
    _events = []

    def on_after_deserialize(self):
        if self._awake_called:
            raise RuntimeError("Awake ran before on_after_deserialize")
        if self._cpp_component.component_id != self.component_id:
            raise RuntimeError("stable component ID was not installed before callback")
        scene = self.game_object.scene
        published_count = sum(len(obj.get_py_components()) for obj in scene.get_all_objects())
        if published_count != 2:
            raise RuntimeError("Python graph was not fully attached before callback")
        type(self)._events.append(("after", self.value))

    def awake(self):
        type(self)._events.append(("awake", self.value))

    def reset(self):
        type(self)._events.append(("reset", self.value))


def _native_record(object_document: dict, type_name: str) -> dict:
    type_id = f"native:infernux.{type_name}"
    return next(record for record in object_document["components"] if record["type_id"] == type_id)


def _python_records(object_document: dict) -> list[dict]:
    return [record for record in object_document["components"] if str(record.get("type_id", "")).startswith("python:")]


def _python_type_id(instance: InxComponent) -> str:
    component_type = type(instance)
    return (
        f"python:{instance._script_guid}:{component_type._get_type_guid()}:"
        f"{component_type.__module__}:{component_type.__qualname__}"
    )


@require_component("Rigidbody")
class _RequiresRigidbodyComponent(InxComponent):
    value: int = 1


@disallow_multiple
class _SingleInstanceSceneComponent(InxComponent):
    value: int = 1


def _cds_alive_count(component_type) -> int:
    class_info = get_class_info(component_type)
    assert class_info is not None
    from infernux import lib

    return lib._cds_alive_count(class_info[0])


class _ObjectRefSceneComponent(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)


class _ObjectGraphRefsComponent(InxComponent):
    target_object = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    target_component = serialized_field(default=None, field_type=FieldType.COMPONENT)


class _CameraCleanupSceneComponent(InxComponent):
    target_component = serialized_field(default=None, field_type=FieldType.COMPONENT)
    _cleanup_results = []

    def on_disable(self):
        camera = self.target_component
        if camera is None:
            type(self)._cleanup_results.append("missing")
            return
        try:
            camera.reset_view_matrix()
        except ReferenceError as exc:
            type(self)._cleanup_results.append(str(exc))
        else:
            type(self)._cleanup_results.append("valid")


class _CloneSettings(SerializableObject):
    gain: float = 1.0
    label: str = "default"


class _RichCloneComponent(InxComponent):
    count: int = 1
    title: str = "source"
    values: list = list_field(element_type=FieldType.INT, default=[])
    settings: _CloneSettings = serialized_field(default=_CloneSettings())
    target_object = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    target_component = serialized_field(default=None, field_type=FieldType.COMPONENT)


def _capture_exception(callback, errors):
    try:
        callback()
    except Exception as exc:
        errors.append(exc)


def test_scene_deleted_id_watermarks_survive_a_fresh_process(tmp_path):
    from infernux.lib import _Infernux

    path = tmp_path / "Deleted.scene"
    source = """
import json, sys
from pathlib import Path
from infernux.lib import SceneManager
manager = SceneManager.instance()
scene = manager.create_scene("IDs")
obj = scene.create_game_object("Removed")
component = obj.add_component("BoxCollider")
object_id, component_id = obj.id, component.component_id
scene.destroy_game_object(obj)
scene.process_pending_destroys()
document = scene.serialize_document()
assert document["nextObjectId"] > object_id
assert document["nextComponentId"] > component_id
Path(sys.argv[1]).write_text(json.dumps(document), encoding="utf-8")
"""
    restore = """
import json, sys
from pathlib import Path
from infernux.lib import SceneManager
document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manager = SceneManager.instance()
scene = manager.create_scene("Restored")
assert scene._commit_document(document)
obj = scene.create_game_object("New")
component = obj.add_component("BoxCollider")
assert obj.id >= document["nextObjectId"]
assert component.component_id >= document["nextComponentId"]
"""
    for script in (source, restore):
        environment = os.environ.copy()
        environment["INFERNUX_NATIVE_MODULE_DIR"] = str(Path(_Infernux.__file__).parent)
        environment["PYTHONPATH"] = str((Path(__file__).resolve().parents[2] / "python"))
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=environment,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_removed_component_watermark_is_persisted_without_saving_the_live_component(scene):
    obj = scene.create_game_object("ComponentIDs")
    component = obj.add_component("BoxCollider")
    removed_id = component.component_id
    assert obj.remove_component(component)
    assert scene.serialize_document()["nextComponentId"] > removed_id


class TestSceneLifecycle:
    def test_public_loaded_scene_directory_is_distinct_from_build_list(self, scene, monkeypatch):
        from infernux.scene import SceneManager as PublicSceneManager

        native = SceneManager.instance()
        loaded_before = native.scene_count
        additive = native.create_scene("AdditiveDirectory")
        try:
            monkeypatch.setattr(
                PublicSceneManager,
                "_load_build_list",
                staticmethod(
                    lambda: ["/project/pytest_scene.scene", "/project/AdditiveDirectory.scene", "/project/Unused.scene"]
                ),
            )
            assert PublicSceneManager.get_scene_count() == loaded_before + 1
            assert PublicSceneManager.get_scene_count_in_build_settings() == 3
            assert any(
                PublicSceneManager.get_scene_at(index) is scene for index in range(PublicSceneManager.get_scene_count())
            )
            assert PublicSceneManager.get_scene_at(loaded_before) is additive
            assert PublicSceneManager.get_scene_at(loaded_before + 1) is None
            assert PublicSceneManager.get_scene_by_name("AdditiveDirectory") is additive
            assert PublicSceneManager.get_scene_by_build_index(1) is additive

            PublicSceneManager.set_active_scene(additive)
            assert PublicSceneManager.get_active_scene() is additive
        finally:
            native.set_active_scene(scene)
            native.unload_scene(additive)

    def test_additive_transaction_remaps_object_component_and_python_references(self, scene, tmp_path, monkeypatch):
        from infernux.engine.scene_manager import SceneFileManager
        from infernux.scene import LoadSceneMode, SceneManager as PublicSceneManager

        native = SceneManager.instance()
        source_root = scene.create_game_object("AdditiveIdentityRoot")
        source_child = scene.create_game_object("AdditiveIdentityChild")
        source_child.set_parent(source_root)
        source_marker = source_child.add_py_component(_StrictSceneComponent())
        source_refs = _ObjectGraphRefsComponent()
        source_refs.target_object = GameObjectRef(source_child)
        source_refs.target_component = ComponentRef(
            go_id=source_child.id,
            component_type="_StrictSceneComponent",
        )
        source_root.add_py_component(source_refs)
        source_document = json.loads(json.dumps(scene.serialize_document()))
        source_document["name"] = "AdditiveIdentityCopy"
        scene_path = tmp_path / "AdditiveIdentityCopy.scene"
        scene_path.write_text(json.dumps(encode_scene_document(source_document)), encoding="utf-8")
        loaded_before = native.scene_count
        # This exercises the runtime/current-schema transaction path.  Other
        # editor tests may have installed a process-wide SceneFileManager,
        # whose authored-project boundary intentionally rejects this isolated
        # temporary scene outside Assets.
        monkeypatch.setattr(SceneFileManager, "_instance", None)
        monkeypatch.setattr(PublicSceneManager, "_runtime_scene_service", None)
        monkeypatch.setattr(PublicSceneManager, "_is_in_play_mode", staticmethod(lambda: False))
        monkeypatch.setattr(
            PublicSceneManager,
            "_load_build_list",
            staticmethod(lambda: [str(scene_path)]),
        )

        additive = None
        try:
            assert PublicSceneManager.load_scene("AdditiveIdentityCopy", LoadSceneMode.ADDITIVE) is True
            assert native.get_active_scene() is scene
            assert native.scene_count == loaded_before + 1
            additive = native.get_scene_at(loaded_before)

            copied_root = additive.find("AdditiveIdentityRoot")
            copied_child = additive.find("AdditiveIdentityChild")
            copied_marker = copied_child.get_py_component(_StrictSceneComponent)
            copied_refs = copied_root.get_py_component(_ObjectGraphRefsComponent)
            assert copied_root.id != source_root.id
            assert copied_child.id != source_child.id
            assert copied_marker.component_id != source_marker.component_id
            assert copied_refs.target_object is copied_child, (
                source_document["objects"][0]["components"],
                copied_refs._serialize_fields_document(),
            )
            assert copied_refs.target_component is copied_marker, copied_refs._serialize_fields_document()
            assert scene.find_by_id(source_root.id) is source_root
            assert scene.find_by_id(source_child.id) is source_child

            cross_scene_ref = _ObjectRefSceneComponent()
            cross_scene_ref.target = GameObjectRef(copied_child)
            source_root.add_py_component(cross_scene_ref)
            native.unload_scene(additive)
            additive = None
            assert cross_scene_ref.target is None
            assert scene.find_by_id(source_root.id) is source_root
            assert source_root.get_py_component(_ObjectRefSceneComponent) is cross_scene_ref
        finally:
            if additive is not None:
                native.unload_scene(additive)

    def test_player_service_additive_loads_cataloged_scene_into_running_world(self, scene, tmp_path):
        from infernux.engine.player_scene import PlayerSceneService
        from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog

        manager = SceneManager.instance()
        source = scene.create_game_object("PlayerWorldSource")
        document = json.loads(json.dumps(scene.serialize_document()))
        document["name"] = "PlayerCatalogAdditive"
        document["objects"][0]["name"] = "PlayerWorldAdditive"

        cooked_path = tmp_path / "Content" / "PlayerCatalogAdditive.scene"
        cooked_path.parent.mkdir(parents=True)
        cooked_path.write_text(json.dumps(encode_runtime_scene_artifact(document)), encoding="utf-8")
        artifact_id = "content:scene-player-additive"
        catalog = PlayerRuntimeAssetCatalog.from_documents(
            str(tmp_path),
            {
                "artifacts": [
                    {
                        "runtime_artifact_id": artifact_id,
                        "runtime_path": "Content/PlayerCatalogAdditive.scene",
                        "package": "Game_Data/Content.inxpkg",
                        "logical_type": "scene",
                        "asset_guid": "player-additive-scene-guid",
                        "dependencies": [],
                    }
                ]
            },
            {
                "entries": [
                    {
                        "guid": "player-additive-scene-guid",
                        "runtime_path": "Assets/Scenes/PlayerCatalogAdditive.scene",
                        "primary_runtime_artifact_id": artifact_id,
                        "runtime_artifact_ids": [artifact_id],
                        "dependencies": [],
                    }
                ]
            },
        )
        service = PlayerSceneService()
        service.bind_runtime_catalog(catalog)
        baseline_worlds = {int(manager.get_scene_at(index).world_id) for index in range(int(manager.scene_count))}
        additive = None
        manager.play()
        try:
            assert (
                service.request_prepared_load(
                    "player-additive-scene-guid",
                    mode="additive",
                )
                is True
            )
            deadline = time.monotonic() + 3.0
            while service.is_load_pending and time.monotonic() < deadline:
                service.process_pending_load()
                time.sleep(0.001)
            assert service.is_load_pending is False, service.last_error
            assert service.last_error == ""
            assert manager.get_active_scene() is scene
            assert scene.find("PlayerWorldSource") is source

            additive = next(
                manager.get_scene_at(index)
                for index in range(int(manager.scene_count))
                if int(manager.get_scene_at(index).world_id) not in baseline_worlds
            )
            assert additive.name == "PlayerCatalogAdditive"
            assert additive.find("PlayerWorldAdditive") is not None
        finally:
            service.cancel_pending_load()
            if manager.is_playing():
                manager.stop()
            if additive is not None:
                manager.unload_scene(additive)

    def test_world_queries_and_root_scene_move_preserve_identity(self, scene):
        from infernux.scene import GameObjectQuery, SceneManager as PublicSceneManager

        manager = SceneManager.instance()
        active_match = scene.create_game_object("SharedQueryName")
        active_match.tag = "WorldQueryTag"
        destination = manager.create_scene("MoveDestination")
        moved_root = scene.create_game_object("MovedAcrossScenes")
        moved_root.tag = "WorldQueryTag"
        moved_root.layer = 7
        moved_child = scene.create_game_object("MovedChild")
        moved_child.set_parent(moved_root)
        marker = moved_root.add_py_component(_StrictSceneComponent())
        original_id = moved_root.id
        original_component_id = marker.component_id
        original_handle = moved_root.handle
        try:
            assert GameObjectQuery.find("SharedQueryName") is active_match
            assert GameObjectQuery.find("MovedAcrossScenes") is moved_root
            assert GameObject.find("MovedAcrossScenes") is moved_root
            assert {obj.id for obj in GameObjectQuery.find_game_objects_with_tag("WorldQueryTag")} == {
                active_match.id,
                moved_root.id,
            }
            assert GameObjectQuery.find_game_objects_in_layer(7) == [moved_root]

            PublicSceneManager.move_game_object_to_scene(moved_root, destination)
            assert manager.get_active_scene() is scene
            assert scene.find_by_id(original_id) is None
            assert destination.find_by_id(original_id) is moved_root
            assert moved_root.handle.world_id == destination.world_id
            assert moved_root.handle != original_handle
            assert moved_root.get_py_component(_StrictSceneComponent) is marker
            assert marker.component_id == original_component_id
            assert GameObjectQuery.find_by_id(original_id) is moved_root

            with pytest.raises(ValueError, match="root GameObject"):
                PublicSceneManager.move_game_object_to_scene(moved_child, scene)

            PublicSceneManager.move_game_object_to_scene(moved_root, scene)
            assert scene.find_by_id(original_id) is moved_root
            assert destination.find_by_id(original_id) is None
            assert moved_root.get_py_component(_StrictSceneComponent) is marker
        finally:
            if moved_root.scene is destination:
                PublicSceneManager.move_game_object_to_scene(moved_root, scene)
            manager.unload_scene(destination)


class TestGameObject:
    def test_python_component_serialization_failure_aborts_entire_document(self, scene):
        game_object = scene.create_game_object("BrokenWriter")
        game_object.add_py_component(_ExplodingSerializationComponent())

        with pytest.raises(RuntimeError, match="intentional serialization failure"):
            game_object.serialize_document()
        with pytest.raises(RuntimeError, match="intentional serialization failure"):
            scene.serialize_document()

    def test_object_handles_reject_rebuilt_scene_lifetimes(self, scene):
        go = scene.create_game_object("HandleOwner")
        rigidbody = go.add_component("Rigidbody")
        reference = GameObjectRef(go)
        component_reference = ComponentRef(go_id=go.id, component_type="Rigidbody")
        game_object_handle = go.handle
        transform_handle = go.transform.handle
        component_handle = rigidbody.handle

        assert game_object_handle.is_valid
        assert game_object_handle.world_id == scene.world_id
        assert scene.resolve_game_object(game_object_handle) is go
        assert scene.resolve_component(transform_handle) is go.transform
        assert scene.resolve_component(component_handle).handle == component_handle
        assert component_reference.resolve().handle == component_handle

        document = scene.serialize_document()
        assert scene._commit_document(document) is True

        restored = scene.find_by_id(game_object_handle.id)
        assert restored is not None
        assert restored.handle.generation != game_object_handle.generation
        assert restored.transform.handle.generation != transform_handle.generation
        assert restored.get_component("Rigidbody").handle.generation != component_handle.generation
        assert scene.resolve_game_object(game_object_handle) is None
        assert scene.resolve_component(transform_handle) is None
        assert scene.resolve_component(component_handle) is None
        assert reference.resolve() is restored
        assert component_reference.resolve().handle == restored.get_component("Rigidbody").handle

    def test_object_handles_are_scoped_to_the_scene_world(self, scene):
        go = scene.create_game_object("WorldScoped")
        wrong_world = type(go.handle)(go.id, go.handle.generation, scene.world_id + 1)

        assert scene.resolve_game_object(wrong_world) is None

    def test_python_component_queries_follow_stable_type_identity_across_reload(self, scene):
        namespace = {"__module__": __name__, "__qualname__": "_ReloadedQueryComponent"}
        old_type = type("_ReloadedQueryComponent", (InxComponent,), dict(namespace))
        new_type = type("_ReloadedQueryComponent", (InxComponent,), dict(namespace))
        assert old_type is not new_type
        assert old_type._get_type_guid() == new_type._get_type_guid()

        parent = scene.create_game_object("ReloadedQueryParent")
        child = scene.create_game_object("ReloadedQueryChild")
        child.set_parent(parent)
        component = parent.add_py_component(old_type())

        assert parent.get_py_component(new_type) is component
        assert parent.get_component(new_type) is component
        assert parent.get_components(new_type) == [component]
        assert parent.get_component_in_children(new_type) is component
        assert child.get_component_in_parent(new_type) is component


class TestPrimitives:
    @staticmethod
    def _triangle_areas(positions, indices):
        areas = []
        for offset in range(0, len(indices), 3):
            a, b, c = (positions[indices[offset + corner]] for corner in range(3))
            ab = tuple(b[axis] - a[axis] for axis in range(3))
            ac = tuple(c[axis] - a[axis] for axis in range(3))
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            areas.append(0.5 * math.sqrt(sum(component * component for component in cross)))
        return areas


class TestInstantiate:
    def test_cloning_a_rich_object_graph_remaps_internal_references_and_retains_external_targets(self, scene):
        external = scene.create_game_object("ExternalTarget")
        external_component = external.add_py_component(_StrictSceneComponent())
        original = scene.create_game_object("RichCloneSource")
        original.add_py_component(_StrictSceneComponent()).value = 31
        child = scene.create_game_object("RichCloneChild")
        child.set_parent(original)
        child.add_py_component(_StrictSceneComponent())
        internal = original.add_py_component(_RichCloneComponent())
        internal.count, internal.title, internal.values = 17, "edited", [2, 3, 5, 8]
        internal.settings = _CloneSettings(gain=2.5, label="nested")
        internal.target_object = GameObjectRef(child)
        internal.target_component = ComponentRef(go_id=child.id, component_type="_StrictSceneComponent")
        outside = original.add_py_component(_RichCloneComponent())
        outside.target_object = GameObjectRef(external)
        outside.target_component = ComponentRef(go_id=external.id, component_type="_StrictSceneComponent")
        clone = clone_game_object_transactionally(scene, original)
        assert clone is not None and clone.get_py_component(_StrictSceneComponent).value == 31
        copied, retained = clone.get_components(_RichCloneComponent)
        assert (copied.count, copied.title, copied.values) == (17, "edited", [2, 3, 5, 8])
        assert copied.settings.gain == pytest.approx(2.5) and copied.settings.label == "nested"
        assert copied.target_object is clone.get_child(0)
        assert copied.target_component is clone.get_child(0).get_py_component(_StrictSceneComponent)
        assert retained.target_object is external and retained.target_component is external_component

    def test_unified_instantiate_for_game_object_ref_preserves_fields_and_parent(self, scene):
        parent = scene.create_game_object("InstantiateReferenceParent")
        original = scene.create_game_object("InstantiateReferenceSource")
        component = original.add_py_component(_RichCloneComponent())
        component.count = 610
        component.values = [3, 5, 8]
        component.settings = _CloneSettings(gain=1.25, label="reference")

        clone = Instantiate(GameObjectRef(original), parent=parent)

        assert clone is not None
        assert clone.get_parent() is parent
        restored = clone.get_py_component(_RichCloneComponent)
        assert restored.count == 610
        assert restored.values == [3, 5, 8]
        assert restored.settings.gain == pytest.approx(1.25)
        assert restored.settings.label == "reference"

    def test_game_object_instantiate_parent_overloads_follow_unity_transform_space(self, scene):
        source_parent = scene.create_game_object("InstantiateSourceParent")
        source_parent.transform.position = Vector3(10.0, 0.0, 0.0)
        source = scene.create_game_object("InstantiateTransformSource")
        source.set_parent(source_parent, False)
        source.transform.local_position = Vector3(2.0, 3.0, 4.0)

        target_parent = scene.create_game_object("InstantiateTargetParent")
        target_parent.transform.position = Vector3(20.0, 0.0, 0.0)

        detached = GameObject.instantiate(source)
        assert detached.get_parent() is None
        assert detached.transform.position.to_tuple() == pytest.approx((12.0, 3.0, 4.0))

        explicit_root = GameObject.instantiate(source, parent=None)
        assert explicit_root.get_parent() is None
        assert explicit_root.transform.position.to_tuple() == pytest.approx((2.0, 3.0, 4.0))

        unified_detached = Instantiate(source)
        assert unified_detached.get_parent() is None
        assert unified_detached.transform.position.to_tuple() == pytest.approx((12.0, 3.0, 4.0))

        local_clone = GameObject.instantiate(source, target_parent)
        assert local_clone.get_parent() is target_parent
        assert local_clone.transform.local_position.to_tuple() == pytest.approx((2.0, 3.0, 4.0))
        assert local_clone.transform.position.to_tuple() == pytest.approx((22.0, 3.0, 4.0))

        keyword_clone = GameObject.instantiate(source, parent=target_parent)
        assert keyword_clone.transform.local_position.to_tuple() == pytest.approx((2.0, 3.0, 4.0))

        world_clone = GameObject.instantiate(source, target_parent, True)
        assert world_clone.get_parent() is target_parent
        assert world_clone.transform.position.to_tuple() == pytest.approx((12.0, 3.0, 4.0))

        positioned = GameObject.instantiate(
            source,
            Vector3(30.0, 5.0, 6.0),
            quatf(),
            target_parent,
        )
        assert positioned.get_parent() is target_parent
        assert positioned.transform.position.to_tuple() == pytest.approx((30.0, 5.0, 6.0))

    def test_copying_connected_level_objects_preserves_rich_fields_and_all_reference_scopes(
        self, scene, editor_history
    ):
        from infernux.engine.interaction import ClipboardService, SceneObjectCommandService, SelectionService

        external = scene.create_game_object("ExternalClipboardTarget")
        external_component = external.add_py_component(_StrictSceneComponent())
        first = scene.create_game_object("ClipboardGroupFirst")
        child = scene.create_game_object("InternalChild")
        child.set_parent(first)
        child.add_py_component(_StrictSceneComponent())
        second = scene.create_game_object("ClipboardGroupSecond")
        second_component = second.add_py_component(_StrictSceneComponent())
        internal = first.add_py_component(_RichCloneComponent())
        internal.count, internal.title, internal.values = 29, "ctrl-cv", [34, 55]
        internal.settings = _CloneSettings(gain=6.5, label="hierarchy")
        internal.target_object = GameObjectRef(child)
        internal.target_component = ComponentRef(go_id=child.id, component_type="_StrictSceneComponent")
        cross_root = first.add_py_component(_RichCloneComponent())
        cross_root.target_object = GameObjectRef(second)
        cross_root.target_component = ComponentRef(go_id=second.id, component_type="_StrictSceneComponent")
        outside = first.add_py_component(_RichCloneComponent())
        outside.target_object = GameObjectRef(external)
        outside.target_component = ComponentRef(go_id=external.id, component_type="_StrictSceneComponent")
        selection = SelectionService.instance()
        selection.replace_scene_objects([first.id, second.id], owner_id="hierarchy", record_history=False)
        commands = SceneObjectCommandService(selection, ClipboardService())
        context = SimpleNamespace(
            selection=selection.snapshot, focus=SimpleNamespace(active_view_id="hierarchy", active_panel_id="hierarchy")
        )
        assert commands.copy(context, cut=False) and commands.paste(context)
        first_id, second_id = selection.scene_object_ids()
        copied_first, copied_second = scene.find_by_id(first_id), scene.find_by_id(second_id)
        inner, linked, retained = copied_first.get_components(_RichCloneComponent)
        assert (inner.count, inner.title, inner.values) == (29, "ctrl-cv", [34, 55])
        assert inner.settings.gain == pytest.approx(6.5) and inner.settings.label == "hierarchy"
        assert inner.target_object is copied_first.get_child(0)
        assert inner.target_component is copied_first.get_child(0).get_py_component(_StrictSceneComponent)
        assert linked.target_object is copied_second and linked.target_object is not second
        assert linked.target_component is copied_second.get_py_component(_StrictSceneComponent)
        assert linked.target_component is not second_component
        assert retained.target_object is external and retained.target_component is external_component

    @staticmethod
    def _transform_values(position, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        return {
            "position": position,
            "rotation": rotation,
            "scale": scale,
        }

    def test_component_add_transaction_rolls_back_failed_python_callback(self, scene):
        from infernux.engine.undo import AddComponentTransactionCommand, UndoManager

        owner = scene.create_game_object("FailedPythonAddTransaction")
        prototype = _ExplodingAfterDeserializeComponent()
        command = AddComponentTransactionCommand(
            owner.id,
            type(prototype).__name__,
            python_instance=prototype,
            invoke_after_deserialize=True,
        )
        previous_manager = UndoManager._instance
        manager = UndoManager()
        try:
            assert not manager.execute(command)
            assert owner.get_py_component(_ExplodingAfterDeserializeComponent) is None
            assert tuple(owner.get_component_order()) == ()
            assert not manager.can_undo
        finally:
            UndoManager._instance = previous_manager

    def test_repeated_asset_script_clone_preserves_fields_across_module_reload(self, scene, tmp_path):
        script_path = tmp_path / "pipe_probe.py"
        script_path.write_text(
            "from infernux.components import InxComponent\n\n"
            "class PipeProbe(InxComponent):\n"
            "    speed: float = 2.8\n"
            "    wrap_distance: float = 18.0\n",
            encoding="utf-8",
        )
        prefab_path = tmp_path / "pipe_probe.prefab"

        class AssetDatabase:
            @staticmethod
            def get_guid_from_path(path):
                identities = {
                    str(script_path): "pipe-probe-guid",
                    str(prefab_path): "pipe-probe-prefab-guid",
                }
                return identities.get(str(path), "")

            @staticmethod
            def get_path_from_guid(guid):
                identities = {
                    "pipe-probe-guid": str(script_path),
                    "pipe-probe-prefab-guid": str(prefab_path),
                }
                return identities.get(str(guid), "")

        from infernux.components.script_loader import load_and_create_component

        original = scene.create_game_object("AssetScriptCloneSource")
        source_component = load_and_create_component(
            str(script_path),
            asset_database=AssetDatabase(),
            type_name="PipeProbe",
            script_guid="pipe-probe-guid",
        )
        original.add_py_component(source_component)
        source_component.speed = 6.25
        source_component.wrap_distance = 42.0

        for _ in range(4):
            clone = GameObject.instantiate(original)
            restored_components = list(clone.get_py_components() or ())
            assert len(restored_components) == 1, [
                (type(item).__name__, getattr(item, "_broken_error", "")) for item in restored_components
            ]
            restored = restored_components[0]
            assert type(restored) is type(source_component)
            assert clone.get_component("PipeProbe") is restored, (
                type(restored).__name__,
                getattr(restored, "type_name", ""),
            )
            assert restored.speed == pytest.approx(6.25)
            assert restored.wrap_distance == pytest.approx(42.0)

        from infernux.engine.prefab_manager import (
            _PREFAB_TEMPLATE_CACHE,
            instantiate_prefab,
            save_prefab,
        )

        _PREFAB_TEMPLATE_CACHE.clear()
        assert save_prefab(original, str(prefab_path)) is True
        for _ in range(3):
            instance = instantiate_prefab(
                file_path=str(prefab_path),
                scene=scene,
                asset_database=AssetDatabase(),
            )
            restored = list(instance.get_py_components() or ())[0]
            assert restored.speed == pytest.approx(6.25)
            assert restored.wrap_distance == pytest.approx(42.0)

    def test_ui_prefab_drop_with_implicit_canvas_is_one_atomic_action(self, scene, tmp_path):
        from infernux.engine.interaction import (
            ClipboardService,
            EditorContextSnapshot,
            SceneObjectCommandService,
            SelectionService,
            SelectionTarget,
        )
        from infernux.engine.prefab_manager import save_prefab
        from infernux.engine.undo import UndoManager
        from infernux.ui import UICanvas

        source = scene.create_game_object("AtomicUIPrefab")
        prefab_path = tmp_path / "atomic_ui.prefab"
        assert save_prefab(
            source,
            str(prefab_path),
            asset_database=AssetRegistry.instance().get_asset_database(),
            source_canvas_name="HUD",
        )

        selection = SelectionService.instance()
        selection.select(
            SelectionTarget.scene_object(source.id),
            owner_id="hierarchy",
            record_history=False,
        )
        previous_manager = UndoManager.instance()
        manager = UndoManager()

        def restore_context(context, _phase):
            selection.apply_snapshot(context.selection, record_history=False)

        manager.set_context_hooks(
            lambda: EditorContextSnapshot(selection=selection.snapshot),
            restore_context,
        )
        commands = SceneObjectCommandService(
            selection,
            ClipboardService.instance(),
        )

        try:
            assert commands.instantiate_prefab(str(prefab_path), 0, False)

            assert len(manager.action_journal.entries) == 1
            assert manager.undo_description == "Instantiate Prefab"
            instance_id = selection.snapshot.primary.scene_object_id()
            instance = scene.find_by_id(instance_id)
            canvas = instance.get_parent()
            canvas_id = canvas.id
            assert canvas.name == "HUD"
            assert canvas.get_py_component(UICanvas) is not None

            manager.undo()
            assert scene.find_by_id(instance_id) is None
            assert scene.find_by_id(canvas_id) is None
            assert selection.snapshot.primary == SelectionTarget.scene_object(source.id)

            manager.redo()
            restored_canvas = scene.find_by_id(canvas_id)
            restored_instance = scene.find_by_id(instance_id)
            assert restored_canvas is not None
            assert restored_instance is not None
            assert restored_instance.get_parent() is restored_canvas
            assert selection.snapshot.primary == SelectionTarget.scene_object(instance_id)
        finally:
            manager.clear()
            UndoManager._instance = previous_manager

    def test_failed_ui_prefab_drop_rolls_back_implicit_canvas(self, scene, tmp_path, monkeypatch):
        from infernux.engine.interaction import (
            ClipboardService,
            SceneObjectCommandService,
            SelectionService,
            SelectionTarget,
        )
        from infernux.engine.prefab_manager import save_prefab
        from infernux.engine.undo import UndoManager

        source = scene.create_game_object("BrokenUIPrefab")
        prefab_path = tmp_path / "broken_ui.prefab"
        assert save_prefab(
            source,
            str(prefab_path),
            source_canvas_name="RollbackCanvas",
        )

        selection = SelectionService.instance()
        selection.select(
            SelectionTarget.scene_object(source.id),
            owner_id="hierarchy",
            record_history=False,
        )
        previous_manager = UndoManager.instance()
        manager = UndoManager()
        commands = SceneObjectCommandService(
            selection,
            ClipboardService.instance(),
        )

        def fail_instantiation(**_kwargs):
            raise RuntimeError("intentional prefab failure")

        monkeypatch.setattr(
            "infernux.engine.prefab_manager.instantiate_prefab",
            fail_instantiation,
        )
        try:
            assert commands.instantiate_prefab(str(prefab_path), 0, False) is False

            assert all(obj.name != "RollbackCanvas" for obj in scene.get_root_objects())
            assert len(manager.action_journal.entries) == 0
            assert selection.snapshot.primary == SelectionTarget.scene_object(source.id)
        finally:
            manager.clear()
            UndoManager._instance = previous_manager

    def test_structural_undo_restores_rich_python_fields(self, scene):
        original = scene.create_game_object("UndoRichSource")
        original_id = original.id
        component = original.add_py_component(_RichCloneComponent())
        component.count = 89
        component.title = "undo"
        component.values = [1, 1, 2, 3, 5]
        component.settings = _CloneSettings(gain=7.5, label="restored")

        from infernux.engine.undo import DeleteGameObjectCommand

        command = DeleteGameObjectCommand(original_id, "Delete rich source")
        command.execute()
        scene.process_pending_destroys()
        assert scene.find_by_id(original_id) is None

        command.undo()

        restored_object = scene.find_by_id(original_id)
        restored = restored_object.get_py_component(_RichCloneComponent)
        assert restored.count == 89
        assert restored.title == "undo"
        assert restored.values == [1, 1, 2, 3, 5]
        assert restored.settings.gain == pytest.approx(7.5)
        assert restored.settings.label == "restored"

    def test_structural_undo_keeps_additive_scene_ownership_after_active_scene_switch(self, scene, editor_history):
        from infernux.engine.undo import (
            CreateGameObjectCommand,
            DeleteGameObjectCommand,
        )

        manager = SceneManager.instance()
        additive = manager.create_scene("UndoAdditiveOwner")
        source_marker = scene.create_game_object("UndoSourceMarker")
        try:
            manager.set_active_scene(additive)
            created = additive.create_game_object("UndoAdditiveObject")
            object_id = int(created.id)
            editor_history.record(CreateGameObjectCommand(object_id, "Create additive object"))

            manager.set_active_scene(scene)
            editor_history.undo()
            assert additive.find_by_id(object_id) is None
            assert scene.find_by_id(source_marker.id) is source_marker

            editor_history.redo()
            assert additive.find_by_id(object_id) is not None
            assert scene.find_by_id(object_id) is None

            assert editor_history.execute(DeleteGameObjectCommand(object_id, "Delete additive object"))
            assert additive.find_by_id(object_id) is None
            editor_history.undo()
            assert additive.find_by_id(object_id) is not None
            assert manager.get_active_scene() is scene
        finally:
            manager.set_active_scene(scene)
            manager.unload_scene(additive)

    def test_hierarchy_move_transfers_ownership_between_loaded_scenes_and_undoes(self, scene, editor_history):
        from infernux.engine.interaction import (
            ClipboardService,
            SceneObjectCommandService,
            SelectionService,
        )

        manager = SceneManager.instance()
        additive = manager.create_scene("CrossSceneDestination")
        destination_parent = additive.create_game_object("DestinationParent")
        source_parent = scene.create_game_object("SourceParent")
        moved = scene.create_game_object("MovedBetweenScenes")
        moved.set_parent(source_parent)
        moved.transform.set_sibling_index(0)
        commands = SceneObjectCommandService(SelectionService(), ClipboardService())
        try:
            assert (
                commands.move_hierarchy(
                    [moved.id],
                    "parent",
                    destination_parent.id,
                    False,
                    additive.world_id,
                )
                is True
            )
            assert moved.scene is additive
            assert moved.get_parent() is destination_parent

            editor_history.undo()
            assert moved.scene is scene
            assert moved.get_parent() is source_parent

            editor_history.redo()
            assert moved.scene is additive
            assert moved.get_parent() is destination_parent
        finally:
            if moved.scene is additive:
                moved.set_parent(None)
                manager.move_game_object_to_scene(moved, scene)
            manager.set_active_scene(scene)
            manager.unload_scene(additive)

    def test_additive_scene_document_owns_dirty_revision_and_save_target(
        self,
        scene,
        tmp_path,
        monkeypatch,
    ):
        from infernux.engine.interaction import DocumentActionStatus, DocumentRegistry
        from infernux.engine.project_context import get_project_root, set_project_root
        from infernux.engine.undo import SetPropertyCommand, UndoManager

        previous_root = get_project_root()
        previous_manager = SceneFileManager._instance
        previous_undo = UndoManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        native = SceneManager.instance()
        additive = native.create_scene("Additive")
        additive_object = additive.create_game_object("AdditiveObject")

        try:
            set_project_root(str(project_root))
            manager = SceneFileManager()

            class _AssetDatabase:
                @staticmethod
                def get_guid_from_path(path):
                    paths = {
                        str(primary_path): "primary-scene-guid",
                        str(additive_path): "additive-scene-guid",
                    }
                    return paths.get(str(path), "")

                @staticmethod
                def get_path_from_guid(guid):
                    paths = {
                        "primary-scene-guid": str(primary_path),
                        "additive-scene-guid": str(additive_path),
                    }
                    return paths.get(str(guid), "")

            manager.set_asset_database(_AssetDatabase())
            monkeypatch.setattr(manager, "_save_camera_state", lambda _path: None)
            monkeypatch.setattr(manager, "_remember_last_scene", lambda _path: None)
            manager._current_scene_path = str(primary_path)
            manager._replace_scene_document(
                kind="scene",
                resource_path=str(primary_path),
                title="Primary",
                dirty=False,
            )
            primary_document_id = manager.document_id
            additive_document_id = manager.register_loaded_scene(
                additive,
                str(additive_path),
            )
            assert additive_document_id != primary_document_id

            assert manager.activate_loaded_scene(additive)
            undo = UndoManager()
            assert undo.execute(
                SetPropertyCommand(
                    additive_object,
                    "name",
                    "AdditiveObject",
                    "EditedAdditiveObject",
                )
            )

            registry = DocumentRegistry.instance()
            assert registry.require(additive_document_id).is_dirty
            assert not registry.require(primary_document_id).is_dirty

            assert manager.activate_loaded_scene(scene)
            result = registry.request_save(additive_document_id)
            assert result.status is DocumentActionStatus.APPLIED
            saved = json.loads(additive_path.read_text(encoding="utf-8"))
            assert saved["name"] == "Additive"
            assert [entry["name"] for entry in saved["objects"]] == ["EditedAdditiveObject"]
            assert native.get_active_scene() is scene

            undo.undo()
            assert additive_object.name == "AdditiveObject"
            assert native.get_active_scene() is scene
        finally:
            native.set_active_scene(scene)
            native.unload_scene(additive)
            SceneFileManager._instance = previous_manager
            UndoManager._instance = previous_undo
            set_project_root(previous_root)

    @pytest.mark.parametrize("immediate", (False, True))
    def test_additive_scene_open_undo_unloads_and_redo_reloads_without_placeholder(
        self,
        scene,
        tmp_path,
        monkeypatch,
        immediate,
    ):
        from infernux.engine.interaction import EditorInteractionCore
        from infernux.engine.project_context import get_project_root, set_project_root
        from infernux.engine.undo import UndoManager

        previous_root = get_project_root()
        previous_core = EditorInteractionCore._instance
        previous_manager = SceneFileManager._instance
        previous_undo = UndoManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        additive_path = assets / "UndoableAdditive.scene"
        document = scene.serialize_document()
        document["name"] = "UndoableAdditive"
        document["objects"] = []
        additive_path.write_text(json.dumps(encode_scene_document(document)), encoding="utf-8")

        core = EditorInteractionCore()
        manager = SceneFileManager()
        undo = UndoManager(core.action_journal)
        baseline_scene_count = int(SceneManager.instance().scene_count)

        class _AssetDatabase:
            @staticmethod
            def get_guid_from_path(path):
                return "undoable-additive-guid" if str(path) == str(additive_path.resolve()) else ""

            @staticmethod
            def get_path_from_guid(guid):
                return str(additive_path.resolve()) if guid == "undoable-additive-guid" else ""

        def drain_additive_load():
            deadline = time.monotonic() + 3.0
            while manager.is_loading and time.monotonic() < deadline:
                manager.poll_deferred_load()
                time.sleep(0.001)
            assert not manager.is_loading

        try:
            set_project_root(str(project_root))
            manager.set_asset_database(_AssetDatabase())
            monkeypatch.setattr(manager, "_prepare_native_scene_swap", lambda: None)
            if immediate:
                assert manager.load_scene_additive_immediate(str(additive_path)) is True
            else:
                assert manager.open_scene_additive(str(additive_path)) is True
                drain_additive_load()

            native = SceneManager.instance()
            loaded = tuple(native.get_scene_at(index) for index in range(int(native.scene_count)))
            additive = next(item for item in loaded if item.name == "UndoableAdditive")
            additive_world_id = int(additive.world_id)
            assert undo.undo_description == "Open Scene UndoableAdditive Additively"

            undo.undo()
            manager.poll_deferred_load()
            assert native.get_scene_by_world_id(additive_world_id) is None
            assert int(native.scene_count) == baseline_scene_count

            undo.redo()
            drain_additive_load()
            assert int(native.scene_count) == baseline_scene_count + 1
            assert native.get_active_scene().name == "UndoableAdditive"
            assert all(native.get_scene_at(index).name for index in range(int(native.scene_count)))
        finally:
            native = SceneManager.instance()
            native.set_active_scene(scene)
            for index in range(int(native.scene_count) - 1, -1, -1):
                candidate = native.get_scene_at(index)
                if candidate is not None and candidate is not scene:
                    native.unload_scene(candidate)
            undo.shutdown()
            core.shutdown()
            EditorInteractionCore._instance = previous_core
            SceneFileManager._instance = previous_manager
            UndoManager._instance = previous_undo
            set_project_root(previous_root)

    def test_additive_unload_rebinds_stale_editor_document_before_retiring_target(
        self,
        scene,
        tmp_path,
    ):
        """A queued Header activation must not leave views on a retired document."""
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.project_context import get_project_root, set_project_root

        previous_root = get_project_root()
        previous_manager = SceneFileManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        primary_path.write_text("{}", encoding="utf-8")
        additive_path.write_text("{}", encoding="utf-8")

        manager = SceneFileManager()
        native = SceneManager.instance()
        additive = native.create_scene("Additive")
        additive_world_id = int(additive.world_id)
        registry = DocumentRegistry.instance()
        callbacks: list[str] = []

        try:
            set_project_root(str(project_root))
            primary_document = manager.register_loaded_scene(scene, str(primary_path))
            additive_document = manager.register_loaded_scene(additive, str(additive_path))
            native.set_active_scene(scene)

            # Reproduce the one-frame divergence caused by an activation
            # immediately followed by Ctrl+Z/unload.
            manager._scene_document_id = additive_document
            manager._current_scene_path = str(additive_path)

            def observe_scene_change():
                callbacks.append(manager.document_id)
                registry.require(manager.document_id)

            manager.set_on_scene_changed(observe_scene_change)
            assert manager.request_unload_scene(additive)
            manager.poll_deferred_load()

            assert native.get_active_scene() is scene
            assert manager.document_id == primary_document
            assert manager.scene_for_document(additive_document) is None
            assert callbacks
            assert all(identifier == primary_document for identifier in callbacks)
        finally:
            native.set_active_scene(scene)
            if native.get_scene_by_world_id(additive_world_id) is not None:
                native.unload_scene(additive)
            SceneFileManager._instance = previous_manager
            set_project_root(previous_root)

    def test_single_scene_open_resolves_every_dirty_resident_scene_document(self, scene, tmp_path, monkeypatch):
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.project_context import get_project_root, set_project_root
        from infernux.engine.ui.dirty_panel_confirmation import (
            DirtyPanelConfirmationCoordinator,
        )

        previous_root = get_project_root()
        previous_manager = SceneFileManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        replacement_path = assets / "Replacement.scene"
        additive = SceneManager.instance().create_scene("DirtyAdditive")

        class _Confirmation:
            document_ids = ()

            def request_documents_replace(self, document_ids, on_complete, on_cancel=None, **_kwargs):
                del on_complete, on_cancel
                self.document_ids = tuple(document_ids)
                return True

        confirmation = _Confirmation()
        try:
            set_project_root(str(project_root))
            manager = SceneFileManager()
            monkeypatch.setattr(manager, "_save_camera_state", lambda _path: None)
            manager._current_scene_path = str(primary_path)
            manager._replace_scene_document(
                kind="scene",
                resource_path=str(primary_path),
                title="Primary",
                dirty=False,
            )
            primary_document_id = manager.document_id
            additive_document_id = manager.register_loaded_scene(
                additive,
                str(additive_path),
            )
            registry = DocumentRegistry.instance()
            registry.mark_changed(primary_document_id)
            registry.mark_changed(additive_document_id)
            monkeypatch.setattr(
                DirtyPanelConfirmationCoordinator,
                "instance",
                classmethod(lambda _cls: confirmation),
            )

            assert manager._continue_open_scene(str(replacement_path)) is False
            assert confirmation.document_ids == (
                primary_document_id,
                additive_document_id,
            )
            assert manager._deferred_load_path is None
        finally:
            native = SceneManager.instance()
            native.set_active_scene(scene)
            native.unload_scene(additive)
            SceneFileManager._instance = previous_manager
            set_project_root(previous_root)

    def test_single_scene_commit_unloads_every_other_resident_scene(self, scene, tmp_path, monkeypatch):
        from infernux.engine.project_context import get_project_root, set_project_root

        previous_root = get_project_root()
        previous_manager = SceneFileManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        replacement_path = assets / "Replacement.scene"
        replacement_path.write_text(
            json.dumps({"identity_format": "guid-v1", "name": "Replacement", "isPlaying": False, "objects": []}),
            encoding="utf-8",
        )
        native = SceneManager.instance()
        additive = native.create_scene("RetiredBySingle")

        try:
            set_project_root(str(project_root))
            manager = SceneFileManager()
            monkeypatch.setattr(manager, "_prepare_native_scene_swap", lambda: None)
            monkeypatch.setattr(manager, "_restore_camera_state", lambda _path: None)
            monkeypatch.setattr(manager, "_remember_last_scene", lambda _path: None)
            monkeypatch.setattr(manager, "sync_all_prefab_instances", lambda _scene: None)
            manager._current_scene_path = str(primary_path)
            manager._replace_scene_document(
                kind="scene",
                resource_path=str(primary_path),
                title="Primary",
                dirty=False,
            )
            additive_document_id = manager.register_loaded_scene(
                additive,
                str(additive_path),
            )
            additive_world_id = int(additive.world_id)

            assert manager._do_open_scene(str(replacement_path)) is True

            assert native.scene_count == 1
            assert native.get_active_scene() is scene
            assert native.get_scene_by_world_id(additive_world_id) is None
            assert manager.scene_for_document(additive_document_id) is None
            assert manager.current_scene_path == str(replacement_path.resolve())
        finally:
            SceneFileManager._instance = previous_manager
            set_project_root(previous_root)


def test_scene_hierarchy_edit_query_and_destroy_workflow(scene):
    assert SceneManager.instance().get_active_scene() is scene
    assert not scene.get_all_objects() and not scene.get_root_objects()
    parent = scene.create_game_object("Parent")
    child = scene.create_game_object("Child")
    assert parent.active and child.active and parent.id != child.id
    assert scene.find("Parent") is parent and scene.find_by_id(child.id) is child
    assert scene.find("MissingObject") is None
    child.set_parent(parent)
    assert child.get_parent() is parent and parent.get_children() == [child]
    assert scene.get_root_objects() == [parent]
    assert {obj.id for obj in scene.get_all_objects()} == {parent.id, child.id}
    parent.transform.position = Vector3(10, 0, 0)
    child.transform.local_position = Vector3(0, 5, 0)
    assert tuple(child.transform.position) == pytest.approx((10, 5, 0))
    child.set_parent(None, True)
    assert child.get_parent() is None
    assert tuple(child.transform.position) == pytest.approx((10, 5, 0))
    child.active = False
    assert not child.active
    child.active = True
    assert child.active
    scene.destroy_game_object(parent)
    scene.process_pending_destroys()
    assert scene.find("Parent") is None and scene.find("Child") is child
    scene.destroy_game_object(child)
    scene.process_pending_destroys()
    assert not scene.get_all_objects()


def test_transform_authoring_value_assignment_and_document_restore(scene):
    owner = scene.create_game_object("TransformAuthoring")
    assert tuple(owner.transform.position) == (0, 0, 0)
    owner.transform.position = Vector3(1, 2, 3)
    position = owner.transform.position
    position.x += 4
    position.y = 9
    owner.transform.position = position
    assert tuple(owner.transform.position) == pytest.approx((5, 9, 3))
    owner.transform.position += Vector3(2, 3, 4)
    owner.transform.local_scale = Vector3(2, 3, 4)
    owner.transform.euler_angles = Vector3(0, 90, 0)
    document = owner.serialize_document()
    owner.transform.position = Vector3(0, 0, 0)
    owner.transform.local_scale = Vector3(1, 1, 1)
    assert deserialize_game_object_document_transactionally(owner, document)
    assert tuple(owner.transform.position) == pytest.approx((7, 12, 7))
    assert tuple(owner.transform.local_scale) == pytest.approx((2, 3, 4))
    assert owner.transform.euler_angles.y == pytest.approx(90, abs=0.5)
