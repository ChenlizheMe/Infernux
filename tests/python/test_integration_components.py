"""Integration tests — Components, rendering objects, assets (real engine)."""

from __future__ import annotations

import importlib
import gc
import json
import sys
import time
from pathlib import Path

import pytest

from infernux.components import InxComponent, serialized_field, FieldType
from infernux.components.ref_wrappers import ComponentRef
from infernux.components.value_document import make_game_object_ref
from infernux.components.builtin import BoxCollider as BoxColliderComponent
from infernux.components.builtin import Camera as CameraComponent
from infernux.components.builtin import PhysicsMaterialCombine
from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.renderstack.render_stack import RenderStack
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline
from infernux.renderstack.effect_slot import EffectSlot
from infernux.core.asset_ref import RenderEffectRef

from infernux.lib import (
    SceneManager,
    Vector3,
    PrimitiveType,
    TextureLoader,
    InxMaterial,
    InxPhysicMaterial,
    LightType,
    LightShadows,
    LightColorMode,
    Physics,
    AssetRegistry,
    ResourceType,
)


def test_explicit_skinned_animation_seek_marks_temporal_discontinuity(scene):
    renderer = scene.create_game_object("AnimationSeekProbe").add_component("SkinnedMeshRenderer")
    native_renderer = renderer._cpp_component
    revision = scene.temporal_discontinuity_revision

    native_renderer.runtime_animation_time = 1.25
    assert scene.temporal_discontinuity_revision == revision + 1

    native_renderer.runtime_animation_time = 1.25
    assert scene.temporal_discontinuity_revision == revision + 1


def test_mesh_cpu_payload_prepares_on_worker_and_rejects_stale_publish(engine):
    registry = AssetRegistry.instance()
    asset_database = registry.get_asset_database()
    source = Path(asset_database.assets_root) / "async-cpu-artifact.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="ascii",
    )
    guid = asset_database.import_asset(str(source)).guid
    assert guid
    artifact = Path(asset_database.get_runtime_artifact_path(guid, ResourceType.Mesh))
    skin_artifact = (
        Path(asset_database.assets_root).parent / "Library" / "Artifacts" / "SkinnedMesh" / f"{guid}.inxskin"
    )
    assert artifact.is_file()
    assert artifact.read_bytes().startswith(b"INXMESH")
    assert skin_artifact.read_bytes().startswith(b"INXSKIN")
    asset_database.flush_derived_index()
    index_document = json.loads(Path(asset_database.asset_index_path).read_text(encoding="utf-8"))
    indexed = next(item for item in index_document["entries"] if item["guid"] == guid)
    assert indexed["artifact_path"] == f"Library/Artifacts/Mesh/{guid}.inxmesh"

    try:
        source.write_text("this source is intentionally invalid\n", encoding="ascii")
        assert registry.get_asset_version(guid) == 0
        ticket = registry.begin_load_mesh_by_guid(guid)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not registry.try_commit_asset_load(ticket):
            time.sleep(0.001)
        assert ticket.committed is True
        assert ticket.produced_on_worker is True
        mesh = registry.get_mesh(guid)
        assert mesh is not None
        assert mesh.vertex_count == 3
        assert mesh.has_skinned_data is False
        assert registry.get_asset_version(guid) == 1
        runtime_record = next(record for record in engine.asset_runtime_records if record.guid == guid)
        assert runtime_record.runtime_version == 1
        assert runtime_record.cpu_resident is True
        assert runtime_record.cpu_bytes > 0
        assert runtime_record.gpu_resident_bytes == 0
        assert runtime_record.gpu_version_synchronized is True

        registry.invalidate_asset(guid)
        source.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="ascii",
        )
        artifact.write_bytes(b"corrupt derived mesh")
        fallback = registry.begin_load_mesh_by_guid(guid)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not registry.try_commit_asset_load(fallback):
            time.sleep(0.001)
        assert fallback.committed is True
        assert registry.get_mesh(guid).vertex_count == 3
        assert registry.get_asset_version(guid) == 2

        registry.invalidate_asset(guid)
        assert asset_database.reimport_asset(str(source))
        asset_database.flush_derived_index()
        artifact.unlink()
        asset_database.refresh()
        assert artifact.is_file()
        assert source.resolve() in {Path(path).resolve() for path in asset_database.last_refresh_imported_paths}

        stale = registry.begin_load_mesh_by_guid(guid)
        registry.invalidate_asset(guid)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not stale.complete:
            time.sleep(0.001)
        assert stale.complete is True
        assert stale.produced_on_worker is True
        with pytest.raises(RuntimeError, match="stale"):
            registry.try_commit_asset_load(stale)
        assert registry.is_loaded(guid) is False
        assert registry.get_asset_version(guid) == 2
        runtime_record = next(record for record in engine.asset_runtime_records if record.guid == guid)
        assert runtime_record.runtime_version == 2
        assert runtime_record.cpu_resident is False
        assert runtime_record.cpu_bytes == 0
    finally:
        registry.invalidate_asset(guid)
        if asset_database.contains_path(str(source)):
            asset_database.delete_asset(str(source))
        assert artifact.exists() is False
        assert skin_artifact.exists() is False
        source.unlink(missing_ok=True)
        Path(f"{source}.meta").unlink(missing_ok=True)


def test_skinned_mesh_companion_artifact_is_atomic_worker_loaded_and_rebuilt(engine, scene):
    registry = AssetRegistry.instance()
    asset_database = registry.get_asset_database()
    fixture = (
        Path(__file__).resolve().parents[2]
        / "external"
        / "assimp"
        / "test"
        / "models"
        / "FBX"
        / "animation_with_skeleton.fbx"
    )
    original_bytes = fixture.read_bytes()
    source = Path(asset_database.assets_root) / "skinned-artifact-probe.fbx"
    source.write_bytes(original_bytes)
    guid = asset_database.import_asset(str(source)).guid
    assert guid
    mesh_artifact = Path(asset_database.get_runtime_artifact_path(guid, ResourceType.Mesh))
    skin_artifact = (
        Path(asset_database.assets_root).parent / "Library" / "Artifacts" / "SkinnedMesh" / f"{guid}.inxskin"
    )
    assert mesh_artifact.read_bytes().startswith(b"INXMESH")
    assert skin_artifact.read_bytes().startswith(b"INXSKIN")
    metadata = asset_database.get_meta_by_guid(guid)
    assert metadata.get_int("bone_count") > 0
    assert metadata.get_int("animation_count") > 0

    def load_on_worker():
        ticket = registry.begin_load_mesh_by_guid(guid)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not registry.try_commit_asset_load(ticket):
            time.sleep(0.001)
        assert ticket.committed is True
        assert ticket.produced_on_worker is True
        loaded = registry.get_mesh(guid)
        assert loaded is not None
        assert loaded.has_skinned_data is True
        assert loaded.skinned_bone_count > 0
        assert loaded.skinned_animation_count > 0
        assert loaded.skinned_animation_names
        return loaded

    try:
        source.write_bytes(b"invalid source: artifact load must not parse this")
        mesh = load_on_worker()
        assert mesh.vertex_count > 0

        registry.invalidate_asset(guid)
        source.write_bytes(original_bytes)
        skin_artifact.write_bytes(b"corrupt skinned companion")
        fallback = load_on_worker()
        assert fallback.has_skinned_data is True

        registry.invalidate_asset(guid)
        assert asset_database.reimport_asset(str(source))
        assert skin_artifact.is_file()
        skin_artifact.unlink()
        asset_database.refresh()
        assert skin_artifact.is_file()
        assert source.resolve() in {Path(path).resolve() for path in asset_database.last_refresh_imported_paths}

        renderer = scene.create_game_object("SkinnedArtifactProbe").add_component("SkinnedMeshRenderer")
        renderer.set_source_model_guid(guid)
        assert renderer.source_model_guid == guid
        assert not hasattr(renderer, "source_model_path")
        assert renderer.animation_take_count > 0
        takes = renderer.get_animation_take_names()
        assert len(takes) == renderer.animation_take_count
        for seconds in (0.0, 0.1, 0.2):
            renderer.submit_animation_pose(takes[0], seconds, 0.0)
        assert renderer.get_animation_take_names() == takes
        renderer.set_source_model_guid("")
        assert not renderer.has_animation_takes
        assert renderer.animation_take_count == 0
        assert renderer.get_animation_take_names() == []
        renderer.set_source_model_guid(guid)
        assert renderer.get_animation_take_names() == takes
        document = renderer.serialize_document()
        assert document["meshAssetGuid"] == guid
        assert "sourceModelGuid" not in document
        assert "sourceModelPath" not in document
        assert "animationTakeNames" not in document
    finally:
        registry.invalidate_asset(guid)
        if asset_database.contains_path(str(source)):
            asset_database.delete_asset(str(source))
        assert mesh_artifact.exists() is False
        assert skin_artifact.exists() is False
        source.unlink(missing_ok=True)
        Path(f"{source}.meta").unlink(missing_ok=True)


def test_texture_cpu_artifact_prepares_on_worker_and_validates_cache(engine):
    registry = AssetRegistry.instance()
    asset_database = registry.get_asset_database()
    source = Path(asset_database.assets_root) / "async-texture-artifact.ppm"
    source_bytes = b"P6\n4 2\n255\n" + bytes(
        (
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            255,
            255,
            255,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            0,
            255,
            32,
            64,
            128,
        )
    )
    source.write_bytes(source_bytes)
    guid = asset_database.import_asset(str(source)).guid
    assert guid
    artifact = Path(asset_database.get_runtime_artifact_path(guid, ResourceType.Texture))
    assert artifact.is_file()
    assert artifact.read_bytes().startswith(b"INXTEX")

    try:
        source.write_bytes(b"invalid texture source")
        ticket = registry.begin_load_texture_by_guid(guid)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not registry.try_commit_asset_load(ticket):
            time.sleep(0.001)
        assert ticket.committed is True
        assert ticket.produced_on_worker is True
        texture = registry.get_texture_asset(guid)
        assert texture is not None
        assert texture.pixel_width == 4
        assert texture.pixel_height == 2
        assert texture.mip_count == 3
        assert not hasattr(texture, "cpu_byte_size")
        assert texture.pixel_storage == "block_compressed"
        assert texture.pixel_format == "bc1_rgba_srgb"

        registry.invalidate_asset(guid)
        source.write_bytes(source_bytes)
        artifact.write_bytes(b"corrupt texture artifact")
        rejected = registry.begin_load_texture_by_guid(guid)
        deadline = time.monotonic() + 10.0
        with pytest.raises((RuntimeError, ValueError), match="texture artifact"):
            while time.monotonic() < deadline and not registry.try_commit_asset_load(rejected):
                time.sleep(0.001)
    finally:
        registry.invalidate_asset(guid)
        if asset_database.contains_path(str(source)):
            asset_database.delete_asset(str(source))
        assert artifact.exists() is False
        source.unlink(missing_ok=True)
        Path(f"{source}.meta").unlink(missing_ok=True)


def test_asset_registry_cpu_residency_budget_respects_live_and_explicit_pins(engine):
    registry = AssetRegistry.instance()
    asset_database = registry.get_asset_database()
    sources = [Path(asset_database.assets_root) / f"residency-{index}.ppm" for index in range(2)]
    payload = b"P6\n2 1\n255\n" + bytes((255, 0, 0, 0, 255, 0))
    for index, source in enumerate(sources):
        source.write_bytes(payload[:-1] + bytes((index,)))

    guids = [asset_database.import_asset(str(source)).guid for source in sources]
    assert all(guids)
    original_budget = registry.cpu_budget_bytes
    baseline_cpu_bytes = registry.total_cpu_bytes
    first = registry.load_texture_by_guid(guids[0])
    second = registry.load_texture_by_guid(guids[1])
    assert first is not None and second is not None
    first_record = registry.get_asset_residency(guids[0])
    second_record = registry.get_asset_residency(guids[1])
    assert first_record.cpu_bytes > 0
    assert second_record.cpu_bytes > 0

    larger_payload = b"P6\n4 2\n255\n" + bytes(range(24))
    sources[0].write_bytes(larger_payload)
    assert asset_database.reimport_asset(str(sources[0]))
    assert registry.reload_asset(guids[0]) is True
    updated_first_record = registry.get_asset_residency(guids[0])
    assert first.pixel_format == "bc1_rgba_srgb"
    assert updated_first_record.runtime_version == first_record.runtime_version + 1
    assert updated_first_record.cpu_bytes == first_record.cpu_bytes
    first_record = updated_first_record
    assert registry.total_cpu_bytes == baseline_cpu_bytes + first_record.cpu_bytes + second_record.cpu_bytes

    try:
        registry.pin_asset(guids[0])
        registry.pin_asset(guids[1])
        registry.cpu_budget_bytes = 1
        assert registry.trim_cpu_budget() == 0
        assert registry.is_loaded(guids[0]) is True
        assert registry.is_loaded(guids[1]) is True

        registry.unpin_asset(guids[0])
        del first
        gc.collect()
        assert registry.trim_cpu_budget() >= 1
        assert registry.is_loaded(guids[0]) is False
        assert registry.is_loaded(guids[1]) is True
        retained = registry.get_asset_residency(guids[1])
        assert retained.explicit_pin_count == 1
        assert retained.external_reference_count >= 1
        assert retained.evictable is False
    finally:
        if registry.is_loaded(guids[1]):
            registry.unpin_asset(guids[1])
        registry.cpu_budget_bytes = original_budget
        del second
        gc.collect()
        for guid, source in zip(guids, sources):
            registry.invalidate_asset(guid)
            if asset_database.contains_path(str(source)):
                asset_database.delete_asset(str(source))
            source.unlink(missing_ok=True)
            Path(f"{source}.meta").unlink(missing_ok=True)


class _RestoreFirst(InxComponent):
    value: int = 1


class _RestoreSecond(InxComponent):
    value: int = 2


class _RestoreReference(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)


class _RestoreComponentReference(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.COMPONENT)


# ═══════════════════════════════════════════════════════════════════════════
# Component add / remove / query
# ═══════════════════════════════════════════════════════════════════════════


class TestComponentLifecycle:
    def test_pending_python_component_reference_identity_is_preflighted(self, scene):
        from infernux.engine.component_restore import (
            PythonComponentRestoreError,
            deserialize_game_object_document_transactionally,
        )

        game_object = scene.create_game_object("InvalidReferenceRestore")
        game_object.add_py_component(_RestoreReference())
        document = game_object.serialize_document()
        document["components"][0]["data"]["target"] = make_game_object_ref(999_999_999)
        document["components"][0]["data"]["target"]["object_id"] = -1

        with pytest.raises(PythonComponentRestoreError, match="non-negative integer"):
            deserialize_game_object_document_transactionally(game_object, document)
        assert len(game_object.get_py_components()) == 1
        assert scene.has_pending_py_components() is False

    def test_pending_python_component_restore_preserves_missing_reference_identity(self, scene):
        from infernux.engine.component_restore import deserialize_game_object_document_transactionally

        game_object = scene.create_game_object("MissingReferenceRestore")
        game_object.add_py_component(_RestoreReference())
        document = game_object.serialize_document()
        document["components"][0]["data"]["target"] = make_game_object_ref(999_999_999)

        assert deserialize_game_object_document_transactionally(game_object, document)
        (restored,) = game_object.get_py_components()
        assert restored.target is None
        assert type(restored).target.get_raw(restored).persistent_id == 999_999_999
        assert game_object.serialize_document()["components"][0]["data"]["target"]["object_id"] == 999_999_999
        assert scene.has_pending_py_components() is False

    def test_component_reference_can_target_same_pending_batch(self, scene):
        from infernux.engine.component_restore import deserialize_game_object_document_transactionally

        game_object = scene.create_game_object("PendingReferenceRestore")
        target = game_object.add_py_component(_RestoreSecond())
        holder = game_object.add_py_component(_RestoreComponentReference())
        holder.target = ComponentRef(
            go_id=game_object.id,
            component_type="_RestoreSecond",
        )
        document = game_object.serialize_document()

        assert deserialize_game_object_document_transactionally(game_object, document) is True

        restored_components = game_object.get_py_components()
        restored_holder = next(
            component for component in restored_components if isinstance(component, _RestoreComponentReference)
        )
        restored_target = next(component for component in restored_components if isinstance(component, _RestoreSecond))
        assert restored_holder.target is restored_target
        assert restored_target is not target

    def test_pending_python_component_restore_rejects_invalid_field_atomically(self, scene):
        from infernux.engine.component_restore import (
            PythonComponentRestoreError,
            deserialize_game_object_document_transactionally,
        )

        game_object = scene.create_game_object("AtomicPythonRestore")
        first = game_object.add_py_component(_RestoreFirst())
        second = game_object.add_py_component(_RestoreSecond())
        first.value = 31
        second.value = 42
        before = game_object.serialize_document()
        document = game_object.serialize_document()
        document["components"][0]["data"]["value"] = 99
        document["components"][1]["data"]["value"] = "invalid"

        with pytest.raises(PythonComponentRestoreError, match="INT field requires an integer"):
            deserialize_game_object_document_transactionally(game_object, document)

        restored = game_object.get_py_components()
        assert restored == [first, second]
        assert first.value == 31
        assert second.value == 42
        assert game_object.serialize_document() == before
        assert scene.has_pending_py_components() is False

    def test_missing_python_component_type_restores_as_data_preserving_placeholder(self, scene):
        from infernux.components.missing_script import MissingScript
        from infernux.engine.component_restore import deserialize_game_object_document_transactionally

        game_object = scene.create_game_object("MissingPythonType")
        game_object.add_py_component(_RestoreFirst())
        document = game_object.serialize_document()
        descriptor = document["components"][0]
        parts = descriptor["type_id"].split(":")
        parts[-1] = "RemovedPythonComponent"
        descriptor["type_id"] = ":".join(parts)

        assert deserialize_game_object_document_transactionally(game_object, document)
        components = game_object.get_py_components()
        assert len(components) == 1
        assert isinstance(components[0], MissingScript)
        assert components[0]._is_broken is True
        assert components[0]._component_name == "RemovedPythonComponent"
        assert components[0]._serialize_fields_document()["value"] == 1
        assert scene.has_pending_py_components() is False

    def test_python_component_constraints_are_enforced_from_registry(self, scene):
        from infernux.components.decorators import disallow_multiple, require_component

        class RegistryDependency(InxComponent):
            pass

        @disallow_multiple
        @require_component(RegistryDependency)
        class RegistryConsumer(InxComponent):
            pass

        go = scene.create_game_object("PythonComponentConstraints")
        consumer = go.add_component(RegistryConsumer)
        dependency = go.get_component(RegistryDependency)
        assert consumer is not None
        assert dependency is not None
        assert go.add_component(RegistryConsumer) is None
        assert go.get_components(RegistryConsumer) == [consumer]
        assert go.remove_component(dependency) is False
        assert go.get_component(RegistryDependency) is dependency

    def test_failed_python_attachment_rolls_back_auto_added_dependencies(self, scene):
        from infernux.components.decorators import require_component

        class AutoDependency(InxComponent):
            pass

        @require_component(AutoDependency)
        class RejectedConsumer(InxComponent):
            _incompatible_components_ = ("MeshRenderer",)

        go = scene.create_game_object("AtomicPythonAttachment")
        assert go.add_component("MeshRenderer") is not None

        assert go.add_component(RejectedConsumer) is None
        assert go.get_component(RejectedConsumer) is None
        assert go.get_component(AutoDependency) is None

    def test_prepared_python_component_requires_valid_complete_set_before_activation(self, scene):
        from infernux.components.decorators import require_component

        class PreparedDependency(InxComponent):
            pass

        @require_component(PreparedDependency)
        class PreparedConsumer(InxComponent):
            pass

        go = scene.create_game_object("PreparedConstraintGate")
        component = PreparedConsumer()
        assert go._attach_prepared_py_component(component, 0) is component

        with pytest.raises(ValueError, match="requires missing component"):
            go._activate_prepared_py_component(component._cpp_component)

        assert go._remove_prepared_py_component(component._cpp_component) is True
        assert go.get_component(PreparedConsumer) is None

    def test_python_component_is_bound_before_reset(self, scene):
        observed = []

        class ResetBindingProbe(InxComponent):
            def reset(self):
                observed.append((self.game_object, self._cpp_component))

        go = scene.create_game_object("ResetBinding")
        component = go.add_component(ResetBindingProbe)

        assert component is not None
        assert observed == [(go, component._cpp_component)]

    def test_game_object_document_restore_recreates_python_components(self, scene):
        class DocumentProbeComponent(InxComponent):
            pass

        go = scene.create_game_object("DocumentRestoreGO")
        original = go.add_component(DocumentProbeComponent)
        document = go.serialize_document()

        from infernux.engine.component_restore import deserialize_game_object_document_transactionally

        assert deserialize_game_object_document_transactionally(go, document) is True

        restored = go.get_component(DocumentProbeComponent)
        assert isinstance(restored, DocumentProbeComponent)
        assert restored is not original

    def test_dynamic_rigidbody_forces_mesh_collider_convex_in_both_component_orders(self, scene):
        mesh_first = scene.create_primitive(PrimitiveType.Cube, "MeshFirst")
        first_mesh = mesh_first.add_component("MeshCollider")
        mesh_first.add_component("Rigidbody")
        assert first_mesh.convex is True

        rigidbody_first = scene.create_primitive(PrimitiveType.Cube, "RigidbodyFirst")
        rigidbody_first.add_component("Rigidbody")
        second_mesh = rigidbody_first.add_component("MeshCollider")
        assert second_mesh.convex is True

        with pytest.raises(ValueError, match="dynamic Rigidbody requires MeshCollider.convex"):
            second_mesh.convex = False
        assert second_mesh.convex is True

    def test_undo_adding_dynamic_rigidbody_restores_mesh_collider_convex(self, scene):
        from infernux.engine.undo import (
            AddComponentTransactionCommand,
            UndoManager,
        )

        previous_manager = UndoManager.instance()
        manager = UndoManager()
        try:
            owner = scene.create_primitive(PrimitiveType.Cube, "UndoDynamicMesh")
            mesh = owner.add_component("MeshCollider")
            assert mesh.convex is False
            assert manager.execute(AddComponentTransactionCommand(owner.id, "Rigidbody"))
            assert mesh.convex is True

            manager.undo()
            assert owner.get_component("Rigidbody") is None
            assert mesh.convex is False

            manager.redo()
            assert owner.get_component("Rigidbody") is not None
            assert mesh.convex is True
        finally:
            UndoManager._instance = previous_manager

    def test_dynamic_mesh_collider_survives_play_mode_document_rebuild(self, scene):
        from infernux.engine.play_mode import PlayModeManager

        owner = scene.create_primitive(PrimitiveType.Cube, "PlayModeDynamicMesh")
        mesh = owner.add_component("MeshCollider")
        owner.add_component("Rigidbody")
        snapshot = scene.serialize_document()
        mesh_document = next(
            component
            for component in snapshot["objects"][0]["components"]
            if component["type_id"] == "native:infernux.MeshCollider"
        )
        assert mesh_document["data"]["convex"] is True

        previous_manager = PlayModeManager.instance()
        manager = PlayModeManager()
        manager.set_asset_database(AssetRegistry.instance().get_asset_database())
        try:
            assert manager._rebuild_active_scene(snapshot, for_play=True)
            runtime_owner = SceneManager.instance().get_active_scene().find("PlayModeDynamicMesh")
            assert runtime_owner.get_component("MeshCollider").convex is True

            assert manager._rebuild_active_scene(snapshot, for_play=False)
            restored_owner = SceneManager.instance().get_active_scene().find("PlayModeDynamicMesh")
            assert restored_owner.get_component("MeshCollider").convex is True
        finally:
            PlayModeManager._instance = previous_manager

    def test_inactive_component_awakens_once_and_orders_enable_disable_and_destruction(self, scene):
        events = []

        class Probe(InxComponent):
            def awake(self):
                events.append("awake")

            def on_enable(self):
                events.append("enable")

            def on_disable(self):
                events.append("disable")

            def on_destroy(self):
                events.append("destroy")

        owner = scene.create_game_object("Lifecycle")
        owner.active = False
        owner.add_component(Probe)
        assert events == []
        owner.active = True
        assert events == ["awake", "enable"]
        owner.active = False
        owner.active = True
        assert events == ["awake", "enable", "disable", "enable"]
        events.clear()
        scene.destroy_game_object(owner)
        scene.process_pending_destroys()
        assert events == ["disable", "destroy"]

    def test_component_added_during_update_joins_next_frame_snapshot(self, scene):
        sm = SceneManager.instance()
        events = []

        class SpawnedComponent(InxComponent):
            def awake(self):
                events.append("spawned_awake")

            def on_enable(self):
                events.append("spawned_on_enable")

            def start(self):
                events.append("spawned_start")

            def late_update(self, delta_time: float):
                events.append("spawned_late_update")

        class SpawnerComponent(InxComponent):
            def awake(self):
                self._spawned = False

            def update(self, delta_time: float):
                if self._spawned:
                    return
                self._spawned = True
                events.append("spawner_update")
                self.game_object.add_component(SpawnedComponent)

            def late_update(self, delta_time: float):
                events.append("spawner_late_update")

        go = scene.create_game_object("StartTimingGO")
        go.add_component(SpawnerComponent)

        sm.play()
        sm.pause()
        events.clear()

        sm.step(1.0 / 60.0)

        assert events == [
            "spawner_update",
            "spawned_awake",
            "spawned_on_enable",
            "spawned_start",
            "spawner_late_update",
        ]

        sm.step(1.0 / 60.0)

        assert events[-2:] == ["spawner_late_update", "spawned_late_update"]

    def test_shared_scheduler_owns_python_update_dispatch(self, scene, runtime_scheduler):
        sm = SceneManager.instance()

        class ProbeComponent(InxComponent):
            def update(self, delta_time: float):
                self.last_delta_time = delta_time

        component = scene.create_game_object("DispatchProbe").add_component(ProbeComponent)

        sm.play()
        sm.pause()
        runtime_scheduler.reset_profiler()

        sm.step(1.0 / 60.0)

        counters = runtime_scheduler.profiler_snapshot()
        # One native fixed step enters the shared frame through the fixed,
        # physics_pre, and physics_post contracts.  Only the authored update
        # phase has an invoker in this probe, so phase_dispatches remains one.
        assert counters["native_phase_dispatches"] == 3
        assert counters["phase_dispatches"] == 1
        assert component.last_delta_time == pytest.approx(1.0 / 60.0)

    def test_disabled_component_coroutine_continues_until_its_game_object_deactivates(self, scene):
        events = []

        class Probe(InxComponent):
            def awake(self):
                self.start_coroutine(self.run())

            def run(self):
                events.append("started")
                yield None
                events.append("continued")
                yield None
                events.append("must-not-resume")

            def update(self, delta_time):
                events.append("update")

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        owner = scene.create_game_object("CoroutineOwner")
        component = owner.add_component(Probe)
        component.enabled = False
        events.clear()
        manager.step(1.0 / 60.0)
        assert events == ["continued"]
        owner.active = False
        owner.active = True
        manager.step(1.0 / 60.0)
        assert events == ["continued"]

    def test_awake_exception_disables_component(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                events.append("awake")
                raise RuntimeError("boom")

            def on_enable(self):
                events.append("on_enable")

        go = scene.create_game_object("AwakeExceptionGO")
        comp = go.add_component(ProbeComponent)

        assert events == ["awake"]
        assert comp.enabled is False

    def test_python_component_destroy_skips_on_destroy_when_never_activated(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def on_destroy(self):
                events.append("on_destroy")

        go = scene.create_game_object("DormantDestroyGO")
        go.active = False
        go.add_component(ProbeComponent)

        scene.destroy_game_object(go)
        scene.process_pending_destroys()

        assert events == []

    def test_renderstack_activation_and_cached_pipeline_lookup_follow_the_same_owner(self, scene):
        from types import SimpleNamespace

        owner = scene.create_game_object("RenderStackOwner")
        stack = owner.add_component(RenderStack)
        pipeline = RenderStackPipeline()
        context = SimpleNamespace(scene=scene)
        assert RenderStack.instance() is stack
        RenderStack.clear_active_instance(scene)
        assert pipeline._find_render_stack(context) is stack
        assert RenderStack.instance() is stack
        RenderStack.clear_active_instance(scene)
        assert pipeline._find_render_stack(context) is stack
        owner.active = False
        assert RenderStack.instance() is None
        RenderStack.clear_active_instance(scene)
        assert pipeline._find_render_stack(context) is None

    def test_renderstack_pipeline_rejects_live_stack_owned_by_previous_scene(self, scene):
        previous_owner = scene.create_game_object("PreviousSceneRenderStack")
        previous_stack = previous_owner.add_component(RenderStack)

        manager = SceneManager.instance()
        next_scene = manager.create_scene("renderstack_next_scene")
        next_owner = next_scene.create_game_object("NextSceneRenderStack")
        next_stack = next_owner.add_component(RenderStack)
        manager.set_active_scene(next_scene)

        class _Context:
            pass

        ctx = _Context()
        ctx.scene = next_scene
        pipeline = RenderStackPipeline()

        # Retained scene transactions can leave the old component completely
        # live for a short period.  Scene ownership, not liveness, decides
        # which graph is allowed to render.
        RenderStack.clear_active_instance(next_scene)
        assert previous_stack.is_valid
        assert previous_owner.is_active_in_hierarchy()
        assert pipeline._find_render_stack(ctx) is next_stack
        assert RenderStack.instance(next_scene) is next_stack

        manager.set_active_scene(scene)
        manager.unload_scene(next_scene)

    def test_renderstack_pipeline_keeps_independent_additive_scene_owners(self, scene):
        first_stack = scene.create_game_object("FirstSceneRenderStack").add_component(RenderStack)
        manager = SceneManager.instance()
        second_scene = manager.create_scene("renderstack_additive_scene")
        second_stack = second_scene.create_game_object("SecondSceneRenderStack").add_component(RenderStack)

        class _Context:
            def __init__(self, owner_scene):
                self.scene = owner_scene

        pipeline = RenderStackPipeline()
        try:
            assert pipeline._find_render_stack(_Context(scene)) is first_stack
            assert pipeline._find_render_stack(_Context(second_scene)) is second_stack
            assert pipeline._find_render_stack(_Context(scene)) is first_stack
            assert RenderStack.instance(scene) is first_stack
            assert RenderStack.instance(second_scene) is second_stack
        finally:
            manager.unload_scene(second_scene)

    def test_renderstack_effect_authoring_survives_component_restore_and_play_rebuild(self, scene):
        from infernux.engine.play_mode import PlayModeManager

        owner = scene.create_game_object("PlayModeRenderStack")
        stack = owner.add_component(RenderStack)
        slots = (
            EffectSlot(slot_id="empty-slot", stage_id="final"),
            EffectSlot(
                slot_id="effect-slot",
                stage_id="final",
                effect=RenderEffectRef(guid="missing-effect-guid", path_hint="Assets/RenderEffects/Missing.effect"),
            ),
        )
        stack.set_effect_stage_slots("final", slots)
        fields = stack._serialize_fields_document()
        stack.effect_slots = []
        stack._deserialize_fields_document(fields)
        restored_slots = stack.get_effect_stage_slots("final")
        assert [slot.slot_id for slot in restored_slots] == ["empty-slot", "effect-slot"]
        assert restored_slots[1].effect_ref.guid == "missing-effect-guid"
        assert not hasattr(stack, "effect_stage_bindings_json")
        snapshot = scene.serialize_document()
        previous = PlayModeManager.instance()
        manager = PlayModeManager()
        manager.set_asset_database(AssetRegistry.instance().get_asset_database())
        try:
            assert RenderStack.instance() is stack
            assert manager._rebuild_active_scene(snapshot, for_play=True)
            rebuilt = SceneManager.instance().get_active_scene().find("PlayModeRenderStack").get_component(RenderStack)
            assert rebuilt is not stack and RenderStack.instance() is rebuilt
            slots = rebuilt.get_effect_stage_slots("final")
            assert [slot.slot_id for slot in slots] == ["empty-slot", "effect-slot"]
            assert slots[1].effect_ref.guid == "missing-effect-guid"
        finally:
            PlayModeManager._instance = previous


# ═══════════════════════════════════════════════════════════════════════════
# Collider properties
# ═══════════════════════════════════════════════════════════════════════════


class TestColliders:
    def test_typed_collider_material_assign_restore_and_clear_preserve_combine_modes(self, scene):
        from infernux.core.asset_ref import PhysicMaterialRef
        from infernux.core.physic_material import PhysicMaterial

        native = InxPhysicMaterial()
        native.friction_combine, native.bounce_combine = 2, 3
        document = native.serialize_document()
        assert document["friction_combine"] == 2 and document["bounce_combine"] == 3
        assert native.friction_combine == 2 and native.bounce_combine == 3
        material = PhysicMaterial()
        material.friction_combine = PhysicsMaterialCombine.Multiply
        material.bounce_combine = PhysicsMaterialCombine.Maximum
        collider = scene.create_game_object("TypedSurface").add_component(BoxColliderComponent)
        collider.physic_material = material
        resolved = collider.physic_material.resolve()
        assert resolved is not None
        assert resolved.friction_combine == PhysicsMaterialCombine.Multiply
        assert resolved.bounce_combine == PhysicsMaterialCombine.Maximum
        collider.physic_material = PhysicMaterialRef()
        assert collider.physic_material.resolve() is None

    def test_physic_material_inspector_edit_is_undoable_and_republishes(self):
        from types import SimpleNamespace
        from infernux.core.physic_material import PhysicMaterial
        from infernux.engine.interaction import (
            DocumentKind,
            DocumentRegistry,
            ensure_editable_resource_document,
        )
        from infernux.engine.ui.asset_details_renderer import _apply_physic_material_edit
        from infernux.engine.undo import UndoManager

        class _ExecutionLayer:
            def __init__(self):
                self.published = []

            def refresh_binding(self, _category, _file_path):
                pass

            def schedule_rw_save(self, resource):
                self.published.append(resource.serialize_document())

        previous_manager = UndoManager.instance()
        previous_registry = DocumentRegistry._instance
        DocumentRegistry()
        manager = UndoManager()
        material = PhysicMaterial()
        original_friction = material.friction
        execution_layer = _ExecutionLayer()
        controller = ensure_editable_resource_document(
            category="physic_material",
            document_kind=DocumentKind.PHYSIC_MATERIAL,
            file_path="Assets/Test.physicmat",
            resource=material,
            guid="physic-material-guid",
            exec_layer=execution_layer,
        )
        state = SimpleNamespace(
            settings=material,
            exec_layer=execution_layer,
            resource_controller=controller,
            document_id=controller.document_id,
        )
        try:
            assert _apply_physic_material_edit(state, "friction", 0.8)
            assert material.friction == pytest.approx(0.8)

            manager.undo()
            assert material.friction == pytest.approx(original_friction)

            manager.redo()
            assert material.friction == pytest.approx(0.8)
            assert [entry["friction"] for entry in execution_layer.published] == pytest.approx(
                [0.8, original_friction, 0.8]
            )
        finally:
            UndoManager._instance = previous_manager
            DocumentRegistry._instance = previous_registry

    @pytest.mark.parametrize(
        "component_type,field,value",
        [
            ("BoxCollider", "size", [1, 0, 1]),
            ("CapsuleCollider", "direction", 9),
            ("MeshCollider", "convex", "yes"),
            ("BoxCollider", "physic_material_guid", 7),
        ],
    )
    def test_collider_documents_reject_invalid_values_transactionally(self, scene, component_type, field, value):
        collider = scene.create_game_object("StrictColliderDocument").add_component(component_type)
        original = collider.serialize_document()
        invalid = dict(original)
        invalid[field] = value

        assert collider.deserialize_document(invalid) is False
        assert collider.serialize_document() == original

    def test_collider_persists_only_physic_material_guid(self, scene):
        registry = AssetRegistry.instance()
        asset_database = registry.get_asset_database()
        asset_path = Path(asset_database.assets_root) / "SharedSurface.physicMaterial"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(
            json.dumps(
                {
                    "friction": 0.65,
                    "bounciness": 0.25,
                    "friction_combine": 2,
                    "bounce_combine": 3,
                }
            ),
            encoding="utf-8",
        )
        guid = asset_database.import_asset(str(asset_path)).guid
        assert guid

        material = registry.load_physic_material_by_guid(guid)
        assert material is not None
        assert registry.get_asset_version(guid) == 1
        assert registry.get_asset_runtime_type_name(guid)
        with pytest.raises(ValueError, match="Runtime asset type mismatch"):
            registry.get_mesh(guid)
        with pytest.raises(ValueError, match="resource type mismatch"):
            registry.load_mesh_by_guid(guid)
        assert registry.get_asset_version(guid) == 1
        collider = scene.create_game_object("SharedSurfaceCollider").add_component("BoxCollider")
        collider.physic_material = material
        document = collider.serialize_document()

        assert document["physic_material_guid"] == guid
        assert "friction" not in document
        assert "bounciness" not in document

        restored = scene.create_game_object("RestoredSurfaceCollider").add_component("BoxCollider")
        restored_document = dict(document)
        restored_document["component_id"] = restored.component_id
        assert restored.deserialize_document(restored_document) is True
        assert restored.physic_material_guid == guid
        assert restored.physic_material.resolve().native is material

        asset_path.write_text(
            json.dumps(
                {
                    "friction": 0.1,
                    "bounciness": 0.9,
                    "friction_combine": 1,
                    "bounce_combine": 2,
                }
            ),
            encoding="utf-8",
        )
        assert AssetManager.reimport_asset(str(asset_path), database=asset_database)
        assert registry.get_asset_version(guid) == 2
        assert restored.physic_material.resolve().native is material
        assert material.friction == pytest.approx(0.1)
        assert material.bounciness == pytest.approx(0.9)

        assert AssetManager.delete_asset(str(asset_path), database=asset_database)
        asset_path.unlink(missing_ok=True)
        assert collider.physic_material.resolve() is None
        assert collider.physic_material.guid == guid
        assert collider.physic_material_guid == guid
        assert restored.physic_material.resolve() is None
        assert restored.physic_material.guid == guid
        assert restored.physic_material_guid == guid


# ═══════════════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Light
# ═══════════════════════════════════════════════════════════════════════════


class TestLight:
    def test_light_color_modes_and_kelvin_round_trip(self, scene):
        light = scene.create_game_object("TemperatureLight").add_component("Light")
        native = light._cpp_component
        light.color = Vector3(0.8, 0.5, 0.25)
        assert light.color_mode == LightColorMode.Color
        assert light.use_color_temperature is False
        assert not type(light).color_temperature.metadata.visible_when(light)
        assert tuple(native.effective_color) == pytest.approx((0.8, 0.5, 0.25))

        light.color_mode = LightColorMode.FilterAndTemperature
        assert light.use_color_temperature is True
        assert type(light).color_temperature.metadata.visible_when(light)
        light.color_temperature = 6500.0
        assert tuple(native.effective_color) == pytest.approx((0.8, 0.5, 0.25))
        light.color_temperature = 2500.0
        warm = tuple(native.effective_color)
        assert warm[0] > warm[1] > warm[2]
        assert warm[2] < 0.25

        document = native.serialize_document()
        assert document["useColorTemperature"] is True
        assert document["colorTemperature"] == pytest.approx(2500.0)
        light.use_color_temperature = False
        assert native.deserialize_document(document) is True
        assert light.color_mode == LightColorMode.FilterAndTemperature
        invalid = dict(document, colorTemperature=999.0)
        assert native.deserialize_document(invalid) is False


# ═══════════════════════════════════════════════════════════════════════════
# MeshRenderer
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Texture (real GPU-side creation)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Material
# ═══════════════════════════════════════════════════════════════════════════


class TestMaterial:
    def test_material_texture_authoring_saves_atomically_and_survives_an_embedded_scene_roundtrip(
        self, engine, scene, tmp_path
    ):
        wrapper = Material.create_lit("WrapperSave")
        wrapper_path = tmp_path / "wrapper-save.mat"
        assert wrapper.save(str(wrapper_path))
        assert json.loads(wrapper_path.read_text(encoding="utf-8"))["name"] == "WrapperSave"
        material = InxMaterial.create_default_unlit()
        material.name = "InlineRuntimeMaterial"
        material.set_texture("texSampler", "white")
        assert material.get_texture("texSampler") == "white"
        with pytest.raises(ValueError, match="texture GUID does not exist"):
            material.set_texture("texSampler", "Assets/Textures/legacy-path.png")
        with pytest.raises(ValueError, match="texture GUID does not exist"):
            material.set_texture("texSampler", "missing-texture-guid")
        assert material.get_texture("texSampler") == "white"
        document = json.loads(material.serialize())
        missing_guid = "52f6a608180f248634255ab26ccbd0a3"
        document["properties"]["texSampler"]["guid"] = missing_guid
        assert material.deserialize(json.dumps(document))
        assert material.get_texture("texSampler") == missing_guid
        assert json.loads(material.serialize())["properties"]["texSampler"]["guid"] == missing_guid
        path = tmp_path / "atomic.mat"
        assert material.save_to(str(path))
        assert "material_version" not in json.loads(path.read_text(encoding="utf-8"))
        assert list(tmp_path.glob("atomic.mat.tmp.*")) == []
        renderer = scene.create_primitive(PrimitiveType.Cube, "InlineMaterialCube").get_component("MeshRenderer")
        renderer.material = material
        slot = json.loads(renderer.serialize())["materials"][0]
        assert isinstance(slot["material"], dict)
        assert "material_version" not in slot["material"] and "material_json" not in slot
        saved = scene.serialize_document()
        assert scene._commit_document(saved)
        restored = scene.find("InlineMaterialCube").get_component("MeshRenderer")
        restored_slot = json.loads(restored.serialize())["materials"][0]
        assert restored_slot["material"]["name"] == "InlineRuntimeMaterial"
        assert restored_slot["material"]["properties"]["texSampler"]["guid"] == missing_guid


# ═══════════════════════════════════════════════════════════════════════════
# Component serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestComponentSerialization:
    def test_missing_audio_resource_preserves_its_guid(self, scene):
        owner = scene.create_game_object("MissingAudioResource")
        component = owner.add_component("AudioSource")
        document = component.serialize_document()
        document["tracks"][0]["clip_guid"] = "missing-audio-resource-guid"

        assert component.deserialize_document(document) is True
        assert component.serialize_document()["tracks"][0]["clip_guid"] == ("missing-audio-resource-guid")
        assert component.get_track_clip(0) is None

    def test_audio_source_uses_fixed_named_bus_contract(self, scene):
        from infernux.lib import AudioEngine

        owner = scene.create_game_object("BusAudio")
        component = owner.add_component("AudioSource")
        component.output_bus = "Music"
        assert component.serialize_document()["output_bus"] == "Music"

        with pytest.raises(ValueError, match="Unknown AudioSource output bus"):
            component.output_bus = "Dialogue"
        assert component.output_bus == "Music"

        engine = AudioEngine.instance()
        try:
            engine.set_bus_volume("Music", 0.35)
            engine.set_bus_muted("Music", True)
            assert engine.get_bus_volume("Music") == pytest.approx(0.35)
            assert engine.get_bus_muted("Music") is True
            with pytest.raises(ValueError, match="Unknown audio bus"):
                engine.get_bus_volume("Dialogue")
        finally:
            engine.set_bus_muted("Music", False)
            engine.set_bus_volume("Music", 1.0)

    def test_audio_priority_round_trip_and_runtime_budget(self, scene):
        from infernux.lib import AudioEngine
        from infernux.components.builtin.audio_source import AudioSource

        owner = scene.create_game_object("PriorityAudio")
        native = owner.add_component("AudioSource")
        source = AudioSource._get_or_create_wrapper(native, owner)
        assert source.priority == 128
        source.priority = 16
        assert native.serialize_document()["priority"] == 16
        document = native.serialize_document()
        document.pop("priority")
        assert native.deserialize_document(document)
        assert source.priority == 128
        for invalid in (-1, 256):
            with pytest.raises(ValueError, match="priority"):
                source.priority = invalid
        assert source.is_track_virtual() is False
        assert source.rejected_one_shot_count == 0
        audio = AudioEngine.instance()
        previous = audio.max_real_voices
        try:
            audio.max_real_voices = 2
            assert audio.max_real_voices == 2
            assert audio.real_voice_count <= 2
            with pytest.raises(ValueError, match="positive"):
                audio.max_real_voices = 0
            assert audio.max_real_voices == 2
        finally:
            audio.max_real_voices = previous

    def test_audio_track_seek_is_runtime_state_and_uses_clip_seconds(self, scene, tmp_path):
        import wave
        from infernux.lib import AudioClip
        from infernux.components.builtin.audio_source import AudioSource

        clip_path = tmp_path / "seek.wav"
        with wave.open(str(clip_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(bytes(22050 * 2 * 2))
        clip = AudioClip()
        assert clip.load_from_file(str(clip_path))
        owner = scene.create_game_object("SeekableAudio")
        native = owner.add_component("AudioSource")
        source = AudioSource._get_or_create_wrapper(native, owner)
        source.track_count = 2
        source.set_track_clip(0, clip)
        source.set_track_clip(1, clip)
        authored = native.serialize_document()

        source.set_track_time(0, 1.25)
        source.set_track_time(1, 0.5)
        assert source.get_track_time() == pytest.approx(1.25)
        assert source.get_track_time(1) == pytest.approx(0.5)
        assert source.is_track_playing(0) is False
        assert native.serialize_document() == authored
        with pytest.raises(IndexError, match="track index"):
            source.set_track_time(2, 0.0)
        with pytest.raises(ValueError, match="finite"):
            source.set_track_time(0, float("nan"))
        with pytest.raises(IndexError, match="duration"):
            source.set_track_time(0, 2.1)
        assert source.get_track_time() == pytest.approx(1.25)
        source.stop()
        assert source.get_track_time() == 0.0
        source.set_track_clip(1, None)
        assert source.get_track_time(1) == 0.0
        with pytest.raises(RuntimeError, match="loaded clip"):
            source.set_track_time(1, 0.0)

    def test_audio_bus_fade_uses_device_time_and_direct_write_cancels_it(self, engine, scene):
        import time
        from infernux.lib import AudioEngine

        audio = AudioEngine.instance()
        if not audio.is_initialized:
            audio.initialize()
        if not audio.is_initialized:
            pytest.skip("No audio output device; dummy-device native regression covers this contract")

        def wait_for_fade():
            deadline = time.monotonic() + 2.0
            while audio.is_bus_fading("Music") and time.monotonic() < deadline:
                time.sleep(0.005)
            assert not audio.is_bus_fading("Music")

        try:
            audio.pause_all()
            audio.set_bus_volume("Music", 1.0)
            audio.fade_bus_volume("Music", 0.0, 0.1)
            assert audio.is_bus_fading("Music") is True
            frozen = audio.output_time
            engine.tick(1.0)
            assert audio.output_time == frozen
            assert audio.get_bus_volume("Music") == 1.0
            audio.resume_all()
            # No scene tick: even a silent device completes automation.
            wait_for_fade()
            assert audio.get_bus_volume("Music") == 0.0
            assert audio.output_time > frozen
            assert audio.output_peak >= 0.0
            assert audio.saturated_sample_count >= 0

            audio.fade_bus_volume("Music", 1.0, 0.1)
            audio.set_bus_muted("Music", True)
            wait_for_fade()
            assert audio.get_bus_volume("Music") == 1.0

            audio.fade_bus_volume("Music", 0.0, 1.0)
            audio.cancel_bus_fade("Music")
            cancelled = audio.get_bus_volume("Music")
            time.sleep(0.05)
            assert audio.get_bus_volume("Music") == cancelled
            assert audio.is_bus_fading("Music") is False

            audio.fade_bus_volume("Music", 0.0, 1.0)
            audio.set_bus_volume("Music", 0.25)
            assert audio.get_bus_volume("Music") == pytest.approx(0.25)
            assert audio.is_bus_fading("Music") is False

            with pytest.raises(ValueError, match="positive and finite"):
                audio.fade_bus_volume("Music", 0.5, 0.0)
            with pytest.raises(ValueError, match="Unknown audio bus"):
                audio.fade_bus_volume("Dialogue", 0.5, 1.0)
        finally:
            audio.resume_all()
            audio.cancel_bus_fade("Music")
            audio.set_bus_muted("Music", False)
            audio.set_bus_volume("Music", 1.0)

    @pytest.mark.parametrize("resource_kind", ["mesh", "material"])
    def test_missing_renderer_resource_preserves_its_guid(self, scene, resource_kind):
        owner = scene.create_game_object(f"Missing{resource_kind.title()}Resource")
        renderer = owner.add_component("MeshRenderer")
        document = renderer.serialize_document()
        missing_guid = f"missing-{resource_kind}-resource-guid"
        if resource_kind == "mesh":
            document["meshAssetGuid"] = missing_guid
        else:
            document["materials"] = [missing_guid]

        assert renderer.deserialize_document(document) is True
        restored = renderer.serialize_document()
        if resource_kind == "mesh":
            assert restored["meshAssetGuid"] == missing_guid
        else:
            assert restored["materials"] == [missing_guid]

    def test_missing_physic_material_preserves_its_guid(self, scene):
        owner = scene.create_game_object("MissingPhysicMaterial")
        collider = owner.add_component("BoxCollider")
        document = collider.serialize_document()
        document["physic_material_guid"] = "missing-physic-material-guid"

        assert collider.deserialize_document(document) is True
        assert collider.serialize_document()["physic_material_guid"] == "missing-physic-material-guid"

    @pytest.mark.parametrize("field,value", [("mass", 0.0), ("constraints", 1), ("collision_detection_mode", 99)])
    def test_rigidbody_document_rejects_invalid_values_transactionally(self, scene, field, value):
        game_object = scene.create_game_object("StrictRigidbody")
        rigidbody = game_object.add_component("Rigidbody")
        rigidbody.mass = 3.5
        original = rigidbody.serialize_document()
        invalid = dict(original)
        invalid[field] = value

        assert rigidbody.deserialize_document(invalid) is False
        assert rigidbody.serialize_document() == original


def test_puzzle_level_components_configure_save_and_reopen_with_authored_state(scene, tmp_path):
    from infernux.engine.component_restore import deserialize_scene_document_transactionally
    from infernux.engine.scene_authoring import decode_scene_document, encode_scene_document

    class LevelProbe(InxComponent):
        value: int = 7

    actor = scene.create_game_object("PuzzleActor")
    actor.transform.position = Vector3(1, 2, 3)
    transform = actor.get_component("Transform")
    assert transform is actor.transform and not actor.remove_component(transform)
    probe = actor.add_component(LevelProbe)
    temporary = actor.add_component(CameraComponent)
    assert actor.get_component(LevelProbe) is actor.get_component("LevelProbe") is probe
    assert actor.get_components(LevelProbe) == [probe]
    assert actor.get_component(CameraComponent) is temporary
    assert actor.remove_component(temporary) and actor.get_component(CameraComponent) is None
    transient_body = actor.add_component("Rigidbody")
    assert actor.get_component("Rigidbody") is transient_body
    assert actor.remove_component(transient_body) and actor.get_component("Rigidbody") is None
    body = actor.add_component("Rigidbody")
    body.mass = 7.77
    sphere = actor.add_component("SphereCollider")
    sphere.radius = 2.5
    assert probe in actor.get_components()
    assert all(type(item).__name__ != "PyComponentProxy" for item in actor.get_components())

    floor = scene.create_game_object("Floor").add_component("BoxCollider")
    floor.size, floor.is_trigger = Vector3(2, 3, 4), True
    capsule = scene.create_game_object("ActorShape").add_component("CapsuleCollider")
    capsule.height, capsule.radius = 3.0, 1.0
    cylinder = scene.create_game_object("Post").add_component("CylinderCollider")
    cylinder.radius, cylinder.height, cylinder.direction = 1.25, 4.0, 2
    camera = scene.create_game_object("AuthorCamera").add_component("Camera")
    assert camera.field_of_view == pytest.approx(60)
    assert 0 < camera.near_clip < camera.far_clip and not camera.dithering and not camera.stop_nans
    camera.field_of_view, camera.depth = 90, 5
    camera.dithering = camera.stop_nans = True
    light = scene.create_game_object("AuthorLight").add_component("Light")
    assert light.light_type == LightType.Directional and light.intensity == pytest.approx(1)
    assert (light.shadow_bias, light.shadow_normal_bias) == pytest.approx((1, 1))
    assert light.affect_geometry and light.affect_particles
    light.light_type, light.shadows = LightType.Point, LightShadows.Hard
    light.intensity, light.color = 2.5, Vector3(1, 0, 0)
    light.affect_geometry, light.affect_particles = False, True
    renderer = scene.create_primitive(PrimitiveType.Cube, "AuthoredMesh").get_component("MeshRenderer")
    assert len(renderer.get_positions()) and len(renderer.get_normals()) and len(renderer.get_indices())
    renderer.material = InxMaterial.create_default_lit()
    assert renderer.get_material(0) is not None
    renderer.casts_shadows = False
    assert light.serialize_document()["influenceDomains"] == 2
    path = tmp_path / "puzzle-components.scene"
    assert scene.save_to_file(str(path))
    authored = json.loads(path.read_text(encoding="utf-8"))
    saved = decode_scene_document(authored)
    assert deserialize_scene_document_transactionally(scene, saved)
    actor = scene.find("PuzzleActor")
    assert tuple(actor.transform.position) == pytest.approx((1, 2, 3))
    assert actor.get_component("Rigidbody").mass == pytest.approx(7.77)
    assert actor.get_component(LevelProbe).value == 7
    floor = scene.find("Floor").get_component("BoxCollider")
    sphere = actor.get_component("SphereCollider")
    capsule = scene.find("ActorShape").get_component("CapsuleCollider")
    cylinder = scene.find("Post").get_component("CylinderCollider")
    camera = scene.find("AuthorCamera").get_component("Camera")
    light = scene.find("AuthorLight").get_component("Light")
    renderer = scene.find("AuthoredMesh").get_component("MeshRenderer")
    assert tuple(floor.size) == pytest.approx((2, 3, 4)) and floor.is_trigger
    assert sphere.radius == pytest.approx(2.5)
    assert (capsule.radius, capsule.height) == pytest.approx((1, 3))
    assert (cylinder.radius, cylinder.height, cylinder.direction) == pytest.approx((1.25, 4, 2))
    assert (camera.field_of_view, camera.depth) == pytest.approx((90, 5))
    assert camera.dithering and camera.stop_nans
    assert light.light_type == LightType.Point and light.shadows == LightShadows.Hard
    assert light.intensity == pytest.approx(2.5) and tuple(light.color) == pytest.approx((1, 0, 0, 1))
    assert not light.affect_geometry and light.affect_particles and not renderer.casts_shadows
    assert encode_scene_document(scene.serialize_document()) == authored
    floor.is_trigger = False
    assert not floor.is_trigger
