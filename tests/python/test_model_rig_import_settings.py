"""Source inventories and published rig/animation payloads are distinct."""

import base64
import json
import struct
import time
from pathlib import Path

import pytest

from infernux.core.assets import AssetManager
from infernux.core.asset_types import read_mesh_import_settings
from infernux.lib import AssetDependencyGraph, AssetRegistry


@pytest.mark.parametrize("apply_mode", ["sync", "async"])
def test_skin_settings_publish_companion_and_reject_empty_weights(engine, tmp_path, monkeypatch, apply_mode):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Weights.fbx"
    source.parent.mkdir()
    root = Path(__file__).resolve().parents[2]
    source.write_bytes((root / "external/assimp/test/models/FBX/animation_with_skeleton.fbx").read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    guid = imported.guid
    sidecar = Path(str(source) + ".meta")
    settings = read_mesh_import_settings(str(source))

    def apply():
        if apply_mode == "sync":
            return AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        owner = AssetManager.begin_model_reimport(str(source), settings)
        deadline = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        return result

    try:
        mesh = registry.load_mesh(str(source))
        original_bones = mesh.skinned_bone_count
        for count in (1, 2, 3, 4):
            settings.max_bones_per_vertex = count
            result = apply()
            assert result and result.guid == guid, result.error
            saved = read_mesh_import_settings(str(source))
            assert saved.max_bones_per_vertex == count
            assert mesh.skinned_bone_count == original_bones
            registry.invalidate_asset(guid)
            mesh = registry.load_mesh(str(source))
            assert mesh.has_skinned_data and mesh.skinned_bone_count == original_bones
        before = sidecar.read_bytes()
        settings.min_bone_weight = 1.0
        result = apply()
        assert not result and "min_bone_weight removes every influence" in result.error
        assert sidecar.read_bytes() == before
        assert mesh.skinned_bone_count == original_bones
        # A normal source reimport retains the last successful author settings.
        result = AssetManager.reimport_asset(str(source), database=database)
        assert result, result.error
        saved = read_mesh_import_settings(str(source))
        assert saved.max_bones_per_vertex == 4 and saved.min_bone_weight == 0
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(source))


def test_generic_skeleton_definition_copy_resolves_guid_artifact_and_is_atomic(engine, tmp_path, monkeypatch):
    """Copy consumes the published definition entity, never a source path."""
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    fixture = Path(__file__).resolve().parents[2] / "external/assimp/test/models/FBX/animation_with_skeleton.fbx"
    definition_source, copy_source = folder / "Definition.fbx", folder / "Copy.fbx"
    definition_source.write_bytes(fixture.read_bytes())
    copy_source.write_bytes(fixture.read_bytes())
    definition_import = AssetManager.import_asset(str(definition_source), database=database)
    copy_import = AssetManager.import_asset(str(copy_source), database=database)
    assert definition_import and copy_import
    try:
        definition_mesh = registry.load_mesh(str(definition_source))
        identity = definition_mesh.skeleton_definition
        assert identity["guid"] == definition_import.guid and identity["subresource_id"] == "skeleton"
        settings = read_mesh_import_settings(str(copy_source))
        settings.skeleton_definition_mode = "copy"
        settings.skeleton_definition_guid = identity["guid"]
        settings.skeleton_definition_id = identity["subresource_id"]
        settings.rig_root_node = ""
        result = AssetManager.reimport_asset(str(copy_source), import_settings=settings.to_dict(), database=database)
        assert result, result.error
        registry.invalidate_asset(copy_import.guid)
        copied_mesh = registry.load_mesh(str(copy_source))
        assert copied_mesh.skeleton_definition == identity
        assert AssetDependencyGraph.instance().has_dependency(copy_import.guid, identity["guid"])

        sidecar_before = Path(str(copy_source) + ".meta").read_bytes()
        settings.skeleton_definition_id = "missing"
        rejected = AssetManager.reimport_asset(str(copy_source), import_settings=settings.to_dict(), database=database)
        assert not rejected and "Copy skeleton definition" in rejected.error
        assert Path(str(copy_source) + ".meta").read_bytes() == sidecar_before
        assert registry.load_mesh(str(copy_source)).skeleton_definition == identity
    finally:
        registry.invalidate_asset(definition_import.guid)
        registry.invalidate_asset(copy_import.guid)
        database.delete_asset(str(definition_source))
        database.delete_asset(str(copy_source))


def test_humanoid_apply_publishes_runtime_report_and_rejects_bad_override_atomically(
    engine,
    tmp_path,
    monkeypatch,
):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Humanoid.fbx"
    source.parent.mkdir()
    fixture = Path(__file__).resolve().parents[2] / "external/assimp/test/models/FBX/animation_with_skeleton.fbx"
    source.write_bytes(fixture.read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    try:
        settings = read_mesh_import_settings(str(source))
        settings.rig_type = "humanoid"
        applied = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert applied, applied.error
        registry.invalidate_asset(imported.guid)
        mesh = registry.load_mesh(str(source))
        report = mesh.humanoid_rig
        assert set(report) == {
            "valid",
            "required_bones_valid",
            "hierarchy_valid",
            "reference_pose_valid",
            "mapping",
            "issues",
        }
        assert isinstance(report["mapping"], dict) and isinstance(report["issues"], list)
        meta = database.get_meta_by_guid(imported.guid).serialize_document()["metadata"]
        assert json.loads(meta["published_humanoid_rig"]["value"]) == report

        sidecar_before = Path(str(source) + ".meta").read_bytes()
        settings.humanoid_bone_overrides = {"hips": "node-that-does-not-exist"}
        rejected = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert not rejected and "humanoid override" in rejected.error
        assert Path(str(source) + ".meta").read_bytes() == sidecar_before
        assert registry.load_mesh(str(source)).humanoid_rig == report
    finally:
        registry.invalidate_asset(imported.guid)
        database.delete_asset(str(source))


@pytest.mark.parametrize(
    "apply_mode,source_kind", [("sync", "fbx"), ("async", "animation_only"), ("async", "skinned_gltf")]
)
def test_rig_animation_apply_replaces_companion_without_losing_source_inventory(
    engine,
    tmp_path,
    monkeypatch,
    source_kind,
    apply_mode,
):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    root = Path(__file__).resolve().parents[2]
    if source_kind == "animation_only":
        source = folder / "EmptyAnimation.gltf"
        data = struct.pack("<8f", 0, 1, 0, 0, 0, 1, 0, 0)
        source.write_text(
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "scene": 0,
                    "scenes": [{"nodes": [0]}],
                    "nodes": [{"name": "AnimatedEmpty"}],
                    "buffers": [
                        {
                            "byteLength": len(data),
                            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode(),
                        }
                    ],
                    "bufferViews": [
                        {"buffer": 0, "byteOffset": 0, "byteLength": 8},
                        {"buffer": 0, "byteOffset": 8, "byteLength": 24},
                    ],
                    "accessors": [
                        {"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1]},
                        {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"},
                    ],
                    "animations": [
                        {
                            "name": "Move",
                            "samplers": [{"input": 0, "output": 1}],
                            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    else:
        fixture = (
            root / "external/assimp/test/models/FBX/animation_with_skeleton.fbx"
            if source_kind == "fbx"
            else root / "tests/native/fixtures/model_uv_basis.gltf"
        )
        source = folder / fixture.name
        source.write_bytes(fixture.read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    guid = imported.guid
    try:
        mesh = registry.load_mesh(str(source))
        assert mesh.has_skinned_data
        original_bones = mesh.skinned_bone_count
        original_clips = mesh.skinned_animation_names
        original_vertices = mesh.vertex_count
        original_meta = database.get_meta_by_guid(guid).serialize_document()["metadata"]
        settings = read_mesh_import_settings(str(source))
        # Re-enable after each exclusion, including both flags changing together.
        for rig, animations in [("generic", False), ("none", False), ("none", True), ("generic", True)]:
            settings.rig_type, settings.import_animations = rig, animations
            if apply_mode == "sync":
                result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
            else:
                owner = AssetManager.begin_model_reimport(str(source), settings)
                deadline = time.monotonic() + 30
                while (result := AssetManager.poll_model_reimport(owner)) is None:
                    assert time.monotonic() < deadline
                    time.sleep(0.002)
            assert result, result.error
            assert mesh.vertex_count == original_vertices
            assert mesh.skinned_animation_names == (original_clips if rig == "generic" and animations else [])
            assert mesh.skinned_bone_count == (original_bones if rig == "generic" else 0)
            assert mesh.has_skinned_data == (rig == "generic" and (bool(original_bones) or animations))
            metadata = database.get_meta_by_guid(guid).serialize_document()["metadata"]
            for key in ("bone_count", "bone_names_csv", "animation_count", "animation_names_csv"):
                assert metadata[key] == original_meta[key]
            saved = read_mesh_import_settings(str(source))
            assert (saved.rig_type, saved.import_animations) == (rig, animations)
            # Discard the live registry object and read the actual binary companion.
            registry.invalidate_asset(guid)
            mesh = registry.load_mesh(str(source))
            assert mesh.skinned_animation_names == (original_clips if rig == "generic" and animations else [])
            assert mesh.has_skinned_data == (rig == "generic" and (bool(original_bones) or animations))
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(source))
