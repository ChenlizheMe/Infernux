from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import time
import threading

import numpy as np
import pytest

from Infernux.lib import AssetDependencyGraph, AssetMutationErrorCode, AssetRegistry, InxMaterial, ResourceType
from Infernux.core.assets import AssetManager
from Infernux.engine.path_utils import same_path
from Infernux.particle import (
    AssetReference,
    ParticleArtifactRegistry,
    ParticleGraphAsset,
    SdfVolume,
    VectorField,
)


def test_mesh_position_publication_preserves_identity_and_source(engine, tmp_path: Path):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    source = tmp_path / "editable.obj"
    document = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    source.write_text(document, encoding="ascii")
    guid = database.import_asset(str(source)).guid
    try:
        pending = registry.begin_load_mesh_by_guid(guid)
        mesh = registry.load_mesh_by_guid(guid)
        original = mesh._particle_sampling_data()
        version = registry.get_asset_residency(guid).runtime_version
        positions = original["positions"][1:2].copy()
        positions[:, 2] = 2.0
        registry.update_mesh_positions(guid, 1, positions)
        positions[:] = 99  # Native publication owns its data.
        assert registry.get_mesh(guid) is mesh
        current = mesh._particle_sampling_data()
        expected = original["positions"].copy()
        expected[1, 2] = 2.0
        np.testing.assert_array_equal(current["positions"], expected)
        np.testing.assert_array_equal(current["indices"], original["indices"])
        assert registry.get_asset_residency(guid).runtime_version == version + 1
        assert source.read_text(encoding="ascii") == document
        deadline = time.monotonic() + 10.0
        while not pending.complete and time.monotonic() < deadline:
            time.sleep(0.001)
        assert pending.complete
        with pytest.raises(RuntimeError, match="stale"):
            registry.try_commit_asset_load(pending)
        registry.update_mesh_positions(guid, mesh.vertex_count, np.empty((0, 3), np.float32))
        for first, values, message in (
            (mesh.vertex_count, np.zeros((1, 3), np.float32), "exceeds"),
            (0, np.full((1, 3), np.nan, np.float32), "finite"),
            (0, np.zeros((1, 2), np.float32), "shape"),
        ):
            with pytest.raises(ValueError, match=message):
                registry.update_mesh_positions(guid, first, values)
        assert registry.get_asset_residency(guid).runtime_version == version + 1
        np.testing.assert_array_equal(mesh._particle_sampling_data()["positions"], expected)
    finally:
        database.delete_asset(str(source))


def test_shared_mesh_position_publication_keeps_collision_until_recook(engine, scene, tmp_path: Path):
    from Infernux.lib import MeshCollider as NativeMeshCollider, Physics, Vector3

    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    source = tmp_path / "shared-surface.obj"
    source.write_text("v -1 0 -1\nv -1 0 1\nv 1 0 -1\nf 1 2 3\n", encoding="ascii")
    guid = database.import_asset(str(source)).guid
    objects = []
    try:
        for x in (0, 5):
            go = scene.create_game_object("shared surface")
            objects.append(go)
            go.transform.position = Vector3(x, 0, 0)
            go.add_component("MeshRenderer").set_mesh_asset_guid(guid)
            go.add_component("MeshCollider")
        Physics.sync_transforms()
        before = NativeMeshCollider.get_cooking_cache_stats()["async_submissions"]
        positions = registry.get_mesh(guid)._particle_sampling_data()["positions"]
        positions[:, 1] = 2
        normals = np.tile(np.array([0.6, 0.8, 0], np.float32), (len(positions), 1))
        registry.update_mesh_positions(guid, 0, positions, normals=normals)
        version = registry.get_asset_residency(guid).runtime_version
        for invalid in (np.zeros((len(positions), 2), np.float32), np.full_like(normals, np.inf)):
            with pytest.raises(ValueError, match="normals"):
                registry.update_mesh_positions(guid, 0, positions + 10, normals=invalid)
        assert registry.get_asset_residency(guid).runtime_version == version
        registry.update_mesh_positions(guid, 0, positions)
        # Omitting normals preserves the previously published lighting attributes.
        Physics.sync_transforms()
        assert NativeMeshCollider.get_cooking_cache_stats()["async_submissions"] == before
        generation_before_recook = int(Physics.query_generation)
        for go, x in zip(objects, (0, 5)):
            np.testing.assert_array_equal(go.get_component("MeshRenderer").get_positions(), positions)
            np.testing.assert_allclose(go.get_component("MeshRenderer").get_normals(), normals)
            assert Physics.raycast(Vector3(x - 0.5, 5, -0.5), Vector3(0, -1, 0), 10).point.y == pytest.approx(0)
        objects[0].get_component("MeshCollider").recook()
        Physics.sync_transforms()
        assert int(Physics.query_generation) > generation_before_recook
        assert Physics.raycast(Vector3(-0.5, 5, -0.5), Vector3(0, -1, 0), 10).point.y == pytest.approx(2)
        assert Physics.raycast(Vector3(4.5, 5, -0.5), Vector3(0, -1, 0), 10).point.y == pytest.approx(0)
    finally:
        for go in objects:
            scene.destroy_game_object(go)
        database.delete_asset(str(source))


def test_authored_mesh_import_restores_edited_geometry_with_independent_identity(engine, tmp_path: Path):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    original = tmp_path / "original.obj"
    authored = tmp_path / "edited.inxmesh"
    original.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    original_guid = database.import_asset(str(original)).guid
    try:
        mesh = registry.load_mesh_by_guid(original_guid)
        positions = mesh._particle_sampling_data()["positions"]
        positions[:, 2] = 3
        normals = np.tile(np.array([0, 0, 1], np.float32), (len(positions), 1))
        registry.update_mesh_positions(original_guid, 0, positions, normals=normals)
        source_bytes = mesh.serialize_source()
        authored.write_bytes(source_bytes)
        imported = database.import_asset(str(authored))
        assert imported.resource_type == ResourceType.Mesh
        assert imported.guid != original_guid
        restored = registry.load_mesh_by_guid(imported.guid)
        assert restored is not mesh
        np.testing.assert_array_equal(restored._particle_sampling_data()["positions"], positions)
        assert restored.serialize_source() == source_bytes
        registry.invalidate_asset(imported.guid)
        reloaded = registry.load_mesh_by_guid(imported.guid)
        assert reloaded is not restored
        assert reloaded.serialize_source() == source_bytes
        registry.update_mesh_positions(original_guid, 0, positions + 5)
        assert reloaded.serialize_source() == source_bytes
        replacement_bytes = mesh.serialize_source()
        authored.write_bytes(replacement_bytes)
        replaced = database.import_asset(str(authored))
        assert replaced.guid == imported.guid
        registry.invalidate_asset(imported.guid)
        assert registry.load_mesh_by_guid(imported.guid).serialize_source() == replacement_bytes
    finally:
        if database.contains_path(str(authored)):
            database.delete_asset(str(authored))
        database.delete_asset(str(original))


def test_authored_mesh_cooked_artifact_validates_and_loads_without_source(engine, tmp_path: Path):
    from Infernux.engine.runtime_artifact_catalog import validate_artifact, artifact_source_hash
    from Infernux.engine.player_package_native import write_pack, read_entry
    from Infernux.engine.game_builder import GameBuilder

    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    root = Path(database.assets_root).parent
    original = Path(database.assets_root) / "041-cooked-source.obj"
    authored = Path(database.assets_root) / "041-cooked-copy.inxmesh"
    settings = root / "ProjectSettings" / "BuildSettings.json"
    previous_settings = settings.read_bytes() if settings.exists() else None
    settings.parent.mkdir(exist_ok=True)
    settings.write_text('{"scenes": []}', encoding="utf-8")
    original.write_text("v 0 0 2\nv 1 0 2\nv 0 1 2\nf 1 2 3\n", encoding="ascii")
    try:
        original_guid = database.import_asset(str(original)).guid
        source_bytes = registry.load_mesh_by_guid(original_guid).serialize_source()
        authored.write_bytes(source_bytes)
        guid = database.import_asset(str(authored)).guid
        database.flush_derived_index()
        index = json.loads(Path(database.asset_index_path).read_text(encoding="utf-8"))
        entry = next(item for item in index["entries"] if item["guid"] == guid)
        artifact = root / entry["artifact_path"]
        binding = validate_artifact(root, entry, artifact)
        assert binding["source_guid"] == guid
        assert artifact_source_hash(artifact) != "infernux.static-mesh.source"
        assert artifact.read_bytes() != source_bytes
        builder = GameBuilder(str(root), str(tmp_path / "Build"))
        builder.freeze_asset_index_entries([entry])
        builder._runtime_artifact_bindings = {}
        builder._runtime_artifact_source_paths = set()
        staged = tmp_path / "Staged"
        builder._stage_library_runtime_artifacts(str(staged))
        assert (staged / entry["artifact_path"]).read_bytes() == artifact.read_bytes()
        assert f"assets/{authored.name}" in builder._runtime_artifact_source_paths
        assert builder._runtime_artifact_bindings[entry["artifact_path"]]["source_guid"] == guid
        assert not (staged / "Assets" / authored.name).exists()
        package = tmp_path / "Content.inxpkg"
        write_pack(((entry["artifact_path"], staged / entry["artifact_path"]),), package)
        assert read_entry(package, entry["artifact_path"]) == artifact.read_bytes()
        authored.unlink()
        registry.invalidate_asset(guid)
        loaded = registry.load_mesh_by_guid(guid)
        assert loaded.serialize_source() == source_bytes
    finally:
        if previous_settings is None:
            settings.unlink(missing_ok=True)
        else:
            settings.write_bytes(previous_settings)
        for source in (authored, original):
            if database.contains_path(str(source)):
                database.delete_asset(str(source))
            source.unlink(missing_ok=True)
            Path(f"{source}.meta").unlink(missing_ok=True)


def test_mesh_copy_command_undo_redo_restores_identity(engine, tmp_path: Path):
    from Infernux.engine.interaction import EditorActionJournal, ProjectAssetCommandService, SelectionService
    from Infernux.engine.undo import UndoManager

    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    assets = tmp_path / "Assets"
    assets.mkdir()
    original = assets / "original.obj"
    target = assets / "copy.inxmesh"
    original.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    original_guid = database.import_asset(str(original)).guid
    manager = UndoManager(EditorActionJournal())
    service = ProjectAssetCommandService(SelectionService())
    service.configure(str(tmp_path), database)
    try:
        mesh = registry.load_mesh_by_guid(original_guid)
        saved = service.save_mesh_copy(mesh, str(target))
        assert Path(saved) == target
        guid = database.get_guid_from_path(str(target))
        assert guid and guid != original_guid
        content = target.read_bytes()
        with pytest.raises(FileExistsError):
            service.save_mesh_copy(mesh, str(target))
        manager.undo()
        assert not target.exists()
        assert not database.contains_guid(guid)
        manager.redo()
        assert database.get_guid_from_path(str(target)) == guid
        assert target.read_bytes() == content
        assert registry.load_mesh_by_guid(guid).serialize_source() == content
        occupied = assets / "occupied.inxmesh"

        class ConcurrentTarget:
            def serialize_source(self):
                occupied.write_bytes(b"another author's file")
                return content

        with pytest.raises(RuntimeError):
            service.save_mesh_copy(ConcurrentTarget(), str(occupied))
        assert occupied.read_bytes() == b"another author's file"
    finally:
        service.shutdown()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))
        database.delete_asset(str(original))


def test_mesh_assignment_command_restores_complete_source(engine, scene, tmp_path: Path, monkeypatch):
    from Infernux.lib import PrimitiveType
    from Infernux.engine.interaction import ComponentCommandService, EditorActionJournal
    from Infernux.engine.undo import UndoManager

    database = engine.get_asset_database()
    source = tmp_path / "assigned.obj"
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    guid = database.import_asset(str(source)).guid
    owner = scene.create_game_object("renderer")
    renderer = owner.add_component("MeshRenderer")
    renderer.set_primitive_mesh(PrimitiveType.Cube)
    before = renderer.serialize_document()
    previous_manager = UndoManager._instance
    manager = UndoManager(EditorActionJournal())
    service = ComponentCommandService()
    try:
        assert service.assign_mesh_asset(renderer, guid)
        after = renderer.serialize_document()
        assert after["meshAssetGuid"] == guid
        assert not after.get("useInlineMesh", False)
        manager.undo()
        assert renderer.serialize_document() == before
        manager.redo()
        assert renderer.serialize_document() == after
        assert not service.assign_mesh_asset(renderer, guid)
        with pytest.raises(ValueError, match="registered Mesh"):
            service.assign_mesh_asset(renderer, "missing-guid")
        assert renderer.serialize_document() == after
        assert len(manager.action_journal.applied_entries()) == 1
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager
        scene.destroy_game_object(owner)
        database.delete_asset(str(source))


def test_audio_import_requires_complete_metadata(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "incomplete_audio.wav"
    source.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    meta_path = Path(f"{source}.meta")
    asset_guid = "a" * 32
    meta_path.write_text(
        json.dumps({
            "metadata": {
                "guid": {"type": "string", "value": asset_guid},
                "resource_type": {
                    "type": "enum infernux::ResourceType",
                    "value": "DefaultText",
                },
            },
        }),
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="current content_hash"):
            asset_db.import_asset(str(source))
        assert not asset_db.contains_path(str(source))
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))


def test_render_effect_import_tracks_group_dependencies(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    graph = AssetDependencyGraph.instance()
    bloom = tmp_path / "Bloom.effect"
    tone = tmp_path / "Tonemapping.effect"
    group = tmp_path / "Basic Post Processing.effectgroup"

    def effect_document(feature_type: str) -> dict:
        return {
            "$schema": "infernux.render_effect",
            "feature_type": feature_type,
            "parameters": {},
            "dependencies": [],
        }

    bloom.write_text(json.dumps(effect_document("infernux.post.bloom")), encoding="utf-8")
    tone.write_text(json.dumps(effect_document("infernux.post.tonemapping")), encoding="utf-8")

    imported_paths = []
    try:
        bloom_result = asset_db.import_asset(str(bloom))
        tone_result = asset_db.import_asset(str(tone))
        imported_paths.extend((bloom, tone))
        assert bloom_result and tone_result
        assert bloom_result.resource_type == ResourceType.RenderEffect
        assert tone_result.resource_type == ResourceType.RenderEffect

        group.write_text(
            json.dumps(
                {
                    "$schema": "infernux.render_effect_group",
                    "entries": [
                        {
                            "entry_id": "bloom",
                            "asset": {"guid": bloom_result.guid},
                            "enabled": True,
                            "overrides": {"intensity": 0.8},
                        },
                        {
                            "entry_id": "tonemapping",
                            "asset": {"guid": tone_result.guid},
                            "enabled": True,
                            "overrides": {},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        group_result = asset_db.import_asset(str(group))
        imported_paths.append(group)

        assert group_result
        assert group_result.resource_type == ResourceType.RenderEffect
        assert set(graph.get_dependencies(group_result.guid)) == {
            bloom_result.guid,
            tone_result.guid,
        }
    finally:
        for path in reversed(imported_paths):
            if asset_db.contains_path(str(path)):
                asset_db.delete_asset(str(path))


def test_render_effect_import_ignores_path_only_dependency(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "Path Only.effectgroup"
    source.write_text(
        json.dumps(
            {
                "$schema": "infernux.render_effect_group",
                "entries": [
                    {
                        "entry_id": "tone",
                        "asset": {
                            "guid": "",
                            "path_hint": "Assets/Rendering/Tone.effect",
                        },
                        "enabled": True,
                        "overrides": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = asset_db.import_asset(str(source))

    assert result
    assert asset_db.contains_path(str(source))
    assert AssetDependencyGraph.instance().get_dependencies(result.guid) == set()


def test_render_effect_import_does_not_consume_mount_scope_in_asset(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "InvalidScope.effect"
    source.write_text(
        json.dumps(
            {
                "$schema": "infernux.render_effect",
                "feature_type": "infernux.route.pixelation",
                "parameters": {},
                "dependencies": [],
                "scope": "final",
            }
        ),
        encoding="utf-8",
    )

    result = asset_db.import_asset(str(source))

    assert result
    assert asset_db.contains_path(str(source))
    assert asset_db.get_guid_from_path(str(source)) == result.guid


def test_particle_graph_import_compiles_and_publishes_aot(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "Smoke.particlegraph"
    wind_guid = "a" * 32
    collision_guid = "b" * 32
    mesh_guid = "c" * 32
    document = ParticleGraphAsset(stable_id="integration-smoke").to_dict()
    document["emitters"][0]["stages"]["rendering"]["nodes"][1]["properties"][
        "shader"
    ] = "Particle Unlit"
    document["emitters"][0]["data_interfaces"] = [
        VectorField(
            stable_id="wind-field",
            texture=AssetReference(guid=wind_guid),
        ).to_dict(),
        SdfVolume(
            stable_id="collision-field",
            texture=AssetReference(guid=collision_guid),
        ).to_dict(),
    ]
    document["parameters"].append(
        {
            "stable_id": "surface-mesh",
            "name": "Surface Mesh",
            "type": {"value_type": "mesh", "space": "none"},
            "default": {
                "guid": mesh_guid,
                "path_hint": "Assets/Models/Surface.fbx",
            },
                "exposed": True,
                "writable": False,
                "category": "",
            "tooltip": "",
            "attributes": [],
        }
    )
    source.write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    ParticleArtifactRegistry.clear()

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)

        assert result, result.error
        assert result.resource_type == ResourceType.ParticleGraph
        assert asset_db.get_meta_by_path(str(source)).get_resource_type() == ResourceType.ParticleGraph
        artifact = ParticleArtifactRegistry.get(str(source), guid=result.guid)
        assert artifact is not None
        assert ParticleArtifactRegistry.get(str(source)) is artifact
        assert artifact.hir["stable_id"] == "integration-smoke"
        assert Path(artifact.artifact_path).name == f"{result.guid}.inxparticle"
        runtime_artifact = Path(
            asset_db.get_runtime_artifact_path(result.guid, ResourceType.ParticleGraph)
        )
        published_artifact = Path(artifact.artifact_path)
        assert runtime_artifact.samefile(published_artifact)
        assert AssetDependencyGraph.instance().get_dependencies(result.guid) == {
            collision_guid,
            mesh_guid,
            wind_guid,
        }
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))
        ParticleArtifactRegistry.clear()


def test_particle_graph_native_importer_rejects_obsolete_emitter_lifecycle_schema(
    engine, tmp_path: Path
):
    asset_db = engine.get_asset_database()
    source = tmp_path / "ObsoleteLifecycle.particlegraph"
    document = ParticleGraphAsset(stable_id="obsolete-lifecycle").to_dict()
    document["emitters"][0]["play_on_start"] = False
    source.write_text(json.dumps(document), encoding="utf-8")

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)

        assert not result
        assert result.error_code == AssetMutationErrorCode.IMPORT_FAILED
        assert "missing or unknown fields" in result.error
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))


def test_particle_graph_native_importer_rejects_separate_emitter_active_state(
    engine, tmp_path: Path
):
    asset_db = engine.get_asset_database()
    source = tmp_path / "SeparateActiveState.particlegraph"
    document = ParticleGraphAsset(stable_id="separate-active-state").to_dict()
    document["emitters"][0]["active"] = True
    source.write_text(json.dumps(document), encoding="utf-8")

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)

        assert not result
        assert result.error_code == AssetMutationErrorCode.IMPORT_FAILED
        assert "missing or unknown fields" in result.error
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))


def test_particle_script_import_uses_script_resource_and_particle_aot(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "Sparks.particle.py"
    source.write_text(
        """from Infernux.particle import (
    AssetReference, EmitterSettings, ParticleEmitter, ParticleScript
)

class SparksGraph(ParticleScript):
    stable_id = "integration-sparks"

    class Sparks(ParticleEmitter):
        stable_id = "sparks"
        settings = EmitterSettings()

        def init(self, ctx, particles):
            pass

        def update(self, ctx, particles):
            pass

        def rendering(self, ctx, particles):
            particles.sprite(shader="Particle Unlit")
""",
        encoding="utf-8",
    )
    ParticleArtifactRegistry.clear()

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)

        assert result
        assert result.resource_type == ResourceType.Script
        artifact = ParticleArtifactRegistry.get(str(source), guid=result.guid)
        assert artifact is not None
        assert artifact.source_kind == "script"
        assert artifact.hir["schedule"] == ["sparks"]
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))
        ParticleArtifactRegistry.clear()


def test_particle_graph_reimport_keeps_last_known_good_on_semantic_failure(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "StableSmoke.particlegraph"
    valid = ParticleGraphAsset(stable_id="stable-smoke")
    source.write_text(json.dumps(valid.to_dict()), encoding="utf-8")
    ParticleArtifactRegistry.clear()

    try:
        imported = AssetManager.import_asset(str(source), database=asset_db)
        assert imported, imported.error
        published = ParticleArtifactRegistry.get(str(source), guid=imported.guid)
        assert published is not None

        invalid = valid.to_dict()
        invalid["emitters"][0]["stages"]["rendering"]["nodes"] = []
        invalid["emitters"][0]["stages"]["rendering"]["links"] = []
        source.write_text(json.dumps(invalid), encoding="utf-8")
        result = AssetManager.reimport_asset(str(source), database=asset_db)

        assert not result
        assert result.database_committed is True
        assert result.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
        assert "keeping last-known-good" in result.error
        assert ParticleArtifactRegistry.get(str(source), guid=imported.guid) == published
    finally:
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))
        ParticleArtifactRegistry.clear()




def test_vector_field_import_exposes_immutable_volume_generations(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "Wind.inxvfield"

    def document(vectors):
        return {
            "$schema": "infernux.vector_field",
            "dimensions": [2, 1, 1],
            "storage_order": "x_fastest",
            "bake_basis": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "vectors": vectors,
        }

    source.write_text(json.dumps(document([[1, 2, 3], [-4, 5.5, 6]])), encoding="utf-8")
    registry = AssetRegistry.instance()

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)
        assert result, result.error
        assert result.resource_type == ResourceType.Texture

        texture = registry.load_texture_by_guid(result.guid)
        assert texture is not None
        assert texture.dimension == "3d"
        assert texture.pixel_format == "rgba16_float"
        assert texture.pixel_depth == 1
        initial_generation = texture.generation
        assert initial_generation > 0
        assert tuple(texture.bake_basis) == tuple(document([])["bake_basis"])
        np.testing.assert_allclose(texture.value_min[:3], (-4.0, 2.0, 3.0))
        np.testing.assert_allclose(texture.value_max[:3], (1.0, 5.5, 6.0))

        assert not hasattr(texture, "volume_array")

        source.write_text(json.dumps(document([[7, 8, 9], [10, 11, 12]])), encoding="utf-8")
        reimported = AssetManager.reimport_asset(str(source), database=asset_db)
        assert reimported, reimported.error

        assert texture.generation == initial_generation + 1
        np.testing.assert_allclose(texture.value_min[:3], (7.0, 8.0, 9.0))
        np.testing.assert_allclose(texture.value_max[:3], (10.0, 11.0, 12.0))
    finally:
        if "result" in locals() and result.guid:
            registry.remove_asset(result.guid)
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))


def test_signed_distance_field_import_exposes_immutable_volume_generations(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "Collider.inxsdf"

    def document(distances):
        return {
            "$schema": "infernux.sdf",
            "dimensions": [2, 1, 1],
            "storage_order": "x_fastest",
            "distance_unit": "field",
            "bake_basis": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "distances": distances,
        }

    source.write_text(json.dumps(document([-0.25, 0.75])), encoding="utf-8")
    registry = AssetRegistry.instance()

    try:
        result = AssetManager.import_asset(str(source), database=asset_db)
        assert result, result.error
        assert result.resource_type == ResourceType.Texture

        texture = registry.load_texture_by_guid(result.guid)
        assert texture is not None
        assert texture.dimension == "3d"
        assert texture.semantic == "signed_distance_field"
        assert texture.pixel_format == "rgba16_float"
        assert texture.pixel_depth == 1
        initial_generation = texture.generation
        assert initial_generation > 0
        assert tuple(texture.bake_basis) == tuple(document([])["bake_basis"])
        assert texture.value_min[0] == pytest.approx(-0.25)
        assert texture.value_max[0] == pytest.approx(0.75)

        assert not hasattr(texture, "volume_array")

        source.write_text(json.dumps(document([-0.5, 1.25])), encoding="utf-8")
        reimported = AssetManager.reimport_asset(str(source), database=asset_db)
        assert reimported, reimported.error

        assert texture.generation == initial_generation + 1
        assert texture.value_min[0] == pytest.approx(-0.5)
        assert texture.value_max[0] == pytest.approx(1.25)
    finally:
        if "result" in locals() and result.guid:
            registry.remove_asset(result.guid)
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))


def test_asset_database_never_indexes_python_bytecode_or_cache_paths(engine):
    asset_db = engine.get_asset_database()
    fixture = Path(asset_db.assets_root) / "python-bytecode-ignore-fixture"
    cache = fixture / "__pycache__"
    similarly_named = fixture / "my__pycache__data"
    cache.mkdir(parents=True, exist_ok=True)
    similarly_named.mkdir(parents=True, exist_ok=True)
    cached_bytecode = cache / "Controller.cpython-312.pyc"
    top_level_bytecode = fixture / "Legacy.PYC"
    control = similarly_named / "control.txt"
    cached_bytecode.write_bytes(b"not real bytecode")
    top_level_bytecode.write_bytes(b"not real bytecode")
    control.write_text("import me", encoding="utf-8")

    try:
        asset_db.refresh()

        assert asset_db.contains_path(str(cached_bytecode)) is False
        assert asset_db.contains_path(str(top_level_bytecode)) is False
        assert asset_db.get_guid_from_path(str(cached_bytecode)) == ""
        assert asset_db.get_guid_from_path(str(top_level_bytecode)) == ""
        assert not Path(f"{cached_bytecode}.meta").exists()
        assert not Path(f"{top_level_bytecode}.meta").exists()
        assert cached_bytecode not in {Path(path) for path in asset_db.last_refresh_imported_paths}
        assert top_level_bytecode not in {Path(path) for path in asset_db.last_refresh_imported_paths}

        assert asset_db.contains_path(str(control)) is True
        assert asset_db.get_guid_from_path(str(control))

        explicit = asset_db.import_asset(str(top_level_bytecode))
        assert not explicit
        assert explicit.error_code == AssetMutationErrorCode.UNSUPPORTED_TYPE
        assert not Path(f"{top_level_bytecode}.meta").exists()
    finally:
        if asset_db.contains_path(str(control)):
            asset_db.delete_asset(str(control))
        shutil.rmtree(fixture, ignore_errors=True)
        asset_db.refresh()


def test_dependency_graph_separates_asset_and_runtime_domains():
    graph = AssetDependencyGraph.instance()
    asset_user = "test-asset-user"
    runtime_user = "test-runtime-user"
    dependency = "test-shared-dependency"
    generation = graph.asset_generation

    for legacy_name in ("add_dependency", "remove_dependency", "clear_dependencies_of", "set_dependencies"):
        assert not hasattr(graph, legacy_name)

    try:
        graph.add_asset_dependency(asset_user, dependency)
        assert graph.asset_generation == generation + 1
        graph.add_runtime_dependency(runtime_user, dependency)
        assert graph.has_dependency(asset_user, dependency)
        assert graph.has_dependency(runtime_user, dependency)
        assert {asset_user, runtime_user} <= set(graph.get_dependents(dependency))

        graph.clear_asset_dependencies_of(asset_user)
        assert not graph.has_dependency(asset_user, dependency)
        assert graph.has_dependency(runtime_user, dependency)
        assert set(graph.get_dependents(dependency)) == {runtime_user}

        with pytest.raises(ValueError, match="cannot depend on itself"):
            graph.add_runtime_dependency(runtime_user, runtime_user)
    finally:
        graph.clear_asset_dependencies_of(asset_user)
        graph.clear_runtime_dependencies_of(runtime_user)


def test_asset_database_canonical_crud_preserves_guid(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "canonical_asset.txt"
    source.write_text("first", encoding="utf-8")

    import_result = asset_db.import_asset(str(source))
    assert import_result
    guid = import_result.guid
    assert asset_db.get_guid_from_path(str(source)) == guid
    assert Path(asset_db.get_path_from_guid(guid)).resolve() == source.resolve()
    assert asset_db.get_meta_by_path(str(source)).get_guid() == guid
    catalog = asset_db.get_directory_catalog(str(tmp_path))
    catalog_entry = next(entry for entry in catalog if entry["guid"] == guid)
    assert Path(catalog_entry["path"]).resolve() == source.resolve()
    assert catalog_entry["name"] == source.name
    assert catalog_entry["size"] == len("first")
    assert asset_db.catalog_generation == asset_db.query_generation

    moved = tmp_path / "canonical_asset_moved.txt"
    source.replace(moved)
    assert asset_db.move_asset(str(source), str(moved))
    assert asset_db.get_guid_from_path(str(moved)) == guid
    assert not asset_db.contains_path(str(source))
    moved_catalog = asset_db.get_directory_catalog(str(tmp_path))
    assert next(entry for entry in moved_catalog if entry["guid"] == guid)["name"] == moved.name

    moved.unlink()
    assert asset_db.delete_asset(str(moved))
    assert not asset_db.contains_guid(guid)
    assert not Path(f"{moved}.meta").exists()
    assert all(entry["guid"] != guid for entry in asset_db.get_directory_catalog(str(tmp_path)))


def test_asset_database_does_not_expose_legacy_resource_crud(engine):
    asset_db = engine.get_asset_database()
    for name in (
        "register_resource",
        "modify_resource",
        "delete_resource",
        "move_resource",
        "get_all_resource_guids",
        "on_asset_created",
        "on_asset_modified",
        "on_asset_deleted",
        "on_asset_moved",
    ):
        assert not hasattr(asset_db, name)


def test_asset_database_moves_registered_assets_as_one_published_batch(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source_a = tmp_path / "batch-a.txt"
    source_b = tmp_path / "batch-b.txt"
    destination = tmp_path / "Moved"
    destination.mkdir()
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    imported_a = asset_db.import_asset(str(source_a))
    imported_b = asset_db.import_asset(str(source_b))
    assert imported_a and imported_b
    generation_before = asset_db.query_generation
    moved_a = destination / source_a.name
    moved_b = destination / source_b.name
    source_a.replace(moved_a)
    source_b.replace(moved_b)

    results = asset_db.move_assets_batch(
        [(str(source_a), str(moved_a)), (str(source_b), str(moved_b))]
    )

    assert len(results) == 2
    assert all(results)
    assert asset_db.query_generation == generation_before + 1
    assert asset_db.get_guid_from_path(str(moved_a)) == imported_a.guid
    assert asset_db.get_guid_from_path(str(moved_b)) == imported_b.guid
    assert not asset_db.contains_path(str(source_a))
    assert not asset_db.contains_path(str(source_b))

    moved_a.replace(source_a)
    moved_b.replace(source_b)
    restored = asset_db.move_assets_batch(
        [(str(moved_a), str(source_a)), (str(moved_b), str(source_b))]
    )
    assert len(restored) == 2 and all(restored)
    assert asset_db.delete_asset(str(source_a))
    assert asset_db.delete_asset(str(source_b))


def test_asset_database_batch_rejects_destination_guid_collision(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "source.txt"
    occupied = tmp_path / "occupied.txt"
    source.write_text("source", encoding="utf-8")
    occupied.write_text("occupied", encoding="utf-8")
    source_result = asset_db.import_asset(str(source))
    occupied_result = asset_db.import_asset(str(occupied))
    assert source_result and occupied_result
    generation_before = asset_db.query_generation

    results = asset_db.move_assets_batch([(str(source), str(occupied))])

    assert len(results) == 1
    assert not results[0]
    assert asset_db.query_generation == generation_before
    assert asset_db.get_guid_from_path(str(source)) == source_result.guid
    assert asset_db.get_guid_from_path(str(occupied)) == occupied_result.guid
    assert asset_db.delete_asset(str(source))
    assert asset_db.delete_asset(str(occupied))


def test_project_directory_relocation_is_one_editor_and_catalog_transaction(
    engine, tmp_path: Path
):
    from Infernux.engine.interaction import (
        AssetMutationService,
        DocumentRegistry,
        SelectionService,
        SelectionTarget,
    )
    from Infernux.engine.ui import project_file_ops

    asset_db = engine.get_asset_database()
    source_dir = tmp_path / "Source"
    destination_dir = tmp_path / "Destination"
    source_dir.mkdir()
    source_a = source_dir / "A.txt"
    source_b = source_dir / "B.txt"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    imported_a = asset_db.import_asset(str(source_a))
    imported_b = asset_db.import_asset(str(source_b))
    assert imported_a and imported_b

    selection = SelectionService()
    selection.select(
        SelectionTarget.asset(imported_a.guid),
        owner_id="project",
        record_history=False,
    )
    mutations = AssetMutationService(DocumentRegistry(), selection)
    published = []
    mutations.add_observer(published.append)
    generation_before = asset_db.query_generation
    moved_a = destination_dir / source_a.name
    moved_b = destination_dir / source_b.name

    try:
        result = project_file_ops.move_path(
            str(source_dir),
            str(destination_dir),
            asset_db,
            origin="user",
            operation_id="directory-transaction",
        )

        assert same_path(result, str(destination_dir))
        assert asset_db.query_generation == generation_before + 1
        assert asset_db.get_guid_from_path(str(moved_a)) == imported_a.guid
        assert asset_db.get_guid_from_path(str(moved_b)) == imported_b.guid
        assert selection.snapshot.primary == SelectionTarget.asset(imported_a.guid)
        assert len(published) == 1
        assert published[0].operation_id == "directory-transaction"
        assert len(published[0].changes) == 2
        assert Path(f"{moved_a}.meta").is_file()
        assert Path(f"{moved_b}.meta").is_file()
        assert not list(destination_dir.rglob("*.meta.meta"))
    finally:
        mutations.shutdown()
        for path in (moved_a, moved_b, source_a, source_b):
            if asset_db.contains_path(str(path)):
                asset_db.delete_asset(str(path))


def test_project_shader_move_preserves_material_guid_reference_without_rewrite(
    engine, tmp_path: Path
):
    from Infernux.engine.ui import project_file_ops

    database = engine.get_asset_database()
    graph = AssetDependencyGraph.instance()
    source_dir = tmp_path / "Shaders"
    destination_dir = tmp_path / "Rendering"
    source_dir.mkdir()
    destination_dir.mkdir()
    vertex = source_dir / "surface.vert"
    fragment = source_dir / "surface.frag"
    material = tmp_path / "Surface.mat"
    vertex.write_text("void main() {}", encoding="utf-8")
    fragment.write_text("void main() {}", encoding="utf-8")
    vertex_guid = database.import_asset(str(vertex)).guid
    fragment_guid = database.import_asset(str(fragment)).guid
    assert vertex_guid and fragment_guid
    document = json.loads(InxMaterial.create_default_lit().serialize())
    document["shaders"] = {
        "vertex": {"guid": vertex_guid, "shader_id": "surface-vertex"},
        "fragment": {"guid": fragment_guid, "shader_id": "surface-fragment"},
    }
    original = json.dumps(document)
    material.write_text(original, encoding="utf-8")
    material_guid = database.import_asset(str(material)).guid
    assert material_guid
    destination = destination_dir / fragment.name

    try:
        result = project_file_ops.move_path(
            str(fragment), str(destination), database,
            origin="user", operation_id="shader-guid-relocation",
        )

        assert same_path(result, str(destination))
        assert material.read_text(encoding="utf-8") == original
        assert database.get_guid_from_path(str(destination)) == fragment_guid
        assert fragment_guid in set(graph.get_dependencies(material_guid))
    finally:
        for path in (material, fragment, destination, vertex):
            if database.contains_path(str(path)):
                database.delete_asset(str(path))
            path.unlink(missing_ok=True)
            Path(f"{path}.meta").unlink(missing_ok=True)


def test_metadata_creation_uses_the_submitted_source_bytes(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    text = tmp_path / "metadata-source.txt"
    texture = tmp_path / "metadata-source.ppm"
    text_bytes = b"first\r\nsecond\r\n"
    texture_bytes = b"P6\n2 1\n255\n" + bytes((255, 0, 0, 0, 255, 0))
    text.write_bytes(text_bytes)
    texture.write_bytes(texture_bytes)

    try:
        text_guid = asset_db.import_asset(str(text)).guid
        texture_guid = asset_db.import_asset(str(texture)).guid
        assert text_guid and texture_guid

        text_meta = asset_db.get_meta_by_guid(text_guid)
        texture_meta = asset_db.get_meta_by_guid(texture_guid)
        text_document = text_meta.serialize_document()
        texture_document = texture_meta.serialize_document()
        assert text_document["metadata"]["file_size"] == {
            "type": "size_t",
            "value": len(text_bytes),
        }
        assert texture_document["metadata"]["file_size"] == {
            "type": "size_t",
            "value": len(texture_bytes),
        }
        assert texture_meta.get_int("width") == 2
        assert texture_meta.get_int("height") == 1
        assert texture_meta.get_int("channels") == 3
        assert texture_meta.get_string("source_container") == "PPM"
        assert texture_meta.get_string("texture_format") == "auto"
    finally:
        for path in (text, texture):
            if asset_db.contains_path(str(path)):
                asset_db.delete_asset(str(path))
            path.unlink(missing_ok=True)
            Path(f"{path}.meta").unlink(missing_ok=True)


def test_material_import_artifact_commits_metadata_and_dependencies_atomically(
    engine, tmp_path: Path
):
    asset_db = engine.get_asset_database()
    graph = AssetDependencyGraph.instance()
    vertex = tmp_path / "artifact.vert"
    fragment = tmp_path / "artifact.frag"
    material = tmp_path / "artifact.mat"
    vertex.write_text("void main() {}", encoding="utf-8")
    fragment.write_text("void main() {}", encoding="utf-8")
    vertex_guid = asset_db.import_asset(str(vertex)).guid
    fragment_guid = asset_db.import_asset(str(fragment)).guid
    assert vertex_guid and fragment_guid

    def write_material(shader_paths: list[tuple[Path, str]]) -> None:
        document = json.loads(InxMaterial.create_default_lit().serialize())
        if shader_paths:
            document["shaders"]["vertex"] = {
                "guid": shader_paths[0][1],
                "shader_id": "artifact-vertex",
                "path_hint": str(shader_paths[0][0]),
            }
        if len(shader_paths) > 1:
            document["shaders"]["fragment"] = {
                "guid": shader_paths[1][1],
                "shader_id": "artifact-fragment",
                "path_hint": str(shader_paths[1][0]),
            }
        material.write_text(json.dumps(document), encoding="utf-8")

    try:
        material.write_text("{ invalid first import", encoding="utf-8")
        failed_import = asset_db.import_asset(str(material))
        assert not failed_import
        assert failed_import.error_code == AssetMutationErrorCode.IMPORT_FAILED
        assert failed_import.database_committed is False
        assert failed_import.error
        assert not asset_db.contains_path(str(material))

        write_material([(vertex, vertex_guid), (fragment, fragment_guid)])
        material_guid = asset_db.import_asset(str(material)).guid
        assert material_guid
        assert set(graph.get_dependencies(material_guid)) == {
            vertex_guid,
            fragment_guid,
        }
        metadata_before = asset_db.get_meta_by_guid(material_guid).serialize_document()
        generation_before = asset_db.query_generation

        material.write_text("{ invalid material", encoding="utf-8")
        assert not asset_db.reimport_asset(str(material))
        assert asset_db.query_generation == generation_before
        assert set(graph.get_dependencies(material_guid)) == {
            vertex_guid,
            fragment_guid,
        }
        assert (
            asset_db.get_meta_by_guid(material_guid).serialize_document()
            == metadata_before
        )

        write_material([(vertex, vertex_guid)])
        assert asset_db.reimport_asset(str(material))
        assert set(graph.get_dependencies(material_guid)) == {vertex_guid}
    finally:
        if asset_db.contains_path(str(material)):
            asset_db.delete_asset(str(material))
        asset_db.delete_asset(str(vertex))
        asset_db.delete_asset(str(fragment))


def test_asset_database_explicit_reimport_preserves_guid(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "reimport.txt"
    source.write_text("first", encoding="utf-8")
    guid = asset_db.import_asset(str(source)).guid

    source.write_text("second", encoding="utf-8")
    assert asset_db.reimport_asset(str(source))
    assert asset_db.get_guid_from_path(str(source)) == guid

    unregistered = tmp_path / "unregistered.txt"
    unregistered.write_text("content", encoding="utf-8")
    missing = asset_db.reimport_asset(str(unregistered))
    assert not missing
    assert missing.error_code == AssetMutationErrorCode.NOT_FOUND
    assert missing.error


def test_asset_database_rejects_worker_thread_mutation(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    source = tmp_path / "worker.txt"
    source.write_text("content", encoding="utf-8")
    errors = []

    def mutate():
        try:
            asset_db.import_asset(str(source))
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=mutate)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert "owner thread" in errors[0]
    assert not asset_db.contains_path(str(source))


def test_asset_database_publishes_concurrent_reader_snapshots(engine, tmp_path: Path):
    asset_db = engine.get_asset_database()
    stable_path = tmp_path / "stable-reader.txt"
    stable_path.write_text("stable", encoding="utf-8")
    stable_guid = asset_db.import_asset(str(stable_path)).guid
    initial_generation = asset_db.query_generation

    start = threading.Event()
    stop = threading.Event()
    errors: list[str] = []

    def read_snapshots():
        start.wait()
        try:
            while not stop.is_set():
                assert asset_db.contains_guid(stable_guid)
                assert asset_db.contains_path(str(stable_path))
                assert asset_db.get_guid_from_path(str(stable_path)) == stable_guid
                assert asset_db.get_path_from_guid(stable_guid)
                meta = asset_db.get_meta_by_guid(stable_guid)
                assert meta is not None and meta.get_guid() == stable_guid
                assert stable_guid in asset_db.get_all_guids()
                assert any(
                    entry["guid"] == stable_guid
                    for entry in asset_db.get_directory_catalog(str(tmp_path))
                )
                assert asset_db.asset_count >= 1
        except BaseException as exc:
            errors.append(repr(exc))
            stop.set()

    readers = [threading.Thread(target=read_snapshots) for _ in range(4)]
    for reader in readers:
        reader.start()

    start.set()
    for index in range(32):
        transient = tmp_path / f"transient-{index}.txt"
        transient.write_text(str(index), encoding="utf-8")
        transient_guid = asset_db.import_asset(str(transient)).guid
        assert transient_guid
        assert asset_db.delete_asset(str(transient))
        transient.unlink()

    stop.set()
    for reader in readers:
        reader.join(timeout=5)
        assert not reader.is_alive()

    assert errors == []
    assert asset_db.query_generation >= initial_generation + 64
    assert asset_db.asset_count == len(asset_db.get_all_guids())

    retained_meta = asset_db.get_meta_by_guid(stable_guid)
    assert retained_meta is not None
    assert asset_db.delete_asset(str(stable_path))
    assert asset_db.get_meta_by_guid(stable_guid) is None
    assert retained_meta.get_guid() == stable_guid


@pytest.mark.skipif(os.name != "nt", reason="Windows short-path normalization regression")
def test_delete_removes_reverse_mapping_after_short_path_source_disappears(engine, tmp_path: Path):
    import ctypes

    asset_db = engine.get_asset_database()
    source = tmp_path / "short-path-delete-regression.txt"
    source.write_text("short path", encoding="utf-8")

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(source), buffer, len(buffer))
    if not length or buffer.value == str(source):
        pytest.skip("8.3 short paths are unavailable on this volume")

    result = asset_db.import_asset(buffer.value)
    assert result
    guid = result.guid
    assert guid
    assert asset_db.get_guid_from_path(str(source)) == guid

    source.unlink()
    assert asset_db.delete_asset(str(source))
    assert not asset_db.contains_guid(guid)
    assert not asset_db.contains_path(str(source))
    assert all(entry["guid"] != guid for entry in asset_db.get_directory_catalog(str(tmp_path)))


def test_refresh_builds_import_artifacts_only_on_workers(engine):
    asset_db = engine.get_asset_database()
    graph = AssetDependencyGraph.instance()
    fixture = Path(asset_db.assets_root) / "worker-import-artifact-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    vertex = fixture / "worker.vert"
    fragment = fixture / "worker.frag"
    material = fixture / "worker.mat"
    model = fixture / "worker.obj"
    vertex.write_text("void main() {}", encoding="utf-8")
    fragment.write_text("void main() {}", encoding="utf-8")
    vertex_guid = asset_db.import_asset(str(vertex)).guid
    fragment_guid = asset_db.import_asset(str(fragment)).guid
    assert vertex_guid and fragment_guid
    material_document = json.loads(InxMaterial.create_default_lit().serialize())
    vertex_shader_id = asset_db.get_meta_by_path(str(vertex)).get_string("shader_id")
    fragment_shader_id = asset_db.get_meta_by_path(str(fragment)).get_string(
        "shader_id"
    )
    material_document["shaders"] = {
        "vertex": {
            "guid": vertex_guid,
            "shader_id": vertex_shader_id,
            "path_hint": str(vertex),
        },
        "fragment": {
            "guid": fragment_guid,
            "shader_id": fragment_shader_id,
            "path_hint": str(fragment),
        },
    }
    material.write_text(json.dumps(material_document), encoding="utf-8")
    model.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="ascii",
    )

    paths = (vertex, fragment, material, model)
    try:
        vertex.write_text("void main() { /* refreshed */ }", encoding="utf-8")
        fragment.write_text("void main() { /* refreshed */ }", encoding="utf-8")
        asset_db.refresh()
        assert asset_db.last_refresh_metadata_task_count >= len(paths)
        assert (
            asset_db.last_refresh_worker_metadata_count
            == asset_db.last_refresh_metadata_task_count
        )
        assert asset_db.last_refresh_index_build_on_worker is True
        assert asset_db.last_refresh_importer_task_count >= len(paths)
        assert (
            asset_db.last_refresh_worker_importer_count
            == asset_db.last_refresh_importer_task_count
        )
        assert set(paths).issubset(
            {Path(path) for path in asset_db.last_refresh_imported_paths}
        )

        material_guid = asset_db.get_guid_from_path(str(material))
        assert set(graph.get_dependencies(material_guid)) == {
            vertex_guid,
            fragment_guid,
        }
        model_meta = asset_db.get_meta_by_path(str(model))
        assert model_meta.get_int("mesh_count") == 1
        assert model_meta.get_int("vertex_count") == 3
        assert model_meta.get_int("index_count") == 3
        assert model_meta.get_int("material_slot_count") == 1
        assert model_meta.get_int("bone_count") == 0
        assert model_meta.get_int("animation_count") == 0
    finally:
        for path in paths:
            if asset_db.contains_path(str(path)):
                asset_db.delete_asset(str(path))
            path.unlink(missing_ok=True)
            Path(f"{path}.meta").unlink(missing_ok=True)
        fixture.rmdir()
        asset_db.refresh()


def test_asset_index_reuses_unchanged_assets_and_recovers_from_corruption(engine):
    asset_db = engine.get_asset_database()
    assert Path(asset_db.assets_root).name == "Assets"
    assert Path(asset_db.assets_root).parent.resolve() == Path(asset_db.project_root).resolve()
    fixture_root = Path(asset_db.assets_root) / "asset-index-fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    paths = [fixture_root / f"asset-{index}.txt" for index in range(16)]
    for index, path in enumerate(paths):
        path.write_text(f"initial-{index}", encoding="utf-8")

    try:
        asset_db.refresh()
        assert asset_db.last_refresh_scan_on_worker is True
        assert asset_db.last_refresh_query_build_on_worker is True
        assert asset_db.last_refresh_query_build_ms >= 0.0
        assert asset_db.last_refresh_owner_merge_slice_count >= 3
        assert asset_db.last_refresh_owner_merge_max_slice_ms >= 0.0
        assert asset_db.last_refresh_scanned_count >= len(paths)
        assert asset_db.last_refresh_scan_ms >= 0.0
        assert asset_db.last_refresh_commit_ms >= 0.0
        original_guids = {path: asset_db.get_guid_from_path(str(path)) for path in paths}
        assert all(original_guids.values())
        assert asset_db.last_refresh_imported_count >= len(paths)

        index_path = Path(asset_db.asset_index_path)
        index_document = json.loads(index_path.read_text(encoding="utf-8"))
        assert set(index_document) == {"project_root", "import_revision", "entries"}

        query_generation = asset_db.query_generation
        catalog_generation = asset_db.catalog_generation
        asset_db.refresh()
        assert asset_db.last_refresh_reused_count >= len(paths)
        assert asset_db.last_refresh_imported_paths == []
        assert asset_db.query_generation == query_generation
        assert asset_db.catalog_generation == catalog_generation
        assert asset_db.last_refresh_restore_ms == 0.0
        assert asset_db.last_refresh_import_ms == 0.0
        assert asset_db.last_refresh_index_build_ms == 0.0
        assert asset_db.last_refresh_index_save_ms == 0.0
        assert asset_db.last_refresh_publish_ms == 0.0
        assert {path: asset_db.get_guid_from_path(str(path)) for path in paths} == original_guids

        changed_meta = Path(f"{paths[3]}.meta")
        changed_meta.write_text(
            changed_meta.read_text(encoding="utf-8") + "\n ",
            encoding="utf-8",
        )
        asset_db.refresh()
        assert asset_db.last_refresh_imported_count >= 1
        assert asset_db.last_refresh_reused_count >= len(paths) - 1
        assert asset_db.get_guid_from_path(str(paths[3])) == original_guids[paths[3]]

        paths[5].write_text("changed-content-with-a-different-size", encoding="utf-8")
        asset_db.refresh()
        assert asset_db.last_refresh_imported_count >= 1
        assert asset_db.last_refresh_reused_count >= len(paths) - 1
        assert asset_db.get_guid_from_path(str(paths[5])) == original_guids[paths[5]]

        legacy = json.loads(index_path.read_text(encoding="utf-8"))
        del legacy["import_revision"]
        index_path.write_text(json.dumps(legacy), encoding="utf-8")
        asset_db.refresh()
        assert asset_db.last_refresh_imported_count >= len(paths)
        assert {path: asset_db.get_guid_from_path(str(path)) for path in paths} == original_guids
        rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
        assert set(rebuilt) == {"project_root", "import_revision", "entries"}
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
            Path(f"{path}.meta").unlink(missing_ok=True)
        fixture_root.rmdir()
        asset_db.refresh()


def test_asset_database_async_refresh_commits_worker_artifact(engine):
    asset_db = engine.get_asset_database()
    asset_db.begin_refresh()
    assert asset_db.refresh_pending is True
    with pytest.raises(RuntimeError, match="already pending"):
        asset_db.begin_refresh()

    asset_db.complete_pending_refresh()

    assert asset_db.refresh_pending is False
    assert asset_db.last_refresh_scan_on_worker is True
    if asset_db.last_refresh_imported_count:
        assert asset_db.last_refresh_query_build_on_worker is True

    with pytest.raises(RuntimeError, match="no pending refresh"):
        asset_db.complete_pending_refresh()


def test_async_refresh_hides_prepared_state_until_worker_import_finalize(
    engine, tmp_path: Path
):
    asset_db = engine.get_asset_database()
    asset_db.refresh()
    fixture = Path(asset_db.assets_root) / "pending-import-visibility"
    fixture.mkdir(parents=True, exist_ok=True)
    model = fixture / "pending.obj"
    model.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="ascii",
    )
    blocked_mutation = tmp_path / "blocked-during-import.txt"
    blocked_mutation.write_text("blocked", encoding="utf-8")
    generation_before = asset_db.query_generation
    count_before = asset_db.asset_count

    try:
        asset_db.begin_refresh()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            assert asset_db.try_commit_refresh() is False
            if asset_db.last_refresh_importer_task_count > 0:
                break
            time.sleep(0.001)
        else:
            pytest.fail("refresh never entered its worker importer phase")

        assert asset_db.refresh_pending is True
        assert asset_db.query_generation == generation_before
        assert asset_db.asset_count == count_before
        assert asset_db.contains_path(str(model)) is False
        assert asset_db.get_meta_by_path(str(model)) is None
        assert Path(f"{model}.meta").exists() is False
        with pytest.raises(RuntimeError, match="refresh commit is pending"):
            asset_db.import_asset(str(blocked_mutation))

        while time.monotonic() < deadline:
            if asset_db.try_commit_refresh():
                break
            time.sleep(0.001)
        else:
            pytest.fail("worker importer phase did not finalize")

        assert asset_db.refresh_pending is False
        assert asset_db.query_generation > generation_before
        assert asset_db.asset_count == count_before + 1
        assert asset_db.contains_path(str(model)) is True
        assert asset_db.get_meta_by_path(str(model)).get_int("mesh_count") == 1
        assert Path(f"{model}.meta").is_file()
    finally:
        if asset_db.refresh_pending:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not asset_db.try_commit_refresh():
                time.sleep(0.001)
        if asset_db.contains_path(str(model)):
            asset_db.delete_asset(str(model))
        model.unlink(missing_ok=True)
        Path(f"{model}.meta").unlink(missing_ok=True)
        fixture.rmdir()
        asset_db.refresh()


def test_asset_database_restarts_async_scan_after_owner_mutation(engine):
    asset_db = engine.get_asset_database()
    fixture = Path(asset_db.assets_root) / "owner-mutation-refresh"
    fixture.mkdir(parents=True, exist_ok=True)
    mutation = fixture / "mutation-during-scan.txt"
    asset_db.begin_refresh()

    mutation.write_text("newer owner state", encoding="utf-8")
    guid = asset_db.import_asset(str(mutation)).guid
    assert guid

    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if asset_db.try_commit_refresh():
                break
            time.sleep(0.001)
        else:
            pytest.fail("replacement AssetDatabase scan did not finish")

        assert asset_db.refresh_pending is False
        assert asset_db.get_guid_from_path(str(mutation)) == guid
    finally:
        if asset_db.contains_path(str(mutation)):
            asset_db.delete_asset(str(mutation))
        mutation.unlink(missing_ok=True)
        Path(f"{mutation}.meta").unlink(missing_ok=True)
        fixture.rmdir()
        asset_db.refresh()


def test_refresh_rejects_invalid_metadata_without_rewriting_it(engine):
    asset_db = engine.get_asset_database()
    fixture = Path(asset_db.assets_root) / "prepare-rollback-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    source = fixture / "rollback.txt"
    source.write_text("stable", encoding="utf-8")

    try:
        asset_db.refresh()
        guid = asset_db.get_guid_from_path(str(source))
        meta_path = Path(f"{source}.meta")
        current_metadata = meta_path.read_text(encoding="utf-8")
        invalid_metadata = "{ broken metadata"
        meta_path.write_text(invalid_metadata, encoding="utf-8")

        with pytest.raises(RuntimeError, match="parse error"):
            asset_db.refresh()
        assert asset_db.get_guid_from_path(str(source)) == guid
        assert meta_path.read_text(encoding="utf-8") == invalid_metadata
        meta_path.write_text(current_metadata, encoding="utf-8")
    finally:
        if asset_db.refresh_pending:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not asset_db.try_commit_refresh():
                time.sleep(0.001)
        if asset_db.contains_path(str(source)):
            asset_db.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(f"{source}.meta").unlink(missing_ok=True)
        fixture.rmdir()
        asset_db.refresh()


def test_builtin_read_only_resources_do_not_create_metadata(engine):
    from Infernux.resources import resources_path

    resource_root = Path(resources_path)
    assert engine.get_asset_database().contains_path(str(resource_root / "shaders" / "standard.vert"))
    assert list(resource_root.rglob("*.meta")) == []
