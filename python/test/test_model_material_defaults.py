"""Source materials follow model selection/reimport without erasing author edits."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from Infernux.core.assets import AssetManager


@pytest.fixture
def imported_model(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "MaterialSelection.gltf"
    fixture = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_hierarchy.gltf"
    document = json.loads(fixture.read_text(encoding="utf-8"))
    document["materials"] = [
        {"name": name, "pbrMetallicRoughness": {"baseColorFactor": color}}
        for name, color in (("Red", [1, 0, 0, 1]), ("Green", [0, 1, 0, 1]))
    ]
    document["meshes"].append(copy.deepcopy(document["meshes"][0]))
    for index, mesh in enumerate(document["meshes"]):
        mesh["primitives"][0]["material"] = index
    document["nodes"][3]["mesh"] = 1
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(document), encoding="utf-8")
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    go = scene.create_game_object("Model material defaults")
    renderer = go.add_component("MeshRenderer")._require_cpp_component()
    renderer.set_mesh_asset_guid(imported.guid)
    yield renderer, document, source, database, scene
    scene.destroy_game_object(go)


def color(renderer, slot=0):
    value = renderer.get_material(slot).get_color("baseColor")
    return [value.x, value.y, value.z, value.w]


@pytest.mark.parametrize("apply_mode", ["sync", "async"])
@pytest.mark.parametrize("override", [False, True])
def test_material_creation_mode_controls_artifacts_and_keeps_overrides(imported_model, apply_mode, override):
    import time
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.lib import AssetRegistry, AssetDependencyGraph

    renderer, _, source, database, _ = imported_model
    registry = AssetRegistry.instance()
    mesh = registry.load_mesh(str(source))
    target = source.with_suffix(".mat")
    blue = mesh.create_material_copy(0)
    blue.set_color("baseColor", [0, 0, 1, 1])
    target.write_text(json.dumps(blue.serialize_document()), encoding="utf-8")
    imported = AssetManager.import_asset(str(target), database=database)
    assert imported, imported.error
    settings = read_mesh_import_settings(str(source))
    settings.material_remaps["material/Green"] = imported.guid
    if override:
        renderer.set_material(0, imported.guid)
    original_vertices = mesh.vertex_count
    original_names = mesh.material_slot_names

    for mode in ("description", "none", "description"):
        settings.material_import_mode = mode
        if apply_mode == "sync":
            result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        else:
            owner = AssetManager.begin_model_reimport(str(source), settings)
            deadline = time.monotonic() + 30
            while (result := AssetManager.poll_model_reimport(owner)) is None:
                assert time.monotonic() < deadline
                time.sleep(.002)
        assert result, result.error
        assert read_mesh_import_settings(str(source)).material_import_mode == mode
        assert read_mesh_import_settings(str(source)).material_remaps == settings.material_remaps
        assert mesh.vertex_count == original_vertices
        assert mesh.material_slot_names == original_names
        assert (imported.guid in AssetDependencyGraph.instance().get_dependencies(mesh.guid)) == (mode == "description")
        if override:
            np.testing.assert_allclose(color(renderer, 0), [0, 0, 1, 1])
        if mode == "none":
            assert mesh.get_material_slot_data() == []
            with pytest.raises(IndexError, match="no imported material"):
                mesh.create_material_copy(0)
            assert renderer.serialize_document()["materials"] == ([imported.guid, None] if override else [None, None])
        else:
            assert len(mesh.get_material_slot_data()) == 2
            np.testing.assert_allclose(color(renderer, 1), [0, 0, 1, 1])
            if not override:
                np.testing.assert_allclose(color(renderer, 0), [1, 0, 0, 1])
        # Read the cooked native artifact, not only the live import publication.
        assert registry.reload_asset(mesh.guid)
        assert bool(registry.load_mesh(str(source)).get_material_slot_data()) == (mode == "description")


def test_disabled_material_import_defers_remap_resolution(imported_model):
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.lib import AssetRegistry, AssetDependencyGraph

    renderer, _, source, database, _ = imported_model
    mesh = AssetRegistry.instance().load_mesh(str(source))
    settings = read_mesh_import_settings(str(source))
    settings.material_import_mode = "none"
    settings.material_remaps = {"material/RemovedSource": "a" * 32}
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert result, result.error
    assert mesh.get_material_slot_data() == []
    assert "a" * 32 not in AssetDependencyGraph.instance().get_dependencies(mesh.guid)
    settings.material_import_mode = "description"
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert not result
    assert read_mesh_import_settings(str(source)).material_import_mode == "none"
    assert mesh.get_material_slot_data() == []


@pytest.mark.parametrize("apply_mode", ["sync", "async"])
def test_material_none_removes_external_texture_dependency(imported_model, apply_mode):
    import time
    from PIL import Image
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.lib import AssetRegistry, AssetDependencyGraph

    renderer, document, source, database, _ = imported_model
    texture = source.with_name("model_color.png")
    Image.new("RGBA", (4, 4), (220, 80, 40, 255)).save(texture)
    imported = AssetManager.import_asset(str(texture), database=database)
    assert imported, imported.error
    document["images"] = [{"uri": texture.name}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    source.write_text(json.dumps(document), encoding="utf-8")
    settings = read_mesh_import_settings(str(source))
    mesh = AssetRegistry.instance().load_mesh(str(source))
    for mode in ("description", "none", "description"):
        settings.material_import_mode = mode
        if apply_mode == "sync":
            result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        else:
            owner = AssetManager.begin_model_reimport(str(source), settings)
            deadline = time.monotonic() + 30
            while (result := AssetManager.poll_model_reimport(owner)) is None:
                assert time.monotonic() < deadline
                time.sleep(.002)
        assert result, result.error
        assert (imported.guid in AssetDependencyGraph.instance().get_dependencies(mesh.guid)) == (mode == "description")
        if mode == "description":
            assert mesh.get_material_slot_data()[0]["base_color_texture_guid"] == imported.guid
            for material in (mesh.create_material_copy(0), renderer.get_material(0)):
                assert material.serialize_document()["properties"]["texSampler"]["guid"] == imported.guid
            assert AssetRegistry.instance().reload_asset(mesh.guid)
            assert mesh.get_material_slot_data()[0]["base_color_texture_guid"] == imported.guid


def test_model_and_texture_first_scan_share_unpublished_guid_catalog(imported_model):
    import time
    from PIL import Image
    from Infernux.lib import AssetRegistry, AssetDependencyGraph

    _, document, source, database, _ = imported_model
    fresh = source.parent / "Fresh Textured Model.gltf"
    image = fresh.with_suffix(".png")
    Image.new("RGBA", (4, 4), (60, 200, 80, 255)).save(image)
    document["images"] = [{"uri": image.name}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    fresh.write_text(json.dumps(document), encoding="utf-8")
    assert not Path(str(image) + ".meta").exists()
    assert not Path(str(fresh) + ".meta").exists()
    database.begin_refresh()
    deadline = time.monotonic() + 60
    while not database.try_commit_refresh():
        assert time.monotonic() < deadline
        time.sleep(.002)
    guid = database.get_guid_from_path(str(image))
    mesh = AssetRegistry.instance().load_mesh(str(fresh))
    assert mesh and guid
    assert mesh.get_material_slot_data()[0]["base_color_texture_guid"] == guid
    assert guid in AssetDependencyGraph.instance().get_dependencies(mesh.guid)
    assert mesh.create_material_copy(0).serialize_document()["properties"]["texSampler"]["guid"] == guid


def test_unregistered_model_texture_rejects_publication_but_none_does_not_use_it(imported_model):
    from Infernux.core.asset_types import read_mesh_import_settings
    renderer, document, source, database, _ = imported_model
    before = Path(str(source) + ".meta").read_bytes()
    document["images"] = [{"uri": "MissingTexture.png"}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert not result and "not a registered project asset" in result.error
    assert Path(str(source) + ".meta").read_bytes() == before
    np.testing.assert_allclose(color(renderer, 0), [1, 0, 0, 1])
    settings = read_mesh_import_settings(str(source))
    settings.material_import_mode = "none"
    result = AssetManager.reimport_asset(str(source), database=database, import_settings=settings.to_dict())
    assert result, result.error


@pytest.mark.parametrize("value", [True, 1, "legacy", "", None])
def test_invalid_material_import_mode_rejected_before_publication(imported_model, value):
    from Infernux.core.asset_types import read_mesh_import_settings, MeshImportSettings
    _, _, source, database, _ = imported_model
    before = source.with_suffix(source.suffix + ".meta").read_bytes()
    data = read_mesh_import_settings(str(source)).to_dict()
    data["material_import_mode"] = value
    with pytest.raises(ValueError, match="material_import_mode"):
        MeshImportSettings.from_dict(data)
    with pytest.raises(ValueError, match="material_import_mode"):
        AssetManager.reimport_asset(str(source), import_settings=data, database=database)
    assert source.with_suffix(source.suffix + ".meta").read_bytes() == before


def test_material_copy_matches_renderer_without_sharing_edits(imported_model):
    from Infernux.lib import AssetRegistry

    renderer, _, source, _, _ = imported_model
    mesh = AssetRegistry.instance().load_mesh(str(source))
    material = mesh.create_material_copy(1)
    assert material.name == "Green"
    assert material.serialize_document() == renderer.get_material(1).serialize_document()
    material.set_color("baseColor", [0, 0, 1, 1])
    np.testing.assert_allclose(color(renderer, 1), [0, 1, 0, 1])
    with pytest.raises(IndexError, match="no imported material"):
        mesh.create_material_copy(999)


@pytest.mark.parametrize("alpha_mode", ["OPAQUE", "MASK", "BLEND"])
@pytest.mark.parametrize("double_sided", [False, True])
def test_source_surface_survives_import_copy_and_binary_reload(imported_model, alpha_mode, double_sided):
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.core.material import Material
    from Infernux.lib import AssetRegistry, InxMaterial

    renderer, document, source, database, _ = imported_model
    authored = document["materials"][0]
    authored.update(alphaMode=alpha_mode, alphaCutoff=0.37, doubleSided=double_sided)
    authored["pbrMetallicRoughness"]["baseColorFactor"][3] = 0.4
    source.write_text(json.dumps(document), encoding="utf-8")
    settings = read_mesh_import_settings(str(source))
    settings.is_readable = True
    result = AssetManager.reimport_asset(
        str(source), import_settings=settings.to_dict(), database=database
    )
    assert result, result.error
    registry = AssetRegistry.instance()
    mesh = registry.load_mesh(str(source))
    data = mesh.get_material_slot_data()[0]
    assert data["alpha_mode"] == alpha_mode.lower()
    assert data["double_sided"] is double_sided
    assert data["alpha_cutoff"] == pytest.approx(0.37)
    assert data["base_color"][3] == pytest.approx(0.4)  # Not 0.4 squared by Assimp OPACITY.

    def check(material):
        state = material.get_render_state()
        assert state.cull_mode == (0 if double_sided else 2)
        assert state.blend_enable is (alpha_mode == "BLEND")
        assert state.depth_write_enable is (alpha_mode != "BLEND")
        assert state.alpha_clip_enabled is (alpha_mode == "MASK")
        assert state.render_queue == {"OPAQUE": 2000, "MASK": 2450, "BLEND": 3000}[alpha_mode]
        assert material.get_float("_AlphaClipThreshold") == pytest.approx(0.37 if alpha_mode == "MASK" else 0)
        assert material.get_color("baseColor").w == pytest.approx(1 if alpha_mode == "OPAQUE" else 0.4)
        if alpha_mode == "BLEND":
            assert state.src_color_blend_factor == 6
            assert state.dst_color_blend_factor == 7
            assert state.src_alpha_blend_factor == 1
            assert state.dst_alpha_blend_factor == 7

    check(renderer.get_material(0))
    check(mesh.create_material_copy(0))
    check(Material.load(f"{source}::submat:0").native)
    restored = InxMaterial.create_default_lit()
    assert restored.deserialize_document(mesh.create_material_copy(0).serialize_document())
    check(restored)  # Standalone extracted material serialization.
    assert registry.reload_asset(mesh.guid)
    check(registry.load_mesh(str(source)).create_material_copy(0))

    # Authored native mesh and Player-oriented binary data retain the same surface.
    native_source = source.with_suffix(".inxmesh")
    native_source.write_bytes(mesh.serialize_source())
    imported = AssetManager.import_asset(str(native_source), database=database)
    assert imported, imported.error
    check(registry.load_mesh(str(native_source)).create_material_copy(0))

    # Reimport resets source defaults, rather than retaining stale transparent state.
    authored.update(alphaMode="OPAQUE", doubleSided=False)
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    state = renderer.get_material(0).get_render_state()
    assert not state.blend_enable and not state.alpha_clip_enabled
    assert state.depth_write_enable and state.cull_mode == 2 and state.render_queue == 2000


def test_material_extraction_undo_redo_and_independent_asset(imported_model):
    from Infernux.lib import AssetRegistry
    from Infernux.engine.interaction import EditorActionJournal, ProjectAssetCommandService, SelectionService
    from Infernux.engine.undo import UndoManager

    renderer, _, source, database, _ = imported_model
    registry = AssetRegistry.instance()
    mesh = registry.load_mesh(str(source))
    target = source.with_name("Extracted.mat")
    occupied = target.with_name("Occupied.mat")
    previous_manager = UndoManager._instance
    manager = UndoManager(EditorActionJournal())
    service = ProjectAssetCommandService(SelectionService())
    service.configure(str(Path(database.assets_root).parent), database)
    original = renderer.serialize_document()
    try:
        assert Path(service.extract_model_material(mesh, 1, str(target))) == target
        guid = database.get_guid_from_path(str(target))
        assert guid and guid != mesh.guid
        content = target.read_bytes()
        saved = registry.load_material_by_guid(guid)
        expected = renderer.get_material(1).serialize_document()
        expected["name"] = target.stem  # Imported assets take their authored filename.
        assert saved.serialize_document() == expected
        assert renderer.serialize_document() == original  # No implicit remap/scene edit.
        for slot in (-1, True, "1"):
            with pytest.raises(ValueError, match="non-negative integer"):
                service.extract_model_material(mesh, slot, str(target))
        with pytest.raises(FileExistsError):
            service.extract_model_material(mesh, 1, str(target))
        with pytest.raises(ValueError, match=".mat"):
            service.extract_model_material(mesh, 1, str(target.with_suffix(".txt")))
        outside = Path(database.assets_root).parent / "Outside.mat"
        with pytest.raises(ValueError, match="under Assets or Packages"):
            service.extract_model_material(mesh, 1, str(outside))
        assert not outside.exists()
        manager.undo()
        assert not target.exists()
        assert not database.contains_guid(guid)
        manager.redo()
        assert database.get_guid_from_path(str(target)) == guid
        assert target.read_bytes() == content
        class ConcurrentTarget:
            def serialize_document(self):
                occupied.write_bytes(b"another author's file")
                return saved.serialize_document()

        with pytest.raises(RuntimeError):
            service.save_material_copy(ConcurrentTarget(), str(occupied))
        assert occupied.read_bytes() == b"another author's file"
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager
        if database.contains_path(str(target)):
            database.delete_asset(str(target))
        occupied.unlink(missing_ok=True)


def test_material_save_dialog_captures_source_before_reimport(imported_model, monkeypatch):
    from types import SimpleNamespace
    from Infernux.lib import AssetRegistry
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.ui import asset_details_renderer as inspector, asset_save_dialog

    _, document, source, database, _ = imported_model
    mesh = AssetRegistry.instance().load_mesh(str(source))
    captured = {}
    saved = []

    class Dialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, **kwargs):
            captured.update(kwargs)

    service = SimpleNamespace(project_root=str(source.parent), save_material_copy=lambda material, path: (
        saved.append(material.serialize_document()) or path))
    monkeypatch.setattr(EditorInteractionCore, "instance", lambda: SimpleNamespace(project_assets=service))
    monkeypatch.setattr(asset_save_dialog, "AssetSaveAsDialog", Dialog)
    expected = mesh.create_material_copy(1).serialize_document()
    inspector._request_model_material_extraction(inspector._State(), mesh, 1)
    document["materials"][1]["name"] = "Changed source"
    document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert captured["default_name"] == "Green"
    assert captured["save_callback"](str(source.with_suffix(".mat")))
    assert saved == [expected]
    assert mesh.create_material_copy(1).serialize_document() != expected


def select(renderer, index, mode):
    if mode == "submesh":
        renderer.submesh_index = index
    else:
        document = renderer.serialize_document()
        if index >= 0:
            document["nodeGroup"] = index
        else:
            document.pop("nodeGroup", None)
        assert renderer.deserialize_document(document)


@pytest.mark.parametrize("mode", ["node", "submesh"])
def test_model_material_selection_and_scene_restore(imported_model, mode):
    renderer, _, _, _, scene = imported_model
    assert renderer.material_count == 2
    np.testing.assert_allclose(color(renderer, 0), [1, 0, 0, 1])
    np.testing.assert_allclose(color(renderer, 1), [0, 1, 0, 1])
    assert renderer.serialize_document()["materials"] == [None, None]
    for index in (1, 0, 1):
        select(renderer, index, mode)
        assert renderer.material_count == 1
        np.testing.assert_allclose(color(renderer), [0, 1, 0, 1] if index else [1, 0, 0, 1])
    saved = renderer.serialize_document()
    assert saved["materials"] == [None]
    restored = scene.create_game_object("Restored").add_component("MeshRenderer")._require_cpp_component()
    saved["component_id"] = restored.component_id
    assert restored.deserialize_document(saved)
    np.testing.assert_allclose(color(restored), [0, 1, 0, 1])
    select(restored, -1, mode)
    assert restored.material_count == 2
    np.testing.assert_allclose(color(restored, 0), [1, 0, 0, 1])
    np.testing.assert_allclose(color(restored, 1), [0, 1, 0, 1])


@pytest.mark.parametrize("override", ["none", "edit", "assign_same", "guid", "inline_document"])
def test_reimport_updates_only_unmodified_source_materials(imported_model, override):
    renderer, source_document, source, database, scene = imported_model
    renderer.submesh_index = 1
    expected = [0, 0, 1, 1]
    if override == "edit":
        renderer.get_material(0).set_color("baseColor", [1, 0, 1, 1])
        expected = [1, 0, 1, 1]
    elif override == "assign_same":
        renderer.set_material(0, renderer.get_material(0))
        expected = [0, 1, 0, 1]
    elif override == "guid":
        target = source.with_suffix(".mat")
        material = renderer.get_material(0).serialize_document()
        target.write_text(json.dumps(material), encoding="utf-8")
        result = AssetManager.import_asset(str(target), database=database)
        assert result, result.error
        renderer.set_material(0, result.guid)
        expected = [0, 1, 0, 1]
    elif override == "inline_document":
        # Existing scene snapshots are authored data; never infer their origin
        # from names or file paths and silently replace them on reimport.
        saved = renderer.serialize_document()
        saved["materials"] = [{"material": renderer.get_material(0).serialize_document()}]
        assert renderer.deserialize_document(saved)
        expected = [0, 1, 0, 1]

    saved = renderer.serialize_document()
    assert (saved["materials"] == [None]) == (override == "none")
    restored = scene.create_game_object("Restored override").add_component("MeshRenderer")._require_cpp_component()
    saved["component_id"] = restored.component_id
    assert restored.deserialize_document(saved)
    source_document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    source.write_text(json.dumps(source_document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    np.testing.assert_allclose(color(renderer), expected)
    np.testing.assert_allclose(color(restored), expected)
    assert (renderer.serialize_document()["materials"] == [None]) == (override == "none")


def test_clearing_model_keeps_existing_inline_appearance(imported_model):
    renderer, _, _, _, _ = imported_model
    renderer.clear_mesh_asset()
    saved = renderer.serialize_document()
    assert "meshAssetGuid" not in saved
    assert all("material" in slot for slot in saved["materials"])


def test_unchanged_model_materials_keep_runtime_instances(imported_model):
    renderer, _, source, database, _ = imported_model
    before = [renderer.get_material(index) for index in range(2)]
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert [renderer.get_material(index) for index in range(2)] == before
    assert renderer.serialize_document()["materials"] == [None, None]


def test_model_material_remap_follows_source_not_slot_and_preserves_overrides(imported_model):
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.lib import AssetRegistry, AssetDependencyGraph

    renderer, document, source, database, scene = imported_model
    mesh = AssetRegistry.instance().load_mesh(str(source))
    assert [row["source_id"] for row in mesh.get_material_slot_data()] == ["material/Red", "material/Green"]
    target = source.with_suffix(".mat")
    blue = mesh.create_material_copy(0)
    blue.set_color("baseColor", [0, 0, 1, 1])
    target.write_text(json.dumps(blue.serialize_document()), encoding="utf-8")
    imported = AssetManager.import_asset(str(target), database=database)
    assert imported, imported.error
    settings = read_mesh_import_settings(str(source))
    settings.material_remaps["material/Green"] = imported.guid
    assert AssetManager.apply_import_settings("mesh", str(source), settings)
    assert read_mesh_import_settings(str(source)).material_remaps == settings.material_remaps
    assert imported.guid in AssetDependencyGraph.instance().get_dependencies(mesh.guid)
    np.testing.assert_allclose(color(renderer, 1), [0, 0, 1, 1])
    assert renderer.serialize_document()["materials"] == [None, None]
    restored = scene.create_game_object("Remapped restored").add_component("MeshRenderer")._require_cpp_component()
    saved = renderer.serialize_document()
    saved["component_id"] = restored.component_id
    assert restored.deserialize_document(saved)
    np.testing.assert_allclose(color(restored, 1), [0, 0, 1, 1])
    # Reorder source objects so the Green source now occupies renderer slot 0.
    document["nodes"][1]["children"] = [3, 2]
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert mesh.get_material_slot_data()[0]["source_id"] == "material/Green"
    np.testing.assert_allclose(color(renderer, 0), [0, 0, 1, 1])
    np.testing.assert_allclose(color(renderer, 1), [1, 0, 0, 1])
    # Explicit assignment of even the currently inherited material is an override.
    renderer.set_material(0, imported.guid)
    settings.material_remaps.clear()
    assert AssetManager.apply_import_settings("mesh", str(source), settings)
    assert renderer.serialize_document()["materials"] == [imported.guid, None]
    assert imported.guid not in AssetDependencyGraph.instance().get_dependencies(mesh.guid)
    np.testing.assert_allclose(color(renderer, 0), [0, 0, 1, 1])
    np.testing.assert_allclose(color(restored, 0), [0, 1, 0, 1])


@pytest.mark.parametrize("failure", ["missing_source", "ambiguous_source", "missing_guid", "wrong_type"])
def test_invalid_model_remap_does_not_publish_settings_or_geometry(imported_model, failure):
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.lib import AssetRegistry

    renderer, document, source, database, _ = imported_model
    mesh = AssetRegistry.instance().load_mesh(str(source))
    target = source.with_suffix(".mat")
    target.write_text(json.dumps(mesh.create_material_copy(0).serialize_document()), encoding="utf-8")
    imported = AssetManager.import_asset(str(target), database=database)
    assert imported, imported.error
    original_meta = Path(str(source) + ".meta").read_bytes()
    original_scene = renderer.serialize_document()
    original_generation = mesh.generation
    settings = read_mesh_import_settings(str(source))
    key = "material/Missing" if failure == "missing_source" else "material/Green"
    guid = ("0" * 32 if failure == "missing_guid" else mesh.guid if failure == "wrong_type" else imported.guid)
    settings.material_remaps[key] = guid
    if failure == "ambiguous_source":
        document["materials"][0]["name"] = "Green"
        source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict())
    assert not result
    assert ("missing or ambiguous" if failure.endswith("source") else "registered Material") in result.error
    assert Path(str(source) + ".meta").read_bytes() == original_meta
    assert renderer.serialize_document() == original_scene
    assert mesh.generation == original_generation


def test_material_remap_settings_copy_and_complete_contract():
    from Infernux.core.asset_types import MeshImportSettings

    settings = MeshImportSettings()
    clone = settings.copy()
    clone.material_remaps["material/Unique"] = "f" * 32
    assert settings.material_remaps == {}
    assert MeshImportSettings.from_dict(clone.to_dict()).material_remaps == clone.material_remaps
    incomplete = settings.to_dict()
    del incomplete["material_remaps"]
    with pytest.raises(ValueError, match="complete current field set"):
        MeshImportSettings.from_dict(incomplete)
