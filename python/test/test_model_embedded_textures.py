"""Model-owned images are reusable, read-only, GUID-addressed Texture assets."""
import base64
import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.lib import AssetRegistry, AssetDependencyGraph
from test_model_material_defaults import imported_model


def embed(document, color=(200, 60, 30), *, name="Embedded Color"):
    output = io.BytesIO()
    Image.new("RGB", (8, 4), color).save(output, format="PNG")
    document["images"] = [{"name": name, "uri": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"].update({
        "baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": 0}})


def records(database, source):
    meta = database.get_meta_by_path(str(source)).serialize_document()["metadata"]
    return json.loads(meta["model_textures"]["value"])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_embedded_texture_is_registered_previewable_reusable_and_cooked(imported_model, asynchronous):
    renderer, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    if asynchronous:
        owner = AssetManager.begin_model_reimport(str(source), read_mesh_import_settings(str(source)))
        deadline = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < deadline
            time.sleep(.002)
    else:
        result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    record, = records(database, source)
    guid = record["guid"]
    path = str(source) + "::subtex:" + guid
    assert database.get_guid_from_path(path) == guid
    metadata = database.get_meta_by_guid(guid).serialize_document()["metadata"]
    assert metadata["artifact_width"]["value"] == 8
    assert metadata["artifact_height"]["value"] == 4
    mesh = AssetRegistry.instance().load_mesh(str(source))
    assert metadata["import_owner_guid"]["value"] == mesh.guid
    texture = AssetRegistry.instance().load_texture_by_guid(guid)
    assert texture is not None and texture.guid == guid
    assert renderer.get_material(0).serialize_document()["properties"]["texSampler"]["guid"] == guid
    assert guid in AssetDependencyGraph.instance().get_dependencies(mesh.guid)

    target = source.with_suffix(".mat")
    target.write_text(json.dumps(mesh.create_material_copy(0).serialize_document()), encoding="utf-8")
    material = AssetManager.import_asset(str(target), database=database)
    assert material, material.error
    assert guid in AssetDependencyGraph.instance().get_dependencies(material.guid)
    assert not Path(path).exists() and not Path(path + ".meta").exists()
    assert not AssetManager.reimport_asset(path, database=database)

    database.flush_derived_index()
    from Infernux.engine.runtime_artifact_catalog import load_asset_index, validate_artifact
    root = Path(database.assets_root).parent
    entry = next(entry for entry in load_asset_index(root) if entry["guid"] == guid)
    binding = validate_artifact(root, entry, root / entry["artifact_path"])
    assert binding["source_guid"] == guid
    assert "::subtex:" in binding["source_path"]
    assert entry["read_only"]


def test_embedded_texture_keeps_guid_on_content_edit_refresh_and_model_move(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    before, = records(database, source)
    texture = AssetRegistry.instance().load_texture_by_guid(before["guid"])
    version = AssetRegistry.instance().get_asset_version(before["guid"])
    embed(document, (10, 40, 230))
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    after, = records(database, source)
    assert before["guid"] == after["guid"]
    assert AssetRegistry.instance().load_texture_by_guid(before["guid"]) is texture
    assert AssetRegistry.instance().get_asset_version(before["guid"]) > version
    assert before["metadata"]["metadata"]["content_hash"] != after["metadata"]["metadata"]["content_hash"]
    database.refresh()
    assert database.get_guid_from_path(str(source) + "::subtex:" + before["guid"]) == before["guid"]
    moved = source.with_name("Renamed.gltf")
    source.rename(moved)
    assert database.move_asset(str(source), str(moved))
    assert database.get_guid_from_path(str(moved) + "::subtex:" + before["guid"]) == before["guid"]
    assert not database.get_guid_from_path(str(source) + "::subtex:" + before["guid"])
    moved.rename(source)
    assert database.move_asset(str(moved), str(source))


def test_same_embedded_image_gets_distinct_color_and_data_views(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 0}
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    views = records(database, source)
    assert {view["semantic"] for view in views} == {"color", "data"}
    assert len({view["guid"] for view in views}) == 2
    for view in views:
        assert view["metadata"]["metadata"]["srgb"]["value"] == (view["semantic"] == "color")
    del document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    assert {view["guid"] for view in records(database, source)} == {view["guid"] for view in views}


def test_failed_embedded_decode_does_not_publish_partial_children(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    before = records(database, source)
    document["images"][0]["uri"] = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert not result
    assert records(database, source) == before


def test_embedded_texture_is_in_picker_and_accepts_guid_or_path_drop(imported_model, monkeypatch):
    from Infernux.engine import project_context
    from Infernux.engine.interaction.object_fields import AssetReferenceCatalog
    from Infernux.engine.ui._inspector_references import _project_texture_guid_and_path
    from Infernux.engine.ui.inspector_material import _texture_display_name
    from Infernux.core.asset_reference_types import asset_type_registry
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    record, = records(database, source)
    guid = record["guid"]
    path = database.get_path_from_guid(guid)
    monkeypatch.setattr(project_context, "_project_root", str(Path(database.assets_root).parent))
    assert any(value == path for label, value in AssetReferenceCatalog().items("Texture", "Embedded Color"))
    for value in (guid, path):
        resolved_guid, resolved_path = _project_texture_guid_and_path(value)
        assert resolved_guid == guid
        assert Path(resolved_path) == Path(path)
    assert not asset_type_registry.require("Texture").incompatibility(guid)
    assert _texture_display_name(database, path) == "Embedded Color"


def test_removing_owner_unregisters_children_without_erasing_material_reference(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    child, = records(database, source)
    target = source.with_suffix(".mat")
    mesh = AssetRegistry.instance().load_mesh(str(source))
    target.write_text(json.dumps(mesh.create_material_copy(0).serialize_document()), encoding="utf-8")
    result = AssetManager.import_asset(str(target), database=database)
    assert result
    source.unlink()
    assert database.delete_asset(str(source))
    assert not database.get_path_from_guid(child["guid"])
    assert json.loads(target.read_text(encoding="utf-8"))["properties"]["texSampler"]["guid"] == child["guid"]


def test_project_selection_roundtrips_and_texture_fields_accept_owned_path():
    from Infernux.engine._bootstrap_selection import _project_selection_target, _project_path_for_target
    from Infernux.core.asset_reference_types import asset_type_registry
    path = str(Path("Assets/Model.glb").resolve()) + "::subtex:" + "a" * 32
    target = _project_selection_target(path)
    assert target.sub_kind == "subtexture"
    assert _project_path_for_target(target) == path
    for name in ("Texture", "Texture.Sampled"):
        assert not asset_type_registry.require(name).incompatibility(path)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_owned_texture_import_settings_publish_in_model_sidecar(imported_model, asynchronous):
    from Infernux.core.asset_types import (
        read_texture_import_settings, FilterMode, WrapMode, TextureCompression,
    )
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    before, = records(database, source)
    path = database.get_path_from_guid(before["guid"])
    settings = read_texture_import_settings(path)
    settings.max_size = 4
    settings.filter_mode = FilterMode.POINT
    settings.wrap_mode = WrapMode.CLAMP
    settings.srgb = False
    settings.generate_mipmaps = False
    settings.compression = TextureCompression.NONE
    if asynchronous:
        owner = AssetManager.begin_model_reimport(path, settings)
        deadline = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < deadline
            time.sleep(.002)
    else:
        result = AssetManager.reimport_asset(path, import_settings=settings.to_dict(), database=database)
    assert result, result.error
    assert read_texture_import_settings(path) == settings
    after, = records(database, source)
    assert after["guid"] == before["guid"]
    assert after["metadata"]["metadata"]["artifact_width"]["value"] == 4
    assert after["metadata"]["metadata"]["artifact_height"]["value"] == 2
    stored = json.loads(Path(str(source) + ".meta").read_text(encoding="utf-8"))
    assert json.loads(stored["metadata"]["model_textures"]["value"])[0] == after
    assert not Path(path + ".meta").exists()
    embed(document, (20, 40, 60))
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    assert read_texture_import_settings(path) == settings
    database.refresh()
    assert read_texture_import_settings(path) == settings


def test_owned_texture_failed_settings_preserve_owner_and_identity(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    child, = records(database, source)
    path = database.get_path_from_guid(child["guid"])
    sidecar = Path(str(source) + ".meta")
    old = sidecar.read_bytes()
    database.begin_model_reimport(path, {"texture_compression": "not-supported"})
    deadline = time.monotonic() + 30
    while (result := database.try_commit_model_reimport()) is None:
        assert time.monotonic() < deadline
        time.sleep(.002)
    assert not result
    assert "texture_compression" in result.error
    assert sidecar.read_bytes() == old
    assert records(database, source) == [child]
    assert database.get_guid_from_path(path) == child["guid"]


def test_owned_texture_inspector_has_its_own_settings_identity(imported_model):
    from Infernux.engine.ui import asset_details_renderer as ui
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    assert AssetManager.reimport_asset(str(source), database=database)
    child, = records(database, source)
    ui._ensure_categories()
    state = ui._State()
    assert state.load(database.get_path_from_guid(child["guid"]), "texture", ui._categories["texture"])
    assert state.meta["guid"] == child["guid"]
    assert state.meta["guid"] != database.get_guid_from_path(str(source))
    assert state.settings.max_size == 2048
