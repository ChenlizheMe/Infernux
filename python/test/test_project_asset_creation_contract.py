"""Project panel creation must publish documents the current loaders accept."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from Infernux.engine.ui import project_file_ops as ops
from Infernux.core.animation_clip import AnimationClip
from Infernux.core.animation_clip3d import AnimationClip3D
from Infernux.core.animation_timeline import AnimationTimeline
from Infernux.core.anim_state_machine import AnimStateMachine
from Infernux.lib import InxMaterial, InxPhysicMaterial


@pytest.fixture
def asset_directory(engine, tmp_path):
    database = engine.get_asset_database()
    directory = Path(database.assets_root) / tmp_path.name
    directory.mkdir()
    try:
        yield database, directory
    finally:
        for asset in tuple(directory.iterdir()):
            if asset.suffix != ".meta":
                if database.contains_path(str(asset)):
                    result = database.delete_asset(str(asset))
                    assert result, result.error
                asset.unlink(missing_ok=True)
                asset.with_name(asset.name + ".meta").unlink(missing_ok=True)
        directory.rmdir()


@pytest.mark.parametrize("creator,extension,loader", [
    (ops.create_animclip, ".animclip2d", AnimationClip),
    (ops.create_animclip3d, ".animclip3d", AnimationClip3D),
    (ops.create_animfsm, ".animfsm", AnimStateMachine),
    (ops.create_timelinefsm, ".timelinefsm", AnimStateMachine),
    (ops.create_animtimeline, ".animtimeline", AnimationTimeline),
])
def test_created_animation_is_complete_and_importable(asset_directory, creator, extension, loader):
    database, directory = asset_directory
    name = "新资产 {01}"
    success, error = creator(str(directory), name + extension, database)
    assert success, error
    path = directory / (name + extension)
    original = path.read_bytes()
    document = json.loads(original)
    loaded = loader.from_dict(document)
    assert loaded.to_dict() == document
    assert loaded.name == name
    if extension == ".timelinefsm":
        assert loaded.mode == "timeline"
    assert original.endswith(b"\n") and b"\r" not in original
    guid = database.get_guid_from_path(str(path))
    assert len(guid) == 32
    database.refresh()
    assert database.get_guid_from_path(str(path)) == guid
    assert path.read_bytes() == original


def test_created_scene_and_material_use_current_native_document_contract(scene, asset_directory):
    database, directory = asset_directory
    success, path = ops.create_scene(str(directory), "新场景 {01}", database)
    assert success, path
    scene_document = json.loads(Path(path).read_bytes())
    from Infernux.engine.component_restore import deserialize_scene_document_transactionally

    assert deserialize_scene_document_transactionally(scene, scene_document, database)
    assert scene.serialize_document()["name"] == "新场景 {01}"
    assert scene_document["nextObjectId"] == scene_document["nextComponentId"] == 1

    success, error = ops.create_material(str(directory), "共享材质 {01}", database)
    assert success, error
    document = json.loads((directory / "共享材质 {01}.mat").read_bytes())
    material = InxMaterial()
    assert material.deserialize_document(document)
    assert document["shaders"]["fragment"]["shader_id"] == "Lit"
    assert document["builtin"] is False
    assert document["properties"]["baseColor"]["type"] == 7

    success, error = ops.create_physic_material(str(directory), "接触材质", database)
    assert success, error
    document = json.loads((directory / "接触材质.physicMaterial").read_bytes())
    physical = InxPhysicMaterial()
    physical.deserialize_document(document)
    assert physical.serialize_document() == document


def test_text_reimport_updates_local_statistics_without_changing_shared_meta(asset_directory):
    database, directory = asset_directory
    success, path = ops.create_scene(str(directory), "Original", database)
    assert success, path
    path = Path(path)
    meta = path.with_name(path.name + ".meta")
    original_meta = meta.read_bytes()
    guid = database.get_guid_from_path(str(path))
    document = json.loads(path.read_bytes())
    document["name"] = "A much longer shared scene name"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = database.reimport_asset(str(path))
    assert result, result.error
    observed = database.get_meta_by_guid(guid).serialize_document()["metadata"]
    assert observed["character_count"]["value"] == path.stat().st_size
    assert meta.read_bytes() == original_meta
    stored = json.loads(original_meta)["metadata"]
    assert "character_count" not in stored and "file_size" not in stored


def test_model_read_write_import_setting_remains_shared_in_meta(asset_directory):
    from Infernux.core.asset_types import read_mesh_import_settings
    from Infernux.core.assets import AssetManager

    database, directory = asset_directory
    source = directory / "Readable.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8", newline="\n")
    imported = database.import_asset(str(source))
    assert imported, imported.error
    settings = read_mesh_import_settings(str(source))
    settings.is_readable = True
    reimported = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert reimported, reimported.error
    meta = source.with_name(source.name + ".meta")
    before = meta.read_bytes()
    fields = json.loads(before)["metadata"]
    assert fields["is_readable"]["value"] is True
    assert "file_size" not in fields
    database.refresh()
    assert read_mesh_import_settings(str(source)) == settings
    assert meta.read_bytes() == before


@pytest.mark.parametrize("creator", [
    ops.create_folder, ops.create_scene, ops.create_material, ops.create_physic_material,
    ops.create_render_texture, ops.create_animclip, ops.create_animclip3d,
    ops.create_animfsm, ops.create_timelinefsm, ops.create_animtimeline,
    ops.create_particlegraph, ops.create_render_effect_group,
])
@pytest.mark.parametrize("name", ["../Escape", "sub\\Escape", "CON", "NUL.mat", "bad:name", ".scene"])
def test_asset_creation_rejects_nonportable_names_before_writing(tmp_path, creator, name):
    success, error = creator(str(tmp_path), name)
    assert not success and error
    assert not list(tmp_path.iterdir())


def test_script_template_uses_explicit_serialization_and_valid_class_names(tmp_path):
    assert not ops.create_script(str(tmp_path), "class.py")[0]
    success, error = ops.create_script(str(tmp_path), "新组件.py")
    assert success, error
    source = (tmp_path / "新组件.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert [item.name for item in tree.body if isinstance(item, ast.ClassDef)] == ["新组件"]
    assert "inx.serialized_field" in source
    assert "automatically serialized" not in source


@pytest.mark.parametrize("creator", [ops.create_physic_material, ops.create_render_texture])
def test_asset_creation_reports_failed_import_instead_of_success(tmp_path, monkeypatch, creator):
    from Infernux.core.assets import AssetManager
    from types import SimpleNamespace

    monkeypatch.setattr(AssetManager, "import_asset", lambda *args, **kwargs:
                        SimpleNamespace(guid="", error="Importer rejected document"))
    success, error = creator(str(tmp_path), "Rejected", object())
    assert not success and error == "Importer rejected document"


@pytest.mark.parametrize("name", ["CON", "NUL.txt", "../Escape", "sub\\Escape", "bad:name", "Hidden."])
def test_rename_rejects_nonportable_names_without_changing_the_source(tmp_path, name):
    source = tmp_path / "Original.txt"
    source.write_bytes(b"authored contents")
    assert ops.rename_destination(str(source), name) == ""
    assert ops.do_rename(str(source), name) is None
    assert source.read_bytes() == b"authored contents"
    assert list(tmp_path.iterdir()) == [source]


def test_rename_preserves_valid_filename_punctuation(tmp_path):
    source = tmp_path / "Original.mat"
    source.write_bytes(b"authored contents")
    assert ops.rename_destination(str(source), "船体 (红色) {01}") == str(tmp_path / "船体 (红色) {01}.mat")


def test_creation_does_not_overwrite_a_file_published_after_the_ui_check(tmp_path):
    source = tmp_path / "Shared.scene"
    external = b'{"name":"Other author"}\n'
    source.write_bytes(external)
    success, error = ops._write_new_text_asset(str(source), '{"name":"New scene"}\n')
    assert not success and "changed outside" in error
    assert source.read_bytes() == external
