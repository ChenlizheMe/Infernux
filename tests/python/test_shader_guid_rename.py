"""Typed shader IDs can change while imported GUIDs and bindings remain stable."""

import json
from pathlib import Path
import time
import uuid

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.core.shader import Shader
from infernux.lib import AssetDependencyGraph, AssetRegistry, InxMaterial


_SHARED_FIELD = "shader.material.float(default=0.25)"


def _vertex_source(shader_id, field=None):
    declaration = f"    renameSignal = {field}\n" if field else ""
    return f"""import infernux.shader as shader

class RenameVertex(shader.vert):
    shader_id = {shader_id!r}
{declaration}
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex
"""


def _surface_source(shader_id, *, shared_field=False):
    declaration = f"    renameSignal = {_SHARED_FIELD}\n" if shared_field else ""
    albedo = "shader.vec3(self.renameSignal)" if shared_field else "shader.vec3(0.1, 0.2, 0.3)"
    return f"""import infernux.shader as shader

class RenameSurface(shader.surface):
    shader_id = {shader_id!r}
    shading_model = shader.ref("infernux/shading/unlit")
    smoothness = shader.material.float(default=0.5)
{declaration}
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo={albedo}, smoothness=self.smoothness)
"""


@pytest.fixture
def rename_assets(engine, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", AssetRegistry.instance())
    directory = Path(database.assets_root)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "Rename_" + uuid.uuid4().hex
    paths = []

    def create(extension, source):
        path = directory / (prefix + extension)
        path.write_text(source, encoding="utf-8")
        paths.append(path)
        imported = AssetManager.import_asset(str(path))
        assert imported, imported.error
        return path, imported.guid

    try:
        vertex_name, fragment_name = prefix + " Vertex", prefix + " Fragment"
        vertex_source, fragment_source = _vertex_source(vertex_name), _surface_source(fragment_name)
        vertex, vertex_guid = create(".vert.py", vertex_source)
        fragment, fragment_guid = create(".frag.py", fragment_source)
        document = InxMaterial.create_default_lit().serialize_document()
        document["builtin"] = False
        document["name"] = prefix
        document["shaders"] = {
            "vertex": {"guid": vertex_guid, "shader_id": vertex_name},
            "fragment": {"guid": fragment_guid, "shader_id": fragment_name},
        }
        material_path, material_guid = create(".mat", json.dumps(document))
        yield {
            "database": database,
            "material_guid": material_guid,
            "material_path": material_path,
            "create": create,
            "vertex": (vertex, vertex_guid, vertex_name, vertex_source),
            "fragment": (fragment, fragment_guid, fragment_name, fragment_source),
        }
    finally:
        for path in reversed(paths):
            path.unlink(missing_ok=True)
            if database.get_guid_from_path(str(path)):
                result = AssetManager.delete_asset(str(path))
                assert result, result.error


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_shader_rename_preserves_loaded_guid_material_in_place(engine, rename_assets, stage):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    assert material is not None
    native = material._native
    assert engine.refresh_material_pipeline(native)
    native.set_float("smoothness", 0.37)
    path, guid, old_name, source = assets[stage]
    dependencies = AssetDependencyGraph.instance()
    assert dependencies.has_dependency(assets["material_guid"], guid)
    new_name = old_name + " V2"
    path.write_text(source.replace(old_name, new_name), encoding="utf-8")
    imported = AssetManager.reimport_asset(str(path))
    assert imported and imported.guid == guid, imported.error
    assert AssetManager.load(assets["material_guid"], Material) is material
    assert material._native is native
    assert native.serialize_document()["shaders"][stage]["guid"] == guid
    assert dependencies.has_dependency(assets["material_guid"], guid)
    assert assets["database"].get_meta_by_guid(guid).get_string("shader_id") == new_name
    assert native.get_float("smoothness") == pytest.approx(0.37)
    assert engine.refresh_material_pipeline(native)
    assert Shader.is_loaded(guid, stage) and Shader.is_loaded(new_name, stage)
    assert not Shader.is_loaded(old_name, stage)
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored and restored.guid == guid, restored.error
    assert native.serialize_document()["shaders"][stage]["guid"] == guid
    assert engine.refresh_material_pipeline(native)
    assert Shader.is_loaded(old_name, stage) and not Shader.is_loaded(new_name, stage)


@pytest.mark.parametrize("rendered,stage", [(True, "vertex"), (True, "fragment")])
def test_shader_rename_updates_runtime_clone_without_replacing_it(engine, rename_assets, stage, rendered):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    assert material is not None
    assert engine.refresh_material_pipeline(material._native)
    clone = material.clone()
    native = clone._native
    native.set_float("smoothness", 0.23)
    assert not clone.guid
    if rendered:
        assert engine.refresh_material_pipeline(clone._native)
    path, guid, old_name, source = assets[stage]
    new_name = old_name + " V2"
    path.write_text(source.replace(old_name, new_name), encoding="utf-8")
    imported = AssetManager.reimport_asset(str(path))
    assert imported and imported.guid == guid, imported.error
    assert clone._native is native
    assert native.serialize_document()["shaders"][stage]["guid"] == guid
    assert Shader.is_loaded(new_name, stage)
    assert native.get_float("smoothness") == pytest.approx(0.23)
    assert engine.refresh_material_pipeline(native)
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored, restored.error
    assert clone._native is native
    assert native.serialize_document()["shaders"][stage]["guid"] == guid
    assert engine.refresh_material_pipeline(clone._native)


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_failed_shader_rename_keeps_live_bindings_and_recovers(engine, rename_assets, stage):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    clone = material.clone()
    assert engine.refresh_material_pipeline(material._native)
    assert engine.refresh_material_pipeline(clone._native)
    before = [value._native.serialize_document() for value in (material, clone)]
    path, _, old_name, source = assets[stage]
    new_name = old_name + " Invalid"
    changed = source.replace(old_name, new_name)
    if stage == "vertex":
        changed = changed.replace("return vertex", "return missing_rename_symbol")
    else:
        changed = changed.replace("shader.vec3(0.1, 0.2, 0.3)", "shader.vec3(missing_rename_symbol)")
    path.write_text(changed, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(path))
    assert not rejected
    assert "missing_rename_symbol" in rejected.error
    assert [value._native.serialize_document() for value in (material, clone)] == before
    assert not engine.is_shader_loaded(new_name, stage)
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored, restored.error
    assert engine.refresh_material_pipeline(material._native)
    assert engine.refresh_material_pipeline(clone._native)
    assert [value._native.serialize_document() for value in (material, clone)] == before


def test_unreferenced_standalone_shader_rename_retires_old_name(rename_assets):
    name = rename_assets["fragment"][2] + " Fullscreen"
    source = f"""import infernux.shader as shader

class RenameFullscreen(shader.fullscreen):
    shader_id = {name!r}
    hidden = True

    def main(self, fragment: shader.fragment_input) -> shader.vec4:
        return shader.vec4(1.0, 0.0, 0.0, 1.0)
"""
    path, guid = rename_assets["create"](".fullscreen.frag.py", source)
    assert Shader.reload(guid, "fragment")
    assert Shader.is_loaded(name, "fragment")
    changed_name = name + " V2"
    path.write_text(source.replace(name, changed_name), encoding="utf-8")
    assert Shader.reload(str(path), "fragment")
    assert Shader.is_loaded(guid, "fragment") and Shader.is_loaded(changed_name, "fragment")
    assert not Shader.is_loaded(name, "fragment")
    with pytest.raises(FileNotFoundError, match="not registered"):
        Shader.reload(name, "fragment")
    with pytest.raises(ValueError, match="matching GPU source stage"):
        Shader.reload(guid, "vertex")
    path.write_text(source, encoding="utf-8")
    assert Shader.reload(guid, "fragment")
    assert Shader.is_loaded(name, "fragment")
    assert not Shader.is_loaded(changed_name, "fragment")


def test_shader_rename_preflights_every_dependent_pair(engine, rename_assets):
    assets = rename_assets
    vertex_path, _, old_name, _ = assets["vertex"]
    linked_source = _vertex_source(old_name, _SHARED_FIELD)
    vertex_path.write_text(linked_source, encoding="utf-8")
    imported = AssetManager.reimport_asset(str(vertex_path))
    assert imported, imported.error
    paired_name = assets["fragment"][2] + " With Signal"
    _, paired_guid = assets["create"](
        ".paired.frag.py", _surface_source(paired_name, shared_field=True),
    )
    material = AssetManager.load(assets["material_guid"], Material)
    clone = material.clone()
    document = clone._native.serialize_document()
    document["shaders"]["fragment"] = {"guid": paired_guid, "shader_id": paired_name}
    assert clone._native.deserialize_document(document)
    for value in (material, clone):
        assert engine.refresh_material_pipeline(value._native)
    before = [value._native.serialize_document() for value in (material, clone)]
    # Only the second pair shares renameSignal with the fragment. Its type
    # conflict prevents publication of an otherwise valid vertex edit.
    changed_name = old_name + " V2"
    changed_source = _vertex_source(changed_name, "shader.material.vec3(default=(0.25, 0.25, 0.25))")
    vertex_path.write_text(changed_source, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(vertex_path))
    assert not rejected
    assert "renameSignal" in rejected.error
    assert [value._native.serialize_document() for value in (material, clone)] == before
    assert not engine.is_shader_loaded(changed_name, "vertex")
    vertex_path.write_text(linked_source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(vertex_path))
    assert restored, restored.error
    for value in (material, clone):
        assert engine.refresh_material_pipeline(value._native)


def test_guid_shader_material_loads_on_worker_without_runtime_registration(rename_assets, engine):
    registry = AssetRegistry.instance()
    guid = rename_assets["material_guid"]
    registry.invalidate_asset(guid)
    ticket = registry.begin_load_material_by_guid(guid)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not registry.try_commit_asset_load(ticket):
        engine.tick(.016)
        time.sleep(0.001)
    assert ticket.committed
    assert ticket.produced_on_worker
    material = AssetManager.load(guid, Material)
    assert material is not None
    assert material.guid == guid
    assert material._native.serialize_document()["shaders"]["vertex"]["guid"] == rename_assets["vertex"][1]
