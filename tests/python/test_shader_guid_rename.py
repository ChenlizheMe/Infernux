"""ShaderInfo names are labels of imported GUIDs, including across reload."""
import json
from pathlib import Path
import time
import uuid

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.lib import AssetRegistry, InxMaterial


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
        imported = AssetManager.import_asset(str(path))
        assert imported, imported.error
        paths.append(path)
        return path, imported.guid

    vertex_name, fragment_name = prefix + " Vertex", prefix + " Fragment"
    vertex_source = '#version 450\nShaderInfo { Name "' + vertex_name + '" }\n'
    fragment_source = ('#version 450\nShaderInfo { Name "' + fragment_name +
                       '" ShadingModel Unlit }\nvoid surface(out SurfaceData s) { '
                       's = InitSurfaceData(); s.albedo = vec3(0.1,0.2,0.3); }\n')
    vertex, vertex_guid = create(".vert", vertex_source)
    fragment, fragment_guid = create(".frag", fragment_source)
    document = InxMaterial.create_default_lit().serialize_document()
    document["builtin"] = False
    document["name"] = prefix
    document["shaders"] = {"vertex": {"guid": vertex_guid, "shader_id": vertex_name},
                           "fragment": {"guid": fragment_guid, "shader_id": fragment_name}}
    material_path, material_guid = create(".mat", json.dumps(document))
    yield {"database": database, "material_guid": material_guid,
           "material_path": material_path,
           "create": create,
           "vertex": (vertex, vertex_guid, vertex_name, vertex_source),
           "fragment": (fragment, fragment_guid, fragment_name, fragment_source)}
    for path in reversed(paths):
        result = AssetManager.delete_asset(str(path))
        assert result, result.error


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_material_load_derives_current_shader_name_from_guid(rename_assets, stage):
    assets = rename_assets
    path, guid, old_name, source = assets[stage]
    new_name = old_name + " V2"
    path.write_text(source.replace(old_name, new_name), encoding="utf-8")
    # Simulate loading an existing saved material after a new shader metadata
    # publication, without first having that material resident in the renderer.
    imported = assets["database"].reimport_asset(str(path))
    assert imported, imported.error
    material = AssetManager.load(assets["material_guid"], Material)
    assert material is not None
    reference = material._native.serialize_document()["shaders"][stage]
    assert reference == {"guid": guid, "shader_id": new_name}
    assert json.loads(assets["material_path"].read_text(encoding="utf-8"))["shaders"][stage]["shader_id"] == old_name


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_shader_rename_migrates_loaded_guid_material_in_place(engine, rename_assets, stage):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    assert material is not None
    native = material._native
    assert engine.refresh_material_pipeline(native)
    native.set_float("smoothness", 0.37)
    path, guid, old_name, source = assets[stage]
    new_name = old_name + " V2"
    path.write_text(source.replace(old_name, new_name), encoding="utf-8")
    imported = AssetManager.reimport_asset(str(path))
    assert imported, imported.error
    assert AssetManager.load(assets["material_guid"], Material) is material
    assert material._native is native
    assert native.serialize_document()["shaders"][stage] == {"guid": guid, "shader_id": new_name}
    assert native.get_float("smoothness") == pytest.approx(0.37)
    assert engine.refresh_material_pipeline(native)
    assert not engine.is_shader_loaded(old_name, stage)
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored, restored.error
    assert native.serialize_document()["shaders"][stage] == {"guid": guid, "shader_id": old_name}
    assert engine.refresh_material_pipeline(native)
    assert not engine.is_shader_loaded(new_name, stage)


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
@pytest.mark.parametrize("rendered", [False, True])
def test_shader_rename_updates_runtime_clone_without_replacing_it(engine, rename_assets, stage, rendered):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    assert material is not None
    assert engine.refresh_material_pipeline(material._native)
    clone = material.clone()
    clone._native.set_float("smoothness", 0.23)
    assert not clone.guid
    if rendered:
        assert engine.refresh_material_pipeline(clone._native)
    path, guid, old_name, source = assets[stage]
    new_name = old_name + " V2"
    path.write_text(source.replace(old_name, new_name), encoding="utf-8")
    imported = AssetManager.reimport_asset(str(path))
    assert imported, imported.error
    assert clone._native.serialize_document()["shaders"][stage] == {"guid": guid, "shader_id": new_name}
    assert clone._native.get_float("smoothness") == pytest.approx(0.23)
    assert engine.refresh_material_pipeline(clone._native)
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored, restored.error
    assert clone._native.serialize_document()["shaders"][stage] == {"guid": guid, "shader_id": old_name}
    assert engine.refresh_material_pipeline(clone._native)


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_failed_shader_rename_keeps_live_bindings_and_recovers(engine, rename_assets, stage):
    assets = rename_assets
    material = AssetManager.load(assets["material_guid"], Material)
    clone = material.clone()
    assert engine.refresh_material_pipeline(material._native)
    assert engine.refresh_material_pipeline(clone._native)
    before = [value._native.serialize_document() for value in (material, clone)]
    path, guid, old_name, source = assets[stage]
    new_name = old_name + " Invalid"
    changed = source.replace(old_name, new_name)
    if stage == "vertex":
        changed += '\nvoid vertex(inout VertexInput v) { v.position.x += missing_rename_symbol; }\n'
    else:
        changed = changed.replace('vec3(0.1,0.2,0.3)', 'vec3(missing_rename_symbol)')
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


def test_unreferenced_standalone_shader_rename_retires_old_name(engine, rename_assets):
    name = rename_assets["fragment"][2] + " Fullscreen"
    source = ('#version 450\nShaderInfo { Name "'+name+'" Hidden On Capabilities [Fullscreen] '
              'Outputs { Float4 outColor } }\nvoid main() { outColor = vec4(1,0,0,1); }\n')
    path, _ = rename_assets["create"](".fullscreen.frag", source)
    published = AssetManager.reimport_asset(str(path))
    assert published, published.error
    assert engine.is_shader_loaded(name, "fragment")
    changed_name = name + " V2"
    path.write_text(source.replace(name, changed_name), encoding="utf-8")
    renamed = AssetManager.reimport_asset(str(path))
    assert renamed, renamed.error
    assert engine.is_shader_loaded(changed_name, "fragment")
    assert not engine.is_shader_loaded(name, "fragment")
    path.write_text(source, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(path))
    assert restored, restored.error
    assert engine.is_shader_loaded(name, "fragment")
    assert not engine.is_shader_loaded(changed_name, "fragment")


def test_shader_rename_preflights_every_dependent_pair(engine, rename_assets):
    assets = rename_assets
    vertex_path, _, old_name, source = assets["vertex"]
    linked_source = source.replace(
        '" }', '" Outputs { Smooth Float renameSignal } }'
    ) + ('VertexOutput vertex(inout VertexInput v) { VertexOutput result; '
         'result.renameSignal = 0.25; return result; }\n')
    vertex_path.write_text(linked_source, encoding="utf-8")
    imported = AssetManager.reimport_asset(str(vertex_path))
    assert imported, imported.error
    paired_name = assets["fragment"][2] + " With Signal"
    _, paired_guid = assets["create"](
        ".paired.frag",
        '#version 450\nShaderInfo { Name "' + paired_name +
        '" ShadingModel Unlit Inputs { Smooth Float renameSignal } }\n'
        'void surface(out SurfaceData s) { s = InitSurfaceData(); '
        's.albedo = vec3(fragmentInput.renameSignal); }\n',
    )
    material = AssetManager.load(assets["material_guid"], Material)
    clone = material.clone()
    document = clone._native.serialize_document()
    document["shaders"]["fragment"] = {"guid": paired_guid, "shader_id": paired_name}
    assert clone._native.deserialize_document(document)
    for value in (material, clone):
        assert engine.refresh_material_pipeline(value._native)
    before = [value._native.serialize_document() for value in (material, clone)]
    # The first pair remains valid, but the second requires the removed output.
    # Neither live binding may migrate when any dependent pair is rejected.
    changed_name = old_name + " V2"
    vertex_path.write_text(source.replace(old_name, changed_name), encoding="utf-8")
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


def test_guid_shader_material_loads_on_worker_without_runtime_registration(rename_assets):
    registry = AssetRegistry.instance()
    guid = rename_assets["material_guid"]
    registry.invalidate_asset(guid)
    ticket = registry.begin_load_material_by_guid(guid)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not registry.try_commit_asset_load(ticket):
        time.sleep(0.001)
    assert ticket.committed
    assert ticket.produced_on_worker
    material = AssetManager.load(guid, Material)
    assert material is not None
    assert material.guid == guid
    assert material._native.serialize_document()["shaders"]["vertex"]["guid"] == rename_assets["vertex"][1]
