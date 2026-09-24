"""Imported PBR maps use the same GUIDs and sampling contract as saved materials."""
import json
import time

import pytest
from PIL import Image

from Infernux.core.asset_types import read_mesh_import_settings, read_texture_import_settings
from Infernux.core.assets import AssetManager
from Infernux.lib import AssetDependencyGraph, AssetRegistry
from test_model_material_defaults import imported_model


def prepare_maps(imported_model, *, linear=True):
    _, document, source, database, _ = imported_model
    identities = []
    for name, color in (("color", (180, 90, 40)), ("normal", (128, 128, 255)),
                        ("orm", (64, 128, 230)), ("emission", (40, 160, 200))):
        image = source.with_name(name + ".png")
        Image.new("RGB", (4, 4), color).save(image)
        result = AssetManager.import_asset(str(image), database=database)
        assert result, result.error
        if linear and name in ("normal", "orm"):
            settings = read_texture_import_settings(str(image))
            settings.srgb = False
            assert AssetManager.apply_import_settings("texture", str(image), settings)
        identities.append(result.guid)
    document["images"] = [{"uri": name + ".png"} for name in ("color", "normal", "orm", "emission")]
    document["textures"] = [{"source": index} for index in range(4)]
    document["materials"][0].update({
        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": 0},
                                "metallicFactor": .8, "roughnessFactor": .6,
                                "metallicRoughnessTexture": {"index": 2}},
        "normalTexture": {"index": 1, "scale": .35},
        "occlusionTexture": {"index": 2, "strength": .4},
        "emissiveTexture": {"index": 3}, "emissiveFactor": [.2, .3, .4],
    })
    source.write_text(json.dumps(document), encoding="utf-8")
    return identities


@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_pbr_maps_survive_binary_reload_extraction_and_none(imported_model, asynchronous):
    renderer, _, source, database, _ = imported_model
    guids = prepare_maps(imported_model)
    registry = AssetRegistry.instance()
    mesh = registry.load_mesh(str(source))
    if asynchronous:
        owner = AssetManager.begin_model_reimport(str(source), read_mesh_import_settings(str(source)))
        deadline = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < deadline
            time.sleep(.002)
    else:
        result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    expected = dict(zip(("texSampler", "normalMap", "metallicMap", "smoothnessMap", "aoMap", "emissionMap"),
                        (guids[0], guids[1], guids[2], guids[2], guids[2], guids[3])))
    for reload in (False, True):
        if reload:
            assert registry.reload_asset(mesh.guid)
        data = mesh.get_material_slot_data()[0]
        assert data["packed_metallic_roughness"]
        assert data["normal_scale"] == pytest.approx(.35)
        assert data["occlusion_strength"] == pytest.approx(.4)
        for material in (renderer.get_material(0), mesh.create_material_copy(0)):
            props = material.serialize_document()["properties"]
            assert {key: props[key]["guid"] for key in expected} == expected
            assert material.get_float("normalScale") == pytest.approx(.35)
            assert material.get_float("smoothness") == pytest.approx(.4)
            assert material.get_float("smoothnessFromRoughness") == 1
            assert material.get_float("occlusionStrength") == pytest.approx(.4)
            metal_channels = material.get_vector4("metallicChannels")
            rough_channels = material.get_vector4("smoothnessChannels")
            assert (metal_channels.x, metal_channels.y, metal_channels.z, metal_channels.w) == (0, 0, 1, 0)
            assert (rough_channels.x, rough_channels.y, rough_channels.z, rough_channels.w) == (0, 1, 0, 0)
        assert set(guids) <= set(AssetDependencyGraph.instance().get_dependencies(mesh.guid))
    # Extracted .mat uses the same references, not paths into an Assimp importer.
    target = source.with_suffix(".mat")
    target.write_text(json.dumps(mesh.create_material_copy(0).serialize_document()), encoding="utf-8")
    extracted = AssetManager.import_asset(str(target), database=database)
    assert extracted, extracted.error
    assert set(guids) <= set(AssetDependencyGraph.instance().get_dependencies(extracted.guid))
    settings = read_mesh_import_settings(str(source))
    settings.material_import_mode = "none"
    assert AssetManager.apply_import_settings("mesh", str(source), settings)
    assert not set(guids) & set(AssetDependencyGraph.instance().get_dependencies(mesh.guid))


def test_data_maps_reject_srgb_without_publishing_partial_model(imported_model):
    _, _, source, database, _ = imported_model
    prepare_maps(imported_model, linear=False)
    mesh = AssetRegistry.instance().load_mesh(str(source))
    generation = mesh.generation
    before = mesh.get_material_slot_data()
    result = AssetManager.reimport_asset(str(source), database=database)
    assert not result and "disable sRGB" in result.error
    assert mesh.generation == generation
    assert mesh.get_material_slot_data() == before


def test_model_sampler_survives_binary_reload_as_per_binding_override(imported_model):
    _, document, source, database, _ = imported_model
    prepare_maps(imported_model)
    document["samplers"] = [{"magFilter": 9728, "minFilter": 9987,
                             "wrapS": 33071, "wrapT": 33648}]
    document["textures"][0]["sampler"] = 0
    source.write_text(json.dumps(document), encoding="utf-8")

    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    mesh = AssetRegistry.instance().load_mesh(str(source))
    assert AssetRegistry.instance().reload_asset(mesh.guid)
    sampler = mesh.get_material_slot_data()[0]["base_color_sampler"]
    assert sampler == {
        "min_filter": 2,   # linear
        "mag_filter": 1,   # nearest
        "mip_filter": 2,   # linear
        "address_u": 2,    # clamp
        "address_v": 3,    # mirror
        "address_w": 1,    # repeat
    }
    document_sampler = mesh.create_material_copy(0).serialize_document()["textureSamplers"]["texSampler"]
    assert document_sampler == {
        "minFilter": 2, "magFilter": 1, "mipFilter": 2,
        "addressU": 2, "addressV": 3, "addressW": 1,
    }
    report = json.loads(database.get_meta_by_path(str(source)).get_string("model_material_diagnostics"))
    assert not [item for item in report if item["code"] == "unsupported_sampler_address"]


def test_each_pbr_binding_consumes_secondary_uv_after_binary_reload(imported_model):
    renderer, document, source, database, _ = imported_model
    prepare_maps(imported_model)
    # Reuse one zero-filled authored stream for both channels so the test
    # isolates binding selection from UV geometry.
    document["buffers"].append({
        "byteLength": 24,
        "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    })
    document["bufferViews"].append({"buffer": 1, "byteOffset": 0, "byteLength": 24})
    document["accessors"].append({
        "bufferView": len(document["bufferViews"]) - 1,
        "componentType": 5126,
        "count": 3,
        "type": "VEC2",
    })
    uv_accessor = len(document["accessors"]) - 1
    for mesh in document["meshes"]:
        attributes = mesh["primitives"][0]["attributes"]
        attributes["TEXCOORD_0"] = uv_accessor
        attributes["TEXCOORD_1"] = uv_accessor
    pbr = document["materials"][0]["pbrMetallicRoughness"]
    pbr["baseColorTexture"]["texCoord"] = 1
    pbr["metallicRoughnessTexture"]["texCoord"] = 1
    document["materials"][0]["normalTexture"]["texCoord"] = 1
    document["materials"][0]["occlusionTexture"]["texCoord"] = 1
    document["materials"][0]["emissiveTexture"]["texCoord"] = 1
    source.write_text(json.dumps(document), encoding="utf-8")

    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    mesh = AssetRegistry.instance().load_mesh(str(source))
    assert AssetRegistry.instance().reload_asset(mesh.guid)
    slot = mesh.get_material_slot_data()[0]
    for key in ("base_color_uv_set", "normal_uv_set", "metallic_uv_set",
                "roughness_uv_set", "occlusion_uv_set", "emission_uv_set"):
        assert slot[key] == 1
    material = mesh.create_material_copy(0)
    for key in ("baseColorUvSet", "normalUvSet", "metallicUvSet",
                "smoothnessUvSet", "occlusionUvSet", "emissionUvSet"):
        assert material.get_int(key) == 1
    assert renderer.get_material(0).get_int("baseColorUvSet") == 1
    report = json.loads(database.get_meta_by_path(str(source)).get_string("model_material_diagnostics"))
    assert not [item for item in report if item["code"] == "unsupported_uv_set"]


def test_unmapped_material_texture_semantic_is_preserved_as_import_report(imported_model):
    _, document, source, database, _ = imported_model
    prepare_maps(imported_model)
    document["extensionsUsed"] = ["KHR_materials_clearcoat"]
    document["materials"][0]["extensions"] = {
        "KHR_materials_clearcoat": {"clearcoatFactor": 1.0, "clearcoatTexture": {"index": 0}}
    }
    source.write_text(json.dumps(document), encoding="utf-8")

    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    report = json.loads(database.get_meta_by_path(str(source)).get_string("model_material_diagnostics"))
    unsupported = [item for item in report if item["code"] == "unsupported_texture_semantic"]
    assert unsupported == [{
        "code": "unsupported_texture_semantic",
        "material": "Red",
        "property": "texture/Clearcoat",
        "detail": "0",
    }]
