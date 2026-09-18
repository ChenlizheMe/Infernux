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
