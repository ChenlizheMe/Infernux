"""Public resource queries preserve Python type, sharing and lifetime."""
from __future__ import annotations

import gc
import weakref

import pytest

import infernux as inx
from infernux.lib import AudioClip as NativeAudioClip


def _resource(kind):
    if kind == "audio":
        return inx.AudioClip.from_native(NativeAudioClip())
    if kind == "material":
        return inx.Material.create_lit("Proxy test")
    if kind == "mesh":
        return inx.Mesh.create("Proxy test")
    return inx.Texture.solid_color(2, 2)


def _component(scene, name, cls):
    owner = scene.create_game_object(name)
    return cls._get_or_create_wrapper(owner.add_component(cls.__name__), owner)


@pytest.mark.parametrize("kind", ["audio", "material", "mesh", "texture"])
def test_resource_conversion_and_constructor_reuse_live_proxy(kind):
    resource = _resource(kind)
    cls, native = type(resource), resource.native
    assert cls.from_native(native) is resource
    assert cls(native) is resource


@pytest.mark.parametrize("kind", ["audio", "material", "mesh", "texture"])
def test_identity_cache_does_not_keep_resources_alive(kind):
    resource = _resource(kind)
    cls, native = type(resource), resource.native
    ref = weakref.ref(resource)
    del resource
    gc.collect()
    assert ref() is None
    replacement = cls.from_native(native)
    assert replacement.native is native
    assert cls.from_native(native) is replacement


def test_rewrapping_material_does_not_reset_pending_save_state():
    material = inx.Material.create_lit()
    material._last_save_time = 23.0
    material._save_pending = True
    assert inx.Material.from_native(material.native) is material
    assert material._last_save_time == 23.0
    assert material._save_pending is True
    material._save_pending = False


def test_rewrapping_texture_preserves_registered_guid():
    texture = inx.Texture.solid_color(2, 2)
    texture._guid = "registered-texture"
    assert inx.Texture.from_native(texture.native) is texture
    assert texture._guid == "registered-texture"


def test_audio_queries_share_proxy_across_tracks_and_sources(scene):
    first = _component(scene, "First", inx.AudioSource)
    second = _component(scene, "Second", inx.AudioSource)
    clip = _resource("audio")
    first.track_count = 2
    first.set_track_clip(0, clip)
    first.set_track_clip(1, clip.native)
    second.set_track_clip(0, clip)
    assert first.get_track_clip(0) is clip
    assert first.get_track_clip(1) is clip
    assert second.get_track_clip(0) is clip
    replacement = _resource("audio")
    first.set_track_clip(0, replacement)
    assert first.get_track_clip(0) is replacement
    assert first.get_track_clip(1) is clip
    first.set_track_clip(0, None)
    assert first.get_track_clip(0) is None


def test_mesh_renderer_all_material_queries_share_proxy(scene):
    renderer = _component(scene, "First", inx.MeshRenderer)
    other = _component(scene, "Second", inx.MeshRenderer)
    material = inx.Material.create_lit()
    renderer.set_material_slot_count(2)
    renderer.materials = [material, material.native]
    other.material = material
    assert renderer.material is material
    assert renderer.sharedMaterial is material
    assert other.material is material
    assert renderer.get_material(0) is material
    assert renderer.get_effective_material(0) is material
    assert all(value is material for value in renderer.materials)
    assert all(value is material for value in renderer.sharedMaterials)
    result = [None]
    assert renderer.get_materials(result) is result
    assert len(result) == 2 and all(value is material for value in result)
    clone = material.clone()
    assert clone is not material
    renderer.set_material(0, clone)
    assert renderer.get_material(0) is clone
    assert renderer.get_material(1) is material
    renderer.material = None
    assert renderer.material is None
    assert isinstance(renderer.get_effective_material(0), inx.Material)


def test_sprite_renderer_material_is_public_proxy(scene):
    renderer = _component(scene, "Sprite", inx.SpriteRenderer)
    material = inx.Material.create_unlit()
    renderer.material = material
    assert renderer.material is material
    assert renderer.shared_material is material


def test_distinct_transient_materials_compare_and_hash_by_live_identity():
    first = inx.Material.create_lit()
    second = first.clone()
    assert first.guid == second.guid == ""
    assert first != second
    assert len({first, second}) == 2
    assert inx.Material.from_native(first.native) == first


@pytest.mark.parametrize("assignment", ["none", "empty-guid", "empty-list-slot"])
def test_clear_transient_material_releases_the_assignment(scene, assignment):
    renderer = _component(scene, "Clear material", inx.MeshRenderer)
    material = inx.Material.create_unlit()
    renderer.material = material
    assert renderer.material is material
    if assignment == "none":
        renderer.material = None
    elif assignment == "empty-guid":
        renderer.material_guid = ""
    else:
        renderer.set_materials([""])
    assert renderer.get_material(0) is None
    assert renderer._cpp_component.get_material(0) is None
    assert isinstance(renderer.get_effective_material(0), inx.Material)


def test_mesh_queries_share_original_proxy_across_renderers(scene):
    first = _component(scene, "First", inx.MeshRenderer)
    second = _component(scene, "Second", inx.MeshRenderer)
    mesh = inx.Mesh.create()
    first.mesh = mesh
    second.mesh = mesh
    assert first.mesh is first.shared_mesh is first.get_mesh_asset() is mesh
    assert second.mesh is mesh
    copy = mesh.copy("Independent mesh")
    assert copy is not mesh
    first.mesh = copy
    assert first.get_mesh_asset() is copy
    assert second.get_mesh_asset() is mesh
    first.mesh = None
    assert first.get_mesh_asset() is None
    copy.destroy()
    mesh.destroy()
