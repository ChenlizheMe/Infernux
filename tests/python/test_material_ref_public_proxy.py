"""Material references expose canonical public proxies for every input route."""
from __future__ import annotations

import pytest

import infernux as inx


@pytest.mark.parametrize("native_input", [False, True])
def test_material_reference_returns_canonical_proxy(native_input):
    material = inx.Material.create_unlit()
    value = material.native if native_input else material
    reference = inx.MaterialRef(value)
    assert reference.resolve() is material
    assert isinstance(reference.resolve(), inx.Material)
    reference.set_float("referenceValue", 0.625)
    assert material.get_float("referenceValue") == pytest.approx(0.625)


@pytest.mark.parametrize("value", [object(), 42])
def test_material_reference_rejects_non_material_objects_at_input(value):
    with pytest.raises(TypeError, match="Material"):
        inx.MaterialRef(value)


def test_native_input_reference_follows_proxy_retirement():
    material = inx.Material.create_unlit()
    reference = inx.MaterialRef(material.native)
    assert reference.resolve() is material
    material.dispose()
    assert reference.resolve() is None


def test_reference_cannot_resurrect_a_disposed_proxy():
    material = inx.Material.create_unlit()
    material.dispose()
    with pytest.raises(ReferenceError, match="disposed"):
        inx.MaterialRef(material)


@pytest.mark.parametrize("native_input", [False, True])
def test_material_conversion_preserves_proxy_and_clone_mutation(native_input):
    material = inx.Material.create_unlit()
    converted = inx.Material.from_native(material.native if native_input else material)
    assert converted is material
    before = material.get_color("baseColor")
    clone = inx.Instantiate(converted)
    clone.set_color("baseColor", 0.2, 0.5, 0.8)
    assert clone.native is not material.native
    assert clone.get_color("baseColor") == pytest.approx((0.2, 0.5, 0.8, 1.0))
    assert material.get_color("baseColor") == before


def test_renderer_material_conversion_remains_a_single_proxy(scene):
    owner = scene.create_game_object("Converted renderer material")
    renderer = inx.MeshRenderer._get_or_create_wrapper(owner.add_component("MeshRenderer"), owner)
    material = inx.Material.create_unlit()
    renderer.material = material
    converted = inx.Material.from_native(renderer.get_effective_material())
    assert converted is material
    assert isinstance(converted.native, inx.lib.InxMaterial)


def test_material_conversion_rejects_a_retired_proxy():
    material = inx.Material.create_unlit()
    material.dispose()
    with pytest.raises(ReferenceError, match="disposed"):
        inx.Material.from_native(material)
