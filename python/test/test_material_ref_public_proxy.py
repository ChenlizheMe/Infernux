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
