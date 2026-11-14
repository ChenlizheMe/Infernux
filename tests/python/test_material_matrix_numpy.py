"""Camera, material and per-draw matrices share NumPy [row, column] input."""

import numpy as np
import pytest

from infernux.core.material import Material
from infernux.lib import InxMaterial, DrawParameterBlock


@pytest.fixture
def material():
    return Material(InxMaterial("Matrix contract", "Unlit"))


def stored_matrix(material, name="projection"):
    return np.asarray(material.native.get_property(name)).reshape((4, 4), order="F")


@pytest.mark.parametrize("layout,setter", [("transpose", "set_matrix"), ("negative_stride", "set_param")])
def test_material_copies_row_column_matrix_independently_of_array_strides(material, layout, setter):
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "transpose":
        values = values.T
    elif layout == "negative_stride":
        values = values[::-1, ::-1]
    expected = values.copy()
    material.set_matrix("projection", np.eye(4))
    getattr(material, setter)("projection", values)
    values[:] = 0
    np.testing.assert_array_equal(stored_matrix(material), expected)
    np.testing.assert_array_equal(material.get_matrix("projection"), expected)
    # The persistent representation is unchanged, not a second matrix format.
    document = material.native.serialize_document()
    assert document["properties"]["projection"]["type"] == 5
    np.testing.assert_array_equal(document["properties"]["projection"]["value"], expected.flatten(order="F"))


@pytest.mark.parametrize("bad", [np.eye(3), 2.0])
def test_invalid_matrix_does_not_mutate_previous_property(material, bad):
    material.set_matrix("projection", np.eye(4))
    for setter in (material.set_matrix, material.set_param):
        with pytest.raises((TypeError, ValueError), match="matrix"):
            setter("projection", bad)
        np.testing.assert_array_equal(stored_matrix(material), np.eye(4))


def test_declared_type_controls_numpy_vectors_and_numeric_reads(material):
    material.set_vector3("direction", 0.0, 0.0, 1.0)
    material.set_param("direction", np.asarray([1.0, 2.0, 3.0]))
    assert material.get_vector3("direction") == (1.0, 2.0, 3.0)
    material.set_int("count", 3)
    material.set_float("strength", 0.5)
    before = material.native.serialize_document()
    with pytest.raises((TypeError, RuntimeError)):
        material.set_param("count", 2.75)
    with pytest.raises(ValueError):
        material.get_int("strength")
    with pytest.raises(ValueError):
        material.set_int("strength", 1)
    with pytest.raises(KeyError):
        material.set_param("missing", 0.5)
    assert material.native.serialize_document() == before


def test_camera_matrix_passes_directly_to_material_renderer_and_draw(scene, material):
    camera = scene.create_game_object("Projection source").add_component("Camera")
    projection = camera.calculate_oblique_matrix((0.1, 0.2, 1.0, -2.0))
    camera.projection_matrix = projection
    material.set_matrix("projection", camera.projection_matrix)
    np.testing.assert_array_equal(stored_matrix(material), camera.projection_matrix)
    renderer = scene.create_game_object("Projection consumer").add_component("MeshRenderer")
    renderer.material = material
    changed = camera.projection_matrix.copy()
    changed[0, 2] = 0.37
    renderer.set_parameter("projection", changed, persistent=True)
    np.testing.assert_array_equal(np.asarray(renderer.get_parameter("projection")).reshape(4, 4, order="F"), changed)
    # Per-renderer overrides do not change the shared material or camera.
    np.testing.assert_array_equal(stored_matrix(material), camera.projection_matrix)
    document = renderer.serialize_document()
    np.testing.assert_array_equal(document["parameterOverrides"][0]["projection"]["value"], changed.flatten(order="F"))
    with pytest.raises(ValueError, match="matrix"):
        renderer.set_parameter("projection", np.eye(3))
    np.testing.assert_array_equal(np.asarray(renderer.get_parameter("projection")).reshape(4, 4, order="F"), changed)
    block = DrawParameterBlock()
    block.set_matrix("projection", camera.projection_matrix)
    assert block.size == 1
    with pytest.raises(ValueError, match="matrix"):
        block.set_matrix("projection", np.eye(2))
    assert block.size == 1
    with pytest.raises(ValueError, match="matrix"):
        block.set_matrix("projection", changed.flatten(order="F").tolist())
    assert block.size == 1


def test_unsigned_bool_and_matrix3_share_material_and_renderer_paths(scene, material):
    material.set_bool("enabled", True)
    material.set_uint("mask", 0xFEDCBA98)
    basis = np.arange(9, dtype=np.float64).reshape(3, 3).T[::-1]
    material.set_matrix("basis", basis)
    material.set_param("mask", 0xFFFFFFFF)
    material.set_param("enabled", False)
    np.testing.assert_array_equal(material.get_matrix("basis"), basis)
    assert material.get_uint("mask") == 0xFFFFFFFF
    assert material.get_bool("enabled") is False
    renderer = scene.create_game_object("Typed material overrides").add_component("MeshRenderer")
    renderer.material = material
    renderer.set_parameter("enabled", True, persistent=True)
    renderer.set_parameter("mask", 0x80000001, persistent=True)
    renderer.set_parameter("basis", basis[:, ::-1], persistent=True)
    assert renderer.get_parameter("enabled") is True
    assert renderer.get_parameter("mask") == 0x80000001
    np.testing.assert_array_equal(np.asarray(renderer.get_parameter("basis")).reshape(3, 3, order="F"), basis[:, ::-1])
    document = renderer.serialize_document()["parameterOverrides"][0]
    assert document["enabled"]["type"] == 10 and document["enabled"]["value"] is True
    assert document["mask"]["type"] == 11 and document["mask"]["value"] == 0x80000001
    np.testing.assert_array_equal(document["basis"]["value"], basis[:, ::-1].flatten(order="F"))
    block = DrawParameterBlock()
    block.set_bool("enabled", True)
    block.set_uint("mask", 0x80000001)
    block.set_matrix("basis", basis)
    assert block.size == 3
    before = material.to_dict()
    for invalid in (-1, 0x100000000):
        with pytest.raises((TypeError, ValueError, RuntimeError, OverflowError)):
            material.set_param("mask", invalid)
    with pytest.raises(TypeError):
        material.set_param("enabled", 1)
    assert material.to_dict() == before
