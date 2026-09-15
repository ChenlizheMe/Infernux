"""Camera, material and per-draw matrices share NumPy [row, column] input."""
import numpy as np
import pytest

from Infernux.core.material import Material
from Infernux.lib import InxMaterial, DrawParameterBlock


@pytest.fixture
def material():
    return Material(InxMaterial('Matrix contract', 'Unlit'))


def stored_matrix(material, name='projection'):
    return np.asarray(material.native.get_property(name)).reshape((4, 4), order='F')


@pytest.mark.parametrize('layout', ['c', 'fortran', 'transpose', 'negative_stride'])
@pytest.mark.parametrize('setter', ['set_matrix', 'set_param'])
def test_material_copies_row_column_matrix_independently_of_array_strides(material, layout, setter):
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    if layout == 'fortran':
        values = np.asfortranarray(values)
    elif layout == 'transpose':
        values = values.T
    elif layout == 'negative_stride':
        values = values[::-1, ::-1]
    expected = values.copy()
    getattr(material, setter)('projection', values)
    values[:] = 0
    np.testing.assert_array_equal(stored_matrix(material), expected)
    # The persistent representation is unchanged, not a second matrix format.
    document = material.native.serialize_document()
    assert document['properties']['projection']['type'] == 5
    np.testing.assert_array_equal(document['properties']['projection']['value'], expected.flatten(order='F'))


@pytest.mark.parametrize('setter', ['set_matrix', 'set_param'])
def test_existing_flat_column_major_contract_remains_explicit(material, setter):
    values = np.arange(16, dtype=float).reshape(4, 4)
    getattr(material, setter)('projection', values.flatten(order='F').tolist())
    np.testing.assert_array_equal(stored_matrix(material), values)
    getattr(material, setter)('projection', values.tolist())
    np.testing.assert_array_equal(stored_matrix(material), values)


@pytest.mark.parametrize('bad', [np.eye(3), np.zeros((2, 8)), np.zeros(15), 2.0])
def test_invalid_matrix_does_not_mutate_previous_property(material, bad):
    material.set_matrix('projection', np.eye(4))
    for setter in (material.set_matrix, material.set_param):
        with pytest.raises((TypeError, ValueError), match='matrix'):
            setter('projection', bad)
        np.testing.assert_array_equal(stored_matrix(material), np.eye(4))


def test_camera_matrix_passes_directly_to_material_renderer_and_draw(scene, material):
    camera = scene.create_game_object('Projection source').add_component('Camera')
    projection = camera.calculate_oblique_matrix((.1, .2, 1., -2.))
    camera.projection_matrix = projection
    material.set_matrix('projection', camera.projection_matrix)
    np.testing.assert_array_equal(stored_matrix(material), camera.projection_matrix)
    renderer = scene.create_game_object('Projection consumer').add_component('MeshRenderer')
    renderer.material = material
    changed = camera.projection_matrix.copy()
    changed[0, 2] = .37
    renderer.set_parameter('projection', changed, persistent=True)
    np.testing.assert_array_equal(np.asarray(renderer.get_parameter('projection')).reshape(4, 4, order='F'), changed)
    # Per-renderer overrides do not change the shared material or camera.
    np.testing.assert_array_equal(stored_matrix(material), camera.projection_matrix)
    document = renderer.serialize_document()
    np.testing.assert_array_equal(document['parameterOverrides'][0]['projection']['value'], changed.flatten(order='F'))
    with pytest.raises(ValueError, match='matrix'):
        renderer.set_parameter('projection', np.eye(3))
    np.testing.assert_array_equal(np.asarray(renderer.get_parameter('projection')).reshape(4, 4, order='F'), changed)
    block = DrawParameterBlock()
    block.set_matrix('projection', camera.projection_matrix)
    assert block.size == 1
    with pytest.raises(ValueError, match='matrix'):
        block.set_matrix('projection', np.eye(3))
    assert block.size == 1
    block.set_matrix('projection', changed.flatten(order='F').tolist())
    assert block.size == 1
