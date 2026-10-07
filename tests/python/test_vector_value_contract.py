"""Mutable vector value semantics and the public axis-bearing angle API."""
import ast
from collections.abc import Hashable
from pathlib import Path

import pytest

from infernux import Vector2, Vector3, vec4f, vector3


@pytest.mark.parametrize('ctor,size', [(Vector2, 2), (Vector3, 3), (vec4f, 4)])
@pytest.mark.parametrize('offset', [0., .0000005])
@pytest.mark.parametrize('operation', ['hash', 'dict', 'set'])
def test_mutable_approximately_equal_vectors_are_unhashable(ctor, size, offset, operation):
    a, b = ctor(*([1.] * size)), ctor(*([1. + offset] * size))
    assert a == b and not (a != b)
    with pytest.raises(TypeError, match='unhashable'):
        if operation == 'hash':
            hash(a)
        elif operation == 'dict':
            {a: 'stored'}.get(b)
        else:
            {a, b}
    assert not isinstance(a, Hashable)
    assert type(a).__hash__ is None


@pytest.mark.parametrize('ctor,size', [(Vector2, 2), (Vector3, 3), (vec4f, 4)])
def test_vector_can_be_snapshotted_as_an_immutable_exact_key(ctor, size):
    value = ctor(*([1.] * size))
    key = tuple(value)
    mapping = {key: 'original'}
    value.x = 2
    assert mapping[key] == 'original'
    assert tuple(value) not in mapping
    assert value != ctor(*key)


@pytest.mark.parametrize('api', [Vector3, vector3])
@pytest.mark.parametrize('a,b,axis', [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
])
@pytest.mark.parametrize('scale', [1., -1., 7., -7.])
def test_signed_angle_uses_the_authored_axis(api, a, b, axis, scale):
    a, b = Vector3(*a), Vector3(*b)
    axis = Vector3(*(coordinate * scale for coordinate in axis))
    expected = 90. if scale > 0 else -90.
    assert api.signed_angle(a, b, axis) == pytest.approx(expected)
    assert api.signed_angle(b, a, axis) == pytest.approx(-expected)
    assert api.angle(a, b) == pytest.approx(90.)


@pytest.mark.parametrize('a,b,axis,expected', [
    ((0, 0, 0), (0, 1, 0), (1, 0, 0), 0.),
    ((0, 1, 0), (0, 0, 0), (1, 0, 0), 0.),
    ((2, 0, 0), (5, 0, 0), (0, 0, 1), 0.),
    ((2, 0, 0), (-5, 0, 0), (0, 0, 1), 180.),
    ((2, 0, 0), (0, 5, 0), (0, 0, 0), 90.),
])
def test_signed_angle_degenerate_orientation_retains_unsigned_angle(a, b, axis, expected):
    assert Vector3.signed_angle(Vector3(*a), Vector3(*b), Vector3(*axis)) == pytest.approx(expected)


@pytest.mark.parametrize('scale', [1e-30, 1., 1e30])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_angles_are_invariant_under_finite_vector_scaling(scale, sign):
    a, b, axis = Vector3(scale, 0, 0), Vector3(0, scale, 0), Vector3(0, 0, sign * scale)
    assert Vector3.angle(a, b) == pytest.approx(90.)
    assert Vector3.signed_angle(a, b, axis) == pytest.approx(sign * 90.)


def test_native_vector_stub_matches_hash_and_axis_contracts():
    source = (Path(__file__).parents[2] / "python") / 'infernux' / 'lib' / '_Infernux.pyi'
    classes = {node.name: node for node in ast.parse(source.read_text(encoding='utf-8')).body
               if isinstance(node, ast.ClassDef)}
    for name in ('Vector2', 'Vector3', 'vec4f'):
        row = next(node for node in classes[name].body
                   if isinstance(node, ast.AnnAssign) and node.target.id == '__hash__')
        assert isinstance(row.annotation, ast.Constant) and row.annotation.value is None
    method = next(node for node in classes['Vector3'].body
                  if isinstance(node, ast.FunctionDef) and node.name == 'signed_angle')
    assert len(method.args.args) == 3
