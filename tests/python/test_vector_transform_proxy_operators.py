"""Native vector operators honor Python dispatch for Transform value proxies."""
import operator

import pytest

from infernux import Vector2, Vector3, vec4f


@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('left_proxy,right_proxy', [(False, False), (True, False), (False, True), (True, True)])
def test_native_and_transform_vectors_support_both_operand_orders(scene, operation, left_proxy, right_proxy):
    a = scene.create_game_object('Left').transform
    b = scene.create_game_object('Right').transform
    a.position, b.position = Vector3(2, 4, 6), Vector3(1, 2, 3)
    left = a.position if left_proxy else Vector3(2, 4, 6)
    right = b.position if right_proxy else Vector3(1, 2, 3)
    assert tuple(operation(left, right)) == pytest.approx(tuple(operation(x, y) for x, y in zip((2,4,6), (1,2,3))))
    assert tuple(a.position) == (2,4,6)
    assert tuple(b.position) == (1,2,3)


@pytest.mark.parametrize('operation', [operator.iadd, operator.isub, operator.imul, operator.itruediv])
def test_transform_inplace_arithmetic_commits_proxy_operand(scene, operation):
    a = scene.create_game_object('Owner').transform
    b = scene.create_game_object('Operand').transform
    a.position, b.position = Vector3(2,4,6), Vector3(1,2,3)
    result = operation(a.position, b.position)
    expected = tuple(operation(x, y) for x, y in zip((2,4,6), (1,2,3)))
    assert tuple(result) == pytest.approx(expected)
    assert tuple(a.position) == pytest.approx(expected)
    assert tuple(b.position) == (1,2,3)


def test_native_equality_defers_to_transform_proxy(scene):
    transform = scene.create_game_object('Vector equality').transform
    transform.position = Vector3(2,4,6)
    assert Vector3(2,4,6) == transform.position
    assert transform.position == Vector3(2,4,6)
    assert not (Vector3(2,4,6) != transform.position)
    assert Vector3(1,4,6) != transform.position


@pytest.mark.parametrize('constructor,dimension', [(Vector2,2), (Vector3,3), (vec4f,4)])
@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul, operator.truediv])
def test_vector_unsupported_operand_uses_python_operator_protocol(constructor, dimension, operation):
    vector = constructor(*([2] * dimension))
    with pytest.raises(TypeError):
        operation(vector, object())
    sentinel = object()
    class Reflected:
        def __radd__(self, other): return sentinel
        def __rsub__(self, other): return sentinel
        def __rmul__(self, other): return sentinel
        def __rtruediv__(self, other): return sentinel
    assert operation(vector, Reflected()) is sentinel


@pytest.mark.parametrize('constructor,dimension', [(Vector2,2), (Vector3,3), (vec4f,4)])
@pytest.mark.parametrize('scalar', [2, 2.5, 2**40])
def test_vector_scalar_operator_contract(constructor, dimension, scalar):
    vector = constructor(*([4] * dimension))
    assert tuple(vector + scalar) == pytest.approx([4 + scalar] * dimension)
    assert tuple(scalar - vector) == pytest.approx([scalar - 4] * dimension)
    assert tuple(vector * scalar) == pytest.approx([4 * scalar] * dimension)
    assert tuple(scalar / vector) == pytest.approx([scalar / 4] * dimension)
