"""Native vector interpolation obeys scalar timing and geometric endpoints."""
import math

import numpy as np
import pytest

from infernux import Mathf, Vector2, Vector3, vec4f


@pytest.mark.parametrize('ctor,dimension,axis', [(ctor, size, axis)
    for ctor, size in ((Vector2, 2), (Vector3, 3), (vec4f, 4)) for axis in range(size)])
@pytest.mark.parametrize('dt', [.016, .1, .5, 1.0])
@pytest.mark.parametrize('start,target,speed,max_speed', [(0., 1., 0., 100.), (3., -2., -.25, .7), (0., 1., 20., 100.)])
def test_smooth_damp_axis_matches_scalar_over_time(ctor, dimension, axis, dt, start, target, speed, max_speed):
    def embed(value):
        coordinates = [0.] * dimension
        coordinates[axis] = value
        return ctor(*coordinates)
    position, velocity = embed(start), embed(speed)
    scalar, scalar_speed = start, speed
    for _ in range(16):
        position, velocity = ctor.smooth_damp(position, embed(target), velocity, .5, max_speed, dt)
        scalar, scalar_speed = Mathf.smooth_damp(scalar, target, scalar_speed, .5, max_speed, dt)
        assert list(position) == pytest.approx(list(embed(scalar)), abs=3e-6, rel=2e-5)
        assert list(velocity) == pytest.approx(list(embed(scalar_speed)), abs=3e-6, rel=2e-5)


@pytest.mark.parametrize('t', [0., .25, .5, 1., -1., 2.])
@pytest.mark.parametrize('lengths', [(1., 1.), (2., 5.), (1e-20, 4e-20), (1e20, 4e20)])
@pytest.mark.parametrize('clamped', [True, False])
def test_slerp_interpolates_angle_and_length(t, lengths, clamped):
    a, b = Vector3(lengths[0], 0, 0), Vector3(0, lengths[1], 0)
    effective = min(1., max(0., t)) if clamped else t
    angle = math.pi / 2 * effective
    length = float(a.x) + (float(b.y) - float(a.x)) * effective
    expected = [math.cos(angle) * length, math.sin(angle) * length, 0.]
    function = Vector3.slerp if clamped else Vector3.slerp_unclamped
    actual = function(a, b, t)
    np.testing.assert_allclose(list(actual), expected, rtol=3e-6, atol=max(abs(length) * 1e-6, 1e-30))
    if effective == 0:
        assert list(actual) == list(a)
    elif effective == 1:
        assert list(actual) == list(b)


@pytest.mark.parametrize('a,b', [
    ((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (2, 3, 4)), ((2, 3, 4), (0, 0, 0)),
    ((2, 0, 0), (5, 0, 0)), ((0, -2, 0), (0, -5, 0)),
])
@pytest.mark.parametrize('t', [0., .5, 1., 1.5])
def test_slerp_parallel_and_zero_vectors_have_linear_magnitude(a, b, t):
    expected = np.asarray(a) + (np.asarray(b) - a) * t
    np.testing.assert_allclose(list(Vector3.slerp_unclamped(Vector3(*a), Vector3(*b), t)), expected, atol=1e-6)


@pytest.mark.parametrize('a', [(2, 0, 0), (0, 3, 0), (0, 0, -4), (1, 2, 3)])
@pytest.mark.parametrize('near', [False, True])
def test_slerp_opposite_vectors_follow_a_deterministic_arc(a, near):
    a = np.asarray(a, dtype=float)
    b = -a * 2
    if near:
        b[(int(np.argmax(np.abs(a))) + 1) % 3] += 1e-4
    start, end = Vector3(*a), Vector3(*b)
    first = np.asarray(list(Vector3.slerp(start, end, .5)))
    for _ in range(10):
        np.testing.assert_array_equal(list(Vector3.slerp(start, end, .5)), first)
    expected_length = (np.linalg.norm(list(start)) + np.linalg.norm(list(end))) * .5
    assert np.linalg.norm(first) == pytest.approx(expected_length, rel=2e-6)
    assert abs(np.dot(first, a) / (np.linalg.norm(a) * expected_length)) < 1e-4
    assert list(Vector3.slerp(start, end, 0)) == list(start)
    assert list(Vector3.slerp(start, end, 1)) == list(end)


@pytest.mark.parametrize('angle', [0., math.pi / 4, math.pi / 2])
def test_rotate_towards_retains_independent_magnitude_limit(angle):
    result = Vector3.rotate_towards(Vector3(2, 0, 0), Vector3(0, 5, 0), angle, .5)
    np.testing.assert_allclose(list(result), [2.5 * math.cos(angle), 2.5 * math.sin(angle), 0], atol=1e-6)


@pytest.mark.parametrize('function', [Vector3.slerp, Vector3.slerp_unclamped])
@pytest.mark.parametrize('t', [float('nan'), float('inf'), float('-inf')])
def test_slerp_rejects_nonfinite_interpolation_parameter(function, t):
    with pytest.raises(ValueError, match='finite'):
        function(Vector3(2, 0, 0), Vector3(0, 3, 0), t)


@pytest.mark.parametrize('a,b', [((0, 0, 0), (3e38, 0, 0)), ((1e38, 0, 0), (0, 3e38, 0))])
def test_slerp_unrepresentable_extrapolation_is_explicit(a, b):
    with pytest.raises(OverflowError, match='non-finite'):
        Vector3.slerp_unclamped(Vector3(*a), Vector3(*b), 4)
