import pytest

from tests.acceptance.primitive_tangents import PRIMITIVES, verify_primitive_tangents


@pytest.mark.parametrize('name', PRIMITIVES)
def test_builtin_primitive_tangent_frame_matches_uv_derivatives(scene, name):
    verify_primitive_tangents(scene, name)
