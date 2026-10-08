import pytest

from tests.acceptance.line_bounds import CURVES, verify_line_bounds


@pytest.mark.parametrize('style', CURVES)
@pytest.mark.parametrize('space', ['world', 'local', 'affine'])
@pytest.mark.parametrize('shape', ['plain', 'rounded', 'loop'])
def test_authored_ribbon_vertices_fit_world_bounds(scene, style, space, shape):
    verify_line_bounds(scene, style, space, shape)
