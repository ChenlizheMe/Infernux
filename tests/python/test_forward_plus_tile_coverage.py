"""Production Forward+ GLSL versus independent pixel-segment/sphere geometry."""
from pathlib import Path
import re
import struct

import numpy as np
import pytest

from infernux.lib import _Infernux


@pytest.fixture(scope="module")
def light_grid(engine):
    source = (Path(__file__).resolve().parents[2] /
              "cpp/infernux/function/renderer/lighting/ForwardPlusLightGrid.cpp").read_text(encoding="utf-8")
    match = re.search(r'ForwardPlusLightGrid::ShaderSource\(\).*?R"glsl\((.*?)\)glsl";', source, re.S)
    assert match is not None
    code = _Infernux._compile_compute_glsl_batch({"grid": match[1]}, "test-forward-plus-coverage")["grid"]
    host = engine._acquire_compute_host()
    kernel = host.create_kernel(code, buffer_binding_count=3, push_constant_bytes=112)
    state = [host, kernel]
    yield state
    kernel.wait()
    state.clear()
    kernel = host = None


def _projection(kind, width, height):
    near, far = .1, 30.
    projection = np.zeros((4, 4), np.float64)
    if kind.startswith("ortho"):
        projection[0, 0], projection[1, 1] = .2, -.25
        projection[2, 2], projection[2, 3], projection[3, 3] = 1/(far-near), -near/(far-near), 1
    else:
        projection[0, 0], projection[1, 1] = height/width, -1
        projection[2, 2], projection[2, 3], projection[3, 2] = far/(far-near), -near*far/(far-near), 1
    if "shift" in kind:
        projection[0] += .43 * projection[3]
        projection[1] -= .27 * projection[3]
    if kind == "oblique":
        projection[0, 1] = .35
        projection[1, 0] = -.21
        projection[2, 0] = .012
    if "infinite" in kind:
        projection[2, 2], projection[2, 3] = 1, -near
    if "reverse" in kind:
        projection[2] = projection[3] - projection[2]
    return projection


def _dispatch(grid, matrix, lights, width, height, domain=1, count=None):
    host, kernel = grid
    count = len(lights) if count is None else count
    nx, ny = (width+15)//16, (height+15)//16
    words = max(1, (count+31)//32)
    # Two directionals precede the local array; their garbage bounds must not
    # become local lights. Local index and bit-word boundaries are exercised.
    canonical = np.zeros(4 + 32*(len(lights)+2), np.uint32)
    canonical[:4] = (2, len(lights), len(lights)+2, 7)
    entries = canonical[4:].reshape(-1, 32)[2:]
    for entry, (center, radius, light_type, affects, area) in zip(entries, lights):
        floats = entry.view(np.float32)
        floats[:4] = (*center, radius)
        floats[19], floats[23] = area
        entry[24:28] = (light_type, 0xFFFFFFFF, 0, affects)
    arrays = [canonical, np.full((nx*ny+1)*4, 0xBAD, np.uint32),
              np.full(nx*ny*words, 0xFFFFFFFF, np.uint32)]
    push = struct.pack("<20f8I", *np.asarray(matrix, np.float32).flatten(order="F"),
                       width, height, 1, 1, nx, ny, count, 16, domain, words, 0, 0)
    buffers = []
    buffer = None
    try:
        for array in arrays:
            buffer = host.create_buffer(array.size, "uint32")
            buffer.set_bytes(array.tobytes())
            buffers.append(buffer)
        kernel.dispatch(buffers, push, nx, ny, 1)
        masks = np.frombuffer(buffers[2].get_bytes(arrays[2].nbytes), np.uint32).copy().reshape(ny, nx, words)
        headers = np.frombuffer(buffers[1].get_bytes(arrays[1].nbytes), np.uint32).reshape(-1, 4)
        assert headers[0].tolist() == [nx, ny, 16, domain]
        np.testing.assert_array_equal(headers[1:, 0], np.arange(nx*ny)*words)
        np.testing.assert_array_equal(headers[1:, 1:], np.tile([words, count, domain], (nx*ny, 1)))
        return masks
    finally:
        kernel.wait()
        buffers.clear()
        buffer = host = kernel = None


def test_known_off_axis_light_keeps_affected_tile(light_grid):
    matrix = _projection("perspective", 2048, 2048)
    light = ((3, 0, 5), 1, 1, 1, (0, 0))
    pixel = np.array([1906.5, 1024.5])
    ray = np.array([2*pixel[0]/2048-1, -(2*pixel[1]/2048-1), 1])
    closest = ray * (ray @ light[0]) / (ray @ ray)
    assert np.linalg.norm(closest-light[0]) < 1
    masks = _dispatch(light_grid, matrix, [light], 2048, 2048)
    assert masks[64, 119, 0] == 1, "Forward+ rejected a pixel segment intersecting the light sphere"
    assert masks[64, 0, 0] == 0, "A disjoint tile must still be culled"


@pytest.mark.parametrize("scale", [.01, 1, 1000])
@pytest.mark.parametrize("offset", [0, 10000])
def test_tangent_sphere_survives_matrix_scale_and_translation(light_grid, scale, offset):
    matrix = np.eye(4)
    matrix[0, 3] = -offset
    matrix *= scale
    lights = [((offset+.25, .125, .5), .25, 1, 1, (0, 0))]
    masks = _dispatch(light_grid, matrix, lights, 256, 256)
    assert masks[8, 7, 0] == 1  # Sphere tangent to the tile's x=0 right plane.
    assert masks[8, 0, 0] == 0  # Far outside remains excluded.


def _pixel_segments(matrix, width, height):
    yy, xx = np.mgrid[:height, :width]
    clip = np.stack((2*(xx+.5)/width-1, 2*(yy+.5)/height-1,
                     np.zeros_like(xx), np.ones_like(xx)), axis=-1).reshape(-1, 4)
    inverse = np.linalg.inv(matrix)
    ends = []
    for depth in (0, 1):
        clip[:, 2] = depth
        ends.append(clip @ inverse.T)
    # Infinite far maps to a homogeneous direction. Represent that end of
    # the ray well beyond the finite test spheres, including reversed depth.
    for index in (0, 1):
        infinite = np.abs(ends[index][:, 3]) < 1e-10
        if infinite.any():
            other = ends[1-index]
            origin = other[infinite, :3] / other[infinite, 3:4]
            ends[index][infinite, :3] = origin + 1e5*ends[index][infinite, :3]
            ends[index][infinite, 3] = 1
    return [end[:, :3]/end[:, 3:4] for end in ends], (yy//16, xx//16)


@pytest.mark.parametrize("kind", ["perspective", "perspective_shift", "ortho", "ortho_shift",
                                  "oblique", "reverse", "infinite", "reverse_infinite"])
@pytest.mark.parametrize("moved", [False, True])
def test_all_illuminated_pixel_tiles_survive(light_grid, kind, moved):
    width, height = 257, 193  # Last tiles are partial in both dimensions.
    projection = _projection(kind, width, height)
    camera_to_world = np.eye(4)
    if moved:
        angle = .73
        camera_to_world[:3, :3] = ((np.cos(angle), 0, np.sin(angle)), (0, 1, 0),
                                   (-np.sin(angle), 0, np.cos(angle)))
        camera_to_world[:3, 3] = (140, -80, 32)
    # Use exactly the float matrix sent to the GPU for independent ray math.
    matrix = (projection @ np.linalg.inv(camera_to_world)).astype(np.float32).astype(np.float64)
    rng = np.random.default_rng(414007)
    centers = rng.uniform((-12, -9, -.5), (12, 9, 26), (70, 3))
    radii = rng.uniform(.15, 3, 70)
    # Camera-inside, near crossing, wholly behind, wholly beyond finite far,
    # far side, negative range and zero range (all light types stay bounded).
    centers[:7] = ((0, 0, 0), (.1, .05, .15), (0, 0, -10), (0, 0, 40),
                   (1000, 0, 5), (0, 0, 5), (0, 0, 5))
    radii[:7] = (1, .2, 1, 1, 1, -1, 0)
    centers = centers @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    lights = [(center, radius, (1, 2, 3)[i % 3], (1, 2, 3)[i % 3],
               (1.4, .8) if i >= 7 and i % 3 == 2 else (0, 0))
              for i, (center, radius) in enumerate(zip(centers, radii))]
    (a, b), (tile_y, tile_x) = _pixel_segments(matrix, width, height)
    direction = b-a
    length_squared = np.einsum("ij,ij->i", direction, direction)
    for domain in (1, 2):
        masks = _dispatch(light_grid, matrix, lights, width, height, domain)
        affected = 0
        for index, (center, radius, light_type, affects, area) in enumerate(lights):
            present = masks[:, :, index//32] & np.uint32(1 << (index % 32)) != 0
            if not affects & domain:
                assert not present.any(), (kind, index, "wrong light domain")
                continue
            extent = max(0, radius) + (np.linalg.norm(area)*.5 if light_type == 3 else 0)
            t = np.clip(np.einsum("ij,ij->i", center-a, direction)/length_squared, 0, 1)
            distance = np.linalg.norm(a+direction*t[:, None]-center, axis=1)
            inside = (distance < extent-1e-4).reshape(height, width)
            affected += np.count_nonzero(inside)
            assert present[tile_y[inside], tile_x[inside]].all(), (kind, moved, domain, index)
            if index in (2, 4) or (index == 3 and "infinite" not in kind):
                assert not present.any(), (kind, index, "disjoint sphere included")
        assert affected > 100, "Fixture must exercise visible light coverage"
        # Padding bits beyond the local light count must remain zero.
        assert not (masks[:, :, -1] >> (len(lights) % 32)).any()
    empty = _dispatch(light_grid, matrix, lights, width, height, count=0)
    assert not empty.any()
