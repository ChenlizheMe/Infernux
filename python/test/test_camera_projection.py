"""The rendered projection, pixel queries and frustum gizmos share one definition."""
import numpy as np
import pytest

from Infernux import lib
from Infernux.components.builtin.camera import Camera


@pytest.fixture
def camera(scene):
    return scene.create_game_object('Projection Camera').add_component('Camera')


def test_override_is_copied_and_preserves_authoring_until_reset(camera):
    before = camera.serialize_document()
    projection = camera.projection_matrix
    projection[0, 2] = .35
    camera.projection_matrix = projection
    assert camera.has_custom_projection_matrix
    np.testing.assert_array_equal(camera.projection_matrix, projection)
    assert camera.serialize_document() == before
    projection[0, 2] = 99
    assert camera.projection_matrix[0, 2] == pytest.approx(.35)
    copy = camera.projection_matrix
    copy[:] = 0
    assert camera.projection_matrix[0, 2] == pytest.approx(.35)
    camera.field_of_view = 75
    # The renderer, not the public wrapper, owns the viewport aspect.
    camera._require_cpp_component().aspect_ratio = .8
    assert camera.projection_matrix[0, 2] == pytest.approx(.35)
    camera.reset_projection_matrix()
    assert not camera.has_custom_projection_matrix
    actual = camera.projection_matrix
    assert actual[0, 2] == 0
    assert actual[1, 1] == pytest.approx(-1 / np.tan(np.radians(75) / 2))
    assert actual[0, 0] == pytest.approx(-actual[1, 1] / .8)


@pytest.mark.parametrize('matrix', [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan),
                                  np.full((4, 4), np.inf), np.ones(16)])
def test_invalid_override_does_not_replace_current_projection(camera, matrix):
    before = camera.projection_matrix
    with pytest.raises(ValueError, match='Camera'):
        camera.projection_matrix = matrix
    np.testing.assert_array_equal(camera.projection_matrix, before)
    assert not camera.has_custom_projection_matrix


@pytest.mark.parametrize('orthographic', [False, True])
@pytest.mark.parametrize('oblique', [False, True])
def test_ray_reprojects_to_same_pixel_and_starts_on_rendered_near_plane(camera, orthographic, oblique):
    camera.projection_mode = lib.CameraProjection.Orthographic if orthographic else lib.CameraProjection.Perspective
    camera.set_clip_planes(.3, 1e8)
    projection = camera.projection_matrix
    if orthographic:
        projection[0, 3] = .2
    else:
        projection[0, 2] = .2
    camera.projection_matrix = projection
    if oblique:
        camera.projection_matrix = camera.calculate_oblique_matrix((.12, .18, 1, -2))
    projection = camera.projection_matrix.astype(np.float64)
    for x, y in [(100, 100), (960, 540), (1820, 980)]:
        origin, direction = camera.screen_point_to_ray(x, y, 1920, 1080)
        origin, direction = np.array(tuple(origin)), np.array(tuple(direction))
        assert np.isfinite(origin).all() and np.isfinite(direction).all()
        assert np.linalg.norm(direction) == pytest.approx(1)
        clip = projection @ np.append(origin, 1)
        assert clip[2] / clip[3] == pytest.approx(0, abs=1e-6)
        for distance in (0, 3, 100):
            point = origin + direction * distance
            clip = projection @ np.append(point, 1)
            pixel = (clip[:2] / clip[3] + 1) * np.array([960, 540])
            np.testing.assert_allclose(pixel, [x, y], atol=1e-3)
            np.testing.assert_allclose(tuple(camera.world_to_screen_point(*point)), [x, y], atol=1e-3)


def test_default_ray_cache_follows_fov_and_explicit_viewport_without_changing_camera(camera):
    original_aspect = camera.aspect_ratio
    _, first = camera.screen_point_to_ray(0, 0, 800, 600)
    _, wide = camera.screen_point_to_ray(0, 0, 1200, 600)
    assert abs(wide.x) > abs(first.x)
    camera.field_of_view = 90
    _, changed = camera.screen_point_to_ray(0, 0, 800, 600)
    assert abs(changed.y) > abs(first.y)
    assert camera.aspect_ratio == original_aspect
    # The projection cache must also work after switching back to orthographic.
    camera.projection_mode = lib.CameraProjection.Orthographic
    origin, direction = camera.screen_point_to_ray(0, 0, 800, 600)
    np.testing.assert_allclose(tuple(direction), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(tuple(origin), [-5 * 4 / 3, 5, camera.near_clip], atol=1e-5)


@pytest.mark.skipif(not hasattr(lib, "PhysicalGateFit"), reason="native PhysicalGateFit binding not built")
@pytest.mark.parametrize("gate, expected_sensor_axis", [
    ("Horizontal", "width"),
    ("Vertical", "height"),
])
def test_physical_camera_uses_vertical_fov_and_selected_gate(camera, gate, expected_sensor_axis):
    """Physical projection must derive glm's vertical FOV from the selected gate.

    Unity's Horizontal gate ignores sensor height; Vertical ignores sensor
    width.  This catches the former bug where horizontal FOV was passed to
    glm::perspective as if it were vertical FOV.
    """
    camera.projection_mode = lib.CameraProjection.Physical
    camera.focal_length = 50.0
    camera.sensor_size = lib.Vector2(36.0, 24.0)
    camera.gate_fit = getattr(lib.PhysicalGateFit, gate)
    camera._require_cpp_component().aspect_ratio = 16.0 / 9.0
    matrix = np.asarray(camera.projection_matrix, dtype=np.float64)
    if expected_sensor_axis == "width":
        gate_height = 36.0 / (16.0 / 9.0)
    else:
        gate_height = 24.0
    fov_y = 2.0 * np.arctan(gate_height / (2.0 * 50.0))
    np.testing.assert_allclose(abs(matrix[1, 1]), 1.0 / np.tan(fov_y / 2.0), rtol=2e-5)


@pytest.mark.skipif(not hasattr(lib, "PhysicalGateFit"), reason="native PhysicalGateFit binding not built")
def test_physical_camera_lens_shift_changes_frustum_center(camera):
    camera.projection_mode = lib.CameraProjection.Physical
    camera.gate_fit = lib.PhysicalGateFit.Horizontal
    camera.lens_shift = lib.Vector2(0.125, -0.1)
    shifted = np.asarray(camera.projection_matrix)
    camera.lens_shift = lib.Vector2(0.0, 0.0)
    centered = np.asarray(camera.projection_matrix)
    # Python exposes matrices in row/column order; the native GLM write to
    # projection[2][0/1] therefore appears at row 0/1, column 2.
    assert shifted[0, 2] != pytest.approx(centered[0, 2])
    assert shifted[1, 2] != pytest.approx(centered[1, 2])


def test_screen_world_round_trip_uses_top_left_pixels(camera):
    camera.set_clip_planes(.5, 50)
    for pixel in [(100, 120), (960, 540), (1800, 900)]:
        world = camera.screen_to_world_point(*pixel, .5)
        recovered = camera.world_to_screen_point(*tuple(world))
        np.testing.assert_allclose(tuple(recovered), pixel, atol=2e-4)
        ray_origin, direction = camera.screen_point_to_ray(*pixel)
        delta = np.array(tuple(world)) - np.array(tuple(ray_origin))
        np.testing.assert_allclose(delta / np.linalg.norm(delta), tuple(direction), atol=1e-6)
    assert camera.screen_to_world_point(960, 100, .5).y > 0


@pytest.mark.parametrize('plane', [(0, 0, 0, 1), (np.nan, 0, 1, -1), (0, 0, -1, -1)])
def test_invalid_oblique_plane_rejected_without_mutation(camera, plane):
    original = camera.projection_matrix
    with pytest.raises(ValueError, match='Camera oblique'):
        camera.calculate_oblique_matrix(plane)
    np.testing.assert_array_equal(camera.projection_matrix, original)


@pytest.mark.parametrize('orthographic', [False, True])
def test_oblique_clips_positive_halfspace_and_gizmos_use_the_same_near_corners(camera, orthographic):
    camera.projection_mode = lib.CameraProjection.Orthographic if orthographic else lib.CameraProjection.Perspective
    camera.set_clip_planes(.2, 40)
    plane = np.array([.1, .2, 1, -2])
    initial = camera.projection_matrix
    oblique = camera.calculate_oblique_matrix(plane)
    np.testing.assert_array_equal(camera.projection_matrix, initial)  # calculation is pure
    camera.projection_matrix = oblique
    for point in ([0, 0, 1, 1], [0, 0, 4, 1], [1, 1, 2, 1]):
        assert np.sign((oblique @ point)[2]) == np.sign(plane @ point)
    points, indices = Camera._frustum_wire_geometry(oblique, camera.view_matrix, 1000)
    assert indices.shape == (12, 2) and points.shape == (8, 3)
    np.testing.assert_allclose(np.column_stack((points[:4], np.ones(4))) @ plane, 0, atol=1e-6)
    if orthographic:
        # Capping display distance must not taper an orthographic volume.
        points, _ = Camera._frustum_wire_geometry(initial, camera.view_matrix, 5)
        np.testing.assert_allclose(points[:4, :2], points[4:, :2], atol=1e-6)


def test_gizmo_infinite_far_is_finite_and_batched(camera, monkeypatch):
    from Infernux.gizmos import Gizmos
    camera.set_clip_planes(.1, 1e8)
    batches = []
    monkeypatch.setattr(Gizmos, 'draw_lines', lambda positions, indices: batches.append((positions, indices)))
    camera.on_draw_gizmos_selected()
    assert len(batches) == 1
    assert np.isfinite(batches[0][0]).all()
    assert batches[0][1].shape == (12, 2)
