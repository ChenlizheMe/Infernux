"""Runtime camera poses share rendering, lighting, gizmo and ray coordinates."""
import numpy as np
import pytest

from Infernux import lib
from Infernux.components.builtin.camera import Camera


@pytest.fixture
def camera(scene):
    return scene.create_game_object('View Camera').add_component('Camera')


def test_history_reset_does_not_edit_camera_pose_or_serialized_settings(camera):
    camera.transform.position = lib.Vector3(1, 2, -3)
    camera.view_matrix = np.diag([1., -1., 1., 1.])
    camera.invert_culling = True
    document = camera.serialize_document()
    pose = camera.transform.position
    projection, view = camera.projection_matrix.copy(), camera.view_matrix.copy()
    assert camera.reset_history() is None
    assert camera.reset_history() is None
    assert camera.serialize_document() == document
    assert camera.transform.position == pose
    np.testing.assert_array_equal(camera.projection_matrix, projection)
    np.testing.assert_array_equal(camera.view_matrix, view)
    assert camera.invert_culling and camera.has_custom_view_matrix


def test_view_override_is_owned_runtime_state_and_reset_follows_current_transform(camera):
    camera.transform.position = lib.Vector3(2, 3, -4)
    document = camera.serialize_document()
    ordinary = camera.view_matrix.copy()
    np.testing.assert_allclose(camera.camera_to_world_matrix @ ordinary, np.eye(4), atol=1e-6)
    reflected = ordinary @ np.diag([1., -1., 1., 1.])
    camera.view_matrix = reflected
    camera.invert_culling = True
    assert camera.has_custom_view_matrix
    assert camera.invert_culling
    assert camera.serialize_document() == document
    np.testing.assert_allclose(camera.view_matrix, reflected)
    np.testing.assert_allclose(camera.camera_to_world_matrix @ reflected, np.eye(4), atol=1e-6)
    reflected[:] = 0
    returned = camera.camera_to_world_matrix
    returned[:] = 0
    assert camera.view_matrix[1, 1] == -1
    assert camera.camera_to_world_matrix[1, 1] == -1
    camera.transform.position = lib.Vector3(8, 9, -10)
    assert camera.camera_to_world_matrix[0, 3] == 2
    camera.reset_view_matrix()
    assert not camera.has_custom_view_matrix
    np.testing.assert_allclose(camera.camera_to_world_matrix[:3, 3], [8, 9, -10])
    assert camera.invert_culling  # Explicit raster policy is not a pose reset.
    camera.invert_culling = False


@pytest.mark.parametrize('bad', [np.eye(3), np.zeros((4, 4)), np.full((4, 4), np.nan),
                               np.full((4, 4), np.inf), np.diag([1., 1., 1., 2.]),
                               np.eye(4) + np.outer([0., 0., 0., .1], [1., 0., 0., 0.])])
def test_invalid_view_is_rejected_at_set_without_losing_current_view(camera, bad):
    camera.view_matrix = np.diag([1., -1., 1., 1.])
    before = camera.view_matrix
    with pytest.raises(ValueError, match='Camera'):
        camera.view_matrix = bad
    np.testing.assert_array_equal(camera.view_matrix, before)


@pytest.mark.parametrize('orthographic', [False, True])
@pytest.mark.parametrize('mirrored', [False, True])
def test_custom_view_ray_and_gizmo_match_rendered_projection(camera, orthographic, mirrored):
    camera.transform.position = lib.Vector3(900, 800, 700)  # Deliberately not the rendered eye.
    camera.projection_mode = lib.CameraProjection.Orthographic if orthographic else lib.CameraProjection.Perspective
    camera.set_clip_planes(.3, 100)
    angle = .6
    camera_to_world = np.array([[np.cos(angle), 0, np.sin(angle), 4],
                               [0, -1 if mirrored else 1, 0, 3],
                               [-np.sin(angle), 0, np.cos(angle), -6], [0, 0, 0, 1.]])
    camera.view_matrix = np.linalg.inv(camera_to_world)
    camera.projection_matrix = camera.calculate_oblique_matrix((.1, .15, 1, -2))
    vp = camera.projection_matrix @ camera.view_matrix
    for x, y in [(100, 100), (960, 540), (1820, 980)]:
        origin, direction = camera.screen_point_to_ray(x, y, 1920, 1080)
        origin, direction = np.array(tuple(origin)), np.array(tuple(direction))
        assert np.linalg.norm(direction) == pytest.approx(1)
        near_clip = vp @ np.append(origin, 1)
        assert near_clip[2] / near_clip[3] == pytest.approx(0, abs=2e-6)
        for distance in (0, 5, 30):
            point = origin + direction * distance
            np.testing.assert_allclose(tuple(camera.world_to_screen_point(*point)), [x, y], atol=.003)
    positions, indices = Camera._frustum_wire_geometry(camera.projection_matrix, camera.view_matrix, 1000)
    clips = np.column_stack((positions[:4], np.ones(4))) @ vp.T
    np.testing.assert_allclose(clips[:, :3] / clips[:, 3:],
                               [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], atol=2e-6)
    assert indices.shape == (12, 2)
