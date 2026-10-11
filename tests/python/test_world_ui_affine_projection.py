"""Billboard picking must invert the same affine camera basis used to draw it."""
import numpy as np
import pytest

from infernux import lib
from infernux.engine.runtime_screen_ui import collect_runtime_ui_input_surfaces, map_world_ui_ray
from infernux.ui import UIButton


@pytest.mark.parametrize('kind', ['rigid', 'scaled', 'sheared', 'mirrored', 'small'])
@pytest.mark.parametrize('constant_size', [False, True])
@pytest.mark.parametrize('orthographic', [False, True])
def test_affine_billboard_camera_roundtrip(scene, kind, constant_size, orthographic):
    camera = scene.create_game_object('Affine Camera').add_component('Camera')
    camera.set_clip_planes(.1, 100)
    camera.projection_mode = lib.CameraProjection.Orthographic if orthographic else lib.CameraProjection.Perspective
    # Keep one effective projection for both routes; explicit ray viewport
    # dimensions must not rebuild it at another authored aspect ratio.
    camera.projection_matrix = camera.projection_matrix
    to_world = np.eye(4)
    if kind == 'rigid':
        angle = .35
        to_world[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    elif kind == 'scaled':
        to_world[0, 0], to_world[1, 1] = 2., .75
    elif kind == 'sheared':
        to_world[0, 1], to_world[1, 0] = .5, .2
        to_world[2, 0], to_world[2, 1] = .15, -.1
    elif kind == 'mirrored':
        to_world[0, 0], to_world[1, 1] = -2., .75
    else:
        to_world[0, 0] = to_world[1, 1] = .0001
    camera.view_matrix = np.linalg.inv(to_world)
    obj = scene.create_game_object('Affine Billboard Button')
    anchor = np.array([0., 0., 10.])
    obj.transform.position = lib.Vector3(*anchor)
    element = UIButton()
    obj.add_py_component(element)
    element.width, element.height = 200., 100.
    element.world_billboard = True
    element.world_constant_screen_size = constant_size
    target, = collect_runtime_ui_input_surfaces(scene)
    # Project the actual drawing basis into the explicit ray viewport.
    view, projection = camera.view_matrix, camera.projection_matrix
    basis = camera.camera_to_world_matrix[:3, :2]
    clip_w = (projection @ view @ np.append(anchor, 1.))[3]
    scale = 200. * clip_w / (abs(projection[1, 1]) * 800.) if constant_size else 1.
    for local in ((.25, .1), (-.85, .4), (1.4, -.75)):
        point = anchor + basis @ np.asarray(local) * scale
        clip = projection @ view @ np.append(point, 1.)
        screen = (clip[:2] / clip[3] + 1.) * 400.
        origin, direction = camera.screen_point_to_ray(*screen, 800, 800)
        mapped = map_world_ui_ray(target, origin, direction, camera=camera, viewport_height=800)
        assert mapped is not None
        expected = (100. + 100.*local[0], 50. - 100.*local[1])
        assert mapped[:2] == pytest.approx(expected, abs=.005)
        assert mapped[2] > 0
        # Drag coordinates remain unbounded; only raycast clips to the button.
        assert (target.raycast(*mapped[:2]) is element) == (abs(local[0]) <= 1 and abs(local[1]) <= .5)
