"""Self-contained GPU proof for effective-projection shadow cascades."""
import json
import time

import pytest

from infernux.lib import ConsolePanel
from tests.gpu.projection_shadow_case import ProjectionShadowCase


@pytest.mark.parametrize('path', ('forward', 'forward_plus', 'deferred'))
@pytest.mark.parametrize('kind', ('perspective', 'orthographic', 'asymmetric', 'oblique', 'infinite', 'affine'))
def test_effective_projection_shadow_pixels(engine, scene, tmp_path, path, kind):
    case = ProjectionShadowCase(scene, path, kind)
    engine.resize_game_render_target(240, 180)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    phase, frames, ticket = 0, 0, None
    failures, observations = [], []
    deadline = time.monotonic() + 60
    try:
        def tick():
            nonlocal phase, frames, ticket
            try:
                engine.request_full_speed_frame()
                frames += 1
                assert time.monotonic() < deadline
                if ticket is None and frames >= 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    assert not ticket.error, ticket.error
                    observations.append(case.observe(ticket.result_numpy()))
                    phase += 1
                    if phase == len(case.phases):
                        engine.exit()
                    else:
                        case.change(phase)
                        ticket, frames = None, 0
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(tick)
        engine.run()
        if failures:
            raise failures[0]
        assert len(observations) == len(case.phases)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        case.close()
        (tmp_path / 'projection-shadows.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')


@pytest.mark.parametrize('path', ('forward', 'forward_plus', 'deferred'))
def test_shadow_receiver_crossing_camera_plane_is_independent_of_tessellation(engine, scene, path):
    import numpy as np

    from infernux.lib import LightShadows, Vector3
    from tests.gpu.additive_shadow_case import AdditiveShadowCase

    case = AdditiveShadowCase(scene, path)
    receiver = next(obj for obj in case.objects if obj.name == 'Receiver')
    receiver.transform.local_scale = Vector3(1, 1, 1)
    renderer = receiver.get_component('MeshRenderer')
    caster = next(obj for obj in case.objects if obj.name == 'Caster')
    caster.transform.position = Vector3(0, 1.25, 0)
    caster.transform.local_scale = Vector3(2, 2.5, 1)
    camera = case.camera.game_object
    camera.transform.position = Vector3(0, 2, -5)
    camera.transform.euler_angles = Vector3(15, 0, 0)
    case.camera.field_of_view = 60.
    case.light.game_object.transform.euler_angles = Vector3(35, 30, 0)
    case.light.shadows = LightShadows.Soft

    def ground(subdivisions):
        coordinates = np.linspace(-60, 60, subdivisions + 1, dtype=np.float32)
        positions = np.array([(x, 0, z) for z in coordinates for x in coordinates], dtype=np.float32)
        indices = []
        for z in range(subdivisions):
            for x in range(subdivisions):
                a = z * (subdivisions + 1) + x
                b, c, d = a + 1, a + subdivisions + 2, a + subdivisions + 1
                indices.extend((a, c, b, a, d, c))
        renderer.set_inline_mesh_data(
            positions, np.tile(np.array([[0, 1, 0]], dtype=np.float32), (len(positions), 1)),
            np.zeros((len(positions), 2), dtype=np.float32), np.array(indices, dtype=np.uint32),
            'Camera-plane receiver',
        )

    ground(1)
    engine.resize_game_render_target(240, 180)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    images, failures = [], []
    frames, ticket = 0, None
    deadline = time.monotonic() + 60

    def tick():
        nonlocal frames, ticket
        try:
            engine.request_full_speed_frame()
            frames += 1
            assert time.monotonic() < deadline
            if ticket is None and frames >= 8:
                ticket = engine.request_render_target_readback(True)
            elif ticket is not None and ticket.done:
                assert not ticket.error, ticket.error
                images.append(ticket.result_numpy().copy())
                if len(images) == 4:
                    engine.exit()
                else:
                    if len(images) == 3:
                        case.light.shadows = LightShadows.NoShadows
                    else:
                        ground(30 if len(images) == 1 else 1)
                    frames, ticket = 0, None
        except BaseException as error:
            failures.append(error)
            engine.exit()

    try:
        engine.set_post_draw_callback(tick)
        engine.run()
        if failures:
            raise failures[0]
        assert len(images) == 4
        shadow = np.max(np.abs(images[1][..., :3] - images[3][..., :3]), axis=-1) > .03
        assert shadow.sum() > 100, 'Fixture must contain a visible shadow'
        # The same coplanar receiver and bounds must produce the same shadow,
        # including triangles with vertices behind the camera.
        for image in (images[0], images[2]):
            difference = np.abs(image.astype(np.float32) - images[1].astype(np.float32))
            assert float(difference[shadow].mean()) < .005, float(difference[shadow].mean())
        assert ConsolePanel().get_error_count() == 0
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        case.close()
