from __future__ import annotations

import math

import pytest


def _send_scene_camera_input(engine, *, right_down: bool, delta_x: float = 0.0, delta_y: float = 0.0) -> None:
    engine.process_scene_view_input(
        0.0,
        right_down,
        False,
        delta_x,
        delta_y,
        0.0,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )


def test_first_editor_camera_look_delta_does_not_snap_focus(engine):
    camera = engine.editor_camera
    _send_scene_camera_input(engine, right_down=False)
    camera.reset()

    before = camera.focus_point
    try:
        _send_scene_camera_input(engine, right_down=True)
        _send_scene_camera_input(engine, right_down=True, delta_x=1.0)
        after = camera.focus_point
    finally:
        _send_scene_camera_input(engine, right_down=False)
        camera.reset()

    focus_shift = math.dist(
        (before.x, before.y, before.z),
        (after.x, after.y, after.z),
    )
    assert focus_shift < 0.05


def _camera_snapshot(camera):
    return tuple(camera.position), tuple(camera.focus_point), camera.focus_distance, camera.rotation


@pytest.mark.parametrize('position,point,distance', [
    ((1., 2., 3.), (1., 2., 3.), 5.),
    ((0., 0., 0.), (0., 0., 0.), 1.),
    ((1., 2., 3.), (1., 2., 8.), 5.),
    ((0., 4., 0.), (0., 0., 0.), 2.),
    ((1.e-25, 0., 0.), (0., 0., 0.), 1.),
    ((1.e20, 0., 0.), (0., 0., 0.), 5.),
    ((0., 0., 0.), (0., 0., 0.), 1.e-25),
    ((0., 0., 0.), (0., 0., 0.), 1.e25),
])
def test_focus_on_is_finite_aligned_and_repeatable(engine, position, point, distance):
    camera = engine.editor_camera
    try:
        camera.restore_state(*position, 0., 0., 0., 5., 37., -12.)
        camera.focus_on(*point, distance)
        actual = tuple(camera.position)
        assert all(math.isfinite(v) for v in (*actual, *camera.rotation))
        assert math.dist(actual, point) == pytest.approx(distance, rel=1.e-5, abs=0.)
        offset = tuple(a-b for a, b in zip(position, point))
        length = math.hypot(*offset)
        direction = tuple(v/length for v in offset) if length else (0., 0., 1.)
        assert actual == pytest.approx(tuple(p+d*distance for p, d in zip(point, direction)))
        before = _camera_snapshot(camera)
        camera.focus_on(*point, distance)
        assert _camera_snapshot(camera) == before
        # A zero mouse delta must retain the synchronized finite orientation.
        _send_scene_camera_input(engine, right_down=True)
        _send_scene_camera_input(engine, right_down=True)
        assert all(math.isfinite(value) for row in camera.view_matrix for value in row)
    finally:
        _send_scene_camera_input(engine, right_down=False)
        camera.reset()


@pytest.mark.parametrize('point,distance', [
    ((float('nan'), 0., 0.), 5.), ((0., float('inf'), 0.), 5.),
    ((0., 0., 0.), float('nan')), ((0., 0., 0.), float('inf')),
    ((0., 0., 0.), 0.), ((0., 0., 0.), -1.),
    ((3.e38, 0., 0.), 1.e-30),
])
def test_focus_on_rejects_invalid_input_without_mutating_camera(engine, point, distance):
    camera = engine.editor_camera
    camera.reset()
    before = _camera_snapshot(camera)
    try:
        with pytest.raises(ValueError):
            camera.focus_on(*point, distance)
        assert _camera_snapshot(camera) == before
    finally:
        camera.reset()
