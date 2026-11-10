"""Frame clocks agree with native simulation while retaining real elapsed time."""

import pytest

from infernux.components import InxComponent
from infernux.lib import RenderPipelineCallback, SceneManager
from infernux.timing import Time


@pytest.mark.parametrize(
    "scale,raw_delta,scaled_delta",
    [(1.0, 1.0, 1.0 / 3.0), (0.0, 0.8, 0.0),
     (0.25, 1.0, 0.25), (4.0, 0.2, 1.0 / 3.0)],
)
def test_frame_clocks_match_native_update(engine, scene, scale, raw_delta, scaled_delta):
    observed = []

    class ClockPipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    class ClockProbe(InxComponent):
        def update(self, delta_time):
            observed.append((delta_time, Time.delta_time, Time.unscaled_delta_time))

    scene.create_game_object("Frame clock").add_component(ClockProbe)
    manager = SceneManager.instance()
    manager.play()
    engine.set_render_pipeline(ClockPipeline())
    engine.set_pre_scene_update_callback(Time._tick)
    try:
        engine.tick(0.0)
        Time._reset()
        Time.time_scale = scale
        engine.tick(raw_delta)
        assert observed[-1] == pytest.approx((scaled_delta, scaled_delta, raw_delta))
        assert Time.time == pytest.approx(scaled_delta)
        assert Time.unscaled_time == pytest.approx(raw_delta)
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_render_pipeline(None)
        Time._reset()


def test_scene_load_reset_discards_loading_time_once():
    Time._reset()
    try:
        Time._reset_frame_delta()
        Time._tick(4.0)
        assert Time.delta_time == Time.unscaled_delta_time == 0.0
        assert Time.time == Time.unscaled_time == 0.0
        Time._tick(0.02)
        assert Time.delta_time == pytest.approx(0.02)
        assert Time.unscaled_delta_time == pytest.approx(0.02)
        assert Time.frame_count == 2
    finally:
        Time._reset()


def test_player_clock_follows_native_pause_and_time_scale(scene):
    from types import SimpleNamespace
    from infernux.engine.player_runtime import PlayerRuntimeSession

    session = PlayerRuntimeSession(scene_service=SimpleNamespace(
        active_scene_path=None, process_pending_load=lambda: None,
    ))
    manager = SceneManager.instance()
    manager.play()
    session._state = "playing"
    Time._reset()
    try:
        assert session.is_paused is False
        session.tick(0.05)
        assert Time.frame_count == 1
        assert Time.time == pytest.approx(0.05)
        assert Time.unscaled_time == pytest.approx(0.05)
        manager.pause()
        assert session.is_paused is True
        assert session.tick(1.0) == 0.0
        assert Time.frame_count == 1
        manager.play()
        Time.time_scale = 0.0
        session.tick(0.1)
        assert Time.time == pytest.approx(0.05)
        assert Time.unscaled_time == pytest.approx(0.15)
        assert Time.frame_count == 2
    finally:
        Time._reset()
