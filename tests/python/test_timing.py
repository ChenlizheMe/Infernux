"""Frame clocks agree with native simulation while retaining real elapsed time."""

import pytest

from infernux.components import InxComponent
from infernux.lib import RenderPipelineCallback, SceneManager
from infernux.timing import Time


@pytest.fixture(autouse=True)
def clock_pipeline(engine):
    class ClockPipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    engine.set_render_pipeline(ClockPipeline())
    yield
    engine.set_render_pipeline(None)


@pytest.mark.parametrize(
    "scale,raw_delta,scaled_delta",
    [(1.0, 1.0, 1.0 / 3.0), (0.0, 0.8, 0.0),
     (0.25, 1.0, 0.25), (4.0, 0.2, 1.0 / 3.0)],
)
def test_frame_clocks_match_native_update(engine, scene, scale, raw_delta, scaled_delta):
    observed = []

    class ClockProbe(InxComponent):
        def update(self, delta_time):
            observed.append((delta_time, Time.delta_time, Time.unscaled_delta_time))

    scene.create_game_object("Frame clock").add_component(ClockProbe)
    manager = SceneManager.instance()
    manager.play()
    try:
        # Publish the newly attached lifecycle plan without advancing time.
        engine.tick(0.0)
        Time.time_scale = scale
        engine.tick(raw_delta)
        assert observed[-1] == pytest.approx((scaled_delta, scaled_delta, raw_delta))
        assert Time.time == pytest.approx(scaled_delta)
        assert Time.unscaled_time == pytest.approx(raw_delta)
        assert Time.frame_count == 2
        assert manager.get_frame_delta_time() == Time.delta_time
    finally:
        Time.time_scale = 1.0


def test_scene_load_reset_discards_loading_time_once(engine, scene):
    manager = SceneManager.instance()
    manager.play()
    engine.tick(0.05)
    try:
        engine.set_pre_scene_update_callback(lambda _: manager.prepare_active_scene_replacement())
        engine.tick(4.0)
        assert Time.delta_time == Time.unscaled_delta_time == 0.0
        assert Time.time == Time.unscaled_time == pytest.approx(0.05)
        engine.set_pre_scene_update_callback(None)
        engine.tick(0.02)
        assert Time.delta_time == pytest.approx(0.02)
        assert Time.unscaled_delta_time == pytest.approx(0.02)
        assert Time.frame_count == 3
        assert Time.time == Time.unscaled_time == pytest.approx(0.07)
    finally:
        engine.set_pre_scene_update_callback(None)


def test_player_preparation_never_advances_native_time(engine, scene):
    from types import SimpleNamespace
    from infernux.engine.player_runtime import PlayerRuntimeSession

    session = PlayerRuntimeSession(scene_service=SimpleNamespace(
        active_scene_path=None, process_pending_load=lambda: None,
    ))
    manager = SceneManager.instance()
    manager.play()
    session._state = "playing"
    try:
        assert session.is_paused is False
        session.prepare_frame()
        assert Time.frame_count == 0
        engine.tick(0.05)
        assert Time.frame_count == 1
        assert Time.time == pytest.approx(0.05)
        assert Time.unscaled_time == pytest.approx(0.05)
        manager.pause()
        assert session.is_paused is True
        session.prepare_frame()
        engine.tick(1.0)
        assert Time.frame_count == 1
        manager.play()
        Time.time_scale = 0.0
        session.prepare_frame()
        engine.tick(0.1)
        assert Time.time == pytest.approx(0.05)
        assert Time.unscaled_time == pytest.approx(0.15)
        assert Time.frame_count == 2
    finally:
        Time.time_scale = 1.0


def test_paused_step_and_stop_publish_one_native_clock(engine, scene):
    manager = SceneManager.instance()
    manager.play()
    engine.tick(0.04)
    manager.pause()
    engine.tick(1.0)
    assert Time.time == pytest.approx(0.04) and Time.frame_count == 1
    manager.step(0.016)
    assert Time.time == pytest.approx(0.056) and Time.frame_count == 2
    assert Time.delta_time == pytest.approx(0.016)
    engine.tick(0.0)
    assert Time.time == pytest.approx(0.056) and Time.frame_count == 2
    with pytest.raises(AttributeError):
        Time._frame_time.time = 17.0
    manager.stop()
    assert Time.time == Time.unscaled_time == 0.0 and Time.frame_count == 0


def test_pending_step_retains_its_admitted_frame_duration(engine, scene):
    from infernux.lib import NativeRuntimeFrameBarrier

    manager = SceneManager.instance()
    old_step = manager.get_fixed_time_step()
    fixed, updates = [], []
    ready = False
    manager.clear_runtime_lifecycle_callbacks()
    manager.set_runtime_lifecycle_callbacks(
        lambda: None, fixed.append, updates.append, lambda _: None, lambda _: None, lambda: None)
    manager.set_runtime_lifecycle_work_available(True)
    manager.set_runtime_lifecycle_plan(1, 1, 1, 0, 0)
    manager.set_runtime_result_consumer_callback(
        lambda boundary: ready or boundary != NativeRuntimeFrameBarrier.FIXED_SCRIPT)
    manager.play()
    manager.pause()
    try:
        manager.step(0.016)
        assert manager.fixed_step_pending and updates == []
        assert Time.frame_count == 1
        manager.set_fixed_time_step(0.03)
        ready = True
        manager.step(0.08)
        for _ in range(120):
            if manager.runtime_frame_count == 1:
                break
            engine.tick(0.0)
        assert manager.runtime_frame_count == 1
        assert fixed == pytest.approx([old_step])
        assert updates == pytest.approx([0.016])
        assert Time.delta_time == Time.time == pytest.approx(0.016)
        assert Time.frame_count == 1
        assert Time.fixed_time == pytest.approx(old_step)
    finally:
        manager.stop()
        manager.set_fixed_time_step(old_step)
        manager.clear_runtime_lifecycle_callbacks()
