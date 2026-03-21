"""Tests for InfEngine.timing — Time class (metaclass-based static timing)."""

from __future__ import annotations

import time as stdlib_time

import pytest

from InfEngine.timing import Time


@pytest.fixture(autouse=True)
def _reset_time():
    """Reset Time state before each test."""
    Time._reset()
    yield
    Time._reset()


class TestTimeDefaults:
    def test_initial_values(self):
        assert Time.time == 0.0
        assert Time.delta_time == 0.0
        assert Time.unscaled_delta_time == 0.0
        assert Time.frame_count == 0
        assert Time.time_scale == 1.0

    def test_fixed_delta_time_default(self):
        assert Time.fixed_delta_time == pytest.approx(0.02)


class TestTimeTick:
    def test_tick_advances_time(self):
        Time._tick(0.016)
        assert Time.delta_time == pytest.approx(0.016)
        assert Time.unscaled_delta_time == pytest.approx(0.016)
        assert Time.time == pytest.approx(0.016)
        assert Time.frame_count == 1

    def test_tick_multiple_frames(self):
        Time._tick(0.016)
        Time._tick(0.016)
        assert Time.frame_count == 2
        assert Time.time == pytest.approx(0.032)

    def test_tick_respects_time_scale(self):
        Time.time_scale = 0.5
        Time._tick(0.016)
        assert Time.delta_time == pytest.approx(0.008)
        assert Time.unscaled_delta_time == pytest.approx(0.016)

    def test_tick_frozen_time(self):
        Time.time_scale = 0.0
        Time._tick(0.016)
        assert Time.delta_time == 0.0
        assert Time.time == 0.0
        assert Time.unscaled_delta_time == pytest.approx(0.016)

    def test_tick_clamps_negative_dt(self):
        Time._tick(-1.0)
        assert Time.unscaled_delta_time == 0.0

    def test_tick_clamps_to_maximum_delta_time(self):
        Time._tick(1.0)
        assert Time.unscaled_delta_time == pytest.approx(Time.maximum_delta_time)


class TestTimeFixedTick:
    def test_tick_fixed(self):
        Time._tick_fixed(0.02)
        assert Time.fixed_time == pytest.approx(0.02)
        assert Time.fixed_unscaled_time == pytest.approx(0.02)

    def test_tick_fixed_respects_time_scale(self):
        Time.time_scale = 2.0
        Time._tick_fixed(0.02)
        assert Time.fixed_time == pytest.approx(0.04)
        assert Time.fixed_unscaled_time == pytest.approx(0.02)


class TestTimeProperties:
    def test_time_scale_setter_clamps(self):
        Time.time_scale = -5.0
        assert Time.time_scale == 0.0

    def test_fixed_delta_time_setter_clamps(self):
        Time.fixed_delta_time = 0.0001
        assert Time.fixed_delta_time >= 0.001

    def test_maximum_delta_time_setter_clamps(self):
        Time.maximum_delta_time = 0.001
        assert Time.maximum_delta_time >= 0.01

    def test_realtime_since_startup_positive(self):
        assert Time.realtime_since_startup >= 0


class TestTimeReset:
    def test_reset_clears_state(self):
        Time._tick(0.5)
        Time._tick(0.5)
        Time._reset()
        assert Time.time == 0.0
        assert Time.delta_time == 0.0
        assert Time.frame_count == 0
        assert Time.time_scale == 1.0

    def test_reset_preserves_maximum_delta_time(self):
        Time.maximum_delta_time = 0.05
        Time._reset()
        assert Time.maximum_delta_time == 0.05
