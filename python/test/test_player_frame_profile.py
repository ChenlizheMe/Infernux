from __future__ import annotations

import pytest

from Infernux.engine.player_gui import _FrameProfileWindow


class _SceneManager:
    runtime_frame_count = 0
    fixed_time = 0.0
    time_scale = 1.0
    fixed_steps = 0
    profile_reads = 0
    playing = True
    paused = False

    def get_last_frame_profile(self):
        self.profile_reads += 1
        return {"fixed_steps": self.fixed_steps}

    def get_fixed_time_step(self):
        return 0.02

    def is_playing(self):
        return self.playing

    def is_paused(self):
        return self.paused


class _Scheduler:
    barriers = 0

    def phase_plan_snapshot(self):
        return {"physics_pre_step": (object(),), "update": (object(),), "fixed_update": ()}

    def profiler_snapshot(self):
        return {"native_frame_begins": 3, "native_barriers": self.barriers, "phase_dispatches": 5}


def test_frame_profile_counts_each_native_frame_once_and_resets_window():
    scene = _SceneManager()
    scheduler = _Scheduler()
    window = _FrameProfileWindow()

    window.sample(scene, scheduler)
    scene.runtime_frame_count = 1
    scene.fixed_time = 0.04
    scene.fixed_steps = 2
    window.sample(scene, scheduler)
    window.sample(scene, scheduler)
    scene.runtime_frame_count = 2
    scene.fixed_time = 0.06
    scene.fixed_steps = 1
    window.sample(scene, scheduler)
    scheduler.barriers = 9

    result = window.finish(scene, scheduler)
    assert scene.profile_reads == 2
    assert result["runtime_frame_count"] == 2
    assert result["runtime_frames_window"] == 2
    assert result["fixed_steps_window"] == 3
    assert result["fixed_seconds_window"] == pytest.approx(0.06)
    assert result["fixed_seconds_total"] == pytest.approx(0.06)
    assert result["playing"] is True
    assert result["paused"] is False
    assert result["time_scale"] == 1.0
    assert result["scheduler_phase_counts"]["physics_pre_step"] == 1
    assert result["scheduler_phase_counts"]["update"] == 1
    assert result["scheduler_phase_counts"]["fixed_update"] == 0
    assert result["scheduler_counters"]["native_barriers"] == 9
    assert result["scheduler_counters_window"]["native_barriers"] == 9

    scene.runtime_frame_count = 3
    scene.fixed_time = 0.06
    scene.fixed_steps = 0
    scene.paused = True
    window.sample(scene, scheduler)
    scheduler.barriers = 11
    next_result = window.finish(scene, scheduler)
    assert next_result["runtime_frames_window"] == 1
    assert next_result["fixed_steps_window"] == 0
    assert next_result["fixed_seconds_window"] == 0.0
    assert next_result["paused"] is True
    assert next_result["scheduler_counters_window"]["native_barriers"] == 2


def test_frame_profile_restarts_baseline_after_runtime_frame_counter_reset():
    scene = _SceneManager()
    scheduler = _Scheduler()
    window = _FrameProfileWindow()
    scene.runtime_frame_count = 8
    scene.fixed_time = 0.16
    window.sample(scene, scheduler)
    scene.runtime_frame_count = 0
    scene.fixed_time = 0.0
    window.sample(scene, scheduler)
    result = window.finish(scene, scheduler)
    assert result["runtime_frames_window"] == 0
    assert result["fixed_steps_window"] == 0
    assert result["fixed_seconds_window"] == 0.0


def test_player_frame_profile_is_not_allocated_when_disabled(monkeypatch):
    from Infernux.engine.player_gui import PlayerGUI

    monkeypatch.delenv("_INFERNUX_PLAYER_PROFILE_FRAMES", raising=False)
    player = PlayerGUI(object())
    assert player._profile_frames is False
    assert player._profile_window is None
