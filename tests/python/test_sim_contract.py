"""Yanshi ``sim.step@1`` / ``rigid.pose@1`` contract on the Automation surface.

The deterministic advance itself is exercised by
``tools/yanshi/rigid_diagnostics_probe.py`` against a headless manual world.
These cases pin the read-only and rejection behavior every driver relies on:
reads never advance, and an invalid step request never touches the world.
"""

from __future__ import annotations

import pytest

import infernux as inx
from infernux.sim import SimStepError


def test_clock_reads_are_stable_and_never_advance():
    first = inx.sim.clock()
    second = inx.sim.clock()
    assert first == second
    assert first["tick"] >= 0
    assert first["fixed_dt"] > 0.0


def test_pose_reads_world_position_and_stays_read_only(scene):
    obj = scene.create_game_object("YanshiPoseProbe")
    obj.get_transform().position = inx.Vector3(1.2, 0.0, 0.0)
    first = inx.sim.pose(obj)
    second = inx.sim.pose(obj)
    assert first == second
    assert first["position_m"] == pytest.approx([1.2, 0.0, 0.0], abs=1e-6)
    assert first["quat_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert first["frame"] == "world"
    assert first["tick"] == second["tick"]


def test_step_rejects_a_delta_that_is_not_the_fixed_step():
    fixed = inx.sim.fixed_delta_time()
    class _NeverTicked:
        def tick(self, _delta):  # pragma: no cover - must not run
            raise AssertionError("an invalid request must not advance the world")

    with pytest.raises(SimStepError) as error:
        inx.sim.step(_NeverTicked(), delta_time=fixed + 0.25)
    assert error.value.code == "sim.delta_mismatch"
    assert error.value.details["fixed_dt"] == pytest.approx(fixed)


def test_step_rejects_a_stale_tick_before_ticking():
    fixed = inx.sim.fixed_delta_time()
    class _NeverTicked:
        def tick(self, _delta):  # pragma: no cover - must not run
            raise AssertionError("a stale request must not advance the world")

    current = inx.sim.clock()["tick"]
    with pytest.raises(SimStepError) as error:
        inx.sim.step(_NeverTicked(), delta_time=fixed, expected_tick=current + 7)
    assert error.value.code == "sim.stale_tick"
    assert error.value.details["tick"] == current
