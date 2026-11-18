"""Yanshi ``sim.step@1`` / ``rigid.pose@1`` contract on the Automation surface.

The deterministic advance itself is exercised by
``tools/yanshi/rigid_diagnostics_probe.py`` against a headless Play Mode world
(11 samples, 0.1 mm / 1 us tolerances). These cases pin the read-only and
rejection behavior every driver relies on: reads never advance, invalid
requests never touch the world, and manual stepping refuses an unprepared
world.
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


def test_step_requires_an_active_paused_world():
    class _NeverTicked:
        def tick(self, _delta):  # pragma: no cover - must not run
            raise AssertionError("an unprepared world must not advance")

    with pytest.raises(SimStepError) as error:
        inx.sim.step(_NeverTicked(), expected_tick=0)
    assert error.value.code in {"sim.not_playing", "sim.not_paused"}

    with pytest.raises(SimStepError) as pause_error:
        inx.sim.pause()
    assert pause_error.value.code == "sim.not_playing"

    with pytest.raises(SimStepError) as settle_error:
        inx.sim.settle(_NeverTicked())
    assert settle_error.value.code == "sim.not_playing"


def test_editor_catalog_exposes_sim_capabilities(tmp_path):
    """Yanshi discovers sim.* through the Automation Host catalog."""
    from infernux.host.editor_operations import build_editor_operations
    from infernux.host.operations import OperationKind

    operations = {op.schema.id: op for op in build_editor_operations(str(tmp_path))}
    assert {"infernux.sim.clock", "infernux.sim.pose", "infernux.sim.step"} <= set(operations)
    step = operations["infernux.sim.step"]
    assert step.schema.kind is OperationKind.COMMAND
    assert "sim.write" in step.schema.capabilities
    pose = operations["infernux.sim.pose"]
    assert pose.schema.kind is OperationKind.QUERY
    assert "sim.read" in pose.schema.capabilities
    assert "object_id" in pose.schema.input_schema.get("required", ())
