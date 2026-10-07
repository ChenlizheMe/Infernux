"""Deferred body publication must apply each body's final state exactly once."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_case(project, mode, repeats, motion):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(project), mode, str(repeats), motion],
        env=os.environ.copy(), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((project / "broadphase-evidence.json").read_text())["passed"]


@pytest.mark.parametrize("motion", ["static", "dynamic", "kinematic"])
@pytest.mark.parametrize("mode", ["add", "remove"])
@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_same_frame_collider_toggles_publish_one_body(tmp_path, mode, repeats, motion):
    run_case(tmp_path, mode, repeats, motion)


@pytest.mark.parametrize("motion", ["static", "dynamic"])
@pytest.mark.parametrize("cancel", [False, True])
def test_pending_add_uses_current_motion_type(tmp_path, motion, cancel):
    run_case(tmp_path, "motion", int(cancel), motion)


def test_reenabled_collider_publishes_disabled_authored_configuration(tmp_path):
    run_case(tmp_path, "configuration", 1, "dynamic")


@pytest.mark.parametrize("force", ["linear", "angular", "position"])
@pytest.mark.parametrize("cancel", [0, 1, 2])
def test_force_after_enable_waits_for_actual_residency(tmp_path, force, cancel):
    run_case(tmp_path, "pending_force_" + force, cancel, "dynamic")


def test_destroy_before_force_publication_does_not_leave_active_bodies(tmp_path):
    run_case(tmp_path, "destroy_pending_force", 32, "dynamic")


def exercise_disabled_configuration(owner, collider, body):
    import numpy as np
    from infernux.lib import ForceMode, Physics, SceneManager, Vector3
    from infernux.physics import Physics as PublicPhysics

    collider.enabled = False
    Physics.sync_transforms()
    body.mass = 4
    body.drag = .25
    body.use_gravity = False
    body.constraints = 2 | 64  # FreezePositionX | FreezeRotationZ
    body.velocity = Vector3(1, 2, 3)
    body.add_force(Vector3(0, 4, 0), ForceMode.Impulse)
    collider.enabled = True
    Physics.sync_transforms()
    state = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(state["inverse_mass"][0], [0, .25, .25], atol=1e-6)
    np.testing.assert_allclose(state["inverse_inertia"][0, 2], 0, atol=1e-6)
    np.testing.assert_allclose(state["linear_velocity"][0], [0, 3, 3], atol=1e-6)
    Physics.sync_transforms()
    np.testing.assert_allclose(PublicPhysics.get_rigidbody_states([body])["linear_velocity"], state["linear_velocity"])
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    dt = manager.get_fixed_time_step()
    for _ in range(30):
        manager.step(dt)
    assert body.velocity.y == pytest.approx(3 * (1 - .25*dt)**30, abs=.001)


def exercise(project, mode, repeats, motion):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    observations = {}
    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        scene = manager.get_active_scene()
        owner = scene.create_game_object("Deferred body")
        collider = owner.add_component("BoxCollider")
        body = owner.add_component("Rigidbody") if motion != "static" or mode == "motion" else None
        if body is not None:
            body.is_kinematic = motion == "kinematic"
            if mode == "motion":
                body.enabled = motion != "dynamic"

        def hits():
            return Physics.raycast_all(Vector3(0, 3, 0), Vector3(0, -1, 0), 6)

        assert len(hits()) == 1
        if mode == "destroy_pending_force":
            from infernux.lib import ForceMode
            manager.play()
            manager.pause()
            for _ in range(repeats):
                collider.enabled = False
                Physics.sync_transforms()
                collider.enabled = True
                body.add_force(Vector3(0, 4, 0), ForceMode.Impulse)
                assert body.velocity.y == 0
                scene.destroy_game_object(owner)
                manager.step(manager.get_fixed_time_step())
                owner = scene.create_game_object("Replacement")
                collider = owner.add_component("BoxCollider")
                body = owner.add_component("Rigidbody")
                Physics.sync_transforms()
                assert len(hits()) == 1
            observations["passed"] = True
            return
        if mode.startswith("pending_force_"):
            from infernux.lib import ForceMode
            collider.enabled = False
            Physics.sync_transforms()
            collider.enabled = True
            angular = mode.endswith("angular")
            if angular:
                body.add_torque(Vector3(0, 0, 2 / 3), ForceMode.Impulse)
            elif mode.endswith("position"):
                body.add_force_at_position(Vector3(0, 4, 0), Vector3(0, 0, 0), ForceMode.Impulse)
            else:
                body.add_force(Vector3(0, 4, 0), ForceMode.Impulse)
            if repeats == 1:
                collider.enabled = False
                collider.enabled = True
            elif repeats == 2:
                body.enabled = False
                body.enabled = True
            def velocity():
                return body.angular_velocity.z if angular else body.velocity.y
            assert velocity() == 0, "nonresident body cannot consume force before publication"
            assert body.is_sleeping()
            Physics.sync_transforms()
            assert velocity() == pytest.approx(4)
            Physics.sync_transforms()
            assert velocity() == pytest.approx(4)
            observations["passed"] = True
            return
        if mode == "configuration":
            exercise_disabled_configuration(owner, collider, body)
            observations["passed"] = True
            return
        if mode in ("add", "motion"):
            collider.enabled = False
            Physics.sync_transforms()
            assert len(hits()) == 0
        if mode == "motion":
            collider.enabled = True
            if repeats:
                collider.enabled = False
            body.enabled = motion == "dynamic"
            collider.enabled = True
        else:
            for index in range(repeats):
                collider.enabled = mode == "add"
                if index + 1 < repeats:
                    collider.enabled = mode != "add"
        Physics.sync_transforms()
        values = hits()
        observations["hit_count"] = len(values)
        observations["body_ids"] = [value.body_id for value in values]
        expected = 0 if mode == "remove" else 1
        assert len(values) == expected, observations
        if values:
            assert values[0].collider.component_id == collider.component_id
        if mode == "motion" and motion == "dynamic":
            assert not body.is_sleeping(), "published dynamic body must be activated before Play wakes the scene"
        manager.play()
        manager.pause()
        for _ in range(30):
            manager.step(manager.get_fixed_time_step())
        if mode != "remove":
            y = owner.transform.position.y
            observations["y"] = y
            if motion == "dynamic":
                assert y < -.5, observations
            else:
                assert y == pytest.approx(0, abs=.001), observations
        observations["passed"] = True
    finally:
        engine.exit()
        (project / "broadphase-evidence.json").write_text(json.dumps(observations), encoding="utf-8")


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
