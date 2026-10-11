"""Independent actors retain world poses across the real frame-cache boundary."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run(project, order, interpolation, sleeping, rotate, reparent=False, graphical=False):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(project),
         order, str(interpolation), str(int(sleeping)), str(int(rotate)), str(int(reparent)), str(int(graphical))],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    output = result.stdout + result.stderr
    (project / "hierarchy-runtime.log").write_text(output, encoding="utf-8")
    assert result.returncode == 0, output
    for diagnostic in ("VUID-", "SYNC-HAZARD", "Validation Error"):
        assert diagnostic not in output, output
    assert "HIERARCHY_PUBLICATION passed" in result.stdout


@pytest.mark.parametrize("order", ["012", "210"])
@pytest.mark.parametrize("interpolation", [0, 1])
@pytest.mark.parametrize("sleeping", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_nested_bodies_preserve_independent_world_trajectories(
    tmp_path, order, interpolation, sleeping, rotate,
):
    _run(tmp_path, order, interpolation, sleeping, rotate)


@pytest.mark.parametrize("order", ["".join(p) for p in itertools.permutations("012")])
@pytest.mark.parametrize("interpolation", [0, 1])
def test_nested_actor_allocation_orders_and_runtime_reparent(tmp_path, order, interpolation):
    _run(tmp_path, order, interpolation, False, True, reparent=True)


@pytest.mark.parametrize("interpolation", [0, 1])
def test_graphical_frames_keep_sleeping_descendant_bodies_fixed(tmp_path, interpolation):
    _run(tmp_path, "210", interpolation, True, True, graphical=True)


def exercise(project, order, interpolation, sleeping, rotate, reparent, graphical):
    from infernux import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Graphical if graphical else RuntimeMode.Headless)
    if graphical:
        engine.init_renderer(64, 64, str(project))
    else:
        engine.init_headless(str(project))
    manager = SceneManager.instance()
    dt = 1.0 / 60.0

    def xyz(value):
        return value.x, value.y, value.z

    def same_rotation(actual, expected):
        dot = sum(getattr(actual, axis) * getattr(expected, axis) for axis in "xyzw")
        assert abs(dot) == pytest.approx(1.0, abs=2e-5)

    try:
        scene = manager.get_active_scene()
        objects = [scene.create_game_object(f"Actor{i}") for i in range(3)]
        objects[1].set_parent(objects[0], False)
        # A non-actor intermediary must not break ancestor ordering.
        spacer = scene.create_game_object("Spacer")
        spacer.set_parent(objects[1], False)
        objects[2].set_parent(spacer, False)
        for i, owner in enumerate(objects):
            owner.transform.position = Vector3(20 * i, 0, 0)
        follower = scene.create_game_object("VisualChild")
        follower.set_parent(objects[0], False)
        follower.transform.local_position = Vector3(0, 3, 0)
        manager.set_fixed_time_step(dt)
        manager.play()
        bodies = {}
        body_ids = {}
        for key in order:
            i = int(key)
            body = objects[i].add_component("Rigidbody")
            body.use_gravity = False
            body.drag = 0.0
            body.angular_drag = 0.0
            body.interpolation = interpolation
            objects[i].add_component("SphereCollider").radius = 0.25
            Physics.sync_transforms()
            bodies[i] = body
            hit = Physics.raycast(Vector3(20 * i, 2, 0), Vector3(0, -1, 0), 4)
            assert hit is not None and hit.game_object is objects[i]
            body_ids[i] = int(hit.body_id)
        assert [i for i in sorted(body_ids, key=body_ids.get)] == [int(key) for key in order]
        engine.tick(dt)
        engine.tick(dt)
        initial_positions = {i: xyz(bodies[i].position) for i in (1, 2)}
        initial_rotations = {i: bodies[i].rotation for i in (1, 2)}
        if sleeping:
            for i in (1, 2):
                bodies[i].sleep()
                assert bodies[i].is_sleeping()
            engine.tick(dt)
            engine.tick(dt)
        bodies[0].velocity = Vector3(2, 0, 0)
        if rotate:
            bodies[0].angular_velocity = Vector3(0, 0, 1)
        alternate_parent = scene.create_game_object("AlternateParent")
        alternate_parent.set_parent(objects[0], False)
        alternate_parent.transform.local_position = Vector3(0, 7, 0)
        for frame in range(12):
            if reparent and frame == 5:
                objects[1].set_parent(alternate_parent, True)
            engine.tick(dt)
            for i in (1, 2):
                assert xyz(bodies[i].position) == pytest.approx(initial_positions[i], abs=2e-4), (
                    frame, i, order, xyz(bodies[i].position), initial_positions[i]
                )
                assert xyz(objects[i].transform.position) == pytest.approx(initial_positions[i], abs=2e-4)
                same_rotation(bodies[i].rotation, initial_rotations[i])
                same_rotation(objects[i].transform.rotation, initial_rotations[i])
                assert xyz(bodies[i].velocity) == pytest.approx((0, 0, 0), abs=1e-6)
                if sleeping:
                    assert bodies[i].is_sleeping(), "physics presentation must not wake a sleeping child"
            expected_follower = objects[0].transform.transform_point(Vector3(0, 3, 0))
            assert xyz(follower.transform.position) == pytest.approx(xyz(expected_follower), abs=2e-5)
        assert bodies[0].position.x > 0.3
        if rotate:
            assert abs(bodies[0].rotation.z) > 0.05
        # A user's explicit hierarchy move is still an authored teleport.
        # Only solver presentation is prevented from dragging nested actors.
        bodies[0].velocity = Vector3(0, 0, 0)
        bodies[0].angular_velocity = Vector3(0, 0, 0)
        old_root = objects[0].transform.position
        objects[0].transform.position = Vector3(old_root.x + 4, old_root.y, old_root.z)
        Physics.sync_transforms()
        for i in (1, 2):
            expected = (initial_positions[i][0] + 4, *initial_positions[i][1:])
            assert xyz(bodies[i].position) == pytest.approx(expected, abs=2e-4)
        engine.tick(dt)
        for i in (1, 2):
            expected = (initial_positions[i][0] + 4, *initial_positions[i][1:])
            assert xyz(bodies[i].position) == pytest.approx(expected, abs=2e-4)
            assert xyz(objects[i].transform.position) == pytest.approx(expected, abs=2e-4)
        print("HIERARCHY_PUBLICATION passed", order, interpolation, sleeping, rotate, reparent, flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], *map(int, sys.argv[3:]))
