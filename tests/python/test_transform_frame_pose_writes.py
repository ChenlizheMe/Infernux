"""Timeline local poses survive the actual frame cache and Jolt publication."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fps", [30, 60, 144, 240])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("write_mode", ["scalar", "batch_local", "batch_world_position"])
def test_animation_final_pose_survives_native_frame_commit(tmp_path, fps, nested, write_mode):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), str(fps), str(int(nested)), write_mode],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ANIMATION_POSE passed" in result.stdout


def exercise(project, fps, nested, write_mode):
    import numpy as np
    from infernux.batch import batch_write
    from infernux import Engine, InxComponent
    from infernux.core.animation_timeline import AnimationTimeline, TimelineKeyframe
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, SceneManager, Vector3, quatf

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    engine.init_headless(str(project))
    try:
        manager = SceneManager.instance()
        scene = manager.get_active_scene()
        owner = scene.create_game_object("Animated bridge")
        if nested:
            parent = scene.create_game_object("Rotated parent")
            parent.transform.set_local_trs(10, 4, -3, 0, 25, 0, 2, 2, 2)
            owner.set_parent(parent, False)
        owner.transform.local_euler_angles = Vector3(0, 90, 0)
        owner.add_component("BoxCollider")
        body = owner.add_component("Rigidbody")
        body.is_kinematic = True
        body.use_gravity = False
        timeline = AnimationTimeline(duration=1.5, apply_mode="absolute", keyframes=[
            TimelineKeyframe(time=0, position=[0, 0, 0], rotation=[0, 90, 0], scale=[2, .6, 12]),
            TimelineKeyframe(time=1.5, position=[1, 0, 0], rotation=[0, 0, 0],
                             scale=[2, .6, 12], interp="ease_in_out"),
        ])
        elapsed = 0.0

        class Animate(InxComponent):
            def update(self, dt):
                nonlocal elapsed
                if elapsed >= timeline.duration:
                    return
                elapsed = min(elapsed + dt, timeline.duration)
                position, rotation, scale = timeline.sample(elapsed)
                if write_mode == "scalar":
                    owner.transform.set_local_trs(*position, *rotation, *scale)
                else:
                    if write_mode == "batch_world_position":
                        world = tuple(parent.transform.transform_point(Vector3(*position))) if nested else position
                        batch_write([owner.transform], np.array([world], dtype=np.float32), "position")
                    else:
                        batch_write([owner.transform], np.array([position], dtype=np.float32), "local_position")
                    batch_write([owner.transform], np.array([rotation], dtype=np.float32), "local_euler_angles")
                    batch_write([owner.transform], np.array([scale], dtype=np.float32), "local_scale")

        owner.add_component(Animate)
        manager.play()
        for _ in range(fps * 2):
            engine.tick(1 / fps)
        assert elapsed == timeline.duration
        expected_position = parent.transform.transform_point(Vector3(1, 0, 0)) if nested else Vector3(1, 0, 0)
        expected_rotation = parent.transform.rotation if nested else quatf.identity()
        for _ in range(8):
            engine.tick(1 / fps)
            assert tuple(owner.transform.local_position) == pytest.approx((1, 0, 0), abs=2e-5)
            assert tuple(body.position) == pytest.approx(tuple(expected_position), abs=3e-5)
            for actual in (body.rotation, owner.transform.rotation):
                sign = 1 if sum(a * b for a, b in zip(actual, expected_rotation)) >= 0 else -1
                assert tuple(actual) == pytest.approx(tuple(sign * v for v in expected_rotation), abs=2e-6)
        print("ANIMATION_POSE passed", fps, nested, write_mode, flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])), sys.argv[4])
