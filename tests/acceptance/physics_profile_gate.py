"""Deterministic fixed-scale profile gate for the headless physics pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from infernux import Engine
from infernux.lib import CollisionDetectionMode, RuntimeMode, SceneManager, Vector3


BODY_COUNT = 64
FIXED_DELTA = 1.0 / 60.0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="infernux-physics-profile-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()

        engine = Engine(mode=RuntimeMode.Headless)
        engine.init_headless(str(project))
        try:
            manager = SceneManager.instance()
            scene = manager.get_active_scene()

            floor = scene.create_game_object("ProfileFloor")
            floor.transform.position = Vector3(0.0, -0.5, 0.0)
            floor_collider = floor.add_component("BoxCollider")
            floor_collider.size = Vector3(24.0, 1.0, 24.0)

            bodies = []
            for index in range(BODY_COUNT):
                row, column = divmod(index, 8)
                obj = scene.create_game_object(f"ProfileBody{index}")
                obj.transform.position = Vector3(column * 1.1 - 3.85, 0.5, row * 1.1 - 3.85)
                rigidbody = obj.add_component("Rigidbody")
                rigidbody.use_gravity = False
                obj.add_component("BoxCollider")
                bodies.append(rigidbody)

            stack = []
            for index in range(4):
                obj = scene.create_game_object(f"StabilityStack{index}")
                obj.transform.position = Vector3(7.0, 0.5 + index * 1.01, 0.0)
                rigidbody = obj.add_component("Rigidbody")
                obj.add_component("BoxCollider")
                stack.append((obj, rigidbody))

            manager.play()
            profiles = []
            for _ in range(8):
                engine.tick(FIXED_DELTA)
                profiles.append(manager.get_last_frame_profile())

            peak_events = max(int(profile["contact_events"]) for profile in profiles)
            peak_step_ms = max(profile["physics_step_ms"] for profile in profiles)
            peak_event_ms = max(profile["physics_events_ms"] for profile in profiles)
            if peak_events < BODY_COUNT:
                raise AssertionError(f"contact storm produced only {peak_events} events for {BODY_COUNT} bodies")
            if peak_step_ms > 50.0:
                raise AssertionError(f"64-body physics step exceeded 50 ms: {peak_step_ms:.3f} ms")
            if peak_event_ms > 50.0:
                raise AssertionError(f"64-body contact dispatch exceeded 50 ms: {peak_event_ms:.3f} ms")

            for _ in range(360):
                engine.tick(FIXED_DELTA)
            for index, (obj, rigidbody) in enumerate(stack):
                position = obj.transform.position
                if abs(position.x - 7.0) > 0.1 or abs(position.z) > 0.1:
                    raise AssertionError(f"stack body {index} drifted laterally to {tuple(position)}")
                if abs(position.y - (0.5 + index)) > 0.12:
                    raise AssertionError(f"stack body {index} settled at invalid height {position.y:.4f}")
                if not rigidbody.is_sleeping():
                    raise AssertionError(f"stack body {index} failed to enter natural sleep")

            for rigidbody in bodies:
                rigidbody.sleep()
            engine.tick(FIXED_DELTA)
            sleeping = manager.get_last_frame_profile()
            if sleeping["rigidbody_sync_candidates"] != 0:
                raise AssertionError("sleeping-body pose readback regressed to a non-empty scan")
            if sleeping["interpolation_candidates"] != 0:
                raise AssertionError("sleeping-body interpolation regressed to a non-empty scan")

            bodies[0].wake_up()
            engine.tick(FIXED_DELTA)
            single_awake = manager.get_last_frame_profile()
            if single_awake["rigidbody_sync_candidates"] != 1:
                raise AssertionError("one awake body did not produce exactly one pose-readback candidate")
            if single_awake["interpolation_candidates"] != 1:
                raise AssertionError("one awake body did not produce exactly one interpolation candidate")

            ccd_pair = []
            for side, velocity in ((-1, 600.0), (1, -600.0)):
                obj = scene.create_game_object(f"DynamicCCD{side}")
                obj.transform.position = Vector3(side * 4.0, 5.0, 8.0)
                rigidbody = obj.add_component("Rigidbody")
                rigidbody.use_gravity = False
                rigidbody.drag = 0.0
                rigidbody.max_linear_velocity = 1000.0
                rigidbody.collision_detection_mode = CollisionDetectionMode.Continuous
                collider = obj.add_component("SphereCollider")
                collider.radius = 0.25
                rigidbody.velocity = Vector3(velocity, 0.0, 0.0)
                ccd_pair.append((obj, rigidbody))

            engine.tick(FIXED_DELTA)
            ccd_profile = manager.get_last_frame_profile()
            if ccd_profile["dynamic_ccd_splits"] < 1:
                raise AssertionError("dynamic Continuous pair did not trigger a TOI split")
            if ccd_pair[0][0].transform.position.x >= 0.0 or ccd_pair[1][0].transform.position.x <= 0.0:
                raise AssertionError("dynamic Continuous pair crossed through each other")

            print(json.dumps({
                "bodies": BODY_COUNT,
                "peak_contact_events": peak_events,
                "peak_physics_step_ms": round(peak_step_ms, 4),
                "peak_contact_dispatch_ms": round(peak_event_ms, 4),
                "sleeping_sync_candidates": int(sleeping["rigidbody_sync_candidates"]),
                "single_awake_sync_candidates": int(single_awake["rigidbody_sync_candidates"]),
                "stack_bodies_naturally_sleeping": len(stack),
                "dynamic_ccd_splits": int(ccd_profile["dynamic_ccd_splits"]),
            }, sort_keys=True))
        finally:
            engine.exit()


if __name__ == "__main__":
    main()
