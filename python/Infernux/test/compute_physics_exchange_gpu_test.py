"""Real Vulkan-to-Jolt regression for resident compact impulse feedback."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np

import Infernux as inx
from Infernux.lib import SceneManager, Vector3
from Infernux.physics import Physics


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-compute-physics-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        engine = inx.Engine()
        native = engine.get_native_engine()
        manager = SceneManager.instance()
        scene = None
        linear = angular = scale = None
        failure = None
        try:
            try:
                engine.init_renderer(64, 64, str(project))
            except (OSError, RuntimeError) as exception:
                print(f"Compute physics exchange test skipped: {exception}")
                return 77

            scene = manager.create_scene("ComputePhysicsExchange")
            manager.set_active_scene(scene)
            bodies = []
            for index, mass in enumerate((2.0, 4.0)):
                game_object = scene.create_game_object(f"Body {index}")
                game_object.transform.position = Vector3(float(index) * 3.0, 4.0, 0.0)
                body = game_object.add_component("Rigidbody")
                body.mass = mass
                body.use_gravity = False
                body.drag = 0.0
                body.angular_drag = 0.0
                game_object.add_component("BoxCollider")
                bodies.append(body)
            manager.play()
            manager.pause()

            rigidbody_state = {
                "position": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "center_of_mass": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "linear_velocity": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "angular_velocity": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "inverse_mass": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "rotation": inx.buffer(shape=4, dtype=inx.vector4, device="gpu"),
                "inverse_inertia": inx.buffer(shape=(4, 3, 3), dtype=float, device="gpu"),
            }
            box_state = {
                "body_index": inx.buffer(shape=4, dtype=np.int32, device="gpu"),
                "center": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "rotation": inx.buffer(shape=4, dtype=inx.vector4, device="gpu"),
                "half_extents": inx.buffer(shape=4, dtype=inx.vector3, device="gpu"),
                "friction": inx.buffer(shape=4, dtype=float, device="gpu"),
                "bounciness": inx.buffer(shape=4, dtype=float, device="gpu"),
            }
            with inx.compute._record_commands():
                combined = Physics.get_rigidbody_states_and_box_states(bodies, rigidbody_state, box_state)
                assert combined[0] is rigidbody_state
                assert combined[1] is box_state
            assert box_state["count"] == 2
            np.testing.assert_allclose(
                rigidbody_state["position"].get_data(offset=0, count=2).numpy(),
                [[0, 4, 0], [3, 4, 0]], rtol=1e-6, atol=1e-6,
            )
            np.testing.assert_array_equal(
                box_state["body_index"].get_data(offset=0, count=2).numpy(), [0, 1]
            )

            # The broad-phase-to-SoA path must not require a Python candidate
            # list followed by a second collider traversal.  It returns the
            # same authoritative body wrappers only for the caller's feedback
            # mapping; the box descriptors are submitted by the native pass.
            queried_state, queried = Physics.query_rigidbody_states_and_box_states_in_bounds(
                (-1.0, 0.0, -2.0), (5.0, 8.0, 2.0), rigidbody_state, box_state
            )
            assert queried_state is rigidbody_state
            assert queried is box_state
            assert queried["count"] == 2
            assert {body.component_id for body in queried["rigidbodies"]} == {
                body.component_id for body in bodies
            }

            source = np.array([[0.25, 0.5, -0.75], [-0.4, 0.8, 0.2], [9, 9, 9], [9, 9, 9]], np.float32)
            linear = inx.buffer(shape=4, dtype=inx.vector3, device="gpu", data=source)
            angular = inx.buffer(shape=4, dtype=inx.vector3, device="gpu")

            @inx.compute.kernel
            def scale(values, factor):
                i = inx.compute.index(values)
                values[i] = values[i] * factor

            # Compile and submit the specialization before measuring the
            # steady-state fixed-step exchange. First-use JIT is reported by
            # compiler tests and is not a per-step synchronization cost.
            inx.compute.launch(scale, params=(linear, 1.0))
            linear.get_data()
            started = time.perf_counter()
            with inx.compute.recording():
                inx.compute.launch(scale, params=(linear, 2.0))
                Physics.apply_rigidbody_impulses(bodies, linear, angular)
            feedback_wait_ms = (time.perf_counter() - started) * 1000.0

            state = Physics.get_rigidbody_states(bodies)
            expected = source[:2] * 2.0 / np.array([[2.0], [4.0]], np.float32)
            np.testing.assert_allclose(state["linear_velocity"], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(state["angular_velocity"], 0)
            before_step = state["position"].copy()
            manager.step()
            after_step = Physics.get_rigidbody_states(bodies)["position"]
            np.testing.assert_allclose(
                after_step,
                before_step + expected * manager.get_fixed_time_step(),
                rtol=1e-5,
                atol=1e-5,
            )
            print(
                "INFERNUX_COMPUTE_PHYSICS_FEEDBACK "
                f"bodies={len(bodies)} readback_bytes={len(bodies) * 2 * 3 * 4} "
                f"wait_ms={feedback_wait_ms:.6f}",
                flush=True,
            )
        except BaseException:
            failure = traceback.format_exc()
        finally:
            try:
                if manager.is_playing():
                    manager.stop()
                if scene is not None:
                    manager.unload_scene(scene)
                inx.compute._release_engine_resources()
                native.cleanup()
            except BaseException:
                if failure is None:
                    failure = traceback.format_exc()
        if failure is not None:
            raise RuntimeError(failure)
    print("INFERNUX_COMPUTE_PHYSICS_EXCHANGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
