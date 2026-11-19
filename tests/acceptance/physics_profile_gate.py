"""Deterministic fixed-scale profile gate through normal headless CPU Play."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from infernux import Engine, InxComponent, PhysicsBackend, RigidbodyInterpolation
from infernux.engine.build_settings import BUILD_SETTINGS_DEFAULTS
from infernux.lib import CollisionDetectionMode, RuntimeMode, SceneManager, Vector3, _Infernux
from infernux.physics import Physics
from infernux.physics.settings import DEFAULT_PHYSICS_SETTINGS, save as save_physics_settings
from infernux.version import ENGINE_RELEASE

BODY_COUNT = 64
FIXED_DELTA = 1.0 / 60.0


class ProfileContacts(InxComponent):
    callbacks: int = 0

    def on_collision_enter(self, collision):
        self.callbacks += 1

    def on_collision_stay(self, collision):
        self.callbacks += 1


def _tick_fixed(engine, manager):
    before = int(manager.completed_fixed_step_index)
    engine.tick(float(manager.get_fixed_time_step()))
    profile = manager.get_last_frame_profile()
    after = int(manager.completed_fixed_step_index)
    deadline = time.monotonic() + 10.0
    while after == before:
        if time.monotonic() >= deadline:
            raise AssertionError(f"one requested Fixed did not complete: {before}, {profile}")
        engine.tick(0.0)
        time.sleep(0.001)
        profile = manager.get_last_frame_profile()
        after = int(manager.completed_fixed_step_index)
    if after != before + 1 or profile["fixed_steps"] != 1:
        raise AssertionError(f"one requested interval must complete one Fixed: {before} -> {after}, {profile}")
    if int(manager.published_fixed_step_index) != after:
        raise AssertionError("completed CPU Fixed was not published")
    return profile


def _run_profile_gate(engine, manager, scene):
    assert scene.physics_backend == PhysicsBackend.CPU
    floor = scene.create_game_object("ProfileFloor")
    floor.transform.position = Vector3(0.0, -0.5, 0.0)
    floor_collider = floor.add_component("BoxCollider")
    floor_collider.size = Vector3(24.0, 1.0, 24.0)
    # contact_events measures delivered receivers, not captured stream records.
    receiver = floor.add_component(ProfileContacts)

    bodies = []
    for index in range(BODY_COUNT):
        row, column = divmod(index, 8)
        obj = scene.create_game_object(f"ProfileBody{index}")
        obj.transform.position = Vector3(column * 1.1 - 3.85, 0.5, row * 1.1 - 3.85)
        rigidbody = obj.add_component("Rigidbody")
        rigidbody.use_gravity = False
        if index == 0:
            rigidbody.interpolation = RigidbodyInterpolation.Interpolate
        obj.add_component("BoxCollider")
        bodies.append(rigidbody)

    stack = []
    for index in range(4):
        obj = scene.create_game_object(f"StabilityStack{index}")
        obj.transform.position = Vector3(7.0, 0.5 + index * 1.01, 0.0)
        rigidbody = obj.add_component("Rigidbody")
        obj.add_component("BoxCollider")
        stack.append((obj, rigidbody))

    profiles = []
    peak_captured_contacts = 0
    for _ in range(8):
        _tick_fixed(engine, manager)
        profiles.append(manager.get_last_frame_profile())
        peak_captured_contacts = max(peak_captured_contacts, len(Physics.get_contact_events()["type"]))

    peak_events = max(int(profile["contact_events"]) for profile in profiles)
    peak_step_ms = max(profile["physics_step_ms"] for profile in profiles)
    peak_event_ms = max(profile["physics_events_ms"] for profile in profiles)
    if peak_events < BODY_COUNT:
        raise AssertionError(f"contact storm produced only {peak_events} events for {BODY_COUNT} bodies")
    if receiver.callbacks < BODY_COUNT or peak_captured_contacts < BODY_COUNT:
        raise AssertionError(f"contact storm missed callbacks/stream: {receiver.callbacks}/{peak_captured_contacts}")
    if peak_step_ms > 50.0:
        raise AssertionError(f"64-body physics step exceeded 50 ms: {peak_step_ms:.3f} ms")
    if peak_event_ms > 50.0:
        raise AssertionError(f"64-body contact dispatch exceeded 50 ms: {peak_event_ms:.3f} ms")

    for _ in range(360):
        _tick_fixed(engine, manager)
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
    # Publish terminal sleep-transition poses before sampling a sleeping Fixed.
    sleep_transition = _tick_fixed(engine, manager)
    _tick_fixed(engine, manager)
    sleeping = manager.get_last_frame_profile()
    if sleeping["rigidbody_sync_candidates"] != 0:
        raise AssertionError("sleeping-body CPU pose publication regressed to non-empty work")
    if sleeping["interpolation_candidates"] != 0:
        raise AssertionError("sleeping-body interpolation regressed to a non-empty scan")

    bodies[0].wake_up()
    bodies[0].velocity = Vector3(1.0, 0.0, 0.0)
    old_x = float(bodies[0].position.x)
    # Wake/velocity commands may legitimately publish their control-boundary
    # pose before integration. Measure that host separately from the Fixed.
    engine.tick(0.0)
    control = manager.get_last_frame_profile()
    if control["fixed_steps"] != 0 or control["rigidbody_sync_candidates"] != 1:
        raise AssertionError(f"one awakened body did not publish one control pose: {control}")
    if not math.isclose(float(bodies[0].game_object.transform.position.x), old_x, abs_tol=1e-4):
        raise AssertionError("control publication unexpectedly advanced the physical position")
    _tick_fixed(engine, manager)
    single_awake = manager.get_last_frame_profile()
    if single_awake["rigidbody_sync_candidates"] != 1:
        raise AssertionError(f"one awake body did not produce one CPU pose publication candidate: {single_awake}")
    if single_awake["interpolation_candidates"] != 1:
        raise AssertionError(f"one awake body did not produce one interpolation candidate: {single_awake}")

    current_x = float(bodies[0].position.x)
    if current_x <= old_x:
        raise AssertionError("the interpolation probe did not physically move")
    displayed_before = float(bodies[0].game_object.transform.position.x)
    if not math.isclose(displayed_before, old_x, abs_tol=1e-4):
        raise AssertionError(f"fixed-boundary presentation did not retain the previous pose: {displayed_before}")
    fixed_before_half = int(manager.completed_fixed_step_index)
    engine.tick(FIXED_DELTA / 2)
    fractional = manager.get_last_frame_profile()
    if fractional["fixed_steps"] != 0 or int(manager.completed_fixed_step_index) != fixed_before_half:
        raise AssertionError("a half host frame unexpectedly advanced a Fixed")
    if fractional["rigidbody_sync_candidates"] != 0 or fractional["interpolation_candidates"] != 1:
        raise AssertionError(f"fractional presentation should only interpolate one body: {fractional}")
    displayed_half = float(bodies[0].game_object.transform.position.x)
    if not math.isclose(displayed_half, (old_x + current_x) / 2, abs_tol=1e-4):
        raise AssertionError(f"half-frame interpolation did not display the midpoint: {displayed_half}")
    # A mode edit at the same alpha must not be mistaken for a duplicate
    # interpolation request: None must display the latest physical endpoint.
    bodies[0].interpolation = RigidbodyInterpolation.None_
    engine.tick(0.0)
    mode_switch = manager.get_last_frame_profile()
    displayed_after_switch = float(bodies[0].game_object.transform.position.x)
    if mode_switch["fixed_steps"] != 0 or int(manager.completed_fixed_step_index) != fixed_before_half:
        raise AssertionError("a zero-delta interpolation mode edit unexpectedly advanced a Fixed")
    if not math.isclose(displayed_after_switch, current_x, abs_tol=1e-4):
        raise AssertionError(f"disabling interpolation did not display the current physical pose: {displayed_after_switch}")
    # Complete the fractional interval before the one-Fixed CCD check.
    engine.tick(FIXED_DELTA / 2)
    if int(manager.completed_fixed_step_index) != fixed_before_half + 1:
        raise AssertionError("two half host frames did not complete exactly one Fixed")

    ccd_pair = []
    for side, velocity in ((-1, 600.0), (1, -600.0)):
        obj = scene.create_game_object(f"DynamicCCD{side}")
        obj.transform.position = Vector3(side * 4.0, 5.0, 8.0)
        rigidbody = obj.add_component("Rigidbody")
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0
        rigidbody.max_linear_velocity = 1000.0
        rigidbody.collision_detection_mode = CollisionDetectionMode.ContinuousDynamic
        collider = obj.add_component("SphereCollider")
        collider.radius = 0.25
        rigidbody.velocity = Vector3(velocity, 0.0, 0.0)
        ccd_pair.append((obj, rigidbody))

    ccd_before = int(manager.completed_fixed_step_index)
    _tick_fixed(engine, manager)
    ccd_positions = [float(obj.transform.position.x) for obj, _ in ccd_pair]
    ccd_velocities = [float(rigidbody.velocity.x) for _, rigidbody in ccd_pair]
    left, right = ccd_positions
    if left >= 0.0 or right <= 0.0:
        raise AssertionError(f"ContinuousDynamic pair crossed through each other: {ccd_positions}")
    if not (-3.75 < left < right < 3.75) or right - left > 0.75:
        raise AssertionError(f"CCD bodies did not actually approach contact: {ccd_positions}")
    if ccd_velocities[0] - ccd_velocities[1] >= 1200.0 * 0.95:
        raise AssertionError(f"CCD did not reduce the 1200 m/s closing speed: {ccd_velocities}")
    if any(not math.isfinite(v) or abs(v) > 600.01 for v in ccd_velocities):
        raise AssertionError(f"CCD produced invalid/excess velocity: {ccd_velocities}")

    print(json.dumps({
        "bodies": BODY_COUNT,
        "peak_contact_events": peak_events,
        "peak_captured_contacts": peak_captured_contacts,
        "contact_callbacks": receiver.callbacks,
        "peak_physics_step_ms": round(peak_step_ms, 4),
        "peak_contact_dispatch_ms": round(peak_event_ms, 4),
        "sleeping_sync_candidates": int(sleeping["rigidbody_sync_candidates"]),
        "single_awake_sync_candidates": int(single_awake["rigidbody_sync_candidates"]),
        "stack_bodies_naturally_sleeping": len(stack),
        "backend": "cpu",
        "native": _Infernux.__file__,
        "single_awake_old_x": old_x,
        "single_awake_current_x": current_x,
        "control_sync_candidates": int(control["rigidbody_sync_candidates"]),
        "control_interpolation_candidates": int(control["interpolation_candidates"]),
        "sleep_transition_sync_candidates": int(sleep_transition["rigidbody_sync_candidates"]),
        "single_awake_interpolation_candidates": int(single_awake["interpolation_candidates"]),
        "fractional_host_frame_fixed_steps": int(fractional["fixed_steps"]),
        "fractional_host_frame_interpolation_candidates": int(fractional["interpolation_candidates"]),
        "interpolated_x": displayed_half,
        "mode_switch_fixed_steps": int(mode_switch["fixed_steps"]),
        "mode_switch_sync_candidates": int(mode_switch["rigidbody_sync_candidates"]),
        "mode_switch_interpolation_candidates": int(mode_switch["interpolation_candidates"]),
        "mode_switch_displayed_x": displayed_after_switch,
        "ccd_completed_before": ccd_before,
        "ccd_completed_after": int(manager.completed_fixed_step_index),
        "ccd_positions_x": ccd_positions,
        "ccd_velocities_x": ccd_velocities,
    }, sort_keys=True))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="infernux-physics-profile-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        settings = project / "ProjectSettings"
        settings.mkdir()
        (project / ".infernux-version").write_text(ENGINE_RELEASE + "\n", encoding="utf-8")
        (settings / "PythonRuntime.json").write_text(
            json.dumps({"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}"}) + "\n",
            encoding="utf-8",
        )
        (settings / "requirements.txt").write_text("# No project dependencies\n", encoding="utf-8")
        (settings / "BuildSettings.json").write_text(json.dumps(BUILD_SETTINGS_DEFAULTS) + "\n", encoding="utf-8")
        save_physics_settings(str(project), {**DEFAULT_PHYSICS_SETTINGS, "fixed_delta_time": FIXED_DELTA})

        engine = Engine(mode=RuntimeMode.Headless)
        try:
            engine.init_headless(str(project))
            assert not engine.get_native_engine().has_renderer
            manager = SceneManager.instance()
            assert math.isclose(manager.get_fixed_time_step(), FIXED_DELTA, abs_tol=1e-7)
            play_mode = engine.get_play_mode_manager()
            assert play_mode.enter_play_mode()
            deadline = time.monotonic() + 20.0
            while not play_mode.is_playing:
                assert time.monotonic() < deadline, play_mode.state
                engine.tick(0.0)
                time.sleep(0.001)
            assert len(manager._runtime_world_ids) == 1
            # Play owns a clone; create this fixture in that actual runtime world.
            with _Infernux._RuntimeWorldScope(manager._runtime_world_ids[0]):
                # Games opt into captured contacts when a callback/solver consumes them.
                Physics.set_contact_event_stream_enabled(True)
                try:
                    _run_profile_gate(engine, manager, manager.get_active_scene())
                finally:
                    Physics.set_contact_event_stream_enabled(False)
            assert play_mode.exit_play_mode()
            deadline = time.monotonic() + 20.0
            while not play_mode.is_edit_mode:
                assert time.monotonic() < deadline, play_mode.state
                engine.tick(0.0)
                time.sleep(0.001)
        finally:
            engine.exit()


if __name__ == "__main__":
    main()
