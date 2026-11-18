"""Yanshi rigid_diagnostics probe on Infernux (050 automation branch).

Builds the least native world the task card allows (no gravity, no contacts,
no damping), enters Play Mode, settles the startup frame, applies the task's
initial conditions to the **runtime** body, then advances exactly one paused
physics step at a time and samples the pose without ever letting a read
advance the world.

Expected samples: x(k) = 1.2 + 0.005 k, k = 0..10; relative time t(10) = 0.10 s.
Tolerance: position 0.1 mm, simulation time 1 us.

Usage:
    python tools/yanshi/rigid_diagnostics_probe.py [--project DIR] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import traceback

DELTA = 0.01
START_X = 1.2
VELOCITY_X = 0.5
SAMPLES = 11
POSITION_TOLERANCE = 1e-4
TIME_TOLERANCE = 1e-6


def _prepare_project(root: str) -> None:
    os.makedirs(os.path.join(root, "Assets"), exist_ok=True)
    settings_dir = os.path.join(root, "ProjectSettings")
    os.makedirs(settings_dir, exist_ok=True)
    with open(os.path.join(root, ".infernux-version"), "w", encoding="utf-8") as handle:
        handle.write("0.4.8\n")
    from infernux.physics.settings import DEFAULT_PHYSICS_SETTINGS

    settings = dict(DEFAULT_PHYSICS_SETTINGS)
    settings.update(
        {
            "gravity": [0.0, 0.0, 0.0],
            "fixed_delta_time": DELTA,
            "max_fixed_delta_time": 1.0 / 3.0,
            "max_concurrency": 1,
        }
    )
    with open(os.path.join(settings_dir, "PhysicsSettings.json"), "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def _build_fixture(scene) -> None:
    import infernux as inx

    body_object = scene.create_game_object("Slider")
    body_object.get_transform().position = inx.Vector3(START_X, 0.0, 0.0)
    body = body_object.add_component(inx.Rigidbody)
    if body is None:
        raise RuntimeError("Rigidbody could not be added to the probe object")
    body.use_gravity = False
    body.drag = 0.0
    body.angular_drag = 0.0
    if body_object.add_component(inx.BoxCollider) is None:
        raise RuntimeError("BoxCollider could not be added to the probe object")


def _enter_play(engine) -> None:
    from infernux.engine.deferred_task import DeferredTaskRunner
    from infernux.engine.play_mode import PlayModeManager
    from infernux.lib import SceneManager

    manager = SceneManager.instance()
    play = PlayModeManager.instance()
    if play is None or not play.enter_play_mode():
        raise RuntimeError("Play Mode entry was rejected")
    runner = DeferredTaskRunner.instance()
    for _ in range(240):
        runner.tick()
        engine.tick(DELTA)
        if manager.is_playing():
            return
    raise RuntimeError("Play Mode did not start within the admission budget")


def run(project: str) -> dict:
    import infernux as inx
    from infernux.lib import RuntimeMode, SceneManager

    engine = inx.Engine(inx.LogLevel.Warn, RuntimeMode.Headless)
    engine.init_headless(project)
    manager = SceneManager.instance()
    scene = manager.get_active_scene() or manager.create_scene("RigidProbe")
    _build_fixture(scene)

    _enter_play(engine)
    base_clock = inx.sim.settle(engine)
    runtime_object = inx.sim.body("Slider")
    runtime_body = runtime_object.get_component(inx.Rigidbody)
    if runtime_body is None:
        raise RuntimeError("the runtime clone lost its Rigidbody")
    runtime_body.velocity = inx.Vector3(VELOCITY_X, 0.0, 0.0)
    inx.sim.pause()
    base = inx.sim.clock()
    base_x = float(runtime_object.get_transform().position.x)
    base_sim_s = float(base["sim_s"])

    samples = [inx.sim.pose(runtime_object)]
    for k in range(1, SAMPLES):
        inx.sim.step(engine, expected_tick=base["tick"] + k - 1)
        samples.append(inx.sim.pose(runtime_object))
    repeated = inx.sim.pose(runtime_object)
    from infernux.engine.play_mode import PlayModeManager

    PlayModeManager.instance().exit_play_mode()
    engine.exit()

    rows = []
    failures = []
    for k, sample in enumerate(samples):
        expected_x = base_x + VELOCITY_X * DELTA * k
        expected_t = base_sim_s + DELTA * k
        dx = abs(sample["position_m"][0] - expected_x)
        dt = abs(sample["sim_s"] - expected_t)
        rows.append(
            {
                "k": k,
                "x": sample["position_m"][0],
                "expected_x": expected_x,
                "dx_m": dx,
                "sim_s": sample["sim_s"],
                "expected_sim_s": expected_t,
                "dt_s": dt,
                "tick": sample["tick"],
            }
        )
        if dx > POSITION_TOLERANCE:
            failures.append(f"k={k} position off by {dx:.9f} m")
        if dt > TIME_TOLERANCE:
            failures.append(f"k={k} sim time off by {dt:.9f} s")
        if sample["tick"] != base["tick"] + k:
            failures.append(f"k={k} tick {sample['tick']} != {base['tick'] + k}")
    if repeated != samples[-1]:
        failures.append("repeated read changed the sample")
    if abs(base_x - START_X) > POSITION_TOLERANCE:
        failures.append(f"runtime body started at {base_x}, expected {START_X}")

    return {
        "ok": not failures,
        "failures": failures,
        "delta": DELTA,
        "base": {"tick": base["tick"], "sim_s": base_sim_s, "x": base_x,
                 "settled_tick": base_clock["tick"]},
        "samples": rows,
        "final_x": samples[-1]["position_m"][0],
        "final_sim_s": samples[-1]["sim_s"],
        "final_tick": samples[-1]["tick"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = args.project or tempfile.mkdtemp(prefix="yanshi_rigid_")
    try:
        _prepare_project(root)
        report = run(root)
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    finally:
        if not args.project and not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
