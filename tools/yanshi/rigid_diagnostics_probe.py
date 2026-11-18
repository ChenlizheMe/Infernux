"""Yanshi rigid_diagnostics probe on Infernux.

Builds the least native world the task card allows (no gravity, no colliders,
no damping), gives one body x=1.2 m and +X velocity 0.5 m/s, then advances
exactly one physics step at a time and samples the pose without ever letting a
read advance the world.

Expected samples: x(k) = 1.2 + 0.005 k, k = 0..10; t(10) = 0.10 s.
Tolerance: position 0.1 mm, simulation time 1 us.

Usage:
    python tools/yanshi/rigid_diagnostics_probe.py [--project DIR]
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
    with open(os.path.join(settings_dir, "PhysicsSettings.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "gravity": [0.0, 0.0, 0.0],
                "fixed_delta_time": DELTA,
                "max_fixed_delta_time": 1.0 / 3.0,
            },
            handle,
            indent=2,
        )


def run(project: str) -> dict:
    import infernux as inx
    from infernux.lib import RuntimeMode, SceneManager

    engine = inx.Engine(inx.LogLevel.Warn, RuntimeMode.Headless)
    engine.init_headless(project)
    manager = SceneManager.instance()
    scene = manager.get_active_scene() or manager.create_scene("RigidProbe")
    body_object = scene.create_game_object("Slider")
    body_object.get_transform().position = inx.Vector3(START_X, 0.0, 0.0)
    body = body_object.add_component(inx.Rigidbody)
    if body is None:
        raise RuntimeError("Rigidbody could not be added to the probe object")
    body.use_gravity = False
    body.drag = 0.0
    body.angular_drag = 0.0
    body.velocity = inx.Vector3(VELOCITY_X, 0.0, 0.0)

    manager.play()
    samples = []
    try:
        first = inx.sim.pose(body_object)
        samples.append(first)
        for k in range(1, SAMPLES):
            inx.sim.step(engine, expected_tick=k - 1)
            samples.append(inx.sim.pose(body_object))
        repeated = inx.sim.pose(body_object)
    finally:
        manager.stop()
        engine.exit()

    rows = []
    failures = []
    for k, sample in enumerate(samples):
        expected_x = START_X + VELOCITY_X * DELTA * k
        expected_t = DELTA * k
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
        if sample["tick"] != k:
            failures.append(f"k={k} tick {sample['tick']}")
    if repeated != samples[-1]:
        failures.append("repeated read changed the sample")

    return {
        "ok": not failures,
        "failures": failures,
        "delta": DELTA,
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
