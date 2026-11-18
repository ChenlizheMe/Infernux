"""Exercise the Automation Host handlers Yanshi binds, without a GUI.

The HTTP/transport layer is not started; this probe calls the same handlers
the host operations dispatch to:

* ``infernux.scene.object.resolve`` -> ``_resolve_object_now``
* ``infernux.sim.step``            -> ``_request_step`` (native paused step)
* ``infernux.sim.clock``           -> ``_read_clock``

It runs a headless Play world with the rigid_diagnostics fixture and asserts
that the editor-path step protocol publishes exactly one physics step per
request, matching the Python ``infernux.sim`` path.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback

DELTA = 0.01


def _prepare_project(root: str) -> None:
    os.makedirs(os.path.join(root, "Assets"), exist_ok=True)
    settings_dir = os.path.join(root, "ProjectSettings")
    os.makedirs(settings_dir, exist_ok=True)
    with open(os.path.join(root, ".infernux-version"), "w", encoding="utf-8") as handle:
        handle.write("0.4.8\n")
    from infernux.physics.settings import DEFAULT_PHYSICS_SETTINGS

    settings = dict(DEFAULT_PHYSICS_SETTINGS)
    settings.update({"gravity": [0.0, 0.0, 0.0], "fixed_delta_time": DELTA,
                     "max_fixed_delta_time": 1.0 / 3.0, "max_concurrency": 1})
    with open(os.path.join(settings_dir, "PhysicsSettings.json"), "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def run(project: str) -> dict:
    import infernux as inx
    from infernux.lib import RuntimeMode, SceneManager
    from infernux.engine.deferred_task import DeferredTaskRunner
    from infernux.engine.play_mode import PlayModeManager
    from infernux.host.scene_operations import _resolve_object_now
    from infernux.host.sim_operations import _read_clock, _request_step

    engine = inx.Engine(inx.LogLevel.Warn, RuntimeMode.Headless)
    engine.init_headless(project)
    manager = SceneManager.instance()
    scene = manager.get_active_scene() or manager.create_scene("HostProbe")
    obj = scene.create_game_object("Slider")
    obj.get_transform().position = inx.Vector3(1.2, 0.0, 0.0)
    body = obj.add_component(inx.Rigidbody)
    body.use_gravity = False
    body.drag = 0.0
    body.angular_drag = 0.0
    obj.add_component(inx.BoxCollider)

    play = PlayModeManager.instance()
    if play is None or not play.enter_play_mode():
        raise RuntimeError("Play Mode entry was rejected")
    runner = DeferredTaskRunner.instance()
    for _ in range(240):
        runner.tick()
        engine.tick(DELTA)
        if manager.is_playing():
            break
    inx.sim.settle(engine)
    resolved = _resolve_object_now("Slider", "")
    inx.sim.pause()
    base = _read_clock()
    runtime_object = inx.sim.body("Slider")
    runtime_body = runtime_object.get_component(inx.Rigidbody)
    runtime_body.velocity = inx.Vector3(0.5, 0.0, 0.0)
    base = _read_clock()

    failures = []
    if not resolved["found"] or resolved["object_id"] != int(runtime_object.id):
        failures.append(f"resolve returned {resolved}, runtime id {runtime_object.id}")
    if not resolved["runtime"]:
        failures.append("resolve did not report runtime mode")

    rows = []
    for k in range(1, 11):
        receipt = _request_step(DELTA, base["tick"] + k - 1)
        if receipt["target_tick"] != base["tick"] + k:
            failures.append(f"k={k} target_tick {receipt['target_tick']}")
        for _ in range(4):
            engine.tick(DELTA)
            clock = _read_clock()
            if clock["tick"] >= receipt["target_tick"] and not clock["stepping"]:
                break
        pose = inx.sim.pose(runtime_object)
        expected_x = 1.2 + 0.005 * k
        dx = abs(pose["position_m"][0] - expected_x)
        rows.append({"k": k, "x": pose["position_m"][0], "expected_x": expected_x,
                     "dx_m": dx, "tick": clock["tick"]})
        if dx > 1e-4:
            failures.append(f"k={k} position off by {dx:.9f}")
        if clock["tick"] != base["tick"] + k:
            failures.append(f"k={k} tick {clock['tick']} != {base['tick'] + k}")

    play.exit_play_mode()
    engine.exit()
    return {"ok": not failures, "failures": failures, "resolve": resolved,
            "base": base, "samples": rows}


def main() -> int:
    root = tempfile.mkdtemp(prefix="yanshi_host_")
    try:
        _prepare_project(root)
        report = run(root)
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
