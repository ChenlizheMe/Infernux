"""Verify the door-demo primitives against a live graphical editor.

The door gameplay script is Yanshi's app package; this probe proves the engine
surface that demo depends on: create/open/save/reload a scene asset, place
objects, enter Play, deliver a native E key through the synthetic input path,
pause and step the physics clock, and read a runtime object's pose.
"""

from __future__ import annotations

import json
import sys
import time

from infernux_automation import AutomationClient, AutomationError


def call(client, operation, arguments=None):
    return client.call(operation, arguments or {})


def wait_load(client, timeout=30.0):
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = call(client, "infernux.project.info")["active_scene"]
        status = (last.get("last_load") or {}).get("status")
        if status in {"loaded", "idle"} and not last.get("loading"):
            if status == "loaded":
                return last
        time.sleep(0.2)
    return last


def main(project: str) -> int:
    report = {"ok": False, "steps": []}
    client = AutomationClient(project, timeout=60.0)

    def step(name, value):
        report["steps"].append({"step": name, "value": value})

    try:
        step("info", client.info())
        kinds = call(client, "infernux.asset.create.kinds")["kinds"]
        step("kinds", [item["kind"] for item in kinds])
        scene_asset = call(client, "infernux.asset.create",
                           {"kind": "scene", "directory": "Assets", "name": "DoorScene"})
        step("scene_asset", scene_asset)
        guid = scene_asset["asset"]["guid"]

        opened = call(client, "infernux.scene.open", {"asset_guid": guid, "discard_changes": True})
        step("open", opened)
        step("loaded", wait_load(client))

        for name, kind in (("Door", "primitive.cube"), ("Player", "primitive.cube")):
            step("create_" + name, call(client, "infernux.scene.object.create", {"kind": kind, "name": name}))
        door = call(client, "infernux.scene.object.resolve", {"name": "Door"})
        step("door", door)

        saved = call(client, "infernux.scene.save")
        step("save", saved)
        step("dirty_after_save", call(client, "infernux.project.info")["active_scene"]["dirty"])

        reloaded = call(client, "infernux.scene.reload", {"discard_changes": True})
        step("reload", reloaded)
        step("loaded_after_reload", wait_load(client))
        step("door_after_reload", call(client, "infernux.scene.object.resolve", {"name": "Door"}))

        step("play", call(client, "infernux.runtime.play"))
        step("pause", call(client, "infernux.runtime.pause"))
        key_down = call(client, "infernux.input.key", {"key": "E", "pressed": True, "wait_for_delivery": True})
        step("key_down", key_down)
        key_up = call(client, "infernux.input.key", {"key": "E", "pressed": False, "wait_for_delivery": True})
        step("key_up", key_up)
        step("input_status", call(client, "infernux.input.status"))

        clock = call(client, "infernux.sim.clock")
        step("clock", clock)
        receipt = call(client, "infernux.sim.step", {"expected_tick": clock["tick"]})
        step("step", receipt)
        deadline = time.monotonic() + 10
        advanced = clock
        while time.monotonic() < deadline:
            advanced = call(client, "infernux.sim.clock")
            if advanced["tick"] >= receipt["target_tick"]:
                break
            time.sleep(0.05)
        step("advanced", advanced)
        runtime_id = call(client, "infernux.scene.object.resolve", {"name": "Door"})["object_id"]
        step("door_runtime", call(client, "infernux.sim.pose", {"object_id": runtime_id}))

        step("stop", call(client, "infernux.runtime.stop"))
        ok = (saved.get("saved") and bool(opened.get("scheduled"))
              and advanced["tick"] == receipt["target_tick"]
              and key_down.get("delivered") and key_up.get("delivered"))
        report["ok"] = bool(ok)
    except AutomationError as error:
        step("error", {"code": error.code, "message": str(error)})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
