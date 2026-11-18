"""End-to-end check of the Automation Host against a running graphical editor.

Uses the standalone SDK the same way Yanshi will: discover the engine through
the project descriptor, read the capability catalog, drive the world, step the
physics clock, capture a Game-view frame and read it back. The editor owns the
loop; this process only sends requests.
"""

from __future__ import annotations

import json
import os
import sys
import time

from infernux_automation import AutomationClient, AutomationError


def call(client, operation, arguments=None):
    return client.call(operation, arguments or {})


def runtime(client) -> dict:
    return call(client, "infernux.runtime.status").get("runtime", {})


def main(project: str) -> int:
    report: dict = {"ok": False, "steps": []}
    client = AutomationClient(project, timeout=30.0)

    def step(name, value):
        report["steps"].append({"step": name, "value": value})

    step("info", client.info())
    catalog = client.operations(limit=500)
    ids = [item["id"] for item in catalog["operations"]]
    step("operations", {"count": len(ids), "has_sim_step": "infernux.sim.step" in ids,
                        "has_resolve": "infernux.scene.object.resolve" in ids})
    step("project", call(client, "infernux.project.info"))

    kinds = call(client, "infernux.scene.object.kinds")["kinds"]
    cube_kind = next((item["kind"] for item in kinds if item.get("label") == "Cube"), "")
    created = call(client, "infernux.scene.object.create", {"kind": cube_kind, "name": "YanshiCube"})
    step("create", created)
    object_id = int(created["id"])
    step("resolve_authored", call(client, "infernux.scene.object.resolve", {"name": "YanshiCube"}))

    try:
        call(client, "infernux.runtime.stop")
    except AutomationError:
        pass
    step("play", call(client, "infernux.runtime.play"))
    step("paused", call(client, "infernux.runtime.pause"))
    runtime_object = call(client, "infernux.scene.object.resolve", {"name": "YanshiCube"})
    step("resolve_runtime", runtime_object)
    object_id = int(runtime_object["object_id"])
    clock = call(client, "infernux.sim.clock")
    step("clock", clock)

    tick = int(clock["tick"])
    try:
        receipt = call(client, "infernux.sim.step", {"expected_tick": tick})
        step("step", receipt)
        deadline = time.monotonic() + 15
        advanced = clock
        while time.monotonic() < deadline:
            advanced = call(client, "infernux.sim.clock")
            if advanced["tick"] >= receipt["target_tick"]:
                break
            time.sleep(0.05)
        step("advanced", advanced)
        step("pose", call(client, "infernux.sim.pose", {"object_id": object_id}))
    except AutomationError as error:
        step("step_error", {"code": error.code, "message": str(error)})

    capture = call(client, "infernux.capture.request", {"source": "game", "file_name": "yanshi_probe.png"})
    step("capture", capture)
    capture_id = int(capture["capture_id"])
    status_value = capture
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status_value = call(client, "infernux.capture.status", {"capture_id": capture_id})
        if status_value.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)
    step("capture_status", status_value)
    output = str(status_value.get("output_path", ""))
    if output and os.path.isfile(output):
        size = os.path.getsize(output)
        step("capture_file", {"path": output, "bytes": size})
        if size > 1000 and int(status_value.get("width", 0)) > 0:
            report["ok"] = True

    call(client, "infernux.runtime.stop")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
