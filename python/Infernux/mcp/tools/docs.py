"""Self-description and workflow documentation MCP tools."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from Infernux.mcp import server
from Infernux.mcp.threading import MainThreadCommandQueue
from Infernux.mcp.tools.common import (
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_VERSION,
    get_asset_database,
    get_tool_metadata,
    list_tool_metadata,
    main_thread,
    ok,
    register_tool_metadata,
)


CONCEPTS: dict[str, dict[str, Any]] = {
    "Scene": {
        "summary": "A loaded world containing root GameObjects and their children.",
        "notes": [
            "Scene object IDs are editor-session IDs; reacquire them after reload.",
            "Use scene.inspect for a compact map and scene.get_hierarchy for nested structure.",
        ],
        "tools": ["scene.inspect", "scene.get_hierarchy", "scene.save", "scene.open", "scene.new"],
    },
    "GameObject": {
        "summary": "A scene entity with a Transform and zero or more Components.",
        "notes": [
            "Every GameObject has a Transform.",
            "Use stable names and hierarchy paths for generated content so agents can find it later.",
        ],
        "tools": ["hierarchy.create_object", "gameobject.find", "gameobject.get", "gameobject.set_parent"],
    },
    "Transform": {
        "summary": "Position, rotation, scale, and hierarchy relationship for a GameObject.",
        "notes": [
            "World fields: position/euler_angles.",
            "Local fields: local_position/local_euler_angles/local_scale.",
        ],
        "tools": ["transform.set", "gameobject.set_parent", "camera.attach_to_target"],
    },
    "Component": {
        "summary": "Behavior or data attached to a GameObject.",
        "notes": [
            "Built-in components are backed by C++ wrappers.",
            "Python components inherit InxComponent and can expose serialized_field metadata.",
        ],
        "tools": ["component.list_types", "component.describe_type", "gameobject.add_component"],
    },
    "Python Component": {
        "summary": "User-authored gameplay script inheriting from Infernux.components.InxComponent.",
        "notes": [
            "Lifecycle methods include awake, start, update, late_update, on_enable, on_disable, on_destroy.",
            "Use asset.write_text to create scripts, asset.refresh to import them, and gameobject.add_component with script_path to attach.",
        ],
        "tools": ["asset.write_text", "asset.refresh", "gameobject.add_component", "runtime.read_errors"],
    },
    "Builtin Component": {
        "summary": "A C++ engine component exposed through Python wrapper metadata.",
        "notes": [
            "Examples: Camera, Light, MeshRenderer, Rigidbody, BoxCollider.",
            "Use component.describe_type before setting fields.",
        ],
        "tools": ["component.list_types", "component.describe_type", "component.set_field"],
    },
    "AssetDatabase": {
        "summary": "Tracks Assets/ files and maps project paths to GUIDs.",
        "notes": [
            "Generated files should stay under Assets/Generated/<GameName>/ when possible.",
            "Call asset.refresh after external writes if the editor does not auto-import immediately.",
        ],
        "tools": ["asset.list", "asset.search", "asset.resolve", "asset.refresh"],
    },
    "PlayMode": {
        "summary": "Runtime simulation mode with editor-scene isolation.",
        "notes": [
            "Use editor.play, runtime.wait, runtime.read_errors, then editor.stop for validation.",
            "Script load errors can block entering Play Mode.",
        ],
        "tools": ["editor.play", "runtime.wait", "runtime.run_for", "runtime.read_errors", "editor.stop"],
    },
    "Camera": {
        "summary": "Rendering view for Game View. Agents should reuse Main Camera when present.",
        "notes": [
            "Use camera.ensure_main instead of blindly creating another camera.",
            "Card games usually want an orthographic camera; third-person games usually attach camera to a target.",
        ],
        "tools": ["camera.find_main", "camera.ensure_main", "camera.setup_2d_card_game", "camera.setup_third_person"],
    },
    "UI Canvas": {
        "summary": "Root object for UI elements such as text and buttons.",
        "notes": [
            "UI tools are planned as semantic wrappers; low-level creation can use hierarchy.create_object with ui.* kinds.",
        ],
        "tools": ["hierarchy.create_object", "workflow.help"],
    },
    "Input": {
        "summary": "Runtime input facade used by Python components.",
        "notes": [
            "Use Infernux.input.Input.get_axis('Horizontal') and get_axis('Vertical') for WASD movement.",
            "Input is Game View focus-gated.",
        ],
        "tools": ["asset.write_text", "runtime.run_for"],
    },
    "Undo": {
        "summary": "Editor undo/dirty tracking layer for hierarchy and inspector changes.",
        "notes": [
            "MCP mutation tools should mark scenes/assets dirty or call editor undo helpers when available.",
        ],
        "tools": ["gameobject.set", "component.set_field", "scene.save"],
    },
}


WORKFLOWS: dict[str, dict[str, Any]] = {
    "physics_test_scene": {
        "summary": "Create a ground plane, dynamic cube, collider, rigidbody, light, and camera.",
        "steps": [
            "Call camera.ensure_main or camera.setup_third_person rather than creating duplicate cameras.",
            "Create primitive.plane and add BoxCollider.",
            "Create primitive.cube, add BoxCollider and Rigidbody.",
            "Run editor.play, runtime.wait, runtime.read_errors.",
        ],
        "tools": ["hierarchy.create_object", "gameobject.add_component", "transform.set", "editor.play"],
    },
    "third_person_controller": {
        "summary": "Create a controllable character with a following camera.",
        "steps": [
            "Write an InxComponent script with Input.get_axis and Rigidbody.velocity.",
            "Create or reuse a player GameObject.",
            "Use camera.ensure_main, then camera.attach_to_target or camera.setup_third_person.",
            "Enter Play Mode and validate with runtime.get_object_state.",
        ],
        "tools": ["asset.write_text", "gameobject.add_component", "camera.setup_third_person", "runtime.read_errors"],
    },
    "card_game_shell": {
        "summary": "Create folders, scripts, JSON data, orthographic camera, and UI roots for a Balatro-style prototype.",
        "steps": [
            "Create Assets/Generated/<GameName>/Scripts, Data, Materials, Scenes.",
            "Write data model and controller scripts.",
            "Use camera.setup_2d_card_game.",
            "Create UI Canvas hierarchy for menu, run screen, hand, shop, and game over.",
            "Run Play Mode and validate console/runtime state.",
        ],
        "tools": ["asset.write_text", "asset.write_json", "camera.setup_2d_card_game", "runtime.read_errors"],
    },
    "script_attach_play_validate": {
        "summary": "Write a gameplay script, attach it, play, read errors, stop.",
        "steps": [
            "asset.write_text(path='Assets/.../MyComponent.py')",
            "asset.refresh",
            "gameobject.add_component(component_type='MyComponent', script_path='Assets/.../MyComponent.py')",
            "editor.play",
            "runtime.wait(play_state='playing')",
            "runtime.read_errors",
        ],
        "tools": ["asset.write_text", "asset.refresh", "gameobject.add_component", "editor.play", "runtime.read_errors"],
    },
}


def register_docs_tools(mcp, project_path: str) -> None:
    _register_metadata()

    @mcp.tool(name="mcp.ping")
    def mcp_ping() -> dict:
        """Return a lightweight MCP liveness response."""
        return ok({"pong": True, "endpoint": server.endpoint_url()})

    @mcp.tool(name="mcp.version")
    def mcp_version() -> dict:
        """Return MCP server version and protocol information."""
        return ok({
            "server": "Infernux Editor",
            "mcp_server_version": MCP_SERVER_VERSION,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "endpoint": server.endpoint_url(),
        })

    @mcp.tool(name="mcp.capabilities")
    def mcp_capabilities() -> dict:
        """Return high-level capability groups and known tool names."""
        return ok({
            "groups": {
                "foundation": ["mcp.ping", "mcp.version", "mcp.health", "mcp.help"],
                "self_description": ["engine.concepts", "engine.concept.get", "workflow.list", "workflow.help"],
                "scene": ["scene.inspect", "scene.get_hierarchy", "scene.save", "scene.open", "scene.new"],
                "gameobject": ["hierarchy.create_object", "gameobject.find", "gameobject.get", "gameobject.set_parent"],
                "component": ["component.list_types", "component.describe_type", "component.set_field"],
                "asset": ["asset.list", "asset.search", "asset.read_text", "asset.write_text", "asset.refresh"],
                "camera": ["camera.find_main", "camera.ensure_main", "camera.setup_2d_card_game"],
                "runtime": ["editor.play", "runtime.wait", "runtime.run_for", "runtime.read_errors"],
            },
            "tools": [meta["name"] for meta in list_tool_metadata()],
        })

    @mcp.tool(name="mcp.health")
    def mcp_health() -> dict:
        """Report editor, scene, asset database, and queue readiness."""

        def _health():
            from Infernux.engine.deferred_task import DeferredTaskRunner
            from Infernux.engine.play_mode import PlayModeManager
            from Infernux.engine.scene_manager import SceneFileManager
            from Infernux.lib import SceneManager

            scene = SceneManager.instance().get_active_scene()
            sfm = SceneFileManager.instance()
            pmm = PlayModeManager.instance()
            runner = DeferredTaskRunner.instance()
            adb = get_asset_database()
            return {
                "server_running": server.is_running(),
                "endpoint": server.endpoint_url(),
                "main_thread_queue_ready": MainThreadCommandQueue.instance().wait_until_ready(0.01),
                "active_scene": {
                    "available": scene is not None,
                    "name": getattr(scene, "name", ""),
                    "path": getattr(sfm, "current_scene_path", "") if sfm else "",
                    "dirty": bool(getattr(sfm, "is_dirty", False)) if sfm else False,
                    "loading": bool(getattr(sfm, "is_loading", False)) if sfm else False,
                },
                "play_state": getattr(getattr(pmm, "state", None), "name", "edit").lower() if pmm else "edit",
                "deferred_task_busy": bool(getattr(runner, "is_busy", False)),
                "asset_database_ready": adb is not None,
                "project_root": project_path,
            }

        return main_thread("mcp.health", _health)

    @mcp.tool(name="mcp.list_tools_verbose")
    def mcp_list_tools_verbose() -> dict:
        """Return registered tool metadata."""
        return ok({"tools": list_tool_metadata()})

    @mcp.tool(name="mcp.help")
    def mcp_help(tool_name: str = "") -> dict:
        """Return detailed help for one tool or all tool groups."""
        if tool_name:
            return ok({"tool": get_tool_metadata(tool_name)})
        return ok({"tools": list_tool_metadata(), "workflows": list(WORKFLOWS), "concepts": list(CONCEPTS)})

    @mcp.tool(name="mcp.batch")
    def mcp_batch(steps: list[dict[str, Any]], continue_on_error: bool = False) -> dict:
        """Execute a list of MCP tool calls and return an operation trace.

        Each step is {"tool": "...", "arguments": {...}, "label": "..."}.
        This intentionally goes through the MCP transport so the batch path
        observes the same envelope and main-thread behavior as external agents.
        """
        trace = []
        client = _NestedMCPClient(server.endpoint_url())
        for index, step in enumerate(steps or []):
            tool_name = str(step.get("tool", ""))
            arguments = step.get("arguments", {}) or {}
            label = str(step.get("label", "")) or tool_name
            try:
                result = client.call(tool_name, arguments)
                ok_flag = bool(result.get("ok", True)) if isinstance(result, dict) else True
                trace.append({
                    "index": index,
                    "label": label,
                    "tool": tool_name,
                    "ok": ok_flag,
                    "result": result,
                })
                if not ok_flag and not continue_on_error:
                    break
            except Exception as exc:
                trace.append({
                    "index": index,
                    "label": label,
                    "tool": tool_name,
                    "ok": False,
                    "error": {"code": "error.batch_step", "message": str(exc)},
                })
                if not continue_on_error:
                    break
        return ok({"ok": all(item.get("ok") for item in trace), "trace": trace}, trace=trace)

    @mcp.tool(name="engine.concepts")
    def engine_concepts() -> dict:
        """List Infernux concepts exposed to agents."""
        return ok({"concepts": [{"name": key, "summary": value["summary"]} for key, value in sorted(CONCEPTS.items())]})

    @mcp.tool(name="engine.concept.get")
    def engine_concept_get(name: str) -> dict:
        """Return a single concept page."""
        key = _lookup_key(CONCEPTS, name)
        if not key:
            return ok({"found": False, "available": sorted(CONCEPTS)})
        return ok({"name": key, **CONCEPTS[key]})

    @mcp.tool(name="workflow.list")
    def workflow_list() -> dict:
        """List documented workflows."""
        return ok({"workflows": [{"name": key, "summary": value["summary"]} for key, value in sorted(WORKFLOWS.items())]})

    @mcp.tool(name="workflow.help")
    def workflow_help(name: str) -> dict:
        """Return detailed workflow guidance."""
        key = _lookup_key(WORKFLOWS, name)
        if not key:
            return ok({"found": False, "available": sorted(WORKFLOWS)})
        return ok({"name": key, **WORKFLOWS[key]})

    @mcp.tool(name="workflow.examples")
    def workflow_examples(name: str = "") -> dict:
        """Return compact workflow examples."""
        if name:
            key = _lookup_key(WORKFLOWS, name)
            if not key:
                return ok({"found": False, "available": sorted(WORKFLOWS)})
            return ok({"examples": [{key: WORKFLOWS[key]}]})
        return ok({"examples": WORKFLOWS})

def _lookup_key(mapping: dict[str, Any], name: str) -> str:
    lowered = str(name).strip().lower()
    for key in mapping:
        if key.lower() == lowered:
            return key
    return ""


def _register_metadata() -> None:
    for name, summary in {
        "project.info": "Return current project, active scene, play state, and selection.",
        "editor.get_state": "Return lightweight editor state.",
        "scene.inspect": "Return compact active scene summary.",
        "scene.get_hierarchy": "Return active scene hierarchy.",
        "hierarchy.create_object": "Create a GameObject using registered HierarchyCreationService kinds.",
        "gameobject.add_component": "Attach a built-in or Python script component to a GameObject.",
        "component.describe_type": "Describe fields and metadata for a component type.",
        "asset.write_text": "Write a UTF-8 project file and notify AssetDatabase.",
        "console.read": "Read recent editor console entries.",
    }.items():
        register_tool_metadata(name, summary=summary)
    for name in [
        "mcp.ping", "mcp.version", "mcp.capabilities", "mcp.health", "mcp.help", "mcp.batch",
        "engine.concepts", "engine.concept.get", "workflow.list", "workflow.help",
        "workflow.examples",
    ]:
        register_tool_metadata(name, summary=f"Self-description tool: {name}.")


class _NestedMCPClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.request_id = 1
        self.session_id = ""
        self._initialize()

    def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        data = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        if "error" in data:
            raise RuntimeError(data["error"])
        result = data.get("result", {})
        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if structured is not None:
                return structured
            content = result.get("content") or []
            if content and isinstance(content[0].get("text"), str):
                try:
                    return json.loads(content[0]["text"])
                except Exception:
                    return {"text": content[0]["text"]}
        return result

    def _initialize(self) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "infernux-nested-batch", "version": MCP_SERVER_VERSION},
            },
        }
        data, headers = self._post_raw(payload, include_headers=True)
        self.session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id") or ""
        if not self.session_id:
            raise RuntimeError(f"Nested MCP initialize did not return a session id: {data}")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data, _headers = self._post_raw(payload, include_headers=True)
        return data

    def _post_raw(self, payload: dict[str, Any], *, include_headers: bool = False):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
            parsed = self._parse_body(body)
            if include_headers:
                return parsed, dict(response.headers)
            return parsed

    def _parse_body(self, body: str) -> dict[str, Any]:
        for line in body.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        return json.loads(body) if body.strip() else {}

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id
