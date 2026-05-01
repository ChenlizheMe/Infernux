"""Self-description and workflow documentation MCP tools."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from Infernux.mcp import capabilities, server
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
            "Generated or agent-authored files should follow the user's requested path or the existing project/module structure.",
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
            "Create project folders under the user-requested feature/module path, such as Assets/<FeatureName>/Scripts, Data, Materials, Scenes.",
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


INTENT_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "camera_frame_subject": {
        "match": ["camera", "frame", "view", "visible", "subject", "target", "照全", "相机", "主体", "看不全"],
        "summary": "Diagnose the main camera view and frame likely subject objects.",
        "tools": [
            "mcp.catalog.search",
            "scene.query.summary",
            "scene.query.subjects",
            "camera.find_main",
            "camera.visibility_report",
            "camera.frame_targets",
        ],
    },
    "find_scene_objects": {
        "match": ["find", "query", "object", "name", "component", "gameobject", "查找", "物体", "名字"],
        "summary": "Find GameObjects by name, path, component, tag, layer, or activity.",
        "tools": ["scene.query.objects", "gameobject.get", "gameobject.describe_spatial"],
    },
    "renderstack_postprocess": {
        "match": ["renderstack", "render stack", "postprocess", "bloom", "pipeline", "渲染", "后处理"],
        "summary": "Inspect and edit the scene RenderStack pipeline and effects.",
        "tools": [
            "renderstack.find_or_create",
            "renderstack.inspect",
            "renderstack.list_pipelines",
            "renderstack.list_passes",
            "renderstack.add_pass",
            "renderstack.set_pass_params",
        ],
    },
    "shader_authoring": {
        "match": ["shader", "glsl", "material", "fragment", "vertex", "shadingmodel", "着色器", "材质"],
        "summary": "Discover shader architecture, annotations, property declarations, and material binding rules.",
        "tools": ["shader.guide", "shader.catalog", "shader.describe", "api.get", "material.create", "asset.create_builtin_resource"],
    },
    "audio_authoring": {
        "match": ["audio", "sound", "music", "sfx", "listener", "audiosource", "audioclip", "音频", "声音"],
        "summary": "Understand AudioSource multi-track playback, AudioClip loading, and AudioListener placement.",
        "tools": ["audio.guide", "api.get", "component.describe_type", "gameobject.add_component"],
    },
}


def register_docs_tools(mcp, project_path: str, config: dict[str, Any] | None = None) -> None:
    config = config or capabilities.current_config()
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

    @mcp.tool(name="mcp.discovery")
    def mcp_discovery() -> dict:
        """Return connection info and project-local discovery file locations."""
        info = server.connection_info()
        return ok({
            **info,
            "project_root": project_path,
            "files": {
                name: client["file"]
                for name, client in info.get("clients", {}).items()
            },
            "config_snippets": {
                name: client.get("config") or client.get("toml")
                for name, client in info.get("clients", {}).items()
            },
        })

    @mcp.tool(name="mcp.capabilities")
    def mcp_capabilities() -> dict:
        """Return high-level capability groups and known tool names."""
        return ok({
            "agent_guidance": [
                "Infernux APIs are new and engine-specific. Do not guess unfamiliar Python, component, shader, audio, or UI APIs.",
                "Use api.search(query) and api.get(name) for Python/stub-backed APIs before writing scripts.",
                "Use component.describe_type(component_type) before mutating component fields.",
                "Use shader.guide, shader.catalog, and shader.describe before creating or binding shaders.",
                "Some write tools require a knowledge_token from the matching guide, e.g. shader.guide, audio.guide, or api.get('ui').",
                "Use mcp.catalog.search or mcp.catalog.recommend before selecting MCP tools.",
            ],
            "catalog": _catalog_tree(),
            "groups": {
                "foundation": ["mcp.ping", "mcp.version", "mcp.discovery", "mcp.health", "mcp.help", "mcp.catalog.list"],
                "api": ["api.subsystems", "api.search", "api.get", "shader.guide", "audio.guide"],
                "shader": ["shader.guide", "shader.catalog", "shader.describe"],
                "audio": ["audio.guide", "component.describe_type"],
                "self_description": ["engine.concepts", "engine.concept.get", "workflow.list", "workflow.help"],
                "scene": ["scene.status", "scene.inspect", "scene.get_hierarchy", "scene.query.summary", "scene.query.objects"],
                "scene_lifecycle": ["scene.save", "scene.open", "scene.new"],
                "gameobject": ["hierarchy.create_object", "gameobject.find", "gameobject.get", "gameobject.describe_spatial"],
                "component": ["component.list_types", "component.describe_type", "component.set_field"],
                "asset": ["asset.ensure_folder", "asset.list", "asset.search", "asset.read_text", "asset.write_text", "asset.refresh"],
                "camera": ["camera.find_main", "camera.describe_view", "camera.visibility_report", "camera.frame_targets"],
                "renderstack": ["renderstack.inspect", "renderstack.list_pipelines", "renderstack.add_pass", "renderstack.set_pass_params"],
                "runtime": ["editor.play", "runtime.wait", "runtime.run_for", "runtime.read_errors"],
                "project_tools": ["project_tools.list", "project_tools.reload", "project_tools.validate", "project_tools.audit"],
                "trace": ["mcp.trace.start", "mcp.trace.stop", "mcp.trace.current", "mcp.trace.list"],
                "session_log": ["mcp.session_log.info", "mcp.session_log.read", "mcp.session_log.clear"],
                "transactions": ["transaction.begin", "transaction.status", "transaction.commit", "transaction.rollback"],
                "research": ["mcp.config.get", "mcp.contracts.list", "mcp.contracts.validate", "mcp.evolution.suggest_tools"],
            },
            "tools": [meta["name"] for meta in list_tool_metadata() if capabilities.tool_enabled(meta["name"])],
            "config": capabilities.current_config(),
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
        return ok({"tools": _visible_metadata()})

    @mcp.tool(name="mcp.help")
    def mcp_help(tool_name: str = "") -> dict:
        """Return detailed help for one tool or all tool groups."""
        if tool_name:
            return ok({"tool": get_tool_metadata(tool_name)})
        return ok({"tools": _visible_metadata(), "workflows": list(WORKFLOWS), "concepts": list(CONCEPTS)})

    @mcp.tool(name="mcp.catalog.list")
    def mcp_catalog_list() -> dict:
        """Return the hierarchical MCP tool catalog."""
        return ok({
            "catalog": _catalog_tree(),
            "categories": _catalog_categories(),
            "recommend": "Use mcp.catalog.search(query) or mcp.catalog.recommend(intent) before choosing tools.",
        })

    @mcp.tool(name="mcp.catalog.get")
    def mcp_catalog_get(category: str = "") -> dict:
        """Return tools under a category such as camera/framing or scene/query."""
        needle = str(category or "").strip().lower()
        tools = []
        for meta in _visible_metadata():
            meta_category = str(meta.get("category", ""))
            lower = meta_category.lower()
            if not needle or lower == needle or lower.startswith(needle + "/"):
                tools.append(meta)
        return ok({"category": category, "tools": tools, "count": len(tools), "categories": _catalog_categories()})

    @mcp.tool(name="mcp.catalog.search")
    def mcp_catalog_search(query: str, category: str = "", limit: int = 20) -> dict:
        """Search tools by name, category, summary, tags, aliases, and concepts."""
        matches = _search_catalog(str(query or ""), category=str(category or ""), limit=int(limit or 20))
        return ok({"query": query, "category": category, "matches": matches})

    @mcp.tool(name="mcp.catalog.recommend")
    def mcp_catalog_recommend(intent: str, limit: int = 12) -> dict:
        """Recommend a tool chain for a natural-language intent."""
        lowered = str(intent or "").lower()
        recommendations = []
        for key, item in INTENT_RECOMMENDATIONS.items():
            score = sum(1 for token in item.get("match", []) if str(token).lower() in lowered)
            if score:
                recommendations.append({
                    "intent": key,
                    "score": score,
                    "summary": item["summary"],
                    "tools": [get_tool_metadata(tool) for tool in item["tools"] if capabilities.tool_enabled(tool)],
                })
        recommendations.sort(key=lambda item: item["score"], reverse=True)
        if not recommendations:
            recommendations.append({
                "intent": "catalog_search",
                "score": 0,
                "summary": "No intent template matched; use search results as candidates.",
                "tools": _search_catalog(str(intent or ""), limit=int(limit or 12)),
            })
        return ok({"intent": intent, "recommendations": recommendations[: max(int(limit or 12), 1)]})

    if capabilities.feature_enabled("batch_execution"):
        @mcp.tool(name="mcp.batch")
        def mcp_batch(steps: list[dict[str, Any]], continue_on_error: bool = False) -> dict:
            """Execute a list of MCP tool calls and return an operation trace.

            Each step is {"tool": "...", "arguments": {...}, "label": "..."}.
            This intentionally goes through the MCP transport so the batch path
            observes the same envelope and main-thread behavior as external agents.
            """
            max_steps = int(capabilities.limit("batch_max_steps", 100) or 100)
            trace = []
            client = _NestedMCPClient(server.endpoint_url())
            for index, step in enumerate((steps or [])[:max_steps]):
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
            truncated = len(steps or []) > max_steps
            return ok({"ok": all(item.get("ok") for item in trace), "trace": trace, "truncated": truncated}, trace=trace)

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


def _visible_metadata() -> list[dict[str, Any]]:
    return [meta for meta in list_tool_metadata() if capabilities.tool_enabled(meta["name"])]


def _catalog_categories() -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for meta in _visible_metadata():
        category = str(meta.get("category", "") or "misc/other")
        parts = category.split("/")
        for idx in range(1, len(parts) + 1):
            key = "/".join(parts[:idx])
            counts[key] = counts.get(key, 0) + (1 if idx == len(parts) else 0)
    return [{"category": key, "tool_count": counts[key]} for key in sorted(counts)]


def _catalog_tree() -> dict[str, Any]:
    root: dict[str, Any] = {}
    for meta in _visible_metadata():
        category = str(meta.get("category", "") or "misc/other")
        node = root
        for part in [p for p in category.split("/") if p]:
            node = node.setdefault(part, {"tools": {}, "children": {}})["children"]
        leaf = root
        for part in [p for p in category.split("/") if p]:
            leaf = leaf[part]["children"]
        target = root
        parts = [p for p in category.split("/") if p]
        for part in parts[:-1]:
            target = target[part]["children"]
        if parts:
            target[parts[-1]].setdefault("tools", {})[meta["name"]] = {
                "summary": meta.get("summary", ""),
                "level": meta.get("level", "semantic"),
                "tags": meta.get("tags", []),
                "signature": meta.get("signature", ""),
                "parameters": meta.get("parameters", {}),
                "required_parameters": meta.get("required_parameters", []),
                "returns": meta.get("returns", {}),
            }
    return root


def _search_catalog(query: str, *, category: str = "", limit: int = 20) -> list[dict[str, Any]]:
    tokens = [token for token in str(query or "").lower().replace("/", " ").replace(".", " ").split() if token]
    category = str(category or "").lower().strip()
    scored = []
    for meta in _visible_metadata():
        meta_category = str(meta.get("category", ""))
        if category and not meta_category.lower().startswith(category):
            continue
        haystack_parts = [
            meta.get("name", ""),
            meta_category,
            meta.get("summary", ""),
            meta.get("doc", ""),
            meta.get("signature", ""),
            " ".join(str(key) for key in (meta.get("parameters", {}) or {}).keys()),
            " ".join(str(value.get("annotation", "")) for value in (meta.get("parameters", {}) or {}).values() if isinstance(value, dict)),
            " ".join(str(example.get("description", "")) for example in meta.get("examples", []) if isinstance(example, dict)),
            " ".join(str(item) for item in meta.get("tags", [])),
            " ".join(str(item) for item in meta.get("aliases", [])),
            " ".join(str(item) for item in (meta.get("concepts", {}) or {}).keys()),
        ]
        haystack = " ".join(str(part).lower() for part in haystack_parts)
        if not tokens:
            score = 1
        else:
            score = sum(3 if token in str(meta.get("name", "")).lower() else 1 for token in tokens if token in haystack)
        if score:
            scored.append((score, meta))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))
    return [{**meta, "score": score} for score, meta in scored[: max(int(limit or 20), 1)]]


def _self_description_category(name: str) -> str:
    if name.startswith("mcp.catalog."):
        return "foundation/catalog"
    if name.startswith("api."):
        return "foundation/api"
    if name.startswith("shader."):
        return "shader/guide"
    if name.startswith("audio."):
        return "audio/guide"
    return "foundation/discovery"


def _self_description_tags(name: str) -> list[str]:
    if name.startswith("mcp.catalog."):
        return ["catalog", "discover", "tools"]
    if name.startswith("api."):
        return ["api", "script", "docs", "subsystem"]
    if name.startswith("shader."):
        return ["shader", "glsl", "material", "docs"]
    if name.startswith("audio."):
        return ["audio", "sound", "script", "docs"]
    return ["self-description", "help"]


def _register_metadata() -> None:
    for name, summary in {
        "project.info": "Return current project, active scene, play state, and selection.",
        "editor.get_state": "Return lightweight editor state.",
        "editor.play": "Enter Play Mode only after the active scene is saved, clean, and not loading.",
        "editor.stop": "Exit Play Mode; idempotent when already in edit mode.",
        "editor.step": "Step one frame only while Play Mode is paused.",
        "scene.status": "Return active scene path, dirty flag, and suggested save path.",
        "scene.inspect": "Return compact active scene summary.",
        "scene.get_hierarchy": "Return active scene hierarchy.",
        "scene.save": "Save the active scene through SceneFileManager; use this instead of asset tools for .scene files.",
        "scene.open": "Open a scene file only when not playing, not dirty, and not already loading.",
        "scene.new": "Create a new empty scene only with force=true and a reason.",
        "hierarchy.create_object": "Create a GameObject using registered HierarchyCreationService kinds.",
        "gameobject.add_component": "Attach a built-in or Python script component to a GameObject.",
        "component.describe_type": "Describe fields and metadata for a component type.",
        "asset.ensure_folder": "Ensure a project folder exists; succeeds when the folder already exists.",
        "asset.write_text": "Write a UTF-8 project file and notify AssetDatabase.",
        "console.read": "Read recent editor console entries.",
    }.items():
        register_tool_metadata(name, summary=summary)
    for name in ["scene.save", "scene.open", "scene.new", "editor.play", "editor.step"]:
        register_tool_metadata(
            name,
            summary=get_tool_metadata(name).get("summary", ""),
            recovery=[
                "Call scene.status before changing scenes or entering Play Mode.",
                "If scene.loading is true, wait and retry later.",
                "If scene.dirty is true, call scene.save before scene.open/scene.new/editor.play.",
                "Do not use asset.write_text/write_json/patch_text for .scene files.",
            ],
            next_suggested_tools=["scene.status", "runtime.wait", "mcp.health"],
            concepts={"Active Scene": "Only the currently open scene may be edited through MCP scene mutation tools."},
            side_effects=["May change editor scene state or Play Mode state."],
        )
    for name in [
        "mcp.ping", "mcp.version", "mcp.discovery", "mcp.capabilities", "mcp.health", "mcp.help", "mcp.batch",
        "mcp.catalog.list", "mcp.catalog.get", "mcp.catalog.search", "mcp.catalog.recommend",
        "api.subsystems", "api.get", "api.search", "shader.guide", "shader.catalog", "shader.describe", "audio.guide",
        "engine.concepts", "engine.concept.get", "workflow.list", "workflow.help",
        "workflow.examples",
    ]:
        register_tool_metadata(
            name,
            summary=f"Self-description tool: {name}.",
            category=_self_description_category(name),
            tags=_self_description_tags(name),
            aliases=["tool menu", "categories", "search tools", "script api", "shader api", "audio api"] if name.startswith(("mcp.catalog.", "api.", "shader.", "audio.")) else [],
            level="foundation",
        )
    for name, summary in {
        "scene.query.objects": "Search scene objects with semantic filters.",
        "scene.query.summary": "Return grouped scene semantics for cameras, lights, renderers, UI, scripts, and subjects.",
        "scene.query.subjects": "Rank likely primary scene subjects.",
        "gameobject.describe_spatial": "Describe object transform, hierarchy path, components, and approximate bounds.",
    }.items():
        register_tool_metadata(
            name,
            summary=summary,
            category="scene/query",
            tags=["scene", "query", "gameobject", "subject", "bounds"],
            aliases=["find object", "scene semantics", "主体", "查找物体", "空间信息"],
            recovery=[
                "If data.stop_repeating is true, do not call the same query again.",
                "For fuzzy object lookup, prefer query.name_contains or query.path_contains.",
                "For exact matching, use query.name_exact or query.path_exact.",
                "If subjects is empty in a bootstrap scene, create or locate a renderable/gameplay object before camera framing.",
            ],
            next_suggested_tools=["camera.visibility_report", "gameobject.get", "component.get"],
        )


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
