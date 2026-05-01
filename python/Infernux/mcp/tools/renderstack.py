"""RenderStack MCP tools for pipeline and post-processing control."""

from __future__ import annotations

from typing import Any

from Infernux.mcp.tools.common import main_thread, register_tool_metadata, serialize_value


def register_renderstack_tools(mcp) -> None:
    _register_metadata()

    @mcp.tool(name="renderstack.find_or_create")
    def renderstack_find_or_create(name: str = "RenderStack", create_if_missing: bool = True) -> dict:
        """Find the active RenderStack or create one."""

        def _find_or_create():
            stack = _find_stack()
            created = False
            if stack is None and create_if_missing:
                from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
                entry = HierarchyCreationService.instance().create("rendering.render_stack", name=name, select=False)
                obj = _find_game_object(int(entry["id"]))
                stack = _find_stack_on_object(obj)
                created = True
            if stack is None:
                raise FileNotFoundError("No RenderStack found and create_if_missing is false.")
            return {"created": created, "stack": _stack_snapshot(stack)}

        return main_thread("renderstack.find_or_create", _find_or_create, arguments={"name": name, "create_if_missing": create_if_missing})

    @mcp.tool(name="renderstack.inspect")
    def renderstack_inspect() -> dict:
        """Inspect the active RenderStack, pipeline, injection points, and mounted passes."""

        def _inspect():
            stack = _find_stack()
            if stack is None:
                return {
                    "exists": False,
                    "reason": "no_renderstack_in_active_scene",
                    "message": "No RenderStack exists in the active scene. Do not repeat renderstack.inspect; call renderstack.find_or_create if rendering stack control is needed.",
                    "next_suggested_tools": [
                        {
                            "tool": "renderstack.find_or_create",
                            "arguments": {"name": "RenderStack", "create_if_missing": True},
                        },
                        {
                            "tool": "hierarchy.create_object",
                            "arguments": {"kind": "rendering.render_stack", "name": "RenderStack"},
                        },
                    ],
                }
            return _stack_snapshot(stack, include_catalog=True)

        return main_thread("renderstack.inspect", _inspect)

    @mcp.tool(name="renderstack.list_pipelines")
    def renderstack_list_pipelines() -> dict:
        """List discovered RenderPipeline classes."""

        def _list():
            from Infernux.renderstack.discovery import discover_pipelines
            return {"pipelines": [_pipeline_entry(name, cls) for name, cls in sorted(discover_pipelines().items())]}

        return main_thread("renderstack.list_pipelines", _list)

    @mcp.tool(name="renderstack.set_pipeline")
    def renderstack_set_pipeline(pipeline: str = "") -> dict:
        """Set the active RenderStack pipeline. Empty string means default forward pipeline."""

        def _set():
            stack = _require_stack()
            stack.set_pipeline(str(pipeline or ""))
            _mark_scene_dirty()
            return _stack_snapshot(stack)

        return main_thread("renderstack.set_pipeline", _set, arguments={"pipeline": pipeline})

    @mcp.tool(name="renderstack.list_passes")
    def renderstack_list_passes(include_mounted: bool = True, include_available: bool = True) -> dict:
        """List mounted and/or available RenderPass effects."""

        def _list():
            stack = _find_stack()
            result: dict[str, Any] = {}
            if include_mounted and stack is not None:
                result["mounted"] = _mounted_passes(stack)
                result["injection_points"] = _injection_points(stack)
            if include_available:
                result["available"] = _available_passes()
            return result

        return main_thread("renderstack.list_passes", _list)

    @mcp.tool(name="renderstack.add_pass")
    def renderstack_add_pass(pass_name: str, enabled: bool = True, params: dict[str, Any] | None = None) -> dict:
        """Add a discovered RenderPass or FullScreenEffect to the active RenderStack."""

        def _add():
            stack = _require_stack()
            cls = _pass_class(pass_name)
            render_pass = cls(enabled=bool(enabled))
            if params:
                _set_pass_params(render_pass, params)
            added = bool(stack.add_pass(render_pass))
            if not added:
                raise ValueError(f"Render pass '{pass_name}' could not be added. It may already be mounted or target an invalid injection point.")
            _mark_scene_dirty()
            return _stack_snapshot(stack)

        return main_thread("renderstack.add_pass", _add, arguments={"pass_name": pass_name, "enabled": enabled, "params": params or {}})

    @mcp.tool(name="renderstack.remove_pass")
    def renderstack_remove_pass(pass_name: str) -> dict:
        """Remove a mounted pass from the active RenderStack."""

        def _remove():
            stack = _require_stack()
            removed = bool(stack.remove_pass(str(pass_name)))
            if not removed:
                raise FileNotFoundError(f"Mounted render pass '{pass_name}' was not found.")
            _mark_scene_dirty()
            return _stack_snapshot(stack)

        return main_thread("renderstack.remove_pass", _remove, arguments={"pass_name": pass_name})

    @mcp.tool(name="renderstack.set_pass_enabled")
    def renderstack_set_pass_enabled(pass_name: str, enabled: bool) -> dict:
        """Enable or disable a mounted pass."""

        def _set_enabled():
            stack = _require_stack()
            _require_mounted_pass(stack, pass_name)
            stack.set_pass_enabled(str(pass_name), bool(enabled))
            _mark_scene_dirty()
            return _stack_snapshot(stack)

        return main_thread("renderstack.set_pass_enabled", _set_enabled, arguments={"pass_name": pass_name, "enabled": enabled})

    @mcp.tool(name="renderstack.set_pass_params")
    def renderstack_set_pass_params(pass_name: str, params: dict[str, Any]) -> dict:
        """Set serialized parameters on a mounted pass/effect."""

        def _set_params():
            stack = _require_stack()
            entry = _require_mounted_pass(stack, pass_name)
            _set_pass_params(entry.render_pass, params or {})
            stack.invalidate_graph()
            _mark_scene_dirty()
            return _stack_snapshot(stack)

        return main_thread("renderstack.set_pass_params", _set_params, arguments={"pass_name": pass_name, "params": params or {}})


def _find_stack():
    try:
        from Infernux.renderstack import RenderStack
        stack = RenderStack.instance()
        if stack is not None:
            return stack
    except Exception:
        pass
    try:
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return None
        for obj in scene.get_all_objects() or []:
            stack = _find_stack_on_object(obj)
            if stack is not None:
                return stack
    except Exception:
        return None
    return None


def _require_stack():
    stack = _find_stack()
    if stack is None:
        raise FileNotFoundError("No RenderStack exists in the active scene. Use renderstack.find_or_create.")
    return stack


def _find_stack_on_object(obj):
    try:
        from Infernux.renderstack import RenderStack
        for comp in obj.get_py_components() or []:
            if isinstance(comp, RenderStack):
                return comp
    except Exception:
        return None
    return None


def _find_game_object(object_id: int):
    from Infernux.mcp.tools.common import find_game_object
    return find_game_object(object_id)


def _stack_snapshot(stack, *, include_catalog: bool = False) -> dict[str, Any]:
    go = stack.game_object
    data = {
        "object_id": int(go.id),
        "object_name": str(go.name),
        "pipeline": str(getattr(stack, "pipeline_class_name", "") or ""),
        "active": bool(stack is type(stack).instance()),
        "enabled": bool(getattr(stack, "enabled", True)),
        "injection_points": _injection_points(stack),
        "mounted_passes": _mounted_passes(stack),
        "build_failed": bool(getattr(stack, "_build_failed", False)),
    }
    if include_catalog:
        data["available_pipelines"] = _available_pipelines()
        data["available_passes"] = _available_passes()
    return data


def _injection_points(stack) -> list[dict[str, Any]]:
    points = []
    try:
        for point in stack.injection_points:
            points.append({
                "name": str(getattr(point, "name", "")),
                "description": str(getattr(point, "description", "")),
            })
    except Exception:
        pass
    return points


def _mounted_passes(stack) -> list[dict[str, Any]]:
    items = []
    for entry in list(getattr(stack, "pass_entries", []) or []):
        render_pass = entry.render_pass
        items.append({
            "name": str(getattr(render_pass, "name", "")),
            "class_name": type(render_pass).__name__,
            "injection_point": str(getattr(render_pass, "injection_point", "")),
            "enabled": bool(getattr(entry, "enabled", getattr(render_pass, "enabled", True))),
            "order": int(getattr(entry, "order", 0) or 0),
            "params": _pass_params(render_pass),
        })
    return items


def _available_pipelines() -> list[dict[str, Any]]:
    from Infernux.renderstack.discovery import discover_pipelines
    return [_pipeline_entry(name, cls) for name, cls in sorted(discover_pipelines().items())]


def _available_passes() -> list[dict[str, Any]]:
    from Infernux.renderstack.discovery import discover_passes
    return [_pass_entry(name, cls) for name, cls in sorted(discover_passes().items())]


def _pipeline_entry(name: str, cls) -> dict[str, Any]:
    return {"name": str(name), "class_name": cls.__name__, "module": cls.__module__, "fields": _field_schema(cls)}


def _pass_entry(name: str, cls) -> dict[str, Any]:
    return {
        "name": str(name),
        "class_name": cls.__name__,
        "module": cls.__module__,
        "injection_point": str(getattr(cls, "injection_point", "")),
        "default_order": int(getattr(cls, "default_order", 0) or 0),
        "menu_path": str(getattr(cls, "menu_path", "")),
        "requires": sorted(str(item) for item in getattr(cls, "requires", set()) or set()),
        "modifies": sorted(str(item) for item in getattr(cls, "modifies", set()) or set()),
        "creates": sorted(str(item) for item in getattr(cls, "creates", set()) or set()),
        "fields": _field_schema(cls),
    }


def _field_schema(cls) -> list[dict[str, Any]]:
    from Infernux.components.serialized_field import get_serialized_fields
    fields = []
    for name, meta in get_serialized_fields(cls).items():
        fields.append({
            "name": str(name),
            "type": getattr(getattr(meta, "field_type", None), "name", str(getattr(meta, "field_type", ""))),
            "default": serialize_value(getattr(meta, "default", None)),
            "range": serialize_value(getattr(meta, "range", None)),
            "tooltip": str(getattr(meta, "tooltip", "")),
            "readonly": bool(getattr(meta, "readonly", False)),
        })
    return fields


def _pass_class(pass_name: str):
    from Infernux.renderstack.discovery import discover_passes
    passes = discover_passes()
    if pass_name in passes:
        return passes[pass_name]
    lowered = str(pass_name).lower()
    for name, cls in passes.items():
        if name.lower() == lowered or cls.__name__.lower() == lowered:
            return cls
    raise FileNotFoundError(f"Render pass '{pass_name}' was not found.")


def _require_mounted_pass(stack, pass_name: str):
    lowered = str(pass_name).lower()
    for entry in list(getattr(stack, "pass_entries", []) or []):
        render_pass = entry.render_pass
        if str(getattr(render_pass, "name", "")).lower() == lowered or type(render_pass).__name__.lower() == lowered:
            return entry
    raise FileNotFoundError(f"Mounted render pass '{pass_name}' was not found.")


def _pass_params(render_pass) -> dict[str, Any]:
    if hasattr(render_pass, "get_params_dict"):
        try:
            return serialize_value(render_pass.get_params_dict())
        except Exception:
            return {}
    return {}


def _set_pass_params(render_pass, params: dict[str, Any]) -> None:
    if hasattr(render_pass, "set_params_dict"):
        render_pass.set_params_dict(params or {})
        return
    for key, value in (params or {}).items():
        setattr(render_pass, key, value)


def _mark_scene_dirty() -> None:
    try:
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.mark_dirty()
    except Exception:
        pass


def _register_metadata() -> None:
    for name, summary in {
        "renderstack.find_or_create": "Find or create the active scene RenderStack.",
        "renderstack.inspect": "Inspect the active RenderStack pipeline and mounted passes.",
        "renderstack.list_pipelines": "List discovered RenderPipeline classes.",
        "renderstack.set_pipeline": "Switch the active RenderStack pipeline.",
        "renderstack.list_passes": "List mounted and available RenderStack passes.",
        "renderstack.add_pass": "Mount a post-process or render pass on the active RenderStack.",
        "renderstack.remove_pass": "Remove a mounted RenderStack pass.",
        "renderstack.set_pass_enabled": "Enable or disable a mounted RenderStack pass.",
        "renderstack.set_pass_params": "Edit serialized parameters on a mounted RenderStack pass.",
    }.items():
        category = "renderstack/pipeline" if "pipeline" in name or name.endswith(("inspect", "find_or_create")) else "renderstack/effects"
        register_tool_metadata(
            name,
            summary=summary,
            category=category,
            tags=["renderstack", "rendering", "pipeline", "postprocess"],
            aliases=["render stack", "post processing", "effects"],
            recovery=[
                "If renderstack.inspect returns exists=false, do not repeat it.",
                "Call renderstack.find_or_create(name='RenderStack', create_if_missing=true) before editing the stack.",
            ],
            next_suggested_tools=["renderstack.find_or_create", "renderstack.inspect", "runtime.read_errors"],
        )
