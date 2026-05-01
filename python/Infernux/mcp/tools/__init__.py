"""MCP tool registration for the embedded Infernux server."""

from __future__ import annotations

from typing import Any

from Infernux.mcp import capabilities


def register_all_tools(mcp, project_path: str, config: dict[str, Any] | None = None) -> None:
    """Register all enabled MCP tool groups.

    The import style is intentionally lazy so disabled groups do not pull in
    optional editor subsystems during startup.
    """
    config = config or capabilities.current_config()
    if not bool(config.get("enabled", True)):
        return
    gated_mcp = _ToolGate(mcp, config)

    if _group(config, "docs"):
        from Infernux.mcp.tools.docs import register_docs_tools
        register_docs_tools(gated_mcp, project_path, config)
    if _group(config, "api"):
        from Infernux.mcp.tools.api import register_api_tools
        register_api_tools(gated_mcp)
    if _group(config, "project"):
        from Infernux.mcp.tools.project import register_project_tools
        register_project_tools(gated_mcp, project_path)
    if _group(config, "editor"):
        from Infernux.mcp.tools.editor import register_editor_tools
        register_editor_tools(gated_mcp)
    if _group(config, "scene"):
        from Infernux.mcp.tools.scene import register_scene_tools
        register_scene_tools(gated_mcp)
    if _group(config, "hierarchy"):
        from Infernux.mcp.tools.hierarchy import register_hierarchy_tools
        register_hierarchy_tools(gated_mcp)
    if _group(config, "asset"):
        from Infernux.mcp.tools.assets import register_asset_tools
        register_asset_tools(gated_mcp, project_path)
    if _group(config, "material"):
        from Infernux.mcp.tools.material import register_material_tools
        register_material_tools(gated_mcp, project_path)
    if _group(config, "renderstack"):
        from Infernux.mcp.tools.renderstack import register_renderstack_tools
        register_renderstack_tools(gated_mcp)
    if _group(config, "console"):
        from Infernux.mcp.tools.console import register_console_tools
        register_console_tools(gated_mcp)
    if _group(config, "camera"):
        from Infernux.mcp.tools.camera import register_camera_tools
        register_camera_tools(gated_mcp)
    if _group(config, "runtime") and _feature(config, "runtime_observation"):
        from Infernux.mcp.tools.runtime import register_runtime_tools
        register_runtime_tools(gated_mcp)
    if _group(config, "ui"):
        from Infernux.mcp.tools.ui import register_ui_tools
        register_ui_tools(gated_mcp)
    if _group(config, "transactions") and _feature(config, "transactions"):
        from Infernux.mcp.tools.transactions import register_transaction_tools
        register_transaction_tools(gated_mcp, project_path)
    if _group(config, "research"):
        from Infernux.mcp.tools.research import register_research_tools
        register_research_tools(gated_mcp, project_path)
    if _group(config, "project_tool_management") and _feature(config, "project_defined_tools"):
        from Infernux.mcp.tools.project_tools import register_project_tool_management
        register_project_tool_management(gated_mcp, project_path)
    if _group(config, "project_defined_tools") and _feature(config, "project_defined_tools"):
        from Infernux.mcp.tools.project_tools import register_project_defined_tools
        register_project_defined_tools(gated_mcp, project_path)


def _feature(config: dict[str, Any], name: str) -> bool:
    return bool((config.get("features") or {}).get(name, True))


def _group(config: dict[str, Any], name: str) -> bool:
    return bool((config.get("tool_groups") or {}).get(name, True))


class _ToolGate:
    def __init__(self, mcp, config: dict[str, Any]) -> None:
        self._mcp = mcp
        self._disabled = set(str(item) for item in config.get("disabled_tools", []))

    def tool(self, *args, **kwargs):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        if name and str(name) in self._disabled:
            def _skip(fn):
                return fn
            return _skip
        decorator = self._mcp.tool(*args, **kwargs)

        def _register(fn):
            try:
                from Infernux.mcp.tools.common import register_tool_signature
                register_tool_signature(str(name or getattr(fn, "__name__", "")), fn)
            except Exception:
                pass
            return decorator(fn)

        return _register

    def __getattr__(self, name: str):
        return getattr(self._mcp, name)
