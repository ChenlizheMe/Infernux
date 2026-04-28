"""MCP tool registration for the embedded Infernux server."""

from __future__ import annotations

from Infernux.mcp.tools.assets import register_asset_tools
from Infernux.mcp.tools.camera import register_camera_tools
from Infernux.mcp.tools.console import register_console_tools
from Infernux.mcp.tools.docs import register_docs_tools
from Infernux.mcp.tools.editor import register_editor_tools
from Infernux.mcp.tools.hierarchy import register_hierarchy_tools
from Infernux.mcp.tools.material import register_material_tools
from Infernux.mcp.tools.project import register_project_tools
from Infernux.mcp.tools.runtime import register_runtime_tools
from Infernux.mcp.tools.scene import register_scene_tools
from Infernux.mcp.tools.ui import register_ui_tools


def register_all_tools(mcp, project_path: str) -> None:
    register_docs_tools(mcp, project_path)
    register_project_tools(mcp, project_path)
    register_editor_tools(mcp)
    register_scene_tools(mcp)
    register_hierarchy_tools(mcp)
    register_asset_tools(mcp, project_path)
    register_material_tools(mcp, project_path)
    register_console_tools(mcp)
    register_camera_tools(mcp)
    register_runtime_tools(mcp)
    register_ui_tools(mcp)
