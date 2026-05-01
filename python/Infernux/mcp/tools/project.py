"""Project-level MCP tools."""

from __future__ import annotations

from Infernux.mcp.tools.common import main_thread


def register_project_tools(mcp, project_path: str) -> None:
    @mcp.tool(name="project.info")
    def project_info() -> dict:
        """Return the currently opened project and scene state."""

        def _read():
            from Infernux.engine.play_mode import PlayModeManager
            from Infernux.engine.scene_manager import SceneFileManager
            from Infernux.engine.ui.selection_manager import SelectionManager
            from Infernux.lib import SceneManager

            sfm = SceneFileManager.instance()
            pmm = PlayModeManager.instance()
            sel = SelectionManager.instance()
            scene = SceneManager.instance().get_active_scene()
            return {
                "engine_version": "0.1.6",
                "project_root": project_path,
                "active_scene": {
                    "name": getattr(scene, "name", ""),
                    "path": getattr(sfm, "current_scene_path", "") if sfm else "",
                    "dirty": bool(getattr(sfm, "is_dirty", False)) if sfm else False,
                },
                "play_state": getattr(getattr(pmm, "state", None), "name", "edit").lower() if pmm else "edit",
                "selected_ids": sel.get_ids() if sel else [],
            }

        return main_thread("project.info", _read)
