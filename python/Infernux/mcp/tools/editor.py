"""Editor-state MCP tools."""

from __future__ import annotations

from Infernux.mcp.tools.common import main_thread


def register_editor_tools(mcp) -> None:
    @mcp.tool(name="editor.get_state")
    def editor_get_state() -> dict:
        """Return lightweight editor state."""

        def _read():
            from Infernux.engine.deferred_task import DeferredTaskRunner
            from Infernux.engine.play_mode import PlayModeManager
            from Infernux.engine.scene_manager import SceneFileManager
            from Infernux.engine.ui.selection_manager import SelectionManager

            pmm = PlayModeManager.instance()
            sfm = SceneFileManager.instance()
            sel = SelectionManager.instance()
            runner = DeferredTaskRunner.instance()
            return {
                "play_state": getattr(getattr(pmm, "state", None), "name", "edit").lower() if pmm else "edit",
                "deferred_task_busy": bool(getattr(runner, "is_busy", False)),
                "selected_ids": sel.get_ids() if sel else [],
                "scene_dirty": bool(sfm.is_dirty) if sfm else False,
                "is_prefab_mode": bool(getattr(sfm, "is_prefab_mode", False)) if sfm else False,
            }

        return main_thread("editor.get_state", _read)

    @mcp.tool(name="editor.play")
    def editor_play() -> dict:
        """Enter Play Mode."""

        def _play():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            return {"accepted": bool(pmm.enter_play_mode()), "state": pmm.state.name.lower()}

        return main_thread("editor.play", _play)

    @mcp.tool(name="editor.stop")
    def editor_stop() -> dict:
        """Exit Play Mode."""

        def _stop():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            return {"accepted": bool(pmm.exit_play_mode()), "state": pmm.state.name.lower()}

        return main_thread("editor.stop", _stop)

    @mcp.tool(name="editor.pause")
    def editor_pause() -> dict:
        """Pause Play Mode."""

        def _pause():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            return {"accepted": bool(pmm.pause()), "state": pmm.state.name.lower()}

        return main_thread("editor.pause", _pause)

    @mcp.tool(name="editor.resume")
    def editor_resume() -> dict:
        """Resume from paused Play Mode."""

        def _resume():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            return {"accepted": bool(pmm.resume()), "state": pmm.state.name.lower()}

        return main_thread("editor.resume", _resume)

    @mcp.tool(name="editor.step")
    def editor_step() -> dict:
        """Step one frame while Play Mode is paused."""

        def _step():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            pmm.step_frame()
            return {"state": pmm.state.name.lower()}

        return main_thread("editor.step", _step)

    @mcp.tool(name="editor.select")
    def editor_select(object_ids: list[int] | None = None, primary_id: int = 0) -> dict:
        """Set the current editor selection."""

        def _select():
            from Infernux.engine.ui.selection_manager import SelectionManager
            sel = SelectionManager.instance()
            ids = [int(i) for i in (object_ids or []) if int(i) > 0]
            if primary_id:
                sel.select(int(primary_id))
            elif ids:
                sel.box_select(ids)
            else:
                sel.clear()
            return {"selected_ids": sel.get_ids()}

        return main_thread("editor.select", _select)
