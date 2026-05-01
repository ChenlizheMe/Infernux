"""Editor-state MCP tools."""

from __future__ import annotations

from Infernux.mcp.tools.common import main_thread, scene_status


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
                "scene_status": scene_status(),
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
            status = scene_status()
            if status["play_state"] != "edit":
                raise RuntimeError("Play Mode is already active.")
            if status["loading"]:
                raise RuntimeError("Cannot enter Play Mode while scene loading is pending.")
            try:
                from Infernux.engine.deferred_task import DeferredTaskRunner
                runner = DeferredTaskRunner.instance()
                if runner and runner.is_busy:
                    raise RuntimeError("Cannot enter Play Mode while a deferred editor task is running.")
            except RuntimeError:
                raise
            except Exception:
                pass
            if not status["saved_to_file"]:
                raise RuntimeError("Cannot enter Play Mode until the active scene is saved. Call scene.save first.")
            if status["dirty"]:
                raise RuntimeError("Cannot enter Play Mode while the active scene is dirty. Call scene.save first.")
            accepted = bool(pmm.enter_play_mode())
            return {
                "accepted": accepted,
                "state": pmm.state.name.lower(),
                "requested_state": "playing" if accepted else pmm.state.name.lower(),
                "deferred": bool(accepted),
                "preflight": status,
                "next_suggested_tools": ["runtime.wait", "mcp.health", "runtime.read_errors"] if accepted else ["scene.status"],
            }

        return main_thread("editor.play", _play)

    @mcp.tool(name="editor.stop")
    def editor_stop() -> dict:
        """Exit Play Mode."""

        def _stop():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            if pmm.state.name.lower() == "edit":
                return {"accepted": True, "already_stopped": True, "state": "edit"}
            return {"accepted": bool(pmm.exit_play_mode()), "already_stopped": False, "state": pmm.state.name.lower()}

        return main_thread("editor.stop", _stop)

    @mcp.tool(name="editor.pause")
    def editor_pause() -> dict:
        """Pause Play Mode."""

        def _pause():
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm is None:
                raise RuntimeError("PlayModeManager is not available.")
            if pmm.state.name.lower() != "playing":
                raise RuntimeError("editor.pause requires Play Mode to be playing.")
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
            if pmm.state.name.lower() != "paused":
                raise RuntimeError("editor.resume requires Play Mode to be paused.")
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
            if pmm.state.name.lower() != "paused":
                raise RuntimeError("editor.step requires paused Play Mode. Call editor.pause after editor.play before stepping.")
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

        return main_thread("editor.select", _select, arguments={"object_ids": object_ids or [], "primary_id": primary_id})
