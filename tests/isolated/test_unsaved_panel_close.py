"""A native dock close must preserve the dirty View until its modal resolves."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

if __name__ != "__main__":
    from tests.isolated import engine_python_path


@pytest.mark.parametrize("choice", ["cancel", "save", "discard", "clean"])
def test_unsaved_dock_close_preserves_presented_document(tmp_path, choice):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), choice],
        env={**os.environ, "PYTHONPATH": engine_python_path()},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    output = result.stdout + result.stderr
    (tmp_path / "native-close.log").write_text(output, encoding="utf-8")
    assert result.returncode == 0, output
    assert "UNSAVED_DOCK_CLOSE passed" in result.stdout, output


def exercise(project, choice):
    from infernux import Engine
    from infernux.engine.interaction import (
        DocumentCapability, DocumentKind, EditorInteractionCore,
        PanelInteractionDescriptor,
    )
    from infernux.engine.ui.closable_panel import ClosablePanel
    from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator
    from infernux.engine.ui.window_manager import WindowManager
    from infernux.lib import (
        InxGUIRenderable, LogLevel, RenderPipelineCallback,
        get_gui_semantic_snapshot, get_gui_window_presented_dock_peer,
        set_gui_semantic_capture_enabled,
    )

    project.mkdir(parents=True, exist_ok=True)
    (project / "ProjectSettings").mkdir(exist_ok=True)
    path = project / "document.txt"
    path.write_text("1", encoding="utf-8")
    engine = Engine(LogLevel.Warn)
    engine.init_renderer(1000, 700, str(project))
    native = engine._engine
    core = EditorInteractionCore()
    registry = core.documents
    frames = []
    samples = {}

    class Controller:
        value = 1

        def save(self, *, ticket, save_as=False):
            registry.capture_save_revision(ticket.ticket_id)
            path.write_text(str(self.value), encoding="utf-8")
            registry.complete_save(ticket.ticket_id, success=True)
            return True

        def discard(self, *, document_id):
            self.value = int(path.read_text(encoding="utf-8"))
            return True

    controller = Controller()
    document = registry.create(
        DocumentKind.GENERIC, "Close acceptance", resource_path=str(path),
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )

    class Panel(ClosablePanel):
        def on_render(self, ctx):
            if not self.is_open:
                return
            visible = self._begin_closable_window(ctx)
            samples[self.window_id] = bool(visible and ctx.is_current_window_content_presented())
            try:
                if visible:
                    ctx.label(f"Document value: {controller.value}" if self.window_id == "animfsm_editor" else "Scene content")
                    if self.window_id == "animfsm_editor" and ctx.button("Edit draft"):
                        controller.value += 1
                        registry.mark_changed(document.document_id, view_id=self.window_id)
            finally:
                ctx.end_window()

    manager = WindowManager(engine, core.panels, engine.register_gui)
    for view_id in ("scene_view", "animfsm_editor"):
        core.panels.register_type(view_id, PanelInteractionDescriptor(records_focus_history=False))
        manager.register_window_type(view_id, Panel, view_id,
                                     factory=lambda identifier=view_id: Panel(identifier, identifier))
    manager.open_window("scene_view")
    panel = manager.open_window("animfsm_editor")
    panel.bind_document(document.document_id)
    confirmation = DirtyPanelConfirmationCoordinator.instance()

    class Portal(InxGUIRenderable):
        def on_render(self, ctx):
            core.modals.render(ctx)

    class Pipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    portal = Portal()
    pipeline = Pipeline()
    native.set_render_pipeline(pipeline)
    engine.register_gui("test.close.portal", portal, priority=100)
    native.set_pre_gui_callback(manager.process_pending_actions)
    set_gui_semantic_capture_enabled(True)
    state = {"phase": "setup", "frame": 0, "after": 0, "click": None, "semantic_frame": -1,
             "click_step": 0, "close_seen": False,
             "decision": False, "done": False}
    deadline = time.monotonic() + 40

    def click_target(target, next_phase):
        state.update(click=tuple(target["click_point"]), click_step=0,
                     next_phase=next_phase)

    def update(_dt):
        if time.monotonic() > deadline or state["frame"] > 240:
            native.exit()
            return
        snapshot = get_gui_semantic_snapshot()
        semantic_frame = snapshot.get("frame", 0)
        # Updates can outnumber rendered frames. Keep motion/down/up in
        # separate GUI frames so this exercises a real tab close click.
        if semantic_frame == state["semantic_frame"]:
            return
        state["semantic_frame"] = semantic_frame
        state["frame"] += 1
        if state["click"] is not None:
            x, y = state["click"]
            step = state["click_step"]
            if step == 0:
                native.queue_synthetic_mouse_motion_input(x, y, 0, 0)
            else:
                native.queue_synthetic_mouse_button_input(0, step == 1, x, y)
            state["click_step"] += 1
            if step == 2:
                state["click"] = None
                state["phase"] = state["next_phase"]
                state["after"] = state["frame"]
            return
        targets = snapshot.get("targets", [])
        phase = state["phase"]
        if phase == "setup" and state["frame"] >= 5:
            native.select_docked_window("animfsm_editor")
            state["phase"] = "close" if choice == "clean" else "edit"
        elif phase == "edit":
            target = next((t for t in targets if t.get("label") == "Edit draft" and t.get("visible")), None)
            if target and samples.get("animfsm_editor"):
                click_target(target, "close")
        elif phase == "close" and (document.is_dirty or choice == "clean") and state["frame"] - state["after"] >= 4:
            target = next((t for t in targets if t.get("semantic_id") == "animfsm_editor.close"), None)
            if target:
                click_target(target, "resolved" if choice == "clean" else "await")
                state["close_seen"] = True
        elif phase == "await":
            target = next((t for t in targets if t.get("semantic_id") == f"editor.dirty_panel.{choice}"), None)
            if target and state["frame"] - state["after"] >= 8:
                state["decision"] = True
                click_target(target, "resolved")
        elif phase == "resolved" and state["frame"] - state["after"] >= 5:
            state["done"] = True
            native.exit()

    def after_draw():
        if not state["close_seen"]:
            return
        gui_frame = get_gui_semantic_snapshot().get("frame", 0)
        if frames and frames[-1]["gui_frame"] == gui_frame:
            return
        frames.append({
            "frame": state["frame"], "gui_frame": gui_frame, "phase": state["phase"],
            "presented": samples.get("animfsm_editor", False),
            "scene_presented": samples.get("scene_view", False),
            "peer": get_gui_window_presented_dock_peer("animfsm_editor"),
            "dirty": document.is_dirty, "open": panel.is_open,
            "modal": confirmation.is_active, "deciding": state["decision"],
        })
        if os.environ.get("INFERNUX_CLOSE_CAPTURE") and state["phase"] == "await":
            native.request_capture("editor", str(project / f"frame-{gui_frame:04}.png"))

    native.set_pre_scene_update_callback(update)
    native.set_post_draw_callback(after_draw)
    try:
        native.run()
        (project / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        assert state["done"], (state, frames[-10:])
        pending = [f for f in frames if f["phase"] == "await" and not f["deciding"]]
        if choice == "clean":
            assert not any(f["modal"] for f in frames), frames
        else:
            assert len(pending) >= 6, frames
            assert all(f["presented"] and f["open"] and not f["scene_presented"] for f in pending), pending
            assert all(f["dirty"] for f in pending), pending
        assert not confirmation.is_active
        if choice == "cancel":
            assert panel.is_open and document.is_dirty and controller.value == 2
            assert registry.document_for_view(panel.window_id) is document
            assert samples[panel.window_id]
        else:
            assert not panel.is_open
            assert not manager.is_window_open(panel.window_id)
        assert path.read_text(encoding="utf-8") == ("2" if choice == "save" else "1")
        if choice == "discard":
            assert controller.value == 1
        print("UNSAVED_DOCK_CLOSE passed", choice)
    finally:
        native.set_pre_scene_update_callback(None)
        native.set_post_draw_callback(None)
        native.set_pre_gui_callback(None)
        core.shutdown()
        WindowManager._instance = None
        DirtyPanelConfirmationCoordinator._instance = None
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
