"""Minimized windows must retain their last drawable editor layout."""
import ctypes
from ctypes import wintypes
import os
import sys
import threading
import time

import pytest

from infernux.lib import InxGUIRenderable, RenderPipelineCallback


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises native Win32 minimize/restore")
def test_minimize_keeps_agent_gui_usable_and_preserves_docked_widths(engine):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.ShowWindowAsync.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    title = f"Infernux minimize regression {os.getpid()}"
    engine.set_window_title(title)
    engine.show()
    engine.set_maximized(True)
    window = user32.FindWindowW(None, title)
    assert window
    started = threading.Event()
    restored = threading.Event()
    complete = threading.Event()
    cancelled = threading.Event()
    observed = {}

    class GuiOnly(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    class LayoutProbe(InxGUIRenderable):
        frames = 0
        widths = {}

        def on_render(self, ctx):
            widths = {}
            for panel in ("hierarchy", "inspector", "toolbar", "scene_view", "project"):
                try:
                    if ctx.begin_window(f"Probe###{panel}", True, 0):
                        widths[panel] = ctx.get_window_width()
                        ctx.button(panel)
                finally:
                    ctx.end_window()
            self.widths = widths
            self.frames += 1
            if self.frames >= 3:
                started.set()
            if restored.is_set() and self.frames >= observed["minimized_frames"] + 3:
                complete.set()

    probe = LayoutProbe()

    def manipulate_window():
        try:
            assert started.wait(10), "Initial editor layout did not render"
            if cancelled.is_set():
                return
            observed["before"] = dict(probe.widths)
            user32.ShowWindowAsync(window, 6)
            deadline = time.monotonic() + 5
            while not user32.IsIconic(window):
                assert time.monotonic() < deadline, "Window did not minimize"
                time.sleep(.01)
            # Let SDL consume the OS event, then demand a real synthetic UI
            # frame. Ordinary presentation already suspends on minimization;
            # this exercises the automation path that used to build at 0x0.
            time.sleep(.05)
            observed["initial_minimized_frames"] = probe.frames
            engine.queue_synthetic_mouse_motion_input(100, 100, 0, 0)
            engine.queue_synthetic_mouse_button_input(0, True, 100, 100)
            release = engine.queue_synthetic_mouse_button_input(0, False, 100, 100)
            deadline = time.monotonic() + 5
            while engine.last_processed_synthetic_input_sequence < release:
                assert time.monotonic() < deadline, "Minimized GUI did not consume the complete input gesture"
                time.sleep(.01)
            time.sleep(.05)
            assert user32.IsIconic(window)
            observed["minimized_frames"] = probe.frames
            observed["during"] = dict(probe.widths)
            user32.ShowWindowAsync(window, 9)
            restored.set()
            assert complete.wait(10), "Editor did not resume rendering"
            observed["after"] = dict(probe.widths)
        except BaseException as exc:
            observed["error"] = repr(exc)
        finally:
            user32.ShowWindowAsync(window, 9)
            engine.exit()

    worker = threading.Thread(target=manipulate_window, daemon=True)
    engine.set_render_pipeline(GuiOnly())
    engine.register_gui_renderable("test.minimize_layout", probe)
    try:
        worker.start()
        engine.run()
        worker.join(5)
        assert not worker.is_alive()
    finally:
        cancelled.set()
        started.set()
        worker.join(5)
        engine.unregister_gui_renderable("test.minimize_layout")
        engine.set_render_pipeline(None)
        engine.set_maximized(False)
        engine.hide()
    assert "error" not in observed, observed
    assert observed["minimized_frames"] > observed["initial_minimized_frames"], observed
    assert set(observed["before"]) == set(observed["during"])
    assert all(abs(width - observed["during"][panel]) <= 2 for panel, width in observed["before"].items()), observed
    assert set(observed["before"]) == set(observed["after"])
    assert all(abs(width - observed["after"][panel]) <= 2 for panel, width in observed["before"].items()), observed
