"""Native registrations own Python callbacks until their native registration ends."""
from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref

import pytest

from infernux.lib import InxGUIRenderable
from infernux.renderstack import RenderStackPipeline


@pytest.mark.parametrize("kind", ["panel", "pipeline"])
def test_registration_retains_python_subclass_and_releases_on_replacement(engine, kind):
    disposed = []

    class Panel(InxGUIRenderable):
        def on_render(self, ctx):
            pass

    class Pipeline(RenderStackPipeline):
        def dispose(self):
            disposed.append(self.tag)
            super().dispose()

    def install(value):
        if kind == "panel":
            engine.register_gui_renderable("native-ownership", value)
        else:
            engine.set_render_pipeline(value)

    def clear():
        if kind == "panel":
            engine.unregister_gui_renderable("native-ownership")
        else:
            engine.set_render_pipeline(None)

    cls = Panel if kind == "panel" else Pipeline
    try:
        for tag in range(8):
            value = cls()
            value.tag = tag
            current = weakref.ref(value)
            install(value)
            del value
            gc.collect()
            assert current() is not None, f"{kind}: native registration lost the Python override"
            if tag:
                assert previous() is None, "replacement retained the old Python instance"
            previous = current
        clear()
        gc.collect()
        assert current() is None, "unregistration retained the Python instance"
        assert disposed == (list(range(8)) if kind == "pipeline" else [])
    finally:
        clear()


def test_same_panel_two_registrations_release_independently(engine):
    class Panel(InxGUIRenderable):
        def on_render(self, ctx):
            pass

    panel = Panel()
    reference = weakref.ref(panel)
    try:
        engine.register_gui_renderable("owner-a", panel)
        engine.register_gui_renderable("owner-b", panel)
        del panel
        gc.collect()
        assert reference() is not None
        engine.unregister_gui_renderable("owner-a")
        gc.collect()
        assert reference() is not None
        engine.unregister_gui_renderable("owner-b")
        gc.collect()
        assert reference() is None
    finally:
        engine.unregister_gui_renderable("owner-a")
        engine.unregister_gui_renderable("owner-b")


@pytest.mark.parametrize("action", ["unregister", "replace"])
def test_temporary_callbacks_render_and_self_unregister_safely(engine, scene, capfd, action):
    calls, errors, disposed = [], [], []
    references = {}
    scene.create_game_object("Ownership camera").add_component("Camera")

    class Replacement(InxGUIRenderable):
        def on_render(self, ctx):
            calls.append("replacement")

    class Panel(InxGUIRenderable):
        def __init__(self):
            super().__init__()
            references["panel"] = weakref.ref(self)

        def on_render(self, ctx):
            try:
                calls.append("panel")
                if action == "unregister":
                    engine.unregister_gui_renderable("temporary-callback")
                else:
                    engine.register_gui_renderable("temporary-callback", Replacement())
                gc.collect()
                assert references["panel"]() is self
                ctx.set_next_window_size(64, 64, 1)
                ctx.begin_window("Temporary callback", True, 0)
                try:
                    ctx.draw_text_ex_aligned(8, 29, 60, 62, "Alive", 1, 1, 1, 1, font_size=14)
                finally:
                    ctx.end_window()
            except BaseException as error:
                errors.append(error)
                engine.exit()

    class Pipeline(RenderStackPipeline):
        def __init__(self):
            super().__init__()
            references["pipeline"] = weakref.ref(self)

        def render(self, context, camera):
            calls.append("pipeline")
            super().render(context, camera)

        def dispose(self):
            disposed.append(True)
            super().dispose()

    frames = 0

    def after_draw():
        nonlocal frames
        try:
            frames += 1
            gc.collect()
            assert references["panel"]() is None
            assert references["pipeline"]() is not None
            if frames == 3:
                engine.exit()
        except BaseException as error:
            errors.append(error)
            engine.exit()

    try:
        engine.resize_game_render_target(64, 64)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(Pipeline())
        engine.register_gui_renderable("temporary-callback", Panel())
        engine.set_post_draw_callback(after_draw)
        gc.collect()
        # This guard also makes a regression fail cleanly before a dead
        # pure-virtual trampoline could abort the test process.
        assert all(reference() is not None for reference in references.values())
        engine.run()
        assert not errors, repr(errors)
        assert frames == 3
        assert calls.count("panel") == 1
        assert calls.count("pipeline") >= 3
        assert calls.count("replacement") == (2 if action == "replace" else 0)
    finally:
        engine.set_game_camera_enabled(False)
        engine.set_post_draw_callback(None)
        if action == "replace":
            engine.unregister_gui_renderable("temporary-callback")
        engine.set_render_pipeline(None)
    gc.collect()
    assert references["pipeline"]() is None
    assert disposed == [True]
    captured = capfd.readouterr()
    assert not any(item in captured.out + captured.err for item in ("VUID-", "SYNC-HAZARD", "Validation Error"))


def test_native_cleanup_releases_registered_python_owners(tmp_path):
    project = tmp_path / "Owner teardown"
    (project / "ProjectSettings").mkdir(parents=True)
    script = textwrap.dedent('''
        import gc, sys, weakref
        from infernux.lib import Infernux, InxGUIRenderable, LogLevel, SceneManager, lib_dir
        from infernux.renderstack import RenderStackPipeline
        from infernux.resources import resources_path
        engine = Infernux(lib_dir)
        engine.set_log_level(LogLevel.Warn)
        engine.init_renderer(64, 64, sys.argv[1], resources_path)
        scene = SceneManager.instance().create_scene("Native ownership cleanup")
        SceneManager.instance().set_active_scene(scene)
        scene.create_game_object("Camera").add_component("Camera")
        engine.resize_game_render_target(64, 64)
        engine.set_game_camera_enabled(True)
        class Panel(InxGUIRenderable):
            def on_render(self, ctx):
                pass
        panel, pipeline = Panel(), RenderStackPipeline()
        references = [weakref.ref(panel), weakref.ref(pipeline)]
        engine.register_gui_renderable("retained-at-cleanup", panel)
        engine.set_render_pipeline(pipeline)
        del panel, pipeline
        gc.collect()
        assert all(reference() is not None for reference in references)
        engine.set_post_draw_callback(engine.exit)
        engine.run()
        engine.cleanup()  # Binding releases GIL while native owners retire.
        gc.collect()
        assert all(reference() is None for reference in references)
        print("native ownership teardown passed")
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(project)],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "native ownership teardown passed" in output
    assert not any(item in output for item in ("VUID-", "SYNC-HAZARD", "Validation Error"))
