"""Exercise ordinary panel authoring through the real native GUI dispatcher."""
from __future__ import annotations

from infernux.engine.ui.editor_panel import EditorPanel
from infernux.lib import ConsolePanel
from infernux.plugins.registry import PluginRegistry
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline


def test_registered_panel_updates_every_frame_with_automatic_read_reuse(engine, tmp_path):
    registry = PluginRegistry(str(tmp_path))
    document = registry.load()
    document["packages"] = [{"reference": "team/first"}]
    registry.save(document)
    peer = PluginRegistry(str(tmp_path))
    observations = []
    loads = []
    load = registry._load

    def counted_load():
        loads.append(1)
        return load()

    registry._load = counted_load

    class CounterPanel(EditorPanel):
        def __init__(self):
            super().__init__("Live counter", "test.live_counter")
            self.counter = 0

        def _pre_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(64., 64., 0)

        def on_render_content(self, ctx):
            # No cache scopes, dirty flags or memoization in user panel code.
            self.counter += 1
            rows = registry.available()
            ctx.label(f"Frame {self.counter}: {rows[0]['reference']}")
            ctx.button("Live button")
            observations.append((self.counter, rows[0]["reference"]))
            assert registry.find(rows[0]["reference"]) is not None

    class EmptyPanel(EditorPanel):
        def __init__(self):
            super().__init__("Empty", "test.empty_registered")

        def _pre_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(64., 64., 0)

        def on_render_content(self, ctx):
            pass

    panel = CounterPanel()
    empty = EmptyPanel()
    native = ConsolePanel()
    frames = 0

    def update(_delta):
        nonlocal frames
        frames += 1
        if frames == 20:
            revised = peer.load()
            revised["packages"] = [{"reference": "team/second"}]
            peer.save(revised)
        if frames >= 75:
            engine.exit()

    engine.register_gui_renderable("test.live_counter", panel)
    engine.register_gui_renderable("test.empty_registered", empty)
    engine.register_gui_renderable("test.native_console", native)
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
        profile = engine.renderer_ui_performance_snapshot
    finally:
        engine.set_pre_scene_update_callback(None)
        for name in ("test.live_counter", "test.empty_registered", "test.native_console"):
            engine.unregister_gui_renderable(name)
        engine.set_render_pipeline(None)
        pipeline.dispose()
    assert panel._content_render_error_signature is None
    assert len(observations) >= 65
    assert [count for count, _ in observations] == list(range(1, len(observations) + 1))
    assert observations[0][1] == "team/first"
    assert observations[-1][1] == "team/second"
    assert len(loads) == 2
    assert profile["panel_times"]["test.live_counter"]["sample_count"] >= 65
