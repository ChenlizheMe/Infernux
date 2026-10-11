"""Exercise native graph rejection, retained output, and repair through Python."""

import json
import tempfile
from pathlib import Path

import numpy as np

import infernux as inx
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.rendergraph import RenderGraph


def define_graph(graph, color):
    graph.set_msaa_samples(1)
    output = graph.create_texture("color", camera_target=True)
    graph.add_pass("Clear").write_color(output).set_clear(color=color).draw_renderers()
    graph.screen_ui_overlay_section(resources={"color"})
    graph.set_output(output)


def description(color, *, invalid=False):
    graph = RenderGraph("Native Publication Audit", output_samples=1)
    define_graph(graph, color)
    desc = graph.build()
    if invalid:
        # Bypass the Python builder's validation through the public native
        # description API. Native validation must still reach its caller.
        textures = desc.textures
        next(texture for texture in textures if texture.name == "color").is_depth = True
        desc.textures = textures
    return desc


class Publication(inx.renderstack.RenderPipeline):
    name = "Native Publication Audit"

    def __init__(self):
        super().__init__()
        self.phase = 0
        self.prepared_phase = -1
        self.accepted = None
        self.failures = []
        self.events = []

    def define_topology(self, graph):
        define_graph(graph, (1, 0, 0, 1))

    def reject(self, context, camera, entry):
        desc = description((0, 0, 1, 1), invalid=True)
        assert not context.is_graph_revision_current(desc.source_revision)
        if self.accepted is not None:
            assert context.is_graph_revision_current(self.accepted.source_revision), (entry, "before rejection")
        try:
            if entry == "apply_graph":
                context.apply_graph(desc)
            else:
                context.render_with_graph(camera, desc)
        except RuntimeError as exc:
            assert "publication rejected by native validation" in str(exc), str(exc)
            self.events.append({"phase": self.phase, "entry": entry, "error": str(exc)})
        else:
            raise AssertionError(f"{entry} returned normally after native rejection")
        assert not context.is_graph_revision_current(desc.source_revision)
        if self.accepted is not None:
            assert context.is_graph_revision_current(self.accepted.source_revision), (entry, "after rejection")

    def render_camera(self, context, camera, culling):
        try:
            # The screen attachment contract can change after the first
            # submission. Refresh the accepted graph as the base pipeline
            # normally does, before attempting an invalid edit.
            if self.accepted is not None and not context.is_graph_revision_current(self.accepted.source_revision):
                context.apply_graph(self.accepted)
            if self.prepared_phase != self.phase:
                if self.phase in (0, 1):
                    for entry in ("apply_graph", "render_with_graph"):
                        self.reject(context, camera, entry)
                if self.phase in (0, 2, 3):
                    color = (1, 0, 0, 1) if self.phase == 0 else (0, 0, 1, 1)
                    desc = description(color)
                    # Phase 3 republishes identical topology with a fresh
                    # revision; phase 2 repairs the rejected topology.
                    context.apply_graph(desc)
                    assert context.is_graph_revision_current(desc.source_revision)
                    if self.accepted is not None:
                        assert not context.is_graph_revision_current(self.accepted.source_revision)
                    self.accepted = desc
                context.apply_graph(self.accepted)
                assert context.is_graph_revision_current(self.accepted.source_revision)
                self.prepared_phase = self.phase
            context.submit_culling(culling)
        except BaseException as exc:
            self.failures.append(exc)
            self.native.exit()


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-graph-publication-") as folder:
        project = Path(folder)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (project / name).mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = Publication()
        pipeline.native = native
        proof = {"phases": [], "rejections": pipeline.events}
        completed = False
        failures = []
        try:
            frontend.init_renderer(160, 120, str(project))
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            owner = scene.create_game_object("Publication Camera")
            owner.transform.position = Vector3(0, 0, -5)
            scene.main_camera = owner.add_component(inx.Camera)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            names = ("initial_rejection_then_valid", "rejected_edit_retains_output", "repaired", "unchanged_replay")
            frame = 0
            changed = 0
            ticket = None

            def after_draw():
                nonlocal completed, frame, changed, ticket
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        rgb = ticket.result_numpy().copy().astype(np.float32)[..., :3]
                        expected = np.array((1, 0, 0) if pipeline.phase < 2 else (0, 0, 1))
                        assert np.all(np.abs(rgb - expected) < .01), (names[pipeline.phase], rgb.min(), rgb.max())
                        proof["phases"].append({"phase": names[pipeline.phase], "pixels": int(rgb.shape[0] * rgb.shape[1]), "revision": int(pipeline.accepted.source_revision)})
                        if pipeline.phase == len(names) - 1:
                            completed = True
                            native.exit()
                            return
                        pipeline.phase += 1
                        changed = frame
                        ticket = None
                    if frame > 100:
                        raise AssertionError("Graph publication GPU audit timed out")
                except BaseException as exc:
                    failures.append(exc)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if pipeline.failures:
                raise pipeline.failures[0]
            if failures:
                raise failures[0]
            assert completed
            assert len(pipeline.events) == 4
            errors = [entry for entry in console._get_visible_log_snapshot(2000) if entry["level"] in ("ERROR", "FATAL")]
            assert len(errors) == 4 and all("cannot be both backbuffer and depth" in entry["message"] for entry in errors), errors
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()
        print(json.dumps(proof, indent=2), flush=True)
        print("PASS native graph rejection, retained GPU output, repair and replay", flush=True)


if __name__ == "__main__":
    main()
