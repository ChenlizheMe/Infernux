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
        # description API. The executor must reject the invalid publication.
        passes = desc.passes
        passes[0].write_colors = [(0, "missing-color")]
        desc.passes = passes
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

    def render_view(self, context, view, culling):
        try:
            if self.prepared_phase != self.phase:
                color = (1, 0, 0, 1) if self.phase == 0 else (0, 0, 1, 1)
                self.description = description(color, invalid=self.phase == 1)
                if self.phase != 1:
                    self.accepted = self.description
                self.prepared_phase = self.phase
            # The view callback publishes a completed-world plan. Native graph
            # validation happens when the host executes it, not during capture.
            # Rejected execution must leave the last completed pixels intact.
            context.apply_graph(self.description)
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
            names = ("valid_publication", "rejected_edit_retains_output", "repaired", "unchanged_replay")
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
                        if pipeline.phase == 1:
                            errors = console._get_visible_log_snapshot(2000)
                            rejected = [entry["message"] for entry in errors
                                        if "unknown color target 'missing-color'" in entry["message"]]
                            assert rejected, "Invalid publication was not rejected by the executor"
                            pipeline.events.append({"phase": 1, "diagnostic": rejected[0]})
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
            assert len(pipeline.events) == 1
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()
        print(json.dumps(proof, indent=2), flush=True)
        print("PASS native graph rejection, retained GPU output, repair and replay", flush=True)


if __name__ == "__main__":
    main()
