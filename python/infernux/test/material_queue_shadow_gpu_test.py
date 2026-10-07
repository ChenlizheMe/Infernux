"""Live material routing changes must update color and shadow draws together."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3, LightShadows

RECEIVER = '''#version 450
ShaderInfo { Name "Queue Test Receiver" ShadingModel PBR Cull Off }
void surface(out SurfaceData s) {
    s=InitSurfaceData(); s.albedo=vec3(0.5); s.smoothness=0.0;
}
'''


class QueueTestPipeline(inx.renderstack.RenderPipeline):
    name = "Material Queue GPU Test"

    def define(self, pipeline):
        pipeline.frame(hdr=True, msaa=1)
        pipeline.shadows(resolution=1024)
        pipeline.lighting(clustered=False)
        with pipeline.opaque() as opaque:
            opaque.otherwise().forward()
        pipeline.sky()
        with pipeline.transparent() as transparent:
            transparent.otherwise().forward()
        pipeline.screen_ui()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    proof = {"passed": False, "package": inx.__file__, "cases": []}
    queues = (0, 2500, 2501, 2999, 3000, 5000, 5001, 9999, 0)
    with tempfile.TemporaryDirectory(prefix="infernux-material-queues-") as folder:
        project = Path(folder)
        for directory in ("Assets", "Packages", "ProjectSettings"):
            (project / directory).mkdir()
        shader = project / "Assets/Receiver.frag"
        shader.write_text(RECEIVER, encoding="ascii")
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = QueueTestPipeline()
        failures = []
        try:
            frontend.init_renderer(240, 180, str(project))
            frontend.resize_game_render_target(240, 180)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            imported = AssetManager.import_asset(str(shader), database=frontend.get_asset_database())
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object("Camera")
            camera.transform.position = Vector3(0, 0, -6)
            camera.add_component(inx.Camera)
            sun = scene.create_game_object("Sun")
            sun.transform.euler_angles = Vector3(0, 45, 0)
            sun.add_component(inx.Light).shadows = LightShadows.Hard
            mesh = inx.Mesh.from_data(
                np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], dtype=np.float32),
                np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32),
                normals=np.array([[0, 0, -1]] * 4, dtype=np.float32),
            )
            receiver = scene.create_game_object("Receiver")
            receiver.transform.local_scale = Vector3(6, 4, 1)
            receive_material = inx.Material.create_lit()
            receive_material.frag_shader_name = "Queue Test Receiver"
            receive_renderer = receiver.add_component(inx.MeshRenderer)
            receive_renderer.mesh = mesh
            receive_renderer.material = receive_material
            caster = scene.create_game_object("Caster")
            caster.transform.position = Vector3(-.5, 0, -1.5)
            renderer = caster.add_component(inx.MeshRenderer)
            renderer.mesh = mesh
            material = inx.Material.create_unlit()
            material.set_color("baseColor", 1, 0, 1, 1)
            renderer.material = material
            renderer.enabled = False
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            phase = frames = changed = 0
            ticket = None
            images = []
            graph_revision = None

            def after_draw():
                nonlocal phase, frames, changed, ticket, graph_revision
                try:
                    frames += 1
                    if ticket is None and frames >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        images.append(pixels)
                        if graph_revision is None:
                            graph_revision = pipeline._standalone_desc.source_revision
                        assert pipeline._standalone_desc.source_revision == graph_revision
                        findings = [e for e in console._get_visible_log_snapshot(2000)
                                    if e["level"] in ("ERROR", "FATAL")]
                        assert not findings, findings
                        if phase == len(queues):
                            baseline, casting = images[:2]
                            silhouette = (casting[..., 0] > .95) & (casting[..., 1] < .05) & (casting[..., 2] > .95)
                            assert silhouette.sum() > 100
                            shadow = (baseline[..., 0] - casting[..., 0] > .08) & ~silhouette
                            assert shadow.sum() > 100
                            for index, queue in enumerate(queues, 1):
                                expected = casting if queue <= 2999 else baseline
                                np.testing.assert_allclose(images[index][shadow], expected[shadow], atol=.005,
                                                           err_msg=f"Queue {queue}: stale shadow routing")
                                rgb = images[index][..., :3]
                                magenta = (rgb[..., 0] > .95) & (rgb[..., 1] < .05) & (rgb[..., 2] > .95)
                                assert (magenta.sum() > 100) == (queue <= 5000), queue
                                proof["cases"].append({"queue": queue, "color_visible": queue <= 5000,
                                                       "casts_shadow": queue <= 2999,
                                                       "compared_shadow_pixels": int(shadow.sum())})
                            np.testing.assert_array_equal(images[-1], casting)
                            proof["passed"] = True
                            print("PASS live queue/color/shadow boundaries and exact restore", flush=True)
                            native.exit()
                            return
                        phase += 1
                        renderer.enabled = True
                        material.render_queue = queues[phase - 1]
                        ticket, changed = None, frames
                    if frames > 240:
                        raise AssertionError("Material queue GPU audit timed out")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert proof["passed"]
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()
    if args.proof:
        args.proof.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
