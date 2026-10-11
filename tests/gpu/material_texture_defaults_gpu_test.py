"""GPU proof of declared defaults, explicit tokens, overrides and shader reload."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3


SHADER = '''#version 450
ShaderInfo {
    Name "Declared Texture Default Probe"
    ShadingModel Unlit
    Queue 2000
    Properties { Texture2D independentName = DEFAULT }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    vec4 texel = sampleAlbedoAlpha(independentName);
    s.albedo = texel.rgb;
    s.alpha = texel.a;
}
'''


class DefaultProbePipeline(inx.renderstack.RenderPipeline):
    name = "Declared Texture Default GPU"

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        color = graph.create_texture("color", camera_target=True)
        depth = graph.create_texture("depth", format=inx.rendergraph.Format.D32_SFLOAT, samples=1)
        graph.add_pass("DefaultBindings").write_color(color).write_depth(depth).set_clear(
            color=(0, 0, 0, 0), depth=1,
        ).draw_renderers(queue_range=(0, 2500))
        graph.screen_ui_section(resources={"color"})
        graph.set_output(color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    phases = [
        ("white_unassigned", "white", "", None, (1, 1, 1)),
        ("black_unassigned", "black", "", None, (0, 0, 0)),
        ("normal_unassigned", "normal", "", None, (.5, .5, 1)),
        ("assigned_red", "normal", "asset", None, (1, 0, 0)),
        ("changed_default_keeps_assigned", "black", "asset", None, (1, 0, 0)),
        ("clear_restores_black", "black", "", None, (0, 0, 0)),
        ("normal_restored", "normal", "", None, (.5, .5, 1)),
        ("explicit_black", "normal", "black", None, (0, 0, 0)),
        ("explicit_white", "normal", "white", None, (1, 1, 1)),
        ("explicit_normal", "white", "normal", None, (.5, .5, 1)),
        ("override_black", "white", "normal", "black", (0, 0, 0)),
        ("override_white", "black", "normal", "white", (1, 1, 1)),
        ("override_normal", "white", "black", "normal", (.5, .5, 1)),
        ("clear_override_restores_material", "white", "black", None, (0, 0, 0)),
        ("white_exact_restore", "white", "", None, (1, 1, 1)),
    ]
    proof = {"passed": False, "package": inx.__file__, "phases": [],
             "scope": "Independent disposable Vulkan renderer, numerical GPU readback only. Property name differs from default tokens. Declared and explicit white/black/normal, per-renderer overrides, assigned asset preservation across shader reload, clear and exact restore. No live Editor viewport access."}
    with tempfile.TemporaryDirectory(prefix="infernux-texture-defaults-") as folder:
        project = Path(folder)
        for directory in ("Assets", "Packages", "ProjectSettings"):
            (project / directory).mkdir()
        shader = project / "Assets/DefaultProbe.frag"
        shader.write_text(SHADER.replace("DEFAULT", "white"), encoding="utf-8")
        texture = project / "Assets/AssignedRed.png"
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(texture)
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = DefaultProbePipeline()
        failures = []
        complete = False
        try:
            frontend.init_renderer(160, 120, folder)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(shader), database=database)
            assert imported, imported.error
            shader_guid = imported.guid
            imported = AssetManager.import_asset(str(texture), database=database)
            assert imported, imported.error
            texture_guid = imported.guid
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera = scene.create_game_object("Default Probe Camera")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera)
            cube = scene.create_primitive(inx.PrimitiveType.Cube)
            material = inx.Material.create_unlit("Default Texture Probe")
            material.frag_shader_name = "Declared Texture Default Probe"
            renderer = cube.get_component(inx.MeshRenderer)
            renderer.set_material(0, material)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            native.set_play_mode_rendering(True)
            phase = frame = changed = 0
            ticket = initial = None

            def after_draw():
                nonlocal phase, frame, changed, ticket, initial, complete
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        covered = pixels[..., 3] > .9
                        assert covered.sum() > 100, (phases[phase][0], int(covered.sum()))
                        expected = np.array(phases[phase][4])
                        display = np.where(expected <= .0031308, expected * 12.92,
                                           1.055 * expected ** (1 / 2.4) - .055)
                        measured = pixels[..., :3][covered]
                        np.testing.assert_allclose(measured, np.broadcast_to(display, measured.shape), atol=.005)
                        issues = [entry for entry in console._get_visible_log_snapshot(1000)
                                  if entry["level"] in ("ERROR", "FATAL", "WARN", "WARNING")]
                        assert not issues, issues
                        proof["phases"].append({"phase": phases[phase][0], "default": phases[phase][1],
                                                "assigned": phases[phase][2], "override": phases[phase][3],
                                                "covered": int(covered.sum()),
                                                "measured_display": measured.mean(axis=0).tolist(),
                                                "expected_display": display.tolist(), "shader_guid": shader_guid})
                        print("PASS " + phases[phase][0], flush=True)
                        if phase == 0:
                            initial = pixels
                        if phase == len(phases) - 1:
                            np.testing.assert_array_equal(pixels, initial)
                            complete = True
                            native.exit()
                            return
                        phase += 1
                        if phases[phase][1] != phases[phase - 1][1]:
                            shader.write_text(SHADER.replace("DEFAULT", phases[phase][1]), encoding="utf-8")
                            reimported = AssetManager.reimport_asset(str(shader), database=database)
                            assert reimported and reimported.guid == shader_guid, reimported.error
                        if phases[phase][2] != phases[phase - 1][2]:
                            assignment = phases[phase][2]
                            material.set_texture_guid("independentName", texture_guid if assignment == "asset" else assignment)
                        if phases[phase][3] != phases[phase - 1][3]:
                            override = phases[phase][3]
                            if override is None:
                                renderer.clear_parameters()
                            else:
                                renderer.set_parameter("independentName", override)
                        changed, ticket = frame, None
                    if frame > 400:
                        raise AssertionError("Declared texture default GPU test did not finish")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()
    proof["passed"] = True
    if args.proof:
        args.proof.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print("PASS declared texture defaults and assigned-asset shader reload", flush=True)


if __name__ == "__main__":
    main()
