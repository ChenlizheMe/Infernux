"""A newly linked material must have shadow bindings on its first draw."""
from __future__ import annotations

import tempfile
from pathlib import Path

from infernux import Engine
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, LightShadows, SceneManager, Vector3
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline, MSAASamples


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-first-shadow-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        mesh = assets / "Probe.obj"
        mesh.write_text("v -1 -1 0\nv 1 -1 0\nv 0 1 0\nvn 0 0 -1\nvn 0 0 1\nf 1//1 2//1 3//1\nf 3//2 2//2 1//2\n", encoding="ascii")
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        completed = False
        failures = []
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            frontend.resize_game_render_target(64, 64)
            result = AssetManager.import_asset(str(mesh), database=frontend.get_asset_database())
            assert result, result.error
            scene = SceneManager.instance().get_active_scene()
            camera = scene.create_game_object("Camera")
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component("Camera")
            sun = scene.create_game_object("Sun")
            light = sun.add_component("Light")
            light.shadows = LightShadows.Hard
            probe = scene.create_game_object("Probe")
            renderer = probe.add_component("MeshRenderer")
            renderer.set_mesh_asset_guid(result.guid)
            renderer.enabled = False
            pipeline = DefaultForwardPipeline()
            pipeline.msaa_samples = MSAASamples.OFF
            pipeline.shadow_resolution = 256
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)
            frame = 0
            initial_descriptors = 0

            def after_draw():
                nonlocal frame, initial_descriptors, completed
                try:
                    frame += 1
                    if frame == 2:
                        initial_descriptors = engine.gpu_residency_snapshot["shadow_material_descriptor_set_count"]
                        fragment = assets / "FirstShadow.frag"
                        fragment.write_text('''#version 450
ShaderInfo {
    Name "First Shadow Surface"
    ShadingModel Unlit
    AlphaClip 0.5
    Properties {
        Color baseColor = [1.0, 0.2, 0.1, 1.0]
        Texture2D mainTexture = white
    }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    vec4 texel = sampleAlbedoAlpha(mainTexture);
    s.albedo = texel.rgb * material.baseColor.rgb;
    s.alpha = texel.a * material.baseColor.a;
}
''', encoding="utf-8")
                        imported = AssetManager.import_asset(str(fragment), database=frontend.get_asset_database())
                        assert imported, imported.error
                        material = InxMaterial.create_default_unlit()
                        material.frag_shader_name = "First Shadow Surface"
                        renderer.set_material(0, material)
                        renderer.enabled = True
                    elif frame == 3:
                        diagnostics = console._get_visible_log_snapshot(1000)
                        shadow_failures = [entry for entry in diagnostics if "EnsureShadowMaterialBinding:" in entry["message"] or "linked Shadow variant is unavailable" in entry["message"]]
                        assert not shadow_failures, shadow_failures
                        descriptors = engine.gpu_residency_snapshot["shadow_material_descriptor_set_count"]
                        assert descriptors > initial_descriptors, (initial_descriptors, descriptors)
                        completed = True
                        print("First material draw: linked shadow resources published without a missing-binding frame")
                        engine.exit()
                    if frame > 30:
                        raise AssertionError("First shadow test did not complete")
                except BaseException as exception:
                    failures.append(exception)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
            return 0
        finally:
            engine.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
