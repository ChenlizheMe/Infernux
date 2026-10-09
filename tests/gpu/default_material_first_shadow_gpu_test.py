"""A fresh primitive's implicit DefaultLit must cast on its first color draw."""
from pathlib import Path
import tempfile

from infernux import Engine
from infernux.lib import ConsolePanel, LightShadows, PrimitiveType, SceneManager, Vector3
from infernux.renderstack import DefaultForwardPipeline


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="infernux-default-material-shadow-") as folder:
        project = Path(folder)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        failures = []
        completed = False
        callbacks = 0
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.resize_scene_render_target(64, 64)
            frontend.resize_game_render_target(64, 64)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera = scene.create_game_object("Camera")
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component("Camera")
            sun = scene.create_game_object("Sun")
            sun.transform.euler_angles = Vector3(0, 45, 0)
            sun.add_component("Light").shadows = LightShadows.Hard
            cube = scene.create_primitive(PrimitiveType.Cube, "First Cube")
            assert cube.get_component("MeshRenderer").get_material(0) is None
            pipeline = DefaultForwardPipeline()
            pipeline.shadow_resolution = 256
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)

            def after_draw():
                nonlocal callbacks, completed
                try:
                    callbacks += 1
                    frame = engine.renderer_frame_snapshot
                    if frame["game_draw_call_count"] > 0:
                        assert frame["game_shadow_draw_call_count"] > 0, frame
                        assert engine.gpu_residency_snapshot["shadow_material_descriptor_set_count"] > 0
                        diagnostics = console._get_visible_log_snapshot(1000)
                        missing = [entry for entry in diagnostics if "linked Shadow variant is unavailable" in entry["message"] or "EnsureShadowMaterialBinding:" in entry["message"]]
                        assert not missing, missing
                        completed = True
                        print("DefaultLit Cube casts shadows on its first color draw", flush=True)
                        engine.exit()
                    elif callbacks > 30:
                        raise AssertionError("Fresh Cube never produced a color draw")
                except BaseException as error:
                    failures.append(error)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            engine.cleanup()


if __name__ == "__main__":
    main()
