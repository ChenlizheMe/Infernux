"""Real Vulkan regression for model Index Format publication and binding.

The same imported geometry is rendered first with UInt16 and then hot-reloaded
as UInt32.  The test observes the actual object-owned Vulkan index allocation,
captures both frames, and requires pixel-identical output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import monotonic

import numpy as np

from infernux import Engine
from infernux.core.asset_types import read_mesh_import_settings
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags, InxMaterial, SceneManager, Vector3
from infernux.renderstack import DefaultForwardPipeline


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-index-format-gpu-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        source = assets / "IndexFormatProbe.obj"
        # Two opposite windings make the flat probe visible regardless of the
        # renderer's front-face convention while retaining one compact index stream.
        source.write_text(
            "v -1 -1 0\n"
            "v 1 -1 0\n"
            "v 0 1 0\n"
            "vn 0 0 -1\n"
            "vn 0 0 1\n"
            "f 1//1 2//1 3//1\n"
            "f 3//2 2//2 1//2\n",
            encoding="ascii",
        )

        frontend = Engine()
        engine = frontend.get_native_engine()
        failures: list[BaseException] = []
        completed = False
        try:
            try:
                frontend.init_renderer(64, 64, str(project))
            except (OSError, RuntimeError) as exception:
                print(f"mesh index format GPU test skipped: {exception}")
                return 77
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            engine.set_scene_view_visible(False)
            frontend.resize_game_render_target(64, 64)
            frontend.set_render_pipeline(DefaultForwardPipeline())

            database = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(source), database=database)
            assert imported, imported.error
            settings = read_mesh_import_settings(str(source))
            settings.index_format = "uint16"
            applied = AssetManager.reimport_asset(
                str(source), import_settings=settings.to_dict(), database=database
            )
            assert applied, applied.error

            scene = SceneManager.instance().get_active_scene()
            assert scene is not None
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera_owner = scene.create_game_object("IndexFormatCamera")
            camera_owner.transform.position = Vector3(0.0, 0.0, -4.0)
            camera = camera_owner.add_component("Camera")
            camera.clear_flags = CameraClearFlags.SolidColor
            engine.set_game_camera_enabled(True)

            probe = scene.create_game_object("IndexFormatProbe")
            renderer = probe.add_component("MeshRenderer")
            renderer.set_mesh_asset_guid(imported.guid)
            renderer.set_material(0, InxMaterial.create_default_unlit())

            deadline = monotonic() + 15.0
            format_observed_at = None
            ticket = None
            first_pixels = None

            def after_frame(_foreground_ready: bool) -> None:
                nonlocal format_observed_at, ticket, first_pixels, completed
                try:
                    expected_format, expected_bytes = ("uint16", 12) if first_pixels is None else ("uint32", 24)
                    frame = engine.renderer_frame_snapshot
                    execution = frame["game_render_graph_execution_count"]
                    if ticket is None:
                        try:
                            info = engine.get_object_mesh_index_gpu_info(probe.id)
                        except IndexError:
                            info = None
                        if info is not None and info["format"] == expected_format:
                            assert info["bytes"] == expected_bytes, info
                            # The allocation can appear before a Game draw. Capture
                            # only after Game accepts a later execution using it.
                            if format_observed_at is None:
                                format_observed_at = execution
                            elif execution > format_observed_at and frame["game_draw_call_count"] > 0:
                                ticket = frontend.request_render_target_readback(True)
                    elif ticket.done:
                        pixels = ticket.result_numpy()
                        assert np.isfinite(pixels).all()
                        assert pixels.shape == (64, 64, 4), pixels.shape
                        assert np.all(pixels[32, 32, :3] > 0.5), (expected_format, execution, pixels[32, 32])
                        assert np.any(pixels[..., :3] != pixels[0, 0, :3])
                        if first_pixels is None:
                            first_pixels = pixels.copy()
                            settings.index_format = "uint32"
                            result = AssetManager.reimport_asset(
                                str(source), import_settings=settings.to_dict(), database=database
                            )
                            assert result, result.error
                            format_observed_at = None
                            ticket = None
                        else:
                            np.testing.assert_array_equal(pixels, first_pixels)
                            completed = True
                            print(
                                "mesh index format GPU regression: UInt16=12 bytes, "
                                "UInt32=24 bytes, pixel output identical"
                            )
                            engine.exit()
                    if not completed and monotonic() >= deadline:
                        raise AssertionError(
                            f"Timed out waiting for {expected_format} Game output. "
                            f"Format seen at {format_observed_at}, Game execution {execution}."
                        )
                except BaseException as exception:
                    failures.append(exception)
                    engine.exit()

            engine.set_end_frame_callback(after_frame)
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
