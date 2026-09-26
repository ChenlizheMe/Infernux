"""Real Vulkan regression for model Index Format publication and binding.

The same imported geometry is rendered first with UInt16 and then hot-reloaded
as UInt32.  The test observes the actual object-owned Vulkan index allocation,
captures both frames, and requires pixel-identical output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from Infernux import Engine
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.core.assets import AssetManager
from Infernux.lib import InxMaterial, SceneManager, Vector3


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
            frontend.resize_game_render_target(64, 64)

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
            camera_owner = scene.create_game_object("IndexFormatCamera")
            camera_owner.transform.position = Vector3(0.0, 0.0, -4.0)
            camera_owner.add_component("Camera")
            engine.set_game_camera_enabled(True)

            light_owner = scene.create_game_object("IndexFormatLight")
            light_owner.add_component("Light")

            probe = scene.create_game_object("IndexFormatProbe")
            renderer = probe.add_component("MeshRenderer")
            renderer.set_mesh_asset_guid(imported.guid)
            renderer.set_material(0, InxMaterial.create_default_lit())

            frames = 0
            stage = "wait16"
            first_ticket = None
            second_ticket = None
            first_pixels = None
            switched_frame = 0

            def after_draw() -> None:
                nonlocal frames, stage, first_ticket, second_ticket
                nonlocal first_pixels, switched_frame, completed
                try:
                    frames += 1
                    if stage == "wait16" and frames >= 8:
                        info = engine.get_object_mesh_index_gpu_info(probe.id)
                        assert info == {"format": "uint16", "bytes": 12}, info
                        first_ticket = frontend.request_render_target_readback(True)
                        stage = "capture16"
                    elif stage == "capture16" and first_ticket.done:
                        first_pixels = first_ticket.result_numpy().copy()
                        assert np.isfinite(first_pixels).all()
                        settings.index_format = "uint32"
                        result = AssetManager.reimport_asset(
                            str(source), import_settings=settings.to_dict(), database=database
                        )
                        assert result, result.error
                        switched_frame = frames
                        stage = "wait32"
                    elif stage == "wait32" and frames >= switched_frame + 4:
                        info = engine.get_object_mesh_index_gpu_info(probe.id)
                        if info["format"] == "uint32":
                            assert info["bytes"] == 24, info
                            second_ticket = frontend.request_render_target_readback(True)
                            stage = "capture32"
                    elif stage == "capture32" and second_ticket.done:
                        second_pixels = second_ticket.result_numpy()
                        assert first_pixels is not None
                        np.testing.assert_array_equal(second_pixels, first_pixels)
                        assert np.any(first_pixels[..., :3] != first_pixels[0, 0, :3])
                        completed = True
                        print(
                            "mesh index format GPU regression: UInt16=12 bytes, "
                            "UInt32=24 bytes, pixel output identical"
                        )
                        engine.exit()
                    if frames >= 180 and not completed:
                        raise AssertionError(f"index format GPU test timed out in {stage}")
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
