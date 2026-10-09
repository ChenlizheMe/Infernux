"""Visible SVG/PNG comparison of imported GPU textures and direct source previews.

Run with the current native build. Writes an editor framebuffer capture and
checks actual ImGui draw commands, rather than merely checking texture IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from PIL import Image
from infernux import Engine
from infernux.core.asset_types import TextureImportSettings, TextureCompression, write_texture_import_settings
from infernux.lib import InxGUIRenderable, LogLevel, TextureLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    args.capture = args.capture.resolve()
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[1] / "native/fixtures/texture_vector.svg"
    with tempfile.TemporaryDirectory(prefix="infernux-svg-preview-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        svg = project / "Assets/Vector.svg"
        svg.write_bytes(source.read_bytes())
        png = project / "Assets/Raster.png"
        pixels = TextureLoader.load_from_file(str(svg), svg_max_size=512)
        Image.frombytes("RGBA", (pixels.width, pixels.height), pixels.get_pixels()).save(png)
        document = project / "Packages/demo/plugin_pages/Document.svg"
        document.parent.mkdir(parents=True)
        document.write_text('<svg width="321" height="123"><rect width="321" height="123" fill="#558899"/></svg>')
        uncatalogued = project / "Library/PackagePreviews/plugin_pages/Uninstalled.svg"
        uncatalogued.parent.mkdir(parents=True)
        uncatalogued.write_bytes(document.read_bytes())
        precise = uncatalogued.with_suffix(".png")
        Image.new("I;16", (321, 123), 32768).save(precise)
        repaired = uncatalogued.with_name("Repaired.svg")
        repaired.write_text("invalid image")
        frontend = Engine(engine_log_level=LogLevel.Warn)
        native = frontend.get_native_engine()
        errors, draws = [], {}
        capture_id = 0
        frame = 0
        try:
            frontend.init_renderer(1040, 760, str(project))
            native.set_window_title("Infernux - SVG texture acceptance")
            native.set_editor_fps_cap(60.)
            native.set_editor_idle_fps(0.)
            native.show()
            database = frontend.get_asset_database()
            result = database.import_asset(str(document))
            assert result, result.error
            for path in (svg, png):
                result = database.import_asset(str(path))
                assert result, result.error
                settings = TextureImportSettings(max_size=512, compression=TextureCompression.NONE)
                assert write_texture_import_settings(str(path), settings)
                result = database.reimport_asset(str(path))
                assert result, result.error

            class Preview(InxGUIRenderable):
                def on_render(self, ctx):
                    ctx.set_next_window_pos(12., 12., 1, 0., 0.)
                    ctx.set_next_window_size(1000., 950., 1)
                    ctx.begin_window("SVG / PNG - native texture previews", True, 0)
                    try:
                        for title, path, imported, edge in (
                            ("SVG imported texture - 512 x 256", svg, True, 512),
                            ("PNG imported texture - 512 x 256", png, True, 512),
                            ("SVG direct source preview - 128 x 64", svg, False, 128),
                        ):
                            ctx.label(title)
                            key = f"svg-check-{edge}-{path.suffix}|{path.as_posix()}"
                            texture_id, width, height = native.query_or_schedule_texture_preview(
                                key, str(path), path.stat().st_mtime_ns, srgb=True, max_size=edge,
                                texture_type="ui", texture_format="rgba8", use_imported_texture=imported)
                            if texture_id:
                                assert (width, height) == (edge, edge // 2), (title, width, height)
                                ctx.image(texture_id, 192., 96.)
                                draws[key.lower()] = (width, height)
                        for title, path in (("Installed document SVG", document),
                                            ("Uninstalled document SVG", uncatalogued),
                                            ("Uninstalled 16-bit PNG - neutral mid-gray", precise),
                                            ("Document repaired after decode failure", repaired)):
                            ctx.label(title)
                            key = f"document|{path.as_posix()}"
                            texture_id, width, height = native.query_or_schedule_texture_preview(
                                key, str(path), path.stat().st_mtime_ns, srgb=True, max_size=65536,
                                texture_type="ui", texture_format="auto", use_imported_texture=True)
                            if texture_id:
                                assert (width, height) == (321, 123), (title, width, height)
                                ctx.image(texture_id, 192., 74.)
                                draws[key.lower()] = (width, height)
                    except BaseException as error:
                        errors.append(repr(error))
                        native.exit()
                    finally:
                        ctx.end_window()

            panel = Preview()
            native.register_gui_renderable("svg_preview", panel)

            def after_draw():
                nonlocal capture_id, frame
                frame += 1
                if frame == 60:
                    repaired.write_bytes(document.read_bytes())
                snapshots = [s for s in native.preview_task_snapshots
                             if s["resource_key"].lower() in draws and s["imgui_draw_command_count"]]
                if len(snapshots) == 7 and frame > 90 and not capture_id:
                    for snapshot in snapshots:
                        if snapshot["resource_key"].lower().endswith(("uninstalled.svg", "uninstalled.png", "repaired.svg")):
                            assert snapshot["pixel_hash"] != 0, snapshot
                            assert snapshot["non_transparent_pixel_count"] == 321 * 123, snapshot
                    capture_id = native.request_capture("editor", str(args.capture))
                if capture_id:
                    result = native.query_capture(capture_id)
                    if result["status"] == "completed":
                        print(json.dumps({"previews": draws, "capture": result}))
                        native.exit()
                    elif result["status"] == "failed":
                        errors.append(repr(result))
                        native.exit()
                if frame >= 600:
                    errors.append("SVG preview/capture timeout: " + repr(native.preview_task_snapshots))
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            assert not errors, errors
            assert args.capture.is_file(), "missing visual evidence"
            print("SVG_PREVIEW_OK imported/source SVG, document SVG/16-bit PNG and repaired image all drawn")
        finally:
            native.set_post_draw_callback(None)
            native.unregister_gui_renderable("svg_preview")
            native.cleanup()


if __name__ == "__main__":
    main()
