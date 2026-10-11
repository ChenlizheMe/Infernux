"""Document illustrations must survive unrelated game texture import settings."""
import subprocess
import sys
import textwrap


def test_registered_document_image_preserves_source_pixels(tmp_path):
    script = textwrap.dedent('''
        import json, sys
        from pathlib import Path
        from PIL import Image
        from infernux import Engine
        from infernux.core.asset_types import TextureImportSettings, write_texture_import_settings
        from infernux.engine.path_utils import resolved_path
        from infernux.lib import InxGUIRenderable

        project = Path(sys.argv[1])
        (project / 'Assets').mkdir()
        (project / 'ProjectSettings').mkdir()
        path = project / 'Assets/diagram.png'
        image = Image.new('RGBA', (128, 64))
        image.putdata([(x % 251, y % 241, (x * 3 + y * 7) % 253,
                        0 if x < 8 and y < 8 else 255)
                       for y in range(64) for x in range(128)])
        image.save(path)
        source = image.tobytes()
        expected_hash = 1469598103934665603
        for value in source:
            expected_hash = ((expected_hash ^ value) * 1099511628211) & 0xffffffffffffffff

        frontend = Engine()
        native = frontend.get_native_engine()
        frames = 0
        errors, records = [], []
        key = 'document|' + Path(resolved_path(str(path))).as_posix()
        if sys.platform == 'win32':
            key = key.lower()
        try:
            frontend.init_renderer(256, 128, str(project))
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database = frontend.get_asset_database()
            result = database.import_asset(str(path))
            assert result, result.error
            settings = TextureImportSettings(max_size=32, srgb=False)
            assert write_texture_import_settings(str(path), settings)
            result = database.reimport_asset(str(path))
            assert result, result.error
            before = Path(str(path) + '.meta').read_bytes()
            metadata = json.loads(before)['metadata']
            assert metadata['artifact_width']['value'] == 32

            class Illustration(InxGUIRenderable):
                def on_render(self, ctx):
                    visible = ctx.begin_window('Document pixels###document_probe', True, 0)
                    try:
                        texture_id, width, height = native.query_or_schedule_texture_preview(
                            key, str(path), path.stat().st_mtime_ns, srgb=True,
                            max_size=65536, texture_format='rgba8', texture_type='ui',
                            use_imported_texture=False,
                        )
                        if texture_id:
                            assert (width, height) == (128, 64)
                            if visible:
                                ctx.image(texture_id, 128., 64.)
                    except BaseException as error:
                        errors.append(repr(error))
                        native.exit()
                    finally:
                        ctx.end_window()

            panel = Illustration()
            native.register_gui_renderable('document_probe', panel)

            def after_draw():
                global frames
                frames += 1
                tasks = [item for item in native.preview_task_snapshots if item['resource_key'] == key]
                if tasks and tasks[0]['texture_id'] and tasks[0]['imgui_draw_command_count']:
                    records.append(tasks[0])
                    native.exit()
                elif frames >= 240:
                    errors.append('document upload/draw timed out: ' + repr(native.preview_task_snapshots))
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            assert not errors, errors
            assert records, 'source illustration never reached an ImGui draw'
            snapshot = records[0]
            assert snapshot['pixel_hash'] == expected_hash, snapshot
            assert snapshot['non_transparent_pixel_count'] == 128 * 64 - 8 * 8
            assert not snapshot['authoring']
            assert Path(str(path) + '.meta').read_bytes() == before
            print('PASS registered document source pixels, alpha, ImGui draw and unchanged asset settings')
        finally:
            native.set_post_draw_callback(None)
            native.unregister_gui_renderable('document_probe')
            native.cleanup()
    ''')
    project = tmp_path / "Document images"
    project.mkdir()
    result = subprocess.run([sys.executable, "-c", script, str(project)],
                            capture_output=True, text=True, encoding="utf-8", timeout=90)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "PASS registered document source pixels" in output
    assert not any(marker in output for marker in ("VUID-", "SYNC-HAZARD", "Validation Error"))
