"""Committed font assets must refresh both native faces and text layout."""
from pathlib import Path
import shutil

import pytest

from infernux.core.asset_ref import create_asset_ref
from infernux.core.assets import AssetManager
from infernux.ui import UIText
from infernux.ui.enums import TextResizeMode
from model_test_support import remove_model_test_folder


FONTS = Path(__file__).resolve().parents[2] / "external/imgui_for_infernux/misc/fonts"
SAMPLE = "WWWWWWiiiiii"


@pytest.fixture
def font_assets(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    root = Path(database.assets_root) / tmp_path.name
    root.mkdir()
    yield database, root, engine.get_screen_ui_renderer()
    remove_model_test_folder(database, root)


def _import(database, path, filename):
    shutil.copyfile(FONTS / filename, path)
    result = AssetManager.import_asset(str(path), database=database)
    assert result, result.error
    return create_asset_ref("Font", guid=result.guid)


def test_same_guid_font_update_refreshes_native_and_intrinsic_layout(font_assets):
    database, root, renderer = font_assets
    path = root / "Shared.ttf"
    reference = _import(database, path, "Roboto-Medium.ttf")
    text = UIText()
    text.font, text.text, text.font_size, text.resize_mode = reference, SAMPLE, 24, TextResizeMode.AutoWidth
    text.resolve_text_layout(renderer.measure_text)
    before = text.get_resolved_size()[0]
    expected = renderer.measure_text(SAMPLE, 24, 0, str(FONTS / "Cousine-Regular.ttf"))[0]
    assert abs(before-expected) > 5
    epoch = renderer.command_packet_epoch()
    shutil.copyfile(FONTS / "Cousine-Regular.ttf", path)
    result = AssetManager.reimport_asset(str(path), database=database)
    assert result and result.guid == reference.guid
    assert renderer.command_packet_epoch()[0] > epoch[0]
    text.resolve_text_layout(renderer.measure_text)  # Normal cache path, not forced measurement.
    assert text.get_resolved_size()[0] == pytest.approx(expected, abs=.01)
    assert renderer.measure_text(SAMPLE, 24, 0, str(path))[0] == pytest.approx(expected, abs=.01)


def test_missing_font_arrival_delete_restore_and_move(font_assets):
    database, root, renderer = font_assets
    path = root / "Arriving.ttf"
    assert renderer.measure_text(SAMPLE, 24, 0, str(path)) == (0., 0.)
    reference = _import(database, path, "Roboto-Medium.ttf")
    expected = renderer.measure_text(SAMPLE, 24, 0, str(FONTS / "Roboto-Medium.ttf"))[0]
    assert renderer.measure_text(SAMPLE, 24, 0, str(path))[0] == pytest.approx(expected)
    text = UIText()
    text.font, text.text, text.font_size, text.resize_mode = reference, SAMPLE, 24, TextResizeMode.AutoWidth
    text.resolve_text_layout(renderer.measure_text)
    sidecar = Path(str(path)+".meta").read_bytes()
    path.unlink()  # Watcher publication follows the external filesystem edit.
    assert AssetManager.delete_asset(str(path), database=database)
    with pytest.raises(FileNotFoundError, match=reference.guid):
        text.resolve_text_layout(renderer.measure_text)
    assert renderer.measure_text(SAMPLE, 24, 0, str(path)) == (0., 0.)
    Path(str(path)+".meta").write_bytes(sidecar)
    restored = _import(database, path, "Cousine-Regular.ttf")
    assert restored.guid == reference.guid
    expected = renderer.measure_text(SAMPLE, 24, 0, str(FONTS / "Cousine-Regular.ttf"))[0]
    text.resolve_text_layout(renderer.measure_text)
    assert text.get_resolved_size()[0] == pytest.approx(expected)
    renamed = root / "Renamed.ttf"
    path.rename(renamed)
    moved = AssetManager.move_asset(str(path), str(renamed), database=database)
    assert moved, moved.error
    text.resolve_text_layout(renderer.measure_text)
    assert text.get_resolved_size()[0] == pytest.approx(expected)
    assert renderer.measure_text(SAMPLE, 24, 0, str(path)) == (0., 0.)


def test_fallback_face_update_invalidates_text_measurement(font_assets):
    database, root, renderer = font_assets
    primary = _import(database, root / "ASCII.ttf", "ProggyClean.ttf")
    fallback = _import(database, root / "Fallback.ttf", "Roboto-Medium.ttf")
    text = UIText()
    text.font, text.fallback_fonts = primary, [fallback]
    text.text, text.font_size, text.resize_mode = "ΩΩΩΩΩΩ", 24, TextResizeMode.AutoWidth
    text.resolve_text_layout(renderer.measure_text)
    before = text.get_resolved_size()[0]
    shutil.copyfile(FONTS / "Cousine-Regular.ttf", root / "Fallback.ttf")
    assert AssetManager.reimport_asset(str(root / "Fallback.ttf"), database=database)
    expected = renderer.measure_text(text.text, 24, 0, str(root / "ASCII.ttf"), 1.2, 0,
                                     [str(FONTS / "Cousine-Regular.ttf")])[0]
    assert abs(before-expected) > 1
    text.resolve_text_layout(renderer.measure_text)
    assert text.get_resolved_size()[0] == pytest.approx(expected, abs=.01)


def test_filesystem_watcher_publishes_font_revision(font_assets, engine, monkeypatch):
    import threading
    from watchdog.observers import Observer
    from infernux.engine.resources_manager import ResourceChangeHandler

    database, root, renderer = font_assets
    path = root / "Watched.ttf"
    reference = _import(database, path, "Roboto-Medium.ttf")
    text = UIText()
    text.font, text.text, text.font_size, text.resize_mode = reference, SAMPLE, 24, TextResizeMode.AutoWidth
    text.resolve_text_layout(renderer.measure_text)
    observed = threading.Event()

    class Handler(ResourceChangeHandler):
        def on_modified(self, event):
            super().on_modified(event)
            if not event.is_directory and Path(event.src_path) == path:
                observed.set()

    handler = Handler(engine, project_path=str(root.parent.parent))
    observer = Observer()
    observer.schedule(handler, str(root), recursive=False)
    observer.start()
    try:
        shutil.copyfile(FONTS / "Cousine-Regular.ttf", path)
        assert observed.wait(5), "Native filesystem watcher did not observe the committed font write"
        handler.process_pending_reloads(force=True)
    finally:
        observer.stop()
        observer.join(5)
    assert not observer.is_alive()
    text.resolve_text_layout(renderer.measure_text)
    expected = renderer.measure_text(SAMPLE, 24, 0, str(FONTS / "Cousine-Regular.ttf"))[0]
    assert text.get_resolved_size()[0] == pytest.approx(expected)
    # Warm text layout does not resolve or stat the font again.
    import infernux.ui.ui_font_asset as fonts

    monkeypatch.setattr(fonts, "ui_font_paths", lambda *_: pytest.fail("warm layout re-resolved font files"))
    for _ in range(30):
        assert not text.resolve_text_layout(renderer.measure_text)


def test_live_gui_releases_changed_fonts_after_submitted_frames(font_assets, engine, scene):
    from infernux.lib import InxGUIRenderable
    from infernux.renderstack import RenderStackPipeline

    database, root, renderer = font_assets
    path = root / "Live.ttf"
    _import(database, path, "Roboto-Medium.ttf")
    names = ("Roboto-Medium.ttf", "Cousine-Regular.ttf")
    widths = [renderer.measure_text(SAMPLE, 24, 0, str(FONTS / name))[0] for name in names]
    assert abs(widths[0]-widths[1]) > 5
    samples, errors = [], []
    phase, frames = 0, 0

    class Panel(InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0, 0, 1, 0, 0)
            ctx.set_next_window_size(64, 64, 1)
            ctx.begin_window("Font revision GPU fixture", True, 0)
            try:
                samples.append(ctx.calc_text_size(SAMPLE, 24, str(path))[0])
                ctx.draw_text_ex_aligned(8, 29, 60, 62, SAMPLE, 1, 1, 1, 1,
                                         font_size=24, font_path=str(path))
            finally:
                ctx.end_window()

    panel = Panel()
    pipeline = RenderStackPipeline()

    def after_draw():
        nonlocal phase, frames
        try:
            frames += 1
            assert frames < 100, "GUI did not publish the requested font frames"
            if not samples or frames % 2:
                return
            assert samples[-1] == pytest.approx(widths[phase % 2], abs=.01)
            if phase == 8:
                engine.exit()
                return
            phase += 1
            shutil.copyfile(FONTS / names[phase % 2], path)
            assert AssetManager.reimport_asset(str(path), database=database)
        except BaseException as error:
            errors.append(error)
            engine.exit()

    try:
        engine.set_render_pipeline(pipeline)
        engine.register_gui_renderable("font-reload-test", panel, 0)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert phase == 8 and len(samples) >= 9
    finally:
        engine.set_post_draw_callback(None)
        engine.unregister_gui_renderable("font-reload-test")
        engine.set_render_pipeline(None)
