"""Native layout metadata follows the geometry actually submitted by UIText."""

import pytest
import subprocess
import sys

from infernux.lib import TextLayout
from infernux.resources import engine_font_path
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def engine_font(engine):
    engine.set_gui_font(engine_font_path, 18)


def test_unicode_segmentation_is_available_without_a_renderer():
    code = '''from infernux.lib import TextLayout
source = "e\\u0301 👨‍👩‍👧 🇨🇳\\r\\n中👍🏽"
parts = TextLayout.split_graphemes(source)
assert parts == ["e\\u0301", " ", "👨‍👩‍👧", " ", "🇨🇳", "\\r\\n", "中", "👍🏽"], parts
assert "".join(parts) == source
assert TextLayout.split_graphemes("") == []
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_graphemes_source_words_and_lines_share_one_layout():
    source = "e\u0301 👨‍👩‍👧 🇨🇳\r\n中文"
    layout = TextLayout(source)
    characters = layout.characters
    encoded = source.encode("utf-8")
    assert [encoded[item.source_start:item.source_end].decode() for item in characters] == [
        "e\u0301", " ", "👨‍👩‍👧", " ", "🇨🇳", "\r\n", "中", "文",
    ]
    assert [(word.character_start, word.character_end) for word in layout.words] == [(0, 1), (2, 3), (4, 5), (6, 8)]
    assert len(layout.lines) == 2
    assert [item.line for item in characters] == [0] * 6 + [1, 1]
    assert characters[1].word is None and characters[5].word is None
    glyphs = layout.glyphs
    assert [item.codepoint for item in glyphs[characters[2].glyph_start:characters[2].glyph_end]] == [ord(c) for c in "👨‍👩‍👧"]
    assert layout.width > 0 and layout.height > layout.font_size


def test_letter_spacing_and_wrap_never_separate_a_grapheme():
    cluster = "e\u0301"
    plain = TextLayout(cluster)
    composed = TextLayout("é")
    assert plain.text == plain.display_text == cluster
    assert [glyph.codepoint for glyph in plain.glyphs] == [ord("é")]
    assert plain.width == pytest.approx(composed.width)
    assert plain.glyphs[0].source_start == 0 and plain.glyphs[0].source_end == 3
    spaced = TextLayout(cluster, letter_spacing=12.0)
    assert spaced.width == pytest.approx(plain.width)
    doubled = TextLayout(cluster * 2, letter_spacing=3.0)
    assert doubled.width == pytest.approx(plain.width * 2 + 3.0)
    wrapped = TextLayout(cluster * 3, wrap_width=plain.width + 1.0, letter_spacing=3.0)
    assert len(wrapped.lines) == 3
    glyphs = wrapped.glyphs
    for character in wrapped.characters:
        assert {item.line for item in glyphs[character.glyph_start:character.glyph_end]} == {character.line}
        assert character.bounds[2] >= character.bounds[0]


def test_layout_snapshots_remain_readable_after_another_layout_is_built():
    first = TextLayout("first")
    characters, glyphs = first.characters, first.glyphs
    del first
    second = TextLayout("second", font_size=36.0)
    assert len(characters) == 5 and second.height > 0
    assert glyphs[0].codepoint == ord("f")
    with pytest.raises(AttributeError):
        characters[0].line = 7


def test_rich_text_keeps_source_indices_baselines_and_graphemes():
    source = 'A<color="#ff000080">e\u0301<size=200%>中</size>Z</color><br>B'
    layout = TextLayout(source, rich_text=True)
    assert layout.text == source and layout.display_text == "Ae\u0301中Z\nB"
    encoded = source.encode("utf-8")
    assert [encoded[c.source_start:c.source_end].decode() for c in layout.characters] == [
        "A", "e\u0301", "中", "Z", "<br>", "B",
    ]
    glyphs = layout.glyphs
    accent = glyphs[layout.characters[1].glyph_start]
    assert accent.codepoint == ord("é")
    assert encoded[accent.source_start:accent.source_end].decode() == "e\u0301"
    large = glyphs[layout.characters[2].glyph_start]
    assert large.font_size == 36
    assert large.color == pytest.approx((1, 0, 0, 128 / 255))
    assert glyphs[-1].color == pytest.approx((1, 1, 1, 1))
    assert layout.lines[0].height == 36 and layout.lines[1].y == pytest.approx(36 * 1.2)
    assert len(layout.styles) == 5
    assert TextLayout("<size=no>literal</size>", rich_text=True).display_text == "<size=no>literal</size>"


def test_ui_rich_text_uses_its_parsed_layout_for_intrinsic_size(scene):
    from infernux.ui import UIText, TextResizeMode

    text = scene.create_game_object("Styled text").add_component(UIText)
    text.rich_text = True
    text.text = "<size=200%>Large</size><br>Small"
    text.resize_mode = TextResizeMode.AutoWidth
    text.resolve_text_layout(lambda *args: pytest.fail("Rich text must not measure literal markup"))
    layout = text.get_text_layout()
    assert text.get_resolved_size()[0] == pytest.approx(layout.width)
    text.resize_mode = TextResizeMode.AutoHeight
    text.resolve_text_layout(lambda *args: pytest.fail("Rich text must not measure literal markup"))
    assert text.get_resolved_size()[1] == pytest.approx(text.get_text_layout().height)
    assert text.create_character_buffer().character_count == len("Large\nSmall")


def test_character_views_retain_storage_and_reject_a_different_text():
    from infernux.lib import TextGeometry, TextGeometryLayer

    geometry = TextGeometry("AB")
    buffer = geometry.create_character_buffer()
    assert buffer.offsets.shape == (2, 2) and buffer.colors.shape == (2, 4, 4)
    assert np.asarray(buffer).dtype == np.float32
    assert np.all(buffer.scales == 1) and np.all(buffer.colors == 1)
    affine = buffer.affine_transforms
    assert affine.shape == (2, 3, 4) and buffer.vertex_depth_offsets.shape == (2, 4)
    assert buffer.color_overrides.shape == (2,) and np.all(buffer.color_overrides == 0)
    assert buffer.columns == 47
    assert affine.strides == (buffer.columns * 4, 16, 4)
    assert np.all(affine == np.eye(3, 4, dtype=np.float32))
    affine[:, 2, 3] = [3, 6]
    buffer.vertex_depth_offsets[0] = [1, 4, -2, 6]
    buffer.color_overrides[:] = [3, 2]
    assert np.asarray(buffer)[:, 41].tolist() == [3, 6]
    assert np.asarray(buffer)[0, 42:46].tolist() == [1, 4, -2, 6]
    assert np.asarray(buffer)[:, 46].tolist() == [3, 2]
    buffer.reset()
    assert np.all(affine == np.eye(3, 4, dtype=np.float32))
    assert np.all(buffer.vertex_depth_offsets == 0)
    assert np.all(buffer.color_overrides == 0)
    offsets = buffer.offsets
    offsets[:, 1] = [2, 4]
    assert np.asarray(buffer)[:, 1].tolist() == [2, 4]
    geometry.set_character_data(buffer, TextGeometryLayer.Effects)
    snapshot, revision = geometry.layout, geometry.layout_revision
    geometry.set_layout("AB", font_size=36)
    assert geometry.layout_revision > revision and snapshot.font_size == 18
    geometry.set_character_data(buffer)  # Font/width changes preserve character identity.
    geometry.set_layout("CD")
    with pytest.raises(ValueError, match="different text or geometry"):
        geometry.set_character_data(buffer)
    del buffer, geometry
    assert offsets[:, 1].tolist() == [2, 4]
    assert np.all(affine == np.eye(3, 4, dtype=np.float32))


def test_character_edits_preserve_layout_and_authored_fields(scene, engine, monkeypatch):
    from types import SimpleNamespace
    from infernux.ui import UIText
    from infernux.ui import ui_render_dispatch as dispatch
    from infernux.ui.ui_command_packets import _CommandGroup
    from infernux.ui.ui_texture_cache import UITextureCache

    text = scene.create_game_object("Animated text").add_component(UIText)
    sibling = scene.create_game_object("Static text").add_component(UIText)
    group = _CommandGroup([text, sibling], UITextureCache())
    group.dirty.clear()
    text.text = "A e\u0301 中文"
    text.resolve_text_layout(engine.get_screen_ui_renderer().measure_text)
    group.dirty.clear()
    layout_key = text._text_layout_key
    layout = text.get_text_layout()
    buffer = text.create_character_buffer()
    buffer.rotations[:] = 30
    text.set_character_data(buffer)
    assert text.get_text_layout() is layout and text._text_layout_key == layout_key
    assert group.dirty == {0}
    group.dirty.clear()
    text.max_visible_characters = 2
    assert text.max_visible_characters == 2 and group.dirty == {0}
    group.dirty.clear()
    text.max_visible_characters = 2
    assert not group.dirty
    assert "max_visible_characters" not in text._serialize_fields_document()
    text.color, text.opacity = [0.8, 0.4, 0.2, 0], 0.5
    material = {"color": [0.5, 0.8, 0.25, 0.5], "texture_guid": "", "signature": (None, 0), "_native": None}
    monkeypatch.setattr(dispatch, "material_visual_state", lambda *args: material)
    editor_calls, runtime_calls = [], []
    context = SimpleNamespace(draw_text_geometry=lambda *args: editor_calls.append(args))
    dispatch._editor_render_text(text, context, 0, 0, 64, 64, 1, lambda _: 0)
    assert editor_calls[0][5:9] == tuple(text.color)
    assert editor_calls[0][-2:] == (0.5, material["color"])
    renderer = SimpleNamespace(add_text_geometry=lambda *args: runtime_calls.append(args))
    dispatch._runtime_render_text(text, renderer, 1, 0, 0, 64, 64, 64, 64, 1, 1, 1, lambda _: 0)
    assert runtime_calls[0][6:10] == tuple(text.color) and runtime_calls[0][-1] == 0.5


def test_native_character_geometry_translates_rotates_hides_and_resets(engine, scene):
    from infernux.lib import TextGeometry, TextGeometryLayer, ScreenUIList, SceneManager, Vector3
    from infernux.engine.runtime_screen_ui_pipeline import RuntimeScreenUIRenderPipeline
    from infernux.renderstack import RenderStackPipeline

    camera = scene.create_game_object("Text geometry camera")
    camera.transform.position = Vector3(0, 0, -2)
    camera.add_component("Camera")
    geometry = TextGeometry("F", font_size=24)
    buffer = geometry.create_character_buffer()
    layer = ScreenUIList.Overlay
    tint = [1, 1, 1, 1]
    box = [16, 16, 60, 60]
    options = {"rotation": 0.0, "clip": False}

    class Submission:
        def submit(self):
            renderer = engine.get_screen_ui_renderer()
            renderer.begin_frame(64, 64)
            renderer.add_filled_rect(layer, 0, 0, 64, 64, 0, 0, 0, 1)
            renderer.add_text_geometry(layer, *box, geometry, *tint, **options)

    def capture(*, mask=True):
        ticket = None
        for _ in range(100):
            engine.tick(1 / 60)
            renderer = engine.get_screen_ui_renderer()
            if ticket is None and renderer.last_submitted_index_count(layer):
                ticket = engine.request_render_target_readback(True)
            elif ticket is not None and ticket.done:
                assert not ticket.error, ticket.error
                pixels = ticket.result_numpy()
                return np.max(pixels[:, :, :3], axis=2) > 0.2 if mask else pixels.astype(np.float32)
        pytest.fail("Text geometry readback did not complete")

    previous_samples = engine.get_msaa_samples()
    try:
        engine.set_game_camera_enabled(True)
        engine.resize_game_render_target(64, 64)
        engine.set_render_pipeline(RuntimeScreenUIRenderPipeline(Submission(), RenderStackPipeline()))
        original = capture()
        y, x = np.nonzero(original)
        assert len(x) > 10
        SceneManager.instance().mark_temporal_discontinuity()
        assert np.array_equal(capture(), original)
        frame = dict(engine.renderer_frame_snapshot)
        assert frame["game_render_view_id"] in frame["game_render_view_ids"]
        buffer.offsets[:, 0] = 10
        geometry.set_character_data(buffer)
        shifted = capture()
        assert np.nonzero(shifted)[1].mean() - x.mean() == pytest.approx(10, abs=0.6)
        buffer.reset()
        buffer.rotations[:] = 90
        geometry.set_character_data(buffer)
        rotated = capture()
        ry, rx = np.nonzero(rotated)
        assert rx.max() - rx.min() == pytest.approx(y.max() - y.min(), abs=2)
        assert ry.max() - ry.min() == pytest.approx(x.max() - x.min(), abs=2)
        second = geometry.create_character_buffer()
        second.offsets[:, 1] = 8
        geometry.set_character_data(second, TextGeometryLayer.Appearance)
        composed = capture()
        cy, cx = np.nonzero(composed)
        assert cy.mean() - ry.mean() == pytest.approx(8, abs=0.6)
        assert cx.mean() == pytest.approx(rx.mean(), abs=0.6)
        geometry.visible_characters = 0
        assert not capture().any()
        geometry.visible_characters = -1
        geometry.reset_character_data()
        assert np.array_equal(capture(), original)
        full = capture(mask=False)
        solid = full[:, :, :3].max(axis=2) > 0.95
        assert solid.any()
        buffer.reset()
        buffer.colors[:, :, :3] = (1, 0, 0)
        buffer.colors[:, :, 3] = 0.5
        geometry.set_character_data(buffer)
        tint[3] = 0.5
        faded = capture(mask=False)
        assert faded[:, :, 0][solid].mean() == pytest.approx(0.25, abs=0.025)
        assert np.max(faded[:, :, 1:3]) < 0.02
        geometry.set_layout("<color=#ff000080>F</color>", font_size=24, rich_text=True)
        tint[:] = [1, 1, 1, 1]
        styled = capture(mask=False)
        assert styled[:, :, 0][solid].mean() == pytest.approx(128 / 255, abs=0.025)
        assert np.max(styled[:, :, 1:3]) < 0.02
        override = geometry.create_character_buffer()
        override.colors[:] = [0, 1, 0, 0.8]
        override.color_overrides[:] = 3
        geometry.set_character_data(override)
        tint[3] = 0
        options["opacity"] = 0.5
        green = capture(mask=False)
        assert green[:, :, 1][solid].mean() == pytest.approx(0.4, abs=0.025)
        assert np.max(green[:, :, (0, 2)]) < 0.02
        geometry.reset_character_data()
        tint[3] = 1
        options["opacity"] = 1
        assert capture(mask=False)[:, :, 0][solid].mean() == pytest.approx(128 / 255, abs=0.025)
        geometry.set_layout("FF", font_size=26)
        distorted = geometry.create_character_buffer()
        distorted.rotations[:] = [25, -30]
        distorted.offsets[:, 1] = 4
        distorted.skews[:] = 0.3
        geometry.set_character_data(distorted)
        box[:] = [16, 16, 44, 34]
        options["rotation"] = 35.0
        uncut = capture()
        options["clip"] = True
        clipped = capture()
        py, px = np.indices(uncut.shape, dtype=np.float32)
        dx, dy = px + 0.5 - 30, py + 0.5 - 25
        angle = np.deg2rad(35)
        local_x = 30 + dx * np.cos(angle) + dy * np.sin(angle)
        local_y = 25 - dx * np.sin(angle) + dy * np.cos(angle)
        distance = np.minimum.reduce([local_x - 16, 44 - local_x, local_y - 16, 34 - local_y])
        assert np.count_nonzero(uncut & (distance > 1.5)) > 8
        assert np.array_equal(clipped[distance > 1.5], uncut[distance > 1.5])
        assert np.count_nonzero(uncut & (distance < -1.5)) > 8
        assert not clipped[distance < -1.5].any()
    finally:
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        engine.get_screen_ui_renderer().begin_frame(64, 64)
        engine.set_msaa_samples(previous_samples)
