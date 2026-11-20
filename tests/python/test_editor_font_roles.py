"""Editor and gameplay share the bundled Latin and CJK font."""

from pathlib import Path

import pytest

from infernux import resources
from infernux.lib import TextLayout


def test_editor_and_player_share_one_bundled_face():
    font = Path(resources.engine_font_path)
    assert font.name == "MapleMono-CN-Regular.ttf"
    assert list(font.parent.glob("*.ttf")) == [font]
    assert resources.engine_ui_font_path == resources.engine_font_path
    assert resources.engine_font_cjk_path == resources.engine_font_path
    assert resources.engine_mono_font_path == resources.engine_font_path


def test_bundled_font_aligns_latin_and_chinese_columns(engine):
    engine.set_gui_fonts(resources.engine_ui_font_path, resources.engine_font_cjk_path,
                         resources.engine_mono_font_path, 16)
    assert TextLayout("iii").width == pytest.approx(TextLayout("WWW").width)
    assert TextLayout("中").width == pytest.approx(TextLayout("AA").width)
