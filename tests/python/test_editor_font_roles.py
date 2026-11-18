"""Editor typography must not replace the default game text face."""

from pathlib import Path

import pytest

from infernux import resources
from infernux.lib import TextLayout


def test_runtime_and_interface_faces_are_explicit():
    assert resources.engine_font_path == resources.engine_mono_font_path
    assert Path(resources.engine_font_path).name == "MapleMono-CN-Regular.ttf"
    assert Path(resources.engine_ui_font_path).name == "Inter-VF.ttf"
    assert Path(resources.engine_font_cjk_path).name == "NotoSansSC-Regular.ttf"
    assert all(Path(path).is_file() for path in (
        resources.engine_font_path, resources.engine_ui_font_path, resources.engine_font_cjk_path,
    ))


def test_font_family_install_keeps_scene_text_monospaced(engine):
    engine.set_gui_fonts(resources.engine_ui_font_path, resources.engine_font_cjk_path,
                         resources.engine_mono_font_path, 16)
    assert TextLayout("iii").width == pytest.approx(TextLayout("WWW").width)
    assert TextLayout("中").width == pytest.approx(TextLayout("AA").width)
