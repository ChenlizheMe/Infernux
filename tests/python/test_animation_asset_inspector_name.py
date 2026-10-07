"""2D clip names come from their standalone asset, without 3D take state."""
from types import SimpleNamespace

import pytest

from infernux.core.animation_clip import AnimationClip
from infernux.engine.ui.asset_details_renderer import _render_animclip_body


class _NameRendered(Exception):
    pass


class _NameFieldContext:
    """Stop this focused paint test at the first read-only name control."""
    def calc_text_width(self, text):
        return len(text) * 8.0

    def dummy(self, *args):
        pass

    def align_text_to_frame_padding(self):
        pass

    def label(self, text):
        pass

    def same_line(self, offset=0.0):
        pass

    def set_next_item_width(self, width):
        pass

    def begin_disabled(self, disabled):
        assert disabled

    def text_input(self, field, value, length):
        assert field == "##animclip_name"
        self.name = value
        raise _NameRendered


@pytest.mark.parametrize("path,expected", [
    ("Assets/Renamed.animclip2d", "Renamed"),
    ("", "ClipModelName"),
])
def test_animclip_inspector_renders_standalone_name(path, expected):
    context = _NameFieldContext()
    state = SimpleNamespace(settings=AnimationClip(name="ClipModelName"), file_path=path)
    with pytest.raises(_NameRendered):
        _render_animclip_body(context, None, state)
    assert context.name == expected


def test_animclip_inspector_rejects_non_clip_before_name_field():
    context = _NameFieldContext()
    _render_animclip_body(context, None, SimpleNamespace(settings=None))
    assert not hasattr(context, "name")
