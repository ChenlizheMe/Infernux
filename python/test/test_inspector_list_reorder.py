"""Inspector drag ordering obeys ImGui's type limit and list ownership."""
from types import SimpleNamespace

import pytest

from infernux.components.fields import FieldType
from infernux.engine.ui import _inspector_list_field as lists
from infernux.engine.ui.igui import IGUI


class DragContext:
    def __init__(self, source_index):
        self.source_index = source_index
        self.index = None
        self.payload = None
        self.capturing = True

    def push_id_str(self, value):
        self.index = int(value.rsplit('_', 1)[1])

    def pop_id(self):
        self.index = None

    def same_line(self, *args):
        pass

    def begin_drag_drop_source(self, flags):
        return self.capturing and self.index == self.source_index

    def set_drag_drop_payload(self, kind, value):
        # ImGuiPayload.DataType is a fixed 33-byte C string. Release
        # truncation and Debug assertion must never decide list ownership.
        assert len(kind.encode('utf-8')) <= 32
        self.payload = (kind, value)

    set_drag_drop_payload_str = set_drag_drop_payload

    def label(self, value):
        pass

    def end_drag_drop_source(self):
        pass


@pytest.mark.parametrize('source_index,target_index,foreign_owner,foreign_field,expected', [
    (1, 0, False, False, [20, 10, 30]),
    (1, 3, False, False, [10, 30, 20]),
    (0, 1, False, False, [10, 20, 30]),
    (1, 0, True, False, [10, 20, 30]),
    (1, 0, False, True, [10, 20, 30]),
])
def test_reorder_preserves_values_and_rejects_other_lists(
    monkeypatch, source_index, target_index, foreign_owner, foreign_field, expected,
):
    from infernux.engine.ui import inspector_utils

    ctx = DragContext(source_index)
    owner = SimpleNamespace()
    field = 'effect_slots_after_opaque_with_a_deliberately_long_name_渲染'
    target = 'before_0' if target_index == 0 else f'after_{target_index-1}'

    def separator(context, name, kind, callback):
        if not context.capturing and name.endswith(target) and kind == context.payload[0]:
            callback(context.payload[1])

    monkeypatch.setattr(IGUI, 'list_body_begin', lambda *_: None)
    monkeypatch.setattr(IGUI, 'list_body_end', lambda *_: None)
    monkeypatch.setattr(IGUI, 'list_item_remove_button', lambda *_: False)
    monkeypatch.setattr(IGUI, 'reorder_separator', separator)
    monkeypatch.setattr(inspector_utils, 'render_serialized_field', lambda *args: args[4])
    # The renderer uses dataclasses.replace to construct element metadata.
    from infernux.components.fields import FieldMetadata
    metadata = FieldMetadata(name=field, field_type=FieldType.LIST, default=[], element_type=FieldType.INT)
    items = [10, 20, 30]

    def render(component, name):
        return lists._render_list_items_body(
            ctx, component, name, metadata, items, FieldType.INT, set(), 2., list(items),
        )

    assert not render(owner, field)
    assert ctx.payload is not None
    ctx.capturing = False
    changed = render(SimpleNamespace() if foreign_owner else owner,
                     field + '_other' if foreign_field else field)
    assert items == expected
    assert changed == (expected != [10, 20, 30])
