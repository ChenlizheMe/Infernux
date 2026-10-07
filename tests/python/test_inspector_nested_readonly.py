"""Nested Inspector authoring preserves read-only data through real transactions."""
import pytest

from infernux.components import InxComponent, SerializableObject, serialized_field, FieldType
from infernux.components.fields import get_serialized_fields
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import UndoManager


class ReadonlyLeaf(SerializableObject):
    measured = serialized_field(default=7, readonly=True)
    editable = serialized_field(default=2)
    hidden = serialized_field(default=31, hidden=True)
    conditional = serialized_field(default=43, visible_when=lambda value: value.editable >= 10)


class ReadonlyGroup(SerializableObject):
    measured = serialized_field(default=11, readonly=True)
    editable = serialized_field(default=3)
    child = serialized_field(default_factory=ReadonlyLeaf)
    locked_child = serialized_field(default_factory=ReadonlyLeaf, readonly=True)


class ReadonlyOwner(InxComponent):
    data = serialized_field(default_factory=ReadonlyGroup)
    locked = serialized_field(default_factory=ReadonlyGroup, readonly=True)
    rows = serialized_field(default=[ReadonlyGroup()])
    locked_rows = serialized_field(default=[ReadonlyGroup()], readonly=True)
    empty = serialized_field(default=None, field_type=FieldType.SERIALIZABLE_OBJECT,
                             serializable_class=ReadonlyGroup, readonly=True)


class EditContext:
    semantic_capture_enabled = False

    def __init__(self):
        self.disabled = 0
        self.edits = []
        self.mutate = True

    def begin_disabled(self, disabled=True):
        assert disabled
        self.disabled += 1

    def end_disabled(self):
        self.disabled -= 1
        assert self.disabled >= 0

    def align_text_to_frame_padding(self): pass
    def label(self, *_args): pass
    def same_line(self, *_args): pass
    def set_next_item_width(self, *_args): pass
    def calc_text_width(self, value): return len(value) * 7.0
    def push_id_str(self, *_args): pass
    def pop_id(self): pass
    def begin_drag_drop_source(self, *_args): return False

    def drag_int(self, widget, value, *_args):
        self.edits.append((widget, bool(self.disabled)))
        # A disabled chrome result must also be excluded from the candidate document.
        return value + 1 if self.mutate else value


@pytest.fixture
def nested_authoring(engine, scene, monkeypatch, tmp_path):
    from infernux.engine.ui import _inspector_references as refs, _inspector_list_field as lists
    from infernux.engine.ui.igui import IGUI

    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(UndoManager, "_instance", None)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    database = engine.get_asset_database()
    core.project_assets.configure(database.project_root, database)
    files = SceneFileManager()
    files._asset_database = database
    owner = scene.create_game_object("Nested authoring").add_py_component(ReadonlyOwner())
    monkeypatch.setattr(refs, "render_compact_section_header", lambda *_a, **_kw: True)
    monkeypatch.setattr(lists, "render_compact_section_header", lambda *_a, **_kw: True)
    monkeypatch.setattr(IGUI, "list_header", lambda *_a, **_kw: True)
    monkeypatch.setattr(IGUI, "list_body_begin", lambda *_a: None)
    monkeypatch.setattr(IGUI, "list_body_end", lambda *_a: None)
    monkeypatch.setattr(IGUI, "list_item_remove_button", lambda *_a: False)
    monkeypatch.setattr(IGUI, "reorder_separator", lambda *_a: None)
    try:
        yield core, files, owner, tmp_path
    finally:
        core.shutdown()


def _render(ctx, owner, field):
    from infernux.engine.ui import _inspector_references as refs, _inspector_list_field as lists

    metadata = get_serialized_fields(type(owner))[field]
    renderer = lists._render_list_field if metadata.field_type == FieldType.LIST else refs._render_serializable_object_field
    renderer(ctx, owner, field, metadata, getattr(owner, field), 80)


def _value(owner, field):
    value = getattr(owner, field)
    return value[0] if isinstance(value, list) else value


@pytest.mark.parametrize("field", ["data", "rows", "locked", "locked_rows"])
def test_nested_readonly_is_preserved_in_committed_and_reopened_scene(nested_authoring, scene, field):
    from infernux.lib import SceneManager
    from infernux.engine.scene_document_transaction import SceneDocumentTransaction

    core, files, owner, folder = nested_authoring
    before = owner._serialize_fields_document()
    ctx = EditContext()
    _render(ctx, owner, field)
    value = _value(owner, field)
    locked = field.startswith("locked")
    assert value.measured == 11
    assert value.editable == (3 if locked else 4)
    assert value.child.measured == 7
    assert value.child.editable == (2 if locked else 3)
    assert value.child.hidden == 31 and value.child.conditional == 43
    assert value.locked_child.editable == 2
    assert value.locked_child.measured == 7
    assert ctx.disabled == 0
    assert ctx.edits
    for widget, disabled in ctx.edits:
        assert not widget.endswith(("_hidden", "_conditional"))
        if locked or "locked_child" in widget or widget.endswith("_measured"):
            assert disabled, widget
    if locked:
        assert owner._serialize_fields_document() == before
        assert core.action_journal.entries == ()
        assert not core.documents.require(files.document_id).is_dirty
    else:
        assert core.documents.require(files.document_id).is_dirty
        UndoManager.instance().undo()
        assert owner._serialize_fields_document() == before
        UndoManager.instance().redo()
        assert _value(owner, field).child.editable == 3
    path = folder / "readonly.scene"
    assert scene.save_to_file(str(path))
    saved = path.read_bytes()
    reopened = SceneManager.instance().create_scene("Readonly reopen")
    transaction = SceneDocumentTransaction(reopened, path=str(path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    restored = reopened.find("Nested authoring").get_py_component(ReadonlyOwner)
    restored_fields = restored._serialize_fields_document()
    original_fields = owner._serialize_fields_document()
    restored_fields.pop("__component_id__")
    original_fields.pop("__component_id__")
    assert restored_fields == original_fields
    assert reopened.save_to_file(str(path))
    assert path.read_bytes() == saved


def test_readonly_empty_object_is_not_initialized_by_rendering(nested_authoring):
    core, files, owner, _ = nested_authoring
    _render(EditContext(), owner, "empty")
    assert owner.empty is None
    assert core.action_journal.entries == ()
    assert not core.documents.require(files.document_id).is_dirty


@pytest.mark.parametrize("field", ["data", "rows"])
def test_runtime_can_update_readonly_values_and_inspector_uses_current_visibility(nested_authoring, field):
    core, _, owner, _ = nested_authoring
    value = _value(owner, field)
    value.measured = 87
    value.child.measured = 93
    value.child.editable = 20
    value.locked_child.measured = 99
    ctx = EditContext()
    _render(ctx, owner, field)
    value = _value(owner, field)
    assert value.measured == 87 and value.child.measured == 93
    assert value.locked_child.measured == 99
    assert value.child.conditional == 44
    assert any(widget.endswith("_child_conditional") and not disabled for widget, disabled in ctx.edits)
    assert core.action_journal.can_undo
    UndoManager.instance().undo()
    value = _value(owner, field)
    assert value.measured == 87 and value.child.measured == 93
    assert value.child.conditional == 43


@pytest.mark.parametrize("field", ["data", "rows", "locked", "locked_rows"])
def test_unchanged_nested_display_creates_no_history_or_document_revision(nested_authoring, field):
    core, files, owner, _ = nested_authoring
    before = owner._serialize_fields_document()
    revision = core.documents.require(files.document_id).revision
    ctx = EditContext()
    ctx.mutate = False
    _render(ctx, owner, field)
    assert owner._serialize_fields_document() == before
    assert core.action_journal.entries == ()
    assert core.documents.require(files.document_id).revision == revision


@pytest.mark.parametrize("field", ["data", "rows"])
def test_readonly_widget_exception_unwinds_disabled_scope_without_publication(nested_authoring, field):
    core, files, owner, _ = nested_authoring
    before = owner._serialize_fields_document()
    ctx = EditContext()

    def failed_widget(*_args):
        assert ctx.disabled > 0
        raise RuntimeError("readonly widget failed")

    ctx.drag_int = failed_widget
    with pytest.raises(RuntimeError, match="readonly widget failed"):
        _render(ctx, owner, field)
    assert ctx.disabled == 0
    assert owner._serialize_fields_document() == before
    assert core.action_journal.entries == ()
    assert not core.documents.require(files.document_id).is_dirty


def test_readonly_list_can_expand_without_exposing_mutation_actions(nested_authoring, monkeypatch):
    from infernux.engine.ui.igui import IGUI

    core, _, owner, _ = nested_authoring
    seen = []

    def header(*_args, **kwargs):
        seen.append(True)
        assert all(kwargs[key] is None for key in ("on_add", "on_remove", "accept_drop", "on_header_drop"))
        return True

    def unavailable(*_args, **_kwargs):
        raise AssertionError("readonly list cannot offer remove/reorder/drag actions")

    monkeypatch.setattr(IGUI, "list_header", header)
    monkeypatch.setattr(IGUI, "list_item_remove_button", unavailable)
    monkeypatch.setattr(IGUI, "reorder_separator", unavailable)
    ctx = EditContext()
    ctx.begin_drag_drop_source = unavailable
    _render(ctx, owner, "locked_rows")
    assert seen and ctx.edits and ctx.disabled == 0
    assert core.action_journal.entries == ()


def test_nested_readonly_asset_clear_cannot_enter_parent_candidate(monkeypatch):
    from infernux.core.asset_ref import TextureRef
    from infernux.components.fields import FieldMetadata, get_raw_field_value
    from infernux.engine.ui import _inspector_list_field as lists

    class ReadonlyAssetRow(SerializableObject):
        texture = serialized_field(default=TextureRef(guid="f" * 32), field_type=FieldType.TEXTURE, readonly=True)
        editable = serialized_field(default=2)

    def clear_disabled(context, *_args, **kwargs):
        assert context.disabled > 0
        kwargs["on_clear"]()

    monkeypatch.setattr(lists, "render_compact_section_header", lambda *_a, **_kw: True)
    monkeypatch.setattr(lists, "render_asset_reference_field", clear_disabled)
    row = ReadonlyAssetRow()
    values = [row]
    meta = FieldMetadata(name="rows", field_type=FieldType.LIST, default=[],
                         element_type=FieldType.SERIALIZABLE_OBJECT, element_class=ReadonlyAssetRow)
    ctx = EditContext()
    assert lists._render_serializable_list_item(ctx, "rows", 0, row, values, meta)
    assert values[0].editable == 3 and row.editable == 2
    assert get_raw_field_value(values[0], "texture").guid == "f" * 32
    assert ctx.disabled == 0
