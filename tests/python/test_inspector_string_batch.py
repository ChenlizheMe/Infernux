"""Actual Python GUI bindings must never manufacture truncated string edits."""
from types import SimpleNamespace

from infernux import lib
from infernux.components.fields import FieldType
from infernux.engine.ui.inspector_utils import build_scalar_desc
from test_property_transaction_scene_owners import resident_authors


def test_python_text_bindings_and_cached_inspector_batches_keep_full_values(engine):
    values = ["", "a" * 255, "b" * 4095, "c" * 5000, "解谜对话" * 1500, "0123456789abcdef" * 2]
    cases = [(value, multiline) for value in values for multiline in (False, True)]
    observed = {"frames": 0, "renders": 0, "errors": []}

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(600., 400., 0)
            opened = ctx.begin_window("String###string_contract", True, 0)
            try:
                if not opened or observed["renders"] >= len(cases):
                    return
                value, multiline = cases[observed["renders"]]
                metadata = SimpleNamespace(field_type=FieldType.STRING, range=None,
                                           slider=False, drag_speed=None, multiline=multiline, tooltip="")
                descriptor = build_scalar_desc("##string", "Dialogue", metadata, value)
                plan = ctx.create_property_batch_plan([descriptor])
                for route in ("plan", "values", "immediate"):
                    ctx.push_id_str(route)
                    try:
                        if route == "plan":
                            changes = ctx.render_property_batch_plan(plan, 12.)
                        elif route == "values":
                            changes = ctx.render_property_batch_plan_values(plan, [value], 12.)
                        else:
                            changes = ctx.render_property_batch([descriptor], 12.)
                        assert changes == {}
                    finally:
                        ctx.pop_id()
                # Capacity arguments are allocation hints, never data limits;
                # existing full GUID callers pass 32 for a 32-byte value.
                for capacity in (0, 1, 32):
                    assert ctx.text_input(f"Input {capacity}", value, capacity) == value
                    assert ctx.input_text_with_hint(f"Hint {capacity}", "Hint", value, capacity) == value
                assert ctx.text_area("Area", value) == value
            except Exception as error:
                observed["errors"].append(error)
            finally:
                observed["renders"] += 1
                ctx.end_window()

    def update(_delta):
        observed["frames"] += 1
        if observed["renders"] >= len(cases) or observed["frames"] > 30:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.string_batch", probe)
    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.string_batch")
    assert observed["renders"] >= len(cases)
    assert not observed["errors"], repr(observed["errors"])


def test_long_component_text_inspection_history_save_and_reopen(engine, resident_authors):
    from infernux.components.fields import get_serialized_fields
    from infernux.engine.ui.inspector_components import _apply_batch_changes_py
    from infernux.ui import UIText

    state = resident_authors
    owner = state.objects[0]
    component = owner.add_py_component(UIText())
    initial = "原始谜题线索\n" * 1000 + "END_TEXT"
    updated = initial + "更多谜题信息\n" * 800
    component.text = initial
    document = state.docs[0]
    state.registry.mark_changed(document.document_id)
    assert state.registry.request_save_to_resource(document.document_id, str(state.paths[0])).accepted
    assert not document.is_dirty
    state.history.clear()
    metadata = get_serialized_fields(UIText)["text"]
    observed = {"frames": 0, "renders": 0, "error": None}

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(600., 400., 0)
            visible = ctx.begin_window("Dialogue###component_text_contract", True, 0)
            try:
                if visible:
                    descriptor = build_scalar_desc("##dialogue", "Text", metadata, component.text)
                    changes = ctx.render_property_batch([descriptor], 80.)
                    _apply_batch_changes_py(component, changes, [("text", metadata, component.text, None)])
                    assert not changes
                    observed["renders"] += 1
            except Exception as error:
                observed["error"] = error
            finally:
                ctx.end_window()

    def update(_delta):
        observed["frames"] += 1
        if observed["frames"] >= 5:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.component_text", probe)
    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.component_text")
    if observed["error"] is not None:
        raise observed["error"]
    assert observed["renders"] > 0
    assert component.text == initial and not document.is_dirty and not state.history.can_undo
    # Feed an accepted edit through the same Inspector transaction route.
    # Native mouse/key tests independently verify production of exact edits.
    _apply_batch_changes_py(component, {0: updated}, [("text", metadata, initial, None)])
    assert component.text == updated and document.is_dirty
    state.history.undo()
    assert component.text == initial and not document.is_dirty
    state.history.redo()
    assert component.text == updated
    assert state.registry.request_save(document.document_id).accepted
    assert state.files.reload_from_resource(document_id=document.document_id, resource_path=str(state.paths[0]))
    restored_owner = state.files.scene_for_document(document.document_id).find("Owner0")
    restored, = (item for item in restored_owner.get_py_components() if isinstance(item, UIText))
    assert restored.text == updated
