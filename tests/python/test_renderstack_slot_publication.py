"""Rejected stage edits preserve the live document and caller-owned values."""
import copy

import pytest

from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.undo import UndoManager
from infernux.renderstack import EffectSlot, RenderStack


@pytest.fixture
def authoring(scene):
    old_core, old_undo = EditorInteractionCore._instance, UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    stack = scene.create_game_object("Render settings").add_component(RenderStack)
    stack.add_effect_slot("after_opaque")
    stack.add_effect_slot("final")
    try:
        yield stack, core.render_stacks, undo
    finally:
        core.shutdown()
        EditorInteractionCore._instance, UndoManager._instance = old_core, old_undo


@pytest.mark.parametrize("editor", [False, True])
@pytest.mark.parametrize("failure", ["duplicate", "cross_stage", "wrong_type", "bad_stage"])
def test_rejected_slot_edit_preserves_document_and_inputs(authoring, editor, failure):
    stack, service, undo = authoring
    peer = stack.get_effect_stage_slots("after_opaque")[0]
    first = EffectSlot(stage_id="caller_scope")
    slots = [first]
    stage = "final"
    if failure == "duplicate":
        slots.append(EffectSlot(slot_id=first.slot_id))
    elif failure == "cross_stage":
        slots.append(peer)
    elif failure == "wrong_type":
        slots.append(object())
    else:
        stage = "absent_stage"
    before = stack._serialize_fields_document()
    old_slots = tuple(stack.effect_slots)
    input_state = [(slot, slot.stage_id, slot.slot_id) for slot in slots if isinstance(slot, EffectSlot)]
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        if editor:
            service.set_effect_stage_slots(stack, stage, slots)
        else:
            stack.set_effect_stage_slots(stage, slots)
    assert stack._serialize_fields_document() == before
    assert tuple(stack.effect_slots) == old_slots
    assert all((slot.stage_id, slot.slot_id) == (stage_id, slot_id) for slot, stage_id, slot_id in input_state)
    assert not undo.can_undo


def test_successful_editor_replacement_save_reload_undo_redo(authoring):
    stack, service, undo = authoring
    before = stack._serialize_fields_document()
    peer_id = stack.get_effect_stage_slots("after_opaque")[0].slot_id
    first, second = EffectSlot(), EffectSlot(enabled=False)
    service.set_effect_stage_slots(stack, "final", [first, second])
    saved = copy.deepcopy(stack._serialize_fields_document())
    assert [slot.slot_id for slot in stack.get_effect_stage_slots("final")] == [first.slot_id, second.slot_id]
    assert stack.get_effect_stage_slots("after_opaque")[0].slot_id == peer_id
    # Authoring commands address the current component document after reloading it.
    stack._deserialize_fields_document(saved)
    undo.undo()
    assert stack._serialize_fields_document() == before
    undo.redo()
    assert stack._serialize_fields_document() == saved


def test_direct_replacement_owns_inputs_and_preserves_other_stage_identity(authoring):
    stack, _, _ = authoring
    peer = stack.get_effect_stage_slots("after_opaque")[0]
    replacement = EffectSlot(stage_id="caller_scope")
    stack.set_effect_stage_slots("final", [replacement])
    assert stack.get_effect_stage_slots("after_opaque")[0] is peer
    assert replacement.stage_id == "caller_scope"
    replacement.enabled = False
    assert stack.get_effect_stage_slots("final")[0].enabled
