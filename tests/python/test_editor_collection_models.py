from __future__ import annotations


from infernux.engine.interaction import (
    CollectionInteractionModel,
)


def test_collection_reconcile_preserves_valid_state_and_retires_removed_transients():
    model = CollectionInteractionModel(("a", "b", "c"))
    model.activate("b")
    model.begin_rename("b", "Before")
    model.set_insertion(3, ("b",))

    assert model.set_items(("a", "c"))
    assert model.selection.selected_ids == ()
    assert model.rename_session is None
    assert model.insertion is not None
    assert model.insertion.index == 2
    assert model.insertion.dragged_ids == ()
