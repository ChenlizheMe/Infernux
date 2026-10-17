import copy
from dataclasses import replace

import pytest

from infernux.engine.interaction import (
    ClipboardDomain,
    ClipboardService,
    DocumentRegistry,
    EditorContextSnapshot,
    GraphElementKind,
    GraphElementRef,
    SelectionService,
)
from infernux.core.anim_state_machine import AnimClipEntry, AnimCondition, AnimState, AnimTransition
from infernux.engine.ui.animfsm_editor_panel import AnimFSMEditorPanel
from infernux.engine.undo import UndoManager


def _panel_with_history():
    DocumentRegistry()
    selection = SelectionService()
    ClipboardService()
    manager = UndoManager()
    manager.set_context_hooks(
        lambda: EditorContextSnapshot(selection=selection.snapshot),
        lambda context, phase: (
            selection.apply_snapshot(
                context.selection,
                reason=phase,
                record_history=False,
            ),
            True,
        )[1],
    )
    panel = AnimFSMEditorPanel()
    panel._graph_selection.bind(selection)
    return panel, manager


def test_animfsm_disk_discard_reconciles_document_revision(tmp_path):
    from infernux.core.anim_state_machine import AnimStateMachine
    from infernux.engine.interaction import DocumentActionStatus

    target = tmp_path / "Saved.animfsm"
    assert AnimStateMachine(name="Saved").save(str(target))
    panel, _manager = _panel_with_history()
    assert panel._authoring_document_controller.open_resource_immediate(str(target))
    registry = DocumentRegistry.instance()
    document = registry.require(panel.document_id)
    loaded_model = panel._fsm
    assert not panel._authoring_document_controller.open_resource_immediate(
        str(tmp_path / "Missing.animfsm")
    )
    assert panel._fsm is loaded_model
    assert panel.document_id == document.document_id
    registry.mark_changed(document.document_id)

    result = registry.request_discard(document.document_id)

    assert result.status is DocumentActionStatus.APPLIED
    assert registry.require(panel.document_id).is_dirty is False
    assert panel._document_is_dirty() is False


def test_node_graph_edit_scene_switch_and_scene_edit_undo_in_exact_order():
    from types import SimpleNamespace

    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import FocusService
    from infernux.engine.undo import GlobalFocusCommand, LambdaCommand

    DocumentRegistry()
    selection = SelectionService()
    focus = FocusService()
    manager = UndoManager()
    panel = AnimFSMEditorPanel()
    panel._graph_selection.bind(selection)

    class _VisibleGraphWindow:
        @staticmethod
        def is_window_content_visible(window_id):
            return window_id == panel.window_id

    panel._window_manager = _VisibleGraphWindow()
    bootstrap = BootstrapSelectionMixin()
    bootstrap.window_manager = panel._window_manager
    bootstrap.interaction_core = SimpleNamespace(
        focus=focus,
        capture_context=lambda **overrides: EditorContextSnapshot(
            overrides.get("focus", focus.snapshot),
            overrides.get("selection", selection.snapshot),
        ),
    )
    focus.add_change_listener(bootstrap._on_global_focus_changed)
    focus.activate_panel(
        panel.window_id,
        view_id=panel.window_id,
        document_id=panel.document_id,
        record_history=False,
    )
    manager.set_context_hooks(
        bootstrap.interaction_core.capture_context,
        lambda context, _phase: (
            focus.apply_snapshot(context.focus, record_history=False),
            selection.apply_snapshot(context.selection, record_history=False),
            True,
        )[-1],
    )

    assert panel._insert_parameter()
    scene_value = {"x": 0.0}
    focus.activate_panel("scene_view", view_id="scene_view", record_history=True)
    assert manager.execute(
        LambdaCommand(
            "Move Camera",
            undo_fn=lambda: scene_value.update(x=0.0),
            redo_fn=lambda: scene_value.update(x=1.0),
        )
    )

    entries = manager.action_journal.entries
    assert [entry.action.description for entry in entries] == [
        "Add parameter",
        "Focus scene_view",
        "Move Camera",
    ]
    assert isinstance(entries[1].action, GlobalFocusCommand)

    manager.undo()
    assert scene_value["x"] == 0.0
    assert focus.snapshot.active_view_id == "scene_view"
    manager.undo()
    assert focus.snapshot.active_view_id == panel.window_id
    manager.undo()
    assert panel._fsm.parameters == []
    assert focus.snapshot.active_view_id == panel.window_id
    assert len(manager.action_journal.entries) == 3


def test_animfsm_parameter_rename_preserves_transition_reference_and_undo():
    panel, manager = _panel_with_history()
    source = AnimState(name="Source")
    target = AnimState(name="Target")
    assert panel._insert_state(source, "Add Source", make_default=True)
    assert panel._insert_state(target, "Add Target", make_default=False)
    assert panel._insert_parameter()
    parameter = panel._fsm.parameters[0]
    old_parameter = parameter.to_dict()
    old_parameter["name"] = "speed"
    assert panel._update_parameter_document(
        parameter,
        old_parameter,
        "Rename parameter",
        merge_key=f"parameter:{parameter.stable_id}:name",
    )
    transition = AnimTransition(
        target_state="Target",
        conditions=[
            AnimCondition(
                parameter_id=parameter.stable_id,
                operator=">",
                threshold=1.0,
            )
        ],
    )
    assert panel._insert_transition(source, transition, "Connect")
    baseline_revision = panel._fsm_document().revision
    manager.clear()

    renamed = panel._fsm.parameters[0].to_dict()
    renamed["name"] = "velocity"
    assert panel._update_parameter_document(
        panel._fsm.parameters[0],
        renamed,
        "Rename parameter",
        merge_key=f"parameter:{parameter.stable_id}:name",
    )
    assert panel._fsm.states[0].transitions[0].conditions[0].parameter_id == (parameter.stable_id)
    rename_diff = manager.action_journal.applied_entries()[-1].action.diff
    assert [mutation.element.kind for mutation in rename_diff.mutations] == [
        GraphElementKind.PARAMETER,
    ]

    manager.undo()
    assert panel._fsm.parameters[0].name == "speed"
    assert panel._fsm.states[0].transitions[0].conditions[0].parameter_id == (parameter.stable_id)
    assert panel._fsm_document().revision == baseline_revision


def test_animfsm_parameter_remove_atomically_removes_conditions_and_undo_restores():
    panel, manager = _panel_with_history()
    source = AnimState(name="Source")
    target = AnimState(name="Target")
    assert panel._insert_state(source, "Add Source", make_default=True)
    assert panel._insert_state(target, "Add Target", make_default=False)
    assert panel._insert_parameter()
    parameter = panel._fsm.parameters[0]
    condition = AnimCondition(
        parameter_id=parameter.stable_id,
        operator=">",
        threshold=0.5,
    )
    transition = AnimTransition(target_state="Target", conditions=[condition])
    assert panel._insert_transition(source, transition, "Connect")
    manager.clear()

    assert panel._remove_parameter(parameter.stable_id)
    assert panel._fsm.parameters == []
    assert panel._fsm.states[0].transitions[0].conditions == []

    manager.undo()
    assert panel._fsm.parameters[0].stable_id == parameter.stable_id
    assert panel._fsm.states[0].transitions[0].conditions == [condition]


@pytest.mark.parametrize("transition", ["reload", "disable"])
def test_animfsm_document_transition_retires_node_drag(tmp_path, transition):
    from infernux.core.anim_state_machine import AnimStateMachine
    from infernux.engine.interaction import TransientInteractionService

    panel, manager = _panel_with_history()
    transients = TransientInteractionService()
    source = AnimStateMachine(
        name="Original",
        states=[AnimState(name="Original", position=[40.0, 60.0])],
    )
    path = tmp_path / "Original.animfsm"
    assert source.save(str(path))
    panel.open_document_resource_immediate(str(path))
    uid = panel._fsm.states[0].stable_id
    original = panel._graph.capture_authoring_state()
    locator = DocumentRegistry.instance().locate(panel.document_id)
    manager.clear()
    panel._on_node_drag_start(uid)
    panel._graph.find_node(uid).pos_x += 130.0

    replacement = AnimStateMachine(
        name="Replacement",
        states=[AnimState(name="Replacement", position=[90.0, 50.0])],
    )
    if transition == "new":
        assert panel.command_new_fsm()
    elif transition in {"open", "reload"}:
        target = path if transition == "reload" else tmp_path / "Replacement.animfsm"
        assert replacement.save(str(target))
        panel.open_document_resource_immediate(str(target))
    elif transition == "restore":
        panel.restore_document_restore_state({"fsm": replacement.to_dict()})
    elif transition == "unbind":
        panel.unbind_document()
        assert panel.restore_dormant_document(locator)
    else:
        panel.on_disable()

    expected = panel._graph.capture_authoring_state()
    assert not transients.can_cancel
    assert not panel._view._dragging_node
    panel._on_node_drag_end(uid)  # A late mouse-up cannot mutate the replacement.
    assert not transients.cancel_active()
    assert panel._graph.capture_authoring_state() == expected
    assert not manager.can_undo
    if transition in {"unbind", "disable"}:
        assert expected == original
    elif transition == "new":
        assert panel.capture_document_restore_state(panel.document_id)["fsm"]["states"] == []
    else:
        assert (
            panel.capture_document_restore_state(panel.document_id)["fsm"]["states"] == replacement.to_dict()["states"]
        )


def test_animfsm_state_delete_undo_restores_tree_links_default_and_selection():
    panel, manager = _panel_with_history()
    idle = AnimState(name="Idle")
    run = AnimState(name="Run")
    jump = AnimState(name="Jump")
    assert panel._insert_state(idle, "Add Idle", make_default=True)
    assert panel._insert_state(run, "Add Run", make_default=False)
    assert panel._insert_state(jump, "Add Jump", make_default=False)
    idle_to_run = AnimTransition(target_state="Run")
    run_to_jump = AnimTransition(target_state="Jump")
    jump_to_run = AnimTransition(target_state="Run")
    assert panel._insert_transition(idle, idle_to_run, "Idle to Run")
    assert panel._insert_transition(run, run_to_jump, "Run to Jump")
    assert panel._insert_transition(jump, jump_to_run, "Jump to Run")
    assert panel._set_default_state("Run", "Default Run")
    panel._graph_selection.select_one(
        GraphElementKind.NODE,
        run.stable_id,
        record_history=False,
    )

    baseline = panel._fsm.to_dict()
    baseline_revision = panel._fsm_document().revision
    manager.clear()

    assert panel._remove_states((run.stable_id,))
    assert [state.name for state in panel._fsm.states] == ["Idle", "Jump"]
    assert panel._fsm.default_state == "Idle"
    assert all(transition.target_state != "Run" for state in panel._fsm.states for transition in state.transitions)
    assert panel._graph_selection.elements == ()
    assert len(manager.action_journal.applied_entries()[-1].action.diff.mutations) == 5

    manager.undo()

    assert panel._fsm.to_dict() == baseline
    assert panel._fsm_document().revision == baseline_revision
    assert panel._graph_selection.primary == GraphElementRef(
        GraphElementKind.NODE,
        run.stable_id,
    )
    assert panel._graph.find_link(idle_to_run.stable_id) is not None
    assert panel._graph.find_link(run_to_jump.stable_id) is not None
    assert panel._graph.find_link(jump_to_run.stable_id) is not None


def test_animfsm_transition_reconnect_is_one_precise_undo_step():
    panel, manager = _panel_with_history()
    source = AnimState(name="Source")
    other_source = AnimState(name="Other Source")
    target = AnimState(name="Target")
    other_target = AnimState(name="Other Target")
    for index, state in enumerate((source, other_source, target, other_target)):
        assert panel._insert_state(
            state,
            f"Add {state.name}",
            make_default=index == 0,
        )
    transition = AnimTransition(target_state="Target")
    assert panel._insert_transition(source, transition, "Connect")
    baseline_revision = panel._fsm_document().revision
    manager.clear()

    panel._on_link_replaced(
        transition.stable_id,
        other_source.stable_id,
        "out",
        other_target.stable_id,
        "in",
    )

    owner, current = panel._fsm.get_transition_by_id(transition.stable_id)
    assert owner.stable_id == other_source.stable_id
    assert current.target_state == "Other Target"
    assert len(manager.action_journal.applied_entries()[-1].action.diff.mutations) == 1

    manager.undo()
    owner, current = panel._fsm.get_transition_by_id(transition.stable_id)
    assert owner.stable_id == source.stable_id
    assert current.target_state == "Target"
    assert panel._fsm_document().revision == baseline_revision


def test_animfsm_copy_paste_uses_global_typed_clipboard_and_is_atomic():
    panel, manager = _panel_with_history()
    first = AnimState(name="First", kind="group", clips=[AnimClipEntry("a", 1.0), AnimClipEntry("b", .6), AnimClipEntry("c", .4)], position=[10.0, 20.0])
    second = AnimState(name="Second", position=[30.0, 40.0])
    assert panel._insert_state(first, "Add First", make_default=True)
    assert panel._insert_state(second, "Add Second", make_default=False)
    assert panel._insert_parameter()
    parameter = panel._fsm.parameters[0]
    transition = AnimTransition(
        target_state="Second",
        conditions=[AnimCondition(parameter_id=parameter.stable_id, operator=">", threshold=0.0)],
    )
    assert panel._insert_transition(first, transition, "Connect")
    panel._graph_selection.select(
        (
            GraphElementRef(GraphElementKind.NODE, first.stable_id),
            GraphElementRef(GraphElementKind.NODE, second.stable_id),
        ),
        record_history=False,
    )
    panel._on_graph_copy()

    payload = ClipboardService.instance().peek(ClipboardDomain.GRAPH_ELEMENT)
    assert payload is not None
    assert len(payload.items) == 1
    assert payload.items[0].sub_kind == "node_graph_subgraph"
    assert len(payload.items[0].data.state.nodes) == 2
    assert len(payload.items[0].data.state.links) == 1
    baseline = panel._fsm.to_dict()
    baseline_revision = panel._fsm_document().revision
    manager.clear()

    panel._on_graph_paste()

    assert [state.name for state in panel._fsm.states] == [
        "First",
        "Second",
        "First 2",
        "Second 2",
    ]
    pasted_first = panel._fsm.get_state("First 2")
    pasted_second = panel._fsm.get_state("Second 2")
    assert pasted_first.position == [58.0, 68.0]
    assert pasted_second.position == [78.0, 88.0]
    assert [clip.to_dict() for clip in pasted_first.clips] == [clip.to_dict() for clip in first.clips]
    assert [item.target_state for item in pasted_first.transitions] == ["Second 2"]
    assert len(manager.action_journal.applied_entries()[-1].action.diff.mutations) == 3

    manager.undo()
    assert panel._fsm.to_dict() == baseline
    assert panel._fsm_document().revision == baseline_revision

    panel._graph_selection.select(
        (GraphElementRef(GraphElementKind.NODE, first.stable_id),
         GraphElementRef(GraphElementKind.NODE, second.stable_id)), record_history=False)
    assert panel.command_edit_cut()
    assert not panel._fsm.states
    manager.undo()
    assert panel._fsm.to_dict() == baseline
    manager.redo()
    assert not panel._fsm.states
    assert panel.command_edit_paste()
    assert len(panel._fsm.states) == 2


@pytest.mark.parametrize("target_count,mode", [(2, "2d"), (2, "timeline")])
def test_cross_document_paste_reserves_names_for_the_whole_batch(tmp_path, mode, target_count):
    panel, manager = _panel_with_history()
    panel._new_fsm_immediate(mode=mode)
    first, second = AnimState(name="A"), AnimState(name=f"A {target_count}")
    assert panel._insert_state(first, "First", make_default=True)
    assert panel._insert_state(second, "Second", make_default=False)
    assert panel._insert_transition(first, AnimTransition(target_state=second.name), "Connect")
    panel._graph_selection.select(
        tuple(GraphElementRef(GraphElementKind.NODE, state.stable_id) for state in (first, second)),
        record_history=False,
    )
    panel._on_graph_copy()
    clipboard = ClipboardService.instance().peek(ClipboardDomain.GRAPH_ELEMENT)
    clipboard_state = clipboard.items[0].data

    panel._new_fsm_immediate(mode=mode)
    for index in range(target_count):
        assert panel._insert_state(AnimState(name="A" if index == 0 else "Existing"), "Target", make_default=index == 0)
    before = panel._fsm.to_dict()
    manager.clear()
    assert panel._node_graph_paste()
    names = [state.name for state in panel._fsm.states]
    assert len(names) == len(set(names)) == target_count + 2
    copied_first, copied_second = panel._fsm.states[-2:]
    assert copied_first.transitions[0].target_state == copied_second.name
    assert ClipboardService.instance().peek(ClipboardDomain.GRAPH_ELEMENT).items[0].data == clipboard_state
    extension = ".timelinefsm" if mode == "timeline" else ".animfsm"
    panel.capture_authoring_save_snapshot(str(tmp_path / ("Pasted" + extension)))
    after = panel._fsm.to_dict()
    manager.undo()
    assert panel._fsm.to_dict() == before
    manager.redo()
    assert panel._fsm.to_dict() == after
    panel.capture_authoring_save_snapshot(str(tmp_path / ("Replayed" + extension)))


@pytest.mark.parametrize("same_named_parameter,mode", [(False, "2d"), (True, "3d")])
def test_cross_document_paste_rejects_missing_parameter_identity_without_mutation(tmp_path, mode, same_named_parameter):
    panel, manager = _panel_with_history()
    panel._new_fsm_immediate(mode=mode)
    first, second = AnimState(name="First"), AnimState(name="Second")
    assert panel._insert_state(first, "First", make_default=True)
    assert panel._insert_state(second, "Second", make_default=False)
    assert panel._insert_parameter()
    parameter = panel._fsm.parameters[0]
    assert panel._insert_transition(
        first,
        AnimTransition(
            target_state=second.name,
            conditions=[AnimCondition(parameter_id=parameter.stable_id, operator=">", threshold=0.0)],
        ),
        "Connect",
    )
    panel._graph_selection.select(
        tuple(GraphElementRef(GraphElementKind.NODE, state.stable_id) for state in (first, second)),
        record_history=False,
    )
    panel._on_graph_copy()
    panel._new_fsm_immediate(mode=mode)
    if same_named_parameter:
        assert panel._insert_parameter()
        assert panel._fsm.parameters[0].name == parameter.name
        assert panel._fsm.parameters[0].stable_id != parameter.stable_id
    before = panel._fsm.to_dict()
    before_graph = panel._graph.capture_authoring_state()
    before_revision = panel._fsm_document().revision
    manager.clear()
    assert not panel._node_graph_paste()
    assert panel._fsm.to_dict() == before
    assert panel._graph.capture_authoring_state() == before_graph
    assert panel._fsm_document().revision == before_revision
    assert not manager.can_undo
    extension = ".timelinefsm" if mode == "timeline" else ".animfsm"
    panel.capture_authoring_save_snapshot(str(tmp_path / ("Unchanged" + extension)))


@pytest.mark.parametrize("mode", ["timeline"])
def test_cross_document_paste_preserves_redo_and_can_retry_after_parameter_declaration(tmp_path, mode):
    panel, manager = _panel_with_history()
    panel._new_fsm_immediate(mode=mode)
    source_states = [AnimState(name=name) for name in ("A", "A 1", "C")]
    for index, state in enumerate(source_states):
        assert panel._insert_state(state, state.name, make_default=index == 0)
    assert panel._insert_parameter()
    parameter = panel._fsm.parameters[0]
    assert panel._insert_transition(source_states[0], AnimTransition(target_state="A 1"), "Unconditional")
    assert panel._insert_transition(
        source_states[1],
        AnimTransition(
            target_state="C",
            conditions=[AnimCondition(parameter_id=parameter.stable_id)],
        ),
        "Conditional",
    )
    panel._graph_selection.select(
        tuple(GraphElementRef(GraphElementKind.NODE, state.stable_id) for state in source_states), record_history=False
    )
    panel._on_graph_copy()
    clipboard = copy.deepcopy(ClipboardService.instance().peek(ClipboardDomain.GRAPH_ELEMENT))

    panel._new_fsm_immediate(mode=mode)
    assert panel._insert_state(AnimState(name="A"), "Resident", make_default=True)
    manager.clear()
    assert panel._insert_state(AnimState(name="Keep redo"), "Pending redo", make_default=False)
    manager.undo()
    before = panel._fsm.to_dict()
    graph_before = panel._graph.capture_authoring_state()
    selection_before = SelectionService.instance().snapshot
    revision_before = panel._fsm_document().revision
    for _ in range(2):
        assert not panel._node_graph_paste()
        assert panel._fsm.to_dict() == before
        assert panel._graph.capture_authoring_state() == graph_before
        assert SelectionService.instance().snapshot == selection_before
        assert panel._fsm_document().revision == revision_before
        assert manager.can_redo and not manager.can_undo
        assert ClipboardService.instance().peek(ClipboardDomain.GRAPH_ELEMENT) == clipboard
    manager.redo()
    assert panel._fsm.get_state("Keep redo") is not None
    manager.undo()

    # Shared parameter identity remains valid even when its display name changes.
    panel._fsm.parameters = [replace(parameter, name="Renamed shared parameter")]
    assert panel._node_graph_paste()
    assert [state.name for state in panel._fsm.states] == ["A", "A 1", "A 1 1", "C"]
    copied_first, copied_second, copied_third = panel._fsm.states[-3:]
    assert copied_first.transitions[0].target_state == copied_second.name
    assert copied_second.transitions[0].target_state == copied_third.name
    assert copied_second.transitions[0].conditions[0].parameter_id == parameter.stable_id
    assert not manager.can_redo
    after = panel._fsm.to_dict()
    manager.undo()
    assert [state.name for state in panel._fsm.states] == ["A"]
    manager.redo()
    assert panel._fsm.to_dict() == after
    extension = ".timelinefsm" if mode == "timeline" else ".animfsm"
    panel.capture_authoring_save_snapshot(str(tmp_path / ("Retried" + extension)))
