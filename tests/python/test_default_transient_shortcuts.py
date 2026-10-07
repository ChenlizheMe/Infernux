"""Exercise production Bootstrap bindings, not a separately rebuilt shortcut map."""
from types import SimpleNamespace

import pytest

from infernux.engine._bootstrap_wiring import BootstrapWiringMixin
from infernux.engine.interaction import (
    EditorInteractionCore, KeyChord, ShortcutEvent, ShortcutRouteStatus,
)
from infernux.engine.preferences_store import PreferencesStore
from infernux.engine.undo import UndoManager


@pytest.fixture(params=['fsm', 'particle'])
def graph_shortcuts(request, tmp_path, monkeypatch):
    monkeypatch.setattr(PreferencesStore(), '_path', str(tmp_path / 'preferences.json'))
    core = EditorInteractionCore()
    previous_history = UndoManager._instance
    UndoManager(core.action_journal)
    if request.param == 'fsm':
        from infernux.engine.ui.animfsm_editor_panel import AnimFSMEditorPanel, _ANIMFSM_PANEL_INTERACTION
        panel = AnimFSMEditorPanel()
        state = panel._fsm.add_state('Puzzle')
        panel._sync_graph_from_fsm()
        uid = state.stable_id
        descriptor = _ANIMFSM_PANEL_INTERACTION
    else:
        from infernux.engine.ui.node_graph_editor_panel import NODE_GRAPH_PANEL_INTERACTION
        from infernux.engine.ui.particle_graph_editor_panel import ParticleGraphEditorPanel
        panel = ParticleGraphEditorPanel()
        uid = 'init::init.velocity'
        descriptor = NODE_GRAPH_PANEL_INTERACTION
    core.panels.register_type(panel.window_id, descriptor)
    core.panels.bind_view(panel.window_id, panel.window_id, panel)
    panel._graph_selection.bind(core.selection)
    bootstrap = BootstrapWiringMixin()
    bootstrap.interaction_core = core
    bootstrap.engine = SimpleNamespace(_play_mode_manager=None)
    bootstrap.hierarchy = bootstrap.project_panel = None
    bootstrap._register_core_editor_commands(SimpleNamespace(), SimpleNamespace())
    core.focus.activate_panel(panel.window_id, view_id=panel.window_id,
                              document_id=panel.document_id, record_history=False)
    assert panel._on_canvas_selection_changed((uid,), '', False)
    try:
        yield core, panel, uid
    finally:
        panel._graph_selection.unbind()
        panel.unbind_document()
        core.shutdown()
        UndoManager._instance = previous_history


@pytest.mark.parametrize('text,captured', [(False, False), (True, False), (False, True), (True, True)])
def test_default_escape_cancels_graph_drag_without_deselecting(graph_shortcuts, text, captured):
    core, panel, uid = graph_shortcuts
    graph = panel._node_graph_authoring_model()
    before = graph.capture_authoring_state()
    selection = core.selection.snapshot
    panel._on_node_drag_start(uid)
    graph.find_node(uid).pos_x += 130.0
    if captured:
        core.focus.set_capture_owner(panel.window_id)
    routed = core.shortcuts.route(ShortcutEvent(KeyChord.parse('Escape'), text_input_active=text))
    assert routed.status is ShortcutRouteStatus.EXECUTED
    assert routed.command_id == 'interaction.cancel'
    assert graph.capture_authoring_state() == before
    assert core.selection.snapshot == selection
    assert not core.transient_interactions.can_cancel


def test_default_escape_without_a_drag_deselects(graph_shortcuts):
    core, panel, uid = graph_shortcuts
    before = panel._node_graph_authoring_model().capture_authoring_state()
    routed = core.shortcuts.route(ShortcutEvent(KeyChord.parse('Escape')))
    assert routed.command_id == 'edit.deselect'
    assert routed.status is ShortcutRouteStatus.EXECUTED
    assert core.selection.snapshot.is_empty
    assert panel._node_graph_authoring_model().capture_authoring_state() == before


def test_default_escape_closes_modal_before_cancelling_underlying_drag(graph_shortcuts):
    core, panel, uid = graph_shortcuts
    graph = panel._node_graph_authoring_model()
    before = graph.capture_authoring_state()
    panel._on_node_drag_start(uid)
    graph.find_node(uid).pos_x += 130.0
    cancelled = []
    core.modals.register('test.modal', is_active=lambda: True,
                         render=lambda ctx: None, cancel=lambda: cancelled.append(True))
    assert core.modals.activate('test.modal', owner_id=panel.window_id)
    first = core.shortcuts.route(ShortcutEvent(KeyChord.parse('Escape')))
    assert first.status is ShortcutRouteStatus.EXECUTED
    assert cancelled == [True] and not core.modals.active_modal_id
    assert core.transient_interactions.can_cancel
    assert graph.capture_authoring_state() != before
    second = core.shortcuts.route(ShortcutEvent(KeyChord.parse('Escape')))
    assert second.command_id == 'interaction.cancel'
    assert second.status is ShortcutRouteStatus.EXECUTED
    assert not core.transient_interactions.can_cancel
    assert graph.capture_authoring_state() == before
