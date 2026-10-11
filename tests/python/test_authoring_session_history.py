"""Save points and cancelled close decisions preserve authored history."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infernux.engine.interaction import (
    AuthoringMutationService, CloseCoordinator, CloseIntent, CloseIntentKind,
    DocumentCapability, DocumentKind, DocumentRegistry,
)
from infernux.engine.undo import UndoManager, SetPropertyCommand, _base


class ScalarController:
    def __init__(self, registry):
        self.registry = registry
        self.value = 0

    def capture_authoring_snapshot(self):
        return {"value": self.value}

    def restore_authoring_snapshot(self, snapshot):
        self.value = snapshot["value"]

    def capture_document_restore_state(self, document_id):
        return self.capture_authoring_snapshot()

    def save(self, *, ticket, save_as=False):
        self.registry.capture_save_revision(ticket.ticket_id)
        self.registry.complete_save(ticket.ticket_id, success=True)
        return True


@pytest.fixture
def authoring():
    types = (DocumentRegistry, UndoManager, AuthoringMutationService)
    previous = [kind._instance for kind in types]
    registry = DocumentRegistry()
    manager = UndoManager()
    service = AuthoringMutationService(registry)
    def create(name):
        controller = ScalarController(registry)
        document = registry.create(DocumentKind.GENERIC, name, document_id=name,
                                   capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
                                   controller=controller)
        registry.attach_view(name, name + "_view")
        return document, controller
    def edit(document, controller, value, merge_key=""):
        assert service.apply(document.document_id, "Set value", lambda: setattr(controller, "value", value),
                             view_id=document.document_id + "_view", merge_key=merge_key)
    try:
        yield registry, manager, service, create, edit
    finally:
        DocumentRegistry._instance = registry
        service.shutdown()
        manager.shutdown()
        registry.clear()
        for kind, instance in zip(types, previous):
            kind._instance = instance


@pytest.mark.parametrize("redo", [False, True])
def test_saved_undo_session_restores_all_drafts_and_allocates_fresh_revision(authoring, redo):
    registry, manager, _, create, edit = authoring
    document, controller = create("first")
    edit(document, controller, 7)
    assert registry.request_save(document.document_id).accepted
    manager.undo()
    if redo:
        manager.redo()
    other, other_controller = create("second")
    edit(other, other_controller, 19)
    captured = json.loads(json.dumps(registry.capture_session_state()))
    restored = DocumentRegistry()
    try:
        assert restored.queue_session_restore(captured) == 2
        doc, snapshot = restored.claim_session_document("first_view", controller=ScalarController(restored))
        assert snapshot == {"value": 7 if redo else 0}
        assert (doc.revision, doc.saved_revision) == (document.revision, document.saved_revision)
        assert doc.is_dirty == document.is_dirty
        assert restored.reserve_changed_revision(doc.document_id, view_id="first_view") > max(doc.revision, doc.saved_revision)
        assert restored.claim_session_document("second_view", controller=ScalarController(restored))[1] == {"value": 19}
    finally:
        restored.clear()
        DocumentRegistry._instance = registry


@pytest.mark.parametrize("strategy", ["snapshot", "revision_wrapper"])
@pytest.mark.parametrize("save_state", ["none", "completed", "pending"])
def test_authoring_merge_preserves_completed_and_pending_save_points(authoring, monkeypatch, strategy, save_state):
    registry, manager, service, create, edit = authoring
    doc, controller = create("first")
    monkeypatch.setattr(_base, "_time", SimpleNamespace(time=lambda: 100.0))
    def change(value):
        if strategy == "snapshot":
            edit(doc, controller, value, merge_key="value")
        else:
            assert service.execute_command(doc.document_id,
                lambda before, after: SetPropertyCommand(controller, "value", controller.value, value), view_id="first_view")
    change(7)
    ticket = None
    if save_state == "completed":
        assert registry.request_save(doc.document_id).accepted
    elif save_state == "pending":
        ticket = registry.begin_save(doc.document_id)
    change(9)
    manager.undo()
    assert controller.value == (0 if save_state == "none" else 7)
    if ticket is not None:
        registry.complete_save(ticket.ticket_id, success=True)
    if save_state != "none":
        assert not doc.is_dirty
    manager.redo()
    assert controller.value == 9 and doc.is_dirty


@pytest.mark.parametrize("terminal", [CloseIntentKind.EXIT_EDITOR, CloseIntentKind.CLOSE_PROJECT])
def test_cancelled_terminal_discard_keeps_drafts_and_future_edits(authoring, terminal):
    registry, _, _, create, edit = authoring
    first, controller = create("first")
    registry.attach_view(first.document_id, "first_other_view")
    second, second_controller = create("second")
    edit(first, controller, 7)
    edit(second, second_controller, 19)
    events = []
    close = CloseCoordinator(registry)
    assert close.request(CloseIntent(terminal), lambda: events.append("closed"), lambda: events.append("cancelled"))
    close.decide_discard()
    assert close.active_document is second
    close.cancel()
    assert events == ["cancelled"]
    assert first.is_dirty and controller.value == 7
    assert not registry.is_session_restore_suppressed("first_view")
    assert not registry.is_session_restore_suppressed("first_other_view")
    edit(first, controller, 42)
    captured = registry.capture_session_state()
    assert {item["document_id"]: item["restore_state"] for item in captured["documents"]} == {
        "first": {"value": 42}, "second": {"value": 19},
    }
    assert registry.request_save(first.document_id).accepted
    assert not first.is_dirty
    assert close.request(CloseIntent(terminal), lambda: events.append("closed"))
    close.decide_discard()
    assert events[-1] == "closed"
    assert not registry.is_session_restore_suppressed("first_view")
    assert registry.is_session_restore_suppressed("second_view")


def test_late_edit_requires_a_new_discard_decision_without_reasking_unchanged_drafts(authoring):
    registry, _, _, create, edit = authoring
    first, first_controller = create("first")
    second, second_controller = create("second")
    edit(first, first_controller, 7)
    edit(second, second_controller, 19)
    completed = []
    close = CloseCoordinator(registry)
    close.request(CloseIntent(CloseIntentKind.EXIT_EDITOR), lambda: completed.append(True))
    close.decide_discard()
    edit(first, first_controller, 42)
    close.decide_discard()
    assert not completed and close.active_document is first
    assert first.is_dirty and second.is_dirty
    close.decide_discard()
    assert completed == [True] and not close.is_active
    assert registry.capture_session_state()["documents"] == []


def test_external_conflict_after_discard_decision_still_blocks_close(authoring):
    registry, _, _, create, edit = authoring
    first, controller = create("first")
    second, other = create("second")
    edit(first, controller, 7)
    edit(second, other, 19)
    close = CloseCoordinator(registry)
    completed = []
    close.request(CloseIntent(CloseIntentKind.EXIT_EDITOR), lambda: completed.append(True))
    close.decide_discard()
    registry.mark_saved(first.document_id)
    registry.mark_conflict(first.document_id)
    close.decide_discard()
    assert not completed and close.active_document is first
    assert close.state.value == "waiting_for_conflict"
    close.cancel()
    assert not registry.is_session_restore_suppressed("second_view")


def test_real_timeline_save_then_undo_can_restore_session(tmp_path):
    # A separate native headless owner keeps this disk-write test independent
    # from the suite's session-scoped Vulkan owner.
    script = r'''
import json, sys
from pathlib import Path
from infernux.core.assets import AssetManager
from infernux.core.document_store import DocumentStore
from infernux.engine.engine import Engine
from infernux.engine.interaction import DocumentRegistry, EditorInteractionCore, SelectionDomain
from infernux.engine.ui import project_file_ops
from infernux.engine.ui.animtimeline_editor_panel import AnimTimelineEditorPanel
from infernux.engine.undo import UndoManager
from infernux.lib import LogLevel, RuntimeMode
project = Path(sys.argv[1])
(project / "Assets").mkdir()
(project / "ProjectSettings").mkdir()
engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
try:
    engine.init_headless(str(project))
    core = EditorInteractionCore.instance()
    core.panels.register_selection_authority("animtimeline_editor", (SelectionDomain.TIMELINE_ELEMENT,))
    result = project_file_ops.create_animtimeline(str(project / "Assets"), "Timeline", AssetManager.require_asset_database())
    assert result.success, result.detail
    path = Path(result.created_path)
    panel = AnimTimelineEditorPanel()
    try:
        panel.open_document_resource_immediate(str(path))
        baseline = panel._timeline.duration
        assert panel._apply_discrete_property("", "duration", 3.0, "Change duration")
        registry = DocumentRegistry.instance()
        document = registry.require(panel.document_id)
        assert registry.request_save(panel.document_id).accepted
        DocumentStore.flush()
        panel._authoring_document_controller.poll_pending_writes()
        assert not document.is_dirty
        UndoManager.instance().undo()
        assert panel._timeline.duration == baseline and document.is_dirty
        captured = registry.capture_session_state()
        restored = DocumentRegistry()
        try:
            assert restored.queue_session_restore(captured) == 1
        finally:
            restored.clear()
            DocumentRegistry._instance = registry
        assert json.loads(path.read_text(encoding="utf-8"))["duration"] == 3.0
    finally:
        panel.unbind_document()
finally:
    engine.exit()
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "python"))
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            capture_output=True, timeout=60, env=env,
                            creationflags=0x08000000 if sys.platform == "win32" else 0)
    assert result.returncode == 0, (result.stdout, result.stderr)
