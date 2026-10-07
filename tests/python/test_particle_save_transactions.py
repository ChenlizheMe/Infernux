"""Particle source and compiled publication share one request ownership gate."""
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from infernux.core.document_store import capture_document_file_state
from infernux.particle import ParticleArtifactRegistry, ParticleArtifactSuperseded, ParticleGraphAsset


@pytest.fixture
def particle_project(tmp_path, monkeypatch):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    ParticleArtifactRegistry.clear()
    path = tmp_path / "Assets" / "Sparks.particlegraph"
    yield path
    ParticleArtifactRegistry.clear()


def _durable_state(path):
    current = ParticleArtifactRegistry.get(str(path))
    return (
        path.read_bytes(),
        Path(current.artifact_path).read_bytes(),
        (path.parent.parent / "Library/Artifacts/Particle/RuntimeIndex.json").read_bytes(),
    )


def _assert_published(path, name):
    current = ParticleArtifactRegistry.get(str(path))
    assert ParticleGraphAsset.load(str(path)).name == name
    assert current.hir["name"] == name
    assert json.loads(Path(current.artifact_path).read_text(encoding="utf-8"))["hir"]["name"] == name


@pytest.mark.parametrize("background", [False, True])
def test_particle_save_file_conflict_preserves_artifact_and_registry(particle_project, background):
    path = particle_project
    original = ParticleGraphAsset(name="Baseline")
    current = ParticleArtifactRegistry.save_graph_asset(original, str(path))
    state = capture_document_file_state(str(path))
    path.write_text("external collaborator edit\n", encoding="utf-8")
    before = _durable_state(path)
    edited = replace(original, name="Edited")
    with pytest.raises(RuntimeError, match="changed outside the editor"):
        if background:
            prepared = ParticleArtifactRegistry.prepare_graph_asset(edited, str(path))
            ticket = ParticleArtifactRegistry.submit_prepared_graph(prepared, expected_file_state=state)
            ticket.wait()
        else:
            ParticleArtifactRegistry.save_graph_asset(edited, str(path), expected_file_state=state)
    assert _durable_state(path) == before
    assert ParticleArtifactRegistry.get(str(path)) is current


def test_particle_prepared_editor_save_cannot_overwrite_newer_public_save(particle_project):
    path = particle_project
    older = ParticleGraphAsset(name="Older editor snapshot")
    prepared = ParticleArtifactRegistry.prepare_graph_asset(older, str(path))
    ParticleArtifactRegistry.save_graph_asset(replace(older, name="Newer public save"), str(path))
    before = _durable_state(path)
    ticket = ParticleArtifactRegistry.submit_prepared_graph(prepared)
    with pytest.raises(ParticleArtifactSuperseded):
        ticket.wait()
    assert ticket.status == "superseded"
    assert ticket.committed_file_state is None
    assert _durable_state(path) == before
    _assert_published(path, "Newer public save")


def test_particle_clear_never_recycles_an_in_flight_request(particle_project):
    path = particle_project
    asset = ParticleGraphAsset(name="Retired session")
    retired = ParticleArtifactRegistry.prepare_graph_asset(asset, str(path))
    ParticleArtifactRegistry.clear()
    ParticleArtifactRegistry.save_graph_asset(replace(asset, name="New session"), str(path))
    before = _durable_state(path)
    ticket = ParticleArtifactRegistry.submit_prepared_graph(retired)
    with pytest.raises(ParticleArtifactSuperseded):
        ticket.wait()
    assert _durable_state(path) == before


def test_particle_move_retires_an_existing_request_for_the_destination(particle_project):
    source = particle_project
    destination = source.with_name("Moved.particlegraph")
    asset = ParticleGraphAsset(name="Moved accepted source")
    ParticleArtifactRegistry.save_graph_asset(asset, str(source))
    pending = ParticleArtifactRegistry.prepare_graph_asset(
        ParticleGraphAsset(name="Uncommitted destination draft"), str(destination),
    )
    source.replace(destination)
    ParticleArtifactRegistry.remap_source(str(source), str(destination))
    before = _durable_state(destination)
    ticket = ParticleArtifactRegistry.submit_prepared_graph(pending)
    with pytest.raises(ParticleArtifactSuperseded):
        ticket.wait()
    assert _durable_state(destination) == before
    assert not source.exists()
    _assert_published(destination, "Moved accepted source")


def test_particle_background_commit_keeps_source_baseline_and_does_not_block_submit(particle_project, monkeypatch):
    from infernux.core import document_store

    path = particle_project
    asset = ParticleGraphAsset(name="Background commit")
    prepared = ParticleArtifactRegistry.prepare_graph_asset(asset, str(path))
    submit = document_store.submit_document_text
    entered = threading.Event()
    release = threading.Event()
    thread_ids = []

    def paused_write(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=5)
        return submit(*args, **kwargs)

    monkeypatch.setattr(document_store, "submit_document_text", paused_write)
    ticket = ParticleArtifactRegistry.submit_prepared_graph(prepared)
    try:
        assert entered.wait(timeout=5)
        assert not ticket.is_complete
        assert len(thread_ids) == 1 and thread_ids[0] != threading.get_ident()
        assert not path.exists()
    finally:
        release.set()
    ticket.wait()
    assert ticket.status == "succeeded"
    _assert_published(path, "Background commit")
    # The baseline belongs to the native source commit and can authorize the
    # next conditional write; it is not a later sample of whatever is on disk.
    document_store.write_document_text(
        str(path), prepared.source_text, expected_file_state=ticket.committed_file_state,
    )


def test_particle_runtime_readers_keep_committed_snapshot_while_save_waits_for_disk(particle_project, monkeypatch):
    from infernux.core import document_store

    path = particle_project
    asset = ParticleGraphAsset(name="Displayed baseline")
    baseline = ParticleArtifactRegistry.save_graph_asset(asset, str(path))
    prepared = ParticleArtifactRegistry.prepare_graph_asset(replace(asset, name="Pending edit"), str(path))
    entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
    submit = document_store.submit_document_text
    observed = []

    def paused_write(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return submit(*args, **kwargs)

    def read_runtime():
        observed.append(ParticleArtifactRegistry.get(str(path)))
        read_done.set()

    monkeypatch.setattr(document_store, "submit_document_text", paused_write)
    ticket = ParticleArtifactRegistry.submit_prepared_graph(prepared)
    reader = threading.Thread(target=read_runtime)
    try:
        assert entered.wait(timeout=5)
        reader.start()
        assert read_done.wait(timeout=1), "runtime queries must not wait for disk publication"
        assert observed == [baseline]
    finally:
        release.set()
        if reader.ident is not None:
            reader.join(timeout=5)
        ticket.wait()
    _assert_published(path, "Pending edit")


def test_particle_editor_completion_rejects_superseded_committed_snapshot(particle_project):
    from infernux.engine.ui.particle_graph_editor_panel import ParticleGraphEditorPanel

    path = particle_project
    panel = ParticleGraphEditorPanel()
    snapshot = panel.capture_authoring_save_snapshot(str(path))
    ticket = snapshot.submit_write()
    ticket.wait()
    newer = replace(panel.asset, name="Newer external save")
    ParticleArtifactRegistry.save_graph_asset(newer, str(path))
    before = _durable_state(path)
    with pytest.raises(ParticleArtifactSuperseded):
        panel.publish_authoring_save_snapshot(snapshot)
    assert _durable_state(path) == before
    _assert_published(path, "Newer external save")


def test_particle_editor_pending_save_stays_dirty_when_public_save_supersedes_it(particle_project, monkeypatch):
    from infernux.core import document_store
    from infernux.engine.interaction import DocumentRegistry, SaveTicketStatus
    from infernux.engine.ui.particle_graph_editor_panel import ParticleGraphEditorPanel

    path = particle_project
    registry = DocumentRegistry()
    panel = ParticleGraphEditorPanel()
    document = registry.require(panel.document_id)
    saved_revision = document.saved_revision
    save_ticket = registry.begin_save(panel.document_id, save_as=True)
    submit = document_store.submit_document_text
    entered = threading.Event()
    release = threading.Event()

    def paused_write(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return submit(*args, **kwargs)

    monkeypatch.setattr(document_store, "submit_document_text", paused_write)
    try:
        assert panel._save_to(str(path), ticket_id=save_ticket.ticket_id)
        assert entered.wait(timeout=5)
        pending = next(iter(panel._authoring_document_controller._pending_writes.values()))
        assert save_ticket.is_pending
    finally:
        release.set()
    pending.io_ticket.wait()
    newer = replace(ParticleGraphAsset.load(str(path)), name="Newer accepted source")
    ParticleArtifactRegistry.save_graph_asset(newer, str(path))
    panel._authoring_document_controller.poll_pending_writes()
    assert save_ticket.status is SaveTicketStatus.FAILED
    assert "superseded" in save_ticket.message
    assert document.saved_revision == saved_revision
    _assert_published(path, "Newer accepted source")
