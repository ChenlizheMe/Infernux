"""Prefab transitions retain the original native world until publication succeeds."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infernux import editor
from infernux.components import InxComponent
from infernux.core import AssetManager
from infernux.engine import project_context
from infernux.engine.component_restore import deserialize_scene_document_transactionally
from infernux.engine.interaction import EditorInteractionCore, SelectionDomain
from infernux.engine.scene_manager import SceneFileManager, PREFAB_MODE_SCENE_NAME
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager, Physics, Vector3
from infernux.renderstack.render_stack import RenderStack


class PrefabPublicationProbe(InxComponent):
    reject_publication: bool = False
    _failure_enabled = True
    _destroyed = []

    def on_after_deserialize(self):
        if self.reject_publication and type(self)._failure_enabled:
            raise RuntimeError("authored Prefab publication failure")

    def on_destroy(self):
        type(self)._destroyed.append(id(self))


@pytest.fixture
def prefab_transition(engine, scene, monkeypatch, tmp_path):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(UndoManager, "_instance", None)
    previous_project = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    files = SceneFileManager()
    files._asset_database = database
    files._engine = engine
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    PrefabPublicationProbe._failure_enabled = True
    PrefabPublicationProbe._destroyed.clear()
    try:
        yield core, files, folder
    finally:
        from infernux.engine.deferred_task import DeferredTaskRunner
        from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator

        DeferredTaskRunner.instance().cancel()
        confirmation = DirtyPanelConfirmationCoordinator.instance()
        if confirmation.is_active:
            confirmation.choose_cancel()
        PrefabPublicationProbe._failure_enabled = False
        if files.is_prefab_mode:
            files._do_exit_prefab_mode()
        core.shutdown()
        project_context.set_project_root(previous_project)


def _source(scene, folder, *, reject_asset=False, reject_scene=False):
    source = scene.create_game_object("Prefab source")
    probe = source.add_py_component(PrefabPublicationProbe())
    probe.reject_publication = reject_asset
    path = folder / "Door.prefab"
    editor.save_as_prefab_asset(source, path)
    probe.reject_publication = reject_scene
    scene.create_game_object("Unsaved main scene work")
    return path, source, probe


@pytest.mark.parametrize("command", [False, True])
def test_rejected_prefab_entry_retains_author_world(prefab_transition, scene, command):
    core, files, folder = prefab_transition
    path, source, probe = _source(scene, folder, reject_asset=True)
    source_id = source.id
    before = copy.deepcopy(scene.serialize_document())
    original_bytes = path.read_bytes()
    document_id = files.document_id
    core.documents.mark_changed(document_id)
    document = core.documents.require(document_id)
    revision = document.revision
    if command:
        with pytest.raises(RuntimeError, match="rejected"):
            core.prefabs.open(path=str(path))
    else:
        assert not files.open_prefab_mode(str(path))
    assert scene.serialize_document() == before
    assert SceneManager.instance().get_active_scene() is scene
    assert SceneManager.instance().get_scene(PREFAB_MODE_SCENE_NAME) is None
    assert not files.is_prefab_mode
    assert files.document_id == document_id
    assert core.documents.require(document_id) is document
    assert document.is_dirty and document.revision == revision
    assert scene.find_by_id(source_id) is source
    assert source.get_py_components()[0] is probe
    assert id(probe) not in PrefabPublicationProbe._destroyed
    assert files._previous_scene is None and files._previous_scene_document is None
    assert path.read_bytes() == original_bytes
    PrefabPublicationProbe._failure_enabled = False
    assert files.open_prefab_mode(str(path))
    assert files._do_exit_prefab_mode()
    assert scene.serialize_document() == before


def test_rejected_prefab_exit_keeps_live_draft_and_can_retry(prefab_transition, scene):
    core, files, folder = prefab_transition
    path, _, _ = _source(scene, folder, reject_scene=True)
    main_before = copy.deepcopy(scene.serialize_document())
    original_bytes = path.read_bytes()
    assert files.open_prefab_mode(str(path))
    manager = SceneManager.instance()
    prefab = manager.get_active_scene()
    root = prefab.get_root_objects()[0]
    root_id = root.id
    probe = root.get_py_components()[0]
    root.name = "Unsaved Prefab draft"
    before = copy.deepcopy(prefab.serialize_document())
    document_id = files.document_id
    core.documents.mark_changed(document_id)
    document = core.documents.require(document_id)
    revision = document.revision
    with pytest.raises(RuntimeError, match="authored Prefab publication failure"):
        files._do_exit_prefab_mode()
    assert manager.get_active_scene() is prefab
    assert manager.get_scene(PREFAB_MODE_SCENE_NAME) is prefab
    assert prefab.serialize_document() == before
    assert prefab.find_by_id(root_id) is root
    assert root.get_py_components()[0] is probe
    assert id(probe) not in PrefabPublicationProbe._destroyed
    assert files.is_prefab_mode and files.document_id == document_id
    assert core.documents.require(document_id) is document
    assert document.is_dirty and document.revision == revision
    assert path.read_bytes() == original_bytes
    PrefabPublicationProbe._failure_enabled = False
    assert files._do_exit_prefab_mode()
    assert manager.get_active_scene() is scene
    assert scene.serialize_document() == main_before
    assert not files.is_prefab_mode


def test_prefab_mode_round_trip_preserves_unsaved_main_document(prefab_transition, scene):
    core, files, folder = prefab_transition
    path, _, _ = _source(scene, folder)
    before = copy.deepcopy(scene.serialize_document())
    original_bytes = path.read_bytes()
    document_id = files.document_id
    core.documents.mark_changed(document_id)
    assert files.open_prefab_mode(str(path))
    assert files.is_prefab_mode
    assert not scene.get_all_objects()
    assert files._do_exit_prefab_mode()
    assert SceneManager.instance().get_active_scene() is scene
    assert scene.serialize_document() == before
    assert files.document_id == document_id
    assert core.documents.require(document_id).is_dirty
    assert path.read_bytes() == original_bytes


def test_single_scene_publication_rollback_control(prefab_transition, scene):
    _, files, folder = prefab_transition
    _, source, probe = _source(scene, folder, reject_scene=True)
    before = copy.deepcopy(scene.serialize_document())
    with pytest.raises(RuntimeError, match="authored Prefab publication failure"):
        deserialize_scene_document_transactionally(scene, before, files._asset_database)
    assert scene.serialize_document() == before
    assert source.get_py_components()[0] is probe


@pytest.mark.parametrize("entering", [True, False])
def test_failed_transition_preserves_residency_render_owner_selection_and_history(prefab_transition, scene, entering):
    core, files, folder = prefab_transition
    path, source, _ = _source(scene, folder, reject_asset=entering, reject_scene=not entering)
    manager = SceneManager.instance()
    other = manager.create_scene("Other resident author world")
    other_probe = other.create_game_object("Additional resident").add_py_component(PrefabPublicationProbe())
    other_id = files.register_loaded_scene(other, "")
    core.documents.mark_changed(other_id)
    other_document = core.documents.require(other_id)
    other_before = copy.deepcopy(other.serialize_document())
    if not entering:
        assert files.open_prefab_mode(str(path))
        source = manager.get_active_scene().get_root_objects()[0]
    active = manager.get_active_scene()
    source.transform.position = Vector3(1000, 0, 0)
    collider = source.add_component("BoxCollider")
    stack = active.create_game_object("Original render owner").add_py_component(RenderStack())
    RenderStack.activate_instance(stack, active)
    Physics.sync_transforms()
    hit = Physics.raycast(Vector3(1000, 5, 0), Vector3(0, -1, 0), 10)
    assert hit is not None and hit.collider.component_id == collider.component_id
    collider_id = collider.component_id
    assert core.scene_objects.rename(source.id, "Journaled draft edit")
    core.selection.select_scene_object(source.id, owner_id="hierarchy", record_history=False)
    core.focus.activate_panel("hierarchy", view_id="hierarchy", document_id=files.document_id,
                              record_history=False)
    context = core.capture_context()
    entries, cursor = core.action_journal.entries, core.action_journal.cursor
    before = copy.deepcopy(active.serialize_document())
    if entering:
        assert not files.open_prefab_mode(str(path))
    else:
        with pytest.raises(RuntimeError, match="authored Prefab publication failure"):
            files._do_exit_prefab_mode()
    assert manager.get_active_scene() is active
    assert active.serialize_document() == before
    assert core.capture_context() == context
    assert core.action_journal.entries == entries and core.action_journal.cursor == cursor
    assert RenderStack.instance(active) is stack
    assert other.serialize_document() == other_before
    assert other.get_root_objects()[0].get_py_components()[0] is other_probe
    assert core.documents.require(other_id) is other_document and other_document.is_dirty
    Physics.sync_transforms()
    restored_hit = Physics.raycast(Vector3(1000, 5, 0), Vector3(0, -1, 0), 10)
    assert restored_hit is not None and restored_hit.collider.component_id == collider_id
    UndoManager.instance().undo()
    assert source.name != "Journaled draft edit"
    UndoManager.instance().redo()
    assert source.name == "Journaled draft edit"


@pytest.mark.parametrize("choice", ["save", "discard", "cancel"])
def test_public_prefab_exit_gate_preserves_mode_when_restore_fails(prefab_transition, scene, choice):
    from infernux.engine.deferred_task import DeferredTaskRunner
    from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator

    core, files, folder = prefab_transition
    path, _, _ = _source(scene, folder, reject_scene=True)
    original_bytes = path.read_bytes()
    assert files.open_prefab_mode(str(path))
    manager = SceneManager.instance()
    prefab = manager.get_active_scene()
    root = prefab.get_root_objects()[0]
    assert core.scene_objects.rename(root.id, "Gate draft")
    document = core.documents.require(files.document_id)
    core.documents.mark_changed(document.document_id)
    assert core.prefabs.exit()
    confirmation = DirtyPanelConfirmationCoordinator.instance()
    assert confirmation.is_active
    getattr(confirmation, f"choose_{choice}")()
    runner = DeferredTaskRunner.instance()
    assert runner.is_busy is (choice != "cancel")
    entries, cursor = core.action_journal.entries, core.action_journal.cursor
    runner.tick()
    assert not runner.is_busy and not files._deferred_exit_prefab
    assert files.is_prefab_mode and manager.get_active_scene() is prefab
    assert prefab.get_root_objects()[0] is root and root.name == "Gate draft"
    assert core.documents.require(files.document_id) is document
    assert core.action_journal.entries == entries and core.action_journal.cursor == cursor
    if choice == "save":
        assert not document.is_dirty
        assert json.loads(path.read_text(encoding="utf-8"))["root_object"]["name"] == "Gate draft"
    else:
        assert document.is_dirty
        assert path.read_bytes() == original_bytes
    # Fixing the author's failing callback permits a normal subsequent request.
    PrefabPublicationProbe._failure_enabled = False
    assert core.prefabs.exit()
    if confirmation.is_active:
        confirmation.choose_discard()
    assert runner.is_busy
    runner.tick()
    assert not files.is_prefab_mode and manager.get_active_scene() is scene


def test_reserved_prefab_scene_is_never_silently_cleared(prefab_transition, scene):
    _, files, folder = prefab_transition
    path, _, _ = _source(scene, folder)
    manager = SceneManager.instance()
    reserved = manager.create_scene(PREFAB_MODE_SCENE_NAME)
    original = reserved.create_game_object("Existing owned object")
    manager.set_active_scene(scene)
    assert not files.open_prefab_mode(str(path))
    assert manager.get_active_scene() is scene
    assert reserved.get_root_objects()[0] is original


def test_exit_source_reconciliation_failure_rolls_back_both_worlds(prefab_transition, scene):
    from infernux.engine.prefab_manager import instantiate_prefab

    _, files, folder = prefab_transition
    path, _, _ = _source(scene, folder)
    database = files._asset_database
    linked = instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                                scene=scene, asset_database=database)
    assert linked.prefab_guid
    assert files.open_prefab_mode(str(path))
    manager = SceneManager.instance()
    prefab = manager.get_active_scene()
    root = prefab.get_root_objects()[0]
    before = copy.deepcopy(prefab.serialize_document())
    document_id = files.document_id
    original_bytes = path.read_bytes()
    path.unlink()  # An external asset deletion while its Prefab draft is open.
    try:
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)) as caught:
            files._do_exit_prefab_mode()
        assert path.name in str(caught.value)
        assert files.is_prefab_mode and files.document_id == document_id
        assert manager.get_active_scene() is prefab
        assert prefab.get_root_objects()[0] is root
        assert prefab.serialize_document() == before
        assert not scene.get_all_objects()
    finally:
        path.write_bytes(original_bytes)
    assert files._do_exit_prefab_mode()
    assert manager.get_active_scene() is scene
