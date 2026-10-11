"""Inspector reference callbacks use real component transactions and native scenes."""
from __future__ import annotations

from pathlib import Path

import pytest

from infernux.components import InxComponent, FieldType, serialized_field, component_field
from infernux.components.fields import get_raw_field_value, get_serialized_fields
from infernux.core import AssetManager
from infernux.engine import project_context
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager


class SceneReferenceOwner(InxComponent):
    body = component_field(component_type="Rigidbody")
    collider = component_field(component_type="BoxCollider")
    object_target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT,
                                     required_component="Rigidbody")
    bodies = serialized_field(default=[], field_type=FieldType.LIST,
                              element_type=FieldType.COMPONENT, component_type="Rigidbody")
    objects = serialized_field(default=[], field_type=FieldType.LIST,
                               element_type=FieldType.GAME_OBJECT, required_component="Rigidbody")


class _LabelContext:
    semantic_capture_enabled = False

    def align_text_to_frame_padding(self): pass
    def label(self, *_args): pass
    def same_line(self, *_args): pass
    def set_next_item_width(self, *_args): pass
    def calc_text_width(self, value): return len(value) * 7.0
    def push_id_str(self, *_args): pass
    def pop_id(self): pass
    def begin_drag_drop_source(self, *_args): return False
    def is_item_hovered(self): return False
    def separator(self): pass


@pytest.fixture
def reference_authoring(engine, scene, monkeypatch, tmp_path):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(UndoManager, "_instance", None)
    previous_project = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    files = SceneFileManager()
    files._asset_database = database
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    owner = scene.create_game_object("Reference owner").add_py_component(SceneReferenceOwner())
    target = scene.create_game_object("Reference target")
    body = target.add_component("Rigidbody")
    try:
        yield core, files, folder, owner, target, body
    finally:
        core.shutdown()
        project_context.set_project_root(previous_project)


def _inline_callbacks(monkeypatch, owner, field):
    from infernux.engine.ui import _inspector_references as refs

    callbacks = {}
    monkeypatch.setattr(refs, "render_object_field", lambda *_a, **kw: callbacks.update(kw))
    metadata = get_serialized_fields(type(owner))[field]
    if metadata.field_type == FieldType.COMPONENT:
        refs._render_component_ref_inline(_LabelContext(), owner, field, metadata, 80)
    else:
        refs._render_gameobject_ref_inline(_LabelContext(), owner, field, metadata, None, 80)
    return callbacks


def _activate_other_document(files):
    manager = SceneManager.instance()
    other = manager.create_scene("Other active document")
    document = files.register_loaded_scene(other, "")
    manager.set_active_scene(other)
    return document


def _reopen(scene, folder):
    from infernux.engine.scene_document_transaction import SceneDocumentTransaction

    path = folder / "references.scene"
    assert scene.save_to_file(str(path))
    saved = path.read_bytes()
    reopened = SceneManager.instance().create_scene("Reopened references")
    transaction = SceneDocumentTransaction(reopened, path=str(path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    assert reopened.save_to_file(str(path))
    assert path.read_bytes() == saved
    return reopened


def _assert_target(reference, target, body, *, component):
    assert reference is not None and reference.resolve() is not None
    if component:
        assert reference.component_id == body.component_id
    else:
        assert reference.resolve().id == target.id


@pytest.mark.parametrize("field", ["body", "object_target"])
@pytest.mark.parametrize("other_active", [False, True])
@pytest.mark.parametrize("source", ["drop", "picker"])
def test_inline_reference_assignment_uses_its_owner_scene(reference_authoring, scene, monkeypatch,
                                                        field, other_active, source):
    core, files, folder, owner, target, body = reference_authoring
    owner_document = files.document_id
    manager = SceneManager.instance()
    if other_active:
        other_document = _activate_other_document(files)
    active_before = manager.get_active_scene()
    callbacks = _inline_callbacks(monkeypatch, owner, field)
    if source == "drop":
        callbacks["on_drop_callback"](int(target.id))
    else:
        items = callbacks["picker_scene_items"]("Reference target")
        assert len(items) == 1
        callbacks["on_pick"](items[0][1])
    ref = get_raw_field_value(owner, field)
    _assert_target(ref, target, body, component=field == "body")
    assert manager.get_active_scene() is active_before
    assert core.documents.require(owner_document).is_dirty
    if other_active:
        assert not core.documents.require(other_document).is_dirty
    UndoManager.instance().undo()
    cleared = get_raw_field_value(owner, field)
    assert cleared is None or cleared.resolve() is None
    UndoManager.instance().redo()
    assert get_raw_field_value(owner, field) == ref
    reopened = _reopen(scene, folder)
    loaded = reopened.find("Reference owner").get_py_component(SceneReferenceOwner)
    loaded_target = reopened.find("Reference target")
    _assert_target(get_raw_field_value(loaded, field), loaded_target,
                   loaded_target.get_component("Rigidbody"), component=field == "body")


@pytest.mark.parametrize("field", ["bodies", "objects"])
@pytest.mark.parametrize("source", ["header_drop", "item_drop", "item_picker"])
def test_list_reference_callbacks_commit_to_inactive_owner_document(
    reference_authoring, scene, monkeypatch, field, source,
):
    from infernux.engine.ui import _inspector_list_field as lists
    from infernux.engine.ui.igui import IGUI

    core, files, folder, owner, target, body = reference_authoring
    owner_document = files.document_id
    other_document = _activate_other_document(files)
    active_before = SceneManager.instance().get_active_scene()
    metadata = get_serialized_fields(type(owner))[field]
    before = [] if source == "header_drop" else [None]
    setattr(owner, field, before)

    def header(*_args, **kwargs):
        if source == "header_drop":
            kwargs["on_header_drop"](int(target.id))
            return False
        return True

    def item(*_args, **kwargs):
        if source == "item_drop":
            kwargs["on_drop"](int(target.id))
        else:
            choices = kwargs["picker_scene_items"]("Reference target")
            assert len(choices) == 1
            kwargs["on_pick"](choices[0][1])

    monkeypatch.setattr(IGUI, "list_header", header)
    monkeypatch.setattr(IGUI, "list_body_begin", lambda *_a: None)
    monkeypatch.setattr(IGUI, "list_body_end", lambda *_a: None)
    monkeypatch.setattr(IGUI, "list_item_remove_button", lambda *_a: False)
    monkeypatch.setattr(IGUI, "reorder_separator", lambda *_a: None)
    monkeypatch.setattr(IGUI, "object_field", item)
    lists._render_list_field(_LabelContext(), owner, field, metadata, before, 80)
    values = get_raw_field_value(owner, field)
    assert len(values) == 1
    _assert_target(values[0], target, body, component=field == "bodies")
    assert SceneManager.instance().get_active_scene() is active_before
    assert core.documents.require(owner_document).is_dirty
    assert not core.documents.require(other_document).is_dirty
    UndoManager.instance().undo()
    assert get_raw_field_value(owner, field) == before
    UndoManager.instance().redo()
    _assert_target(get_raw_field_value(owner, field)[0], target, body, component=field == "bodies")
    reopened = _reopen(scene, folder)
    loaded = reopened.find("Reference owner").get_py_component(SceneReferenceOwner)
    loaded_target = reopened.find("Reference target")
    _assert_target(get_raw_field_value(loaded, field)[0], loaded_target,
                   loaded_target.get_component("Rigidbody"), component=field == "bodies")


def test_picker_preserves_exact_component_and_rejects_deleted_selection(
    reference_authoring, scene, monkeypatch,
):
    from infernux.components.ref_wrappers import ComponentRef

    core, files, folder, owner, target, _ = reference_authoring
    target.add_component("BoxCollider")
    selected = target.add_component("BoxCollider")
    selected.is_trigger = True
    _activate_other_document(files)
    callbacks = _inline_callbacks(monkeypatch, owner, "collider")
    choices = callbacks["picker_scene_items"]("Reference target")
    assert len(choices) == 2
    callbacks["on_pick"](choices[1][1])
    assert get_raw_field_value(owner, "collider").component_id == selected.component_id
    UndoManager.instance().undo()
    assert owner.collider is None
    UndoManager.instance().redo()
    assert owner.collider.component_id == selected.component_id
    reopened = _reopen(scene, folder)
    loaded = reopened.find("Reference owner").get_py_component(SceneReferenceOwner)
    assert loaded.collider.is_trigger

    removed = target.add_component("BoxCollider")
    stale_choice = ComponentRef(removed)
    target.remove_component(removed)
    journal_revision = core.action_journal.revision
    callbacks["on_pick"](stale_choice)
    assert owner.collider.component_id == selected.component_id
    assert core.action_journal.revision == journal_revision


@pytest.mark.parametrize("payload_kind", ["guid", "path"])
def test_prefab_asset_drop_keeps_asset_identity_separate_from_scene_components(
    reference_authoring, scene, monkeypatch, payload_kind,
):
    from infernux.components.ref_wrappers import PrefabRef
    from infernux.engine.prefab_manager import save_prefab

    core, files, folder, owner, target, _ = reference_authoring
    path = folder / "target.prefab"
    assert save_prefab(target, str(path))
    database = AssetManager._asset_database
    imported = database.import_asset(str(path))
    assert imported.succeeded, imported.error
    guid = str(database.get_guid_from_path(str(path)))
    assert guid
    _activate_other_document(files)
    payload = guid if payload_kind == "guid" else str(path)
    _inline_callbacks(monkeypatch, owner, "object_target")["on_drop_callback"](payload)
    reference = get_raw_field_value(owner, "object_target")
    assert isinstance(reference, PrefabRef) and reference.guid == guid
    serialized = owner._serialize_fields_document()["object_target"]
    assert serialized["guid"] == guid and "path_hint" not in serialized
    revision = core.action_journal.revision
    callbacks = _inline_callbacks(monkeypatch, owner, "body")
    assert callbacks["accept_drag_type"] == "HIERARCHY_GAMEOBJECT"
    callbacks["on_drop_callback"](payload)
    assert owner.body is None
    assert core.action_journal.revision == revision
    UndoManager.instance().undo()
    assert owner.object_target is None
    UndoManager.instance().redo()
    assert get_raw_field_value(owner, "object_target").guid == guid


@pytest.mark.parametrize("source", ["drop", "picker"])
def test_native_joint_reference_commits_to_inactive_owner_document(
    reference_authoring, scene, monkeypatch, source,
):
    from infernux.engine.ui import inspector_components as inspector

    core, files, folder, owner, target, body = reference_authoring
    owner.game_object.add_component("Rigidbody")
    joint = owner.game_object.add_component("HingeJoint")
    owner_document = files.document_id
    other_document = _activate_other_document(files)
    active_before = SceneManager.instance().get_active_scene()
    callbacks = {}
    monkeypatch.setattr(inspector, "render_component_reference_field", lambda *_a, **kw: callbacks.update(kw))
    inspector._render_builtin_component_reference(
        _LabelContext(), joint, "connected_body", "connected_body",
        type(joint).connected_body.metadata, None, 80,
    )
    if source == "drop":
        callbacks["on_drop_callback"](int(target.id))
    else:
        choices = callbacks["picker_scene_items"]("Reference target")
        assert len(choices) == 1
        callbacks["on_pick"](choices[0][1])
    assert joint.connected_body.component_id == body.component_id
    assert SceneManager.instance().get_active_scene() is active_before
    assert core.documents.require(owner_document).is_dirty
    assert not core.documents.require(other_document).is_dirty
    UndoManager.instance().undo()
    assert joint.connected_body is None
    UndoManager.instance().redo()
    assert joint.connected_body.component_id == body.component_id
    reopened = _reopen(scene, folder)
    loaded = reopened.find("Reference owner").get_component("HingeJoint")
    loaded_body = reopened.find("Reference target").get_component("Rigidbody")
    assert loaded.connected_body.component_id == loaded_body.component_id


@pytest.mark.parametrize("kind", ["target", "game_object", "component"])
@pytest.mark.parametrize("source", ["drop", "picker"])
def test_ui_event_reference_callbacks_commit_to_inactive_owner_document(
    reference_authoring, scene, monkeypatch, kind, source,
):
    from infernux.engine.ui import inspector_components as widgets, inspector_ui_components as ui
    from infernux.ui import UIButton
    from infernux.ui.ui_event_entry import UIEventArgument, UIEventEntry, UIEventMethodParameter

    core, files, folder, owner, target, body = reference_authoring
    button = owner.game_object.add_py_component(UIButton())
    argument = UIEventArgument(kind=kind, name="selected", component_type="Rigidbody")
    button.on_click_entries = [UIEventEntry(arguments=[argument])]
    before = button._serialize_fields_document()
    owner_document = files.document_id
    other_document = _activate_other_document(files)
    active_before = SceneManager.instance().get_active_scene()
    callbacks = {}
    monkeypatch.setattr(widgets, "render_object_field", lambda *_a, **kw: callbacks.update(kw))
    entries = button.on_click_entries
    if kind == "target":
        ui._render_onclick_target_field(_LabelContext(), button, entries, 0, entries[0], 80)
    else:
        spec = UIEventMethodParameter("selected", kind, "Rigidbody")
        ui._render_onclick_argument_field(
            _LabelContext(), button, entries, 0, 0, spec, entries[0].arguments[0], 80,
            ui._clone_onclick_entries, lambda payload: ui._resolve_onclick_go(payload, component=button),
        )
    if source == "drop":
        callbacks["on_drop_callback"](int(target.id))
    else:
        choices = callbacks["picker_scene_items"]("Reference target")
        assert len(choices) == 1
        callbacks["on_pick"](choices[0][1])

    def assigned(comp):
        entry = comp.on_click_entries[0]
        return get_raw_field_value(entry if kind == "target" else entry.arguments[0], kind)

    _assert_target(assigned(button), target, body, component=kind == "component")
    assert SceneManager.instance().get_active_scene() is active_before
    assert core.documents.require(owner_document).is_dirty
    assert not core.documents.require(other_document).is_dirty
    UndoManager.instance().undo()
    assert button._serialize_fields_document() == before
    UndoManager.instance().redo()
    _assert_target(assigned(button), target, body, component=kind == "component")
    reopened = _reopen(scene, folder)
    loaded = reopened.find("Reference owner").get_py_component(UIButton)
    loaded_target = reopened.find("Reference target")
    _assert_target(assigned(loaded), loaded_target, loaded_target.get_component("Rigidbody"),
                   component=kind == "component")


@pytest.mark.parametrize("field", ["body", "object_target"])
@pytest.mark.parametrize("candidate", ["missing", "wrong_type", "other_scene", "fractional"])
def test_invalid_scene_drop_keeps_field_document_and_history(
    reference_authoring, scene, monkeypatch, field, candidate,
):
    from infernux.engine.ui import _inspector_references as refs

    core, files, _, owner, target, _ = reference_authoring
    owner_document = files.document_id
    _inline_callbacks(monkeypatch, owner, field)["on_drop_callback"](int(target.id))
    before = owner._serialize_fields_document()
    journal = (core.action_journal.entries, core.action_journal.cursor, core.action_journal.revision)
    document = core.documents.require(owner_document)
    revision = document.revision
    other_document = _activate_other_document(files)
    if candidate == "missing":
        payload = 2 ** 62
    elif candidate == "wrong_type":
        payload = int(scene.create_game_object("No Rigidbody").id)
    elif candidate == "fractional":
        payload = float(target.id) + 0.5
    else:
        foreign = SceneManager.instance().get_active_scene().create_game_object("Other target")
        foreign.add_component("Rigidbody")
        payload = int(foreign.id)
    calls = []

    def unexpected_asset_lookup(value):
        calls.append(value)
        raise AssertionError("A scene reference must not query the asset database")

    monkeypatch.setattr(refs, "_resolve_guid_and_path", unexpected_asset_lookup)
    _inline_callbacks(monkeypatch, owner, field)["on_drop_callback"](payload)
    assert calls == []
    assert owner._serialize_fields_document() == before
    assert (core.action_journal.entries, core.action_journal.cursor, core.action_journal.revision) == journal
    assert document.revision == revision
    assert not core.documents.require(other_document).is_dirty


@pytest.mark.parametrize("source", ["drop", "picker"])
def test_particle_mesh_parameter_uses_its_owner_scene(gpu_particle_authoring, scene, monkeypatch, source):
    from infernux.components import ParticleSystem
    from infernux.core.asset_ref import ParticleGraphRef
    from infernux.components.ref_wrappers import ComponentRef
    from infernux.engine.ui import _inspector_extra_renderers as extra
    from infernux.graph.types import TypeRef, ValueType, builtin_mesh_reference
    from infernux.particle import ParticleGraphAsset, ParticleParameter

    core, files, folder, owner, target, _ = gpu_particle_authoring
    renderer = target.add_component("SkinnedMeshRenderer")
    asset = ParticleGraphAsset(parameters=(ParticleParameter(
        stable_id="source", name="Source", value_type=TypeRef(ValueType.MESH),
        default=builtin_mesh_reference("Cube").to_dict(),
    ),))
    path = folder / "mesh.particlegraph"
    asset.save(str(path))
    database = AssetManager._asset_database
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported.succeeded, imported.error
    particle = owner.game_object.add_py_component(ParticleSystem())
    particle.graph = ParticleGraphRef(guid=str(database.get_guid_from_path(str(path))))
    before = particle.get_parameter("source")
    owner_document = files.document_id
    other_document = _activate_other_document(files)
    callbacks = {}
    monkeypatch.setattr(extra, "render_asset_reference_field", lambda *_a, **kw: callbacks.update(kw))
    monkeypatch.setattr(extra, "render_compact_section_header", lambda _ctx, title, **_kw: "parameters_" in title)
    extra._render_particle_system_parameters(_LabelContext(), particle)
    if source == "drop":
        callbacks["on_assign"](int(target.id))
    else:
        choices = callbacks["picker_scene_items"]("Reference target")
        assert len(choices) == 1
        callbacks["on_assign"](choices[0][1])
    ref = ComponentRef._from_dict(particle.get_parameter("source"))
    assert ref.component_id == renderer.component_id
    assert ref.resolve().component_id == renderer.component_id
    assert core.documents.require(owner_document).is_dirty
    assert not core.documents.require(other_document).is_dirty
    UndoManager.instance().undo()
    assert particle.get_parameter("source") == before
    UndoManager.instance().redo()
    assert ComponentRef._from_dict(particle.get_parameter("source")).component_id == renderer.component_id


@pytest.fixture
def gpu_particle_authoring(reference_authoring, engine):
    from infernux.components import ParticleSystem
    from infernux.runtime_services import install_runtime_service, remove_runtime_service

    install_runtime_service("gpu-particles", engine)
    try:
        yield reference_authoring
    finally:
        owner = reference_authoring[3].game_object
        particle = owner.get_py_component(ParticleSystem)
        if particle is not None:
            owner.remove_py_component(particle)
        assert remove_runtime_service("gpu-particles", engine)
