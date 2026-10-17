from types import SimpleNamespace
import json

from infernux.engine.interaction import (
    DocumentCapability,
    DocumentKey,
    DocumentKind,
    DocumentRegistry,
    EditableResourceDocumentController,
    ensure_editable_resource_document,
)
from infernux.engine.undo import EditableDocumentDraftCommand, UndoManager


class _Resource:
    def __init__(self, value=1.0):
        self.value = float(value)

    def serialize_document(self):
        return {"value": self.value}

    def deserialize_document(self, document):
        self.value = float(document["value"])
        return True


class _ExecutionLayer:
    def __init__(self):
        self.pending = False
        self.saved = []
        self.binding = None

    def refresh_binding(self, category, file_path):
        self.binding = (category, file_path)

    def schedule_rw_save(self, resource):
        self.pending = True
        self._resource = resource

    def flush_rw_autosave(self, *, force=False):
        del force
        if not self.pending:
            return False
        self.pending = False
        self.saved.append(self._resource.serialize_document())
        return True


class _AsyncWriteTicket:
    def __init__(self):
        self.is_complete = False
        self.status = "pending"

    def complete(self, status="succeeded"):
        self.status = str(status)
        self.is_complete = True


class _AsyncExecutionLayer(_ExecutionLayer):
    def __init__(self):
        super().__init__()
        self.tickets = []

    def flush_rw_autosave(self, *, force=False):
        del force
        if not self.pending:
            return False
        self.pending = False
        self.saved.append(self._resource.serialize_document())
        ticket = _AsyncWriteTicket()
        self.tickets.append(ticket)
        return ticket


class _ViewScopedExecutionLayer(_ExecutionLayer):
    view_scoped_persistence = True

    def schedule_rw_save(self, resource):
        raise AssertionError("view-scoped persistence must not own the write")

    def flush_rw_autosave(self, *, force=False):
        raise AssertionError("view-scoped persistence must not own the flush")


def test_linked_resource_reload_cannot_take_over_reused_inspector_state():
    from infernux.engine.ui.asset_details_renderer import _State

    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    state = _State()
    try:
        source_controller = ensure_editable_resource_document(
            category="render_effect",
            document_kind=DocumentKind.RENDER_EFFECT,
            file_path="Assets/Edge Fade.effect",
            resource=_Resource(0.35),
            guid="source-effect-guid",
            view_id="inspector",
            state=state,
        )
        # Selection invalidation reuses this same view for the containing group.
        state.reset()
        group = _Resource(0.6)
        group_controller = ensure_editable_resource_document(
            category="render_effect",
            document_kind=DocumentKind.RENDER_EFFECT,
            file_path="Assets/Group.effectgroup",
            resource=group,
            guid="group-guid",
            view_id="inspector",
            state=state,
        )
        reloaded_source = _Resource(0.75)
        linked_controller = ensure_editable_resource_document(
            category="render_effect",
            document_kind=DocumentKind.RENDER_EFFECT,
            file_path="Assets/Edge Fade.effect",
            resource=reloaded_source,
            guid="source-effect-guid",
        )

        assert linked_controller is source_controller
        assert linked_controller.resource is reloaded_source
        assert state.resource_controller is group_controller
        assert state.settings is group
        assert registry.document_for_view("inspector").document_id == group_controller.document_id
    finally:
        DocumentRegistry._instance = previous_registry


def test_dirty_editable_resource_keeps_authoritative_instance_across_views():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    original = _Resource(3.0)
    replacement = _Resource(1.0)
    first_state = SimpleNamespace(resource_controller=None, settings=original)
    second_state = SimpleNamespace(resource_controller=None, settings=replacement)
    execution = _ExecutionLayer()
    try:
        controller = ensure_editable_resource_document(
            category="material",
            document_kind=DocumentKind.MATERIAL,
            file_path="Assets/Ocean.mat",
            resource=original,
            guid="ocean-guid",
            state=first_state,
            exec_layer=execution,
        )
        registry.mark_changed(controller.document_id)

        rebound = ensure_editable_resource_document(
            category="material",
            document_kind=DocumentKind.MATERIAL,
            file_path="Assets/Ocean.mat",
            resource=replacement,
            guid="ocean-guid",
            state=second_state,
            exec_layer=execution,
        )

        assert rebound is controller
        assert controller.resource is original
        assert second_state.resource_controller is controller
        assert second_state.settings is original
    finally:
        DocumentRegistry._instance = previous_registry


def test_editable_resource_document_tracks_autosave_and_undo_revisions():
    previous_registry = DocumentRegistry._instance
    previous_manager = UndoManager._instance
    registry = DocumentRegistry()
    manager = UndoManager()
    resource = _Resource()
    execution = _ExecutionLayer()
    state = SimpleNamespace(resource_controller=None, settings=resource)
    controller = EditableResourceDocumentController("physic_material", "Assets/Ice.physmat", resource)
    document = registry.create(
        DocumentKind.PHYSIC_MATERIAL,
        "Ice.physmat",
        key=DocumentKey.asset(DocumentKind.PHYSIC_MATERIAL, "ice-guid"),
        resource_path="Assets/Ice.physmat",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    state.resource_controller = controller
    controller.bind(
        file_path="Assets/Ice.physmat",
        resource=resource,
        exec_layer=execution,
        state=state,
    )
    try:
        next_revision = registry.reserve_content_revision(document.document_id)
        assert manager.execute(
            EditableDocumentDraftCommand(
                controller,
                {"value": 1.0},
                {"value": 2.0},
                document.revision,
                next_revision,
                edit_key="value",
                description="Set value",
            )
        )
        assert resource.value == 2.0
        assert document.is_dirty
        assert execution.pending

        assert controller.flush_autosave()
        assert not document.is_dirty
        assert execution.saved == [{"value": 2.0}]

        manager.undo()
        assert resource.value == 1.0
        assert document.is_dirty
        assert controller.flush_autosave()
        assert not document.is_dirty
        assert execution.saved[-1] == {"value": 1.0}
    finally:
        DocumentRegistry._instance = previous_registry
        UndoManager._instance = previous_manager


def test_explicit_resource_save_commits_only_its_captured_revision():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    resource = _Resource(4.0)
    execution = _ExecutionLayer()
    controller = EditableResourceDocumentController("render_effect", "Assets/Bloom.effect", resource)
    document = registry.create(
        DocumentKind.RENDER_EFFECT,
        "Bloom.effect",
        resource_path="Assets/Bloom.effect",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    controller.bind(
        file_path="Assets/Bloom.effect",
        resource=resource,
        exec_layer=execution,
        state=None,
    )
    registry.mark_changed(document.document_id)
    original_flush = execution.flush_rw_autosave

    def _flush_with_publication(*, force=False):
        flushed = original_flush(force=force)
        if flushed:
            registry.mark_changed(document.document_id)
        return flushed

    execution.flush_rw_autosave = _flush_with_publication
    try:
        result = registry.request_save(document.document_id)

        assert result.accepted
        assert execution.saved == [{"value": 4.0}]
        assert document.saved_revision == 1
        assert document.revision == 2
        assert document.is_dirty is True
    finally:
        DocumentRegistry._instance = previous_registry


def test_async_resource_save_stays_pending_until_io_succeeds():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    resource = _Resource(4.0)
    execution = _AsyncExecutionLayer()
    controller = EditableResourceDocumentController("render_effect", "Assets/Bloom.effect", resource)
    document = registry.create(
        DocumentKind.RENDER_EFFECT,
        "Bloom.effect",
        resource_path="Assets/Bloom.effect",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    controller.bind(
        file_path="Assets/Bloom.effect",
        resource=resource,
        exec_layer=execution,
        state=None,
    )
    registry.mark_changed(document.document_id)
    try:
        result = registry.request_save(document.document_id)

        assert result.status.value == "pending"
        assert document.is_dirty
        assert registry.active_save_ticket(document.document_id) is not None

        execution.tickets[-1].complete()
        assert registry.process_pending_saves() == 1
        assert registry.active_save_ticket(document.document_id) is None
        assert document.is_dirty is False
    finally:
        DocumentRegistry._instance = previous_registry


def test_async_resource_save_failure_never_moves_save_point():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    resource = _Resource(4.0)
    execution = _AsyncExecutionLayer()
    controller = EditableResourceDocumentController("render_effect", "Assets/Bloom.effect", resource)
    document = registry.create(
        DocumentKind.RENDER_EFFECT,
        "Bloom.effect",
        resource_path="Assets/Bloom.effect",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    controller.bind(
        file_path="Assets/Bloom.effect",
        resource=resource,
        exec_layer=execution,
        state=None,
    )
    registry.mark_changed(document.document_id)
    try:
        assert registry.request_save(document.document_id).status.value == "pending"
        execution.tickets[-1].complete("failed")

        registry.process_pending_saves()

        assert registry.active_save_ticket(document.document_id) is None
        assert document.saved_revision == 0
        assert document.is_dirty
    finally:
        DocumentRegistry._instance = previous_registry


def test_discard_supersedes_an_already_submitted_background_write():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    resource = _Resource(1.0)
    execution = _AsyncExecutionLayer()
    controller = EditableResourceDocumentController("render_effect", "Assets/Bloom.effect", resource)
    document = registry.create(
        DocumentKind.RENDER_EFFECT,
        "Bloom.effect",
        resource_path="Assets/Bloom.effect",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    controller.bind(
        file_path="Assets/Bloom.effect",
        resource=resource,
        exec_layer=execution,
        state=None,
    )
    resource.value = 9.0
    registry.mark_changed(document.document_id)
    controller.schedule_autosave()
    assert controller.flush_autosave(force=True)
    stale_ticket = execution.tickets[-1]
    try:
        result = registry.request_discard(document.document_id)

        assert result.accepted
        assert resource.value == 1.0
        assert document.is_dirty is False
        assert len(execution.tickets) == 2
        baseline_ticket = execution.tickets[-1]

        stale_ticket.complete("superseded")
        baseline_ticket.complete("succeeded")
        assert registry.process_pending_saves() == 2
        assert controller._saved_document == {"value": 1.0}
        assert document.revision == 0
        assert document.saved_revision == 0
    finally:
        DocumentRegistry._instance = previous_registry


def test_material_document_save_completes_from_native_document_store(tmp_path):
    from infernux.core.assets import AssetManager
    from infernux.core.document_store import DocumentStore
    from infernux.engine.path_utils import path_key
    from infernux.engine.ui.asset_execution_layer import (
        AssetAccessMode,
        AssetExecutionLayer,
    )

    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    path = tmp_path / "Surface.mat"

    class _MaterialResource(_Resource):
        file_path = str(path)

        def serialize(self):
            return json.dumps(self.serialize_document())

    resource = _MaterialResource(0.4)
    execution = AssetExecutionLayer(
        "material",
        str(path),
        AssetAccessMode.READ_WRITE_RESOURCE,
        autosave_debounce_sec=0.0,
    )
    controller = EditableResourceDocumentController("material", str(path), resource)
    document = registry.create(
        DocumentKind.MATERIAL,
        "Surface.mat",
        resource_path=str(path),
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    controller.bind(
        file_path=str(path),
        resource=resource,
        exec_layer=execution,
        state=None,
    )
    registry.mark_changed(document.document_id)
    normalized = path_key(str(path))
    try:
        result = registry.request_save(document.document_id)

        assert result.status.value in {"pending", "applied"}
        if result.status.value == "pending":
            assert document.is_dirty
        else:
            assert path.exists()
            assert document.is_dirty is False

        DocumentStore.flush(str(path))
        AssetManager.poll_pending_asset_writes()
        registry.process_pending_saves()

        assert json.loads(path.read_text(encoding="utf-8")) == {"value": 0.4}
        assert document.is_dirty is False
        assert registry.active_save_ticket(document.document_id) is None
    finally:
        AssetManager._scheduled_saves.pop(str(path), None)
        AssetManager._material_save_snapshots.pop(normalized, None)
        AssetManager._pending_document_write_records.pop(normalized, None)
        DocumentRegistry._instance = previous_registry


def test_editable_resource_controller_applies_user_document_transaction():
    previous_registry = DocumentRegistry._instance
    previous_manager = UndoManager._instance
    registry = DocumentRegistry()
    manager = UndoManager()
    resource = _Resource()
    execution = _ExecutionLayer()
    state = SimpleNamespace(resource_controller=None, settings=resource)
    controller = EditableResourceDocumentController("render_effect", "Assets/Bloom.effect", resource)
    document = registry.create(
        DocumentKind.RENDER_EFFECT,
        "Bloom.effect",
        resource_path="Assets/Bloom.effect",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    state.resource_controller = controller
    controller.bind(
        file_path="Assets/Bloom.effect",
        resource=resource,
        exec_layer=execution,
        state=state,
    )
    try:
        assert controller.apply_document(
            {"value": 3.0},
            view_id="inspector",
            edit_key="value",
            description="Set value",
        )
        assert resource.value == 3.0
        assert document.is_dirty
        assert execution.pending
        assert manager.action_journal.applied_entries()[-1].origin.value == "user"

        manager.undo()
        assert resource.value == 1.0
        assert document.is_dirty is False
    finally:
        DocumentRegistry._instance = previous_registry
        UndoManager._instance = previous_manager


def test_editable_resource_discard_restores_last_durable_document():
    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    resource = _Resource()
    execution = _ExecutionLayer()
    state = SimpleNamespace(resource_controller=None, settings=resource)
    controller = EditableResourceDocumentController("physic_material", "Assets/Ice.physmat", resource)
    document = registry.create(
        DocumentKind.PHYSIC_MATERIAL,
        "Ice.physmat",
        resource_path="Assets/Ice.physmat",
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=controller,
    )
    controller.document_id = document.document_id
    state.resource_controller = controller
    controller.bind(
        file_path="Assets/Ice.physmat",
        resource=resource,
        exec_layer=execution,
        state=state,
    )
    try:
        revision = registry.reserve_content_revision(document.document_id)
        controller.restore_document({"value": 9.0}, revision, persist=False)
        assert document.is_dirty

        result = registry.request_discard(document.document_id)
        assert result.accepted
        assert resource.value == 1.0
        assert not document.is_dirty
    finally:
        DocumentRegistry._instance = previous_registry


def test_animation_clip_documents_replace_state_without_losing_asset_identity():
    from infernux.core.animation_clip import AnimationClip, AnimationFrame
    from infernux.core.animation_clip3d import AnimationClip3D

    clip_2d = AnimationClip(
        name="Run",
        authoring_texture_guid="texture-guid",
        frames=[AnimationFrame(sprite_frame_id=f"{index + 1:032x}") for index in (0, 1)],
        fps=12.0,
    )
    clip_2d.file_path = "Assets/Run.animclip2d"
    document_2d = clip_2d.serialize_document()
    document_2d["fps"] = 24.0
    assert clip_2d.deserialize_document(document_2d)
    assert clip_2d.fps == 24.0
    assert clip_2d.file_path == "Assets/Run.animclip2d"
    assert not clip_2d.deserialize_document({"fps": 60.0})

    clip_3d = AnimationClip3D(
        name="Walk",
        source_model_guid="a" * 32,
        take_name="Walk",
    )
    clip_3d.file_path = "Assets/Walk.animclip3d"
    document_3d = clip_3d.serialize_document()
    document_3d["take_name"] = "Run"
    assert clip_3d.deserialize_document(document_3d)
    assert clip_3d.take_name == "Run"
    assert clip_3d.file_path == "Assets/Walk.animclip3d"
    assert not clip_3d.deserialize_document({"take_name": "Idle"})


def test_material_asset_and_inline_inspectors_share_one_native_document():
    from infernux.engine.ui.asset_details_renderer import (
        AssetCategoryDef,
        _State,
        _bind_editable_resource_document,
    )
    from infernux.engine.ui.asset_execution_layer import AssetAccessMode
    from infernux.engine.ui.inspector_material import _build_inline_state

    class _NativeMaterial:
        guid = "material-guid"
        file_path = "Assets/Test.mat"
        is_builtin = False

        def __init__(self):
            self.document = {"properties": {"roughness": {"value": 0.5}}}

        def get_version(self):
            return 1

        def serialize_document(self):
            return self.document

        def deserialize_document(self, document):
            self.document = document
            return True

    class _MaterialWrapper:
        def __init__(self, native):
            self.native = native

        def serialize_document(self):
            return self.native.serialize_document()

        def deserialize_document(self, document):
            return self.native.deserialize_document(document)

    previous_registry = DocumentRegistry._instance
    registry = DocumentRegistry()
    native = _NativeMaterial()
    wrapper = _MaterialWrapper(native)
    state = _State()
    state.category = "material"
    state.file_path = native.file_path
    state.settings = wrapper
    state.meta = {"guid": native.guid}
    state.extra = {"native_mat": native}
    state.exec_layer = _ExecutionLayer()
    category = AssetCategoryDef(
        display_name="Material",
        access_mode=AssetAccessMode.READ_WRITE_RESOURCE,
        load_fn=lambda _path: None,
        autosave_debounce=0.35,
    )
    try:
        _bind_editable_resource_document(state, category)
        asset_controller = state.resource_controller
        inline_state = _build_inline_state(SimpleNamespace(), native)

        document = registry.require(state.document_id)
        assert document.kind is DocumentKind.MATERIAL
        assert document.key == DocumentKey.asset(DocumentKind.MATERIAL, native.guid)
        assert asset_controller.resource is native
        assert state.settings is wrapper
        assert inline_state.resource_controller is asset_controller
        assert inline_state.document_id == state.document_id
        assert document.view_ids == {"inspector"}
    finally:
        DocumentRegistry._instance = previous_registry


def test_sprite_clip_inspector_and_editor_share_edits_history_and_save(engine, monkeypatch):
    from pathlib import Path
    from PIL import Image

    from infernux.core.animation_clip import AnimationClip, AnimationFrame
    from infernux.core.asset_types import (
        SpriteFrame, TextureType, read_texture_import_settings, write_texture_import_settings,
    )
    from infernux.core.assets import AssetManager
    from infernux.core.document_store import DocumentStore
    from infernux.engine._bootstrap_selection import BootstrapSelectionMixin
    from infernux.engine.interaction import (
        AssetMutationService, CloseCoordinator, DocumentOpenService, DocumentOpenStatus,
        ModalService, SelectionService,
    )
    from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator
    from infernux.engine.ui.animclip2d_editor_panel import AnimClip2DEditorPanel
    from infernux.engine.ui.asset_details_renderer import (
        AssetCategoryDef, _State, _apply_editable_resource_document, _bind_asset_document, _load_animclip,
    )
    from infernux.engine.ui.asset_execution_layer import AssetAccessMode, AssetExecutionLayer
    from infernux.engine.ui import asset_details_renderer as inspector
    from infernux.engine.ui.project_file_ops import on_asset_mutation
    from infernux.debug import Debug

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AnimClip2DEditorPanel, "_get_native_engine", staticmethod(lambda: engine))
    assets = Path(database.assets_root)
    texture = assets / "shared-animation.png"
    Image.new("RGBA", (16, 8), "white").save(texture)
    imported = AssetManager.import_asset(str(texture), database=database)
    assert imported, imported.error
    settings = read_texture_import_settings(str(texture))
    settings.texture_type = TextureType.SPRITE
    settings.sprite_frames = [SpriteFrame(x=0, y=0, w=16, h=8)]
    assert write_texture_import_settings(str(texture), settings)
    DocumentStore.flush(str(texture) + ".meta")
    category = AssetCategoryDef(
        display_name="Animation Clip", access_mode=AssetAccessMode.READ_WRITE_RESOURCE,
        load_fn=_load_animclip,
    )
    registry = DocumentRegistry.instance()
    monkeypatch.setattr(AssetMutationService, "_instance", None)
    mutations = AssetMutationService(registry, SelectionService.instance())
    mutations.add_observer(on_asset_mutation)
    observer_errors = []
    monkeypatch.setattr(Debug, "log_suppressed", lambda context, error: observer_errors.append((context, str(error))))
    confirmation = DirtyPanelConfirmationCoordinator(CloseCoordinator(registry), ModalService())
    monkeypatch.setattr(DirtyPanelConfirmationCoordinator, "instance", classmethod(lambda cls: confirmation))
    bootstrap = BootstrapSelectionMixin()
    bootstrap.interaction_core = SimpleNamespace(documents=registry)
    bootstrap.window_manager = SimpleNamespace(
        open_window=lambda _panel_id: panel,
    )
    opener = DocumentOpenService(registry)
    opener.register(
        DocumentKind.ANIMATION_CLIP,
        lambda locator: bootstrap._restore_panel_document(locator, panel_id="animclip2d_editor"),
    )

    def open_clip(path):
        result = opener.open_resource(DocumentKind.ANIMATION_CLIP, str(path))
        if result.status is DocumentOpenStatus.PENDING:
            confirmation.choose_discard()
            assert Path(registry.require(panel.document_id).resource_path) == path
            opened_owner = panel._model
            result = opener.open_resource(DocumentKind.ANIMATION_CLIP, str(path))
            assert panel._model is opened_owner
        assert result.status is DocumentOpenStatus.READY, result.message
        assert panel.document_id == result.document.document_id

    def bind_inspector():
        assert state.load(str(path), "animclip", category)
        state.exec_layer = AssetExecutionLayer("animclip", str(path), category.access_mode)
        _bind_asset_document(state, category)

    for inspector_first in (False, True):
        manager = UndoManager()
        path = assets / f"SharedClip-{inspector_first}.animclip2d"
        source = AnimationClip(authoring_texture_guid=imported.guid, frames=[
            AnimationFrame(sprite_frame_id=settings.sprite_frames[0].stable_id),
        ])
        assert source.save(str(path))
        assert AssetManager.import_asset(str(path), database=database)
        panel = AnimClip2DEditorPanel()
        state = _State()
        monkeypatch.setattr(inspector, "_state", state)
        assert state.load(str(path), "animclip", category)
        state.exec_layer = AssetExecutionLayer("animclip", str(path), category.access_mode)
        if inspector_first:
            bind_inspector()
            document_id = state.document_id
            draft = state.settings.serialize_document()
            draft["fps"] = 18.0
            assert _apply_editable_resource_document(state, draft, edit_key="fps", description="Inspector FPS")
        open_clip(path)
        bind_inspector()
        assert state.document_id == panel.document_id
        assert state.resource_controller is registry.require(panel.document_id).controller
        assert registry.require(panel.document_id).view_ids == {"inspector", panel.window_id}
        assert panel._apply_authoring_mutation("Empty sequence", panel._active_clip.frames.clear)
        bind_inspector()
        assert state.settings.frame_count == 0
        manager.undo()
        if inspector_first:
            assert panel.document_id == document_id and panel._active_clip.fps == 18.0
            manager.undo()
            assert panel._active_clip.fps == 12.0
            manager.redo()
            assert panel._active_clip.fps == 18.0
        assert panel._apply_authoring_mutation("Panel FPS", lambda: setattr(panel._active_clip, "fps", 24.0))
        active_owner = panel._model
        open_clip(path)
        assert panel._model is active_owner and panel._active_clip.fps == 24.0
        bind_inspector()
        assert state.settings.fps == 24.0
        draft = state.settings.serialize_document()
        draft["fps"] = 30.0
        assert _apply_editable_resource_document(state, draft, edit_key="fps", description="Inspector FPS")
        assert panel._active_clip.fps == 30.0
        manager.undo()
        assert panel._active_clip.fps == 24.0
        manager.redo()
        assert panel._active_clip.fps == 30.0
        AssetManager.flush_scheduled_saves(str(path), force=True)
        DocumentStore.flush(str(path))
        assert AnimationClip.load(str(path)).fps == (18.0 if inspector_first else 12.0)
        assert registry.require(panel.document_id).is_dirty
        document_id = panel.document_id
        owner = panel._model
        controller = registry.require(document_id).controller
        with monkeypatch.context() as pending:
            pending.setattr(controller, "poll_pending_writes", lambda: 0)
            result = registry.request_save(document_id)
            assert result.accepted, result.message
            assert panel._new_clip_document_immediate()
            assert panel._model is not owner
            DocumentStore.flush(str(path))
        registry.process_pending_saves()
        assert not observer_errors, observer_errors
        document = registry.require(document_id)
        assert not document.is_dirty, controller.publication_error
        assert panel._active_clip.name == "NewClip" and panel._active_clip.frames == []
        bind_inspector()
        assert state.settings is owner.resource
        with monkeypatch.context() as closed_view:
            def unexpected_gpu_access():
                raise AssertionError("inactive document accessed the preview renderer")
            closed_view.setattr(panel, "_get_native_engine", unexpected_gpu_access)
            draft = state.settings.serialize_document()
            draft["fps"] = 36.0
            assert _apply_editable_resource_document(state, draft, edit_key="fps", description="Inactive clip FPS")
            manager.undo()
            assert owner.resource.fps == 30.0
            manager.redo()
            assert owner.resource.fps == 36.0
            assert registry.request_save(document_id).accepted
            DocumentStore.flush(str(path))
            registry.process_pending_saves()
        assert not registry.require(document_id).is_dirty
        assert panel._active_clip.name == "NewClip" and panel._active_clip.frames == []
        open_clip(path)
        bind_inspector()
        assert panel._model is owner
        assert panel._active_clip.fps == state.settings.fps == AnimationClip.load(str(path)).fps == 36.0
        assert state.settings.frames[0].stable_id == source.frames[0].stable_id
        panel.on_disable()
        panel.unbind_document()
        import gc
        import weakref

        closed_panel = weakref.ref(panel)
        panel = None
        gc.collect()
        assert closed_panel() is None
        draft = state.settings.serialize_document()
        draft["fps"] = 48.0
        assert _apply_editable_resource_document(state, draft, edit_key="fps", description="Closed clip FPS")
        manager.undo()
        assert owner.resource.fps == 36.0
        manager.redo()
        assert owner.resource.fps == 48.0
        assert registry.request_save(document_id).accepted
        DocumentStore.flush(str(path))
        registry.process_pending_saves()
        assert not registry.require(document_id).is_dirty
        assert AnimationClip.load(str(path)).fps == 48.0
        registry.detach_view("inspector")
        assert not observer_errors, observer_errors
    mutations.shutdown()
