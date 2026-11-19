"""Scene physics settings in the existing Project asset Inspector."""
from __future__ import annotations

import copy
import json
import os

from infernux.engine.interaction import (
    AuthoringAssetSnapshot, AuthoringDocumentController, AuthoringMutationService,
    DocumentCapability, DocumentKey, DocumentKind, DocumentRegistry,
    document_content_token,
)
from infernux.engine.undo import UndoCommand
from infernux.engine.i18n import t


def _scene_files():
    from infernux.engine.scene_manager import SceneFileManager
    return SceneFileManager.instance()


def _loaded_scene(document_id):
    files = _scene_files()
    return files.scene_for_document(document_id) if files is not None else None


def _backend_name(value):
    if isinstance(value, str):
        if value not in {"cpu", "gpu"}:
            raise ValueError(f"Unknown Scene physics backend: {value}")
        return value
    from infernux.physics import PhysicsBackend
    if not isinstance(value, PhysicsBackend):
        raise ValueError(f"Unknown Scene physics backend: {value}")
    if value == PhysicsBackend.CPU:
        return "cpu"
    if value == PhysicsBackend.GPU:
        return "gpu"
    raise ValueError(f"Unknown Scene physics backend: {value}")


def _can_edit(document_id):
    scene = _loaded_scene(document_id)
    if scene is None:
        return True
    files = _scene_files()
    # Includes queued Play/preparation, Play, Pause and scene transactions.
    return bool(scene.physics_backend_writable and files is not None and files._can_replace_scene())


class _SceneAssetDraft:
    """A world-free document model; persistence uses the shared controller."""

    def __init__(self, path, data):
        self.path = path
        self.data = copy.deepcopy(data)
        self.saved = copy.deepcopy(data)
        self.document_id = ""

    def capture_authoring_snapshot(self):
        return copy.deepcopy(self.data)

    def restore_authoring_snapshot(self, snapshot):
        _backend_name(snapshot.get("physics_backend", "cpu"))
        self.data = copy.deepcopy(snapshot)

    def current_authoring_content_token(self):
        return document_content_token(self.data)

    def capture_authoring_save_snapshot(self, target):
        snapshot = self.capture_authoring_snapshot()
        return AuthoringAssetSnapshot(
            target, json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            document_content_token(snapshot), os.path.splitext(os.path.basename(target))[0], snapshot,
        )

    def publish_authoring_save_snapshot(self, snapshot):
        self.path = snapshot.target_path
        self.saved = copy.deepcopy(snapshot.payload)
        return ""

    def discard(self, *, document_id):
        if document_id != self.document_id:
            return False
        self.data = copy.deepcopy(self.saved)
        return True

    def open_document_resource_immediate(self, path):
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        _backend_name(data.get("physics_backend", "cpu"))
        self.path = path
        self.data = data
        self.saved = copy.deepcopy(data)
        return True

    def resource_moved(self, *, document_id, source_path, destination_path, guid):
        if document_id == self.document_id:
            self.path = destination_path

    def capture_document_restore_state(self, document_id):
        return self.path, self.capture_authoring_snapshot(), copy.deepcopy(self.saved)

    def restore_document_restore_state(self, state):
        self.path, self.data, self.saved = copy.deepcopy(state)


def _draft(document):
    controller = document.controller
    if not isinstance(controller, AuthoringDocumentController):
        raise RuntimeError("Scene asset has no authoring document")
    model = controller.view
    if not isinstance(model, _SceneAssetDraft):
        raise RuntimeError("Scene asset has another document owner")
    return model


def _read_backend(document):
    scene = _loaded_scene(document.document_id)
    return _backend_name(scene.physics_backend) if scene is not None else _backend_name(
        _draft(document).data.get("physics_backend", "cpu")
    )


class _SetSceneBackend(UndoCommand):
    # AuthoringMutationService's document adapter owns the revision exactly once.
    marks_dirty = False

    def __init__(self, document, old, new):
        super().__init__(t("asset.scene_physics_backend"))
        self._locator = DocumentRegistry.instance().locate(document.document_id)
        self._old, self._new = old, new
        scene = _loaded_scene(document.document_id)
        self._scene_world_id = int(scene.world_id) if scene is not None else 0

    def _apply(self, value):
        document = DocumentRegistry.instance().resolve_locator(self._locator)
        if document is None:
            raise RuntimeError("The selected Scene document is no longer open")
        if not _can_edit(document.document_id):
            raise RuntimeError("Stop Play before changing a loaded Scene physics backend")
        scene = _loaded_scene(document.document_id)
        if scene is not None:
            from infernux.physics import PhysicsBackend
            scene.physics_backend = PhysicsBackend.CPU if value == "cpu" else PhysicsBackend.GPU
        else:
            _draft(document).data["physics_backend"] = value

    def execute(self):
        self._apply(self._new)

    def undo(self):
        self._apply(self._old)


def set_scene_backend(document_id, value):
    """One document-bound action, shared by the field and automation tests."""
    document = DocumentRegistry.instance().require(document_id)
    value = _backend_name(value)
    old = _read_backend(document)
    if old == value:
        return False
    if not _can_edit(document_id):
        return False
    return AuthoringMutationService.require().execute_command(
        document_id, lambda _before, _after: _SetSceneBackend(document, old, value), view_id="inspector",
    )


def load_scene_asset(path):
    if path.startswith("scene-document:"):
        document_id = path.removeprefix("scene-document:")
        DocumentRegistry.instance().require(document_id)
        return document_id, {}
    from infernux.core.assets import AssetManager
    database = AssetManager.require_asset_database()
    guid = str(database.get_guid_from_path(path) or "")
    if not guid:
        raise ValueError("Scene Inspector requires a registered Scene asset")
    registry = DocumentRegistry.instance()
    key = DocumentKey.asset(DocumentKind.SCENE, guid)
    document = registry.get_by_key(key)
    if document is None:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Scene source must contain an object")
        _backend_name(data.get("physics_backend", "cpu"))
        model = _SceneAssetDraft(path, data)
        controller = AuthoringDocumentController(model)
        document, _ = registry.open_or_create(
            key, os.path.splitext(os.path.basename(path))[0], resource_path=path,
            capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
            controller=controller,
        )
        controller.bind_resource_document(document.document_id)
    return document.document_id, {}


def render_scene_asset(ctx, panel, state):
    from .inspector_utils import field_label, max_label_w, render_compact_section_header
    registry = DocumentRegistry.instance()
    document = registry.get(state.settings)
    if document is None:
        ctx.label(t("asset.scene_unavailable"))
        return
    state.document_id = document.document_id
    registry.attach_view(document.document_id, "inspector")
    ctx.label(f"{t('asset.display_scene')}: {document.title}")
    if document.resource_path:
        ctx.label(t("asset.path_label").format(path=document.resource_path))
    ctx.separator()
    if not render_compact_section_header(ctx, t("asset.scene_physics"), level="primary"):
        return
    editable = _can_edit(document.document_id)
    value = _read_backend(document)
    label = t("asset.scene_physics_backend")
    field_label(ctx, label, max_label_w(ctx, [label]))
    ctx.begin_disabled(not editable)
    try:
        chosen = ctx.combo("##scene_physics_backend", 0 if value == "cpu" else 1, ["CPU", "GPU"])
        ctx.record_semantic_item("combo", label, editable, "asset.scene.physics_backend")
        if editable and chosen in (0, 1):
            desired = "cpu" if chosen == 0 else "gpu"
            if desired != value:
                set_scene_backend(document.document_id, desired)
    finally:
        ctx.end_disabled()
    ctx.text_wrapped(t("asset.scene_backend_scope"))
    if not editable:
        ctx.text_wrapped(t("asset.scene_backend_stop_play"))
    ctx.begin_disabled(not document.is_dirty or registry.is_save_pending(document.document_id) or not editable)
    try:
        if ctx.button(t("asset.scene_save")) and editable:
            registry.request_save(document.document_id)
        ctx.record_semantic_item("button", t("asset.scene_save"), editable and document.is_dirty, "asset.scene.save")
    finally:
        ctx.end_disabled()
