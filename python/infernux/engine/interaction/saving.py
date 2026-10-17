"""Single focus-aware save authority for Editor commands and automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .contexts import FocusService, FocusSnapshot
from .documents import DocumentActionResult, DocumentActionStatus, DocumentKind, DocumentRegistry

if TYPE_CHECKING:
    from .panels import PanelInteractionRegistry


@dataclass(frozen=True, slots=True)
class FocusedSaveResult:
    result: DocumentActionResult
    target: str = ""
    document_id: str = ""
    panel_id: str = ""
    dirty_before: bool = False

    @property
    def accepted(self) -> bool:
        return self.result.accepted


class EditorSaveService:
    """Save the active authoring view while auxiliary panels own keyboard focus."""

    _instance: Optional["EditorSaveService"] = None

    def __init__(self, registry: DocumentRegistry, panels: PanelInteractionRegistry) -> None:
        self._registry = registry
        self._panels = panels
        self._focus = FocusService.instance()
        self._last_primary_view_id = ""
        self._focus.add_listener(self._observe_focus)
        self._observe_focus(self._focus.snapshot)
        EditorSaveService._instance = self

    def _primary_document(self, view_id: str):
        document = self._registry.document_for_view(view_id)
        if document is not None and (
            document.kind in {DocumentKind.SCENE, DocumentKind.PREFAB}
            or self._panels.is_document_backed(type_id=view_id, view_id=view_id)
        ):
            return document
        return None

    def _observe_focus(self, snapshot: FocusSnapshot) -> None:
        view_id = snapshot.active_view_id or snapshot.active_panel_id
        if self._primary_document(view_id) is not None:
            self._last_primary_view_id = view_id

    @classmethod
    def instance(cls) -> Optional["EditorSaveService"]:
        return cls._instance

    def save_focused(self, *, save_as: bool = False) -> FocusedSaveResult:
        from .continuous_edits import ContinuousEditService

        focus = self._focus.snapshot
        self._observe_focus(focus)
        view_id = self._last_primary_view_id
        document = self._primary_document(view_id) if view_id else None
        if view_id and document is None:
            return FocusedSaveResult(
                DocumentActionResult(
                    DocumentActionStatus.REJECTED,
                    "last authoring view no longer owns a document",
                ),
                panel_id=view_id,
            )

        if document is None:
            try:
                from infernux.engine.scene_manager import SceneFileManager

                scene_files = SceneFileManager.instance()
            except (ImportError, RuntimeError):
                scene_files = None
            scene_document_id = str(getattr(scene_files, "document_id", "") or "")
            document = self._registry.get(scene_document_id) if scene_document_id else None
            if document is None:
                return FocusedSaveResult(
                    DocumentActionResult(
                        DocumentActionStatus.REJECTED,
                        "no focused document or active scene document is available",
                    ),
                    panel_id=focus.active_panel_id,
                )

        ContinuousEditService.instance().commit_document(document.document_id)
        dirty_before = bool(document.is_dirty)
        if document.kind is DocumentKind.SCENE and not save_as:
            from infernux.engine.scene_manager import SceneFileManager

            scene_files = SceneFileManager.instance()
            if scene_files is None:
                raise RuntimeError("Scene save requires an active SceneFileManager")
            scene_document_ids = scene_files.loaded_scene_document_ids()

            resident_documents = tuple(
                candidate
                for identifier in scene_document_ids
                if (candidate := self._registry.get(identifier)) is not None
                and candidate.kind is DocumentKind.SCENE
            )
            continuous_edits = ContinuousEditService.instance()
            for candidate in resident_documents:
                continuous_edits.commit_document(candidate.document_id)
            dirty_documents = tuple(
                candidate for candidate in resident_documents if candidate.is_dirty
            )
            if not dirty_documents:
                return FocusedSaveResult(
                    DocumentActionResult(DocumentActionStatus.NO_OP),
                    target="scenes",
                    document_id=document.document_id,
                    dirty_before=False,
                )
            for candidate in dirty_documents:
                result = self._registry.defer_save(candidate.document_id)
                if not result.accepted:
                    return FocusedSaveResult(
                        result,
                        target="scenes",
                        document_id=candidate.document_id,
                        dirty_before=True,
                    )
            return FocusedSaveResult(
                DocumentActionResult(DocumentActionStatus.PENDING),
                target="scenes",
                document_id=document.document_id,
                dirty_before=True,
            )

        result = self._registry.defer_save(document.document_id, save_as=save_as)
        return FocusedSaveResult(
            result,
            target="scene" if document.kind is DocumentKind.SCENE else "document",
            document_id=document.document_id,
            panel_id=view_id if document.kind is not DocumentKind.SCENE else "",
            dirty_before=dirty_before,
        )

    def shutdown(self) -> None:
        self._focus.remove_listener(self._observe_focus)
        self._last_primary_view_id = ""
        if EditorSaveService._instance is self:
            EditorSaveService._instance = None
