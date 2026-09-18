"""Single command authority for editor-visible Prefab operations."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from Infernux.debug import Debug
from Infernux.engine.path_utils import is_path_within, resolved_path, same_path

from .action_journal import ActionOrigin
from .descriptors import SelectionTarget
from .document_open import DocumentOpenStatus
from .documents import DocumentKind
from .navigation import NavigationService
from .project_assets import ProjectAssetCommandService
from .selection import SelectionService


class PrefabCommandService:
    """Resolve and execute every Prefab command independently of Panel focus."""

    _instance: Optional["PrefabCommandService"] = None

    def __init__(
        self,
        selection: SelectionService,
        navigation: NavigationService,
        document_open,
        project_assets: ProjectAssetCommandService,
        context_provider: Optional[Callable[[], object]] = None,
    ) -> None:
        self._selection = selection
        self._navigation = navigation
        self._document_open = document_open
        self._project_assets = project_assets
        self._context_provider = context_provider
        self._contents = {}
        PrefabCommandService._instance = self

    @classmethod
    def instance(cls) -> Optional["PrefabCommandService"]:
        return cls._instance

    def can_execute(
        self,
        action: str,
        *,
        object_id: int = 0,
        path: str = "",
    ) -> bool:
        action = str(action or "").strip().lower()
        if action == "create":
            return self._scene_object(object_id) is not None
        if action in {"locate", "open"}:
            return bool(self._resolve_path(object_id=object_id, path=path))
        if action == "exit":
            from Infernux.engine.scene_manager import SceneFileManager

            scene_files = SceneFileManager.instance()
            return bool(scene_files and scene_files.is_prefab_mode)
        if action in {"apply", "revert", "unpack"}:
            root = self._instance_root(object_id)
            if root is None:
                return False
            if action == "unpack":
                return True
            return bool(self._resolve_path(instance_root=root))
        return False

    def create_from_object(
        self,
        object_id: int,
        current_path: str = "",
        *,
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> str:
        game_object = self._scene_object(object_id)
        if game_object is None:
            raise ValueError(f"Prefab source object is unavailable: {object_id}")
        destination = resolved_path(current_path) if current_path else ""
        if not destination:
            destination = os.path.join(self._project_assets.project_root, "Assets")
        scene = getattr(game_object, "scene", None)
        if scene is None:
            raise RuntimeError("Prefab source object is not bound to an active scene")

        def capture_linkage():
            snapshot = []
            pending = [game_object]
            while pending:
                obj = pending.pop()
                snapshot.append((
                    int(obj.id),
                    str(getattr(obj, "prefab_guid", "") or ""),
                    bool(getattr(obj, "prefab_root", False)),
                    int(getattr(obj, "prefab_source_id", 0)),
                    obj._prefab_source_document,
                ))
                pending.extend(list(obj.get_children()))
            return tuple(snapshot)

        def restore_linkage(snapshot) -> None:
            for linked_id, prefab_guid, prefab_root, source_id, source_document in snapshot or ():
                obj = scene.find_by_id(int(linked_id))
                if obj is None:
                    raise RuntimeError(
                        f"Prefab source object {linked_id} is unavailable"
                    )
                obj.prefab_guid = prefab_guid
                obj.prefab_root = prefab_root
                obj.prefab_source_id = source_id
                obj._prefab_source_document = source_document

        source_canvas_name = self._source_canvas_name(game_object)
        from Infernux.engine.ui import project_file_ops

        result = self._project_assets.create_prefab(
            destination,
            lambda: project_file_ops.create_prefab_from_gameobject(
                game_object,
                destination,
                self._project_assets.asset_database,
                source_canvas_name=source_canvas_name,
            ),
            capture_linkage,
            restore_linkage,
            origin=origin,
        )
        if not isinstance(result, tuple) or len(result) < 2 or not result[0]:
            raise RuntimeError(f"Prefab creation failed: {result!r}")
        return resolved_path(result[1])

    def locate(
        self,
        *,
        object_id: int = 0,
        path: str = "",
        record_history: bool = True,
    ) -> bool:
        prefab_path = self._resolve_path(object_id=object_id, path=path)
        if not prefab_path:
            return False
        return self._navigation.locate(
            SelectionTarget.asset(prefab_path),
            owner_id="prefab",
            reason="prefab_locate",
            record_history=record_history,
        )

    def open(
        self,
        *,
        object_id: int = 0,
        path: str = "",
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> bool:
        prefab_path = self._resolve_path(object_id=object_id, path=path)
        if not prefab_path:
            return False
        from Infernux.engine.scene_manager import SceneFileManager

        scene_files = SceneFileManager.instance()
        if scene_files is None:
            return False
        if scene_files.is_prefab_mode:
            return same_path(scene_files.prefab_mode_path or "", prefab_path)
        from Infernux.engine.undo import PrefabModeCommand

        self._execute(PrefabModeCommand(prefab_path, enter_mode=True), origin)
        result = self._document_open.open_resource(
            DocumentKind.PREFAB,
            prefab_path,
        )
        if result.status is DocumentOpenStatus.FAILED:
            raise RuntimeError(result.message or "Prefab document did not open")
        return result.status is DocumentOpenStatus.READY

    def apply(
        self,
        object_id: int,
        *,
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> bool:
        root = self._require_instance_root(object_id)
        prefab_path = self._require_instance_path(root)
        from Infernux.engine.prefab_overrides import build_prefab_apply_command

        self._execute(
            build_prefab_apply_command(
                root,
                prefab_path,
                self._project_assets.asset_database,
            ),
            origin,
        )
        return True

    def exit(
        self,
        *,
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> bool:
        """Resolve close policy, then record the completed mode transition."""
        from Infernux.engine.scene_manager import SceneFileManager
        from Infernux.engine.undo import PrefabModeCommand, UndoManager

        scene_files = SceneFileManager.instance()
        manager = UndoManager.instance()
        if (
            scene_files is None
            or not scene_files.is_prefab_mode
            or manager is None
            or not manager.enabled
            or manager.is_executing
            or self._context_provider is None
        ):
            return False
        prefab_path = str(scene_files.prefab_mode_path or "")
        before_context = self._context_provider()

        def record_completed_exit() -> None:
            command = PrefabModeCommand(prefab_path, enter_mode=False)
            if manager.record(
                command,
                before_context=before_context,
                after_context=self._context_provider(),
                origin=ActionOrigin(origin),
            ):
                return
            if not scene_files.open_prefab_mode(
                prefab_path,
                preserve_undo_history=True,
            ):
                Debug.log_error(
                    "Exit Prefab Mode completed but could not be recorded or rolled back"
                )

        return bool(scene_files._request_prefab_exit(
            on_complete=record_completed_exit,
            preserve_undo_history=True,
        ))

    def revert(
        self,
        object_id: int,
        *,
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> bool:
        root = self._require_instance_root(object_id)
        prefab_path = self._require_instance_path(root)
        from Infernux.engine.prefab_overrides import build_prefab_revert_command

        self._execute(
            build_prefab_revert_command(
                root,
                prefab_path,
                self._project_assets.asset_database,
            ),
            origin,
        )
        self._selection.select(
            SelectionTarget.scene_object(root.id),
            owner_id="prefab",
            reason="prefab_revert",
            record_history=False,
        )
        return True

    def unpack(
        self,
        object_id: int,
        *,
        origin: ActionOrigin = ActionOrigin.USER,
    ) -> bool:
        root = self._require_instance_root(object_id)
        from Infernux.engine.undo import PrefabUnpackCommand

        self._execute(PrefabUnpackCommand(root.id), origin)
        return True

    def revert_property(self, component, field_name: str, *,
                        origin: ActionOrigin = ActionOrigin.USER) -> bool:
        root = self._require_instance_root(component.game_object.id)
        path = self._require_instance_path(root)
        from Infernux.engine.prefab_overrides import build_prefab_property_revert_command

        command = build_prefab_property_revert_command(
            component, field_name, path, self._project_assets.asset_database,
        )
        if command is None:
            return False
        self._execute(command, origin)
        return True

    def property_modifications(self, object_id: int):
        from Infernux.engine.prefab_overrides import get_property_modifications

        root = self._instance_root(object_id)
        if root is None:
            return ()
        return get_property_modifications(root, self._require_instance_path(root))

    def is_property_override(self, component, field_name: str) -> bool:
        from Infernux.engine.prefab_overrides import is_property_override

        root = self._instance_root(component.game_object.id)
        return is_property_override(
            component, field_name, self._require_instance_path(root) if root else "",
        )

    def load_contents(self, path: str):
        """Load an asset into an isolated authoring world, not the active Scene."""
        from Infernux.lib import SceneManager
        from Infernux.engine import prefab_manager as pm
        from Infernux.engine.component_restore import (
            preflight_game_object_python_components, instantiate_prepared_game_object_document,
        )

        target = self._project_assets._registered_file(path)
        self._require_closed_prefab_mode(target)
        if not self._is_prefab_asset(target):
            raise ValueError("load_prefab_contents requires a .prefab asset")
        content = self._project_assets.read_text(target)
        document = pm._read_prefab_document(target)
        database = self._project_assets.asset_database
        guid = str(database.get_guid_from_path(target))
        payload = pm._load_prefab_template_payload(target, guid, database)
        if payload is None:
            raise ValueError(f"Cannot load Prefab contents: {target}")
        manager = SceneManager.instance()
        scene = manager._create_preview_scene(os.path.basename(target))
        try:
            # The Editor's published class is authoritative. Re-importing a
            # script here would attempt a new dispatch publication mid-UI-frame.
            prepared = preflight_game_object_python_components(
                payload, database, preserve_document_ids=False,
                prefer_loaded_types=True, reference_scene=scene,
            )
            root = instantiate_prepared_game_object_document(scene, payload, prepared)
            if root is None:
                raise RuntimeError(f"Cannot instantiate Prefab contents: {target}")
            self._contents[int(root.id)] = (scene, target, content, document)
            return root
        except BaseException:
            manager._close_preview_scene(scene)
            raise

    def save_contents(self, root, path: str) -> str:
        """Save detached contents, or create a new asset from an authored tree.

        Existing assets can only be replaced by contents loaded from that asset.
        Asset writes share Project Undo; editing the temporary tree does not.
        """
        import json
        from Infernux.engine import prefab_manager as pm

        target = self._project_assets._project_path(path)
        self._require_closed_prefab_mode(target)
        if not target.lower().endswith(pm.PREFAB_EXTENSION):
            raise ValueError("Prefab paths require the .prefab extension")
        session = self._contents.get(int(root.id))
        source = session[3] if session else None
        if session and same_path(session[1], target):
            if self._project_assets.read_text(target) != session[2]:
                raise RuntimeError("Prefab source changed since load; reload contents before saving")
            if str(self._project_assets.asset_database.get_guid_from_path(target)) != root.prefab_guid:
                raise RuntimeError("Prefab source identity changed since load; reload contents before saving")
        elif os.path.exists(target):
            raise FileExistsError("Load the target Prefab contents before overwriting it")
        document = pm.serialize_prefab_document(
            root, root_document_template=source["root_object"] if source else None,
            source_canvas_name=source.get("source_canvas_name", "") if source else "",
            next_local_id=source["next_local_id"] if source else 1,
            next_component_id=source["next_component_id"] if source else 1,
            preserve_root_properties=False,
        )
        database = self._project_assets.asset_database
        guid = str(database.get_guid_from_path(target) or "")
        pm._validate_nested_source_ancestry(document["root_object"], (guid,) if guid else ())
        content = json.dumps(document, indent=2, ensure_ascii=False)
        if os.path.exists(target):
            if self._project_assets.read_text(target) != content:
                from Infernux.engine.prefab_overrides import build_prefab_asset_edit_command

                self._execute(build_prefab_asset_edit_command(target, document, database), ActionOrigin.USER)
                self._project_assets._notify_changed()
        else:
            self._project_assets._save_resource_copy(
                lambda: content, target, extension=pm.PREFAB_EXTENSION,
                description="Save Prefab Asset", origin=ActionOrigin.USER,
            )
        guid = str(database.get_guid_from_path(target))
        pm._invalidate_prefab_template_cache(target, guid)
        if session:
            pm._link_prefab_hierarchy(root, document["root_object"], guid)
            self._contents[int(root.id)] = (session[0], target, content, document)
        return target

    @staticmethod
    def _require_closed_prefab_mode(path):
        from Infernux.engine.scene_manager import SceneFileManager

        scene_files = SceneFileManager.instance()
        if scene_files and scene_files.is_prefab_mode and same_path(scene_files.prefab_mode_path, path):
            raise RuntimeError("Close this asset's Prefab Mode before editing its offline contents")

    def unload_contents(self, root) -> None:
        from Infernux.lib import SceneManager

        identity = int(root.id)
        session = self._contents.get(identity)
        if session is None:
            raise ValueError("Expected a root returned by load_prefab_contents")
        SceneManager.instance()._close_preview_scene(session[0])
        del self._contents[identity]

    def shutdown(self) -> None:
        from Infernux.lib import SceneManager

        for scene, *_ in self._contents.values():
            SceneManager.instance()._close_preview_scene(scene)
        self._contents.clear()
        self._context_provider = None
        if PrefabCommandService._instance is self:
            PrefabCommandService._instance = None

    def _resolve_path(
        self,
        *,
        object_id: int = 0,
        path: str = "",
        instance_root: Any = None,
    ) -> str:
        explicit = resolved_path(path) if path else ""
        if explicit:
            return explicit if self._is_prefab_asset(explicit) else ""
        root = instance_root or self._instance_root(object_id)
        guid = str(getattr(root, "prefab_guid", "") or "") if root else ""
        asset_database = self._project_assets.asset_database
        if not guid or asset_database is None:
            return ""
        try:
            resolved = resolved_path(asset_database.get_path_from_guid(guid))
        except Exception as exc:
            Debug.log_suppressed("PrefabCommandService.resolve_path", exc)
            return ""
        return resolved if self._is_prefab_asset(resolved) else ""

    def _is_prefab_asset(self, path: str) -> bool:
        if not path or not path.lower().endswith(".prefab") or not os.path.isfile(path):
            return False
        project_root = self._project_assets.project_root
        return not project_root or is_path_within(path, project_root)

    def _require_instance_root(self, object_id: int):
        root = self._instance_root(object_id)
        if root is None:
            raise ValueError(f"Object is not part of a Prefab instance: {object_id}")
        return root

    def _require_instance_path(self, root) -> str:
        path = self._resolve_path(instance_root=root)
        if not path:
            raise RuntimeError("Prefab source asset is unavailable")
        return path

    @staticmethod
    def _scene_object(object_id: int):
        try:
            object_id = int(object_id)
        except (TypeError, ValueError):
            return None
        if object_id <= 0:
            return None
        from Infernux.lib import SceneManager

        manager = SceneManager.instance()
        scene = manager.get_active_scene() if manager else None
        return scene.find_by_id(object_id) if scene else None

    def _instance_root(self, object_id: int):
        from Infernux.engine.prefab_overrides import resolve_prefab_instance_root

        return resolve_prefab_instance_root(self._scene_object(object_id))

    @staticmethod
    def _source_canvas_name(game_object) -> str:
        current = game_object.get_parent()
        while current is not None:
            try:
                from Infernux.ui import UICanvas

                if any(
                    isinstance(component, UICanvas)
                    for component in (current.get_py_components() or [])
                ):
                    return str(current.name or "")
            except (AttributeError, ReferenceError, RuntimeError):
                pass
            current = current.get_parent()
        return ""

    @staticmethod
    def _execute(command, origin: ActionOrigin) -> None:
        from Infernux.engine.undo import UndoManager

        manager = UndoManager.instance()
        if manager is None or not manager.enabled or manager.is_executing:
            command.dispose()
            raise RuntimeError("Global editor history is unavailable")
        if not manager.execute(command, origin=ActionOrigin(origin)):
            raise RuntimeError(f"Prefab command was rejected: {command.description}")
