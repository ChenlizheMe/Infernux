"""ScenePrefabMixin — extracted from SceneFileManager."""
from __future__ import annotations

"""
Scene file management for Infernux.

Handles:
- Tracking the current scene file path (.scene)
- Saving / loading scene files (delegates to C++ Scene::SaveToFile / LoadFromFile)
- Python component serialization during save, recreation during load
- Remembering last opened scene per project (EditorSettings.json)
- Default scene fallback when a scene file is missing
- File-dialog for "Save As" when the scene has no file yet
- Enforcing that scenes must be saved under Assets/

The C++ layer already provides ``Scene.serialize / deserialize / save_to_file /
load_from_file`` and ``PendingPyComponent`` for Python component recreation.
This module orchestrates those primitives into a complete workflow.
"""

import os
import copy
from typing import Optional, Callable

from Infernux.debug import Debug
from Infernux.engine.project_context import get_project_root
from Infernux.engine.path_utils import resolved_path, safe_path as _safe_path
from .scene_manager import (
    DEFAULT_SCENE_NAME,
    PREFAB_MODE_SCENE_NAME,
    _empty_scene_document,
    _get_scene_root_objects,
)


class ScenePrefabMixin:
    """ScenePrefabMixin method group for SceneFileManager."""

    def open_prefab_mode(self, prefab_path: str, preserve_undo_history: bool = False):
        """Enter Prefab Mode, pushing the current scene to memory."""
        if self.is_prefab_mode or not prefab_path or not os.path.isfile(prefab_path):
            return False

        if self._is_play_mode():
            Debug.log_warning("Cannot enter Prefab Mode while in Play mode.")
            return False

        from Infernux.lib import SceneManager
        from Infernux.engine.component_restore import (
            deserialize_scene_document_transactionally,
            instantiate_prepared_game_object_document,
            preflight_game_object_python_components,
            serialize_game_object_document_authoritatively,
        )
        from Infernux.engine.prefab_manager import (
            _read_prefab_document,
            _stamp_prefab_guid,
            _strip_prefab_runtime_fields,
        )
        from Infernux.engine.interaction import SelectionService

        active_scene = SceneManager.instance().get_active_scene()
        if active_scene is None:
            Debug.log_warning("No active scene available for Prefab Mode.")
            return False

        try:
            prefab_data = _read_prefab_document(prefab_path)
        except (OSError, ValueError) as exc:
            Debug.log_error(f"Failed to open prefab for Prefab Mode: {exc}")
            return False

        root_obj_data = copy.deepcopy(prefab_data["root_object"])
        _strip_prefab_runtime_fields(root_obj_data)
        _stamp_prefab_guid(root_obj_data, "")
        try:
            prepared_prefab = preflight_game_object_python_components(
                root_obj_data,
                asset_database=self._asset_database,
                preserve_document_ids=False,
            )
        except RuntimeError as exc:
            Debug.log_error(f"Failed to preflight prefab for Prefab Mode: {exc}")
            return False

        self._previous_scene_document = active_scene.serialize_document()
        self._previous_scene = active_scene
        self._previous_scene_document["objects"] = [
            serialize_game_object_document_authoritatively(obj)
            for obj in _get_scene_root_objects(active_scene)
        ]
        self._previous_scene_path = self._current_scene_path
        self._previous_scene_document_id = self._scene_document_id
        self.prefab_envelope = prefab_data
        self._prefab_entry_document = copy.deepcopy(prefab_data)

        # Clear the RenderStack singleton before the swap — matches the
        # pattern in _do_open_scene / _do_new_scene to avoid stale refs.
        from Infernux.renderstack.render_stack import RenderStack
        RenderStack.clear_active_instance(active_scene)

        self._prepare_native_scene_swap()

        # Destroy ALL objects in the original scene so their physics bodies
        # are removed from the global PhysicsWorld.  Without this, invisible
        # colliders from the main scene interfere with the prefab scene.
        if not deserialize_scene_document_transactionally(
            active_scene,
            _empty_scene_document(active_scene.name),
            asset_database=self._asset_database,
            clear_registries=True,
        ):
            prepared_prefab.discard()
            Debug.log_error("Failed to clear the previous scene for Prefab Mode.")
            return False

        sm = SceneManager.instance()
        new_scene = sm.get_scene(PREFAB_MODE_SCENE_NAME)
        if new_scene is None:
            new_scene = sm.create_scene(PREFAB_MODE_SCENE_NAME)
        # Always deserialize empty JSON to clear old objects and component
        # registries (MeshRenderer, physics, etc.) — prevents the previous
        # scene's renderers from leaking into Prefab Mode.
        if not deserialize_scene_document_transactionally(
            new_scene,
            _empty_scene_document(PREFAB_MODE_SCENE_NAME),
            asset_database=self._asset_database,
            clear_registries=True,
        ):
            prepared_prefab.discard()
            Debug.log_error("Failed to initialize the Prefab Mode scene.")
            return False
        sm.set_active_scene(new_scene)

        try:
            root_obj = instantiate_prepared_game_object_document(
                new_scene,
                root_obj_data,
                prepared_prefab,
            )
        except RuntimeError as exc:
            Debug.log_error(f"Failed to preflight prefab for Prefab Mode: {exc}")
            return False
        if root_obj is None:
            Debug.log_error("Failed to instantiate prefab in Prefab Mode.")
            return False

        roots = _get_scene_root_objects(new_scene)
        if roots:
            SelectionService.instance().select_scene_object(
                roots[0].id,
                owner_id="hierarchy",
                reason="enter_prefab_mode",
                record_history=False,
            )

        self.is_prefab_mode = True
        self.prefab_mode_path = resolved_path(prefab_path)
        self._current_scene_path = prefab_path
        self._replace_scene_document(
            kind="prefab",
            resource_path=self.prefab_mode_path,
            title=os.path.splitext(os.path.basename(self.prefab_mode_path))[0],
            dirty=False,
            preserve_previous=True,
        )
        if not preserve_undo_history:
            self._reset_undo_history()

        if self._on_scene_changed:
            self._on_scene_changed()
        return True

    def exit_prefab_mode(self):
        """Resolve the Prefab document, then schedule its deferred exit."""
        return self._request_prefab_exit()

    def _request_prefab_exit(
        self,
        on_complete: Optional[Callable[[], None]] = None,
        *,
        preserve_undo_history: bool = False,
    ) -> bool:
        """Route every Prefab replacement through the shared close transaction."""
        if not self.is_prefab_mode:
            return False

        from Infernux.engine.ui.dirty_panel_confirmation import (
            DirtyPanelConfirmationCoordinator,
        )

        return DirtyPanelConfirmationCoordinator.instance().request_document_replace(
            self.document_id,
            on_complete=lambda: self._schedule_prefab_exit(
                on_complete,
                preserve_undo_history=preserve_undo_history,
            ),
        )

    def _schedule_prefab_exit(
        self,
        on_complete: Optional[Callable[[], None]] = None,
        *,
        preserve_undo_history: bool = False,
    ) -> bool:
        """Schedule exit from Prefab Mode on a later frame.

        Uses ``DeferredTaskRunner`` instead of ``poll_deferred_load`` so the
        exit cannot be consumed again later in the same GUI frame. This avoids
        tearing down the prefab scene while its resources may still be in use
        by the just-submitted frame.
        """
        if not self.is_prefab_mode:
            return False
        if self._deferred_exit_prefab:
            return False

        from Infernux.engine.deferred_task import DeferredTaskRunner

        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            Debug.log_warning("Cannot exit Prefab Mode: a deferred task is already running")
            return False

        self._deferred_exit_prefab = True
        self._post_prefab_exit_callback = on_complete
        submitted = runner.submit(
            "Exit Prefab Mode",
            [
                (
                    "Exiting Prefab Mode...",
                    0.6,
                    lambda: self._run_deferred_exit_prefab_task(
                        preserve_undo_history=preserve_undo_history,
                    ),
                )
            ],
        )
        if not submitted:
            self._deferred_exit_prefab = False
            self._post_prefab_exit_callback = None
        return bool(submitted)

    def _run_deferred_exit_prefab_task(
        self,
        *,
        preserve_undo_history: bool = False,
    ):
        """DeferredTaskRunner step wrapper for prefab-mode exit."""
        self._deferred_exit_prefab = False
        callback = self._post_prefab_exit_callback
        self._post_prefab_exit_callback = None
        succeeded = bool(
            self._do_exit_prefab_mode(
                preserve_undo_history=preserve_undo_history,
            )
        )
        if succeeded and callable(callback):
            callback()
        return succeeded

    def _do_exit_prefab_mode(self, preserve_undo_history: bool = False):
        """Internal: perform the actual Prefab Mode exit (called by poll_deferred_load)."""
        if not self.is_prefab_mode:
            return False

        from Infernux.lib import SceneManager
        from Infernux.engine.component_restore import deserialize_scene_document_transactionally

        # Resolve the prefab GUID so instances are refreshed from whichever
        # on-disk state the user explicitly chose (Save or Discard).
        saved_prefab_guid = None
        if self.prefab_mode_path and self._asset_database:
            try:
                saved_prefab_guid = self._asset_database.get_guid_from_path(
                    self.prefab_mode_path
                ) or None
            except Exception as exc:
                Debug.log_suppressed("ScenePrefabMixin.exit_prefab_mode.resolve_prefab_guid", exc)

        # Clear the RenderStack singleton before the swap — matches the
        # pattern in _do_open_scene / _do_new_scene to avoid stale refs.
        from Infernux.renderstack.render_stack import RenderStack
        sm = SceneManager.instance()
        RenderStack.clear_active_instance(sm.get_active_scene())

        self._prepare_native_scene_swap()

        # Destroy all objects in the prefab scene FIRST so their physics
        # bodies (Colliders, Rigidbodies) are removed from the global
        # PhysicsWorld before we restore the main scene.
        prefab_scene = sm.get_scene(PREFAB_MODE_SCENE_NAME)
        if prefab_scene is not None:
            if not deserialize_scene_document_transactionally(
                prefab_scene,
                _empty_scene_document(PREFAB_MODE_SCENE_NAME),
                asset_database=self._asset_database,
                clear_registries=True,
            ):
                Debug.log_error("Cannot exit Prefab Mode: failed to clear prefab scene.")
                return False

        # Keep the native Scene bound to the existing editor document. Creating
        # a replacement Scene leaves additive/save routing pointing at the
        # emptied original Scene even when the viewport looks restored.
        scene = self._previous_scene
        sm.set_active_scene(scene)
        if prefab_scene is not None:
            sm.unload_scene(prefab_scene)

        if self._previous_scene_document:
            if not deserialize_scene_document_transactionally(
                scene,
                self._previous_scene_document,
                asset_database=self._asset_database,
                clear_registries=True,
            ):
                Debug.log_error("Cannot exit Prefab Mode: previous scene transaction failed.")
                return False
        elif not deserialize_scene_document_transactionally(
            scene,
            _empty_scene_document(scene.name),
            asset_database=self._asset_database,
            clear_registries=True,
        ):
            Debug.log_error("Cannot exit Prefab Mode: failed to initialize restore scene.")
            return False

        # Merge the saved source delta onto each restored instance. Rebuilding
        # from the asset would discard scene overrides and change reference IDs.
        instances_changed = False
        if saved_prefab_guid:
            from Infernux.engine.prefab_manager import _read_prefab_document
            from Infernux.engine.prefab_overrides import (
                _snapshot_linked_instances, _propagate_applied_prefab,
            )
            base_root = self._prefab_entry_document["root_object"]
            updated_root = _read_prefab_document(self.prefab_mode_path)["root_object"]
            if base_root != updated_root:
                snapshots = _snapshot_linked_instances(
                    scene, saved_prefab_guid, base_root=base_root,
                )
                instances_changed = bool(snapshots)
                if not _propagate_applied_prefab(
                    base_root, updated_root, snapshots, saved_prefab_guid, self._asset_database,
                ):
                    return False

        from Infernux.engine.interaction import DocumentRegistry

        registry = DocumentRegistry.instance()
        prefab_document_id = self._scene_document_id
        previous_document_id = self._previous_scene_document_id
        self.is_prefab_mode = False
        self.prefab_mode_path = None
        self._current_scene_path = self._previous_scene_path
        if previous_document_id and registry.get(previous_document_id) is not None:
            self._scene_document_id = previous_document_id
        else:
            self._replace_scene_document(
                kind="scene",
                resource_path=self._current_scene_path or "",
                title=(
                    os.path.splitext(os.path.basename(self._current_scene_path))[0]
                    if self._current_scene_path
                    else DEFAULT_SCENE_NAME
                ),
                dirty=False,
            )
        if instances_changed:
            registry.mark_changed(self._scene_document_id)
        self.prefab_envelope = {}
        self._prefab_entry_document = None
        self._previous_scene_document = None
        self._previous_scene = None
        self._previous_scene_document_id = ""
        self._previous_scene_path = None
        prefab_document = registry.get(prefab_document_id)
        if prefab_document is not None and not prefab_document.view_ids:
            registry.unregister(prefab_document_id)
        if not preserve_undo_history:
            self._reset_undo_history()

        if self._on_scene_changed:
            self._on_scene_changed()
        return True

    @staticmethod
    def _refresh_prefab_instances(scene, prefab_guid: str, prefab_path: str,
                                  asset_database=None):
        """Merge source changes against each instance's persisted baseline."""
        from Infernux.engine.prefab_manager import _read_prefab_document
        from Infernux.engine.prefab_overrides import (
            _snapshot_linked_instances, _propagate_applied_prefab,
        )

        updated_root = _read_prefab_document(prefab_path)["root_object"]
        snapshots = [
            snapshot for snapshot in _snapshot_linked_instances(scene, prefab_guid, base_root=updated_root)
            if snapshot[1].get("prefab_source") != updated_root
        ]
        if not snapshots:
            return False
        if not _propagate_applied_prefab(None, updated_root, snapshots, prefab_guid, asset_database):
            raise RuntimeError(f"Failed to synchronize prefab instances: {prefab_path}")
        return True

    def sync_all_prefab_instances(self, scene=None):
        """Sync every prefab instance in *scene* to its latest on-disk data.

        Called after scene load and after exiting Prefab Mode so that all
        prefab instances reflect the most recent prefab files.
        """
        if scene is None:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
        if scene is None or not self._asset_database:
            return

        roots = _get_scene_root_objects(scene)
        if not roots:
            return

        # Collect unique (prefab_guid → prefab_path) pairs
        guid_to_path: dict[str, str] = {}

        def _walk(objects):
            for obj in objects:
                guid = getattr(obj, 'prefab_guid', '')
                is_root = getattr(obj, 'prefab_root', False)
                if guid and is_root and guid not in guid_to_path:
                    try:
                        p = self._asset_database.get_path_from_guid(guid)
                        if p and os.path.isfile(p):
                            guid_to_path[guid] = p
                    except Exception as exc:
                        Debug.log_suppressed(
                            f"ScenePrefabMixin.refresh_prefab_instances.resolve[{guid[:8]}]",
                            exc,
                        )
                children = list(obj.get_children()) if hasattr(obj, 'get_children') else []
                _walk(children)

        _walk(roots)

        changed = False
        for guid, path in guid_to_path.items():
            changed |= self._refresh_prefab_instances(
                scene, guid, path, self._asset_database
            )
        if changed:
            from Infernux.engine.interaction import DocumentRegistry
            document_id = self.document_id_for_scene(scene)
            if document_id:
                DocumentRegistry.instance().mark_changed(document_id)

