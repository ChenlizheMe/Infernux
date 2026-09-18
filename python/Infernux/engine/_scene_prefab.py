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
            _read_resolved_prefab_document,
            _load_prefab_template_payload,
        )
        from Infernux.engine.interaction import SelectionService

        active_scene = SceneManager.instance().get_active_scene()
        if active_scene is None:
            Debug.log_warning("No active scene available for Prefab Mode.")
            return False

        try:
            prefab_data = _read_resolved_prefab_document(prefab_path, self._asset_database)
        except (OSError, ValueError) as exc:
            Debug.log_error(f"Failed to open prefab for Prefab Mode: {exc}")
            return False

        root_obj_data = _load_prefab_template_payload(prefab_path, "", self._asset_database)
        if root_obj_data is None:
            return False
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
        self._prefab_variant_overrides = copy.deepcopy(prefab_data.get("variant", {}).get("property_overrides", []))
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
        self._prefab_mode_scene = new_scene
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

    def capture_prefab_mode_document(self):
        """Capture live values and draft inheritance intent in one source domain."""
        from Infernux.lib import SceneManager
        from Infernux.engine.component_restore import serialize_game_object_document_authoritatively
        from Infernux.engine.prefab_manager import _serialize_prefab_snapshot, _make_prefab_baseline
        from Infernux.engine.prefab_variant import edit_variant_document

        if not self.is_prefab_mode:
            raise RuntimeError("Prefab draft requires an open Prefab Mode")
        roots = _get_scene_root_objects(SceneManager.instance().get_active_scene())
        if len(roots) != 1:
            raise RuntimeError("Prefab Mode requires exactly one root object")
        snapshot = serialize_game_object_document_authoritatively(roots[0])
        authored = copy.deepcopy(snapshot)
        authored["prefab_source"] = _make_prefab_baseline(self.prefab_envelope["root_object"])
        document, objects, components = _serialize_prefab_snapshot(
            authored, source_canvas_name=self.prefab_envelope.get("source_canvas_name", ""),
            next_local_id=self.prefab_envelope.get("next_local_id", 1),
            next_component_id=self.prefab_envelope.get("next_component_id", 1))
        if "variant" in self.prefab_envelope:
            previous = copy.deepcopy(self.prefab_envelope)
            previous["variant"]["property_overrides"] = copy.deepcopy(self._prefab_variant_overrides)
            document = edit_variant_document(previous, document)
        return document, snapshot, objects, components

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

        # The suspended world may contain derived Variants and nested sources,
        # not only direct instances of the asset opened in Prefab Mode. Use the
        # same baseline-aware resolver as scene loading, preserving overrides
        # and runtime identities, and marking only actually changed documents.
        self.sync_all_prefab_instances(scene)

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
        self.prefab_envelope = {}
        self._prefab_variant_overrides = []
        self._prefab_mode_scene = None
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
        from Infernux.engine.prefab_manager import _read_resolved_prefab_document, _make_prefab_baseline
        from Infernux.engine.prefab_overrides import (
            _snapshot_linked_instances, _propagate_applied_prefab,
        )

        updated_root = _read_resolved_prefab_document(prefab_path, asset_database)["root_object"]
        snapshots = [
            snapshot for snapshot in _snapshot_linked_instances(scene, prefab_guid, base_root=updated_root)
            if snapshot[1].get("prefab_source") != _make_prefab_baseline(
                updated_root, outer_source_id=snapshot[1].get("prefab_source", {}).get("outer_source_id", 0),
            )
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
        self._sync_prefab_scenes((scene,))

    def sync_prefab_dependents(self, source_guid):
        """Publish an external source revision into affected open worlds only."""
        from Infernux.engine.prefab_variant import dependent_variant_guids
        from Infernux.engine.prefab_overrides import _loaded_prefab_scenes
        from Infernux.lib import SceneManager

        if not source_guid or self._asset_database is None:
            return
        affected = {source_guid, *dependent_variant_guids(source_guid, self._asset_database)}
        editing = SceneManager.instance().get_active_scene() if self.is_prefab_mode else None
        scenes = [scene for scene in _loaded_prefab_scenes() if scene is not editing
                  and any(obj.prefab_root and obj.prefab_guid in affected for obj in scene.get_all_objects())]
        self._sync_prefab_scenes(scenes, affected_guids=affected)

    def _sync_prefab_scenes(self, scenes, *, affected_guids=None):
        from Infernux.lib import GameObject
        from Infernux.engine.component_restore import (
            serialize_game_object_document_authoritatively,
            preflight_game_object_python_components,
        )
        from Infernux.engine.prefab_manager import _read_resolved_prefab_document, PrefabDocumentError
        from Infernux.engine.prefab_overrides import (
            resolve_scene_prefab_documents, _publish_applied_prefab,
        )

        sources = {}

        def load_source(guid):
            if guid not in sources:
                path = self._asset_database.get_path_from_guid(guid)
                if not path:
                    raise PrefabDocumentError(f"Prefab source cannot be resolved: {guid}")
                sources[guid] = _read_resolved_prefab_document(path, self._asset_database)["root_object"]
            return sources[guid]

        prepared_updates, originals, changed_worlds = [], [], []
        try:
            for scene in scenes:
                roots = _get_scene_root_objects(scene)
                before = {"objects": [serialize_game_object_document_authoritatively(root) for root in roots]}
                after = resolve_scene_prefab_documents(
                    before, load_source, reserve_ids=GameObject._reserve_document_ids, affected_guids=affected_guids,
                )
                for obj, old, new in zip(roots, before["objects"], after["objects"]):
                    if old == new:
                        continue
                    prepared = preflight_game_object_python_components(
                        new, self._asset_database, preserve_document_ids=True, reference_scene=scene,
                        prefer_loaded_types=True,
                    )
                    originals.append((obj, old, scene))
                    prepared_updates.append((obj, new, prepared))
                    changed_worlds.append(scene.world_id)
        except Exception:
            for _obj, _document, prepared in prepared_updates:
                prepared.discard()
            raise
        if not _publish_applied_prefab(prepared_updates):
            rollback = []
            try:
                for obj, old, scene in originals:
                    rollback.append((obj, old, preflight_game_object_python_components(
                        old, self._asset_database, preserve_document_ids=True, reference_scene=scene,
                        prefer_loaded_types=True,
                    )))
            except Exception:
                for _obj, _document, prepared in rollback:
                    prepared.discard()
                raise
            if not _publish_applied_prefab(rollback):
                raise RuntimeError("Failed to restore scene Prefab instances after publication failure")
            raise RuntimeError("Failed to synchronize scene Prefab instances")
        from Infernux.engine.interaction import DocumentRegistry
        for world_id in dict.fromkeys(changed_worlds):
            document_id = self.document_id_for_scene(world_id)
            if document_id:
                DocumentRegistry.instance().mark_changed(document_id)

