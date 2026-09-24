"""Stable editor-automation capability boundary for transport plugins."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Iterable

from Infernux.debug import DebugConsole
from Infernux.engine.path_utils import resolved_path

from .operations import OperationError


_LEVEL_ALIASES = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "LOG": "INFO",
    "WARN": "WARN",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "ASSERT": "ERROR",
    "EXCEPTION": "ERROR",
    "FATAL": "FATAL",
}


class EditorAutomationHost:
    """JSON-oriented access to editor capabilities used by automation plugins.

    This class is the supported boundary. Transport plugins should not import
    editor managers or native bindings directly. Tests may replace the process
    provider with :meth:`set_provider`.
    """

    _provider: "EditorAutomationHost | None" = None

    @classmethod
    def instance(cls) -> "EditorAutomationHost":
        if cls._provider is None:
            cls._provider = cls()
        return cls._provider

    @classmethod
    def set_provider(cls, provider: "EditorAutomationHost | None") -> None:
        cls._provider = provider

    def interaction_core(self):
        from Infernux.engine.interaction import EditorInteractionCore

        core = EditorInteractionCore.instance()
        if core is None:
            raise OperationError(
                "editor.unavailable", "Editor interaction services are unavailable."
            )
        return core

    def plugin_manager(self):
        from Infernux.plugins import PluginManager

        manager = PluginManager.instance()
        if manager is None:
            raise OperationError(
                "editor.unavailable", "Plugin project session is unavailable."
            )
        return manager

    def asset_database(self):
        database = self.interaction_core().project_assets.asset_database
        if database is None:
            raise OperationError("editor.unavailable", "AssetDatabase is unavailable.")
        return database

    def active_scene(self):
        from Infernux.lib import SceneManager

        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            raise OperationError("editor.unavailable", "No active scene is available.")
        return scene

    def project_info(self, project_root: str) -> dict[str, object]:
        from Infernux.engine.play_mode import PlayModeManager
        from Infernux.engine.scene_manager import SceneFileManager
        from Infernux.engine.interaction import DocumentRegistry

        scene_files = SceneFileManager.instance()
        play_mode = PlayModeManager.instance()
        scene = self.active_scene()
        document = (DocumentRegistry.instance().get(scene_files.document_id)
                    if scene_files is not None else None)
        return {
            "project_root": str(project_root),
            "active_scene": {
                "name": str(getattr(scene, "name", "")),
                "path": str(getattr(scene_files, "current_scene_path", ""))
                if scene_files
                else "",
                "dirty": bool(getattr(scene_files, "is_dirty", False))
                if scene_files
                else False,
                "document_state": document.state.value if document else "",
                "loading": bool(scene_files and scene_files.is_loading),
                "last_load": scene_files.last_scene_load if scene_files else None,
            },
            "play_state": str(
                getattr(getattr(play_mode, "state", None), "name", "edit")
            ).lower(),
        }

    def request_editor_close(self) -> None:
        from Infernux.engine.scene_manager import SceneFileManager

        manager = SceneFileManager.instance()
        if manager is None:
            raise OperationError(
                "editor.unavailable",
                "Scene lifecycle service is unavailable for normal shutdown.",
            )
        manager.request_close()

    def runtime_status(self) -> dict[str, object]:
        manager = self._play_mode_manager()
        return self._runtime_status_value(manager)

    def runtime_transition(self, method: str) -> dict[str, object]:
        manager = self._play_mode_manager()
        result = getattr(manager, str(method))()
        return {
            "accepted": True if result is None else bool(result),
            "runtime": self._runtime_status_value(manager),
        }

    def set_time_scale(self, value: float) -> dict[str, object]:
        manager = self._play_mode_manager()
        manager.time_scale = float(value)
        return {"runtime": self._runtime_status_value(manager)}

    def editor_camera_state(self) -> dict[str, object]:
        return self._camera_state(self._editor_camera())

    def restore_editor_camera(
        self,
        position: Iterable[float],
        focus: Iterable[float],
        distance: float,
        yaw: float,
        pitch: float,
    ) -> dict[str, object]:
        camera = self._editor_camera()
        camera.restore_state(
            *[float(value) for value in position],
            *[float(value) for value in focus],
            float(distance),
            float(yaw),
            float(pitch),
        )
        return self._camera_state(camera)

    def focus_editor_camera(
        self, point: Iterable[float], distance: float
    ) -> dict[str, object]:
        camera = self._editor_camera()
        camera.focus_on(*[float(value) for value in point], float(distance))
        return self._camera_state(camera)

    def game_camera_state(self) -> dict[str, object] | None:
        camera = self.active_scene().effective_game_camera
        if camera is None:
            return None
        owner = getattr(camera, "game_object", None)
        serializer = getattr(camera, "serialize_document", None)
        return {
            "object_id": int(getattr(owner, "id", 0) or 0),
            "component_id": int(getattr(camera, "component_id", 0) or 0),
            "document": serializer() if callable(serializer) else {},
        }

    def queue_input(self, kind: str, **arguments: object) -> dict[str, int]:
        native = self._native_engine()
        native.request_full_speed_frame()
        handlers = {
            "key": lambda: native.queue_synthetic_key_input(
                self.resolve_scancode(arguments["key"]),
                bool(arguments.get("pressed", True)),
                bool(arguments.get("repeat", False)),
            ),
            "pointer_move": lambda: native.queue_synthetic_mouse_motion_input(
                float(arguments["x"]),
                float(arguments["y"]),
                float(arguments.get("delta_x", 0.0)),
                float(arguments.get("delta_y", 0.0)),
            ),
            "pointer_button": lambda: native.queue_synthetic_mouse_button_input(
                int(arguments["button"]),
                bool(arguments["pressed"]),
                float(arguments["x"]),
                float(arguments["y"]),
            ),
            "wheel": lambda: native.queue_synthetic_mouse_wheel_input(
                float(arguments.get("horizontal", 0.0)),
                float(arguments.get("vertical", 0.0)),
            ),
            "text": lambda: native.queue_synthetic_text_input(
                str(arguments.get("text", ""))
            ),
            "close": native.queue_synthetic_close_request,
        }
        callback = handlers.get(str(kind))
        if callback is None:
            raise OperationError(
                "operation.invalid_arguments", f"Unknown input event kind: {kind}"
            )
        sequence = int(callback() or 0)
        if sequence <= 0:
            raise OperationError("input.rejected", "Synthetic input was rejected.")
        return {"sequence": sequence, **self.input_status()}

    def input_status(self) -> dict[str, int]:
        native = self._native_engine()
        return {
            "last_processed_sequence": int(
                native.last_processed_synthetic_input_sequence
            ),
            "pending_event_count": int(native.pending_synthetic_input_count),
        }

    def resolve_scancode(self, key: object) -> int:
        from Infernux.lib import InputManager

        aliases = {
            "ctrl": 224,
            "control": 224,
            "shift": 225,
            "alt": 226,
            "option": 226,
            "cmd": 227,
            "command": 227,
            "super": 227,
            "win": 227,
            "windows": 227,
            "esc": 41,
            # SDL scancode for Space. Keep the automation vocabulary
            # explicit because older native name tables did not expose the
            # literal "space" alias consistently across platforms.
            "space": 44,
        }
        if isinstance(key, bool):
            raise OperationError(
                "operation.invalid_arguments", "key cannot be a boolean"
            )
        if isinstance(key, int):
            result = key
        else:
            text = str(key).strip()
            candidate = aliases.get(text.casefold(), text)
            result = (
                int(candidate)
                if isinstance(candidate, int)
                else int(InputManager.name_to_scancode(candidate))
            )
        if result <= 0:
            raise OperationError("operation.invalid_arguments", f"Unknown key: {key}")
        return result

    def semantic_capture_enabled(self, enabled: bool) -> bool:
        from Infernux.lib import set_gui_semantic_capture_enabled

        set_gui_semantic_capture_enabled(bool(enabled))
        return bool(enabled)

    def request_semantic_snapshot(self) -> int:
        from Infernux.lib import request_gui_semantic_snapshot

        return int(request_gui_semantic_snapshot() or 0)

    def semantic_snapshot(self) -> dict[str, object]:
        from Infernux.lib import get_gui_semantic_snapshot

        return dict(get_gui_semantic_snapshot() or {})

    def request_capture(self, source: str, output_path: str, camera_component_id: int = 0) -> int:
        return int(self._native_engine().request_capture(str(source), str(output_path), int(camera_component_id)))

    def begin_renderer_performance_window(self, sample_count: int = 240) -> int:
        return int(self._native_engine().begin_renderer_performance_window(sample_count))

    def renderer_performance_window(self) -> dict[str, object]:
        native = self._native_engine()
        result = dict(native.get_renderer_performance_window())
        result["resources"] = {
            "resident_mesh_vertex_buffers": int(native.resident_mesh_vertex_buffer_count),
            "pending_mesh_uploads": int(native.pending_mesh_gpu_upload_count),
            "submitted_mesh_uploads": int(native.submitted_mesh_gpu_upload_count),
            "completed_mesh_uploads": int(native.completed_mesh_gpu_upload_count),
            "pending_texture_cpu_loads": int(native.pending_texture_cpu_load_count),
            "pending_texture_uploads": int(native.pending_texture_gpu_upload_count),
            "submitted_texture_uploads": int(native.submitted_texture_gpu_upload_count),
            "completed_texture_uploads": int(native.completed_texture_gpu_upload_count),
        }
        result["gpu_residency"] = dict(native.gpu_residency_snapshot)
        return result

    def gizmo_collection_observation(self) -> dict[str, object]:
        """Read the Editor's last completed Gizmo collection phases."""
        from Infernux.engine.bootstrap import EditorBootstrap

        bootstrap = EditorBootstrap.instance()
        engine = bootstrap.engine if bootstrap is not None else None
        if engine is None:
            raise OperationError(
                "editor.unavailable",
                "A running graphical Editor session is required.",
            )
        return dict(engine.get_gizmo_collection_observation())

    def capture_status(self, capture_id: int) -> dict[str, object]:
        return dict(self._native_engine().query_capture(int(capture_id)))

    def compute_statistics(self, *, reset: bool = False) -> dict[str, object]:
        """Snapshot the engine's existing compute counters, not a second profiler."""
        from dataclasses import asdict
        from Infernux import compute

        return asdict(compute.statistics(reset=reset))

    def set_compute_profiling(self, enabled: bool) -> dict[str, object]:
        from Infernux import compute

        compute.set_profiling_enabled(enabled)
        return {"enabled": enabled}

    def cancel_capture(self, capture_id: int) -> bool:
        return bool(self._native_engine().cancel_capture(int(capture_id)))

    def request_scene_pick(
        self, x: float, y: float, width: float, height: float
    ) -> int:
        return int(
            self._native_engine().request_scene_object_pick(x, y, width, height)
        )

    def scene_pick_status(self, request_id: int) -> dict[str, object]:
        return dict(self._native_engine().query_scene_object_pick(int(request_id)))

    def console_read(
        self, limit: int = 100, levels: Iterable[str] = ()
    ) -> dict[str, object]:
        allowed = {self._canonical_level(level) for level in levels}
        panel = self._native_console()
        reader = getattr(panel, "_get_visible_log_snapshot", None)
        if callable(reader):
            entries = [dict(item) for item in reader(max(1, int(limit)))]
            entries = [
                item
                for item in entries
                if not allowed
                or self._canonical_level(item.get("level", "")) in allowed
            ]
            status_bar = None
            try:
                message, level, info, warnings, errors, uid = panel._get_status_snapshot()
                status_bar = {
                    "surface": "status_bar",
                    "message": message,
                    "level": self._canonical_level(level),
                    "counts": {
                        "info": info,
                        "warnings": warnings,
                        "errors": errors,
                    },
                    "mirrors_console_uid": uid or None,
                }
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            return {
                "entries": entries,
                "source": "native_console",
                "surface": "console",
                "status_bar": status_bar,
            }
        entries = []
        for entry in DebugConsole.instance().get_entries()[-max(1, int(limit)) :]:
            value = {
                "time": entry.get_formatted_time(),
                "level": self._canonical_level(entry.log_type),
                "message": entry.message,
                "source_file": entry.source_file,
                "source_line": entry.source_line,
                "stack_trace": entry.stack_trace,
                "count": 1,
            }
            if not allowed or value["level"] in allowed:
                entries.append(value)
        return {
            "entries": entries,
            "source": "python_debug_fallback",
            "surface": "console",
            "status_bar": None,
        }

    def create_project_asset(
        self,
        kind: str,
        directory: str,
        name: str,
        extension: str,
        variant: str = "",
    ) -> str:
        return str(
            self.interaction_core().project_asset_interactions.create(
                kind, directory, name, extension, variant
            )
            or ""
        )

    def save_mesh_copy(self, asset_guid: str, destination: str) -> str:
        from Infernux.lib import AssetRegistry
        from Infernux.engine.interaction.action_journal import ActionOrigin

        mesh = AssetRegistry.instance().load_mesh_by_guid(asset_guid)
        if mesh is None:
            raise OperationError("asset.not_found", "The source mesh could not be loaded.")
        return self.interaction_core().project_assets.save_mesh_copy(
            mesh, destination, origin=ActionOrigin.AUTOMATION,
        )

    def extract_model_material(self, asset_guid: str, slot: int, destination: str) -> str:
        from Infernux.lib import AssetRegistry
        from Infernux.engine.interaction.action_journal import ActionOrigin

        mesh = AssetRegistry.instance().load_mesh_by_guid(asset_guid)
        if mesh is None:
            raise OperationError("asset.not_found", "The source model could not be loaded.")
        return self.interaction_core().project_assets.extract_model_material(
            mesh, slot, destination, origin=ActionOrigin.AUTOMATION,
        )

    def project_asset_text(self, path: str) -> str:
        return self.interaction_core().project_assets.read_text(path)

    def set_project_asset_text(self, path: str, content: str) -> str:
        from Infernux.engine.interaction.action_journal import ActionOrigin

        return self.interaction_core().project_assets.set_text(
            path,
            content,
            origin=ActionOrigin.AUTOMATION,
        )

    def material_document(self, path: str) -> tuple[Any, dict[str, object]]:
        from Infernux.core.material import Material

        material = Material.load(path)
        if material is None:
            raise OperationError("material.load_failed", f"Material could not be loaded: {path}")
        return material, dict(material.serialize_document())

    def publish_material_document(
        self,
        path: str,
        guid: str,
        document: dict[str, object],
        *,
        edit_key: str,
        description: str,
    ) -> None:
        from Infernux.engine.interaction import (
            DocumentKind,
            ensure_editable_resource_document,
        )

        material, _before = self.material_document(path)
        controller = ensure_editable_resource_document(
            category="material",
            document_kind=DocumentKind.MATERIAL,
            file_path=path,
            resource=material,
            guid=guid,
            title=material.name,
            view_id="automation",
        )
        changed = controller.apply_document(
            document,
            view_id="automation",
            edit_key=edit_key,
            description=description,
        )
        if not changed:
            raise OperationError(
                "material.edit_rejected", "Material edit was rejected or unchanged."
            )
        controller.flush_autosave(force=True)
        native = getattr(
            getattr(self.plugin_manager(), "engine", None),
            "get_native_engine",
            lambda: None,
        )()
        refresh = getattr(native, "refresh_material_pipeline", None)
        if callable(refresh):
            refresh(material.native)

    def data_asset_document(self, path: str) -> tuple[Any, dict[str, object]]:
        from Infernux.core.data_asset import DataAsset
        from Infernux.engine.interaction import DocumentKey, DocumentKind, DocumentRegistry

        resolved = resolved_path(path)
        guid = str(self.asset_database().get_guid_from_path(resolved) or "")
        document = (
            DocumentRegistry.instance().get_by_key(
                DocumentKey.asset(DocumentKind.DATA_ASSET, guid)
            )
            if guid
            else None
        )
        controller = getattr(document, "controller", None)
        live_asset = getattr(controller, "resource", None)
        if isinstance(live_asset, DataAsset):
            return live_asset, dict(live_asset.serialize_document())
        asset = DataAsset.load(path)
        return asset, dict(asset.serialize_document())

    def data_asset_schema(self, path: str) -> dict[str, object]:
        from Infernux.components.fields import get_field_schema, get_serialized_fields

        asset, _document = self.data_asset_document(path)
        asset_type = type(asset)
        return {
            "type_id": str(asset_type.__serialized_type_id__),
            "fields": [
                get_field_schema(asset_type, name).to_document()
                for name in get_serialized_fields(asset_type)
            ],
        }

    def publish_data_asset_document(
        self,
        path: str,
        guid: str,
        document: dict[str, object],
        *,
        edit_key: str,
        description: str,
    ) -> None:
        from Infernux.engine.interaction import (
            ActionOrigin,
            DocumentKind,
            ensure_editable_resource_document,
        )

        asset, _before = self.data_asset_document(path)
        controller = ensure_editable_resource_document(
            category="data_asset",
            document_kind=DocumentKind.DATA_ASSET,
            file_path=path,
            resource=asset,
            guid=guid,
            title=os.path.basename(path),
            view_id="automation",
        )
        changed = controller.apply_document(
            document,
            view_id="automation",
            edit_key=edit_key,
            description=description,
            origin=ActionOrigin.AUTOMATION,
        )
        if not changed:
            raise OperationError(
                "data_asset.edit_rejected",
                "DataAsset edit was rejected or unchanged.",
            )
        controller.flush_autosave(force=True)

    def particle_graph_document(self, path: str) -> tuple[Any, dict[str, object]]:
        from Infernux.particle.asset import ParticleGraphAsset

        graph = ParticleGraphAsset.load(path)
        return graph, dict(graph.to_dict())

    def particle_graph_from_document(self, document: dict[str, object]):
        from Infernux.particle.asset import ParticleGraphAsset

        return ParticleGraphAsset.from_dict(document)

    def publish_particle_graph(
        self, path: str, guid: str, before: Any, after: Any, description: str
    ) -> None:
        from Infernux.core.assets import AssetManager
        from Infernux.engine.undo import UndoCommand, UndoManager
        from Infernux.particle.artifact import ParticleArtifactRegistry

        if before.to_dict() == after.to_dict():
            raise OperationError(
                "particle.edit_rejected", "Particle Graph edit is unchanged."
            )

        class ParticleGraphCommand(UndoCommand):
            marks_dirty = False

            def __init__(command_self):
                super().__init__(description)

            @staticmethod
            def _publish(graph):
                ParticleArtifactRegistry.save_graph_asset(graph, path, guid=guid)
                result = AssetManager.reimport_asset(path)
                if not result:
                    result = AssetManager.import_asset(path)
                if not result:
                    raise RuntimeError(
                        str(getattr(result, "error", "Particle Graph import failed"))
                    )

            def _apply(command_self, graph, rollback):
                try:
                    command_self._publish(graph)
                except Exception:
                    try:
                        command_self._publish(rollback)
                    except Exception:
                        pass
                    raise

            def execute(command_self):
                command_self._apply(after, before)

            def undo(command_self):
                command_self._apply(before, after)

        manager = UndoManager.instance()
        if manager is None or not manager.enabled or manager.is_executing:
            raise OperationError(
                "particle.edit_rejected",
                "Editor history cannot accept Particle Graph edits.",
            )
        if not manager.execute(ParticleGraphCommand()):
            raise OperationError(
                "particle.edit_rejected",
                "Particle Graph edit failed validation or publication.",
            )

    def hierarchy_create_kinds(self) -> list[dict[str, object]]:
        from Infernux.engine.hierarchy_creation_service import HierarchyCreationService

        return list(HierarchyCreationService.instance().list_create_kinds())

    def create_scene_object(self, kind: str, parent_id: int, name: str):
        from Infernux.engine.hierarchy_creation_service import HierarchyCreationService

        service = HierarchyCreationService.instance()
        if not service.can_create(kind, parent_id=int(parent_id)):
            raise OperationError(
                "scene.create_rejected", f"Cannot create object kind {kind!r}."
            )
        return service.create(
            kind,
            parent_id=int(parent_id),
            name=str(name or "") or None,
            select=False,
            selection_owner_id="automation",
            selection_reason="host_create_game_object",
        )

    def instantiate_scene_model(self, asset_guid: str, parent_id: int, name: str):
        """Instantiate a model through the Project/Hierarchy mutation service."""
        value = self.interaction_core().scene_objects.create_model_object(
            str(asset_guid),
            parent_id=int(parent_id),
            is_guid=True,
            name=str(name or ""),
            select=False,
            selection_owner_id="automation",
            selection_reason="host_instantiate_model",
        )
        if value is None:
            raise OperationError(
                "scene.create_rejected", "The model asset could not be instantiated."
            )
        return value

    def scene_object(self, object_id: int):
        """Resolve one live scene object without exposing SceneManager lookup rules."""
        normalized = int(object_id)
        value = self.active_scene().find_by_id(normalized)
        if value is None:
            raise OperationError(
                "scene.object_not_found", f"GameObject was not found: {normalized}"
            )
        return value

    def scene_components(self, object_id: int) -> list[Any]:
        """Return each logical component once, preferring Python API wrappers."""
        from Infernux.components.builtin_component import BuiltinComponent

        owner = self.scene_object(object_id)
        result: list[Any] = []
        seen: set[int] = set()
        for value in list(owner.get_py_components()) + list(owner.get_components()):
            component_id = int(getattr(value, "component_id", 0) or 0)
            if component_id > 0 and component_id in seen:
                continue
            if not bool(getattr(value, "_is_builtin_component_wrapper", False)):
                wrapper_type = BuiltinComponent._builtin_registry.get(
                    str(getattr(value, "type_name", "") or "")
                )
                if wrapper_type is not None:
                    value = wrapper_type._get_or_create_wrapper(value, owner)
            if component_id > 0:
                seen.add(component_id)
            result.append(value)
        return result

    def scene_component(self, object_id: int, component_id: int):
        target_id = int(component_id)
        for value in self.scene_components(object_id):
            if int(getattr(value, "component_id", 0) or 0) == target_id:
                return value
        raise OperationError(
            "scene.component_not_found",
            f"Component {target_id} was not found on GameObject {int(object_id)}.",
        )

    def scene_component_schema(
        self, object_id: int, component_id: int
    ) -> dict[str, object]:
        """Describe writable component fields using authoritative serializer metadata."""
        from Infernux.components.builtin_component import BuiltinComponent
        from Infernux.components.fields import get_serialized_fields

        value = self.scene_component(object_id, component_id)
        fields: list[dict[str, object]] = []
        type_name = str(getattr(value, "type_name", "") or type(value).__name__)
        if type_name == "Transform" or isinstance(value, BuiltinComponent):
            from Infernux.field_schema import get_native_field_schemas

            for schema in get_native_field_schemas(f"native:infernux.{type_name}"):
                attributes = schema.to_document()["attributes"]
                entry: dict[str, object] = {
                    "name": str(schema.attributes["serialized_name"]),
                    "type": schema.value_type.rsplit(".", 1)[-1].lower(),
                    "readonly": bool(schema.read_only),
                    "hidden": bool(schema.attributes["hidden"]),
                }
                if "range" in attributes:
                    entry["range"] = [float(item) for item in attributes["range"]]
                if attributes.get("asset_type"):
                    entry["asset_type"] = attributes["asset_type"]
                    entry["nullable"] = bool(attributes["nullable"])
                enum = attributes.get("enum")
                if enum is not None:
                    entry["enum"] = [
                        {"name": str(member["name"]), "value": int(member["value"])}
                        for member in enum["members"]
                    ]
                fields.append(entry)
        else:
            for name, metadata in get_serialized_fields(type(value)).items():
                enum_type = getattr(metadata, "enum_type", None)
                if isinstance(enum_type, str):
                    try:
                        import Infernux.lib as native

                        enum_type = getattr(native, enum_type, None)
                    except (ImportError, AttributeError):
                        enum_type = None
                members = getattr(enum_type, "__members__", {}) or {}
                field_type = getattr(getattr(metadata, "field_type", None), "name", "unknown")
                entry = {
                    "name": str(name),
                    "type": str(field_type).lower(),
                    "readonly": bool(getattr(metadata, "readonly", False)),
                    "hidden": bool(getattr(metadata, "hidden", False)),
                }
                value_range = getattr(metadata, "range", None)
                if value_range is not None:
                    entry["range"] = [float(item) for item in value_range]
                if members:
                    entry["enum"] = [
                        {
                            "name": str(member_name),
                            "value": int(getattr(member, "value", member)),
                        }
                        for member_name, member in members.items()
                    ]
                fields.append(entry)
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "component_type": type_name,
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": fields,
        }

    def delete_scene_objects(self, object_ids: Iterable[int]) -> list[int]:
        values = [int(value) for value in object_ids]
        if not self.interaction_core().scene_objects.delete_ids(values):
            raise OperationError("scene.edit_rejected", "No GameObjects were deleted.")
        return values

    def set_scene_object_property(
        self, object_id: int, property_name: str, value: object
    ) -> None:
        if not self.interaction_core().scene_objects.set_object_property(
            int(object_id), str(property_name), value
        ):
            raise OperationError(
                "scene.edit_rejected", "GameObject property edit was rejected or unchanged."
            )

    def set_scene_transforms(self, object_id: int, transform: dict[str, object]) -> None:
        if not self.interaction_core().scene_objects.set_transforms(
            [int(object_id)], [dict(transform)]
        ):
            raise OperationError(
                "scene.edit_rejected", "Transform edit was rejected or unchanged."
            )

    def set_scene_parent(self, object_ids: Iterable[int], parent_id: int = 0) -> None:
        values = [int(value) for value in object_ids]
        target = int(parent_id)
        mode = "parent" if target else "root"
        if not self.interaction_core().scene_objects.move_hierarchy(values, mode, target):
            raise OperationError(
                "scene.edit_rejected", "Hierarchy edit was rejected or unchanged."
            )

    def add_scene_component(
        self, object_id: int, component_type: str, *, script_guid: str = ""
    ):
        return self.interaction_core().components.add(
            self.scene_object(object_id),
            str(component_type),
            script_guid=str(script_guid or ""),
        )

    def remove_scene_component(self, object_id: int, component_id: int) -> None:
        owner = self.scene_object(object_id)
        value = self.scene_component(object_id, component_id)
        if not self.interaction_core().components.remove(owner, value):
            raise OperationError(
                "scene.edit_rejected", "Component removal was rejected or unchanged."
            )

    def set_scene_component_field(
        self, object_id: int, component_id: int, field: str, value: object
    ):
        from Infernux.components.builtin_component import BuiltinComponent, CppProperty

        target = self.scene_component(object_id, component_id)
        service = self.interaction_core().components
        if isinstance(target, BuiltinComponent):
            # scene_component_schema and serialize_document expose the native
            # document vocabulary (including integer enums), not wrapper names.
            # Preserve live setter semantics (e.g. mute must not reload tracks).
            from Infernux.field_schema import get_native_field_schemas

            schema = next((item for item in get_native_field_schemas(f"native:infernux.{target.type_name}")
                           if item.attributes["serialized_name"] == str(field)), None)
            if schema is None or schema.read_only:
                raise OperationError("scene.edit_rejected", f"Native field {field!r} is not declared writable.")
            document = target.serialize_document()
            document[str(field)] = value
            attribute = str(schema.attributes["field_id"])
            prop = getattr(type(target), attribute, None)
            if isinstance(prop, CppProperty) and schema.attributes.get("setter_owns_document_shape", False):
                # This field's public value is not its native storage shape
                # (for example a RenderTexture reference versus a GUID string).
                # The shared property transaction owns normalization/setter use.
                changed = service.set_field(target, attribute, value)
            elif isinstance(prop, CppProperty):
                from Infernux.components.fields import FieldType
                from Infernux.components.value_codec import VALUE_CODECS

                if prop.schema is None:
                    # Catalog-backed properties already preflight in their
                    # shared transaction. Older value adapters still need it here.
                    target._require_cpp_component().validate_document(document)
                if schema.value_type == "FieldType.ENUM":
                    if type(value) is not int:
                        raise TypeError(f"{schema.property_path} requires an integer enum value")
                    native_value = prop.metadata.enum_type(value)
                else:
                    native_value = VALUE_CODECS.decode(value, FieldType[schema.value_type.removeprefix("FieldType.")],
                                                       schema.property_path)
                public_value = prop.get_converter(native_value) if prop.get_converter is not None else native_value
                changed = service.set_field(target, attribute, public_value)
            else:
                changed = service.restore_document(target, document,
                                                   description=f"Set {target.type_name}.{field}", edit_key=str(field))
        else:
            changed = service.set_field(target, str(field), value)
        if not changed:
            raise OperationError(
                "scene.edit_rejected", "Component field edit was rejected or unchanged."
            )
        return target

    def assign_scene_mesh(self, object_id: int, component_id: int, asset_guid: str, *, node_path=None):
        from Infernux.engine.interaction.action_journal import ActionOrigin

        target = self.scene_component(object_id, component_id)
        self.interaction_core().components.assign_mesh_asset(
            target, asset_guid, node_path=node_path, origin=ActionOrigin.AUTOMATION,
        )
        return target

    def open_scene(self, path: str) -> bool:
        from Infernux.engine.scene_manager import SceneFileManager

        manager = SceneFileManager.instance()
        return bool(manager is not None and manager.open_scene(path))

    def loaded_scenes(self) -> dict[str, object]:
        from Infernux.lib import SceneManager

        manager = SceneManager.instance()
        active = manager.get_active_scene()
        scenes = []
        for index in range(int(manager.scene_count)):
            scene = manager.get_scene_at(index)
            if scene is None:
                continue
            scenes.append({
                "index": index,
                "world_id": int(scene.world_id),
                "name": str(scene.name),
                "active": scene is active,
                "root_count": len(scene.get_root_objects()),
            })
        return {
            "active_world_id": int(active.world_id) if active is not None else 0,
            "scenes": scenes,
        }

    def activate_loaded_scene(self, world_id: int) -> dict[str, object]:
        from Infernux.engine.scene_manager import SceneFileManager

        identifier = int(world_id)
        manager = SceneFileManager.instance()
        if identifier <= 0 or manager is None or not manager.activate_loaded_scene(identifier):
            raise OperationError(
                "scene.activate_rejected",
                "The requested World is not a loaded authoring Scene.",
            )
        return self.loaded_scenes()

    def load_additive_scene(self, path: str) -> bool:
        from Infernux.engine.play_mode import PlayModeManager, PlayModeState
        from Infernux.engine.scene_manager import SceneFileManager

        play_mode = PlayModeManager.instance()
        if play_mode is not None and play_mode.state is not PlayModeState.EDIT:
            raise OperationError(
                "scene.additive.edit_mode_required",
                "Additive scene authoring is only available in Edit Mode.",
            )
        manager = SceneFileManager.instance()
        return bool(
            manager is not None
            and manager.load_scene_additive_immediate(path)
        )

    def save_scene(self) -> str:
        from Infernux.engine.scene_manager import SceneFileManager
        from Infernux.engine.interaction import DocumentRegistry, DocumentState

        manager = SceneFileManager.instance()
        if manager is not None:
            document = DocumentRegistry.instance().get(manager.document_id)
            if document is not None and document.state is DocumentState.CONFLICT:
                raise OperationError(
                    "scene.save_rejected",
                    "The scene changed outside the Editor. Resolve the external "
                    "conflict by reloading, keeping the local draft, or saving a copy.",
                    details={"document_id": document.document_id,
                             "document_state": document.state.value},
                )
        if manager is None or not manager.save_current_scene():
            raise OperationError(
                "scene.save_rejected", "The active scene could not be saved synchronously."
            )
        return str(manager.current_scene_path or "")

    def reload_scene(self, *, discard_changes: bool = False) -> dict[str, object]:
        """Reload the active scene through the normal deferred scene swap."""
        from Infernux.engine.scene_manager import SceneFileManager

        manager = SceneFileManager.instance()
        path = str(manager.current_scene_path or "") if manager is not None else ""
        if manager is None or not manager.reload_current_scene(
            discard_changes=bool(discard_changes)
        ):
            raise OperationError(
                "scene.reload_rejected",
                "The active scene could not be reloaded. Stop Play Mode and pass "
                "discard_changes=true when the document is dirty or conflicted.",
            )
        return {
            "scheduled": True,
            "path": path,
            "discarded_changes": bool(discard_changes),
        }

    def player_build_targets(self) -> dict[str, object]:
        """List targets currently owned by enabled platform plugins."""
        from dataclasses import asdict

        from Infernux.engine.build import (
            current_host_player_target,
            exporter_registry,
            platform_support_catalog,
        )

        targets = exporter_registry.targets()
        desktop = current_host_player_target(targets)
        support = {item.target_id: item for item in platform_support_catalog()}
        def _capabilities(item) -> dict[str, object]:
            result = asdict(item.capabilities)
            result["features"] = sorted(result["features"])
            return result

        return {
            "current_host_target": str(desktop.id) if desktop is not None else "",
            "targets": [
                {
                    "id": str(item.id),
                    "display_name": item.display_name,
                    "platform": item.platform,
                    "architecture": item.architecture,
                    "capabilities": _capabilities(item),
                    "available": True,
                    "plugin_reference": (
                        support[item.id].package_reference if item.id in support else ""
                    ),
                }
                for item in targets
            ]
            + [
                {
                    "id": item.target_id,
                    "display_name": item.target_id.replace("-", " ").title(),
                    "platform": item.target_id.split("-", 1)[0],
                    "architecture": "",
                    "capabilities": None,
                    "available": False,
                    "plugin_reference": item.package_reference,
                    "installed": item.installed,
                    "enabled": item.enabled,
                    "cached": item.cached,
                }
                for item in support.values()
                if not item.registered
            ],
        }

    def build_player(
        self,
        project_root: str,
        *,
        target: str = "",
        output_dir: str = "",
        game_name: str = "",
        debug_mode: bool | None = None,
        lto: bool | None = None,
        android_artifact: str = "",
        compress_resources: bool | None = None,
        persist_settings: bool = True,
    ) -> dict[str, object]:
        """Build one registered Player target through the shared build service."""
        import json
        import os

        from dataclasses import replace
        from Infernux.engine.build import (
            BuildConfiguration,
            BuildProfile,
            BuildRequest,
            BuildService,
            BuildUnavailableError,
            build_progress_fraction,
            current_host_player_target,
            exporter_registry,
            required_platform_plugin,
        )
        from Infernux.engine.interaction import normalize_build_settings
        from Infernux.engine.path_utils import resolved_path
        from Infernux.engine.player_build_preflight import (
            publish_player_asset_catalog_for_host,
        )

        root = resolved_path(project_root)
        settings_path = os.path.join(root, "ProjectSettings", "BuildSettings.json")
        try:
            with open(settings_path, "r", encoding="utf-8") as stream:
                settings = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise OperationError(
                "player.build_settings",
                f"Build Settings could not be read: {settings_path}",
            ) from exc
        if not isinstance(settings, dict):
            raise OperationError("player.build_settings", "Build Settings must be an object.")
        try:
            settings = normalize_build_settings(settings)
        except (TypeError, ValueError) as exc:
            raise OperationError("player.build_settings", str(exc)) from exc

        raw_output = str(output_dir or settings.get("output_dir", "") or "").strip()
        final_output = resolved_path(raw_output) if raw_output else ""
        final_name = str(game_name or settings.get("game_name", "") or "").strip() or os.path.basename(root)
        if not final_output:
            raise OperationError(
                "player.build_settings",
                "Player build requires output_dir.",
            )
        final_debug = bool(settings.get("debug_mode", False)) if debug_mode is None else bool(debug_mode)
        final_lto = bool(settings.get("lto", True)) if lto is None else bool(lto)
        final_artifact = str(
            android_artifact or settings.get("android_artifact", "apk") or "apk"
        ).strip().casefold()
        if final_artifact not in {"apk", "aab"}:
            raise OperationError(
                "player.build_settings",
                "android_artifact must be apk or aab.",
            )
        available_targets = exporter_registry.targets()
        desktop = current_host_player_target(available_targets)
        final_target = str(
            target
            or settings.get("build_target", "")
            or (desktop.id if desktop is not None else "")
        ).strip()
        if not final_target or all(
            item.id != final_target for item in available_targets
        ):
            required = required_platform_plugin(final_target, root)
            if required is not None:
                raise OperationError(
                    "platform_plugin_required",
                    "Player build target requires an installed and enabled "
                    f"platform plugin: {required.package_reference}",
                    details={
                        "requested_target": final_target,
                        "plugin_reference": required.package_reference,
                        "plugin_name": required.package_name,
                        "installed": required.installed,
                        "enabled": required.enabled,
                        "cached": required.cached,
                        "source": dict(required.source),
                    },
                )
            raise OperationError(
                "player.target_unavailable",
                f"Player build target is not installed or enabled: {final_target or '<none>'}",
                details={
                    "requested_target": final_target,
                    "available_targets": [
                        {
                            "id": str(item.id),
                            "display_name": item.display_name,
                            "platform": item.platform,
                            "architecture": item.architecture,
                        }
                        for item in available_targets
                    ],
                },
            )
        selected_target = next(
            item for item in available_targets if item.id == final_target
        )
        final_jit = bool(
            selected_target.capabilities.cpu_jit
            or "gpu-jit" in selected_target.capabilities.features
        )
        final_compress = (
            not final_debug
            if compress_resources is None
            else bool(compress_resources)
        )
        settings.update(
            {
                "build_target": final_target,
                "android_artifact": final_artifact,
                "output_dir": final_output,
                "game_name": final_name,
                "debug_mode": final_debug,
                "lto": final_lto,
            }
        )
        progress: list[dict[str, object]] = []
        phase_counts: dict[str, int] = {}
        omitted_verbose = 0
        last_fraction = 0.0

        def record(item) -> None:
            nonlocal last_fraction, omitted_verbose
            phase = str(item.phase)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            source = str(item.detail.get("source", "")).casefold()
            if source in {"cmake", "gradle", "web-toolchain"}:
                omitted_verbose += 1
                return
            last_fraction = max(last_fraction, build_progress_fraction(item))
            progress.append(
                {
                    "phase": phase,
                    "completed": item.completed,
                    "total": item.total,
                    "message": item.message,
                    "progress": last_fraction,
                    "detail": dict(item.detail),
                }
            )
            if len(progress) > 128:
                del progress[:-128]

        request = BuildRequest(
            root,
            final_target,
            final_output,
            BuildProfile(
                configuration=(
                    BuildConfiguration.DEVELOPMENT
                    if final_debug
                    else BuildConfiguration.RELEASE
                ),
                debug_symbols=final_debug,
                compress_resources=final_compress,
                options={
                    "android_artifact": final_artifact,
                    "build_settings": settings,
                },
            ),
            progress=record,
        )
        service = BuildService(exporter_registry)
        try:
            plan = service.create_plan(request)
            catalog = publish_player_asset_catalog_for_host(root)
            request = replace(
                request,
                asset_catalog_entries=tuple(catalog["entries"]),
            )
            result = service.execute(request, plan)
        except BuildUnavailableError as exc:
            raise OperationError(
                "player.target_unavailable",
                "; ".join(item.message for item in exc.diagnostics),
                details={
                    "target": final_target,
                    "diagnostics": [
                        {
                            "severity": item.severity.value,
                            "code": item.code,
                            "message": item.message,
                            "source": item.source,
                            "detail": dict(item.detail),
                        }
                        for item in exc.diagnostics
                    ],
                },
            ) from exc
        diagnostics = [
            {
                "severity": item.severity.value,
                "code": item.code,
                "message": item.message,
                "source": item.source,
                "detail": dict(item.detail),
            }
            for item in result.diagnostics
        ]
        artifacts = [
            {
                "path": item.path,
                "kind": item.kind,
                "size": item.size,
            }
            for item in result.artifacts
        ]
        if not result.success:
            raise OperationError(
                "player.build_failed",
                "; ".join(item["message"] for item in diagnostics)
                or "The platform exporter did not produce a Player artifact.",
                details={
                    "target": final_target,
                    "diagnostics": diagnostics,
                    "manifest": dict(result.manifest),
                    "logs": list(result.logs[-100:]),
                },
            )

        if persist_settings:
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".build-settings-",
                suffix=".tmp",
                dir=os.path.dirname(settings_path),
                text=True,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(settings, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, settings_path)
            except Exception:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
                raise

        is_desktop = desktop is not None and final_target == desktop.id
        desktop_output = str(result.manifest.get("output_dir", final_output))
        executable = (
            os.path.join(
                desktop_output,
                final_name + (".exe" if os.name == "nt" else ""),
            )
            if is_desktop
            else ""
        )
        return {
            "target": final_target,
            "output_dir": final_output,
            "game_name": final_name,
            "executable_path": executable,
            "executable_exists": bool(executable and os.path.isfile(executable)),
            "debug_mode": final_debug,
            "lto": final_lto,
            "jit_enabled": final_jit,
            "compress_resources": final_compress,
            "android_artifact": final_artifact,
            "elapsed_seconds": result.elapsed_seconds,
            "artifacts": artifacts,
            "diagnostics": diagnostics,
            "manifest": dict(result.manifest),
            "progress": progress,
            "progress_summary": {
                "event_count": sum(phase_counts.values()),
                "retained_count": len(progress),
                "omitted_verbose_count": omitted_verbose,
                "phase_counts": dict(sorted(phase_counts.items())),
            },
        }

    @staticmethod
    def _canonical_level(value: object) -> str:
        raw = getattr(value, "name", value)
        return _LEVEL_ALIASES.get(str(raw).upper(), str(raw).upper())

    @staticmethod
    def _runtime_status_value(manager: Any) -> dict[str, object]:
        from Infernux.engine.deferred_task import DeferredTaskRunner

        runner = DeferredTaskRunner.instance()
        transition_task = runner.active_task_name
        return {
            "state": str(manager.state.name).lower(),
            "playing": bool(manager.is_playing),
            "paused": bool(manager.is_paused),
            "time_scale": float(manager.time_scale),
            "delta_time": float(manager.delta_time),
            "total_play_time": float(manager.total_play_time),
            "step_sequence": int(manager.step_sequence),
            "transition_timings_ms": dict(manager.last_transition_timings_ms),
            "transition_pending": transition_task in {
                "Enter Play Mode",
                "Exit Play Mode",
            },
        }

    @staticmethod
    def _camera_state(camera: Any) -> dict[str, object]:
        return {
            "position": list(camera.position),
            "rotation": list(camera.rotation),
            "focus": list(camera.focus_point),
            "distance": float(camera.focus_distance),
            "fov": float(camera.fov),
            "near_clip": float(camera.near_clip),
            "far_clip": float(camera.far_clip),
            "orthographic": bool(camera.orthographic),
            "orthographic_size": float(camera.orthographic_size),
        }

    @staticmethod
    def _play_mode_manager():
        from Infernux.engine.play_mode import PlayModeManager

        manager = PlayModeManager.instance()
        if manager is None:
            raise OperationError("editor.unavailable", "PlayModeManager is unavailable.")
        return manager

    def _editor_camera(self):
        camera = getattr(getattr(self.plugin_manager(), "engine", None), "editor_camera", None)
        if camera is None:
            raise OperationError("editor.unavailable", "Editor camera is unavailable.")
        return camera

    @staticmethod
    def _native_console():
        try:
            from Infernux.engine.bootstrap import EditorBootstrap

            bootstrap = EditorBootstrap.instance()
            return getattr(bootstrap, "console", None) if bootstrap else None
        except (AttributeError, ImportError, RuntimeError):
            return None

    @staticmethod
    def _native_engine():
        from Infernux.engine.bootstrap import EditorBootstrap

        bootstrap = EditorBootstrap.instance()
        engine = bootstrap.engine if bootstrap is not None else None
        native = engine.get_native_engine() if engine is not None else None
        if native is None:
            raise OperationError(
                "editor.unavailable",
                "A running graphical Editor session is required.",
            )
        return native


__all__ = ["EditorAutomationHost"]
