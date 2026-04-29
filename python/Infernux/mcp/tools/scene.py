"""Scene inspection and mutation MCP tools."""

from __future__ import annotations

from typing import Any

from Infernux.mcp.tools.common import (
    coerce_vector3,
    find_game_object,
    main_thread,
    resolve_asset_path,
    serialize_component,
    serialize_value,
    serialize_vector,
)


def register_scene_tools(mcp) -> None:
    @mcp.tool(name="scene.get_hierarchy")
    def scene_get_hierarchy(
        depth: int = 6,
        include_components: bool = True,
        include_inactive: bool = True,
    ) -> dict:
        """Return the active scene hierarchy."""

        def _read():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            roots = list(scene.get_root_objects() or [])
            return {
                "scene": getattr(scene, "name", ""),
                "roots": [
                    _serialize_object(
                        root,
                        depth=max(int(depth), 0),
                        include_components=bool(include_components),
                        include_inactive=bool(include_inactive),
                    )
                    for root in roots
                    if include_inactive or bool(getattr(root, "active", True))
                ],
            }

        return main_thread("scene.get_hierarchy", _read)

    @mcp.tool(name="scene.save")
    def scene_save(path: str = "") -> dict:
        """Save the active scene. If path is provided, it must be under Assets/."""

        def _save():
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm is None:
                raise RuntimeError("SceneFileManager is not available.")
            if path:
                from Infernux.engine.project_context import get_project_root
                save_path = resolve_asset_path(get_project_root(), path)
                ok = bool(sfm._do_save(save_path))
            else:
                ok = bool(sfm.save_current_scene())
                save_path = getattr(sfm, "current_scene_path", "") or ""
            return {"saved": ok, "path": save_path, "dirty": bool(getattr(sfm, "is_dirty", False))}

        return main_thread("scene.save", _save)

    @mcp.tool(name="scene.open")
    def scene_open(path: str) -> dict:
        """Open a .scene file under Assets/."""

        def _open():
            from Infernux.engine.project_context import get_project_root
            from Infernux.engine.scene_manager import SceneFileManager
            scene_path = resolve_asset_path(get_project_root(), path)
            sfm = SceneFileManager.instance()
            if sfm is None:
                raise RuntimeError("SceneFileManager is not available.")
            accepted = bool(sfm.open_scene(scene_path))
            return {"accepted": accepted, "path": scene_path, "loading": bool(getattr(sfm, "is_loading", False))}

        return main_thread("scene.open", _open)

    @mcp.tool(name="scene.new")
    def scene_new() -> dict:
        """Create a new empty scene through SceneFileManager."""

        def _new():
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm is None:
                raise RuntimeError("SceneFileManager is not available.")
            sfm.new_scene()
            return {"accepted": True, "loading": bool(getattr(sfm, "is_loading", False))}

        return main_thread("scene.new", _new)

    @mcp.tool(name="scene.serialize")
    def scene_serialize() -> dict:
        """Return the active scene JSON."""

        def _serialize():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            return {"scene": getattr(scene, "name", ""), "json": scene.serialize()}

        return main_thread("scene.serialize", _serialize)

    @mcp.tool(name="scene.inspect")
    def scene_inspect(depth: int = 2, include_components: bool = True) -> dict:
        """Return a compact scene summary for agents."""

        def _inspect():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            objects = list(scene.get_all_objects() or [])
            roots = list(scene.get_root_objects() or [])
            component_counts: dict[str, int] = {}
            cameras = []
            lights = []
            scripts = []
            for obj in objects:
                comps = _all_components(obj)
                for comp in comps:
                    type_name = comp.get("type", "")
                    component_counts[type_name] = component_counts.get(type_name, 0) + 1
                    if type_name == "Camera":
                        cameras.append({"id": int(obj.id), "name": str(obj.name)})
                    elif type_name == "Light":
                        lights.append({"id": int(obj.id), "name": str(obj.name)})
                    elif comp.get("python"):
                        scripts.append({
                            "object_id": int(obj.id),
                            "object_name": str(obj.name),
                            "type": type_name,
                            "script_guid": comp.get("script_guid", ""),
                        })
            return {
                "scene": getattr(scene, "name", ""),
                "object_count": len(objects),
                "root_count": len(roots),
                "component_counts": component_counts,
                "cameras": cameras,
                "lights": lights,
                "scripts": scripts,
                "roots": [
                    _serialize_object(
                        root,
                        depth=max(int(depth), 0),
                        include_components=bool(include_components),
                        include_inactive=True,
                    )
                    for root in roots
                ],
            }

        return main_thread("scene.inspect", _inspect)

    @mcp.tool(name="gameobject.add_component")
    def gameobject_add_component(
        object_id: int,
        component_type: str,
        script_path: str = "",
        fields: dict[str, Any] | None = None,
    ) -> dict:
        """Add a native, built-in wrapper, registered Python, or script component."""

        def _add():
            obj = find_game_object(object_id)
            before_ids = _component_ids(obj)
            is_py = False
            comp = None

            if script_path:
                from Infernux.components import load_and_create_component
                from Infernux.mcp.tools.common import get_asset_database
                comp = load_and_create_component(script_path, asset_database=get_asset_database(), type_name=component_type)
                if comp is None:
                    raise RuntimeError(f"Script did not create component '{component_type}'.")
                comp = obj.add_py_component(comp)
                is_py = True
            else:
                comp = obj.add_component(component_type)
                is_py = _is_python_script_component(comp)

            if comp is None:
                raise RuntimeError(f"Failed to add component '{component_type}'.")

            for key, value in (fields or {}).items():
                setattr(comp, key, _coerce_property_value(key, value))

            from Infernux.engine.ui._inspector_undo import _record_add_component_compound
            _record_add_component_compound(obj, component_type, comp, before_ids, is_py=is_py)
            return {
                "object_id": int(obj.id),
                "component": serialize_component(comp),
                "components": _all_components(obj),
            }

        return main_thread("gameobject.add_component", _add)

    @mcp.tool(name="gameobject.get")
    def gameobject_get(object_id: int, depth: int = 1, include_components: bool = True) -> dict:
        """Return a GameObject snapshot by id."""

        def _get():
            obj = find_game_object(object_id)
            return _serialize_object(
                obj,
                depth=max(int(depth), 0),
                include_components=bool(include_components),
                include_inactive=True,
            )

        return main_thread("gameobject.get", _get)

    @mcp.tool(name="gameobject.get_children")
    def gameobject_get_children(object_id: int = 0, include_components: bool = False) -> dict:
        """Return root objects (object_id=0) or direct children of a GameObject."""

        def _children():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            if int(object_id) == 0:
                children = list(scene.get_root_objects() or [])
                parent_id = 0
            else:
                obj = find_game_object(object_id)
                children = list(obj.get_children() or [])
                parent_id = int(obj.id)
            return {
                "parent_id": parent_id,
                "children": [
                    _serialize_object(
                        child,
                        depth=0,
                        include_components=bool(include_components),
                        include_inactive=True,
                    )
                    for child in children
                ],
            }

        return main_thread("gameobject.get_children", _children)

    @mcp.tool(name="gameobject.find")
    def gameobject_find(
        name: str = "",
        path: str = "",
        tag: str = "",
        component_type: str = "",
        include_inactive: bool = True,
        limit: int = 50,
    ) -> dict:
        """Find GameObjects by name, tag, and/or component type."""

        def _find():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            matches = []
            for obj in list(scene.get_all_objects() or []):
                if not include_inactive and not bool(getattr(obj, "active", True)):
                    continue
                if name and name.lower() not in str(obj.name).lower():
                    continue
                if path and path.lower() not in _object_path(obj).lower():
                    continue
                if tag and str(getattr(obj, "tag", "")) != tag:
                    continue
                if component_type and _find_component(obj, component_type, 0) is None:
                    continue
                data = _serialize_object(obj, depth=0, include_components=True, include_inactive=True)
                data["path"] = _object_path(obj)
                matches.append(data)
                if len(matches) >= max(int(limit), 1):
                    break
            return {"matches": matches}

        return main_thread("gameobject.find", _find)

    @mcp.tool(name="scene.find")
    def scene_find(query: dict[str, Any], limit: int = 50) -> dict:
        """Search scene objects by name/path/tag/layer/component."""

        def _scene_find():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            criteria = query or {}
            q_name = str(criteria.get("name", "")).lower()
            q_path = str(criteria.get("path", "")).lower()
            q_tag = str(criteria.get("tag", ""))
            q_component = str(criteria.get("component_type", ""))
            q_layer = criteria.get("layer", None)
            matches = []
            for obj in list(scene.get_all_objects() or []):
                if q_name and q_name not in str(obj.name).lower():
                    continue
                obj_path = _object_path(obj)
                if q_path and q_path not in obj_path.lower():
                    continue
                if q_tag and str(getattr(obj, "tag", "")) != q_tag:
                    continue
                if q_layer is not None and int(getattr(obj, "layer", -1)) != int(q_layer):
                    continue
                if q_component and _find_component(obj, q_component, 0) is None:
                    continue
                data = _serialize_object(obj, depth=0, include_components=True, include_inactive=True)
                data["path"] = obj_path
                matches.append(data)
                if len(matches) >= max(int(limit), 1):
                    break
            return {"matches": matches}

        return main_thread("scene.find", _scene_find)

    @mcp.tool(name="gameobject.path")
    def gameobject_path(object_id: int) -> dict:
        """Return the hierarchy path for a GameObject."""

        def _path():
            obj = find_game_object(object_id)
            return {"object_id": int(obj.id), "path": _object_path(obj)}

        return main_thread("gameobject.path", _path)

    @mcp.tool(name="gameobject.find_by_path")
    def gameobject_find_by_path(path: str) -> dict:
        """Find a GameObject by exact hierarchy path."""

        def _find_by_path():
            obj = _find_by_path_exact(path)
            if obj is None:
                raise FileNotFoundError(f"GameObject path not found: {path}")
            data = _serialize_object(obj, depth=1, include_components=True, include_inactive=True)
            data["path"] = _object_path(obj)
            return data

        return main_thread("gameobject.find_by_path", _find_by_path)

    @mcp.tool(name="gameobject.ensure_path")
    def gameobject_ensure_path(path: str, kind: str = "empty", select: bool = False) -> dict:
        """Ensure a slash-separated hierarchy path exists."""

        def _ensure_path():
            from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
            if not path or not str(path).strip("/"):
                raise ValueError("path must not be empty.")
            created = []
            parent_id = 0
            current_path = ""
            target_path = str(path).replace("\\", "/").strip("/")
            for part in [p for p in target_path.split("/") if p]:
                current_path = f"{current_path}/{part}" if current_path else part
                existing = _find_by_path_exact(current_path)
                if existing is not None:
                    parent_id = int(existing.id)
                    continue
                entry = HierarchyCreationService.instance().create(
                    kind if current_path == target_path else "empty",
                    parent_id=parent_id,
                    name=part,
                    select=False,
                )
                created.append(entry)
                parent_id = int(entry["id"])
            if select and parent_id:
                from Infernux.engine.ui.selection_manager import SelectionManager
                SelectionManager.instance().select(parent_id)
            final = find_game_object(parent_id)
            return {"object_id": parent_id, "path": _object_path(final), "created": created}

        return main_thread("gameobject.ensure_path", _ensure_path)

    @mcp.tool(name="gameobject.set")
    def gameobject_set(object_id: int, values: dict[str, Any]) -> dict:
        """Set GameObject fields: name, active, tag, layer, is_static."""

        def _set():
            obj = find_game_object(object_id)
            allowed = {"name", "active", "tag", "layer", "is_static"}
            changed = {}
            from Infernux.engine.ui._inspector_undo import _record_property
            for key, value in (values or {}).items():
                if key not in allowed:
                    raise ValueError(f"Unsupported GameObject field: {key}")
                old_value = getattr(obj, key)
                new_value = int(value) if key == "layer" else bool(value) if key in {"active", "is_static"} else str(value)
                _record_property(obj, key, old_value, new_value, f"Set GameObject {key}")
                changed[key] = serialize_value(getattr(obj, key))
            return {"object_id": int(obj.id), "changed": changed}

        return main_thread("gameobject.set", _set)

    @mcp.tool(name="gameobject.delete")
    def gameobject_delete(object_id: int) -> dict:
        """Delete a GameObject through the hierarchy undo tracker."""

        def _delete():
            obj = find_game_object(object_id)
            name = str(obj.name)
            from Infernux.engine.undo._trackers import HierarchyUndoTracker
            HierarchyUndoTracker().record_delete(int(object_id), "MCP Delete GameObject")
            return {"deleted": True, "object_id": int(object_id), "name": name}

        return main_thread("gameobject.delete", _delete)

    @mcp.tool(name="gameobject.batch_delete")
    def gameobject_batch_delete(object_ids: list[int]) -> dict:
        """Delete multiple GameObjects."""

        def _batch_delete():
            deleted = []
            from Infernux.engine.undo._trackers import HierarchyUndoTracker
            tracker = HierarchyUndoTracker()
            for object_id in object_ids or []:
                obj = find_game_object(int(object_id))
                deleted.append({"id": int(obj.id), "name": str(obj.name), "path": _object_path(obj)})
                tracker.record_delete(int(object_id), "MCP Batch Delete GameObject")
            return {"deleted": deleted}

        return main_thread("gameobject.batch_delete", _batch_delete)

    @mcp.tool(name="gameobject.batch_create")
    def gameobject_batch_create(items: list[dict[str, Any]]) -> dict:
        """Create multiple hierarchy objects."""

        def _batch_create():
            from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
            created = []
            for item in items or []:
                parent_id = int(item.get("parent_id", 0) or 0)
                parent_path = item.get("parent_path", "")
                if parent_path:
                    parent = _find_by_path_exact(parent_path)
                    if parent is None:
                        raise FileNotFoundError(f"Parent path not found: {parent_path}")
                    parent_id = int(parent.id)
                entry = HierarchyCreationService.instance().create(
                    str(item.get("kind", "empty")),
                    parent_id=parent_id,
                    name=item.get("name") or None,
                    select=bool(item.get("select", False)),
                )
                obj = find_game_object(int(entry["id"]))
                entry["path"] = _object_path(obj)
                created.append(entry)
            return {"created": created}

        return main_thread("gameobject.batch_create", _batch_create)

    @mcp.tool(name="gameobject.duplicate")
    def gameobject_duplicate(object_id: int, parent_id: int = 0, name: str = "", select: bool = True) -> dict:
        """Duplicate a GameObject using Scene.instantiate_game_object."""

        def _duplicate():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            source = find_game_object(object_id)
            parent = scene.find_by_id(int(parent_id)) if parent_id else None
            obj = scene.instantiate_game_object(source, parent)
            if obj is None:
                raise RuntimeError("Failed to duplicate GameObject.")
            if name:
                obj.name = str(name)
            from Infernux.engine.undo._trackers import HierarchyUndoTracker
            HierarchyUndoTracker().record_create(int(obj.id), "MCP Duplicate GameObject")
            if select:
                from Infernux.engine.ui.selection_manager import SelectionManager
                SelectionManager.instance().select(int(obj.id))
            return _serialize_object(obj, depth=1, include_components=True, include_inactive=True)

        return main_thread("gameobject.duplicate", _duplicate)

    @mcp.tool(name="gameobject.set_parent")
    def gameobject_set_parent(object_id: int, parent_id: int = 0, world_position_stays: bool = True) -> dict:
        """Set or clear a GameObject parent."""

        def _set_parent():
            obj = find_game_object(object_id)
            old_parent = obj.get_parent()
            new_parent = find_game_object(parent_id) if parent_id else None
            obj.set_parent(new_parent, bool(world_position_stays))
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm:
                sfm.mark_dirty()
            return {
                "object_id": int(obj.id),
                "parent_id": int(getattr(obj.get_parent(), "id", 0) or 0),
                "old_parent_id": int(getattr(old_parent, "id", 0) or 0),
            }

        return main_thread("gameobject.set_parent", _set_parent)

    @mcp.tool(name="gameobject.set_sibling_index")
    def gameobject_set_sibling_index(object_id: int, index: int) -> dict:
        """Move a GameObject within its current sibling list."""

        def _set_sibling_index():
            obj = find_game_object(object_id)
            old_index = int(obj.transform.get_sibling_index())
            obj.transform.set_sibling_index(int(index))
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm:
                sfm.mark_dirty()
            return {
                "object_id": int(obj.id),
                "old_index": old_index,
                "index": int(obj.transform.get_sibling_index()),
                "parent_id": int(getattr(obj.get_parent(), "id", 0) or 0),
            }

        return main_thread("gameobject.set_sibling_index", _set_sibling_index)

    @mcp.tool(name="scene.clear_generated")
    def scene_clear_generated(name_prefix: str = "MCP", root_path: str = "") -> dict:
        """Delete generated scene objects by root path or name prefix."""

        def _clear():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            targets = []
            if root_path:
                root = _find_by_path_exact(root_path)
                if root is not None:
                    targets.append(root)
            else:
                for obj in scene.get_root_objects() or []:
                    if str(obj.name).startswith(name_prefix):
                        targets.append(obj)
            from Infernux.engine.undo._trackers import HierarchyUndoTracker
            tracker = HierarchyUndoTracker()
            deleted = []
            for obj in targets:
                deleted.append({"id": int(obj.id), "name": str(obj.name), "path": _object_path(obj)})
                tracker.record_delete(int(obj.id), "MCP Clear Generated")
            return {"deleted": deleted}

        return main_thread("scene.clear_generated", _clear)

    @mcp.tool(name="transform.set")
    def transform_set(object_id: int, values: dict[str, Any]) -> dict:
        """Set Transform fields such as position, euler_angles, or local_scale."""

        def _set():
            obj = find_game_object(object_id)
            trans = obj.transform
            if trans is None:
                raise RuntimeError(f"GameObject {object_id} has no Transform.")
            allowed = {
                "position",
                "local_position",
                "euler_angles",
                "local_euler_angles",
                "local_scale",
            }
            changed: dict[str, Any] = {}
            from Infernux.engine.ui._inspector_undo import _record_property
            for key, value in values.items():
                if key not in allowed:
                    raise ValueError(f"Unsupported Transform field: {key}")
                old_value = getattr(trans, key)
                new_value = coerce_vector3(value)
                _record_property(trans, key, old_value, new_value, f"Set Transform {key}")
                changed[key] = serialize_vector(getattr(trans, key))
            return {"object_id": int(obj.id), "changed": changed}

        return main_thread("transform.set", _set)

    @mcp.tool(name="component.set_field")
    def component_set_field(
        object_id: int,
        component_type: str,
        field: str,
        value: Any,
        ordinal: int = 0,
    ) -> dict:
        """Set a field/property on a component attached to a GameObject."""

        def _set():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            old_value = getattr(comp, field)
            new_value = _coerce_property_value(field, value)
            from Infernux.engine.ui._inspector_undo import _record_property
            _record_property(comp, field, old_value, new_value, f"Set {component_type}.{field}")
            return {
                "object_id": int(obj.id),
                "component": serialize_component(comp),
                "field": field,
                "value": serialize_value(getattr(comp, field)),
            }

        return main_thread("component.set_field", _set)

    @mcp.tool(name="component.set_fields")
    def component_set_fields(object_id: int, component_type: str, values: dict[str, Any], ordinal: int = 0) -> dict:
        """Set multiple fields/properties on a component."""

        def _set_fields():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            from Infernux.engine.ui._inspector_undo import _record_property
            changed = {}
            for field, value in (values or {}).items():
                old_value = getattr(comp, field)
                new_value = _coerce_property_value(field, value)
                _record_property(comp, field, old_value, new_value, f"Set {component_type}.{field}")
                changed[field] = serialize_value(getattr(comp, field))
            return {"object_id": int(obj.id), "component": serialize_component(comp), "changed": changed}

        return main_thread("component.set_fields", _set_fields)

    @mcp.tool(name="component.ensure")
    def component_ensure(
        object_id: int,
        component_type: str,
        script_path: str = "",
        fields: dict[str, Any] | None = None,
    ) -> dict:
        """Return an existing component or add it if missing."""

        def _ensure():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, 0)
            created = False
            if comp is None:
                before_ids = _component_ids(obj)
                if script_path:
                    from Infernux.components import load_and_create_component
                    from Infernux.mcp.tools.common import get_asset_database
                    comp = load_and_create_component(script_path, asset_database=get_asset_database(), type_name=component_type)
                    if comp is None:
                        raise RuntimeError(f"Script did not create component '{component_type}'.")
                    comp = obj.add_py_component(comp)
                    is_py = True
                else:
                    comp = obj.add_component(component_type)
                    is_py = _is_python_script_component(comp)
                if comp is None:
                    raise RuntimeError(f"Failed to add component '{component_type}'.")
                from Infernux.engine.ui._inspector_undo import _record_add_component_compound
                _record_add_component_compound(obj, component_type, comp, before_ids, is_py=is_py)
                created = True
            for key, value in (fields or {}).items():
                setattr(comp, key, _coerce_property_value(key, value))
            return {"object_id": int(obj.id), "created": created, "component": serialize_component(comp), "components": _all_components(obj)}

        return main_thread("component.ensure", _ensure)

    @mcp.tool(name="component.list_on_object")
    def component_list_on_object(object_id: int) -> dict:
        """List components attached to a GameObject."""

        def _list_on_object():
            obj = find_game_object(object_id)
            return {"object_id": int(obj.id), "components": _all_components(obj)}

        return main_thread("component.list_on_object", _list_on_object)

    @mcp.tool(name="component.get_field")
    def component_get_field(object_id: int, component_type: str, field: str, ordinal: int = 0) -> dict:
        """Read a field/property from a component attached to a GameObject."""

        def _get():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            return {
                "object_id": int(obj.id),
                "component": serialize_component(comp),
                "field": field,
                "value": serialize_value(getattr(comp, field)),
            }

        return main_thread("component.get_field", _get)

    @mcp.tool(name="component.get")
    def component_get(object_id: int, component_type: str, ordinal: int = 0) -> dict:
        """Return component metadata and serializable field values."""

        def _get_component():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            return _component_snapshot(obj, comp)

        return main_thread("component.get", _get_component)

    @mcp.tool(name="component.remove")
    def component_remove(object_id: int, component_type: str, ordinal: int = 0) -> dict:
        """Remove a native or Python component from a GameObject."""

        def _remove():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            is_py = _is_python_script_component(comp)
            type_name = getattr(comp, "type_name", type(comp).__name__)
            from Infernux.engine.undo import UndoManager
            mgr = UndoManager.instance()
            if mgr:
                if is_py and hasattr(obj, "remove_py_component"):
                    from Infernux.engine.undo._component_commands import RemovePyComponentCommand
                    mgr.execute(RemovePyComponentCommand(int(obj.id), comp, "MCP Remove Component"))
                else:
                    from Infernux.engine.undo._component_commands import RemoveNativeComponentCommand
                    mgr.execute(RemoveNativeComponentCommand(int(obj.id), str(type_name), comp, "MCP Remove Component"))
            elif is_py and hasattr(obj, "remove_py_component"):
                obj.remove_py_component(comp)
            else:
                obj.remove_component(comp)
            return {"object_id": int(obj.id), "removed": str(type_name), "components": _all_components(obj)}

        return main_thread("component.remove", _remove)

    @mcp.tool(name="component.list_types")
    def component_list_types() -> dict:
        """List known built-in and Python component types."""

        def _list():
            import Infernux.components.builtin  # noqa: F401 - ensure built-ins register
            from Infernux.components.builtin_component import BuiltinComponent
            from Infernux.components.registry import get_all_types
            builtin = [
                _component_type_entry(name, cls, builtin=True)
                for name, cls in sorted(BuiltinComponent._builtin_registry.items())
            ]
            scripts = [
                _component_type_entry(name, cls, builtin=False)
                for name, cls in sorted(get_all_types().items())
                if not _is_builtin_family_class(cls)
            ]
            return {"builtin": builtin, "scripts": scripts}

        return main_thread("component.list_types", _list)

    @mcp.tool(name="component.describe_type")
    def component_describe_type(component_type: str) -> dict:
        """Describe serialized/inspector fields for a component type."""

        def _describe():
            cls = _component_class_for_name(component_type)
            if cls is None:
                raise FileNotFoundError(f"Component type '{component_type}' was not found.")
            return _component_type_entry(component_type, cls, builtin=_is_builtin_component_class(cls), include_fields=True)

        return main_thread("component.describe_type", _describe)

    @mcp.tool(name="component.describe_field")
    def component_describe_field(component_type: str, field: str) -> dict:
        """Describe one field on a component type."""

        def _describe_field():
            cls = _component_class_for_name(component_type)
            if cls is None:
                raise FileNotFoundError(f"Component type '{component_type}' was not found.")
            for item in _component_field_schema(cls):
                if item["name"] == field:
                    return {"component_type": component_type, "field": item}
            raise FileNotFoundError(f"Field '{field}' was not found on component type '{component_type}'.")

        return main_thread("component.describe_field", _describe_field)

    @mcp.tool(name="component.get_snapshot")
    def component_get_snapshot(object_id: int, component_type: str, ordinal: int = 0) -> dict:
        """Serialize a component snapshot for restore_snapshot."""

        def _snapshot():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            payload = ""
            if hasattr(comp, "serialize"):
                try:
                    payload = comp.serialize()
                except Exception:
                    payload = ""
            return {**_component_snapshot(obj, comp), "serialized": payload}

        return main_thread("component.get_snapshot", _snapshot)

    @mcp.tool(name="component.restore_snapshot")
    def component_restore_snapshot(object_id: int, component_type: str, serialized: str, ordinal: int = 0) -> dict:
        """Restore a component from a serialized component JSON snapshot."""

        def _restore():
            obj = find_game_object(object_id)
            comp = _find_component(obj, component_type, int(ordinal))
            if comp is None:
                raise FileNotFoundError(f"Component '{component_type}' was not found on GameObject {object_id}.")
            if not hasattr(comp, "deserialize"):
                raise ValueError(f"Component '{component_type}' does not support deserialize().")
            comp.deserialize(serialized)
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm:
                sfm.mark_dirty()
            return _component_snapshot(obj, comp)

        return main_thread("component.restore_snapshot", _restore)


def _serialize_object(obj, *, depth: int, include_components: bool, include_inactive: bool) -> dict[str, Any]:
    children = []
    if depth > 0:
        try:
            raw_children = list(obj.get_children() or [])
        except Exception:
            raw_children = []
        children = [
            _serialize_object(
                child,
                depth=depth - 1,
                include_components=include_components,
                include_inactive=include_inactive,
            )
            for child in raw_children
            if include_inactive or bool(getattr(child, "active", True))
        ]

    data = {
        "id": int(obj.id),
        "name": str(obj.name),
        "active": bool(getattr(obj, "active", True)),
        "children": children,
    }
    if include_components:
        data["components"] = _all_components(obj)
    return data


def _all_components(obj) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def _append(comp) -> None:
        data = serialize_component(comp)
        component_id = int(data.get("component_id", 0) or 0)
        key = (str(data.get("type", "")), component_id)
        if component_id and key in seen:
            return
        if component_id:
            seen.add(key)
        items.append(data)

    try:
        for comp in (obj.get_components() or []):
            _append(comp)
    except Exception:
        pass
    try:
        for comp in (obj.get_py_components() or []):
            _append(comp)
    except Exception:
        pass
    return items


def _component_ids(obj) -> set[int]:
    ids: set[int] = set()
    try:
        for comp in obj.get_components() or []:
            component_id = getattr(comp, "component_id", 0)
            if component_id:
                ids.add(int(component_id))
    except Exception:
        pass
    return ids


def _object_path(obj) -> str:
    parts = []
    current = obj
    while current is not None:
        parts.append(str(current.name))
        try:
            current = current.get_parent()
        except Exception:
            current = None
    return "/".join(reversed(parts))


def _find_by_path_exact(path: str):
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        raise RuntimeError("No active scene.")
    normalized = str(path).replace("\\", "/").strip("/")
    if not normalized:
        return None
    for obj in list(scene.get_all_objects() or []):
        if _object_path(obj).strip("/") == normalized:
            return obj
    return None


def _find_component(obj, component_type: str, ordinal: int):
    matches = []
    try:
        for comp in obj.get_components() or []:
            if getattr(comp, "type_name", type(comp).__name__) == component_type:
                matches.append(comp)
    except Exception:
        pass
    try:
        for comp in obj.get_py_components() or []:
            if getattr(comp, "type_name", type(comp).__name__) == component_type or type(comp).__name__ == component_type:
                matches.append(comp)
    except Exception:
        pass
    if ordinal < 0 or ordinal >= len(matches):
        return None
    return matches[ordinal]


def _component_snapshot(obj, comp) -> dict[str, Any]:
    fields = {}
    try:
        from Infernux.components.serialized_field import get_serialized_fields
        for name in get_serialized_fields(type(comp)):
            try:
                fields[name] = serialize_value(getattr(comp, name))
            except Exception:
                pass
    except Exception:
        pass
    return {
        "object_id": int(obj.id),
        "object_name": str(obj.name),
        "component": serialize_component(comp),
        "fields": fields,
    }


def _coerce_property_value(field: str, value: Any) -> Any:
    if isinstance(value, dict) and {"x", "y", "z"}.issubset(value.keys()):
        return coerce_vector3(value)
    return value


def _is_python_script_component(comp) -> bool:
    if getattr(comp, "_cpp_type_name", ""):
        return False
    script_guid = getattr(comp, "_script_guid", None)
    if script_guid:
        return True
    try:
        from Infernux.components.component import InxComponent
        return isinstance(comp, InxComponent) and not getattr(comp, "_cpp_type_name", "")
    except Exception:
        return False


def _is_builtin_component_class(cls) -> bool:
    return bool(getattr(cls, "_cpp_type_name", ""))


def _is_builtin_family_class(cls) -> bool:
    module_name = str(getattr(cls, "__module__", ""))
    return module_name.startswith("Infernux.components.builtin") or bool(getattr(cls, "_cpp_type_name", ""))


def _component_class_for_name(component_type: str):
    import Infernux.components.builtin  # noqa: F401 - ensure built-ins register
    from Infernux.components.builtin_component import BuiltinComponent
    from Infernux.components.registry import get_type

    if component_type in BuiltinComponent._builtin_registry:
        return BuiltinComponent._builtin_registry[component_type]
    for name, cls in BuiltinComponent._builtin_registry.items():
        if cls.__name__ == component_type or name.lower() == component_type.lower():
            return cls
    return get_type(component_type)


def _component_type_entry(name: str, cls, *, builtin: bool, include_fields: bool = False) -> dict[str, Any]:
    type_name = getattr(cls, "_cpp_type_name", "") or getattr(cls, "__name__", name)
    entry = {
        "type": str(type_name),
        "class_name": getattr(cls, "__name__", str(name)),
        "category": str(getattr(cls, "_component_category_", "")),
        "builtin": bool(builtin),
    }
    if include_fields:
        entry["fields"] = _component_field_schema(cls)
    return entry


def _component_field_schema(cls) -> list[dict[str, Any]]:
    from Infernux.components.serialized_field import get_serialized_fields

    fields = []
    for name, meta in get_serialized_fields(cls).items():
        enum_values = []
        enum_type = getattr(meta, "enum_type", None)
        if enum_type is not None:
            try:
                enum_values = [
                    {"name": str(item.name), "value": int(item.value)}
                    for item in list(enum_type)
                ]
            except Exception:
                enum_values = []
        fields.append({
            "name": name,
            "type": getattr(getattr(meta, "field_type", None), "name", str(getattr(meta, "field_type", ""))),
            "default": serialize_value(getattr(meta, "default", None)),
            "range": serialize_value(getattr(meta, "range", None)),
            "tooltip": str(getattr(meta, "tooltip", "")),
            "readonly": bool(getattr(meta, "readonly", False)),
            "enum_values": enum_values,
            "asset_type": str(getattr(meta, "asset_type", "") or ""),
            "component_type": str(getattr(meta, "component_type", "") or ""),
        })
    return fields
