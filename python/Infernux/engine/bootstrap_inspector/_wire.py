"""Main wiring function for the C++ InspectorPanel."""
from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

from Infernux.debug import Debug
from Infernux.engine.bootstrap_inspector._helpers import (
    _can_remove_component,
    _get_add_component_entries,
    _get_components_safe,
    _get_py_components_safe,
    _load_script_component,
    _remove_component_impl,
)

if TYPE_CHECKING:
    from Infernux.engine.bootstrap import EditorBootstrap


class _Ctx:
    """Thin namespace shared across inspector wiring helpers."""


# ═══════ Cache initialisation ═══════════════════════════════════

def _wire_cache_init(ctx):
    """Create component and material caches, invalidation helpers."""
    _component_cache = {
        "object_id": 0, "scene_version": -1, "structure_version": -1,
        "items": [], "native_map": {}, "py_map": {},
    }
    _material_section_cache = {
        "object_id": 0, "scene_version": -1, "structure_version": -1,
        "signature": (), "entries": [],
    }
    ctx.component_cache = _component_cache
    ctx.material_section_cache = _material_section_cache

    def _invalidate_material_section_cache():
        _material_section_cache.update(
            object_id=0, scene_version=-1, structure_version=-1,
            signature=(), entries=[])

    def _invalidate_component_cache():
        _component_cache.update(
            object_id=0, scene_version=-1, structure_version=-1,
            items=[], native_map={}, py_map={})
        _invalidate_material_section_cache()

    ctx.invalidate_component_cache = _invalidate_component_cache

    def _current_scene_and_versions():
        scene = ctx.SceneManager.instance().get_active_scene()
        scene_version = getattr(scene, 'structure_version', -1) if scene else -1
        structure_version = ctx._inspector_support.get_component_structure_version()
        return scene, scene_version, structure_version

    ctx.current_scene_and_versions = _current_scene_and_versions


# ═══════ Component enumeration ═════════════════════════════════

def _wire_component_list(ctx):
    """Wire get_component_list and helper resolvers."""
    SceneManager = ctx.SceneManager
    InspectorComponentInfo = ctx.InspectorComponentInfo
    InxComponent = ctx.InxComponent
    _component_cache = ctx.component_cache
    _invalidate = ctx.invalidate_component_cache
    _versions = ctx.current_scene_and_versions

    def _is_py_entry(component):
        return isinstance(component, InxComponent) or hasattr(component, 'get_py_component')

    def _get_component_payload(obj_id):
        scene, scene_ver, struct_ver = _versions()
        if (
            _component_cache["object_id"] == obj_id
            and _component_cache["scene_version"] == scene_ver
            and _component_cache["structure_version"] == struct_ver
        ):
            items = _component_cache["items"]
            native_map = _component_cache["native_map"]
            py_map = _component_cache["py_map"]
            stale = False
            for item in items:
                comp = native_map.get(item.component_id) if item.is_native else py_map.get(item.component_id)
                if comp is None:
                    stale = True
                    break
                item.enabled = bool(getattr(comp, 'enabled', True))
                if not item.is_native:
                    item.is_broken = bool(getattr(comp, '_is_broken', False))
                    item.broken_error = (
                        getattr(comp, '_broken_error', '') or ''
                        if item.is_broken else ''
                    )
            if not stale:
                return scene, items, native_map, py_map

        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            _invalidate()
            return scene, [], {}, {}

        items, native_map, py_map = [], {}, {}

        # Single pass over get_components() preserves the actual insertion
        # order (C++ m_components vector) so the Inspector shows components
        # in chronological add-order.
        for comp in _get_components_safe(obj):
            tn = getattr(comp, 'type_name', type(comp).__name__)
            if tn == "Transform":
                continue

            if _is_py_entry(comp):
                # CastToPython already resolved PyComponentProxy to the
                # actual Python instance, so *comp* IS the Python component.
                py_comp = comp
                cid = getattr(py_comp, 'component_id', id(py_comp))
                ci = InspectorComponentInfo()
                ci.type_name = getattr(py_comp, 'type_name', type(py_comp).__name__)
                ci.component_id = cid
                ci.enabled = bool(getattr(py_comp, 'enabled', True))
                ci.is_native = False
                ci.is_script = True
                ci.is_broken = bool(getattr(py_comp, '_is_broken', False))
                ci.broken_error = (
                    getattr(py_comp, '_broken_error', '') or ''
                    if ci.is_broken else ''
                )
                ci.icon_id = ctx.get_component_icon_id(ci.type_name, True)
                items.append(ci)
                py_map[cid] = py_comp
            else:
                cid = getattr(comp, 'component_id', id(comp))
                ci = InspectorComponentInfo()
                ci.type_name = tn
                ci.component_id = cid
                ci.enabled = bool(getattr(comp, 'enabled', True))
                ci.is_native = True
                ci.is_script = False
                ci.is_broken = False
                ci.icon_id = ctx.get_component_icon_id(tn, False)
                items.append(ci)
                native_map[cid] = comp

        _component_cache.update(
            object_id=obj_id, scene_version=scene_ver,
            structure_version=struct_ver,
            items=items, native_map=native_map, py_map=py_map)
        return scene, items, native_map, py_map

    def _get_cached_maps(obj_id):
        scene, scene_ver, struct_ver = _versions()
        if (
            _component_cache["object_id"] == obj_id
            and _component_cache["scene_version"] == scene_ver
            and _component_cache["structure_version"] == struct_ver
        ):
            return (scene, _component_cache["items"],
                    _component_cache["native_map"], _component_cache["py_map"])
        return _get_component_payload(obj_id)

    ctx.get_cached_component_maps = _get_cached_maps

    def _resolve_component(obj_id, comp_id, is_native):
        _scene, _items, native_map, py_map = _get_cached_maps(obj_id)
        return native_map.get(comp_id) if is_native else py_map.get(comp_id)

    ctx.resolve_component = _resolve_component
    ctx.ip.get_component_list = lambda obj_id: _get_component_payload(obj_id)[1]


# ═══════ Object info & properties ══════════════════════════════

def _wire_object_info(ctx):
    """Wire get_object_info and set_object_property."""
    SceneManager = ctx.SceneManager
    InspectorObjectInfo = ctx.InspectorObjectInfo
    ip = ctx.ip
    _bump = ctx._bump_inspector_values

    def _get_object_info(obj_id):
        info = InspectorObjectInfo()
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return info
        info.name = obj.name
        info.active = obj.active
        info.tag = getattr(obj, 'tag', 'Untagged')
        info.layer = getattr(obj, 'layer', 0)
        info.prefab_guid = getattr(obj, 'prefab_guid', '') or ''
        info.hide_transform = getattr(obj, 'hide_transform', False)
        return info

    ip.get_object_info = _get_object_info

    def _set_object_property(obj_id, prop_name, value_str):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return
        from Infernux.engine.undo import UndoManager, SetPropertyCommand
        mgr = UndoManager.instance()
        old_val = getattr(obj, prop_name, None)
        if prop_name == "active":
            new_val = value_str.lower() in ("true", "1")
        elif prop_name in ("name", "tag"):
            new_val = value_str
        elif prop_name == "layer":
            new_val = int(value_str)
        else:
            new_val = value_str
        if mgr:
            mgr.execute(SetPropertyCommand(obj, prop_name, old_val, new_val,
                                           f"Set {prop_name}"))
        else:
            setattr(obj, prop_name, new_val)
            _bump()
        if prop_name == "active":
            actual = getattr(obj, prop_name, None)
            if actual != new_val:
                Debug.log_warning(
                    f"[Inspector] SetActive failed: old={old_val}, "
                    f"requested={new_val}, actual={actual}, obj={obj_id}")

    ip.set_object_property = _set_object_property


# ═══════ Transform ═════════════════════════════════════════════

def _wire_transform(ctx):
    """Wire transform get/set callbacks."""
    SceneManager = ctx.SceneManager
    ip = ctx.ip
    _bump = ctx._bump_inspector_values

    from Infernux.lib import InspectorTransformData

    def _get_transform_data(obj_id):
        td = InspectorTransformData()
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return td
        trans = obj.get_transform()
        if trans is None:
            return td
        lp = trans.local_position
        le = trans.local_euler_angles
        ls = trans.local_scale
        td.px, td.py_, td.pz = lp.x, lp.y, lp.z
        td.rx, td.ry, td.rz = le.x, le.y, le.z
        td.sx, td.sy, td.sz = ls.x, ls.y, ls.z
        return td

    ip.get_transform_data = _get_transform_data

    from Infernux.engine.undo import (
        UndoManager, InspectorSnapshotCommand,
        snapshot_live_transform, restore_live_transform,
    )

    def _set_transform_data(obj_id, td):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return
        trans = obj.get_transform()
        if trans is None:
            return

        mgr = UndoManager.instance()
        old_snap = None
        if mgr and mgr.enabled:
            try:
                old_snap = snapshot_live_transform(obj_id)
            except Exception:
                old_snap = None

        from Infernux.lib import Vector3
        trans.local_position = Vector3(td.px, td.py_, td.pz)
        trans.local_euler_angles = Vector3(td.rx, td.ry, td.rz)
        trans.local_scale = Vector3(td.sx, td.sy, td.sz)

        if mgr and mgr.enabled and old_snap is not None:
            try:
                new_snap = snapshot_live_transform(obj_id)
            except Exception:
                new_snap = None
            if new_snap is not None and new_snap != old_snap:
                def _restore(snap, _oid=obj_id):
                    restore_live_transform(_oid, snap)
                    _bump()
                cmd = InspectorSnapshotCommand(
                    f"transform:{obj_id}", old_snap, new_snap,
                    _restore, "Edit Transform")
                mgr.record(cmd)

        _bump()

    ip.set_transform_data = _set_transform_data


# ═══════ Icons & body rendering ════════════════════════════════

def _wire_icons_and_body(ctx):
    """Wire component icons, body rendering, and enabled toggle."""
    ip = ctx.ip
    engine = ctx.engine
    _bump = ctx._bump_inspector_values
    _record_count = ctx._record_profile_count
    _record_timing = ctx._record_profile_timing
    _component_cache = ctx.component_cache
    _inspector_support = ctx._inspector_support

    _icon_cache = {}
    _icons_loaded = [False]

    def _ensure_icons():
        if _icons_loaded[0]:
            return
        _icons_loaded[0] = True
        native_engine = engine.get_native_engine()
        if not native_engine:
            return
        import os
        import Infernux.resources as _resources
        from Infernux.lib import TextureLoader
        icons_dir = _resources.component_icons_dir
        if not os.path.isdir(icons_dir):
            return
        for fname in os.listdir(icons_dir):
            if not fname.startswith("component_") or not fname.endswith(".png"):
                continue
            key = fname[len("component_"):-len(".png")]
            tex_name = f"__compicon__{key}"
            if native_engine.has_imgui_texture(tex_name):
                _icon_cache[key] = native_engine.get_imgui_texture_id(tex_name)
                continue
            icon_path = os.path.join(icons_dir, fname)
            tex_data = TextureLoader.load_from_file(icon_path)
            if tex_data and tex_data.is_valid():
                pixels, w, h = _inspector_support.prepare_component_icon_pixels(tex_data)
                if w > 0 and h > 0:
                    tid = native_engine.upload_texture_for_imgui(tex_name, pixels, w, h)
                    if tid != 0:
                        _icon_cache[key] = tid

    def _get_component_icon_id(type_name, is_script):
        _ensure_icons()
        tid = _icon_cache.get(type_name.lower(), 0)
        if tid == 0 and is_script:
            tid = _icon_cache.get("script", 0)
        return tid

    ctx.get_component_icon_id = _get_component_icon_id
    ip.get_component_icon_id = _get_component_icon_id

    from Infernux.engine.ui import inspector_components as comp_ui

    def _render_component_body(ctx_arg, obj_id, type_name, comp_id, is_native):
        _record_count("bodyResolve_count")
        _resolve_t0 = _time.perf_counter()
        comp = ctx.resolve_component(obj_id, comp_id, is_native)
        _record_timing("bodyResolve", (_time.perf_counter() - _resolve_t0) * 1000.0)
        if comp is None:
            return
        if is_native:
            _record_count("bodyNativeDispatch_count")
            _t0 = _time.perf_counter()
            comp_ui.render_component(ctx_arg, comp)
            _record_timing("bodyNativeDispatch", (_time.perf_counter() - _t0) * 1000.0)
            return
        _record_count("bodyPyCheck_count")
        _t0 = _time.perf_counter()
        _script_err = None
        if getattr(comp, '_is_broken', False):
            _script_err = getattr(comp, '_broken_error', '') or 'Script failed to load'
        else:
            _py_guid = getattr(comp, '_script_guid', None)
            adb = engine.get_asset_database()
            if _py_guid and adb:
                from Infernux.components.script_loader import get_script_error_by_path
                _py_path = adb.get_path_from_guid(_py_guid)
                if _py_path:
                    _script_err = get_script_error_by_path(_py_path)
        _record_timing("bodyPyCheck", (_time.perf_counter() - _t0) * 1000.0)
        if _script_err:
            from Infernux.engine.ui.theme import Theme, ImGuiCol
            ctx_arg.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
            ctx_arg.text_wrapped(_script_err)
            ctx_arg.pop_style_color(1)
        else:
            from Infernux.engine.ui.inspector_components import render_py_component
            _record_count("bodyPyDispatch_count")
            _t0 = _time.perf_counter()
            render_py_component(ctx_arg, comp)
            _record_timing("bodyPyDispatch", (_time.perf_counter() - _t0) * 1000.0)

    ip.render_component_body = _render_component_body

    def _set_component_enabled(obj_id, comp_id, new_enabled, is_native):
        comp = ctx.resolve_component(obj_id, comp_id, is_native)
        if comp is None:
            return
        from Infernux.engine.undo import UndoManager, SetPropertyCommand
        mgr = UndoManager.instance()
        old_val = comp.enabled
        if mgr:
            mgr.execute(SetPropertyCommand(
                comp, "enabled", old_val, new_enabled,
                f"Toggle {getattr(comp, 'type_name', '?')}"))
        else:
            comp.enabled = new_enabled
            _bump()
        for item in _component_cache["items"]:
            if item.component_id == comp_id:
                item.enabled = bool(new_enabled)
                break

    ip.set_component_enabled = _set_component_enabled


# ═══════ Clipboard & context menu ══════════════════════════════

def _wire_clipboard_and_context(ctx):
    """Wire clipboard operations and component context menu."""
    ip = ctx.ip
    engine = ctx.engine
    _resolve = ctx.resolve_component
    _invalidate = ctx.invalidate_component_cache
    _bump = ctx._bump_inspector_values
    _t = ctx._t
    SceneManager = ctx.SceneManager

    _comp_clipboard = {
        "type_name": "", "is_native": True, "script_guid": "", "json": "",
    }

    def _copy_to_clipboard(comp, type_name, is_native):
        _comp_clipboard["type_name"] = type_name
        _comp_clipboard["is_native"] = is_native
        _comp_clipboard["script_guid"] = getattr(comp, '_script_guid', '') or ''
        try:
            if is_native and hasattr(comp, "serialize"):
                _comp_clipboard["json"] = comp.serialize()
            elif hasattr(comp, "_serialize_fields"):
                _comp_clipboard["json"] = comp._serialize_fields()
            else:
                _comp_clipboard["json"] = ""
        except Exception:
            _comp_clipboard["json"] = ""

    def _has_clip():
        return bool(_comp_clipboard["type_name"] and _comp_clipboard["json"])

    def _can_paste_values(comp, type_name, is_native):
        if not _has_clip():
            return False
        return (_comp_clipboard["type_name"] == type_name and
                _comp_clipboard["is_native"] == is_native)

    def _paste_as_new(obj):
        tn = _comp_clipboard["type_name"]
        native = _comp_clipboard["is_native"]
        json_data = _comp_clipboard["json"]
        guid = _comp_clipboard["script_guid"]
        if native:
            result = obj.add_component(tn)
            if result and json_data and hasattr(result, "deserialize"):
                try:
                    result.deserialize(json_data)
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        else:
            from Infernux.engine.component_restore import create_component_instance
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            asset_db = sfm._asset_database if sfm else None
            instance, _sp = create_component_instance(
                guid, tn, asset_database=asset_db)
            if instance is None:
                Debug.log_warning(f"Cannot paste: failed to create '{tn}'")
                return
            if json_data:
                try:
                    instance._deserialize_fields(json_data, _skip_on_after_deserialize=True)
                except TypeError:
                    instance._deserialize_fields(json_data)
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            if guid:
                try:
                    instance._script_guid = guid
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            obj.add_py_component(instance)
        _invalidate()

    def _paste_values(comp, is_native):
        json_data = _comp_clipboard["json"]
        if not json_data:
            return
        if is_native and hasattr(comp, "deserialize"):
            try:
                comp.deserialize(json_data)
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        elif hasattr(comp, "_deserialize_fields"):
            try:
                comp._deserialize_fields(json_data, _skip_on_after_deserialize=True)
            except TypeError:
                comp._deserialize_fields(json_data)
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        _bump()

    def _get_script_path(comp):
        guid = getattr(comp, '_script_guid', None)
        if not guid:
            return ''
        adb = engine.get_asset_database()
        if adb:
            path = adb.get_path_from_guid(guid)
            if path:
                return path
        return ''

    _project_path = ctx.project_path

    def _render_component_context_menu(ctx_arg, obj_id, type_name, comp_id, is_native):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return False
        comp = _resolve(obj_id, comp_id, is_native)
        if comp is None:
            return False

        if not is_native:
            script_path = _get_script_path(comp)
            if script_path:
                if ctx_arg.selectable(_t("inspector.show_script")):
                    from Infernux.engine.ui import project_utils
                    project_utils.open_file_with_system(
                        script_path, project_root=_project_path)
                    ctx_arg.close_current_popup()
                    return False
                ctx_arg.separator()

        if ctx_arg.selectable(_t("inspector.copy_properties")):
            _copy_to_clipboard(comp, type_name, is_native)
            ctx_arg.close_current_popup()
            return False

        has_clip = _has_clip()
        if not has_clip:
            ctx_arg.begin_disabled()
        if ctx_arg.selectable(_t("inspector.paste_as_new")):
            _paste_as_new(obj)
            ctx_arg.close_current_popup()
            if not has_clip:
                ctx_arg.end_disabled()
            return False
        if not has_clip:
            ctx_arg.end_disabled()

        can_paste_v = _can_paste_values(comp, type_name, is_native)
        if not can_paste_v:
            ctx_arg.begin_disabled()
        if ctx_arg.selectable(_t("inspector.paste_properties")):
            _paste_values(comp, is_native)
            ctx_arg.close_current_popup()
            if not can_paste_v:
                ctx_arg.end_disabled()
            return False
        if not can_paste_v:
            ctx_arg.end_disabled()

        ctx_arg.separator()

        if ctx_arg.selectable(_t("inspector.remove")):
            if not _can_remove_component(obj, comp, type_name, is_native):
                return False
            if is_native:
                from Infernux.engine.undo import UndoManager, RemoveNativeComponentCommand
                mgr = UndoManager.instance()
                if mgr:
                    mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
                    _invalidate()
                elif obj.remove_component(comp) is False:
                    return False
                else:
                    _invalidate()
            else:
                from Infernux.engine.undo import UndoManager, RemovePyComponentCommand
                mgr = UndoManager.instance()
                if mgr:
                    mgr.execute(RemovePyComponentCommand(obj.id, comp))
                    _invalidate()
                elif obj.remove_py_component(comp) is False:
                    return False
                else:
                    _invalidate()
            ctx_arg.close_current_popup()
            return True
        return False

    ip.render_component_context_menu = _render_component_context_menu


# ═══════ Add / remove / script-drop ════════════════════════════

def _wire_add_remove_and_drop(ctx):
    """Wire add_component, remove_component, and handle_script_drop."""
    ip = ctx.ip
    engine = ctx.engine
    SelectionManager = ctx.SelectionManager
    SceneManager = ctx.SceneManager
    _invalidate = ctx.invalidate_component_cache
    _bump = ctx._bump_inspector_values
    _resolve = ctx.resolve_component

    ip.get_add_component_entries = _get_add_component_entries

    def _add_component(type_name_or_path, is_native, script_path):
        sel = SelectionManager.instance()
        primary = sel.get_primary()
        if not primary:
            return
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(primary) if scene else None
        if obj is None:
            return
        from Infernux.engine.ui.inspector_components import (
            _record_add_component_compound, _get_component_ids
        )
        if is_native:
            before_ids = _get_component_ids(obj)
            result = obj.add_component(type_name_or_path)
            if result is not None:
                Debug.log_internal(f"Added component: {type_name_or_path}")
                _record_add_component_compound(
                    obj, type_name_or_path, result, before_ids, is_py=False)
                _invalidate()
                _bump()
            else:
                Debug.log_error(f"Failed to add component: {type_name_or_path}")
        elif not script_path:
            _engine_py_map = {"RenderStack": None}
            try:
                from Infernux.renderstack.render_stack import RenderStack as _RS
                _engine_py_map["RenderStack"] = _RS
            except ImportError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            comp_cls = _engine_py_map.get(type_name_or_path)
            if comp_cls is None:
                Debug.log_error(f"Unknown engine component: {type_name_or_path}")
                return
            if getattr(comp_cls, '_disallow_multiple_', False):
                for pc in _get_py_components_safe(obj):
                    if isinstance(pc, comp_cls):
                        Debug.log_warning(
                            f"Cannot add another '{comp_cls.__name__}' — "
                            f"only one per scene is allowed")
                        return
            instance = comp_cls()
            before_ids = _get_component_ids(obj)
            obj.add_py_component(instance)
            _record_add_component_compound(
                obj, comp_cls.__name__, instance, before_ids, is_py=True)
            _invalidate()
            _bump()
            Debug.log_internal(f"Added component {comp_cls.__name__}")
        else:
            adb = engine.get_asset_database()
            instance = _load_script_component(script_path, adb)
            if instance is None:
                return
            before_ids = _get_component_ids(obj)
            obj.add_py_component(instance)
            _record_add_component_compound(
                obj, instance.type_name, instance, before_ids, is_py=True)
            _invalidate()
            _bump()
            Debug.log_internal(f"Added component {instance.type_name}")

    ip.add_component = _add_component

    def _remove_component(obj_id, type_name, comp_id, is_native):
        return _remove_component_impl(
            obj_id, type_name, comp_id, is_native,
            _resolve, _can_remove_component, _invalidate, _bump)

    ip.remove_component = _remove_component

    def _handle_script_drop(script_path):
        sel = SelectionManager.instance()
        primary = sel.get_primary()
        if not primary:
            return
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(primary) if scene else None
        if obj is None:
            return
        adb = engine.get_asset_database()
        instance = _load_script_component(script_path, adb)
        if instance is None:
            return
        from Infernux.engine.ui.inspector_components import (
            _record_add_component_compound, _get_component_ids
        )
        before_ids = _get_component_ids(obj)
        obj.add_py_component(instance)
        _record_add_component_compound(
            obj, instance.type_name, instance, before_ids, is_py=True)
        _invalidate()
        _bump()

    ip.handle_script_drop = _handle_script_drop


# ═══════ Asset / file preview ══════════════════════════════════

def _wire_asset_preview(ctx):
    """Wire asset inspector and file preview callbacks."""
    ip = ctx.ip
    _t = ctx._t

    def _render_asset_inspector(ctx_arg, file_path, category):
        from Infernux.engine.ui.asset_inspector import render_asset_inspector
        try:
            render_asset_inspector(ctx_arg, ip, file_path, category)
        except Exception as exc:
            Debug.log_error(f"Asset inspector render failed for '{file_path}': {exc}")

    ip.render_asset_inspector = _render_asset_inspector

    def _render_file_preview(ctx_arg, file_path):
        import os
        if os.path.isdir(file_path):
            ctx_arg.label(_t("inspector.folder_label").format(name=os.path.basename(file_path)))
            ctx_arg.separator()
            ctx_arg.label(_t("inspector.path_label").format(path=file_path))
        else:
            ctx_arg.label(_t("inspector.file_label").format(name=os.path.basename(file_path)))
            ctx_arg.separator()
            ctx_arg.label(_t("inspector.no_previewer"))

    ip.render_file_preview = _render_file_preview


# ═══════ Prefab, tags, window manager ══════════════════════════

def _wire_prefab_and_misc(ctx):
    """Wire prefab info/actions, tags, layers, and window manager."""
    ip = ctx.ip
    engine = ctx.engine
    bs = ctx.bs
    _t = ctx._t
    SceneManager = ctx.SceneManager
    SelectionManager = ctx.SelectionManager
    _bump = ctx._bump_inspector_values
    _invalidate = ctx.invalidate_component_cache

    from Infernux.lib import InspectorPrefabInfo

    def _get_prefab_info(obj_id):
        pinfo = InspectorPrefabInfo()
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return pinfo
        guid = getattr(obj, 'prefab_guid', '') or ''
        if not guid:
            return pinfo
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm and not sfm.is_prefab_mode:
            pinfo.is_readonly = True
        return pinfo

    ip.get_prefab_info = _get_prefab_info

    def _prefab_action(obj_id, action):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return
        guid = getattr(obj, 'prefab_guid', '') or ''
        if not guid:
            return
        adb = engine.get_asset_database()
        if action == "select":
            if adb:
                path = adb.get_path_from_guid(guid)
                if path:
                    bs.project_panel.set_current_path(
                        __import__('os').path.dirname(path))
        elif action == "open":
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm and adb:
                path = adb.get_path_from_guid(guid)
                if path:
                    sfm.open_prefab_mode_with_undo(path)
        elif action == "apply":
            from Infernux.engine.prefab import apply_prefab_overrides
            apply_prefab_overrides(obj)
        elif action == "revert":
            from Infernux.engine.prefab import revert_prefab_overrides
            revert_prefab_overrides(obj)

    ip.prefab_action = _prefab_action

    from Infernux.lib import TagLayerManager
    ip.get_all_tags = lambda: TagLayerManager.instance().get_all_tags()
    ip.get_all_layers = lambda: TagLayerManager.instance().get_all_layers()

    wm = bs.window_manager
    ip.open_window = lambda win_id: wm.open_window(win_id) if wm else None


# ═══════ Main entry point ══════════════════════════════════════

def wire_inspector_callbacks(bs: EditorBootstrap) -> None:
    """Wire C++ InspectorPanel callbacks to Python managers."""
    ip = bs.inspector_panel
    engine = bs.engine
    from Infernux.engine.i18n import t as _t
    from Infernux.engine.ui import inspector_support as _inspector_support
    from Infernux.engine.ui.selection_manager import SelectionManager
    from Infernux.lib import SceneManager, InspectorObjectInfo, InspectorComponentInfo
    from Infernux.components.component import InxComponent

    ctx = _Ctx()
    ctx.ip = ip
    ctx.engine = engine
    ctx.bs = bs
    ctx._t = _t
    ctx._inspector_support = _inspector_support
    ctx._bump_inspector_values = _inspector_support.bump_inspector_value_generation
    ctx._record_profile_count = _inspector_support.record_inspector_profile_count
    ctx._record_profile_timing = _inspector_support.record_inspector_profile_timing
    ctx.SceneManager = SceneManager
    ctx.InspectorObjectInfo = InspectorObjectInfo
    ctx.InspectorComponentInfo = InspectorComponentInfo
    ctx.InxComponent = InxComponent
    ctx.SelectionManager = SelectionManager
    ctx.project_path = bs.project_path

    ip.translate = _t

    sel = SelectionManager.instance()
    ip.is_multi_selection = lambda: sel.is_multi()
    ip.get_selected_ids = lambda: sel.get_ids()
    ip.get_value_generation = _inspector_support.get_inspector_value_generation
    ip.consume_component_body_profile = _inspector_support.consume_inspector_profile_metrics

    _wire_cache_init(ctx)
    _wire_component_list(ctx)
    _wire_icons_and_body(ctx)
    _wire_object_info(ctx)
    _wire_transform(ctx)
    _wire_clipboard_and_context(ctx)
    _wire_add_remove_and_drop(ctx)
    _wire_asset_preview(ctx)

    from Infernux.engine.bootstrap_inspector._materials import wire_material_sections
    wire_material_sections(
        ip, _t, engine, _inspector_support,
        ctx.get_cached_component_maps, ctx.current_scene_and_versions,
        ctx.material_section_cache)

    _wire_prefab_and_misc(ctx)
