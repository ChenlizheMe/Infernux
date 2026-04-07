"""
Inspector callback wiring — extracted from EditorBootstrap.

Provides :func:`wire_inspector_callbacks` which attaches all Python-side
callbacks to a C++ ``InspectorPanel`` instance.
"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Infernux.engine.bootstrap import EditorBootstrap


def _can_remove_component(obj, comp, type_name, is_native):
    """Check whether *comp* may be removed from *obj* (standalone, no closure deps)."""
    if is_native:
        blockers = []
        if hasattr(obj, 'get_remove_component_blockers'):
            try:
                blockers = list(obj.get_remove_component_blockers(comp) or [])
            except RuntimeError:
                blockers = []
        can_remove = not blockers
        if can_remove and hasattr(obj, 'can_remove_component'):
            can_remove = bool(obj.can_remove_component(comp))
        if not can_remove:
            from Infernux.debug import Debug
            suffix = (
                f" required by: {', '.join(blockers)}"
                if blockers else
                "another component depends on it"
            )
            Debug.log_warning(f"Cannot remove '{type_name}' — {suffix}")
            return False
    return True


def _wire_material_sections(ip, _t, engine, _inspector_support,
                            _get_cached_component_maps,
                            _current_scene_and_versions,
                            _material_section_cache):
    """Wire material-section rendering callback onto *ip*."""
    _inline_material_state = {
        "cache": {},
        "exec_layer": None,
    }

    def _make_inline_material_panel_adapter():
        class _Adapter:
            def __init__(self):
                self._inline_material_cache = _inline_material_state["cache"]
                self._inline_material_exec_layer = _inline_material_state["exec_layer"]

            def _get_native_engine(self):
                return engine.get_native_engine()

            def _ensure_material_file_path(self, material):
                return _inspector_support.ensure_material_file_path(material)

            def _sync_back(self):
                _inline_material_state["cache"] = self._inline_material_cache
                _inline_material_state["exec_layer"] = self._inline_material_exec_layer

        return _Adapter()

    def _collect_material_renderers(items, native_map, obj, wrapper_cls):
        """Collect MeshRenderer tuples and their signature parts."""
        from Infernux.components.builtin_component import BuiltinComponent

        renderers = []
        signature_parts = []
        for item in items:
            if not item.is_native or item.type_name != "MeshRenderer":
                continue
            renderer = native_map.get(item.component_id)
            if renderer is None:
                continue
            if wrapper_cls is not None and not isinstance(renderer, BuiltinComponent):
                try:
                    renderer = wrapper_cls._get_or_create_wrapper(renderer, obj)
                except Exception:
                    pass
            mat_count = getattr(renderer, 'material_count', 0) or 1
            try:
                material_guids = tuple(renderer.get_material_guids() or [])
            except Exception:
                material_guids = ()
            try:
                slot_names = tuple(renderer.get_material_slot_names() or [])
            except Exception:
                slot_names = ()
            renderers.append((renderer, mat_count, material_guids, slot_names))
            signature_parts.append((
                getattr(renderer, 'component_id', id(renderer)),
                mat_count,
                material_guids,
                slot_names,
            ))
        return renderers, tuple(signature_parts)

    def _rebuild_material_entries(renderers):
        """Build the valid_entries list from collected renderers."""
        valid_entries = []
        for renderer, mat_count, material_guids, slot_names in renderers:
            for slot_idx in range(mat_count):
                try:
                    mat = renderer.get_effective_material(slot_idx)
                except Exception:
                    mat = None
                if mat is None:
                    continue
                if slot_idx < len(slot_names) and slot_names[slot_idx]:
                    label = f"{slot_names[slot_idx]} (Slot {slot_idx})"
                else:
                    label = f"Element {slot_idx}"
                is_default = slot_idx >= len(material_guids) or not material_guids[slot_idx]
                valid_entries.append({
                    "label": label,
                    "material": mat,
                    "is_default": is_default,
                })
        return valid_entries

    def _render_material_sections(ctx, obj_id):
        from Infernux.components.builtin_component import BuiltinComponent
        from Infernux.engine.ui import inspector_material as mat_ui
        from Infernux.engine.ui.inspector_utils import render_compact_section_header, render_info_text
        from Infernux.engine.ui.theme import Theme, ImGuiCol, ImGuiStyleVar

        scene, items, native_map, _py_map = _get_cached_component_maps(obj_id)
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return

        wrapper_cls = BuiltinComponent._builtin_registry.get("MeshRenderer")
        renderers, signature = _collect_material_renderers(items, native_map, obj, wrapper_cls)

        if not renderers:
            return

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP * 1.5)
        ctx.separator()
        ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT)
        ctx.label(_t("inspector.material_overrides"))
        ctx.pop_style_color(1)
        ctx.separator()
        if not render_compact_section_header(
            ctx, "Materials##obj_mat_sections", level="primary", default_open=True
        ):
            return

        _scene, scene_version, structure_version = _current_scene_and_versions()
        if (
            _material_section_cache["object_id"] == obj_id
            and _material_section_cache["scene_version"] == scene_version
            and _material_section_cache["structure_version"] == structure_version
            and _material_section_cache["signature"] == signature
        ):
            valid_entries = _material_section_cache["entries"]
        else:
            valid_entries = _rebuild_material_entries(renderers)
            _material_section_cache["object_id"] = obj_id
            _material_section_cache["scene_version"] = scene_version
            _material_section_cache["structure_version"] = structure_version
            _material_section_cache["signature"] = signature
            _material_section_cache["entries"] = valid_entries

        owner_name = getattr(obj, 'name', '') or ''
        multiple_renderers = len(renderers) > 1

        if not valid_entries:
            return

        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)
        for index, entry in enumerate(valid_entries):
            title = entry["label"]
            if multiple_renderers and owner_name:
                title = f"{owner_name} / {title}"
            if not render_compact_section_header(
                ctx, f"{title}##mat_entry_{index}", level="secondary", default_open=True
            ):
                continue

            if entry["is_default"]:
                render_info_text(ctx, "Using the renderer's effective default material")

            adapter = _make_inline_material_panel_adapter()
            ctx.push_id(index)
            try:
                mat_ui.render_inline_material_body(ctx, adapter, entry["material"], cache_key=f"obj_mat_{obj_id}_{index}")
            finally:
                ctx.pop_id()
                adapter._sync_back()

            if index != len(valid_entries) - 1:
                ctx.separator()
        ctx.pop_style_var(2)

    ip.render_material_sections = _render_material_sections


def _get_add_component_entries():
    """Build the list of addable component entries (native + engine + scripts).

    No closure dependencies — safe as a module-level function.
    """
    from Infernux.lib import InspectorAddComponentEntry, get_registered_component_types

    entries = []
    for type_name in sorted(get_registered_component_types()):
        if type_name == "Transform":
            continue
        e = InspectorAddComponentEntry()
        e.display_name = type_name
        e.category = "Built-in"
        e.is_native = True
        entries.append(e)

    from Infernux.renderstack.render_stack import RenderStack
    for display_name, _comp_cls in [("RenderStack", RenderStack)]:
        e = InspectorAddComponentEntry()
        e.display_name = display_name
        e.category = "Engine"
        e.is_native = False
        e.script_path = ""
        entries.append(e)

    import os
    from Infernux.engine.project_context import get_project_root
    from Infernux.components.script_loader import load_component_from_file, ScriptLoadError

    project_root = get_project_root()
    if project_root and os.path.isdir(project_root):
        for dirpath, _dirnames, filenames in os.walk(project_root):
            rel = os.path.relpath(dirpath, project_root)
            if any(part.startswith('.') or part in (
                    '__pycache__', 'build', 'Library',
                    'ProjectSettings', 'Logs', 'Temp')
                   for part in rel.split(os.sep)):
                continue
            for fn in filenames:
                if not fn.endswith('.py') or fn.startswith('_'):
                    continue
                full = os.path.join(dirpath, fn)
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4096)
                if 'InxComponent' not in content:
                    continue
                try:
                    comp_class = load_component_from_file(full)
                except (ScriptLoadError, Exception):
                    continue
                e = InspectorAddComponentEntry()
                e.display_name = comp_class.__name__
                e.category = "Scripts"
                e.is_native = False
                e.script_path = full
                entries.append(e)
    return entries


def _load_script_component(script_path, asset_database):
    """Load a script component from *script_path*, returning the instance or ``None``.

    Logs errors via ``Debug`` on failure.
    """
    from Infernux.components import load_and_create_component
    from Infernux.debug import Debug
    try:
        instance = load_and_create_component(
            script_path, asset_database=asset_database)
    except Exception as exc:
        Debug.log_error(f"Failed to load script '{script_path}': {exc}")
        return None
    if instance is None:
        Debug.log_error(f"No InxComponent found in '{script_path}'")
    return instance


def _remove_component_impl(obj_id, type_name, comp_id, is_native,
                           resolve_component, can_remove_component,
                           invalidate_cache, bump_values):
    """Remove a component from an object — module-level, no closure deps."""
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    obj = scene.find_by_id(obj_id) if scene else None
    if obj is None:
        return False
    comp = resolve_component(obj_id, comp_id, is_native)
    if comp is None:
        return False
    if not can_remove_component(obj, comp, type_name, is_native):
        return False
    from Infernux.engine.undo import UndoManager
    mgr = UndoManager.instance()
    if is_native:
        from Infernux.engine.undo import RemoveNativeComponentCommand
        if mgr:
            mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
            invalidate_cache()
            bump_values()
            return True
        ok = obj.remove_component(comp) is not False
    else:
        from Infernux.engine.undo import RemovePyComponentCommand
        if mgr:
            mgr.execute(RemovePyComponentCommand(obj.id, comp))
            invalidate_cache()
            bump_values()
            return True
        ok = obj.remove_py_component(comp) is not False
    if ok:
        invalidate_cache()
        bump_values()
    return ok


def wire_inspector_callbacks(bs: EditorBootstrap) -> None:
    """Wire C++ InspectorPanel callbacks to Python managers."""
    ip = bs.inspector_panel
    engine = bs.engine
    from Infernux.engine.i18n import t as _t
    from Infernux.engine.ui import inspector_support as _inspector_support
    _bump_inspector_values = _inspector_support.bump_inspector_value_generation
    _record_profile_count = _inspector_support.record_inspector_profile_count
    _record_profile_timing = _inspector_support.record_inspector_profile_timing

    # ── Translation ────────────────────────────────────────────────
    ip.translate = _t

    # ── Selection ──────────────────────────────────────────────────
    from Infernux.engine.ui.selection_manager import SelectionManager

    ip.is_multi_selection = lambda: SelectionManager.instance().is_multi()
    ip.get_selected_ids = lambda: SelectionManager.instance().get_ids()
    ip.get_value_generation = _inspector_support.get_inspector_value_generation
    ip.consume_component_body_profile = _inspector_support.consume_inspector_profile_metrics

    # ── Object info ────────────────────────────────────────────────
    from Infernux.lib import SceneManager, InspectorObjectInfo

    _component_cache = {
        "object_id": 0,
        "scene_version": -1,
        "structure_version": -1,
        "items": [],
        "native_map": {},
        "py_map": {},
    }
    _material_section_cache = {
        "object_id": 0,
        "scene_version": -1,
        "structure_version": -1,
        "signature": (),
        "entries": [],
    }

    def _invalidate_material_section_cache():
        _material_section_cache["object_id"] = 0
        _material_section_cache["scene_version"] = -1
        _material_section_cache["structure_version"] = -1
        _material_section_cache["signature"] = ()
        _material_section_cache["entries"] = []

    def _invalidate_component_cache():
        _component_cache["object_id"] = 0
        _component_cache["scene_version"] = -1
        _component_cache["structure_version"] = -1
        _component_cache["items"] = []
        _component_cache["native_map"] = {}
        _component_cache["py_map"] = {}
        _invalidate_material_section_cache()

    def _current_scene_and_versions():
        scene = SceneManager.instance().get_active_scene()
        scene_version = getattr(scene, 'structure_version', -1) if scene else -1
        structure_version = _inspector_support.get_component_structure_version()
        return scene, scene_version, structure_version

    def _get_component_payload(obj_id):
        scene, scene_version, structure_version = _current_scene_and_versions()
        if (
            _component_cache["object_id"] == obj_id
            and _component_cache["scene_version"] == scene_version
            and _component_cache["structure_version"] == structure_version
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
                return scene, _component_cache["items"], native_map, py_map

        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            _invalidate_component_cache()
            return scene, [], {}, {}

        items = []
        native_map = {}
        py_map = {}

        for comp in (obj.get_components() or []):
            if _is_python_component_entry(comp):
                continue
            comp_type_name = getattr(comp, 'type_name', type(comp).__name__)
            if comp_type_name == "Transform":
                continue
            comp_id = getattr(comp, 'component_id', id(comp))
            ci = InspectorComponentInfo()
            ci.type_name = comp_type_name
            ci.component_id = comp_id
            ci.enabled = bool(getattr(comp, 'enabled', True))
            ci.is_native = True
            ci.is_script = False
            ci.is_broken = False
            ci.icon_id = _get_component_icon_id(comp_type_name, False)
            items.append(ci)
            native_map[comp_id] = comp

        for py_comp in (obj.get_py_components() or []):
            comp_id = getattr(py_comp, 'component_id', id(py_comp))
            ci = InspectorComponentInfo()
            ci.type_name = getattr(py_comp, 'type_name', type(py_comp).__name__)
            ci.component_id = comp_id
            ci.enabled = bool(getattr(py_comp, 'enabled', True))
            ci.is_native = False
            ci.is_script = True
            ci.is_broken = bool(getattr(py_comp, '_is_broken', False))
            ci.broken_error = (
                getattr(py_comp, '_broken_error', '') or ''
                if ci.is_broken else ''
            )
            ci.icon_id = _get_component_icon_id(ci.type_name, True)
            items.append(ci)
            py_map[comp_id] = py_comp

        _component_cache["object_id"] = obj_id
        _component_cache["scene_version"] = scene_version
        _component_cache["structure_version"] = structure_version
        _component_cache["items"] = items
        _component_cache["native_map"] = native_map
        _component_cache["py_map"] = py_map
        return scene, items, native_map, py_map

    def _get_cached_component_maps(obj_id):
        scene, scene_version, structure_version = _current_scene_and_versions()
        if (
            _component_cache["object_id"] == obj_id
            and _component_cache["scene_version"] == scene_version
            and _component_cache["structure_version"] == structure_version
        ):
            return scene, _component_cache["items"], _component_cache["native_map"], _component_cache["py_map"]
        return _get_component_payload(obj_id)

    def _resolve_component(obj_id, comp_id, is_native):
        _scene, _items, native_map, py_map = _get_cached_component_maps(obj_id)
        if is_native:
            return native_map.get(comp_id)
        return py_map.get(comp_id)

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
        elif prop_name == "name":
            new_val = value_str
        elif prop_name == "tag":
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
            _bump_inspector_values()
        if prop_name == "active":
            actual = getattr(obj, prop_name, None)
            if actual != new_val:
                from Infernux.debug import Debug
                Debug.log_warning(
                    f"[Inspector] SetActive failed: old={old_val}, "
                    f"requested={new_val}, actual={actual}, obj={obj_id}")

    ip.set_object_property = _set_object_property

    # ── Transform ──────────────────────────────────────────────────
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

    def _set_transform_data(obj_id, td):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return
        trans = obj.get_transform()
        if trans is None:
            return
        from Infernux.lib import Vector3
        trans.local_position = Vector3(td.px, td.py_, td.pz)
        trans.local_euler_angles = Vector3(td.rx, td.ry, td.rz)
        trans.local_scale = Vector3(td.sx, td.sy, td.sz)
        _bump_inspector_values()

    ip.set_transform_data = _set_transform_data

    # ── Component enumeration ──────────────────────────────────────
    from Infernux.lib import InspectorComponentInfo
    from Infernux.components.component import InxComponent

    def _is_python_component_entry(component) -> bool:
        return isinstance(component, InxComponent) or hasattr(component, 'get_py_component')

    def _get_component_list(obj_id):
        _scene, items, _native_map, _py_map = _get_component_payload(obj_id)
        return items

    ip.get_component_list = _get_component_list

    # ── Component icons ────────────────────────────────────────────
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

    ip.get_component_icon_id = _get_component_icon_id

    # ── Component body rendering ───────────────────────────────────
    from Infernux.engine.ui import inspector_components as comp_ui

    def _render_component_body(ctx, obj_id, type_name, comp_id, is_native):
        _record_profile_count("bodyResolve_count")
        _resolve_t0 = _time.perf_counter()
        comp = _resolve_component(obj_id, comp_id, is_native)
        _record_profile_timing("bodyResolve", (_time.perf_counter() - _resolve_t0) * 1000.0)
        if comp is None:
            return
        if is_native:
            _record_profile_count("bodyNativeDispatch_count")
            _native_t0 = _time.perf_counter()
            comp_ui.render_component(ctx, comp)
            _record_profile_timing("bodyNativeDispatch", (_time.perf_counter() - _native_t0) * 1000.0)
            return
        else:
            _record_profile_count("bodyPyCheck_count")
            _py_check_t0 = _time.perf_counter()
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
            _record_profile_timing("bodyPyCheck", (_time.perf_counter() - _py_check_t0) * 1000.0)
            if _script_err:
                from Infernux.engine.ui.theme import Theme, ImGuiCol
                ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
                ctx.text_wrapped(_script_err)
                ctx.pop_style_color(1)
            else:
                from Infernux.engine.ui.inspector_components import render_py_component
                _record_profile_count("bodyPyDispatch_count")
                _py_render_t0 = _time.perf_counter()
                render_py_component(ctx, comp)
                _record_profile_timing("bodyPyDispatch", (_time.perf_counter() - _py_render_t0) * 1000.0)
            return

    ip.render_component_body = _render_component_body

    # ── Component clipboard & context menu ─────────────────────────
    _comp_clipboard = {
        "type_name": "",
        "is_native": True,
        "script_guid": "",
        "json": "",
    }

    def _copy_component_to_clipboard(comp, type_name, is_native):
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

    def _has_comp_clipboard():
        return bool(_comp_clipboard["type_name"] and _comp_clipboard["json"])

    def _can_paste_values(comp, type_name, is_native):
        if not _has_comp_clipboard():
            return False
        return (_comp_clipboard["type_name"] == type_name and
                _comp_clipboard["is_native"] == is_native)

    def _paste_as_new_component(obj):
        from Infernux.engine.undo import UndoManager
        tn = _comp_clipboard["type_name"]
        native = _comp_clipboard["is_native"]
        json_data = _comp_clipboard["json"]
        guid = _comp_clipboard["script_guid"]
        mgr = UndoManager.instance()
        if native:
            result = obj.add_component(tn)
            if result and json_data and hasattr(result, "deserialize"):
                try:
                    result.deserialize(json_data)
                except Exception:
                    pass
        else:
            from Infernux.engine.component_restore import create_component_instance
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            asset_db = sfm._asset_database if sfm else None
            instance, _sp = create_component_instance(
                guid, tn, asset_database=asset_db)
            if instance is None:
                from Infernux.debug import Debug
                Debug.log_warning(f"Cannot paste: failed to create '{tn}'")
                return
            if json_data:
                try:
                    instance._deserialize_fields(json_data, _skip_on_after_deserialize=True)
                except TypeError:
                    instance._deserialize_fields(json_data)
                except Exception:
                    pass
            if guid:
                try:
                    instance._script_guid = guid
                except Exception:
                    pass
            obj.add_py_component(instance)
        _invalidate_component_cache()

    def _paste_values_to_component(comp, is_native):
        json_data = _comp_clipboard["json"]
        if not json_data:
            return
        from Infernux.engine.undo import UndoManager
        if is_native and hasattr(comp, "deserialize"):
            try:
                comp.deserialize(json_data)
            except Exception:
                pass
        elif hasattr(comp, "_deserialize_fields"):
            try:
                comp._deserialize_fields(json_data, _skip_on_after_deserialize=True)
            except TypeError:
                comp._deserialize_fields(json_data)
            except Exception:
                pass
        _bump_inspector_values()

    def _get_script_path_for_component(comp):
        guid = getattr(comp, '_script_guid', None)
        if not guid:
            return ''
        adb = engine.get_asset_database()
        if adb:
            path = adb.get_path_from_guid(guid)
            if path:
                return path
        return ''

    _project_path = bs.project_path

    def _render_component_context_menu(ctx, obj_id, type_name, comp_id, is_native):
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(obj_id) if scene else None
        if obj is None:
            return False
        comp = _resolve_component(obj_id, comp_id, is_native)
        if comp is None:
            return False

        if not is_native:
            script_path = _get_script_path_for_component(comp)
            if script_path:
                if ctx.selectable(_t("inspector.show_script")):
                    from Infernux.engine.ui import project_utils
                    project_utils.open_file_with_system(
                        script_path, project_root=_project_path)
                    ctx.close_current_popup()
                    return False
                ctx.separator()

        if ctx.selectable(_t("inspector.copy_properties")):
            _copy_component_to_clipboard(comp, type_name, is_native)
            ctx.close_current_popup()
            return False

        has_clip = _has_comp_clipboard()
        if not has_clip:
            ctx.begin_disabled()
        if ctx.selectable(_t("inspector.paste_as_new")):
            _paste_as_new_component(obj)
            ctx.close_current_popup()
            if not has_clip:
                ctx.end_disabled()
            return False
        if not has_clip:
            ctx.end_disabled()

        can_paste_vals = _can_paste_values(comp, type_name, is_native)
        if not can_paste_vals:
            ctx.begin_disabled()
        if ctx.selectable(_t("inspector.paste_properties")):
            _paste_values_to_component(comp, is_native)
            ctx.close_current_popup()
            if not can_paste_vals:
                ctx.end_disabled()
            return False
        if not can_paste_vals:
            ctx.end_disabled()

        ctx.separator()

        if ctx.selectable(_t("inspector.remove")):
            if not _can_remove_component(obj, comp, type_name, is_native):
                return False
            if is_native:
                from Infernux.engine.undo import UndoManager, RemoveNativeComponentCommand
                mgr = UndoManager.instance()
                if mgr:
                    mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
                    _invalidate_component_cache()
                elif obj.remove_component(comp) is False:
                    return False
                else:
                    _invalidate_component_cache()
            else:
                from Infernux.engine.undo import UndoManager, RemovePyComponentCommand
                mgr = UndoManager.instance()
                if mgr:
                    mgr.execute(RemovePyComponentCommand(obj.id, comp))
                    _invalidate_component_cache()
                elif obj.remove_py_component(comp) is False:
                    return False
                else:
                    _invalidate_component_cache()
            ctx.close_current_popup()
            return True
        return False

    ip.render_component_context_menu = _render_component_context_menu

    # ── Component enabled toggle ───────────────────────────────────
    def _set_component_enabled(obj_id, comp_id, new_enabled, is_native):
        comp = _resolve_component(obj_id, comp_id, is_native)
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
            _bump_inspector_values()
        for item in _component_cache["items"]:
            if item.component_id == comp_id:
                item.enabled = bool(new_enabled)
                break

    ip.set_component_enabled = _set_component_enabled

    # ── Add Component ──────────────────────────────────────────────
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
        from Infernux.debug import Debug
        if is_native:
            before_ids = _get_component_ids(obj)
            result = obj.add_component(type_name_or_path)
            if result is not None:
                Debug.log_internal(f"Added component: {type_name_or_path}")
                _record_add_component_compound(
                    obj, type_name_or_path, result, before_ids, is_py=False)
                _invalidate_component_cache()
                _bump_inspector_values()
            else:
                Debug.log_error(f"Failed to add component: {type_name_or_path}")
        elif not script_path:
            _engine_py_map = {"RenderStack": None}
            try:
                from Infernux.renderstack.render_stack import RenderStack as _RS
                _engine_py_map["RenderStack"] = _RS
            except ImportError:
                pass
            comp_cls = _engine_py_map.get(type_name_or_path)
            if comp_cls is None:
                Debug.log_error(f"Unknown engine component: {type_name_or_path}")
                return
            if getattr(comp_cls, '_disallow_multiple_', False):
                for pc in (obj.get_py_components() or []):
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
            _invalidate_component_cache()
            _bump_inspector_values()
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
            _invalidate_component_cache()
            _bump_inspector_values()
            Debug.log_internal(f"Added component {instance.type_name}")

    ip.add_component = _add_component

    # ── Remove Component ───────────────────────────────────────────
    def _remove_component(obj_id, type_name, comp_id, is_native):
        return _remove_component_impl(
            obj_id, type_name, comp_id, is_native,
            _resolve_component, _can_remove_component,
            _invalidate_component_cache, _bump_inspector_values)

    ip.remove_component = _remove_component

    # ── Asset / File preview ───────────────────────────────────────
    def _render_asset_inspector(ctx, file_path, category):
        from Infernux.engine.ui.asset_inspector import render_asset_inspector
        try:
            render_asset_inspector(ctx, ip, file_path, category)
        except Exception as exc:
            from Infernux.debug import Debug
            Debug.log_error(f"Asset inspector render failed for '{file_path}': {exc}")

    ip.render_asset_inspector = _render_asset_inspector

    def _render_file_preview(ctx, file_path):
        import os
        if os.path.isdir(file_path):
            ctx.label(_t("inspector.folder_label").format(name=os.path.basename(file_path)))
            ctx.separator()
            ctx.label(_t("inspector.path_label").format(path=file_path))
        else:
            ctx.label(_t("inspector.file_label").format(name=os.path.basename(file_path)))
            ctx.separator()
            ctx.label(_t("inspector.no_previewer"))

    ip.render_file_preview = _render_file_preview

    # ── Material sections ──────────────────────────────────────────
    _wire_material_sections(
        ip, _t, engine, _inspector_support,
        _get_cached_component_maps, _current_scene_and_versions,
        _material_section_cache,
    )

    # ── Prefab ─────────────────────────────────────────────────────
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
            pinfo.is_transform_readonly = True
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

    # ── Tags & Layers ──────────────────────────────────────────────
    from Infernux.lib import TagLayerManager

    ip.get_all_tags = lambda: TagLayerManager.instance().get_all_tags()
    ip.get_all_layers = lambda: TagLayerManager.instance().get_all_layers()

    # ── Script drop ────────────────────────────────────────────────
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
        _invalidate_component_cache()
        _bump_inspector_values()

    ip.handle_script_drop = _handle_script_drop

    # ── Window manager ─────────────────────────────────────────────
    wm = bs.window_manager

    def _open_window(win_id):
        if wm:
            wm.open_window(win_id)

    ip.open_window = _open_window
