"""
Material Asset Inspector — body renderer for the unified asset inspector.

This module provides ``render_material_body(ctx, panel, state)`` which renders
the material-specific UI sections: shader selection, render settings,
dynamic properties, and auto-save scheduling.

State is managed by the unified ``asset_inspector`` module.
"""

from __future__ import annotations

import json
import os
import time as _time
from types import SimpleNamespace
from typing import Optional

from Infernux.lib import InxGUIContext
from Infernux.engine.i18n import t
from . import inspector_support as _inspector_support
from .asset_execution_layer import AssetAccessMode, get_asset_execution_layer
from .inspector_utils import (
    max_label_w,
    field_label,
    render_compact_section_header,
    _render_color_bar,
    LABEL_PAD,
)
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from . import inspector_shader_utils as shader_utils
from Infernux.debug import Debug
import logging


def _record_profile_timing(bucket: str, start_time: float) -> None:
    _inspector_support.record_inspector_profile_timing(
        bucket, (_time.perf_counter() - start_time) * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Material property renderer (JSON type system: ptype 0-7)
# ═══════════════════════════════════════════════════════════════════════════

def _get_asset_database():
    """Get the asset database from EditorServices or AssetRegistry (fallback)."""
    try:
        from .editor_services import EditorServices
        adb = EditorServices.instance()._asset_database
        if adb:
            return adb
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    try:
        from Infernux.lib import AssetRegistry
        return AssetRegistry.instance().get_asset_database()
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None


def _resolve_path_to_guid(path_str):
    """Resolve a filesystem path to an asset GUID."""
    adb = _get_asset_database()
    return adb.get_guid_from_path(path_str) or "" if adb else ""


def _resolve_texture_display(prop):
    """Return display text for a texture property's GUID."""
    import os
    tex_guid = prop.get("guid", "")
    if not isinstance(tex_guid, str) or not tex_guid:
        return t("igui.none")
    adb = _get_asset_database()
    if adb:
        tex_path = adb.get_path_from_guid(tex_guid)
        if tex_path:
            return os.path.basename(tex_path)
    return tex_guid[:8] + "..."


def _render_texture2d_property(ctx, prop, prop_name, wid_prefix, plw):
    """Render a Texture2D material property (GUID-only v2). Returns True if changed."""
    changed = False
    display = _resolve_texture_display(prop)
    field_label(ctx, prop_name, plw)

    from .igui import IGUI

    def _on_tex_drop(payload):
        nonlocal changed
        dropped = str(payload).replace("\\", "/")
        guid = _resolve_path_to_guid(dropped)
        if not guid:
            logging.getLogger(__name__).warning(
                "Cannot resolve dropped texture path to GUID: %s", dropped)
            return
        prop["guid"] = guid
        changed = True

    def _on_tex_pick(picked_path):
        nonlocal changed
        picked = str(picked_path).replace("\\", "/")
        guid = _resolve_path_to_guid(picked)
        if not guid:
            logging.getLogger(__name__).warning(
                "Cannot resolve picked texture path to GUID: %s", picked)
            return
        prop["guid"] = guid
        changed = True

    def _on_tex_clear():
        nonlocal changed
        prop["guid"] = ""
        changed = True

    def _tex_asset_items(filt):
        from .inspector_components import _picker_assets
        return _picker_assets(filt, "*.png", assets_only=True) + _picker_assets(filt, "*.jpg", assets_only=True)

    IGUI.object_field(
        ctx,
        f"{wid_prefix}_{prop_name}_tex",
        display, "Texture",
        clickable=True,
        accept="TEXTURE_FILE",
        on_drop=_on_tex_drop,
        picker_asset_items=_tex_asset_items,
        on_pick=_on_tex_pick,
        on_clear=_on_tex_clear,
    )
    return changed


def render_material_property(
    ctx: InxGUIContext,
    prop_name: str,
    prop: dict,
    ptype: int,
    value,
    plw: float,
    wid_prefix: str = "mp",
) -> bool:
    """Render one material property row.  Returns ``True`` if changed."""
    changed = False
    wid = f"##{wid_prefix}_{prop_name}"

    if ptype == 0:  # Float
        field_label(ctx, prop_name, plw)
        nv = float(ctx.drag_float(wid, float(value), 0.1, 0.0, 100.0))
        if nv != float(value):
            prop["value"] = nv
            changed = True

    elif ptype == 1:  # Float2
        x, y = value[0], value[1]
        nx, ny = ctx.vector2(prop_name, float(x), float(y), 0.1, plw)
        if [nx, ny] != [x, y]:
            prop["value"] = [nx, ny]
            changed = True

    elif ptype == 2:  # Float3
        x, y, z = value[0], value[1], value[2]
        nx, ny, nz = ctx.vector3(
            prop_name, float(x), float(y), float(z), 0.1, plw,
        )
        if [nx, ny, nz] != [x, y, z]:
            prop["value"] = [nx, ny, nz]
            changed = True

    elif ptype == 3:  # Float4
        x, y, z, w = value[0], value[1], value[2], value[3]
        nx, ny, nz, nw = ctx.vector4(
            prop_name, float(x), float(y), float(z), float(w), 0.1, plw,
        )
        if [nx, ny, nz, nw] != [x, y, z, w]:
            prop["value"] = [nx, ny, nz, nw]
            changed = True

    elif ptype == 4:  # Int
        field_label(ctx, prop_name, plw)
        nv = int(ctx.drag_int(wid, int(value), 1.0, 0, 0))
        if nv != int(value):
            prop["value"] = nv
            changed = True

    elif ptype == 5:  # Mat4
        arr = list(value)
        mat_changed = False
        for row in range(4):
            base = row * 4
            nx, ny, nz, nw = ctx.vector4(
                f"{prop_name}[{row}]",
                float(arr[base]), float(arr[base + 1]),
                float(arr[base + 2]), float(arr[base + 3]),
            )
            nr = [nx, ny, nz, nw]
            if nr != arr[base:base + 4]:
                arr[base:base + 4] = nr
                mat_changed = True
        if mat_changed:
            prop["value"] = arr
            changed = True

    elif ptype == 6:  # Texture2D
        changed = _render_texture2d_property(ctx, prop, prop_name, wid_prefix, plw)

    elif ptype == 7:  # Color
        if value is not None and len(value) >= 4:
            x, y, z, w = value[0], value[1], value[2], value[3]
        else:
            x, y, z, w = 1.0, 1.0, 1.0, 1.0
        field_label(ctx, prop_name, plw)
        allow_hdr = bool(prop.get("hdr", False))
        nr, ng, nb, na = _render_color_bar(
            ctx, wid, float(x), float(y), float(z), float(w), allow_hdr=allow_hdr)
        if (nr, ng, nb, na) != (x, y, z, w):
            prop["value"] = [nr, ng, nb, na]
            changed = True

    else:
        ctx.label(f"{prop_name}: (type {ptype})")

    return changed


# ═══════════════════════════════════════════════════════════════════════════
# Module-level shortcuts (set per render call from unified state)
# ═══════════════════════════════════════════════════════════════════════════

_native_mat = None
_cached_data: Optional[dict] = None
_shader_cache: dict = {".vert": None, ".frag": None}


# ═══════════════════════════════════════════════════════════════════════════
# Extracted section renderers (called from render_material_body)
# ═══════════════════════════════════════════════════════════════════════════

def _render_shader_section(ctx, mat_data, state, is_builtin, default_open):
    """Render the vertex/fragment shader selection section.

    Returns ``(changed, requires_deserialize, requires_pipeline_refresh, change_key)``.
    """
    changed = False
    requires_deserialize = False
    requires_pipeline_refresh = False
    change_key = ""

    if is_builtin:
        ctx.begin_disabled(True)
    section_t0 = _time.perf_counter()
    if render_compact_section_header(ctx, t("material.shader_section"), level="secondary",
                                     default_open=default_open):
        shaders = mat_data.setdefault("shaders", {})
        vert_path = shaders.get("vertex", "")
        frag_path = shaders.get("fragment", "")
        s_lw = max_label_w(ctx, [t("material.vertex"), t("material.fragment")])
        from .inspector_components import _picker_assets

        def _apply_shader(shader_key, new_value, other_key):
            nonlocal changed, requires_deserialize, requires_pipeline_refresh, change_key
            old_val = shaders.get(shader_key, "")
            shaders[shader_key] = new_value
            changed = True
            change_key = f"shader.{shader_key}"
            requires_deserialize = True
            requires_pipeline_refresh = True
            if new_value != old_val:
                other_id = shaders.get(other_key, "")
                v, f = (new_value, other_id) if shader_key == "vertex" else (other_id, new_value)
                shader_utils.sync_all_shader_properties(mat_data, v, f, remove_unknown=True)
                state.extra["shader_sync_key"] = f"{v}|{f}:{shader_utils.get_shader_property_generation()}"

        # Vertex shader
        field_label(ctx, t("material.vertex"), s_lw)
        vert_items = shader_utils.get_shader_candidates(".vert", _shader_cache)
        vert_display = shader_utils.shader_display_from_value(vert_path, vert_items)

        if _render_obj_field(ctx, "mat_vert", vert_display, "Vert", "SHADER_FILE",
                             lambda p: _on_shader_drop(p, ".vert", shaders),
                             picker_asset_items=lambda filt: _picker_assets(filt, "*.vert"),
                             on_pick=lambda picked: _apply_shader("vertex", picked, "fragment")):
            ctx.open_popup("mat_vert_popup")
        if ctx.begin_popup("mat_vert_popup"):
            for display, value in vert_items:
                if ctx.selectable(display, value == vert_path):
                    _apply_shader("vertex", value, "fragment")
            ctx.end_popup()

        # Fragment shader
        field_label(ctx, t("material.fragment"), s_lw)
        frag_items = shader_utils.get_shader_candidates(".frag", _shader_cache)
        frag_display = shader_utils.shader_display_from_value(frag_path, frag_items)

        if _render_obj_field(ctx, "mat_frag", frag_display, "Frag", "SHADER_FILE",
                             lambda p: _on_shader_drop(p, ".frag", shaders),
                             picker_asset_items=lambda filt: _picker_assets(filt, "*.frag"),
                             on_pick=lambda picked: _apply_shader("fragment", picked, "vertex")):
            ctx.open_popup("mat_frag_popup")
        if ctx.begin_popup("mat_frag_popup"):
            for display, value in frag_items:
                if ctx.selectable(display, value == frag_path):
                    _apply_shader("fragment", value, "vertex")
            ctx.end_popup()
    _record_profile_timing("materialShader", section_t0)
    if is_builtin:
        ctx.end_disabled()
    return changed, requires_deserialize, requires_pipeline_refresh, change_key


def _render_surface_options_section(ctx, mat_data, is_builtin, default_open):
    """Render surface options (cull, depth, blend, alpha clip, render queue).

    Returns ``(changed, requires_deserialize, requires_pipeline_refresh, change_key)``.
    """
    changed = False
    requires_deserialize = False
    requires_pipeline_refresh = False
    change_key = ""

    if is_builtin:
        ctx.begin_disabled(True)
    section_t0 = _time.perf_counter()
    if render_compact_section_header(ctx, t("material.surface_options"), level="secondary",
                                     default_open=default_open):
        rs = mat_data.setdefault("renderState", {})
        overrides = int(mat_data.get("renderStateOverrides", 0))

        so_labels = [t("material.surface_type"), t("material.cull_mode"), t("material.depth_write"),
                     t("material.depth_test"), t("material.blend_mode"), t("material.alpha_clip"),
                     t("material.render_queue")]
        so_lw = max_label_w(ctx, so_labels)

        def _mark(key):
            nonlocal changed, requires_deserialize, requires_pipeline_refresh, change_key
            changed = True
            change_key = key
            requires_deserialize = True
            requires_pipeline_refresh = True

        # --- Surface Type (Opaque / Transparent) ---
        surface_items = [t("material.opaque"), t("material.transparent")]
        cur_surface = 1 if rs.get("blendEnable", False) else 0
        field_label(ctx, t("material.surface_type"), so_lw)
        new_surface = ctx.combo("##mat_surface_type", cur_surface, surface_items)
        if new_surface != cur_surface:
            if new_surface == 1:  # Transparent
                rs["blendEnable"] = True
                rs["srcColorBlendFactor"] = 6
                rs["dstColorBlendFactor"] = 7
                rs["colorBlendOp"] = 0
                rs["srcAlphaBlendFactor"] = 0
                rs["dstAlphaBlendFactor"] = 1
                rs["alphaBlendOp"] = 0
                rs["depthWriteEnable"] = False
                rs["renderQueue"] = 3000
                overrides |= 0x80 | 0x10 | 0x20 | 0x02 | 0x40
            else:  # Opaque
                rs["blendEnable"] = False
                rs["depthWriteEnable"] = True
                rs["renderQueue"] = 2000
                overrides |= 0x80 | 0x10 | 0x02 | 0x40
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.surface_type")

        # --- Cull Mode ---
        cull_items = [t("material.cull_none"), t("material.cull_front"), t("material.cull_back")]
        cull_val = int(rs.get("cullMode", 2))
        cull_idx = {0: 0, 1: 1, 2: 2}.get(cull_val, 2)
        field_label(ctx, t("material.cull_mode"), so_lw)
        new_cull_idx = ctx.combo("##mat_cull_mode", cull_idx, cull_items)
        if new_cull_idx != cull_idx:
            rs["cullMode"] = new_cull_idx
            overrides |= 0x01
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.cull_mode")

        # --- Depth Write ---
        dw_val = rs.get("depthWriteEnable", True)
        field_label(ctx, t("material.depth_write"), so_lw)
        new_dw = ctx.checkbox("##mat_depth_write", dw_val)
        if new_dw != dw_val:
            rs["depthWriteEnable"] = new_dw
            overrides |= 0x02
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.depth_write")

        # --- Depth Test ---
        compare_items = [t("material.compare_never"), t("material.compare_less"), t("material.compare_equal"), t("material.compare_less_equal"),
                         t("material.compare_greater"), t("material.compare_not_equal"), t("material.compare_greater_equal"), t("material.compare_always")]
        dt_enable = rs.get("depthTestEnable", True)
        dt_op = int(rs.get("depthCompareOp", 1))
        field_label(ctx, t("material.depth_test"), so_lw)
        if dt_enable:
            new_op = ctx.combo("##mat_depth_test", dt_op, compare_items)
        else:
            new_op = ctx.combo("##mat_depth_test", 7, ["Off"] + compare_items[1:])
            new_op = 0 if new_op == 0 else new_op
        if not dt_enable and new_op > 0:
            rs["depthTestEnable"] = True
            rs["depthCompareOp"] = new_op
            overrides |= 0x04 | 0x08
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.depth_test")
        elif dt_enable and new_op != dt_op:
            rs["depthCompareOp"] = new_op
            overrides |= 0x08
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.depth_test")

        # --- Blend Mode (only visible when transparent) ---
        if rs.get("blendEnable", False):
            blend_items = [t("material.blend_alpha"), t("material.blend_additive"), t("material.blend_premultiply")]
            src = int(rs.get("srcColorBlendFactor", 6))
            dst = int(rs.get("dstColorBlendFactor", 7))
            if src == 1 and dst == 1:
                cur_blend_idx = 1
            elif src == 1 and dst == 7:
                cur_blend_idx = 2
            else:
                cur_blend_idx = 0
            field_label(ctx, t("material.blend_mode"), so_lw)
            new_blend_idx = ctx.combo("##mat_blend_mode", cur_blend_idx, blend_items)
            if new_blend_idx != cur_blend_idx:
                _BLEND = {0: (6, 7), 1: (1, 1), 2: (1, 7)}
                rs["srcColorBlendFactor"], rs["dstColorBlendFactor"] = _BLEND[new_blend_idx]
                rs["colorBlendOp"] = 0
                overrides |= 0x20
                mat_data["renderStateOverrides"] = overrides
                _mark("render_state.blend_mode")

        # --- Alpha Clip ---
        ac_enabled = rs.get("alphaClipEnabled", False)
        ac_threshold = float(rs.get("alphaClipThreshold", 0.5))
        field_label(ctx, t("material.alpha_clip"), so_lw)
        new_ac = ctx.checkbox("##mat_alpha_clip", ac_enabled)
        if new_ac != ac_enabled:
            rs["alphaClipEnabled"] = new_ac
            if new_ac and "alphaClipThreshold" not in rs:
                rs["alphaClipThreshold"] = 0.5
            overrides |= 0x100
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.alpha_clip")
        if rs.get("alphaClipEnabled", False):
            field_label(ctx, t("material.threshold"), so_lw)
            new_threshold = ctx.float_slider("##mat_alpha_threshold", ac_threshold, 0.0, 1.0)
            if abs(new_threshold - ac_threshold) > 1e-5:
                rs["alphaClipThreshold"] = new_threshold
                overrides |= 0x100
                mat_data["renderStateOverrides"] = overrides
                _mark("render_state.alpha_threshold")

        # --- Render Queue ---
        is_transparent = rs.get("blendEnable", False)
        rq_min, rq_max = (2501, 5000) if is_transparent else (0, 2500)
        rq = int(rs.get("renderQueue", 2000))
        rq = max(rq_min, min(rq, rq_max))
        field_label(ctx, t("material.render_queue"), so_lw)
        new_rq = int(ctx.drag_int("##mat_render_queue", rq, 1.0, rq_min, rq_max))
        if new_rq != rq:
            rs["renderQueue"] = new_rq
            overrides |= 0x40
            mat_data["renderStateOverrides"] = overrides
            _mark("render_state.render_queue")

    _record_profile_timing("materialSurface", section_t0)
    if is_builtin:
        ctx.end_disabled()
    return changed, requires_deserialize, requires_pipeline_refresh, change_key


def _render_properties_section(ctx, mat_data, is_builtin, default_open):
    """Render material shader properties.

    Returns ``(changed, change_key, requires_deserialize)``.
    """
    changed = False
    change_key = ""
    requires_deserialize = False

    if is_builtin:
        ctx.begin_disabled(True)
    section_t0 = _time.perf_counter()
    if render_compact_section_header(ctx, t("material.properties_section"), level="secondary",
                                     default_open=default_open):
        props = mat_data.get("properties", {})
        if not props:
            ctx.label(t("material.no_properties"))
        else:
            prop_names = shader_utils.get_material_property_display_order(mat_data)
            plw = max_label_w(ctx, prop_names)
            for prop_name in prop_names:
                prop = props[prop_name]
                ptype = int(prop.get("type", 0))
                value = prop.get("value")
                prop_changed = render_material_property(
                    ctx, prop_name, prop, ptype, value, plw,
                )
                if prop_changed:
                    if ptype == 6:
                        _apply_native_prop(prop_name, prop.get("guid", ""), ptype)
                    else:
                        _apply_native_prop(prop_name, prop["value"], ptype)
                    changed = True
                    change_key = f"property.{prop_name}"
                    if ptype == 6:
                        requires_deserialize = True
    _record_profile_timing("materialProperties", section_t0)
    if is_builtin:
        ctx.end_disabled()
    return changed, change_key, requires_deserialize


def _apply_material_changes(panel, state, mat_data, native_mat,
                            requires_pipeline_refresh, old_json, change_key, exec_layer):
    """Serialize and save material changes, record undo."""
    try:
        native_mat.deserialize(json.dumps(mat_data))
        if requires_pipeline_refresh:
            _refresh_pipeline(panel)
        _ensure_material_file_path(panel, native_mat)
        if exec_layer:
            exec_layer.schedule_rw_save(native_mat)
        new_json = json.dumps(mat_data)
        state.extra["cached_json"] = new_json
        from Infernux.engine.undo import UndoManager, MaterialJsonCommand
        mgr = UndoManager.instance()
        if mgr and not mgr.is_executing and mgr.enabled and old_json:
            if new_json != old_json:
                mgr.record(MaterialJsonCommand(
                    native_mat,
                    old_json,
                    new_json,
                    "Edit Material",
                    refresh_callback=lambda _mat: _refresh_pipeline(panel),
                    edit_key=change_key,
                ))
    except (RuntimeError, ValueError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Body renderer (called from asset_inspector)
# ═══════════════════════════════════════════════════════════════════════════


def render_material_body(ctx: InxGUIContext, panel, state):
    """Render the material-specific inspector body.

    *state* is the ``_State`` object from ``asset_inspector``.  Relevant
    fields: ``state.settings`` (Material wrapper), ``state.extra``
    (native_mat, cached_data, shader_cache), ``state.exec_layer``.
    """
    global _native_mat, _cached_data, _shader_cache

    _native_mat = state.extra["native_mat"]
    _cached_data = state.extra["cached_data"]
    _shader_cache = state.extra["shader_cache"]
    exec_layer = state.exec_layer
    old_json = state.extra.get("cached_json", "")

    mat_data = _cached_data
    is_builtin = bool(getattr(_native_mat, "is_builtin", False) or mat_data.get("builtin", False))
    if is_builtin:
        mat_data["builtin"] = True

    default_open_sections = bool(state.extra.get("default_open_sections", True))

    if is_builtin:
        ctx.label(t("material.builtin_locked"))

    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)

    changed = False
    requires_deserialize = False
    requires_pipeline_refresh = False
    change_key = ""

    # Sync shader annotations (both vertex + fragment properties)
    vert_shader_id = mat_data.get("shaders", {}).get("vertex", "")
    frag_shader_id = mat_data.get("shaders", {}).get("fragment", "")
    prop_gen = shader_utils.get_shader_property_generation()
    sync_key = f"{vert_shader_id}|{frag_shader_id}:{prop_gen}"
    last_sync_key = state.extra.get("shader_sync_key", "")
    last_validation_key = state.extra.get("shader_validation_key", "")
    mat_version = state.extra.get("mat_version", -1)
    last_validation_version = state.extra.get("shader_validation_mat_version", -2)
    needs_validation = (
        sync_key != last_validation_key
        or not mat_data.get("_shader_property_order")
        or (mat_version != -1 and mat_version != last_validation_version)
    )
    missing_shader_props = False
    if needs_validation and (vert_shader_id or frag_shader_id):
        current_props = mat_data.get("properties", {})
        expected_shader_props = shader_utils.get_all_shader_property_names(vert_shader_id, frag_shader_id)
        missing_shader_props = any(name not in current_props for name in expected_shader_props)
        state.extra["shader_validation_key"] = sync_key
        if mat_version != -1:
            state.extra["shader_validation_mat_version"] = mat_version

    if (vert_shader_id or frag_shader_id) and (sync_key != last_sync_key or missing_shader_props):
        old_key = last_sync_key.rsplit(":", 1)[0] if last_sync_key else ""
        # On hot-reload (same shaders, new generation) remove stale properties
        remove = (f"{vert_shader_id}|{frag_shader_id}" == old_key) and bool(old_key)
        state.extra["shader_sync_key"] = sync_key
        if vert_shader_id or frag_shader_id:
            old_prop_names = set(mat_data.get("properties", {}).keys())
            shader_utils.sync_all_shader_properties(mat_data, vert_shader_id, frag_shader_id,
                                                    remove_unknown=remove)
            new_prop_names = set(mat_data.get("properties", {}).keys())
            if new_prop_names != old_prop_names:
                # Sync added/removed properties — push to native material so
                # the C++ UBO gets the correct default values and so the merge
                # in _refresh_material preserves them on subsequent frames.
                changed = True
                requires_deserialize = True

    # ── Shader Section ─────────────────────────────────────────────────
    s_ch, s_ds, s_pr, s_ck = _render_shader_section(ctx, mat_data, state, is_builtin, default_open_sections)
    changed |= s_ch
    requires_deserialize |= s_ds
    requires_pipeline_refresh |= s_pr
    if s_ck:
        change_key = s_ck

    ctx.separator()

    # ── Surface Options (Render Settings) ──────────────────────────────
    so_ch, so_ds, so_pr, so_ck = _render_surface_options_section(ctx, mat_data, is_builtin, default_open_sections)
    changed |= so_ch
    requires_deserialize |= so_ds
    requires_pipeline_refresh |= so_pr
    if so_ck:
        change_key = so_ck

    ctx.separator()

    # ── Properties ─────────────────────────────────────────────────────
    p_ch, p_ck, p_ds = _render_properties_section(ctx, mat_data, is_builtin, default_open_sections)
    changed |= p_ch
    requires_deserialize |= p_ds
    if p_ck:
        change_key = p_ck

    ctx.separator()

    # ── Auto-save on change ─────────────────────────────────────────────
    if changed:
        _apply_material_changes(panel, state, mat_data, _native_mat,
                                requires_pipeline_refresh, old_json, change_key, exec_layer)

    ctx.pop_style_var(2)


def render_inline_material_body(ctx: InxGUIContext, panel, native_mat, cache_key: str | None = None) -> None:
    """Render a MeshRenderer-linked material using the shared material inspector."""
    if native_mat is None:
        return
    inline_t0 = _time.perf_counter()
    state = _build_inline_state(panel, native_mat)
    ctx.push_id_str(cache_key or f"inline_material_{id(native_mat)}")
    try:
        render_material_body(ctx, panel, state)
    finally:
        ctx.pop_id()
        _record_profile_timing("materialInline", inline_t0)

    # Inline materials don't go through render_asset_inspector's footer,
    # so the debounced save would never be flushed.  Drive it here.
    _inline_layer = getattr(panel, "_inline_material_exec_layer", None)
    if _inline_layer is not None:
        _inline_layer.flush_rw_autosave()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _on_shader_drop(path: str, required_ext: str, shaders_dict: dict):
    if path.lower().endswith(required_ext):
        key = "vertex" if required_ext == ".vert" else "fragment"
        old = shaders_dict.get(key, "")
        shaders_dict[key] = path
        if path != old and _cached_data:
            vert_id = shaders_dict.get("vertex", "")
            frag_id = shaders_dict.get("fragment", "")
            shader_utils.sync_all_shader_properties(_cached_data, vert_id, frag_id, remove_unknown=True)


def _render_obj_field(ctx: InxGUIContext, fid: str, display: str, type_hint: str,
                      drag_type: str, on_drop,
                      picker_asset_items=None, on_pick=None, on_clear=None) -> bool:
    """Simplified object-field renderer accepting drag-drop."""
    from . import inspector_components as comp_ui
    return comp_ui.render_object_field(ctx, fid, display, type_hint,
                                       clickable=True,
                                       accept_drag_type=drag_type,
                                       on_drop_callback=on_drop,
                                       picker_asset_items=picker_asset_items,
                                       on_pick=on_pick,
                                       on_clear=on_clear)


def _apply_native_prop(prop_name: str, value, ptype: int):
    """Forward a property change to the C++ material."""
    if not _native_mat:
        return
    if ptype == 0:
        _native_mat.set_float(prop_name, float(value))
    elif ptype == 1:
        _native_mat.set_vector2(prop_name, (float(value[0]), float(value[1])))
    elif ptype == 2:
        _native_mat.set_vector3(prop_name, (float(value[0]), float(value[1]), float(value[2])))
    elif ptype == 3:
        _native_mat.set_vector4(prop_name, (float(value[0]), float(value[1]), float(value[2]), float(value[3])))
    elif ptype == 4:
        _native_mat.set_int(prop_name, int(value))
    elif ptype == 5:
        _native_mat.set_matrix(prop_name, [float(v) for v in value])
    elif ptype == 6:
        _native_mat.set_texture_guid(prop_name, str(value))
    elif ptype == 7:
        _native_mat.set_color(prop_name, (float(value[0]), float(value[1]), float(value[2]), float(value[3])))


class _InlineMaterialExecLayer:
    """Lightweight adapter so inline material editing uses the same autosave path."""

    def __init__(self, panel):
        self._panel = panel

    def schedule_rw_save(self, resource_obj):
        file_path = _ensure_material_file_path(self._panel, resource_obj)
        if not file_path:
            return
        current_layer = getattr(self._panel, "_inline_material_exec_layer", None)
        layer = get_asset_execution_layer(
            current_layer,
            "material",
            file_path,
            AssetAccessMode.READ_WRITE_RESOURCE,
            autosave_debounce_sec=0.35,
        )
        self._panel._inline_material_exec_layer = layer
        layer.schedule_rw_save(resource_obj)


def _build_inline_state(panel, native_mat):
    extra = _get_inline_material_extra(panel, native_mat)
    return SimpleNamespace(extra=extra, exec_layer=_InlineMaterialExecLayer(panel))


def _get_inline_material_extra(panel, native_mat) -> dict:
    cache = getattr(panel, "_inline_material_cache", None)
    if cache is None:
        cache = {}
        panel._inline_material_cache = cache

    mat_id = id(native_mat)
    try:
        mat_version = native_mat.get_version()
    except (AttributeError, RuntimeError):
        mat_version = -1

    extra = cache.get(mat_id)
    cache_hit = extra is not None and mat_version != -1 and extra.get("mat_version", -1) == mat_version
    if cache_hit:
        return extra

    try:
        current_json = native_mat.serialize()
    except RuntimeError:
        current_json = ""
    try:
        fresh = json.loads(current_json) if current_json else {}
    except (ValueError, json.JSONDecodeError):
        fresh = {}

    old_data = extra.get("cached_data", {}) if isinstance(extra, dict) else {}
    if isinstance(old_data, dict) and fresh:
        if "_shader_property_order" in old_data and "_shader_property_order" not in fresh:
            fresh["_shader_property_order"] = old_data["_shader_property_order"]
        old_props = old_data.get("properties") if isinstance(old_data.get("properties"), dict) else {}
        new_props = fresh.get("properties") if isinstance(fresh.get("properties"), dict) else {}
        for prop_name, fresh_prop in new_props.items():
            if isinstance(fresh_prop, dict) and prop_name in old_props and isinstance(old_props[prop_name], dict):
                for meta_key, meta_value in old_props[prop_name].items():
                    if meta_key not in ("value", "guid"):
                        fresh_prop[meta_key] = meta_value
        shader_order = old_data.get("_shader_property_order", [])
        for prop_name in set(shader_order) if shader_order else set():
            if prop_name not in new_props and prop_name in old_props:
                new_props[prop_name] = old_props[prop_name]
        fresh["properties"] = new_props

    shader_cache = extra.get("shader_cache") if isinstance(extra, dict) else None
    if not isinstance(shader_cache, dict):
        shader_cache = {".vert": None, ".frag": None}

    extra = {
        "native_mat": native_mat,
        "cached_data": fresh,
        "cached_json": current_json or json.dumps(fresh),
        "shader_cache": shader_cache,
        "shader_sync_key": extra.get("shader_sync_key", "") if isinstance(extra, dict) else "",
        "mat_version": mat_version,
        "mat_ref": native_mat,
        "default_open_sections": True,
    }
    cache[mat_id] = extra
    return extra


def _refresh_pipeline(panel):
    """Ask the engine to rebuild the material pipeline."""
    engine = None
    if panel and hasattr(panel, '_get_native_engine'):
        engine = panel._get_native_engine()
    if engine is None:
        try:
            from Infernux.engine.ui.editor_services import EditorServices
            svc = EditorServices.instance()
            if svc:
                engine = svc.native_engine
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    if engine and _native_mat and hasattr(engine, 'refresh_material_pipeline'):
        engine.refresh_material_pipeline(_native_mat)


def _ensure_material_file_path(panel, native_mat) -> str:
    """Ensure a native material has a stable file path for autosave/undo replay."""
    if not native_mat:
        return ""
    file_path = getattr(native_mat, "file_path", "") or ""
    if file_path:
        return file_path
    if panel and hasattr(panel, "_ensure_material_file_path"):
        return panel._ensure_material_file_path(native_mat)
    return ""


