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
from typing import Optional

from InfEngine.lib import InfGUIContext
from InfEngine.engine.i18n import t
from .inspector_utils import (
    max_label_w,
    field_label,
    render_compact_section_header,
    _render_color_bar,
    LABEL_PAD,
)
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from . import inspector_shader_utils as shader_utils


# ═══════════════════════════════════════════════════════════════════════════
#  Material property renderer (JSON type system: ptype 0-7)
# ═══════════════════════════════════════════════════════════════════════════

def render_material_property(
    ctx: InfGUIContext,
    prop_name: str,
    prop: dict,
    ptype: int,
    value,
    plw: float,
    wid_prefix: str = "mp",
) -> bool:
    """Render one material property row.  Returns ``True`` if changed."""
    import os
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

    elif ptype == 6:  # Texture2D — GUID-only (v2)
        tex_guid = prop.get("guid", "")
        if not isinstance(tex_guid, str):
            tex_guid = ""
        has_texture = bool(tex_guid)
        display = t("igui.none")
        if has_texture:
            try:
                from .editor_services import EditorServices
                adb = EditorServices.instance()._asset_database
                if adb:
                    tex_path = adb.get_path_from_guid(tex_guid)
                    if tex_path:
                        display = os.path.basename(tex_path)
                    else:
                        display = tex_guid[:8] + "..."
                else:
                    display = tex_guid[:8] + "..."
            except Exception:
                display = tex_guid[:8] + "..."
        field_label(ctx, prop_name, plw)

        from .igui import IGUI

        def _get_asset_database():
            try:
                from .editor_services import EditorServices
                adb = EditorServices.instance()._asset_database
                if adb:
                    return adb
            except Exception:
                pass
            try:
                from InfEngine.lib import AssetRegistry
                return AssetRegistry.instance().get_asset_database()
            except Exception:
                return None

        def _resolve_path_to_guid(path_str):
            adb = _get_asset_database()
            if adb:
                return adb.get_guid_from_path(path_str) or ""
            return ""

        def _on_tex_drop(payload):
            nonlocal changed
            dropped = str(payload).replace("\\", "/")
            guid = _resolve_path_to_guid(dropped)
            if guid:
                prop["guid"] = guid
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Cannot resolve dropped texture path to GUID: %s", dropped)
                return
            changed = True

        def _on_tex_pick(picked_path):
            nonlocal changed
            picked = str(picked_path).replace("\\", "/")
            guid = _resolve_path_to_guid(picked)
            if guid:
                prop["guid"] = guid
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Cannot resolve picked texture path to GUID: %s", picked)
                return
            changed = True

        def _on_tex_clear():
            nonlocal changed
            prop["guid"] = ""
            changed = True

        def _tex_asset_items(filt):
            from .inspector_components import _picker_assets
            return _picker_assets(filt, "*.png") + _picker_assets(filt, "*.jpg")

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
# Body renderer (called from asset_inspector)
# ═══════════════════════════════════════════════════════════════════════════


def render_material_body(ctx: InfGUIContext, panel, state):
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
    try:
        old_json = _native_mat.serialize()
    except RuntimeError:
        old_json = ""

    mat_data = _cached_data
    is_builtin = mat_data.get("builtin", False)

    if is_builtin:
        ctx.label(t("material.builtin_locked"))

    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)

    changed = False
    requires_deserialize = False
    requires_pipeline_refresh = False

    # Sync shader annotations
    frag_shader_id = mat_data.get("shaders", {}).get("fragment", "")
    if frag_shader_id and not mat_data.get("_shader_property_order"):
        state.extra["shader_sync_key"] = ""
    prop_gen = shader_utils.get_shader_property_generation()
    sync_key = f"{frag_shader_id}:{prop_gen}"
    last_sync_key = state.extra.get("shader_sync_key", "")
    if sync_key != last_sync_key:
        old_id = last_sync_key.rsplit(":", 1)[0] if last_sync_key else ""
        # On hot-reload (same shader, new generation) remove stale properties
        remove = (frag_shader_id == old_id) and bool(old_id)
        state.extra["shader_sync_key"] = sync_key
        if frag_shader_id:
            shader_utils.sync_properties_from_shader(mat_data, frag_shader_id, ".frag", remove_unknown=remove)

    # ── Shader Section ─────────────────────────────────────────────────
    if is_builtin:
        ctx.begin_disabled(True)
    if render_compact_section_header(ctx, t("material.shader_section"), level="secondary"):
        shaders = mat_data.setdefault("shaders", {})
        vert_path = shaders.get("vertex", "")
        frag_path = shaders.get("fragment", "")
        s_lw = max_label_w(ctx, [t("material.vertex"), t("material.fragment")])
        from .inspector_components import _picker_assets

        # Vertex shader
        field_label(ctx, t("material.vertex"), s_lw)
        vert_items = shader_utils.get_shader_candidates(".vert", _shader_cache)
        vert_display = shader_utils.shader_display_from_value(vert_path, vert_items)

        def _on_vert_pick(picked):
            nonlocal changed, requires_deserialize, requires_pipeline_refresh
            shaders["vertex"] = picked
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

        if _render_obj_field(ctx, "mat_vert", vert_display, "Vert", "SHADER_FILE",
                             lambda p: _on_shader_drop(p, ".vert", shaders),
                             picker_asset_items=lambda filt: _picker_assets(filt, "*.vert"),
                             on_pick=_on_vert_pick):
            ctx.open_popup("mat_vert_popup")
        if ctx.begin_popup("mat_vert_popup"):
            for display, value in vert_items:
                if ctx.selectable(display, value == vert_path):
                    shaders["vertex"] = value
                    changed = True
                    requires_deserialize = True
                    requires_pipeline_refresh = True
            ctx.end_popup()

        # Fragment shader
        field_label(ctx, t("material.fragment"), s_lw)
        frag_items = shader_utils.get_shader_candidates(".frag", _shader_cache)
        frag_display = shader_utils.shader_display_from_value(frag_path, frag_items)

        def _on_frag_pick(picked):
            nonlocal changed, requires_deserialize, requires_pipeline_refresh
            old_frag = shaders.get("fragment", "")
            shaders["fragment"] = picked
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True
            if picked != old_frag:
                shader_utils.sync_properties_from_shader(mat_data, picked, ".frag", remove_unknown=True)
                state.extra["shader_sync_key"] = f"{picked}:{shader_utils.get_shader_property_generation()}"

        if _render_obj_field(ctx, "mat_frag", frag_display, "Frag", "SHADER_FILE",
                             lambda p: _on_shader_drop(p, ".frag", shaders),
                             picker_asset_items=lambda filt: _picker_assets(filt, "*.frag"),
                             on_pick=_on_frag_pick):
            ctx.open_popup("mat_frag_popup")
        if ctx.begin_popup("mat_frag_popup"):
            for display, value in frag_items:
                if ctx.selectable(display, value == frag_path):
                    old_frag = shaders.get("fragment", "")
                    shaders["fragment"] = value
                    changed = True
                    requires_deserialize = True
                    requires_pipeline_refresh = True
                    if value != old_frag:
                        shader_utils.sync_properties_from_shader(mat_data, value, ".frag", remove_unknown=True)
                        state.extra["shader_sync_key"] = f"{value}:{shader_utils.get_shader_property_generation()}"
            ctx.end_popup()
    if is_builtin:
        ctx.end_disabled()

    ctx.separator()

    # ── Surface Options (Render Settings) ──────────────────────────────
    if is_builtin:
        ctx.begin_disabled(True)
    if render_compact_section_header(ctx, t("material.surface_options"), level="secondary"):
        rs = mat_data.setdefault("renderState", {})
        overrides = int(mat_data.get("renderStateOverrides", 0))

        so_labels = [t("material.surface_type"), t("material.cull_mode"), t("material.depth_write"),
                     t("material.depth_test"), t("material.blend_mode"), t("material.alpha_clip"),
                     t("material.render_queue")]
        so_lw = max_label_w(ctx, so_labels)

        # --- Surface Type (Opaque / Transparent) ---
        surface_items = [t("material.opaque"), t("material.transparent")]
        cur_surface = 1 if rs.get("blendEnable", False) else 0
        field_label(ctx, t("material.surface_type"), so_lw)
        new_surface = ctx.combo("##mat_surface_type", cur_surface, surface_items)
        if new_surface != cur_surface:
            if new_surface == 1:  # Transparent
                rs["blendEnable"] = True
                rs["srcColorBlendFactor"] = 6   # SRC_ALPHA
                rs["dstColorBlendFactor"] = 7   # ONE_MINUS_SRC_ALPHA
                rs["colorBlendOp"] = 0          # ADD
                rs["srcAlphaBlendFactor"] = 0   # ZERO  (preserve dst alpha)
                rs["dstAlphaBlendFactor"] = 1   # ONE
                rs["alphaBlendOp"] = 0          # ADD
                rs["depthWriteEnable"] = False
                rs["renderQueue"] = 3000
                overrides |= 0x80   # SurfaceType
                overrides |= 0x10   # BlendEnable
                overrides |= 0x20   # BlendMode
                overrides |= 0x02   # DepthWrite
                overrides |= 0x40   # RenderQueue
            else:  # Opaque
                rs["blendEnable"] = False
                rs["depthWriteEnable"] = True
                rs["renderQueue"] = 2000
                overrides |= 0x80   # SurfaceType
                overrides |= 0x10   # BlendEnable
                overrides |= 0x02   # DepthWrite
                overrides |= 0x40   # RenderQueue
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

        # --- Cull Mode ---
        cull_items = [t("material.cull_none"), t("material.cull_front"), t("material.cull_back")]
        cull_map = {0: 0, 1: 1, 2: 2}     # VkCullModeFlags: 0=None, 1=Front, 2=Back
        cull_val = int(rs.get("cullMode", 2))
        cull_idx = {0: 0, 1: 1, 2: 2}.get(cull_val, 2)
        field_label(ctx, t("material.cull_mode"), so_lw)
        new_cull_idx = ctx.combo("##mat_cull_mode", cull_idx, cull_items)
        if new_cull_idx != cull_idx:
            rs["cullMode"] = cull_map[new_cull_idx]
            overrides |= 0x01  # CullMode
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

        # --- Depth Write ---
        dw_val = rs.get("depthWriteEnable", True)
        field_label(ctx, t("material.depth_write"), so_lw)
        new_dw = ctx.checkbox("##mat_depth_write", dw_val)
        if new_dw != dw_val:
            rs["depthWriteEnable"] = new_dw
            overrides |= 0x02  # DepthWrite
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

        # --- Depth Test ---
        compare_items = [t("material.compare_never"), t("material.compare_less"), t("material.compare_equal"), t("material.compare_less_equal"),
                         t("material.compare_greater"), t("material.compare_not_equal"), t("material.compare_greater_equal"), t("material.compare_always")]
        dt_enable = rs.get("depthTestEnable", True)
        dt_op = int(rs.get("depthCompareOp", 1))  # VkCompareOp: default Less=1
        field_label(ctx, t("material.depth_test"), so_lw)
        if dt_enable:
            new_op = ctx.combo("##mat_depth_test", dt_op, compare_items)
        else:
            new_op = ctx.combo("##mat_depth_test", 7, ["Off"] + compare_items[1:])
            new_op = 0 if new_op == 0 else new_op  # map "Off" to Never
        # Allow toggling depth test off via selecting index 0 when "Never" is chosen
        if not dt_enable and new_op > 0:
            rs["depthTestEnable"] = True
            rs["depthCompareOp"] = new_op
            overrides |= 0x04  # DepthTest
            overrides |= 0x08  # DepthCompareOp
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True
        elif dt_enable and new_op != dt_op:
            rs["depthCompareOp"] = new_op
            overrides |= 0x08  # DepthCompareOp
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

        # --- Blend Mode (only visible when transparent) ---
        if rs.get("blendEnable", False):
            blend_items = [t("material.blend_alpha"), t("material.blend_additive"), t("material.blend_premultiply")]
            # Detect current blend mode from factors
            src = int(rs.get("srcColorBlendFactor", 6))
            dst = int(rs.get("dstColorBlendFactor", 7))
            if src == 1 and dst == 1:       # ONE, ONE
                cur_blend_idx = 1  # Additive
            elif src == 1 and dst == 7:     # ONE, ONE_MINUS_SRC_ALPHA
                cur_blend_idx = 2  # Premultiply
            else:
                cur_blend_idx = 0  # Alpha (default)
            field_label(ctx, t("material.blend_mode"), so_lw)
            new_blend_idx = ctx.combo("##mat_blend_mode", cur_blend_idx, blend_items)
            if new_blend_idx != cur_blend_idx:
                if new_blend_idx == 0:      # Alpha
                    rs["srcColorBlendFactor"] = 6   # SRC_ALPHA
                    rs["dstColorBlendFactor"] = 7   # ONE_MINUS_SRC_ALPHA
                elif new_blend_idx == 1:    # Additive
                    rs["srcColorBlendFactor"] = 1   # ONE
                    rs["dstColorBlendFactor"] = 1   # ONE
                elif new_blend_idx == 2:    # Premultiply
                    rs["srcColorBlendFactor"] = 1   # ONE
                    rs["dstColorBlendFactor"] = 7   # ONE_MINUS_SRC_ALPHA
                rs["colorBlendOp"] = 0  # ADD
                overrides |= 0x20  # BlendMode
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

        # --- Alpha Clip ---
        ac_enabled = rs.get("alphaClipEnabled", False)
        ac_threshold = float(rs.get("alphaClipThreshold", 0.5))
        field_label(ctx, t("material.alpha_clip"), so_lw)
        new_ac = ctx.checkbox("##mat_alpha_clip", ac_enabled)
        if new_ac != ac_enabled:
            rs["alphaClipEnabled"] = new_ac
            if new_ac and "alphaClipThreshold" not in rs:
                rs["alphaClipThreshold"] = 0.5
            overrides |= 0x100  # AlphaClip
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True
        if rs.get("alphaClipEnabled", False):
            field_label(ctx, t("material.threshold"), so_lw)
            new_threshold = ctx.float_slider("##mat_alpha_threshold", ac_threshold, 0.0, 1.0)
            if abs(new_threshold - ac_threshold) > 1e-5:
                rs["alphaClipThreshold"] = new_threshold
                overrides |= 0x100  # AlphaClip
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

        # --- Render Queue (clamped by surface type) ---
        is_transparent = rs.get("blendEnable", False)
        rq_min, rq_max = (2501, 5000) if is_transparent else (0, 2500)
        rq = int(rs.get("renderQueue", 2000))
        rq = max(rq_min, min(rq, rq_max))  # clamp display value
        field_label(ctx, t("material.render_queue"), so_lw)
        new_rq = int(ctx.drag_int("##mat_render_queue", rq, 1.0, rq_min, rq_max))
        if new_rq != rq:
            rs["renderQueue"] = new_rq
            overrides |= 0x40  # RenderQueue
            mat_data["renderStateOverrides"] = overrides
            changed = True
            requires_deserialize = True
            requires_pipeline_refresh = True

    if is_builtin:
        ctx.end_disabled()

    ctx.separator()

    # ── Properties ─────────────────────────────────────────────────────
    if render_compact_section_header(ctx, t("material.properties_section"), level="secondary"):
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
                    if ptype == 6:  # Texture needs full deserialize
                        requires_deserialize = True

    ctx.separator()

    # ── Auto-save on change ─────────────────────────────────────────────
    if changed:
        try:
            if requires_deserialize:
                _native_mat.deserialize(json.dumps(mat_data))
            if requires_pipeline_refresh:
                _refresh_pipeline(panel)
            _ensure_material_file_path(panel, _native_mat)
            if exec_layer:
                exec_layer.schedule_rw_save(_native_mat)
            from InfEngine.engine.undo import UndoManager, MaterialJsonCommand
            mgr = UndoManager.instance()
            if mgr and not mgr.is_executing and mgr.enabled and old_json:
                new_json = _native_mat.serialize()
                if new_json != old_json:
                    mgr.record(MaterialJsonCommand(
                        _native_mat,
                        old_json,
                        new_json,
                        "Edit Material",
                        refresh_callback=lambda _mat: _refresh_pipeline(panel),
                    ))
        except (RuntimeError, ValueError):
            pass

    ctx.pop_style_var(2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _on_shader_drop(path: str, required_ext: str, shaders_dict: dict):
    if path.lower().endswith(required_ext):
        key = "vertex" if required_ext == ".vert" else "fragment"
        old = shaders_dict.get(key, "")
        shaders_dict[key] = path
        if key == "fragment" and path != old and _cached_data:
            shader_utils.sync_properties_from_shader(_cached_data, path, ".frag", remove_unknown=True)


def _render_obj_field(ctx: InfGUIContext, fid: str, display: str, type_hint: str,
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


def _refresh_pipeline(panel):
    """Ask the engine to rebuild the material pipeline."""
    engine = panel._get_native_engine() if panel else None
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


