"""
Unified Asset Inspector — data-driven inspector for all asset types.

One loader, one state machine, one renderer entry point.  Categories register
via ``AssetCategoryDef``, each specifying how to load data, which editable
fields to expose, and optional custom sections (preview, shader editing, etc.).

Read-only assets (texture, audio, shader) share an Apply / Revert bar.
Read-write assets (material) use automatic debounced save.

Public API::

    render_asset_inspector(ctx, panel, file_path, category)
    invalidate()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from InfEngine.lib import InfGUIContext
from InfEngine.engine.i18n import t
from InfEngine.core.asset_types import (
    TextureType,
    ShaderAssetInfo,
    FontAssetInfo,
    read_meta_file,
    read_texture_import_settings,
    read_audio_import_settings,
    read_mesh_import_settings,
)
from .inspector_utils import max_label_w, field_label, render_apply_revert
from .theme import Theme, ImGuiCol
from .asset_execution_layer import AssetAccessMode, get_asset_execution_layer


# ═══════════════════════════════════════════════════════════════════════════
# Field descriptor
# ═══════════════════════════════════════════════════════════════════════════


class WidgetType(Enum):
    """Widget type for an editable import-settings field."""
    CHECKBOX = "checkbox"
    COMBO = "combo"
    FLOAT = "float"


@dataclass
class FieldDef:
    """Describes one editable field on an import-settings object.

    * *key* — attribute name on the settings dataclass.
    * *label* — display text in the Inspector.
    * *field_type* — which ImGui widget to render.
    * *combo_entries* — ``[(display_label, value), ...]`` for COMBO fields.
    * *float_speed* — drag speed for FLOAT fields (default 0.001).
    * *float_range* — ``(min, max)`` clamp for FLOAT fields (default None = unclamped).
    """
    key: str
    label: str
    field_type: WidgetType
    combo_entries: List[Tuple[str, Any]] = field(default_factory=list)
    float_speed: float = 0.001
    float_range: Optional[Tuple[float, float]] = None


# ═══════════════════════════════════════════════════════════════════════════
# Category definition
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AssetCategoryDef:
    """Registration for one asset category.

    * *load_fn* returns ``(settings_obj, extra_dict)`` or ``None`` on failure.
      For read-only assets the settings object must implement ``.copy()``
      and ``__eq__`` for dirty tracking.
    * *refresh_fn* is called every frame when the asset is already loaded
      (e.g. material re-serializes native data).
    * *custom_header_fn(ctx, panel, state)* renders after the standard header
      (e.g. texture preview).
    * *custom_body_fn(ctx, panel, state)* replaces the auto-generated
      import-settings field list (e.g. material properties, shader path editing).
    """
    display_name: str
    access_mode: AssetAccessMode
    load_fn: Callable[[str], Optional[Tuple[Any, dict]]]
    refresh_fn: Optional[Callable] = None
    editable_fields: List[FieldDef] = field(default_factory=list)
    extra_meta_keys: List[str] = field(default_factory=list)
    custom_header_fn: Optional[Callable] = None
    custom_body_fn: Optional[Callable] = None
    autosave_debounce: float = 0.35
    show_header: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Unified state
# ═══════════════════════════════════════════════════════════════════════════


class _State:
    """Per-asset inspector state (only one asset is inspected at a time)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.file_path: str = ""
        self.category: str = ""
        self.meta: Optional[dict] = None
        self.settings: Any = None
        self.disk_settings: Any = None   # snapshot for dirty check (read-only)
        self.exec_layer = None
        self.extra: dict = {}

    def load(self, file_path: str, category: str,
             cat_def: AssetCategoryDef) -> bool:
        # Already loaded — just refresh.
        if (self.file_path == file_path
                and self.category == category
                and self.settings is not None):
            if cat_def.refresh_fn:
                cat_def.refresh_fn(self)
            return True
        # Fresh load
        self.reset()
        self.file_path = file_path
        self.category = category
        self.meta = read_meta_file(file_path)
        result = cat_def.load_fn(file_path)
        if result is None:
            return False
        settings, extra = result
        if settings is None:
            return False
        self.settings = settings
        self.extra = extra
        if (cat_def.access_mode == AssetAccessMode.READ_ONLY_RESOURCE
                and hasattr(settings, "copy")):
            self.disk_settings = settings.copy()
        return True

    def is_dirty(self) -> bool:
        if self.disk_settings is None:
            return False
        return self.settings != self.disk_settings


_state = _State()


# ═══════════════════════════════════════════════════════════════════════════
# Category registry
# ═══════════════════════════════════════════════════════════════════════════

_categories: Dict[str, AssetCategoryDef] = {}
_initialized = False


def _ensure_categories():
    global _initialized
    if _initialized:
        return
    _initialized = True

    # ── Texture ────────────────────────────────────────────────────────
    _categories["texture"] = AssetCategoryDef(
        display_name="asset.display_texture",
        access_mode=AssetAccessMode.READ_ONLY_RESOURCE,
        load_fn=_load_texture,
        editable_fields=[
            FieldDef("texture_type", "asset.texture_type", WidgetType.COMBO,
                     [("asset.tex_default", TextureType.DEFAULT),
                      ("asset.tex_normalmap", TextureType.NORMAL_MAP),
                      ("asset.tex_ui", TextureType.UI)]),
            FieldDef("srgb", "asset.srgb", WidgetType.CHECKBOX),
            FieldDef("max_size", "asset.max_size", WidgetType.COMBO,
                     [(str(s), s) for s in
                      (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]),
        ],
        custom_header_fn=_render_texture_preview,
    )

    # ── Audio ──────────────────────────────────────────────────────────
    _categories["audio"] = AssetCategoryDef(
        display_name="asset.display_audio",
        access_mode=AssetAccessMode.READ_ONLY_RESOURCE,
        load_fn=_load_audio,
        editable_fields=[
            FieldDef("force_mono", "asset.force_mono", WidgetType.CHECKBOX),
        ],
        extra_meta_keys=["file_size", "extension"],
    )

    # ── Shader ─────────────────────────────────────────────────────────
    _categories["shader"] = AssetCategoryDef(
        display_name="asset.display_shader",
        access_mode=AssetAccessMode.READ_ONLY_RESOURCE,
        load_fn=_load_shader,
        custom_body_fn=_render_shader_body,
    )

    _categories["font"] = AssetCategoryDef(
        display_name="asset.display_font",
        access_mode=AssetAccessMode.READ_ONLY_RESOURCE,
        load_fn=_load_font,
        custom_body_fn=_render_font_body,
        extra_meta_keys=["file_size", "extension"],
    )

    # ── Mesh ─────────────────────────────────────────────────────────────────
    _categories["mesh"] = AssetCategoryDef(
        display_name="asset.display_mesh",
        access_mode=AssetAccessMode.READ_ONLY_RESOURCE,
        load_fn=_load_mesh,
        editable_fields=[
            FieldDef("scale_factor", "asset.scale_factor", WidgetType.FLOAT,
                     float_speed=0.001, float_range=(0.0001, 1000.0)),
            FieldDef("generate_normals", "asset.generate_normals", WidgetType.CHECKBOX),
            FieldDef("generate_tangents", "asset.generate_tangents", WidgetType.CHECKBOX),
            FieldDef("flip_uvs", "asset.flip_uvs", WidgetType.CHECKBOX),
            FieldDef("optimize_mesh", "asset.optimize_mesh", WidgetType.CHECKBOX),
        ],
        custom_header_fn=_render_mesh_info,
        extra_meta_keys=["mesh_count", "vertex_count", "index_count", "material_slot_count"],
        show_header=False,
    )

    # ── Material ───────────────────────────────────────────────────────
    _categories["material"] = AssetCategoryDef(
        display_name="asset.display_material",
        access_mode=AssetAccessMode.READ_WRITE_RESOURCE,
        load_fn=_load_material,
        refresh_fn=_refresh_material,
        custom_body_fn=_render_material_body,
        autosave_debounce=0.35,
    )

    # ── Prefab ─────────────────────────────────────────────────────────
    _categories["prefab"] = AssetCategoryDef(
        display_name="asset.display_prefab",
        access_mode=AssetAccessMode.READ_WRITE_RESOURCE,
        load_fn=_load_prefab,
        custom_body_fn=_render_prefab_body,
        autosave_debounce=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-category loaders
# ═══════════════════════════════════════════════════════════════════════════


def _load_texture(path: str):
    return read_texture_import_settings(path), {"preview_height": 200.0}


def _load_audio(path: str):
    return read_audio_import_settings(path), {}


def _load_shader(path: str):
    meta = read_meta_file(path)
    guid = (meta or {}).get("guid", "")
    return ShaderAssetInfo.from_path(path, guid=guid), {}


def _load_font(path: str):
    meta = read_meta_file(path)
    guid = (meta or {}).get("guid", "")
    return FontAssetInfo.from_path(path, guid=guid), {}


def _load_mesh(path: str):
    return read_mesh_import_settings(path), {}


def _load_material(path: str):
    from InfEngine.core.material import Material
    mat = Material.load(path)
    if mat is None:
        return None
    native = mat.native
    try:
        cached = json.loads(native.serialize())
    except (RuntimeError, ValueError, json.JSONDecodeError):
        cached = {"name": mat.name, "properties": {}}
    _sync_material_shader_metadata(cached)
    return mat, {
        "native_mat": native,
        "cached_data": cached,
        "shader_cache": {".vert": None, ".frag": None},
        "shader_sync_key": "",
    }


def _load_prefab(path: str):
    """Load a .prefab file and return its root object JSON as the settings."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    root = data.get("root_object")
    if root is None:
        return None
    version = data.get("prefab_version", 0)
    return root, {
        "prefab_version": version,
        "prefab_path": path,
        "prefab_dirty": False,
        "prefab_envelope": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Prefab body — full component-level editor (mirrors hierarchy inspector)
# ═══════════════════════════════════════════════════════════════════════════

_PREFAB_DRAG_SPEED = 0.1
_PREFAB_DRAG_SPEED_FINE = 0.01
_PREFAB_SKIP_COMP_KEYS = frozenset({
    "type", "component_id", "schema_version", "enabled",
})
_PREFAB_SKIP_PY_KEYS = frozenset({
    "py_type_name", "component_id", "schema_version", "enabled", "script_guid",
})


def _render_prefab_body(ctx: InfGUIContext, panel, state: _State):
    """Render the prefab asset inspector body — full object editor."""
    from .theme import ImGuiTreeNodeFlags
    from .inspector_utils import render_component_header, render_info_text

    root = state.settings
    if not isinstance(root, dict):
        ctx.label(t("asset.invalid_prefab"))
        return

    dirty = _render_prefab_object(ctx, root, "root", is_root=True)

    # Recurse children
    children = root.get("children", [])
    if children:
        ctx.dummy(0, 6)
        ctx.separator()
        ctx.dummy(0, 4)
        ctx.push_style_color(ImGuiCol.Text, *Theme.PREFAB_TEXT)
        ctx.label(t("asset.prefab_children").format(n=len(children)))
        ctx.pop_style_color(1)
        ctx.dummy(0, 2)
        dirty |= _render_prefab_children(ctx, children, "root")

    # Summary
    ctx.dummy(0, 8)
    ctx.separator()
    ctx.dummy(0, 4)
    total_nodes = _count_descendants(root) + 1
    total_comps = len(root.get("components", [])) + len(root.get("py_components", []))
    render_info_text(ctx, t("asset.prefab_summary").format(nodes=total_nodes, comps=total_comps))

    # Save if dirty
    if dirty:
        _save_prefab_state(state)


def _render_prefab_children(ctx: InfGUIContext, children: list, parent_uid: str) -> bool:
    """Recursively render child objects as collapsible tree nodes."""
    from .theme import ImGuiTreeNodeFlags
    dirty = False
    flags = (ImGuiTreeNodeFlags.OpenOnArrow
             | ImGuiTreeNodeFlags.SpanAvailWidth
             | ImGuiTreeNodeFlags.FramePadding)
    for i, child in enumerate(children):
        child_name = child.get("name", f"Child_{i}")
        child_uid = f"{parent_uid}_ch{i}"
        ctx.push_id_str(child_uid)
        ctx.push_style_color(ImGuiCol.Text, *Theme.PREFAB_TEXT)
        is_open = ctx.tree_node_ex(child_name, flags)
        ctx.pop_style_color(1)
        if is_open:
            dirty |= _render_prefab_object(ctx, child, child_uid)
            grandchildren = child.get("children", [])
            if grandchildren:
                dirty |= _render_prefab_children(ctx, grandchildren, child_uid)
            ctx.tree_pop()
        ctx.pop_id()
    return dirty


def _render_prefab_object(ctx: InfGUIContext, obj: dict, uid: str,
                          is_root: bool = False) -> bool:
    """Render all properties of a single prefab object node. Returns True if modified."""
    from .inspector_utils import render_component_header

    dirty = False
    ctx.push_id_str(f"pobj_{uid}")

    # ── Name ──
    old_name = obj.get("name", "GameObject")
    field_label(ctx, t("asset.prefab_name"))
    new_name = ctx.text_input(f"##name_{uid}", old_name, 256)
    if new_name != old_name:
        obj["name"] = new_name
        dirty = True

    # ── Active ──
    active = obj.get("active", True)
    new_active = ctx.checkbox(f"{t('asset.prefab_active')}##{uid}", active)
    if new_active != active:
        obj["active"] = new_active
        dirty = True

    # ── Tag ──
    tag = obj.get("tag", "Untagged")
    field_label(ctx, t("asset.prefab_tag"))
    new_tag = ctx.text_input(f"##tag_{uid}", tag, 128)
    if new_tag != tag:
        obj["tag"] = new_tag
        dirty = True

    # ── Layer ──
    layer = obj.get("layer", 0)
    field_label(ctx, t("asset.prefab_layer"))
    new_layer = ctx.drag_int(f"##layer_{uid}", layer, 0.1, 0, 31)
    if new_layer != layer:
        obj["layer"] = new_layer
        dirty = True

    ctx.dummy(0, 4)
    ctx.separator()
    ctx.dummy(0, 4)

    # ── Transform ──
    transform = obj.get("transform")
    if transform and isinstance(transform, dict):
        header_open, _ = render_component_header(
            ctx, t("asset.prefab_transform"), show_enabled=False, default_open=is_root,
        )
        if header_open:
            dirty |= _render_prefab_transform(ctx, transform, uid)

    # ── C++ Components ──
    components = obj.get("components", [])
    for ci, comp in enumerate(components):
        type_name = comp.get("type", "Component")
        if type_name == "Transform":
            continue
        enabled = comp.get("enabled", True)
        comp_uid = f"{uid}_c{ci}"
        header_open, new_enabled = render_component_header(
            ctx, type_name, show_enabled=True, is_enabled=enabled,
            default_open=is_root,
        )
        if new_enabled != enabled:
            comp["enabled"] = new_enabled
            dirty = True
        if header_open:
            dirty |= _render_prefab_component_fields(ctx, comp, comp_uid, _PREFAB_SKIP_COMP_KEYS)

    # ── Python Components ──
    py_components = obj.get("py_components", [])
    for pi, pyc in enumerate(py_components):
        type_name = pyc.get("py_type_name", "PyComponent")
        enabled = pyc.get("enabled", True)
        pyc_uid = f"{uid}_p{pi}"
        header_open, new_enabled = render_component_header(
            ctx, type_name, show_enabled=True, is_enabled=enabled,
            suffix=t("asset.prefab_script"), default_open=is_root,
        )
        if new_enabled != enabled:
            pyc["enabled"] = new_enabled
            dirty = True
        if header_open:
            dirty |= _render_prefab_component_fields(ctx, pyc, pyc_uid, _PREFAB_SKIP_PY_KEYS)
            # Also render py_fields if present
            py_fields = pyc.get("py_fields")
            if isinstance(py_fields, dict):
                dirty |= _render_prefab_json_dict(ctx, py_fields, f"{pyc_uid}_fields")

    ctx.pop_id()
    return dirty


def _render_prefab_transform(ctx: InfGUIContext, transform: dict, uid: str) -> bool:
    """Render position/rotation/scale vector3 controls from transform JSON."""
    dirty = False
    lw = max_label_w(ctx, [t("asset.prefab_position"), t("asset.prefab_rotation"), t("asset.prefab_scale")])

    # Position — array format: [x, y, z]
    pos = transform.get("position", [0.0, 0.0, 0.0])
    if isinstance(pos, list) and len(pos) == 3:
        px, py_, pz = float(pos[0]), float(pos[1]), float(pos[2])
        npx, npy, npz = ctx.vector3(f"{t('asset.prefab_position')}##{uid}", px, py_, pz, _PREFAB_DRAG_SPEED, lw)
        if abs(npx - px) > 1e-6 or abs(npy - py_) > 1e-6 or abs(npz - pz) > 1e-6:
            transform["position"] = [npx, npy, npz]
            dirty = True

    # Rotation
    rot = transform.get("rotation", [0.0, 0.0, 0.0])
    if isinstance(rot, list) and len(rot) == 3:
        rx, ry, rz = float(rot[0]), float(rot[1]), float(rot[2])
        nrx, nry, nrz = ctx.vector3(f"{t('asset.prefab_rotation')}##{uid}", rx, ry, rz, _PREFAB_DRAG_SPEED, lw)
        if abs(nrx - rx) > 1e-6 or abs(nry - ry) > 1e-6 or abs(nrz - rz) > 1e-6:
            transform["rotation"] = [nrx, nry, nrz]
            dirty = True

    # Scale
    scl = transform.get("scale", [1.0, 1.0, 1.0])
    if isinstance(scl, list) and len(scl) == 3:
        sx, sy, sz = float(scl[0]), float(scl[1]), float(scl[2])
        nsx, nsy, nsz = ctx.vector3(f"{t('asset.prefab_scale')}##{uid}", sx, sy, sz, _PREFAB_DRAG_SPEED_FINE, lw)
        if abs(nsx - sx) > 1e-6 or abs(nsy - sy) > 1e-6 or abs(nsz - sz) > 1e-6:
            transform["scale"] = [nsx, nsy, nsz]
            dirty = True

    return dirty


def _render_prefab_component_fields(ctx: InfGUIContext, comp: dict,
                                     uid: str, skip_keys: frozenset) -> bool:
    """Render all fields of a serialized component as editable widgets."""
    dirty = False
    keys = [k for k in comp.keys() if k not in skip_keys]
    if not keys:
        ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
        ctx.label("  " + t("asset.no_editable_fields"))
        ctx.pop_style_color(1)
        return False

    for key in keys:
        value = comp[key]
        new_value, changed = _render_prefab_json_value(ctx, key, value, f"{uid}_{key}")
        if changed:
            comp[key] = new_value
            dirty = True
    return dirty


def _render_prefab_json_value(ctx: InfGUIContext, key: str, value, uid: str):
    """Render a single JSON value with an appropriate widget. Returns (new_value, changed)."""
    display_key = key.replace("_", " ").title()

    if isinstance(value, bool):
        field_label(ctx, display_key)
        new_val = ctx.checkbox(f"##{uid}", value)
        return new_val, new_val != value

    if isinstance(value, int):
        field_label(ctx, display_key)
        new_val = ctx.drag_int(f"##{uid}", value, 0.1, -999999, 999999)
        return new_val, new_val != value

    if isinstance(value, float):
        field_label(ctx, display_key)
        new_val = ctx.drag_float(f"##{uid}", value, 0.01, -999999.0, 999999.0)
        return new_val, abs(new_val - value) > 1e-7

    if isinstance(value, str):
        field_label(ctx, display_key)
        new_val = ctx.text_input(f"##{uid}", value, 512)
        return new_val, new_val != value

    if isinstance(value, list):
        # Array of 3 floats → vector3
        if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
            lw = max_label_w(ctx, [display_key])
            x, y, z = float(value[0]), float(value[1]), float(value[2])
            nx, ny, nz = ctx.vector3(f"{display_key}##{uid}", x, y, z, 0.01, lw)
            changed = abs(nx - x) > 1e-7 or abs(ny - y) > 1e-7 or abs(nz - z) > 1e-7
            return [nx, ny, nz] if changed else value, changed

        # Array of 4 floats → render as 4 drag floats (color-like)
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            ctx.label(f"{display_key}:")
            changed = False
            new_arr = list(value)
            labels_4 = ["X", "Y", "Z", "W"]
            for i in range(4):
                field_label(ctx, f"  {labels_4[i]}")
                nv = ctx.drag_float(f"##{uid}_{i}", float(value[i]), 0.01, -999999.0, 999999.0)
                if abs(nv - float(value[i])) > 1e-7:
                    new_arr[i] = nv
                    changed = True
            return new_arr if changed else value, changed

        # Other arrays — show as read-only text
        ctx.label(f"{display_key}: {json.dumps(value, ensure_ascii=False)}")
        return value, False

    if isinstance(value, dict):
        # Nested dict — render as collapsible section
        from .theme import ImGuiTreeNodeFlags
        flags = ImGuiTreeNodeFlags.OpenOnArrow | ImGuiTreeNodeFlags.SpanAvailWidth
        if ctx.tree_node_ex(f"{display_key}##{uid}", flags):
            changed = _render_prefab_json_dict(ctx, value, uid)
            ctx.tree_pop()
            return value, changed
        return value, False

    # Fallback — display as text
    ctx.label(f"{display_key}: {value}")
    return value, False


def _render_prefab_json_dict(ctx: InfGUIContext, d: dict, uid: str) -> bool:
    """Render all key-value pairs in a JSON dict. Returns True if modified."""
    dirty = False
    for key, value in list(d.items()):
        new_value, changed = _render_prefab_json_value(ctx, key, value, f"{uid}_{key}")
        if changed:
            d[key] = new_value
            dirty = True
    return dirty


def _save_prefab_state(state: _State):
    """Write the modified prefab data back to disk."""
    path = state.extra.get("prefab_path")
    envelope = state.extra.get("prefab_envelope")
    if not path or not envelope:
        return
    envelope["root_object"] = state.settings
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def _count_descendants(node: dict) -> int:
    """Count total descendant nodes (excluding the node itself)."""
    children = node.get("children", [])
    total = len(children)
    for child in children:
        total += _count_descendants(child)
    return total


def _refresh_material(state: _State):
    native = state.extra.get("native_mat")
    if native:
        try:
            fresh = json.loads(native.serialize())
            merged = _merge_material_cached_data(state.extra.get("cached_data"), fresh)
            _sync_material_shader_metadata(merged)
            state.extra["cached_data"] = merged
        except (RuntimeError, ValueError, json.JSONDecodeError):
            pass


def _merge_material_cached_data(existing: Optional[dict], fresh: dict) -> dict:
    """Merge native material values into cached inspector data.

    Native material JSON does not preserve Python-only metadata such as
    shader-derived Color/HDR annotations. Preserve those keys while taking the
    latest live values from the native material.
    """
    if not isinstance(existing, dict):
        return fresh

    merged = dict(existing)
    merged.update(fresh)

    # Preserve Python-only metadata that C++ serialiser does not round-trip
    if "_shader_property_order" in existing and "_shader_property_order" not in fresh:
        merged["_shader_property_order"] = existing["_shader_property_order"]

    fresh_props = fresh.get("properties") if isinstance(fresh.get("properties"), dict) else {}
    existing_props = existing.get("properties") if isinstance(existing.get("properties"), dict) else {}

    merged_props = {}
    for name, fresh_prop in fresh_props.items():
        if not isinstance(fresh_prop, dict):
            merged_props[name] = fresh_prop
            continue
        merged_prop = dict(fresh_prop)
        existing_prop = existing_props.get(name)
        if isinstance(existing_prop, dict):
            for key, value in existing_prop.items():
                if key not in ("value", "guid"):
                    merged_prop[key] = value
        merged_props[name] = merged_prop

    merged["properties"] = merged_props
    return merged


def _sync_material_shader_metadata(mat_data: dict):
    shaders = mat_data.get("shaders") if isinstance(mat_data.get("shaders"), dict) else {}
    frag_shader_id = shaders.get("fragment", "")
    if frag_shader_id:
        from . import inspector_shader_utils as shader_utils

        shader_utils.sync_properties_from_shader(mat_data, frag_shader_id, ".frag", remove_unknown=True)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def render_asset_inspector(ctx: InfGUIContext, panel,
                           file_path: str, category: str):
    """Single entry point for all asset inspectors."""
    _ensure_categories()
    cat_def = _categories.get(category)
    if cat_def is None:
        ctx.label(t("asset.unknown_asset_type").format(cat=category))
        return

    if not _state.load(file_path, category, cat_def):
        ctx.label(t("asset.failed_load").format(name=t(cat_def.display_name)))
        ctx.label(file_path)
        return

    _state.exec_layer = get_asset_execution_layer(
        _state.exec_layer, category, file_path, cat_def.access_mode,
        autosave_debounce_sec=cat_def.autosave_debounce,
    )

    # ── Header (shared for all categories) ─────────────────────────────
    if cat_def.show_header:
        _render_header(ctx, cat_def, _state)

    # ── Custom header additions (e.g. texture preview) ─────────────────
    if cat_def.custom_header_fn:
        cat_def.custom_header_fn(ctx, panel, _state)

    # ── Body (auto-generated fields or custom) ─────────────────────────
    if cat_def.custom_body_fn:
        cat_def.custom_body_fn(ctx, panel, _state)
    elif cat_def.editable_fields:
        _render_import_fields(ctx, cat_def, _state)

    # ── Footer ─────────────────────────────────────────────────────────
    if (cat_def.access_mode == AssetAccessMode.READ_ONLY_RESOURCE
            and cat_def.editable_fields):
        render_apply_revert(
            ctx, _state.is_dirty(),
            on_apply=lambda: _on_apply(),
            on_revert=_on_revert,
        )
    elif cat_def.access_mode == AssetAccessMode.READ_WRITE_RESOURCE:
        if _state.exec_layer:
            _state.exec_layer.flush_rw_autosave()


def invalidate():
    """Reset all inspector state (called on selection change)."""
    _state.reset()


def invalidate_asset(path: str):
    """Clear inspector cache if *path* is the currently inspected asset.

    Call this when an asset file is deleted so that re-creating a file with
    the same name performs a fresh load instead of reusing stale cached data.
    """
    if not _state.file_path or not path:
        return
    if os.path.normpath(_state.file_path) == os.path.normpath(path):
        _state.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Shared rendering helpers
# ═══════════════════════════════════════════════════════════════════════════


def _render_header(ctx: InfGUIContext, cat_def: AssetCategoryDef,
                   state: _State):
    """Render the standard asset header: name, GUID, path, extra meta."""
    filename = os.path.basename(state.file_path)
    ctx.label(f"{t(cat_def.display_name)}: {filename}")

    # GUID — try .meta first, then serialized data (material stores it inside)
    guid = (state.meta or {}).get("guid", "")
    if not guid:
        cached = state.extra.get("cached_data")
        if cached:
            guid = cached.get("guid", "")
    if guid:
        ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
        ctx.label(t("asset.guid_label").format(guid=guid))
        ctx.pop_style_color(1)

    # Path
    ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
    ctx.label(t("asset.path_label").format(path=state.file_path))
    ctx.pop_style_color(1)

    # Extra metadata from .meta (e.g. file_size, extension for audio)
    if cat_def.extra_meta_keys and state.meta:
        for key in cat_def.extra_meta_keys:
            val = state.meta.get(key, "")
            if not val:
                continue
            ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
            if key == "file_size":
                _render_file_size(ctx, val)
            else:
                ctx.label(f"{key.replace('_', ' ').title()}: {val}")
            ctx.pop_style_color(1)

    ctx.separator()


def _render_file_size(ctx: InfGUIContext, val):
    try:
        size = int(val)
        if size >= 1048576:
            ctx.label(t("asset.size_mb").format(size=f"{size / 1048576:.2f}"))
        elif size >= 1024:
            ctx.label(t("asset.size_kb").format(size=f"{size / 1024:.1f}"))
        else:
            ctx.label(t("asset.size_bytes").format(size=size))
    except (ValueError, TypeError):
        ctx.label(t("asset.size_bytes").format(size=val))


def _render_import_fields(ctx: InfGUIContext, cat_def: AssetCategoryDef,
                          state: _State):
    """Auto-render editable import-settings fields from descriptors."""
    from .inspector_utils import render_compact_section_header, render_inspector_checkbox

    if render_compact_section_header(ctx, t("asset.import_settings"), level="secondary"):
        labels = [t(f.label) for f in cat_def.editable_fields]
        lw = max_label_w(ctx, labels)

        for fdef in cat_def.editable_fields:
            cur = getattr(state.settings, fdef.key)
            wid = f"##{fdef.key}"

            if fdef.field_type == WidgetType.CHECKBOX:
                # Disable sRGB when texture_type is NORMAL_MAP
                disabled = (fdef.key == "srgb"
                            and hasattr(state.settings, "texture_type")
                            and state.settings.texture_type == TextureType.NORMAL_MAP)
                if disabled:
                    ctx.begin_disabled(True)
                new_val = render_inspector_checkbox(ctx, t(fdef.label), cur)
                if new_val != cur:
                    setattr(state.settings, fdef.key, new_val)
                if disabled:
                    ctx.end_disabled()

            elif fdef.field_type == WidgetType.COMBO:
                field_label(ctx, t(fdef.label), lw)
                display_labels = [t(e[0]) if e[0].startswith("asset.") else e[0] for e in fdef.combo_entries]
                values = [e[1] for e in fdef.combo_entries]
                try:
                    idx = values.index(cur)
                except ValueError:
                    idx = 0
                new_idx = ctx.combo(wid, idx, display_labels)
                if new_idx != idx:
                    setattr(state.settings, fdef.key, values[new_idx])
                    if hasattr(state.settings, '_sync_derived_fields'):
                        state.settings._sync_derived_fields()

            elif fdef.field_type == WidgetType.FLOAT:
                field_label(ctx, t(fdef.label), lw)
                speed = fdef.float_speed
                v_min = fdef.float_range[0] if fdef.float_range else 0.0
                v_max = fdef.float_range[1] if fdef.float_range else 0.0
                new_val = ctx.drag_float(wid, float(cur), speed, v_min, v_max)
                if new_val != cur:
                    setattr(state.settings, fdef.key, new_val)


# ── Apply / Revert actions ─────────────────────────────────────────────


def _on_apply():
    if _state.settings is None or _state.exec_layer is None:
        return
    ok = _state.exec_layer.apply_import_settings(_state.settings)
    if ok and hasattr(_state.settings, "copy"):
        _state.disk_settings = _state.settings.copy()


def _on_revert():
    _state.file_path = ""  # force full reload next frame


# ═══════════════════════════════════════════════════════════════════════════
# Mesh — info section (custom_header_fn)
# ═══════════════════════════════════════════════════════════════════════════


def _render_mesh_info(ctx: InfGUIContext, panel, state: _State):
    """Render mesh metadata summary (vertex count, submesh count, etc.)."""
    meta = state.meta
    if not meta:
        return

    from .inspector_utils import render_compact_section_header

    if render_compact_section_header(ctx, t("asset.mesh_info"), level="secondary"):
        labels = [t("asset.mesh_file"), t("asset.mesh_meshes"), t("asset.mesh_vertices"), t("asset.mesh_indices"), t("asset.mesh_material_slots")]
        lw = max_label_w(ctx, labels)

        filename = os.path.basename(state.file_path)
        ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
        field_label(ctx, t("asset.mesh_file"), lw)
        ctx.label(filename)
        ctx.pop_style_color(1)

        mesh_count = meta.get("mesh_count", "?")
        vertex_count = meta.get("vertex_count", "?")
        index_count = meta.get("index_count", "?")
        mat_slots = meta.get("material_slot_count", "?")
        mat_names = meta.get("material_slots", "")

        ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
        field_label(ctx, t("asset.mesh_meshes"), lw)
        ctx.label(str(mesh_count))
        field_label(ctx, t("asset.mesh_vertices"), lw)
        ctx.label(str(vertex_count))
        field_label(ctx, t("asset.mesh_indices"), lw)
        ctx.label(str(index_count))
        field_label(ctx, t("asset.mesh_material_slots"), lw)
        ctx.label(str(mat_slots))
        if mat_names:
            field_label(ctx, t("asset.mesh_materials"), lw)
            ctx.label(str(mat_names))
        ctx.pop_style_color(1)

    ctx.separator()


# ═══════════════════════════════════════════════════════════════════════════
# Texture — preview section (custom_header_fn)
# ═══════════════════════════════════════════════════════════════════════════

_PREVIEW_MIN_H = 60.0
_PREVIEW_MAX_H = 800.0
_SPLITTER_H = 14.0


def _render_texture_preview(ctx: InfGUIContext, panel, state: _State):
    """Render texture preview image + drag-to-resize splitter."""
    if not panel or not hasattr(panel, "_InspectorPanel__preview_manager"):
        return
    pm = panel._InspectorPanel__preview_manager
    if not pm or not pm.load_preview(state.file_path):
        return

    settings = state.settings
    display_mode = 1 if settings.texture_type == TextureType.NORMAL_MAP else 0
    preview_max = min(settings.max_size, 512)
    pm.set_preview_settings(display_mode, preview_max, settings.srgb)

    avail_w = ctx.get_content_region_avail_width()
    preview_h = min(avail_w, state.extra.get("preview_height", 200.0))
    if avail_w > 0 and preview_h > 0:
        pm.render_preview(ctx, avail_w, preview_h)

    # ── Drag splitter ──────────────────────────────────────────────────
    ctx.separator()
    avail_w = ctx.get_content_region_avail_width()
    ctx.invisible_button("##TexPreviewSplitter", avail_w, _SPLITTER_H)
    if ctx.is_item_hovered() or ctx.is_item_active():
        ctx.set_mouse_cursor(3)  # ResizeNS
    if ctx.is_item_active():
        dy = ctx.get_mouse_drag_delta_y(0)
        if abs(dy) > 1.0:
            h = state.extra.get("preview_height", 200.0)
            state.extra["preview_height"] = max(
                _PREVIEW_MIN_H, min(_PREVIEW_MAX_H, h + dy))
            ctx.reset_mouse_drag_delta(0)
    ctx.separator()


# ═══════════════════════════════════════════════════════════════════════════
# Shader — custom body (path editing + source preview)
# ═══════════════════════════════════════════════════════════════════════════


def _render_shader_body(ctx: InfGUIContext, panel, state: _State):
    info = state.settings  # ShaderAssetInfo

    # Shader type (read-only)
    lw = max_label_w(ctx, [t("asset.shader_type")])
    field_label(ctx, t("asset.shader_type"), lw)
    ctx.label(info.shader_type.capitalize() if info.shader_type else t("asset.shader_unknown"))
    ctx.separator()

    # ── Path editing ───────────────────────────────────────────────────
    from .inspector_utils import render_compact_section_header

    if render_compact_section_header(ctx, t("asset.shader_path"), level="secondary"):
        plw = max_label_w(ctx, [t("asset.shader_source_path")])
        field_label(ctx, t("asset.shader_source_path"), plw)
        new_path = ctx.text_input("##shader_src_path", info.source_path, 512)

        if new_path != info.source_path:
            ext = os.path.splitext(new_path)[1].lower()
            valid = {".vert", ".frag", ".geom", ".comp", ".tesc", ".tese"}
            if ext not in valid:
                ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
                ctx.label(t("asset.shader_invalid_ext").format(ext=ext))
                ctx.pop_style_color(1)
            else:
                if not os.path.isfile(new_path):
                    ctx.push_style_color(ImGuiCol.Text, *Theme.WARNING_TEXT)
                    ctx.label(t("asset.file_not_exist_warning"))
                    ctx.pop_style_color(1)
                ctx.button(t("asset.apply_path_change"),
                           lambda np=new_path: _apply_shader_path(
                               state, np))

    ctx.separator()

    # ── Source preview ─────────────────────────────────────────────────
    if render_compact_section_header(ctx, t("asset.shader_source_preview"), default_open=False, level="secondary"):
        _render_shader_source(ctx, state.file_path)


def _render_font_body(ctx: InfGUIContext, panel, state: _State):
    info = state.settings
    lw = max_label_w(ctx, [t("asset.font_format"), t("asset.font_source_path")])
    field_label(ctx, t("asset.font_format"), lw)
    ctx.label(info.font_type.capitalize() if info.font_type else t("asset.font_unknown"))
    field_label(ctx, t("asset.font_source_path"), lw)
    ctx.label(info.source_path)


def _apply_shader_path(state: _State, new_path: str):
    info = state.settings
    old_path = info.source_path
    if state.exec_layer:
        state.exec_layer.move_asset_path(new_path)
    from InfEngine.core.shader import Shader
    shader_id = os.path.splitext(os.path.basename(old_path))[0]
    Shader.invalidate(shader_id)
    info.source_path = new_path
    info.shader_type = ShaderAssetInfo.from_path(new_path).shader_type


def _render_shader_source(ctx: InfGUIContext, file_path: str):
    if not os.path.isfile(file_path):
        ctx.label(t("asset.file_not_found"))
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[:40]
        text = "".join(lines)
        if len(lines) == 40:
            text += "\n" + t("asset.shader_truncated")
        ctx.push_style_color(ImGuiCol.Text, *Theme.SUCCESS_TEXT)
        ctx.label(text)
        ctx.pop_style_color(1)
    except OSError:
        ctx.label(t("asset.failed_read_source"))


# ═══════════════════════════════════════════════════════════════════════════
# Material — custom body (delegates to inspector_material)
# ═══════════════════════════════════════════════════════════════════════════


def _render_material_body(ctx: InfGUIContext, panel, state: _State):
    from . import inspector_material as mat_ui
    mat_ui.render_material_body(ctx, panel, state)
