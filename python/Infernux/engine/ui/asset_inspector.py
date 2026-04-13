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

import copy
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from Infernux.lib import InxGUIContext
from Infernux.engine.i18n import t
from Infernux.core.asset_types import (
    FilterMode,
    SpriteFrame,
    TextureType,
    TextureImportSettings,
    WrapMode,
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
from Infernux.debug import Debug


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
                      ("asset.tex_ui", TextureType.UI),
                      ("asset.tex_sprite", TextureType.SPRITE)]),
            FieldDef("srgb", "asset.srgb", WidgetType.CHECKBOX),
            FieldDef("filter_mode", "asset.filter_mode", WidgetType.COMBO,
                     [("asset.filter_point", FilterMode.POINT),
                      ("asset.filter_bilinear", FilterMode.BILINEAR),
                      ("asset.filter_trilinear", FilterMode.TRILINEAR)]),
            FieldDef("wrap_mode", "asset.wrap_mode", WidgetType.COMBO,
                     [("asset.wrap_repeat", WrapMode.REPEAT),
                      ("asset.wrap_clamp", WrapMode.CLAMP),
                      ("asset.wrap_mirror", WrapMode.MIRROR)]),
            FieldDef("max_size", "asset.max_size", WidgetType.COMBO,
                     [(str(s), s) for s in
                      (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]),
        ],
        custom_header_fn=_render_texture_preview,
        custom_body_fn=_render_sprite_body,
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
    from Infernux.core.material import Material
    mat = Material.load(path)
    if mat is None:
        return None
    native = mat.native
    try:
        cached = json.loads(native.serialize())
    except (RuntimeError, ValueError, json.JSONDecodeError):
        cached = {"name": mat.name, "properties": {}}
    old_prop_names = set(cached.get("properties", {}).keys())
    _sync_material_shader_metadata(cached)
    new_prop_names = set(cached.get("properties", {}).keys())
    if new_prop_names != old_prop_names:
        # Vertex/fragment shader sync added new properties — push them to the
        # native C++ material so the UBO picks up the correct default values.
        try:
            native.deserialize(json.dumps(cached))
        except (RuntimeError, ValueError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    return mat, {
        "native_mat": native,
        "cached_data": cached,
        "cached_json": json.dumps(cached),
        "shader_cache": {".vert": None, ".frag": None},
        "shader_sync_key": "",
    }


def _load_prefab(path: str):
    """Load a .prefab file into a safe data-only representation.

    The previous implementation instantiated a hidden preview scene and then
    routed the prefab through the full object inspector. That path re-used
    editor-only object rendering on non-active temporary scene objects and
    allocated a new native scene for each selection, which is not safe with
    the current SceneManager API surface.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None

    root_json = data.get("root_object")
    if root_json is None:
        return None

    root_copy = copy.deepcopy(root_json)
    return root_copy, {
        "prefab_version": data.get("prefab_version", 0),
        "prefab_path": path,
        "prefab_envelope": data,
        "root_name": root_copy.get("name", "GameObject"),
        "node_count": _count_prefab_nodes(root_copy),
        "component_count": _count_prefab_components(root_copy),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Prefab body — exact same components rendering as hierarchy instances
# ═══════════════════════════════════════════════════════════════════════════

def _render_prefab_body(ctx: InxGUIContext, panel, state: _State):
    """Render a safe prefab summary plus Prefab Mode entry point.

    Inline full-object rendering is intentionally disabled here for stability.
    Prefab editing remains available through Prefab Mode.
    """
    from .inspector_utils import render_info_text, render_compact_section_header

    prefab_root = state.settings
    if not isinstance(prefab_root, dict):
        ctx.label(t("asset.invalid_prefab"))
        return

    def _open_prefab_mode():
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm and hasattr(sfm, "open_prefab_mode_with_undo"):
            sfm.open_prefab_mode_with_undo(state.extra["prefab_path"])

    ctx.dummy(0, 4)
    ctx.push_style_color(ImGuiCol.Button, *Theme.PREFAB_BTN_NORMAL)
    ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.PREFAB_BTN_HOVERED)
    ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.PREFAB_BTN_ACTIVE)
    try:
        ctx.button(t("asset.open_prefab_mode"), _open_prefab_mode, -1, 32)
    finally:
        ctx.pop_style_color(3)

    ctx.dummy(0, 8)
    render_info_text(ctx, t("asset.prefab_safe_mode"))
    ctx.separator()
    ctx.dummy(0, 4)

    labels = [
        t("asset.prefab_root"),
        t("asset.prefab_nodes"),
        t("asset.prefab_components_count"),
        t("asset.prefab_scripts_count"),
        t("asset.prefab_path"),
    ]
    lw = max_label_w(ctx, labels)

    root_name = state.extra.get("root_name", prefab_root.get("name", "GameObject"))
    component_count = state.extra.get("component_count", 0)
    script_count = _count_prefab_script_components(prefab_root)
    node_count = state.extra.get("node_count", 1)

    field_label(ctx, t("asset.prefab_root"), lw)
    ctx.label(str(root_name))
    field_label(ctx, t("asset.prefab_nodes"), lw)
    ctx.label(str(node_count))
    field_label(ctx, t("asset.prefab_components_count"), lw)
    ctx.label(str(component_count))
    field_label(ctx, t("asset.prefab_scripts_count"), lw)
    ctx.label(str(script_count))
    field_label(ctx, t("asset.prefab_path"), lw)
    ctx.label(state.extra.get("prefab_path", state.file_path))

    ctx.dummy(0, 6)
    ctx.separator()

    if render_compact_section_header(ctx, t("asset.prefab_root_object"), level="secondary"):
        _render_prefab_root_summary(ctx, prefab_root)

    if render_compact_section_header(ctx, t("asset.prefab_raw_json_preview"), default_open=False, level="secondary"):
        preview = json.dumps(prefab_root, indent=2, ensure_ascii=False)
        if len(preview) > 8000:
            preview = preview[:8000] + "\n..."
        ctx.label(preview)


def _count_prefab_nodes(node: dict) -> int:
    total = 1
    for child in node.get("children", []):
        if isinstance(child, dict):
            total += _count_prefab_nodes(child)
    return total


def _count_prefab_components(node: dict) -> int:
    total = len(node.get("components", []))
    total += len(node.get("py_components", []))
    for child in node.get("children", []):
        if isinstance(child, dict):
            total += _count_prefab_components(child)
    return total


def _count_prefab_script_components(node: dict) -> int:
    total = len(node.get("py_components", []))
    for child in node.get("children", []):
        if isinstance(child, dict):
            total += _count_prefab_script_components(child)
    return total


def _render_prefab_root_summary(ctx: InxGUIContext, root: dict):
    transform = root.get("transform", {}) if isinstance(root.get("transform"), dict) else {}
    position = transform.get("position", [0.0, 0.0, 0.0])
    rotation = transform.get("rotation", [0.0, 0.0, 0.0])
    scale = transform.get("scale", [1.0, 1.0, 1.0])
    ctx.label(f"{t('asset.prefab_name')}: {root.get('name', 'GameObject')}")
    ctx.label(f"{t('asset.prefab_active')}: {bool(root.get('active', True))}")
    ctx.label(f"{t('asset.prefab_tag')}: {root.get('tag', 'Untagged')}")
    ctx.label(f"{t('asset.prefab_layer')}: {root.get('layer', 0)}")
    ctx.label(f"{t('asset.prefab_position')}: {position}")
    ctx.label(f"{t('asset.prefab_rotation')}: {rotation}")
    ctx.label(f"{t('asset.prefab_scale')}: {scale}")
    ctx.label(f"{t('asset.prefab_native_components')}: {len(root.get('components', []))}")
    ctx.label(f"{t('asset.prefab_script_components')}: {len(root.get('py_components', []))}")
    ctx.label(f"{t('asset.prefab_children_count')}: {len(root.get('children', []))}")


def _refresh_material(state: _State):
    native = state.extra.get("native_mat")
    if native:
        try:
            fresh = json.loads(native.serialize())
            merged = _merge_material_cached_data(state.extra.get("cached_data"), fresh)
            _sync_material_shader_metadata(merged)
            state.extra["cached_data"] = merged
            state.extra["cached_json"] = json.dumps(merged)
        except (RuntimeError, ValueError, json.JSONDecodeError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
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

    # Preserve shader-declared properties that exist in cached data but not yet
    # in the native serialisation (e.g. vertex shader properties just synced).
    shader_order = existing.get("_shader_property_order", [])
    shader_declared = set(shader_order) if shader_order else set()
    for name in shader_declared:
        if name not in merged_props and name in existing_props:
            merged_props[name] = existing_props[name]

    merged["properties"] = merged_props
    return merged


def _sync_material_shader_metadata(mat_data: dict):
    shaders = mat_data.get("shaders") if isinstance(mat_data.get("shaders"), dict) else {}
    vert_shader_id = shaders.get("vertex", "")
    frag_shader_id = shaders.get("fragment", "")
    if vert_shader_id or frag_shader_id:
        from . import inspector_shader_utils as shader_utils

        shader_utils.sync_all_shader_properties(mat_data, vert_shader_id, frag_shader_id, remove_unknown=True)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def render_asset_inspector(ctx: InxGUIContext, panel,
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
    _sprite_state.reset()


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


def _render_header(ctx: InxGUIContext, cat_def: AssetCategoryDef,
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


def _render_file_size(ctx: InxGUIContext, val):
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


def _render_import_fields(ctx: InxGUIContext, cat_def: AssetCategoryDef,
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
                    # Only sync derived fields when texture_type itself changes
                    if fdef.key == "texture_type" and hasattr(state.settings, '_sync_derived_fields'):
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


def _render_mesh_info(ctx: InxGUIContext, panel, state: _State):
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
# Sprite slice editor (SPRITE-mode custom body)
# ═══════════════════════════════════════════════════════════════════════════

class _SpriteEditorState:
    """Persistent state for the sprite slice editor (one at a time)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.file_path: str = ""
        self.tex_w: int = 0
        self.tex_h: int = 0
        self.texture_id: int = 0
        self.selected_frame: int = -1
        self.slice_rows: int = 1
        self.slice_cols: int = 1
        self.drag_edge: str = ""  # "", "left", "right", "top", "bottom"
        self.drag_frame_idx: int = -1


_sprite_state = _SpriteEditorState()


def _ensure_sprite_texture(state: _State) -> bool:
    """Load the texture dimensions + ImGui texture ID for the sprite editor."""
    from Infernux.debug import Debug

    ss = _sprite_state
    # Track sRGB + filter_mode so we re-upload when either changes
    cur_srgb = getattr(state.settings, 'srgb', False)
    cur_filter = getattr(state.settings, 'filter_mode', None)
    if (ss.file_path == state.file_path and ss.tex_w > 0
            and getattr(ss, '_srgb', None) == cur_srgb
            and getattr(ss, '_filter', None) == cur_filter):
        return True
    ss.reset()
    ss.file_path = state.file_path
    ss._srgb = cur_srgb
    ss._filter = cur_filter

    # Load texture data once
    try:
        from Infernux.lib import TextureLoader
        td = TextureLoader.load_from_file(state.file_path)
        if not td or td.width <= 0 or not td.is_valid():
            Debug.log_warning(f"[SpriteEditor] TextureLoader failed for: {state.file_path}")
            return False
        ss.tex_w = td.width
        ss.tex_h = td.height
    except Exception as exc:
        Debug.log_warning(f"[SpriteEditor] TextureLoader exception: {exc}")
        return False

    # Upload for ImGui preview
    try:
        from Infernux.engine.ui.editor_services import EditorServices
        svc = EditorServices.instance()
        native = svc.native_engine if svc else None
        if not native:
            Debug.log_warning("[SpriteEditor] No native engine via EditorServices")
            return ss.tex_w > 0

        filter_tag = cur_filter.name if cur_filter else "default"
        srgb_tag = "srgb" if cur_srgb else "linear"
        cache_name = (f"__sprite_preview__{srgb_tag}_{filter_tag}__"
                      f"{os.path.normpath(state.file_path)}")
        if native.has_imgui_texture(cache_name):
            ss.texture_id = native.get_imgui_texture_id(cache_name)
        else:
            pixels = td.get_pixels_list()
            w, h = td.width, td.height

            if cur_srgb:
                import array
                _LUT = [int(((i / 255.0) ** (1.0 / 2.2)) * 255.0 + 0.5)
                        for i in range(256)]
                pixels = list(array.array('B', (
                    _LUT[pixels[j]] if (j % 4) != 3 else pixels[j]
                    for j in range(len(pixels))
                )))

            # Simulate POINT filtering: nearest-neighbor 2x upscale so the
            # GPU's linear sampler can't blur adjacent pixels together.
            if cur_filter == FilterMode.POINT and w <= 1024 and h <= 1024:
                up_w, up_h = w * 2, h * 2
                upscaled = [0] * (up_w * up_h * 4)
                for sy in range(up_h):
                    src_y = sy // 2
                    for sx in range(up_w):
                        src_x = sx // 2
                        si = (src_y * w + src_x) * 4
                        di = (sy * up_w + sx) * 4
                        upscaled[di] = pixels[si]
                        upscaled[di + 1] = pixels[si + 1]
                        upscaled[di + 2] = pixels[si + 2]
                        upscaled[di + 3] = pixels[si + 3]
                pixels = upscaled
                w, h = up_w, up_h

            ss.texture_id = native.upload_texture_for_imgui(
                cache_name, pixels, w, h)
        if not ss.texture_id:
            Debug.log_warning(f"[SpriteEditor] upload_texture_for_imgui returned 0 for {cache_name}")
    except Exception as exc:
        Debug.log_warning(f"[SpriteEditor] Upload exception: {exc}")
        ss.texture_id = 0
    return ss.tex_w > 0


def _render_sprite_body(ctx: InxGUIContext, panel, state: _State):
    """Render the full SPRITE-mode inspector: import fields + slice editor."""
    from .inspector_utils import render_compact_section_header

    settings: TextureImportSettings = state.settings

    # ── Standard import fields (same as generic texture) ────────────────
    cat_def = _categories.get("texture")
    if cat_def and cat_def.editable_fields:
        _render_import_fields(ctx, cat_def, state)

    # ── Sprite slice editor ─────────────────────────────────────────────
    if not isinstance(settings, TextureImportSettings):
        return
    if settings.texture_type != TextureType.SPRITE:
        return

    ctx.separator()
    if not render_compact_section_header(ctx, t("sprite.slice_editor")):
        return

    if not _ensure_sprite_texture(state):
        ctx.push_style_color(ImGuiCol.Text, *Theme.WARNING_TEXT)
        ctx.label(t("sprite.cannot_load"))
        ctx.pop_style_color(1)
        return

    ss = _sprite_state
    labels = [t("sprite.image_size"), t("sprite.rows"), t("sprite.cols")]
    lw = max_label_w(ctx, labels)

    # Image dimensions (read-only)
    field_label(ctx, t("sprite.image_size"), lw)
    ctx.label(f"{ss.tex_w} x {ss.tex_h}")

    # Auto-slice controls
    field_label(ctx, t("sprite.rows"), lw)
    ctx.set_next_item_width(120)
    ss.slice_rows = max(1, ctx.input_int("##sprite_rows", ss.slice_rows, 1, 4))

    field_label(ctx, t("sprite.cols"), lw)
    ctx.set_next_item_width(120)
    ss.slice_cols = max(1, ctx.input_int("##sprite_cols", ss.slice_cols, 1, 4))

    ctx.button(t("sprite.auto_slice"), lambda: (_auto_slice(settings, ss),
                                                   _auto_save_sprite(state)))
    ctx.dummy(0, 4)

    # ── Visual preview with divider lines ────────────────────────────────
    _render_sprite_preview(ctx, settings, ss, state)


def _auto_slice(settings: TextureImportSettings, ss: _SpriteEditorState):
    """Generate uniform sprite_frames from rows × cols grid."""
    rows, cols = ss.slice_rows, ss.slice_cols
    if rows < 1 or cols < 1 or ss.tex_w < 1 or ss.tex_h < 1:
        return
    fw = ss.tex_w // cols
    fh = ss.tex_h // rows
    frames = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            frames.append(SpriteFrame(
                name=f"frame_{idx}",
                x=c * fw, y=r * fh,
                w=fw, h=fh,
            ))
            idx += 1
    settings.sprite_frames = frames
    ss.selected_frame = -1


def _auto_save_sprite(state: _State):
    """Persist sprite_frames to .meta immediately."""
    if state.exec_layer and state.settings:
        state.exec_layer.apply_import_settings(state.settings)
        if hasattr(state.settings, 'copy'):
            state.disk_settings = state.settings.copy()


def _collect_dividers(settings: TextureImportSettings,
                      tex_w: int, tex_h: int):
    """Extract unique vertical (X) and horizontal (Y) divider positions
    from sprite_frames, excluding image edges (0 and max)."""
    v_set: set[int] = set()
    h_set: set[int] = set()
    for f in settings.sprite_frames:
        v_set.add(f.x)
        v_set.add(f.x + f.w)
        h_set.add(f.y)
        h_set.add(f.y + f.h)
    # Remove image boundary values — they are implicit
    v_set.discard(0)
    v_set.discard(tex_w)
    h_set.discard(0)
    h_set.discard(tex_h)
    return sorted(v_set), sorted(h_set)


def _rebuild_frames_from_dividers(settings: TextureImportSettings,
                                  v_divs: list[int], h_divs: list[int],
                                  tex_w: int, tex_h: int):
    """Regenerate sprite_frames from the current divider positions."""
    xs = [0] + v_divs + [tex_w]
    ys = [0] + h_divs + [tex_h]
    frames = []
    idx = 0
    for ri in range(len(ys) - 1):
        for ci in range(len(xs) - 1):
            frames.append(SpriteFrame(
                name=f"frame_{idx}",
                x=xs[ci], y=ys[ri],
                w=xs[ci + 1] - xs[ci],
                h=ys[ri + 1] - ys[ri],
            ))
            idx += 1
    settings.sprite_frames = frames


def _render_sprite_preview(ctx: InxGUIContext, settings: TextureImportSettings,
                           ss: _SpriteEditorState, state: _State):
    """Draw the texture image with white divider lines overlaid."""
    avail_w = ctx.get_content_region_avail_width() - 8
    if avail_w < 32:
        return

    # Fit image into available width, maintain aspect ratio
    scale = min(avail_w / ss.tex_w, 600.0 / ss.tex_h) if ss.tex_h > 0 else 1.0
    draw_w = ss.tex_w * scale
    draw_h = ss.tex_h * scale

    # Render the texture image
    if ss.texture_id:
        ctx.image(ss.texture_id, draw_w, draw_h)
    else:
        cx = ctx.get_cursor_pos_x()
        cy = ctx.get_cursor_pos_y()
        wx = ctx.get_window_pos_x()
        wy = ctx.get_window_pos_y()
        sy = ctx.get_scroll_y()
        fb_x = wx + cx
        fb_y = wy + cy - sy
        ctx.invisible_button("##sprite_preview_fb", draw_w, draw_h)
        ctx.draw_filled_rect(fb_x, fb_y, fb_x + draw_w, fb_y + draw_h,
                             0.15, 0.15, 0.15, 1.0)

    # Screen coords of the image widget
    img_x = ctx.get_item_rect_min_x()
    img_y = ctx.get_item_rect_min_y()
    img_hovered = ctx.is_item_hovered()

    # Outer border
    ctx.draw_rect(img_x, img_y, img_x + draw_w, img_y + draw_h,
                  1.0, 1.0, 1.0, 0.6, 1.0)

    # Collect divider lines from sprite_frames
    v_divs, h_divs = _collect_dividers(settings, ss.tex_w, ss.tex_h)

    # Draw divider lines: thin shadow + white line
    for vx in v_divs:
        sx = img_x + vx * scale
        ctx.draw_line(sx, img_y, sx, img_y + draw_h,
                      0.0, 0.0, 0.0, 0.4, 3.0)
        ctx.draw_line(sx, img_y, sx, img_y + draw_h,
                      1.0, 1.0, 1.0, 0.9, 1.0)

    for hy in h_divs:
        sy_line = img_y + hy * scale
        ctx.draw_line(img_x, sy_line, img_x + draw_w, sy_line,
                      0.0, 0.0, 0.0, 0.4, 3.0)
        ctx.draw_line(img_x, sy_line, img_x + draw_w, sy_line,
                      1.0, 1.0, 1.0, 0.9, 1.0)

    # ── Interaction: click near a line to start dragging it ─────────────
    _GRAB_THRESHOLD = 5.0  # pixels

    if img_hovered:
        mx = ctx.get_mouse_pos_x() - img_x
        my = ctx.get_mouse_pos_y() - img_y

        if ctx.is_mouse_button_clicked(0):
            ss.drag_edge = ""
            ss.drag_frame_idx = -1
            # Check vertical dividers
            for i, vx in enumerate(v_divs):
                if abs(mx - vx * scale) < _GRAB_THRESHOLD:
                    ss.drag_edge = "v"
                    ss.drag_frame_idx = i  # index into v_divs
                    break
            else:
                # Check horizontal dividers
                for i, hy in enumerate(h_divs):
                    if abs(my - hy * scale) < _GRAB_THRESHOLD:
                        ss.drag_edge = "h"
                        ss.drag_frame_idx = i  # index into h_divs
                        break

        # Drag a divider line
        if ctx.is_mouse_dragging(0) and ss.drag_edge and ss.drag_frame_idx >= 0:
            if ss.drag_edge == "v" and ss.drag_frame_idx < len(v_divs):
                new_px = max(1, min(ss.tex_w - 1, round(mx / scale)))
                v_divs[ss.drag_frame_idx] = new_px
                v_divs.sort()
                _rebuild_frames_from_dividers(settings, v_divs, h_divs,
                                              ss.tex_w, ss.tex_h)
                try:
                    ss.drag_frame_idx = v_divs.index(new_px)
                except ValueError:
                    ss.drag_frame_idx = -1
            elif ss.drag_edge == "h" and ss.drag_frame_idx < len(h_divs):
                new_py = max(1, min(ss.tex_h - 1, round(my / scale)))
                h_divs[ss.drag_frame_idx] = new_py
                h_divs.sort()
                _rebuild_frames_from_dividers(settings, v_divs=v_divs,
                                              h_divs=h_divs,
                                              tex_w=ss.tex_w, tex_h=ss.tex_h)
                try:
                    ss.drag_frame_idx = h_divs.index(new_py)
                except ValueError:
                    ss.drag_frame_idx = -1

    if not ctx.is_mouse_button_down(0):
        # Save on drag release
        if ss.drag_edge:
            _auto_save_sprite(state)
        ss.drag_edge = ""
        ss.drag_frame_idx = -1


# _render_frame_table removed — divider lines replace per-frame editing


# ═══════════════════════════════════════════════════════════════════════════
# Texture — preview section (custom_header_fn)
# ═══════════════════════════════════════════════════════════════════════════

_PREVIEW_MIN_H = 60.0
_PREVIEW_MAX_H = 800.0
_SPLITTER_H = 14.0


def _render_texture_preview(ctx: InxGUIContext, panel, state: _State):
    """Render a safe placeholder instead of the native texture preview.

    The native preview path is temporarily disabled while investigating asset
    selection crashes that reproduce reliably on texture clicks.
    """
    ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
    ctx.label("Texture preview temporarily disabled in safe mode.")
    ctx.pop_style_color(1)
    ctx.separator()


# ═══════════════════════════════════════════════════════════════════════════
# Shader — custom body (path editing + source preview)
# ═══════════════════════════════════════════════════════════════════════════


def _render_shader_body(ctx: InxGUIContext, panel, state: _State):
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


def _render_font_body(ctx: InxGUIContext, panel, state: _State):
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
    from Infernux.core.shader import Shader
    shader_id = os.path.splitext(os.path.basename(old_path))[0]
    Shader.invalidate(shader_id)
    info.source_path = new_path
    info.shader_type = ShaderAssetInfo.from_path(new_path).shader_type


def _render_shader_source(ctx: InxGUIContext, file_path: str):
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


def _render_material_body(ctx: InxGUIContext, panel, state: _State):
    from . import inspector_material as mat_ui
    mat_ui.render_material_body(ctx, panel, state)
