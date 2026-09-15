"""Unified UI element rendering dispatch.

Provides a registry-based pattern so that adding a new UI component type
(e.g. UIButton) only requires registering one renderer per back-end,
rather than scattering ``isinstance`` chains across multiple panels.

Two back-ends:
- **editor**: ImGui draw-list commands via ``InxGUIContext`` (UI Editor panel).
- **runtime**: GPU ScreenUI renderer commands (Game View panel).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from Infernux.ui.enums import TextAlignH, TextAlignV, TextOverflow
from Infernux.ui.ui_render_revision import (
    get_runtime_ui_revision, mark_runtime_ui_dirty, _get_shared_visual_revision,
    _get_resource_binding_revision, _invalidate_ui_command_groups,
)


# ── Shared attribute helpers ─────────────────────────────────────────

def _pad_rgba(color, default=None) -> list:
    """Ensure *color* is a 4-element RGBA list, padding missing channels with 1.0."""
    if not color:
        if default is not None:
            return list(default)
        return [1.0, 1.0, 1.0, 1.0]
    color = list(color)
    while len(color) < 4:
        color.append(1.0)
    return color


def _multiply_rgba(left, right) -> list:
    return [float(left[index]) * float(right[index]) for index in range(4)]


def _visual_material(elem, slot):
    material = getattr(elem, slot, None)
    if material is None:
        return None
    native = getattr(material, "native", material)
    if native._texture_assets_pending:
        from Infernux.application import Application

        engine = Application._current_engine()
        if engine is not None:
            engine.get_native_engine()._prepare_material_texture_assets(native)
    return native


def _material_signature(native, runtime_texture):
    identity = str(getattr(native, "guid", "") or "") or id(native)
    return (identity, id(native), int(native.get_version()),
            runtime_texture.revision if runtime_texture is not None else 0)


def material_visual_revision(elem, slot: str = "material"):
    """Inspect dependencies without reading colors or resolving texture paths.

    Material references are still resolved here so a reimport replacing the
    native owner is observed in the same frame. Full draw data is read by
    command construction, not by this dependency check.
    """
    native = _visual_material(elem, slot)
    if native is None:
        return None, 0
    return _material_signature(native, native._get_render_texture("texSampler"))


def material_visual_state(elem, slot: str = "material") -> dict:
    """Resolve the small, shared UI material contract from a normal .mat asset.

    UI geometry keeps using the dedicated screen/world vertex path. The
    material owns its visual parameters through the same asset/GUID system as
    every other renderer: ``baseColor`` tints vertices and ``texSampler`` may
    provide the sampled texture. An empty slot means the engine UI material
    (white tint and no authored texture).
    """
    native = _visual_material(elem, slot)
    if native is None:
        return {
            "color": [1.0, 1.0, 1.0, 1.0],
            "texture_path": "",
            "texture": None,
            "signature": (None, 0),
        }
    runtime_texture = native._get_render_texture('texSampler')
    signature = _material_signature(native, runtime_texture)
    color = (
        list(native.get_color("baseColor"))
        if native.has_property("baseColor")
        else [1.0, 1.0, 1.0, 1.0]
    )
    texture_guid = (
        str(native.get_texture("texSampler") or "")
        if runtime_texture is None and native.has_property("texSampler")
        else ""
    )
    texture_path = ""
    if texture_guid:
        from Infernux.core.assets import AssetManager

        texture_path = str(
            AssetManager.require_asset_database().get_path_from_guid(texture_guid) or ""
        )
    return {
        "color": _pad_rgba(color),
        "texture_path": texture_path,
        "texture": runtime_texture,
        "signature": signature,
    }


def image_texture_source(elem, material_state):
    if hasattr(elem, '_image_texture_source'):
        source = elem._image_texture_source()
        # Missing GUIDs do not turn path hints into a second loading route.
        return source if source is not None else (material_state.get('texture') or
                                                   material_state['texture_path'] or '')
    source = getattr(elem, 'texture', None)
    if source is not None and not isinstance(source, str):
        return source
    return (source or material_state.get('texture') or
            material_state['texture_path'] or getattr(elem, 'texture_path', '') or '')


def extract_common(
    elem,
    *,
    world_transform_owned: bool = False,
    material_state: dict | None = None,
) -> dict:
    """Extract shared visual attributes from any InxUIScreenComponent."""
    material_state = material_state or material_visual_state(elem)
    color = _multiply_rgba(
        _pad_rgba(getattr(elem, "color", None)), material_state["color"]
    )
    opacity = max(0.0, min(1.0, float(getattr(elem, "opacity", 1.0))))
    group_state = getattr(elem, "get_effective_group_state", None)
    if callable(group_state):
        opacity *= group_state()[0]
    return {
        "color": color,
        "opacity": opacity,
        # Canvas-free UI is submitted as ordinary scene geometry: its full
        # rotation is already present in the element's world matrix.
        "rotation": 0.0 if world_transform_owned else float(elem.get_layout_rotation()),
        "mirror_h": bool(getattr(elem, "mirror_x", False)),
        "mirror_v": bool(getattr(elem, "mirror_y", False)),
        "corner_radius": float(getattr(elem, "corner_radius", 0.0)),
        "material_texture_path": material_state["texture_path"],
        "material_signature": material_state["signature"],
    }


def _extract_text_attrs(elem, scale: float = 1.0) -> dict:
    """Extract text-rendering attributes from a text-bearing element.

    *scale* is applied to letter_spacing (zoom for editor, text_scale for
    runtime).  font_size is returned raw (caller applies own scaling).
    """
    ah = getattr(elem, "text_align_h", TextAlignH.Left)
    av = getattr(elem, "text_align_v", TextAlignV.Top)
    ax, ay = text_align_to_float(ah, av)
    font_path = _resolve_font_asset_path(getattr(elem, "font_path", ""))
    fallback_font_paths = [
        _resolve_font_asset_path(path)
        for path in (getattr(elem, "fallback_font_paths", None) or ())
    ]
    return {
        "font_path": font_path,
        "fallback_font_paths": fallback_font_paths,
        "font_size": float(elem.font_size),
        "line_height": float(elem.line_height),
        "letter_spacing": float(elem.letter_spacing) * scale,
        "align_x": ax,
        "align_y": ay,
    }


def _resolve_font_asset_path(path) -> str:
    """Resolve one authored font alias through the active asset catalog.

    Editor projects resolve ``Assets/...`` and ``Packages/...`` against their
    source roots. Players resolve the same authoring alias to the extracted
    GUID-owned blob from ``Content.inxpkg``. Absolute tool/test paths outside
    a project are retained for the editor only; a packaged scene never emits
    those because the cook rewrites project-owned absolute paths.
    """
    authored = str(path or "")
    if not authored:
        return ""
    from Infernux.engine.project_context import resolve_asset_path

    return resolve_asset_path(authored) or authored


def text_align_to_float(align_h, align_v) -> tuple[float, float]:
    """Convert TextAlignH/V enums to (0.0/0.5/1.0) floats."""
    ax = 0.0 if align_h == TextAlignH.Left else (0.5 if align_h == TextAlignH.Center else 1.0)
    ay = 0.0 if align_v == TextAlignV.Top else (0.5 if align_v == TextAlignV.Center else 1.0)
    return ax, ay


def resolve_text_layout(elem, measure_text, scale: float = 1.0) -> bool:
    """Resolve a text element's intrinsic box through the shared layout path."""
    resolver = getattr(elem, "resolve_text_layout", None)
    return bool(resolver(measure_text, scale)) if callable(resolver) else False


# ── Registry ─────────────────────────────────────────────────────────

# Keyed by (component class name, backend_name)
_RENDERERS: Dict[tuple[str, str], Callable] = {}
_RESOLVED_RENDERERS: Dict[tuple[type, str], Optional[Callable]] = {}
_REVISION_MASK = (1 << 64) - 1


class _UICommandDependencies:
    """Retain topology once; inspect poses natively and only bound resources.

    Visual setters already publish the shared UI revision. Native scene
    structure covers membership, activation and component enablement. Geometry
    observes native Transform storage, including parent motion and physics;
    no Python setter interception is needed for native-side edits.
    """

    def __init__(self):
        self.key = None
        self.generation = 0
        self.geometry = None
        self.geometry_revision = None
        self.elements = ()
        self.bindings_revision = None
        self.materials = ()
        self.images = ()
        self.resource_states = None
        self.snapshot = None
        self.element_resources = {}

    def revision(self, key, canvases, world_elements):
        if key != self.key:
            from Infernux.lib._Infernux import _UITransformDependencies

            screen = tuple(element for canvas in canvases for element in canvas._get_elements())
            self.geometry = _UITransformDependencies(
                [element.game_object for element in screen],
                [element.game_object for element in world_elements],
            )
            self.elements = screen + world_elements
            self.screen_count = len(screen)
            self.screen_children = [[] for _ in screen]
            screen_owners = {element.game_object.id: index for index, element in enumerate(screen)}
            # A screen rect is relative to its nearest UI ancestor, including
            # through ordinary scene nodes. This membership-only graph avoids
            # walking every hierarchy again when one Transform moves.
            for index, element in enumerate(screen):
                parent = element.game_object.get_parent()
                while parent is not None:
                    parent_index = screen_owners.get(parent.id)
                    if parent_index is not None:
                        self.screen_children[parent_index].append(index)
                        break
                    parent = parent.get_parent()
            self.key = key
            self.generation += 1
            self.geometry_revision = None
            self.bindings_revision = None
        geometry_revision = self.geometry.poll()
        if geometry_revision != self.geometry_revision:
            affected = set()
            # World packets retain local geometry and sample their native
            # Transform/layer when appended. Only screen poses change the
            # authored 2D geometry (including dependent child layout).
            pending = [index for index in self.geometry.changed_entries
                       if index < self.screen_count]
            while pending:
                index = pending.pop()
                if index in affected:
                    continue
                affected.add(index)
                if index < self.screen_count:
                    pending.extend(self.screen_children[index])
            if any(index < self.screen_count for index in affected):
                from .inx_ui_screen_component import clear_rect_cache

                # Child layout depends on ancestor poses, even when the
                # child's own local Transform signature is unchanged.
                clear_rect_cache((id(self), self.generation, geometry_revision))
            for index in affected:
                _invalidate_ui_command_groups(self.elements[index])
            self.geometry_revision = geometry_revision
        bindings_revision = _get_resource_binding_revision()
        if bindings_revision != self.bindings_revision:
            materials = []
            images = []
            for element in self.elements:
                for slot in ("material", "text_material"):
                    field = getattr(type(element), slot, None)
                    # Retain unresolved authored references too: publication
                    # must be noticed without requiring another assignment.
                    reference = field.get_raw(element) if field is not None else None
                    if reference is not None and (reference.guid or reference._cached is not None):
                        materials.append((element, slot, reference))
                texture = getattr(type(element), "texture", None)
                if texture is not None and (
                    bool(texture.get_raw(element)) or
                    element.__dict__.get("_render_texture") is not None
                ):
                    images.append(element)
            self.materials = tuple(materials)
            self.images = tuple(images)
            self.bindings_revision = bindings_revision
            self.resource_states = None
        image_revisions = []
        for element in self.images:
            source = element._image_texture_source()
            revision = getattr(source, "revision", None)
            state = (id(source), revision) if revision is not None else None
            image_revisions.append(state)
        # Inspect each shared material once in this publication. Retain normal
        # reference resolution (including unresolved GUIDs), but do not read
        # the same native version/texture for every control that uses it.
        material_revisions = {}
        material_states = []
        for element, slot, reference in self.materials:
            owner = reference._cached
            identity = ("owner", id(owner)) if owner is not None else ("guid", reference.guid)
            if identity not in material_revisions:
                material_revisions[identity] = material_visual_revision(element, slot)
            material_states.append(material_revisions[identity])
        resource_states = tuple(material_states), tuple(image_revisions)
        if resource_states != self.resource_states:
            resources = {element: [None, None, state]
                         for element, state in zip(self.images, image_revisions)}
            for (element, slot, _), state in zip(self.materials, material_states):
                resources.setdefault(element, [None, None, None])[slot == "text_material"] = state
            current_resources = {element: tuple(state) for element, state in resources.items()}
            for element in self.element_resources.keys() | current_resources.keys():
                if self.element_resources.get(element) != current_resources.get(element):
                    _invalidate_ui_command_groups(element)
            self.element_resources = current_resources
            self.resource_states = resource_states
        self.snapshot = (
            self.generation,
            geometry_revision,
            # A resolved reference and an unresolved alias can share one
            # material; lazy resolution must not change the dependency set.
            tuple(dict.fromkeys(material_revisions.values())),
            tuple(image_revisions),
        )
        return self.snapshot


_runtime_dependencies = _UICommandDependencies()


def _runtime_command_epoch():
    """Only inherited/topology state invalidates every element's packet."""
    return (_runtime_dependencies.generation, _get_shared_visual_revision())


def runtime_ui_revision(scene, canvases, width: int, height: int,
                        texture_generation: int = 0, world_elements=(),
                        persistent_scene=None) -> int:
    """Build the content revision for native command-list reuse.

    UI components increment a shared generation on visual mutation. Scene
    structure covers hierarchy and active-state changes. Transform, material
    and RenderTexture generations also invalidate already-recorded commands.
    """
    canvases = tuple(canvases)
    world_elements = tuple(world_elements)
    visual_revision = get_runtime_ui_revision()
    membership = (
        id(scene),
        int(getattr(scene, "world_id", 0)),
        int(getattr(scene, "structure_version", 0)),
        int(getattr(scene, "temporal_discontinuity_revision", 0)),
        id(persistent_scene),
        int(getattr(persistent_scene, "world_id", 0)),
        int(getattr(persistent_scene, "structure_version", 0)),
        int(getattr(persistent_scene, "temporal_discontinuity_revision", 0)),
        canvases, world_elements,
    )
    dependencies = _runtime_dependencies.revision(
        membership, canvases, world_elements,
    )
    from .inx_ui_screen_component import _get_layout_revision

    signature = hash((
        int(width),
        int(height),
        int(texture_generation),
        visual_revision,
        _get_layout_revision(),
        dependencies,
    )) & _REVISION_MASK
    return signature or 1


def register_ui_renderer(component_cls_name: str, backend: str, fn: Callable):
    """Register a renderer function for a UI component type.

    Args:
        component_cls_name: e.g. ``"UIText"``, ``"UIImage"``.
        backend: ``"editor"`` or ``"runtime"``.
        fn: Callable with backend-specific signature (see below).
    """
    _RENDERERS[(component_cls_name, backend)] = fn
    _RESOLVED_RENDERERS.clear()
    mark_runtime_ui_dirty()


def get_ui_renderer(component_cls_name: str, backend: str) -> Optional[Callable]:
    """Look up a registered renderer."""
    return _RENDERERS.get((component_cls_name, backend))


def _resolve_renderer(elem_type: type, backend: str) -> Optional[Callable]:
    cache_key = (elem_type, backend)
    cached = _RESOLVED_RENDERERS.get(cache_key, None)
    if cache_key in _RESOLVED_RENDERERS:
        return cached

    fn = _RENDERERS.get((elem_type.__name__, backend))
    if fn is None:
        for base in elem_type.__mro__[1:]:
            fn = _RENDERERS.get((base.__name__, backend))
            if fn is not None:
                break

    _RESOLVED_RENDERERS[cache_key] = fn
    return fn


def dispatch(elem, backend: str, **kwargs):
    """Dispatch rendering of *elem* to the registered handler.

    Returns True if a handler was found and called, False otherwise.
    """
    fn = _resolve_renderer(type(elem), backend)
    if fn is None:
        return False
    fn(elem, **kwargs)
    return True


# ══════════════════════════════════════════════════════════════════════
#  Built-in renderers — EDITOR back-end (ImGui draw-list)
# ══════════════════════════════════════════════════════════════════════

def _editor_render_text(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, get_tex_id, **_kw):
    """Render a UIText element in the UI Editor panel."""
    attrs = extract_common(elem)
    color = attrs["color"]
    ta = _extract_text_attrs(elem, scale=zoom)
    clip = getattr(elem, "overflow", TextOverflow.Visible) != TextOverflow.Visible
    text_size = max(1.0, ta["font_size"] * zoom)
    editor_wrap_width = (
        0.0 if getattr(elem, "is_auto_width", lambda: False)() else base_sw
    )
    arguments = (
        base_sx, base_sy, base_sx + base_sw, base_sy + base_sh,
        elem.text,
        color[0], color[1], color[2], color[3] * attrs["opacity"],
        ta["align_x"], ta["align_y"], text_size,
        editor_wrap_width,
        attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"], clip,
        ta["font_path"], ta["line_height"], ta["letter_spacing"],
    )
    ctx.draw_text_ex_aligned(
        *arguments,
        *([ta["fallback_font_paths"]] if ta["fallback_font_paths"] else []),
    )


def _editor_render_image(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, get_tex_id, **_kw):
    """Render a UIImage element in the UI Editor panel."""
    attrs = extract_common(elem)
    color = attrs["color"]
    cr, cg, cb = color[0], color[1], color[2]
    ca = color[3] * attrs["opacity"]
    tex_path = image_texture_source(elem, material_visual_state(elem))
    tex_id = get_tex_id(tex_path) if tex_path else 0
    rounding = attrs["corner_radius"] * zoom
    if tex_id:
        ctx.draw_image_rect(
            tex_id, base_sx, base_sy,
            base_sx + base_sw, base_sy + base_sh,
            0.0, 0.0, 1.0, 1.0,
            cr, cg, cb, ca,
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            rounding,
        )
    else:
        _draw_editor_placeholder(ctx, base_sx, base_sy, base_sw, base_sh,
                                 cr, cg, cb, ca, rounding, attrs["corner_radius"] * zoom,
                                 attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"])

def _draw_editor_placeholder(ctx, x, y, w, h, cr, cg, cb, ca, rounding, rect_rounding, rotation=0.0, mirror_h=False, mirror_v=False):
    """Draw a placeholder rect with tint + cross pattern (editor backend)."""
    # Editor styling is an editor-only capability.  Keeping this import at the
    # editor rendering boundary prevents Player/Web runtime UI from loading the
    # native editor theme registry.
    from Infernux.engine.ui.theme import Theme

    tint = Theme.UI_EDITOR_PLACEHOLDER_TINT
    alpha = Theme.UI_EDITOR_PLACEHOLDER_ALPHA
    ctx.draw_filled_rect_rotated(x, y, x + w, y + h,
                         cr * tint, cg * tint, cb * tint, ca * alpha,
                         rotation, mirror_h, mirror_v, rounding)
    ctx.draw_rect(x, y, x + w, y + h, 0.5, 0.5, 0.5, 0.5, 1.0, rect_rounding)
    ctx.draw_line(x, y, x + w, y + h, 0.5, 0.5, 0.5, 0.3, 1.0)
    ctx.draw_line(x + w, y, x, y + h, 0.5, 0.5, 0.5, 0.3, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Built-in renderers — RUNTIME back-end (GPU ScreenUI renderer)
# ══════════════════════════════════════════════════════════════════════

def _runtime_render_text(elem, renderer, ui_list, sx, sy, sw, sh,
                         ref_w, ref_h, scale_x, scale_y, text_scale, get_tex_id, **_kw):
    """Render a UIText element via the GPU ScreenUI renderer."""
    world_transform_owned = bool(_kw.get("world_transform_owned", False))
    material_state = material_visual_state(elem)

    attrs = extract_common(
        elem,
        world_transform_owned=world_transform_owned,
        material_state=material_state,
    )
    color = attrs["color"]
    ta = _extract_text_attrs(elem, scale=text_scale)
    font_size_raw = ta["font_size"]

    wrap_width = float(elem.get_wrap_width()) if hasattr(elem, "get_wrap_width") else 0.0
    scaled_wrap_width = 0.0 if wrap_width <= 0.0 else wrap_width * text_scale

    font_size = max(1.0, font_size_raw * text_scale)
    ca = color[3] * attrs["opacity"]
    arguments = (
        ui_list,
        sx, sy, sx + sw, sy + sh,
        elem.text,
        color[0], color[1], color[2], ca,
        ta["align_x"], ta["align_y"], font_size,
        0.0 if getattr(elem, "is_auto_width", lambda: False)() else scaled_wrap_width,
        attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
        ta["font_path"], ta["line_height"], ta["letter_spacing"],
        getattr(elem, "overflow", TextOverflow.Visible) != TextOverflow.Visible,
    )
    if ta["fallback_font_paths"]:
        arguments += (ta["fallback_font_paths"],)
    renderer.add_text(*arguments)


def _runtime_render_image(elem, renderer, ui_list, sx, sy, sw, sh,
                          scale_x, scale_y, get_tex_id, **_kw):
    """Render a UIImage element via the GPU ScreenUI renderer."""
    material_state = material_visual_state(elem)
    tex_path = image_texture_source(elem, material_state)
    tex_id = get_tex_id(tex_path) if tex_path else 0
    world_transform_owned = bool(_kw.get("world_transform_owned", False))

    attrs = extract_common(
        elem,
        world_transform_owned=world_transform_owned,
        material_state=material_state,
    )
    color = attrs["color"]
    cr, cg, cb = color[0], color[1], color[2]
    ca = color[3] * attrs["opacity"]
    rounding = attrs["corner_radius"] * min(scale_x, scale_y)
    if tex_id:
        kind = "image"
        arguments = (
            ui_list, tex_id,
            sx, sy, sx + sw, sy + sh,
            0.0, 0.0, 1.0, 1.0,
            cr, cg, cb, ca,
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            rounding,
        )
    else:
        kind = "rect"
        arguments = (
            ui_list,
            sx, sy, sx + sw, sy + sh,
            cr, cg, cb, ca,
            rounding,
        )
    if kind == "image":
        renderer.add_image(*arguments)
    else:
        renderer.add_filled_rect(*arguments)


# ── Register built-in renderers ──────────────────────────────────────

register_ui_renderer("UIText", "editor", _editor_render_text)
register_ui_renderer("UIImage", "editor", _editor_render_image)
register_ui_renderer("UIText", "runtime", _runtime_render_text)
register_ui_renderer("UIImage", "runtime", _runtime_render_image)


# ══════════════════════════════════════════════════════════════════════
#  UIButton renderers — shared helpers + per-backend glue
# ══════════════════════════════════════════════════════════════════════

def _get_button_bg(elem, material_state=None):
    """Return the button's background colour as a 4-element list."""
    material_state = material_state or material_visual_state(elem)
    background = _pad_rgba(elem.background_color)
    tint = _pad_rgba(elem.get_current_tint())
    return _multiply_rgba(_multiply_rgba(background, tint), material_state["color"])


def _get_label_attrs(elem, scale: float, material_state=None):
    """Return (label, color, text_attrs) for a button's label text."""
    label = elem.label or ""
    material_state = material_state or material_visual_state(elem, "text_material")
    lc = _multiply_rgba(_pad_rgba(elem.label_color), material_state["color"])
    # Buttons default to Center/Center alignment
    ah = getattr(elem, "text_align_h", TextAlignH.Center)
    av = getattr(elem, "text_align_v", TextAlignV.Center)
    ax, ay = text_align_to_float(ah, av)
    ta = {
        "font_path": _resolve_font_asset_path(getattr(elem, "font_path", "")),
        "fallback_font_paths": [
            _resolve_font_asset_path(path)
            for path in (getattr(elem, "fallback_font_paths", None) or ())
        ],
        "font_size": float(elem.font_size),
        "line_height": float(elem.line_height),
        "letter_spacing": float(elem.letter_spacing) * scale,
        "align_x": ax,
        "align_y": ay,
    }
    return label, lc, ta


def _editor_render_button(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, get_tex_id, **_kw):
    """Render a UIButton element in the UI Editor panel."""
    background_material = material_visual_state(elem)
    text_material = material_visual_state(elem, "text_material")
    attrs = extract_common(elem, material_state=background_material)
    bg = _get_button_bg(elem, background_material)
    rounding = attrs["corner_radius"] * zoom

    # Background: texture image or solid fill
    tex_path = image_texture_source(elem, background_material)
    tex_id = get_tex_id(tex_path) if tex_path else 0
    if tex_id:
        ctx.draw_image_rect(
            tex_id,
            base_sx, base_sy, base_sx + base_sw, base_sy + base_sh,
            0.0, 0.0, 1.0, 1.0,
            bg[0], bg[1], bg[2], bg[3] * attrs["opacity"],
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            rounding,
        )
    else:
        ctx.draw_filled_rect_rotated(
            base_sx, base_sy, base_sx + base_sw, base_sy + base_sh,
            bg[0], bg[1], bg[2], bg[3] * attrs["opacity"],
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            rounding,
        )

    # Label
    label, lc, ta = _get_label_attrs(elem, scale=zoom, material_state=text_material)
    if label:
        font_size = max(1.0, ta["font_size"] * zoom)
        arguments = (
            base_sx, base_sy, base_sx + base_sw, base_sy + base_sh,
            label,
            lc[0], lc[1], lc[2], lc[3] * attrs["opacity"],
            ta["align_x"], ta["align_y"], font_size,
            base_sw,
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"], False,
            ta["font_path"], ta["line_height"], ta["letter_spacing"],
        )
        ctx.draw_text_ex_aligned(
            *arguments,
            *([ta["fallback_font_paths"]] if ta["fallback_font_paths"] else []),
        )


def _runtime_render_button(elem, renderer, ui_list, sx, sy, sw, sh,
                           scale_x, scale_y, text_scale, get_tex_id, **_kw):
    """Render a UIButton element via the GPU ScreenUI renderer."""
    background_material = material_visual_state(elem)
    text_material = material_visual_state(elem, "text_material")
    tex_path = image_texture_source(elem, background_material)
    tex_id = get_tex_id(tex_path) if tex_path else 0
    world_transform_owned = bool(_kw.get("world_transform_owned", False))

    attrs = extract_common(
        elem,
        world_transform_owned=world_transform_owned,
        material_state=background_material,
    )
    bg = _get_button_bg(elem, background_material)
    r, g, b = bg[0], bg[1], bg[2]
    a = bg[3] * attrs["opacity"]
    rounding = attrs["corner_radius"] * min(scale_x, scale_y)

    # Background: texture image or solid fill
    commands = []
    if tex_id:
        arguments = (
            ui_list, tex_id,
            sx, sy, sx + sw, sy + sh,
            0.0, 0.0, 1.0, 1.0,
            r, g, b, a,
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            rounding,
        )
        commands.append(("image", arguments))
    else:
        commands.append(("rect", (ui_list, sx, sy, sx + sw, sy + sh, r, g, b, a, rounding)))

    # Label
    label, lc, ta = _get_label_attrs(
        elem, scale=text_scale, material_state=text_material
    )
    if label:
        font_size = max(1.0, ta["font_size"] * text_scale)
        text_arguments = (
            ui_list,
            sx, sy, sx + sw, sy + sh,
            label,
            lc[0], lc[1], lc[2], lc[3] * attrs["opacity"],
            ta["align_x"], ta["align_y"], font_size,
            sw,
            attrs["rotation"], attrs["mirror_h"], attrs["mirror_v"],
            ta["font_path"], ta["line_height"], ta["letter_spacing"],
        )
        if ta["fallback_font_paths"]:
            text_arguments += (False, ta["fallback_font_paths"])
        commands.append(("text", text_arguments))

    for kind, arguments in commands:
        if kind == "image":
            renderer.add_image(*arguments)
        elif kind == "rect":
            renderer.add_filled_rect(*arguments)
        else:
            renderer.add_text(*arguments)


register_ui_renderer("UIButton", "editor", _editor_render_button)
register_ui_renderer("UIButton", "runtime", _runtime_render_button)


def _value_fill_rect(elem, x, y, width, height):
    fraction = float(elem.normalized_value)
    direction = elem.fill_direction
    from Infernux.ui.enums import UIFillDirection
    if direction == UIFillDirection.RightToLeft:
        fill_width = width * fraction
        return x + width - fill_width, y, fill_width, height
    if direction == UIFillDirection.BottomToTop:
        fill_height = height * fraction
        return x, y + height - fill_height, width, fill_height
    if direction == UIFillDirection.TopToBottom:
        return x, y, width, height * fraction
    return x, y, width * fraction, height


def _editor_render_progress(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, **_kw):
    attrs = extract_common(elem)
    background = _multiply_rgba(_pad_rgba(elem.background_color), attrs["color"])
    fill = _multiply_rgba(_pad_rgba(elem.fill_color), attrs["color"])
    rounding = attrs["corner_radius"] * zoom
    ctx.draw_filled_rect(
        base_sx, base_sy, base_sx + base_sw, base_sy + base_sh,
        background[0], background[1], background[2],
        background[3] * attrs["opacity"], rounding,
    )
    x, y, width, height = _value_fill_rect(elem, base_sx, base_sy, base_sw, base_sh)
    if width > 0.0 and height > 0.0:
        ctx.draw_filled_rect(
            x, y, x + width, y + height,
            fill[0], fill[1], fill[2], fill[3] * attrs["opacity"], rounding,
        )


def _runtime_render_progress(elem, renderer, ui_list, sx, sy, sw, sh,
                             scale_x, scale_y, **_kw):
    attrs = extract_common(
        elem,
        world_transform_owned=bool(_kw.get("world_transform_owned", False)),
    )
    background = _multiply_rgba(_pad_rgba(elem.background_color), attrs["color"])
    fill = _multiply_rgba(_pad_rgba(elem.fill_color), attrs["color"])
    rounding = attrs["corner_radius"] * min(scale_x, scale_y)
    renderer.add_filled_rect(
        ui_list, sx, sy, sx + sw, sy + sh,
        background[0], background[1], background[2],
        background[3] * attrs["opacity"], rounding,
    )
    x, y, width, height = _value_fill_rect(elem, sx, sy, sw, sh)
    if width > 0.0 and height > 0.0:
        renderer.add_filled_rect(
            ui_list, x, y, x + width, y + height,
            fill[0], fill[1], fill[2], fill[3] * attrs["opacity"], rounding,
        )


def _editor_render_slider(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, **kwargs):
    _editor_render_progress(elem, ctx, base_sx, base_sy, base_sw, base_sh, zoom, **kwargs)
    attrs = extract_common(elem)
    handle = _multiply_rgba(_pad_rgba(elem.handle_color), attrs["color"])
    x, y, width, height = _value_fill_rect(elem, base_sx, base_sy, base_sw, base_sh)
    from Infernux.ui.enums import UIFillDirection
    size = float(elem.handle_size) * zoom
    if elem.fill_direction in (UIFillDirection.LeftToRight, UIFillDirection.RightToLeft):
        center_x = x + width if elem.fill_direction == UIFillDirection.LeftToRight else x
        center_y = base_sy + base_sh * 0.5
    else:
        center_x = base_sx + base_sw * 0.5
        center_y = y if elem.fill_direction == UIFillDirection.BottomToTop else y + height
    ctx.draw_filled_rect(
        center_x - size * 0.5, center_y - size * 0.5,
        center_x + size * 0.5, center_y + size * 0.5,
        handle[0], handle[1], handle[2], handle[3] * attrs["opacity"], size * 0.5,
    )


def _runtime_render_slider(elem, renderer, ui_list, sx, sy, sw, sh,
                           scale_x, scale_y, **kwargs):
    _runtime_render_progress(
        elem, renderer, ui_list, sx, sy, sw, sh, scale_x, scale_y, **kwargs
    )
    attrs = extract_common(
        elem,
        world_transform_owned=bool(kwargs.get("world_transform_owned", False)),
    )
    handle = _multiply_rgba(_pad_rgba(elem.handle_color), attrs["color"])
    x, y, width, height = _value_fill_rect(elem, sx, sy, sw, sh)
    from Infernux.ui.enums import UIFillDirection
    size = float(elem.handle_size) * min(scale_x, scale_y)
    if elem.fill_direction in (UIFillDirection.LeftToRight, UIFillDirection.RightToLeft):
        center_x = x + width if elem.fill_direction == UIFillDirection.LeftToRight else x
        center_y = sy + sh * 0.5
    else:
        center_x = sx + sw * 0.5
        center_y = y if elem.fill_direction == UIFillDirection.BottomToTop else y + height
    renderer.add_filled_rect(
        ui_list,
        center_x - size * 0.5, center_y - size * 0.5,
        center_x + size * 0.5, center_y + size * 0.5,
        handle[0], handle[1], handle[2], handle[3] * attrs["opacity"], size * 0.5,
    )


def _render_visual_neutral_container(_elem, **_kwargs):
    return None


register_ui_renderer("UIFrame", "editor", _render_visual_neutral_container)
register_ui_renderer("UIFrame", "runtime", _render_visual_neutral_container)
register_ui_renderer("UIProgressBar", "editor", _editor_render_progress)
register_ui_renderer("UIProgressBar", "runtime", _runtime_render_progress)
register_ui_renderer("UISlider", "editor", _editor_render_slider)
register_ui_renderer("UISlider", "runtime", _runtime_render_slider)


# These renderers emit only the ScreenUI drawing commands. Third-party
# renderers may use other renderer operations; their existing contract is
# unchanged and they are not captured as retained native geometry.
_RETAINED_RUNTIME_RENDERERS = frozenset({
    _runtime_render_text, _runtime_render_image, _runtime_render_button,
    _runtime_render_progress, _runtime_render_slider, _render_visual_neutral_container,
})


def _can_retain_runtime_commands(element) -> bool:
    return _resolve_renderer(type(element), "runtime") in _RETAINED_RUNTIME_RENDERERS
