"""
Component rendering functions for the Inspector panel.

Each function renders the ImGui inspector UI for a specific component type.
Functions take the GUI context and component as arguments — they do not depend
on InspectorPanel state.

New components get a usable Inspector UI automatically via the generic
serialize→edit→deserialize renderer.  To provide a custom renderer, call
``register_component_renderer("TypeName", my_render_fn)`` at module level.
"""

import json
import math
from dataclasses import replace
from InfEngine.lib import InfGUIContext
from InfEngine.engine.i18n import t
from .inspector_utils import (
    max_label_w, field_label, render_serialized_field, has_field_changed,
    render_compact_section_header, render_info_text, render_component_header,
    render_inspector_checkbox, pretty_field_name,
    float_close as _float_close,
    get_enum_members as _get_enum_members,
    get_enum_member_name as _get_enum_member_name,
    get_enum_member_value as _get_enum_member_value,
    find_enum_index as _find_enum_index,
    DRAG_SPEED_DEFAULT, DRAG_SPEED_FINE, DRAG_SPEED_INT,
)
from .theme import Theme, ImGuiCol


def _is_in_play_mode() -> bool:
    """Return True if the engine is currently in runtime (play/pause) mode."""
    from InfEngine.engine.play_mode import PlayModeManager, PlayModeState
    pm = PlayModeManager.instance()
    if pm and pm.state != PlayModeState.EDIT:
        return True
    return False


# Enum helpers are now in inspector_utils — imported above as
# _get_enum_members, _get_enum_member_name, _get_enum_member_value, _find_enum_index


def _notify_scene_modified():
    """Mark the active scene as dirty (unsaved) in SceneFileManager."""
    from InfEngine.engine.scene_manager import SceneFileManager
    sfm = SceneFileManager.instance()
    if sfm:
        sfm.mark_dirty()


def _record_property(target, prop_name: str, old_value, new_value,
                     description: str = ""):
    """Record a property change through the undo system.

    Falls back to direct ``setattr`` + dirty-mark if UndoManager is
    unavailable.
    """
    from InfEngine.engine.undo import UndoManager, SetPropertyCommand
    mgr = UndoManager.instance()
    if mgr:
        mgr.execute(SetPropertyCommand(
            target, prop_name, old_value, new_value,
            description or f"Set {prop_name}"))
        return
    # Fallback
    setattr(target, prop_name, new_value)
    _notify_scene_modified()


def _record_generic_component(comp, old_json: str, new_json: str):
    """Record a generic C++ component JSON edit through the undo system."""
    from InfEngine.engine.undo import UndoManager, GenericComponentCommand
    mgr = UndoManager.instance()
    if mgr:
        mgr.execute(GenericComponentCommand(
            comp, old_json, new_json, f"Edit {comp.type_name}"))
        return
    # Fallback
    comp.deserialize(new_json)
    _notify_scene_modified()


def _record_add_component(obj, type_name: str, comp_ref,
                          is_py: bool = False):
    """Record the addition of a component through the undo system."""
    from InfEngine.engine.undo import (
        UndoManager, AddNativeComponentCommand, AddPyComponentCommand)
    mgr = UndoManager.instance()
    if mgr:
        if is_py:
            mgr.record(AddPyComponentCommand(
                obj.id, comp_ref,
                f"Add {getattr(comp_ref, 'type_name', type_name)}"))
        else:
            mgr.record(AddNativeComponentCommand(
                obj.id, type_name, comp_ref, f"Add {type_name}"))
        return
    _notify_scene_modified()

# ============================================================================
# Component renderer registry
# ============================================================================

_COMPONENT_RENDERERS: dict = {}   # type_name -> render_fn(ctx, comp)
_PY_COMPONENT_RENDERERS: dict = {}  # type_name -> render_fn(ctx, py_comp)
_COMPONENT_EXTRA_RENDERERS: dict = {}  # type_name -> render_fn(ctx, comp) appended after generic


def register_component_renderer(type_name: str, render_fn):
    """Register a custom Inspector renderer for a C++ component type.

    Args:
        type_name: The value returned by ``comp.type_name`` (e.g. "Camera").
        render_fn: ``fn(ctx: InfGUIContext, comp) -> None``
    """
    _COMPONENT_RENDERERS[type_name] = render_fn


def register_component_extra_renderer(type_name: str, render_fn):
    """Register extra Inspector UI appended after generic CppProperty rendering.

    Unlike ``register_component_renderer`` (which *replaces* the entire renderer),
    this appends additional UI *after* the generic CppProperty fields.  Use this
    when a component's standard properties can be handled generically but it needs
    extra custom sections (e.g. AudioSource per-track UI).

    Args:
        type_name: The value returned by ``comp.type_name``.
        render_fn: ``fn(ctx: InfGUIContext, comp) -> None``
    """
    _COMPONENT_EXTRA_RENDERERS[type_name] = render_fn


def register_py_component_renderer(type_name: str, render_fn):
    """Register a custom Inspector renderer for a Python component type.

    When registered, ``render_py_component()`` will use the custom renderer
    instead of the generic serialize-based renderer.

    Args:
        type_name: The ``type_name`` of the InfComponent (e.g. "RenderStack").
        render_fn: ``fn(ctx: InfGUIContext, py_comp) -> None``
    """
    _PY_COMPONENT_RENDERERS[type_name] = render_fn


def render_component(ctx: InfGUIContext, comp):
    """Unified entry point — dispatches to a custom renderer, then wraps
    BuiltinComponent subclasses into their Python wrapper and renders via
    the same ``render_py_component`` path that user scripts use, and
    finally falls back to the generic serialize-based renderer."""
    renderer = _COMPONENT_RENDERERS.get(comp.type_name)
    if renderer:
        renderer(ctx, comp)
        return

    # If a BuiltinComponent wrapper class exists for this C++ component,
    # wrap it and render through the unified render_py_component path.
    # CppProperty descriptors already populate _serialized_fields_ via
    # __init_subclass__, so the same field iteration logic applies.
    from InfEngine.components.builtin_component import BuiltinComponent
    wrapper_cls = BuiltinComponent._builtin_registry.get(comp.type_name)
    if wrapper_cls:
        raw_cpp = comp  # preserve original C++ component for fallback
        try:
            if not isinstance(comp, BuiltinComponent):
                go = getattr(comp, 'game_object', None)
                if go is not None:
                    comp = wrapper_cls._get_or_create_wrapper(comp, go)
                else:
                    render_cpp_component_generic(ctx, raw_cpp)
                    return
            render_py_component(ctx, comp)
        except Exception as exc:
            import traceback
            from InfEngine.debug import Debug
            tb_str = traceback.format_exc()
            Debug.log_warning(
                f"[Inspector] render_component fallback for "
                f"{getattr(raw_cpp, 'type_name', '?')}: {exc}\n{tb_str}"
            )
            try:
                render_cpp_component_generic(ctx, raw_cpp)
            except Exception as fallback_exc:
                Debug.log_warning(
                    f"[Inspector] fallback also failed for "
                    f"{getattr(raw_cpp, 'type_name', '?')}: {fallback_exc}"
                )
        extra = _COMPONENT_EXTRA_RENDERERS.get(getattr(comp, 'type_name', ''))
        if extra:
            extra(ctx, comp)
        return

    render_cpp_component_generic(ctx, comp)


# ============================================================================
# Built-in component renderers
# ============================================================================


def render_transform_component(ctx: InfGUIContext, trans):
    """Render Transform component fields (Position, Rotation, Scale).
    
    Displays LOCAL values in the inspector (matching Unity convention where
    the Inspector shows localPosition / localEulerAngles / localScale).
    """
    from InfEngine.lib import Vector3
    lw = max_label_w(ctx, ["Position", "Rotation", "Scale"])

    # Position (local space — offset from parent)
    pos = trans.local_position
    px, py, pz = pos[0], pos[1], pos[2]
    npx, npy, npz = ctx.vector3("Position", px, py, pz, DRAG_SPEED_DEFAULT, lw)
    if any(not _float_close(a, b) for a, b in [(npx, px), (npy, py), (npz, pz)]):
        _record_property(trans, "local_position", pos, Vector3(npx, npy, npz), "Set Position")

    # Rotation (local euler angles)
    rot = trans.local_euler_angles
    rx, ry, rz = rot[0], rot[1], rot[2]
    nrx, nry, nrz = ctx.vector3("Rotation", rx, ry, rz, DRAG_SPEED_DEFAULT, lw)
    if any(not _float_close(a, b) for a, b in [(nrx, rx), (nry, ry), (nrz, rz)]):
        _record_property(trans, "local_euler_angles", rot, Vector3(nrx, nry, nrz), "Set Rotation")

    # Scale (local — Unity convention: scale is always local)
    scl = trans.local_scale
    sx, sy, sz = scl[0], scl[1], scl[2]
    nsx, nsy, nsz = ctx.vector3("Scale", sx, sy, sz, DRAG_SPEED_FINE, lw)
    if any(not _float_close(a, b) for a, b in [(nsx, sx), (nsy, sy), (nsz, sz)]):
        _record_property(trans, "local_scale", scl, Vector3(nsx, nsy, nsz), "Set Scale")



# ============================================================================
# BuiltinComponent property-setter renderer
# ============================================================================

_CPP_PROPS_CACHE: dict = {}  # wrapper_cls -> list[(attr_name, CppProperty)]


def _collect_cpp_properties(wrapper_cls):
    """Collect CppProperty descriptors from *wrapper_cls* MRO (top→base).

    Returns a list of ``(python_attr_name, CppProperty)`` in definition
    order, skipping duplicates.  Results are cached per class.
    """
    cached = _CPP_PROPS_CACHE.get(wrapper_cls)
    if cached is not None:
        return cached
    seen = set()
    result = []
    # Walk the MRO in reverse so that the most-derived class wins
    for cls in reversed(wrapper_cls.__mro__):
        for attr_name, attr in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if getattr(attr, "_is_cpp_property", False) and attr_name not in seen:
                seen.add(attr_name)
                result.append((attr_name, attr))
    _CPP_PROPS_CACHE[wrapper_cls] = result
    return result


def render_builtin_via_setters(ctx: InfGUIContext, comp, wrapper_cls):
    """Render a C++ component by iterating CppProperty descriptors.

    If *comp* is a raw C++ component, it is wrapped in a BuiltinComponent
    wrapper so that CppProperty converters (e.g. COLOR get/set) are applied.
    """
    from InfEngine.components.serialized_field import FieldType
    from InfEngine.components.builtin_component import BuiltinComponent

    # Ensure we have a Python wrapper for correct CppProperty behavior
    if not isinstance(comp, BuiltinComponent):
        go = getattr(comp, 'game_object', None)
        if go is not None:
            comp = wrapper_cls._get_or_create_wrapper(comp, go)

    props = _collect_cpp_properties(wrapper_cls)
    if not props:
        # Fallback — no descriptors found
        render_cpp_component_generic(ctx, comp)
        return

    labels = [name for name, _ in props]
    lw = max_label_w(ctx, labels)

    for py_name, cpp_prop in props:
        meta = cpp_prop.metadata  # FieldMetadata
        cpp_attr = cpp_prop.cpp_attr

        # Conditional visibility
        if meta.visible_when is not None:
            try:
                if not meta.visible_when(comp):
                    continue
            except (RuntimeError, TypeError):
                pass  # On error, show the field

        # Headers / spacing
        if meta.header:
            ctx.separator()
            ctx.label(meta.header)
        if meta.space and meta.space > 0:
            ctx.dummy(0, meta.space)

        # Read current value from C++ component
        current = getattr(comp, cpp_attr)

        # Read-only fields: render as label and skip editing
        if meta.readonly:
            ctx.label(f"{py_name}: {current}")
            continue

        # ----- ASSET (AudioClip for built-in AudioSource.clip) -----
        if meta.field_type == FieldType.ASSET and cpp_attr == "clip":
            display_name = "None"
            if current is not None and hasattr(current, "name"):
                try:
                    display_name = current.name or "None"
                except (RuntimeError, AttributeError):
                    display_name = "None"

            field_label(ctx, py_name, lw)
            render_object_field(
                ctx,
                f"audio_clip_{py_name}",
                display_name,
                "AudioClip",
                accept_drag_type="AUDIO_FILE",
                on_drop_callback=lambda payload, _comp=comp, _attr=cpp_attr: _apply_builtin_audio_clip_drop(_comp, _attr, payload),
            )
            continue

        # ----- All scalar types via unified renderer -----
        new_value = render_serialized_field(
            ctx, f"##{py_name}", py_name, meta, current, lw,
        )

        if has_field_changed(meta.field_type, current, new_value):
            _record_builtin_property(comp, cpp_attr, current, new_value,
                                     f"Set {py_name}")

    # Append extra renderer if registered (e.g. AudioSource per-track section)
    extra = _COMPONENT_EXTRA_RENDERERS.get(getattr(comp, 'type_name', ''))
    if extra:
        extra(ctx, comp)


def _record_builtin_property(comp, cpp_attr: str, old_value, new_value,
                             description: str):
    """Apply a property change to a C++ component via direct setter, with undo.

    The setter path (e.g. ``comp.size = …``) goes through the pybind11
    property → C++ ``SetSize()`` → ``RebuildShape()`` → physics sync,
    which is exactly what we need for runtime changes.
    """
    from InfEngine.engine.undo import UndoManager, BuiltinPropertyCommand
    mgr = UndoManager.instance()
    if mgr:
        cmd = BuiltinPropertyCommand(comp, cpp_attr, old_value, new_value,
                                     description)
        mgr.execute(cmd)
        return
    # Fallback — just set the property directly
    setattr(comp, cpp_attr, new_value)
    _notify_scene_modified()


class _TrackVolumeCommand:
    """Lightweight undo command for AudioSource track volume.

    Implements the same interface that UndoManager expects from
    ``UndoCommand`` without pulling in a heavy ABC import.
    """
    supports_redo = True
    marks_dirty = True
    MERGE_WINDOW = 0.3

    def __init__(self, comp, track_index: int, old_vol: float, new_vol: float):
        import time as _time
        self.description = f"Set Track {track_index} Volume"
        self.timestamp = _time.time()
        self._comp = comp
        self._track = track_index
        self._old = old_vol
        self._new = new_vol
        self._comp_id = getattr(comp, "component_id", id(comp))

    def execute(self):
        self._comp.set_track_volume(self._track, self._new)

    def undo(self):
        self._comp.set_track_volume(self._track, self._old)

    def redo(self):
        self.execute()

    def can_merge(self, other):
        return (isinstance(other, _TrackVolumeCommand)
                and self._comp_id == other._comp_id
                and self._track == other._track
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other):
        self._new = other._new
        self.timestamp = other.timestamp


def _record_track_volume(comp, track_index: int, old_vol: float, new_vol: float):
    """Record an AudioSource track volume change through undo."""
    from InfEngine.engine.undo import UndoManager
    mgr = UndoManager.instance()
    if mgr:
        mgr.record(_TrackVolumeCommand(comp, track_index, old_vol, new_vol))
        return
    _notify_scene_modified()


def _apply_builtin_audio_clip_drop(comp, cpp_attr: str, payload):
    """Handle an AUDIO_FILE drag-drop onto a built-in component AudioClip field."""
    try:
        file_path = str(payload) if not isinstance(payload, str) else payload
        from InfEngine.core.audio_clip import AudioClip as PyAudioClip

        clip = PyAudioClip.load(file_path)
        if clip is None:
            return

        old_val = getattr(comp, cpp_attr)
        _record_builtin_property(comp, cpp_attr, old_val, clip.native, f"Set {cpp_attr}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"Audio clip drop failed: {e}")


def _asset_guid_from_path(file_path: str) -> str:
    from InfEngine.core.assets import AssetManager

    guid = ""
    adb = getattr(AssetManager, '_asset_database', None)
    if adb:
        try:
            guid = adb.get_guid_from_path(file_path) or ""
        except RuntimeError:
            pass
    return guid


def _create_reference_value_from_payload(element_type, payload, required_component: str = None):
    from InfEngine.components.serialized_field import FieldType

    if element_type == FieldType.GAME_OBJECT:
        from InfEngine.lib import SceneManager as _SM
        from InfEngine.components.ref_wrappers import GameObjectRef

        scene = _SM.instance().get_active_scene()
        if scene is None:
            return None

        obj_id = int(payload) if not isinstance(payload, int) else payload
        game_object = scene.find_by_id(obj_id)
        if game_object is None:
            return None
        if required_component and game_object.get_cpp_component(required_component) is None:
            # Also check Python type map
            from InfEngine.components.component import InfComponent
            tmap = InfComponent._type_map.get(game_object.id, {})
            if required_component not in tmap:
                return None
        return GameObjectRef(game_object)

    file_path = str(payload) if not isinstance(payload, str) else payload

    if element_type == FieldType.MATERIAL:
        from InfEngine.core.material import Material
        from InfEngine.components.ref_wrappers import MaterialRef

        mat = Material.load(file_path)
        if mat is None:
            return None
        return MaterialRef(mat, path_hint=file_path)

    if element_type == FieldType.TEXTURE:
        from InfEngine.core.asset_ref import TextureRef
        return TextureRef(guid=_asset_guid_from_path(file_path), path_hint=file_path)

    if element_type == FieldType.SHADER:
        from InfEngine.core.asset_ref import ShaderRef
        return ShaderRef(guid=_asset_guid_from_path(file_path), path_hint=file_path)

    if element_type == FieldType.ASSET:
        from InfEngine.core.asset_ref import AudioClipRef
        return AudioClipRef(guid=_asset_guid_from_path(file_path), path_hint=file_path)

    if element_type == FieldType.COMPONENT:
        from InfEngine.lib import SceneManager as _SM
        from InfEngine.components.ref_wrappers import ComponentRef

        scene = _SM.instance().get_active_scene()
        if scene is None:
            return None
        obj_id = int(payload) if not isinstance(payload, int) else payload
        game_object = scene.find_by_id(obj_id)
        if game_object is None:
            return None

        comp_type = required_component or ''
        if comp_type:
            # Validate the GO actually has this component via type map
            from InfEngine.components.component import InfComponent
            tmap = InfComponent._type_map.get(obj_id, {})
            if comp_type not in tmap:
                from InfEngine.debug import Debug
                Debug.log_warning(
                    f"GameObject '{game_object.name}' has no '{comp_type}' component."
                )
                return None
        else:
            # No type filter — pick the first Python component on this GO
            from InfEngine.components.component import InfComponent
            py_comps = InfComponent._active_instances.get(obj_id, [])
            if py_comps:
                comp_type = py_comps[0].__class__.__name__

        return ComponentRef(go_id=obj_id, component_type=comp_type)

    return None


def _get_reference_display_name(element_type, value) -> str:
    from InfEngine.components.serialized_field import FieldType

    if value is None:
        return "None"

    if element_type == FieldType.COMPONENT:
        if hasattr(value, 'display_name'):
            return value.display_name
        return "None"

    if element_type == FieldType.GAME_OBJECT:
        obj = value.resolve() if hasattr(value, 'resolve') else value
        return obj.name if obj and hasattr(obj, 'name') else "None"

    if hasattr(value, 'display_name'):
        return value.display_name

    if element_type == FieldType.MATERIAL:
        mat = value.resolve() if hasattr(value, 'resolve') else value
        return mat.name if mat and hasattr(mat, 'name') else "None"

    if element_type == FieldType.SHADER:
        resolved = value.resolve() if hasattr(value, 'resolve') else value
        if resolved and hasattr(resolved, 'source_path'):
            return resolved.source_path

    if hasattr(value, 'name'):
        return value.name or "None"

    return str(value)


def _make_list_default_element(metadata, element_type):
    from InfEngine.components.serialized_field import FieldType
    from InfEngine.math import Vector2, Vector3, vec4f

    if element_type == FieldType.INT:
        return 0
    if element_type == FieldType.FLOAT:
        return 0.0
    if element_type == FieldType.BOOL:
        return False
    if element_type == FieldType.STRING:
        return ""
    if element_type == FieldType.VEC2:
        return Vector2(0.0, 0.0)
    if element_type == FieldType.VEC3:
        return Vector3(0.0, 0.0, 0.0)
    if element_type == FieldType.VEC4:
        return vec4f(0.0, 0.0, 0.0, 0.0)
    if element_type == FieldType.COLOR:
        return [1.0, 1.0, 1.0, 1.0]
    if element_type == FieldType.ENUM and metadata.enum_type is not None:
        members = _get_enum_members(metadata.enum_type)
        return members[0] if members else metadata.default
    if element_type == FieldType.GAME_OBJECT:
        from InfEngine.components.ref_wrappers import GameObjectRef
        return GameObjectRef(persistent_id=0)
    if element_type == FieldType.MATERIAL:
        from InfEngine.components.ref_wrappers import MaterialRef
        return MaterialRef(guid="")
    if element_type == FieldType.TEXTURE:
        from InfEngine.core.asset_ref import TextureRef
        return TextureRef()
    if element_type == FieldType.SHADER:
        from InfEngine.core.asset_ref import ShaderRef
        return ShaderRef()
    if element_type == FieldType.ASSET:
        from InfEngine.core.asset_ref import AudioClipRef
        return AudioClipRef()
    if element_type == FieldType.SERIALIZABLE_OBJECT:
        elem_cls = metadata.element_class
        if elem_cls is not None:
            return elem_cls()
        return None
    if element_type == FieldType.COMPONENT:
        from InfEngine.components.ref_wrappers import ComponentRef
        comp_type = getattr(metadata, 'component_type', '') or ''
        return ComponentRef(component_type=comp_type)
    return None


def _infer_list_element_type(metadata, current_value):
    from InfEngine.components.serialized_field import infer_field_type_from_value, FieldType

    if metadata.element_type is not None:
        return metadata.element_type

    for container in (current_value, metadata.default):
        if not isinstance(container, (list, tuple)):
            continue
        for item in container:
            if item is None:
                continue
            inferred = infer_field_type_from_value(item)
            if inferred != FieldType.UNKNOWN:
                return inferred

    return FieldType.STRING


def _list_drag_drop_type(element_type):
    from InfEngine.components.serialized_field import FieldType

    mapping = {
        FieldType.GAME_OBJECT: "HIERARCHY_GAMEOBJECT",
        FieldType.MATERIAL: "MATERIAL_FILE",
        FieldType.TEXTURE: "TEXTURE_FILE",
        FieldType.SHADER: "SHADER_FILE",
        FieldType.ASSET: "AUDIO_FILE",
        FieldType.COMPONENT: "HIERARCHY_GAMEOBJECT",
    }
    return mapping.get(element_type)


def _list_type_hint(element_type, metadata):
    from InfEngine.components.serialized_field import FieldType

    if element_type == FieldType.GAME_OBJECT:
        return f"GameObject:{metadata.required_component}" if metadata.required_component else "GameObject"
    if element_type == FieldType.MATERIAL:
        return "Material"
    if element_type == FieldType.TEXTURE:
        return "Texture"
    if element_type == FieldType.SHADER:
        return "Shader"
    if element_type == FieldType.ASSET:
        return "AudioClip"
    if element_type == FieldType.COMPONENT:
        comp_type = getattr(metadata, 'component_type', '') or ''
        return comp_type or "Component"
    return "Element"


def _render_list_field(ctx: InfGUIContext, comp, field_name: str, metadata, current_value, lw: float):
    import copy as _copy
    from InfEngine.components.serialized_field import FieldType
    from .igui import IGUI

    items = list(current_value) if isinstance(current_value, list) else []
    element_type = _infer_list_element_type(metadata, items)
    changed = False
    button_spacing = Theme.INSPECTOR_INLINE_BTN_GAP

    reference_types = {
        FieldType.GAME_OBJECT,
        FieldType.MATERIAL,
        FieldType.TEXTURE,
        FieldType.SHADER,
        FieldType.ASSET,
        FieldType.COMPONENT,
    }

    # ── Callbacks for IGUI.list_header ──
    def _on_add():
        nonlocal changed
        items.append(_make_list_default_element(metadata, element_type))
        changed = True

    def _on_remove_last():
        nonlocal changed
        if items:
            items.pop()
            changed = True

    # Header drop callback (for reference types)
    _hdr_drag_type = _list_drag_drop_type(element_type) if element_type in reference_types else None
    _hdr_req = (metadata.component_type if element_type == FieldType.COMPONENT
                else metadata.required_component) if element_type in reference_types else None

    def _header_drop(payload):
        nonlocal changed
        value = _create_reference_value_from_payload(element_type, payload, _hdr_req)
        if value is not None:
            items.append(value)
            changed = True

    # ── Unified list header: [label [N] ........... [-][+]] ──
    header_open = IGUI.list_header(
        ctx, field_name, len(items),
        on_add=_on_add,
        on_remove=_on_remove_last if items else None,
        accept_drop=_hdr_drag_type,
        on_header_drop=_header_drop if _hdr_drag_type else None,
    )

    if not header_open:
        if changed and not metadata.readonly:
            _record_property(comp, field_name, current_value, items, f"Set {field_name}")
            if hasattr(comp, '_call_on_validate'):
                comp._call_on_validate()
        return

    # ── Drag-reorder state ──
    _list_drag_id = f"IGUI_LIST_{id(comp)}_{field_name}"
    move_from = None
    move_to = None

    def _make_reorder_cb(target_index):
        def _cb(payload):
            nonlocal move_from, move_to
            src = int(payload)
            if src != target_index and src != target_index - 1:
                move_from = src
                move_to = target_index
        return _cb

    # Separator before first item
    IGUI.reorder_separator(ctx, f"##sep_{field_name}_before_0", _list_drag_id, _make_reorder_cb(0))

    remove_index = None
    element_meta = replace(metadata, field_type=element_type, default=_make_list_default_element(metadata, element_type))

    for index, item in enumerate(items):
        ctx.push_id_str(f"{field_name}_{index}")

        # [-] remove button per item
        remove_clicked = IGUI.list_item_remove_button(ctx, f"{field_name}_{index}")
        ctx.same_line(0, button_spacing)

        # Drag source for reordering
        if ctx.begin_drag_drop_source(0):
            ctx.set_drag_drop_payload(_list_drag_id, index)
            ctx.label(f"[{index}]")
            ctx.end_drag_drop_source()

        if element_type in reference_types:
            _req = metadata.component_type if element_type == FieldType.COMPONENT else metadata.required_component
            def _replace_item(payload, _index=index, _req=_req):
                nonlocal changed
                value = _create_reference_value_from_payload(element_type, payload, _req)
                if value is not None:
                    items[_index] = value
                    changed = True

            # Build picker providers for list items
            _li_scene, _li_assets = _make_list_picker_providers(element_type, metadata)

            def _li_on_pick(value, _index=index, _et=element_type, _req=_req):
                nonlocal changed
                ref = _create_list_pick_ref(_et, value, _req)
                if ref is not None:
                    items[_index] = ref
                    changed = True

            def _li_on_clear(_index=index, _et=element_type):
                nonlocal changed
                items[_index] = _make_list_default_element(metadata, _et)
                changed = True

            IGUI.object_field(
                ctx,
                f"list_{field_name}_{index}",
                _get_reference_display_name(element_type, item),
                _list_type_hint(element_type, metadata),
                accept=_list_drag_drop_type(element_type),
                on_drop=_replace_item,
                picker_scene_items=_li_scene,
                picker_asset_items=_li_assets,
                on_pick=_li_on_pick,
                on_clear=_li_on_clear,
            )
        elif element_type == FieldType.SERIALIZABLE_OBJECT:
            so_class = type(item) if item is not None else metadata.element_class
            so_label = f"[{index}]" + (f" ({so_class.__name__})" if so_class else "")
            if render_compact_section_header(ctx, so_label, level="tertiary"):
                from InfEngine.components.serialized_field import get_serialized_fields as _gsf
                if so_class and item is not None:
                    so_fields = _gsf(so_class)
                    so_lw = max_label_w(ctx, list(so_fields.keys())) if so_fields else 0.0
                    elem_changes = {}
                    for so_fn, so_meta in so_fields.items():
                        so_val = getattr(item, so_fn, so_meta.default)
                        new_val = render_serialized_field(
                            ctx, f"##{field_name}_{index}_{so_fn}", so_fn,
                            so_meta, so_val, so_lw,
                        )
                        if has_field_changed(so_meta.field_type, so_val, new_val):
                            elem_changes[so_fn] = new_val
                    if elem_changes:
                        edited = _copy.deepcopy(item)
                        for fn, fv in elem_changes.items():
                            setattr(edited, fn, fv)
                        items[index] = edited
                        changed = True
        else:
            new_item = render_serialized_field(
                ctx, f"##{field_name}_{index}", "", element_meta, item, 0.0,
            )
            if has_field_changed(element_type, item, new_item):
                items[index] = new_item
                changed = True

        if remove_clicked:
            remove_index = index

        ctx.pop_id()

        # Separator after each item (reorder drop zone)
        IGUI.reorder_separator(ctx, f"##sep_{field_name}_after_{index}", _list_drag_id, _make_reorder_cb(index + 1))

    if remove_index is not None:
        items.pop(remove_index)
        changed = True

    # Apply reorder
    if move_from is not None and move_to is not None:
        elem = items.pop(move_from)
        insert_at = move_to if move_to < move_from else move_to - 1
        items.insert(insert_at, elem)
        changed = True

    # Bottom drop zone for reference types
    if element_type in reference_types:
        _req = metadata.component_type if element_type == FieldType.COMPONENT else metadata.required_component
        def _append_item(payload):
            nonlocal changed
            value = _create_reference_value_from_payload(element_type, payload, _req)
            if value is not None:
                items.append(value)
                changed = True

        IGUI.object_field(
            ctx,
            f"list_add_{field_name}",
            "",
            _list_type_hint(element_type, metadata),
            clickable=False,
            accept=_list_drag_drop_type(element_type),
            on_drop=_append_item,
        )

    if changed and not metadata.readonly:
        _record_property(comp, field_name, current_value, items, f"Set {field_name}")
        if hasattr(comp, '_call_on_validate'):
            comp._call_on_validate()


def render_cpp_component_generic(ctx: InfGUIContext, comp):
    """Render generic fields for a C++ component based on its serialized JSON."""
    original_json = comp.serialize()
    data = json.loads(original_json)

    ignore_keys = {"schema_version", "type", "enabled", "component_id"}
    changed = False

    visible_keys = [k for k in data if k not in ignore_keys]
    lw = max_label_w(ctx, visible_keys) if visible_keys else 0.0

    for key, value in data.items():
        if key in ignore_keys:
            continue

        new_value = value
        if isinstance(value, bool):
            new_value = render_inspector_checkbox(ctx, key, bool(value))
        elif isinstance(value, int):
            field_label(ctx, key, lw)
            new_value = int(ctx.drag_int(f"##{key}", int(value), DRAG_SPEED_INT, -1000000, 1000000))
        elif isinstance(value, float):
            field_label(ctx, key, lw)
            new_value = float(ctx.drag_float(f"##{key}", float(value), DRAG_SPEED_DEFAULT, -1e6, 1e6))
        elif isinstance(value, str):
            field_label(ctx, key, lw)
            new_value = ctx.text_input(f"##{key}", value, 256)
        elif isinstance(value, list):
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                nx, ny = ctx.vector2(key, float(value[0]), float(value[1]), DRAG_SPEED_DEFAULT, lw)
                new_value = [nx, ny]
            elif len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
                nx, ny, nz = ctx.vector3(key, float(value[0]), float(value[1]), float(value[2]), DRAG_SPEED_DEFAULT, lw)
                new_value = [nx, ny, nz]
            elif len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                nx, ny, nz, nw = ctx.vector4(key, float(value[0]), float(value[1]), float(value[2]), float(value[3]), DRAG_SPEED_DEFAULT, lw)
                new_value = [nx, ny, nz, nw]
            else:
                ctx.label(f"{key}: {value}")
        else:
            ctx.label(f"{key}: {value}")

        # Detect change — use tolerance for floats to avoid phantom edits
        if isinstance(value, float) and isinstance(new_value, float):
            value_changed = not _float_close(new_value, value)
        elif isinstance(value, list) and isinstance(new_value, list):
            value_changed = any(
                not _float_close(float(a), float(b))
                for a, b in zip(new_value, value)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
            ) or (len(new_value) != len(value))
        else:
            value_changed = (new_value != value)

        if value_changed:
            data[key] = new_value
            changed = True

    if changed:
        new_json = json.dumps(data)
        _record_generic_component(comp, original_json, new_json)


# _render_info_text is now render_info_text from inspector_utils
_render_info_text = render_info_text


def _render_serializable_object_field(
    ctx: InfGUIContext, comp, field_name: str, metadata, current_value, lw: float,
):
    """Render a SerializableObject field as an inline collapsible section.

    Sub-fields are rendered recursively.  Changes are applied via
    deep-copy + _record_property so undo works correctly.
    """
    import copy as _copy
    from InfEngine.components.serialized_field import get_serialized_fields, FieldType

    so_class = type(current_value) if current_value is not None else getattr(metadata, 'serializable_class', None)
    if so_class is None:
        field_label(ctx, field_name, lw)
        ctx.label(t("inspector.unknown_type"))
        return

    header = f"{field_name} ({so_class.__name__})"
    if not render_compact_section_header(ctx, header, level="secondary"):
        return

    so_fields = get_serialized_fields(so_class)
    so_lw = max_label_w(ctx, list(so_fields.keys())) if so_fields else 0.0

    if current_value is None:
        current_value = so_class()
        _record_property(comp, field_name, None, current_value, f"Init {field_name}")

    changes: dict = {}
    for so_fn, so_meta in so_fields.items():
        so_val = getattr(current_value, so_fn, so_meta.default)

        if so_meta.field_type == FieldType.SERIALIZABLE_OBJECT:
            # Nested SO — recursive rendering
            # We handle nested changes by deep-copying the parent when done.
            _render_nested_so(ctx, field_name, so_fn, so_meta, so_val, so_lw, changes)
        else:
            new_val = render_serialized_field(
                ctx, f"##{field_name}_{so_fn}", so_fn, so_meta, so_val, so_lw,
            )
            if has_field_changed(so_meta.field_type, so_val, new_val):
                changes[so_fn] = new_val

    if changes and not metadata.readonly:
        edited = _copy.deepcopy(current_value)
        for fn, fv in changes.items():
            setattr(edited, fn, fv)
        _record_property(comp, field_name, current_value, edited, f"Set {field_name}")
        if hasattr(comp, '_call_on_validate'):
            comp._call_on_validate()


def _render_nested_so(
    ctx: InfGUIContext, parent_id: str, so_fn: str, so_meta, so_val, so_lw: float,
    changes: dict,
):
    """Render a nested SerializableObject sub-field and collect changes."""
    import copy as _copy
    from InfEngine.components.serialized_field import get_serialized_fields, FieldType

    so_class = type(so_val) if so_val is not None else getattr(so_meta, 'serializable_class', None)
    if so_class is None:
        ctx.label(f"{so_fn}: " + t("inspector.unknown_type"))
        return

    header = f"{so_fn} ({so_class.__name__})"
    if not render_compact_section_header(ctx, header, level="tertiary"):
        return

    inner_fields = get_serialized_fields(so_class)
    inner_lw = max_label_w(ctx, list(inner_fields.keys())) if inner_fields else 0.0

    if so_val is None:
        so_val = so_class()
        changes[so_fn] = so_val
        return

    inner_changes: dict = {}
    for ifn, imeta in inner_fields.items():
        ival = getattr(so_val, ifn, imeta.default)
        new_val = render_serialized_field(
            ctx, f"##{parent_id}_{so_fn}_{ifn}", ifn, imeta, ival, inner_lw,
        )
        if has_field_changed(imeta.field_type, ival, new_val):
            inner_changes[ifn] = new_val

    if inner_changes:
        edited = _copy.deepcopy(so_val)
        for fn, fv in inner_changes.items():
            setattr(edited, fn, fv)
        changes[so_fn] = edited


def render_py_component(ctx: InfGUIContext, py_comp):
    """Render a Python InfComponent's serialized fields.

    If a custom renderer is registered via ``register_py_component_renderer``,
    it is used instead of the generic field-based renderer.

    Supports:
    - ``group``: fields with the same group name are wrapped in a
      ``collapsing_header`` section.
    - ``info_text``: dimmed description line rendered after the field.
    """
    renderer = _PY_COMPONENT_RENDERERS.get(py_comp.type_name)
    if renderer:
        renderer(ctx, py_comp)
        return
    from InfEngine.components.serialized_field import get_serialized_fields, FieldType

    fields = get_serialized_fields(py_comp.__class__)
    lw = max_label_w(ctx, list(fields.keys())) if fields else 0.0

    # Track which collapsible group is currently open so we can close it
    _current_group: str = ""
    _group_visible: bool = True

    for field_name, metadata in fields.items():
        # ── Collapsible group management ──
        field_group = metadata.group or ""
        if field_group != _current_group:
            _current_group = field_group
            if field_group:
                _group_visible = render_compact_section_header(ctx, field_group, level="secondary")
            else:
                _group_visible = True

        if not _group_visible:
            continue

        # Conditional visibility (visible_when callback)
        if metadata.visible_when is not None:
            try:
                if not metadata.visible_when(py_comp):
                    continue
            except (RuntimeError, TypeError):
                pass  # On error, show the field

        # Add header if specified (simple label, NOT collapsible)
        if metadata.header:
            ctx.separator()
            ctx.label(metadata.header)

        # Add space if specified
        if metadata.space > 0:
            ctx.dummy(0, metadata.space)

        # Get current value. For reference-like fields, use the raw stored ref
        # so the Inspector keeps access to path_hint / missing-ref state.
        try:
            if metadata.field_type in (
                FieldType.GAME_OBJECT,
                FieldType.MATERIAL,
                FieldType.TEXTURE,
                FieldType.SHADER,
                FieldType.ASSET,
                FieldType.COMPONENT,
            ):
                from InfEngine.components.serialized_field import get_raw_field_value
                current_value = get_raw_field_value(py_comp, field_name)
            else:
                current_value = getattr(py_comp, field_name, metadata.default)
        except RuntimeError:
            current_value = metadata.default

        # Readonly scalar fields: render as label, skip interactive widgets
        if metadata.readonly and metadata.field_type in (FieldType.INT, FieldType.FLOAT, FieldType.STRING, FieldType.BOOL):
            field_label(ctx, field_name, lw)
            ctx.label(f"{current_value}")
            if metadata.tooltip and ctx.is_item_hovered():
                ctx.set_tooltip(metadata.tooltip)
            continue

        if metadata.field_type == FieldType.LIST:
            from InfEngine.components.serialized_field import get_raw_field_value
            _raw_list = get_raw_field_value(py_comp, field_name)
            _render_list_field(ctx, py_comp, field_name, metadata, _raw_list, lw)
            if metadata.tooltip and ctx.is_item_hovered():
                ctx.set_tooltip(metadata.tooltip)
            if metadata.info_text:
                _render_info_text(ctx, metadata.info_text)
            continue

        # ── SerializableObject — inline collapsible section ──
        if metadata.field_type == FieldType.SERIALIZABLE_OBJECT:
            _render_serializable_object_field(ctx, py_comp, field_name, metadata, current_value, lw)
            if metadata.tooltip and ctx.is_item_hovered():
                ctx.set_tooltip(metadata.tooltip)
            if metadata.info_text:
                _render_info_text(ctx, metadata.info_text)
            continue

        # ── ComponentRef — object field with component type filtering ──
        if metadata.field_type == FieldType.COMPONENT:
            from InfEngine.components.ref_wrappers import ComponentRef
            from InfEngine.components.serialized_field import get_raw_field_value
            _comp_ref = get_raw_field_value(py_comp, field_name)
            if not isinstance(_comp_ref, ComponentRef):
                _comp_ref = ComponentRef()
            _display = _comp_ref.display_name
            _type_hint = metadata.component_type or "Component"
            _ct = metadata.component_type

            def _comp_scene(filt, _ct=_ct):
                return _picker_scene_gameobjects(filt, required_component=_ct)

            def _comp_on_pick(go, _fn=field_name, _comp=py_comp, _ct=_ct):
                ref = _create_component_ref_from_go(go, _ct)
                if ref is not None:
                    old = get_raw_field_value(_comp, _fn)
                    _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _comp_on_clear(_fn=field_name, _comp=py_comp):
                old = get_raw_field_value(_comp, _fn)
                _record_property(_comp, _fn, old, ComponentRef(), f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"comp_ref_{field_name}", _display, _type_hint,
                accept_drag_type=["HIERARCHY_GAMEOBJECT", "PREFAB_GUID", "PREFAB_FILE"],
                on_drop_callback=lambda payload, _fn=field_name, _comp=py_comp, _ct=metadata.component_type: _apply_component_ref_drop(_comp, _fn, payload, _ct),
                picker_scene_items=_comp_scene,
                on_pick=_comp_on_pick,
                on_clear=_comp_on_clear,
            )
            if metadata.tooltip and ctx.is_item_hovered():
                ctx.set_tooltip(metadata.tooltip)
            if metadata.info_text:
                _render_info_text(ctx, metadata.info_text)
            continue

        # ── Reference types need context-specific callbacks — handle separately ──
        if metadata.field_type == FieldType.GAME_OBJECT:
            _display_obj = current_value
            if hasattr(current_value, 'resolve'):
                _display_obj = current_value.resolve()
            display = _display_obj.name if _display_obj and hasattr(_display_obj, 'name') else "None"
            _type_hint = "GameObject"
            _req_comp = metadata.required_component
            if _req_comp:
                _type_hint = f"GameObject:{_req_comp}"

            def _go_scene(filt, _rc=_req_comp):
                return _picker_scene_gameobjects(filt, required_component=_rc)

            def _go_on_pick(go, _fn=field_name, _comp=py_comp):
                from InfEngine.components.ref_wrappers import GameObjectRef
                ref = GameObjectRef(go)
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _go_on_clear(_fn=field_name, _comp=py_comp):
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, None, f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"go_ref_{field_name}", display, _type_hint,
                accept_drag_type=["HIERARCHY_GAMEOBJECT", "PREFAB_GUID", "PREFAB_FILE"],
                on_drop_callback=lambda payload, _fn=field_name, _comp=py_comp, _rc=_req_comp: _apply_gameobject_or_prefab_drop(_comp, _fn, payload, _rc),
                picker_scene_items=_go_scene,
                on_pick=_go_on_pick,
                on_clear=_go_on_clear,
            )

        elif metadata.field_type == FieldType.MATERIAL:
            _display_mat = current_value
            if hasattr(current_value, 'resolve'):
                _display_mat = current_value.resolve()
            display = _display_mat.name if _display_mat and hasattr(_display_mat, 'name') else "None"

            def _mat_on_pick(path, _fn=field_name, _comp=py_comp):
                ref = _create_reference_value_from_payload(FieldType.MATERIAL, path)
                if ref is not None:
                    old = getattr(_comp, _fn, None)
                    _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _mat_on_clear(_fn=field_name, _comp=py_comp):
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, None, f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"mat_ref_{field_name}", display, "Material",
                accept_drag_type="MATERIAL_FILE",
                on_drop_callback=lambda payload, _fn=field_name, _comp=py_comp: _apply_material_drop(_comp, _fn, payload),
                picker_asset_items=lambda filt: _picker_assets(filt, "*.mat"),
                on_pick=_mat_on_pick,
                on_clear=_mat_on_clear,
            )

        elif metadata.field_type == FieldType.TEXTURE:
            _display_name = "None"
            if hasattr(current_value, 'display_name'):
                _display_name = current_value.display_name
            elif current_value and hasattr(current_value, 'name'):
                _display_name = current_value.name

            def _tex_on_pick(path, _fn=field_name, _comp=py_comp):
                ref = _create_reference_value_from_payload(FieldType.TEXTURE, path)
                if ref is not None:
                    old = getattr(_comp, _fn, None)
                    _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _tex_on_clear(_fn=field_name, _comp=py_comp):
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, None, f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"tex_ref_{field_name}", _display_name, "Texture",
                accept_drag_type="TEXTURE_FILE",
                on_drop_callback=lambda payload, _fn=field_name, _comp=py_comp: _apply_texture_drop(_comp, _fn, payload),
                picker_asset_items=lambda filt: _picker_assets(filt, "*.png") + _picker_assets(filt, "*.jpg"),
                on_pick=_tex_on_pick,
                on_clear=_tex_on_clear,
            )

        elif metadata.field_type == FieldType.SHADER:
            _display_name = "None"
            if hasattr(current_value, 'display_name'):
                _display_name = current_value.display_name
            elif current_value and hasattr(current_value, 'source_path'):
                _display_name = current_value.source_path

            def _shd_on_pick(path, _fn=field_name, _comp=py_comp):
                ref = _create_reference_value_from_payload(FieldType.SHADER, path)
                if ref is not None:
                    old = getattr(_comp, _fn, None)
                    _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _shd_on_clear(_fn=field_name, _comp=py_comp):
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, None, f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"shd_ref_{field_name}", _display_name, "Shader",
                accept_drag_type="SHADER_FILE",
                on_drop_callback=lambda payload, _fn=field_name, _comp=py_comp: _apply_shader_drop(_comp, _fn, payload),
                picker_asset_items=lambda filt: _picker_assets(filt, "*.vert") + _picker_assets(filt, "*.frag"),
                on_pick=_shd_on_pick,
                on_clear=_shd_on_clear,
            )

        elif metadata.field_type == FieldType.ASSET:
            _display_name = _get_reference_display_name(FieldType.ASSET, current_value)

            def _apply_audio_ref(payload, _fn=field_name, _comp=py_comp):
                ref = _create_reference_value_from_payload(FieldType.ASSET, payload)
                if ref is None:
                    return
                old_val = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old_val, ref, f"Set {_fn}")

            def _aud_on_pick(path, _fn=field_name, _comp=py_comp):
                ref = _create_reference_value_from_payload(FieldType.ASSET, path)
                if ref is not None:
                    old = getattr(_comp, _fn, None)
                    _record_property(_comp, _fn, old, ref, f"Set {_fn}")

            def _aud_on_clear(_fn=field_name, _comp=py_comp):
                old = getattr(_comp, _fn, None)
                _record_property(_comp, _fn, old, None, f"Clear {_fn}")

            field_label(ctx, field_name, lw)
            render_object_field(
                ctx, f"aud_ref_{field_name}", _display_name, "AudioClip",
                accept_drag_type="AUDIO_FILE",
                on_drop_callback=_apply_audio_ref,
                picker_asset_items=lambda filt: _picker_assets(filt, "*.wav") + _picker_assets(filt, "*.mp3") + _picker_assets(filt, "*.ogg"),
                on_pick=_aud_on_pick,
                on_clear=_aud_on_clear,
            )

        else:
            # ── All scalar types via unified renderer ──
            new_value = render_serialized_field(
                ctx, f"##{field_name}", field_name, metadata, current_value, lw,
            )

            if has_field_changed(metadata.field_type, current_value, new_value) and not metadata.readonly:
                _record_property(py_comp, field_name, current_value, new_value, f"Set {field_name}")
                if hasattr(py_comp, '_call_on_validate'):
                    py_comp._call_on_validate()

        # Show tooltip if available
        if metadata.tooltip and ctx.is_item_hovered():
            ctx.set_tooltip(metadata.tooltip)

        # Show info text if available
        if metadata.info_text:
            _render_info_text(ctx, metadata.info_text)



def _apply_gameobject_drop(comp, field_name: str, payload, required_component: str = None):
    """Handle a HIERARCHY_GAMEOBJECT drag-drop onto a GAME_OBJECT field.

    Wraps the result in a ``GameObjectRef`` for null safety.
    If *required_component* is set, rejects objects that lack that C++ component.
    """
    try:
        from InfEngine.components.serialized_field import FieldType
        ref = _create_reference_value_from_payload(FieldType.GAME_OBJECT, payload, required_component)
        if ref is None:
            return
        old_val = getattr(comp, field_name, None)
        _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"GameObject drop failed: {e}")


def _apply_gameobject_or_prefab_drop(comp, field_name: str, payload, required_component: str = None):
    """Handle a HIERARCHY_GAMEOBJECT or PREFAB drag-drop onto a GAME_OBJECT field.

    If the payload is a string (file path or GUID from a prefab), the prefab
    is instantiated first, then the resulting root GO is used as the reference.
    """
    if isinstance(payload, str):
        # Prefab drop — instantiate and use root GO
        try:
            from InfEngine.engine.prefab_manager import instantiate_prefab
            from InfEngine.lib import SceneManager, AssetRegistry
            scene = SceneManager.instance().get_active_scene()
            adb = None
            registry = AssetRegistry.instance()
            if registry:
                adb = registry.get_asset_database()

            # Determine if it's a GUID or a file path
            import os
            if os.path.isfile(payload):
                new_obj = instantiate_prefab(file_path=payload, scene=scene, asset_database=adb)
            else:
                new_obj = instantiate_prefab(guid=payload, scene=scene, asset_database=adb)

            if new_obj is None:
                from InfEngine.debug import Debug
                Debug.log_warning("Failed to instantiate prefab for field drop.")
                return

            from InfEngine.components.ref_wrappers import GameObjectRef
            ref = GameObjectRef(new_obj)
            old_val = getattr(comp, field_name, None)
            _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
        except Exception as e:
            from InfEngine.debug import Debug
            Debug.log_error(f"Prefab drop failed: {e}")
    else:
        # HIERARCHY_GAMEOBJECT drop (payload is int ID)
        _apply_gameobject_drop(comp, field_name, payload, required_component)


def _apply_material_drop(comp, field_name: str, payload):
    """Handle a MATERIAL_FILE drag-drop onto a MATERIAL field.

    Wraps the result in a ``MaterialRef`` for null safety and GUID persistence.
    """
    try:
        from InfEngine.components.serialized_field import FieldType
        ref = _create_reference_value_from_payload(FieldType.MATERIAL, payload)
        if ref is None:
            return
        old_val = getattr(comp, field_name, None)
        _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"Material drop failed: {e}")


def _apply_texture_drop(comp, field_name: str, payload):
    """Handle a TEXTURE_FILE / IMAGE_FILE drag-drop onto a TEXTURE field.

    Creates a ``TextureRef`` that stores the GUID for scene serialization.
    """
    try:
        from InfEngine.components.serialized_field import FieldType
        ref = _create_reference_value_from_payload(FieldType.TEXTURE, payload)
        if ref is None:
            return
        old_val = getattr(comp, field_name, None)
        _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"Texture drop failed: {e}")


def _apply_shader_drop(comp, field_name: str, payload):
    """Handle a SHADER_FILE drag-drop onto a SHADER field.

    Creates a ``ShaderRef`` that stores the GUID for scene serialization.
    """
    try:
        from InfEngine.components.serialized_field import FieldType
        ref = _create_reference_value_from_payload(FieldType.SHADER, payload)
        if ref is None:
            return
        old_val = getattr(comp, field_name, None)
        _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"Shader drop failed: {e}")


def _apply_component_ref_drop(comp, field_name: str, payload, component_type: str = ""):
    """Handle a HIERARCHY_GAMEOBJECT drag-drop onto a COMPONENT field.

    Creates a ``ComponentRef`` pointing to the target component on the
    dropped GameObject.  Delegates to ``_create_reference_value_from_payload``
    for consistency with list fields.
    """
    try:
        from InfEngine.components.serialized_field import FieldType
        ref = _create_reference_value_from_payload(FieldType.COMPONENT, payload, component_type)
        if ref is None:
            return
        from InfEngine.components.serialized_field import get_raw_field_value
        old_val = get_raw_field_value(comp, field_name)
        _record_property(comp, field_name, old_val, ref, f"Set {field_name}")
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"ComponentRef drop failed: {e}")


# ============================================================================
# Picker item providers for object fields
# ============================================================================

def _create_component_ref_from_go(game_object, component_type: str = ""):
    """Create a ComponentRef from a picked GameObject (for picker popup)."""
    from InfEngine.components.ref_wrappers import ComponentRef
    if game_object is None:
        return None
    go_id = game_object.id
    ct = component_type or ''
    if ct:
        # Verify the GO has this component via type map
        from InfEngine.components.component import InfComponent
        tmap = InfComponent._type_map.get(go_id, {})
        if ct not in tmap:
            return None
    else:
        from InfEngine.components.component import InfComponent
        py_comps = InfComponent._active_instances.get(go_id, [])
        if py_comps:
            ct = py_comps[0].__class__.__name__
    return ComponentRef(go_id=go_id, component_type=ct)


def _picker_scene_gameobjects(filter_text: str, required_component: str = None):
    """Return ``[(name, go), ...]`` for all scene GameObjects matching *filter_text*.

    If *required_component* is set, only include GameObjects with that component.
    """
    from InfEngine.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        return []
    items = []
    filt = filter_text.lower()
    for go in scene.get_all_objects():
        if filt and filt not in go.name.lower():
            continue
        if required_component:
            # Check type map for the required component
            from InfEngine.components.component import InfComponent
            tmap = InfComponent._type_map.get(go.id, {})
            if required_component not in tmap:
                continue
        items.append((go.name, go))
    return items


def _picker_assets(filter_text: str, pattern: str):
    """Return ``[(display_name, path), ...]`` for assets matching *pattern*."""
    import os
    from InfEngine.core.assets import AssetManager
    paths = AssetManager.find_assets(pattern)
    items = []
    filt = filter_text.lower()
    for p in paths:
        name = os.path.basename(p)
        if filt and filt not in name.lower():
            continue
        items.append((name, p))
    return items


def _make_list_picker_providers(element_type, metadata):
    """Return ``(scene_items_fn_or_None, asset_items_fn_or_None)`` for a list element type."""
    from InfEngine.components.serialized_field import FieldType

    if element_type in (FieldType.GAME_OBJECT, FieldType.COMPONENT):
        _rc = metadata.component_type if element_type == FieldType.COMPONENT else metadata.required_component
        return (lambda filt, _rc=_rc: _picker_scene_gameobjects(filt, required_component=_rc), None)

    if element_type == FieldType.MATERIAL:
        return (None, lambda filt: _picker_assets(filt, "*.mat"))
    if element_type == FieldType.TEXTURE:
        return (None, lambda filt: _picker_assets(filt, "*.png") + _picker_assets(filt, "*.jpg"))
    if element_type == FieldType.SHADER:
        return (None, lambda filt: _picker_assets(filt, "*.vert") + _picker_assets(filt, "*.frag"))
    if element_type == FieldType.ASSET:
        return (None, lambda filt: _picker_assets(filt, "*.wav") + _picker_assets(filt, "*.mp3") + _picker_assets(filt, "*.ogg"))
    return (None, None)


def _create_list_pick_ref(element_type, value, required_component=None):
    """Create a reference wrapper from a picker selection for a list element."""
    from InfEngine.components.serialized_field import FieldType

    if element_type == FieldType.GAME_OBJECT:
        from InfEngine.components.ref_wrappers import GameObjectRef
        return GameObjectRef(value) if value is not None else None

    if element_type == FieldType.COMPONENT:
        return _create_component_ref_from_go(value, required_component or '')

    # Asset types: value is a file path
    if element_type == FieldType.MATERIAL:
        return _create_reference_value_from_payload(FieldType.MATERIAL, value)
    if element_type == FieldType.TEXTURE:
        return _create_reference_value_from_payload(FieldType.TEXTURE, value)
    if element_type == FieldType.SHADER:
        return _create_reference_value_from_payload(FieldType.SHADER, value)
    if element_type == FieldType.ASSET:
        return _create_reference_value_from_payload(FieldType.ASSET, value)
    return None


def render_object_field(ctx: InfGUIContext, field_id: str, display_text: str,
                        type_hint: str, selected: bool = False, clickable: bool = True,
                        accept_drag_type: str = None, on_drop_callback=None,
                        picker_scene_items=None, picker_asset_items=None,
                        on_pick=None, on_clear=None) -> bool:
    """Render a Unity-style object field (selectable box showing an object reference).

    Delegates to ``IGUI.object_field`` for consistent behaviour
    (white outline on hover, picker dot, etc.).
    """
    from .igui import IGUI
    return IGUI.object_field(
        ctx, field_id, display_text, type_hint,
        selected=selected, clickable=clickable,
        accept=accept_drag_type, on_drop=on_drop_callback,
        picker_scene_items=picker_scene_items,
        picker_asset_items=picker_asset_items,
        on_pick=on_pick, on_clear=on_clear,
    )


# ============================================================================
# Camera extra renderer ("Set as Main Camera" button)
# ============================================================================


# ============================================================================
# AudioSource extra renderer (per-track section only)
# ============================================================================


def _render_audio_source_extra(ctx: InfGUIContext, comp):
    """Extra Inspector section for AudioSource: per-track clip & volume.

    Source-level properties (volume, pitch, mute, spatial, etc.) are handled
    by the generic CppProperty renderer.  This function only renders the
    dynamic per-track section that cannot be expressed as CppProperty.
    """
    track_count = comp.track_count

    ctx.separator()
    ctx.label("Tracks")

    track_labels = ["Clip", "Volume"]
    track_lw = max_label_w(ctx, track_labels)

    for i in range(track_count):
        ctx.set_next_item_open(True)
        if ctx.collapsing_header(f"Track {i}"):
            # Track clip
            clip = comp.get_track_clip(i)
            clip_name = "None"
            if clip is not None:
                try:
                    clip_name = clip.name or "None"
                except (RuntimeError, AttributeError):
                    clip_name = "None"

            field_label(ctx, "Clip", track_lw)
            render_object_field(
                ctx,
                f"audio_track_clip_{i}",
                clip_name,
                "AudioClip",
                accept_drag_type="AUDIO_FILE",
                on_drop_callback=lambda payload, _c=comp, _i=i: _apply_track_audio_clip_drop(_c, _i, payload),
            )

            # Track volume
            tv = comp.get_track_volume(i)
            field_label(ctx, "Volume", track_lw)
            new_tv = ctx.float_slider(f"##track_vol_{i}", float(tv), 0.0, 1.0)
            if not _float_close(float(new_tv), float(tv)):
                comp.set_track_volume(i, float(new_tv))
                _record_track_volume(comp, i, float(tv), float(new_tv))

            # Play / Stop buttons (only in play mode for feedback)
            if _is_in_play_mode():
                is_playing = comp.is_track_playing(i)
                if is_playing:
                    if ctx.button(f"Stop##track_stop_{i}"):
                        comp.stop(i)
                else:
                    if ctx.button(f"Play##track_play_{i}"):
                        comp.play(i)
                ctx.same_line()
                status = "Playing" if is_playing else ("Paused" if comp.is_track_paused(i) else "Stopped")
                ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
                ctx.label(status)
                ctx.pop_style_color(1)


def _apply_track_audio_clip_drop(comp, track_index: int, payload):
    """Handle an AUDIO_FILE drag-drop onto a track clip field.

    Tries GUID-based loading via AssetRegistry first, falls back to path.
    """
    try:
        file_path = str(payload) if not isinstance(payload, str) else payload

        # Try GUID-based loading via AssetRegistry
        from InfEngine.lib import AssetRegistry
        registry = AssetRegistry.instance()
        adb = registry.get_asset_database()
        if adb:
            guid = adb.get_guid_from_path(file_path)
            if guid and hasattr(comp, 'set_track_clip_by_guid'):
                comp.set_track_clip_by_guid(track_index, guid)
                _notify_scene_modified()
                return

        # Fallback: load from file path directly
        from InfEngine.core.audio_clip import AudioClip as PyAudioClip

        clip = PyAudioClip.load(file_path)
        if clip is None:
            return

        comp.set_track_clip(track_index, clip.native)
        _notify_scene_modified()
    except Exception as e:
        from InfEngine.debug import Debug
        Debug.log_error(f"Audio clip drop failed: {e}")


# ============================================================================
# Auto-register built-in component renderers
# ============================================================================
register_component_renderer("Transform", render_transform_component)
register_component_extra_renderer("AudioSource", _render_audio_source_extra)
# MeshRenderer is registered by InspectorPanel.__init__ (bound method) because
# its renderer needs panel-level state (material selection, drag-drop).

# ============================================================================
# Auto-register Python component renderers
# ============================================================================
from .inspector_renderstack import render_renderstack_inspector
register_py_component_renderer("RenderStack", render_renderstack_inspector)
from . import inspector_ui_components  # Registers UI component inspectors.
