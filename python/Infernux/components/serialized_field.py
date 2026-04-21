"""
Serialized Field Decorator for InxComponent.

This module provides the @serialized_field decorator that marks class attributes
as serializable and inspector-visible fields.

Usage:
    class MyComponent(InxComponent):
        speed: float = serialized_field(default=5.0, range=(0, 100), tooltip="Movement speed")
        name: str = serialized_field(default="Player")
        target: 'GameObject' = serialized_field(default=None)
"""

from enum import Enum, auto
from typing import Any, Tuple, Optional, Type, Dict, Callable, TYPE_CHECKING
from dataclasses import dataclass
import copy
import weakref
import threading
from Infernux.debug import Debug

if TYPE_CHECKING:
    from .component import InxComponent


# ── Injectable callbacks (set once by EditorBootstrap) ─────────────
#
# These break the circular dependency:
#     components.serialized_field  →  engine.undo / play_mode / scene_manager
#
# ``on_field_will_change(instance, field_name, old, new) -> bool``
#   Called before storing; returns True if undo handled the write (caller
#   should ``return`` early).
#
# ``on_field_did_change(instance, field_name, old, new) -> None``
#   Called after storing; responsible for marking-dirty / play-mode guard.

_on_field_will_change: Optional[Callable] = None
_on_field_did_change: Optional[Callable] = None


def set_field_change_hooks(
    will_change: Optional[Callable] = None,
    did_change: Optional[Callable] = None,
) -> None:
    """Inject editor-level callbacks (called once at startup)."""
    global _on_field_will_change, _on_field_did_change
    _on_field_will_change = will_change
    _on_field_did_change = did_change


class FieldType(Enum):
    """Supported field types for serialization and inspector rendering."""
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    COLOR = auto()
    GAME_OBJECT = auto()  # Reference to another GameObject
    COMPONENT = auto()     # Reference to a component
    MATERIAL = auto()      # Reference to a Material asset
    TEXTURE = auto()       # Reference to a Texture asset
    SHADER = auto()        # Reference to a Shader asset
    ASSET = auto()         # Generic asset reference
    ENUM = auto()
    LIST = auto()
    SERIALIZABLE_OBJECT = auto()  # Custom data class (SerializableObject subclass)
    UNKNOWN = auto()


@dataclass
class FieldMetadata:
    """Metadata for a serialized field."""
    name: str
    field_type: FieldType
    default: Any
    range: Optional[Tuple[float, float]] = None  # (min, max) for numeric types
    tooltip: str = ""
    readonly: bool = False
    header: str = ""  # Group header shown above this field
    space: float = 0.0  # Vertical space before this field
    enum_type: Optional[Type[Enum]] = None  # For ENUM fields (or str for lazy resolve)
    enum_labels: Optional[list] = None  # Override display names for ENUM members
    element_type: Optional[FieldType] = None  # For LIST fields
    group: str = ""  # Collapsible group name (fields sharing the same group are folded together)
    info_text: str = ""  # Non-editable description shown below the field (dimmed)
    multiline: bool = False  # STRING: use multiline text input widget
    slider: bool = True  # When range is set, True = slider widget, False = bounded drag
    drag_speed: Optional[float] = None  # Override default drag speed (None = type default)
    required_component: Optional[str] = None  # For GAME_OBJECT: only accept objects with this C++ component
    visible_when: Optional[Callable] = None  # fn(component) → bool; hides field when False
    element_class: Optional[Type] = None       # For LIST: the SerializableObject subclass for elements
    serializable_class: Optional[Type] = None  # For SERIALIZABLE_OBJECT: the concrete class to instantiate
    component_type: Optional[str] = None       # For COMPONENT (ComponentRef): target component type name
    hdr: bool = False                            # For COLOR: allow HDR mode toggle
    asset_type: Optional[str] = None             # For ASSET: registered asset type name (e.g. "AudioClip", "AnimStateMachine")

    # For internal use
    python_type: Optional[Type] = None
    getter: Optional[Callable] = None
    setter: Optional[Callable] = None


class SerializedFieldDescriptor:
    """
    Descriptor that handles get/set for serialized fields.
    This enables proper attribute access while maintaining metadata.
    
    Uses weak references to automatically clean up values when instances
    are garbage collected, preventing memory leaks.

    Numeric fields (INT, FLOAT, BOOL, VEC2, VEC3, VEC4) are backed by the
    C++ ComponentDataStore for cache-friendly batch access.  The CDS
    identifiers are stamped on this descriptor by ``_cds_bridge.register_class``.
    """
    
    def __init__(self, metadata: FieldMetadata):
        self.metadata = metadata
        self._values: Dict[int, Any] = {}  # instance id -> value
        self._weak_refs: Dict[int, weakref.ref] = {}  # instance id -> weak ref
        self._lock = threading.Lock()  # Thread-safe access
        self._set_count: int = 0  # Counter for periodic dead-ref cleanup
        # CDS backing (set by _cds_bridge.register_class; None = Python-only).
        self._cds_class_id: Optional[int] = None
        self._cds_field_id: Optional[int] = None
        self._cds_type_code: Optional[int] = None

    def _make_ref_callback(self, inst_id: int):
        """Create a weak-ref callback that auto-cleans on GC."""
        def _on_gc(_ref, _iid=inst_id, _self=self):
            with _self._lock:
                _self._values.pop(_iid, None)
                _self._weak_refs.pop(_iid, None)
        return _on_gc
    
    def __set_name__(self, owner: Type, name: str):
        self.metadata.name = name
        # Register this field in the owner class
        if '_serialized_fields_' not in owner.__dict__:
            owner._serialized_fields_ = {}
        owner._serialized_fields_[name] = self.metadata
    
    def _cleanup_dead_refs(self):
        """Remove entries for garbage-collected instances."""
        with self._lock:
            dead_ids = [inst_id for inst_id, ref in self._weak_refs.items() if ref() is None]
            for inst_id in dead_ids:
                self._values.pop(inst_id, None)
                self._weak_refs.pop(inst_id, None)
    
    def get_raw(self, instance: 'InxComponent') -> Any:
        """Get the raw stored value without auto-resolution."""
        if self._cds_class_id is not None:
            slot = getattr(instance, '_cds_slot', None)
            if slot is not None and getattr(instance, '_cds_class_id', None) == self._cds_class_id:
                from ._cds_bridge import cds_get
                return cds_get(self._cds_class_id, self._cds_field_id, self._cds_type_code, slot)
        inst_id = id(instance)
        with self._lock:
            return self._values.get(inst_id, self.metadata.default)

    # FieldTypes that are stored as Ref wrappers with a _cached attribute.
    _REF_FIELD_TYPES = frozenset({
        FieldType.MATERIAL,
        FieldType.TEXTURE,
        FieldType.SHADER,
        FieldType.ASSET,
        FieldType.GAME_OBJECT,
        FieldType.COMPONENT,
    })

    def __get__(self, instance: Optional['InxComponent'], owner: Type) -> Any:
        if instance is None:
            return self
        # CDS fast path for numeric fields.
        if self._cds_class_id is not None:
            slot = getattr(instance, '_cds_slot', None)
            if slot is not None and getattr(instance, '_cds_class_id', None) == self._cds_class_id:
                from ._cds_bridge import cds_get
                return cds_get(self._cds_class_id, self._cds_field_id, self._cds_type_code, slot)
        inst_id = id(instance)
        with self._lock:
            value = self._values.get(inst_id, self.metadata.default)
        # Fast path: for ref-type fields that already have a cached resolved
        # object, return it directly — skips resolve_runtime_field_value and
        # the entire _resolve_single_reference dispatch chain.
        if self.metadata.field_type in self._REF_FIELD_TYPES:
            if value is None:
                return None
            cached = getattr(value, '_cached', None)
            if cached is not None:
                return cached
        return resolve_runtime_field_value(value, self.metadata)
    
    def __set__(self, instance: 'InxComponent', value: Any):
        if self.metadata.readonly and not getattr(instance, '_inf_deserializing', False):
            raise AttributeError(f"Field '{self.metadata.name}' is readonly")

        value = normalize_runtime_field_value(value, self.metadata)

        # CDS fast path for numeric fields.
        if self._cds_class_id is not None:
            slot = getattr(instance, '_cds_slot', None)
            if slot is not None and getattr(instance, '_cds_class_id', None) == self._cds_class_id:
                # Undo hook (before write)
                if not getattr(instance, '_inf_deserializing', False) and _on_field_will_change is not None:
                    from ._cds_bridge import cds_get
                    old_value = cds_get(self._cds_class_id, self._cds_field_id, self._cds_type_code, slot)
                    if old_value != value:
                        if _on_field_will_change(instance, self.metadata.name, old_value, value):
                            return
                from ._cds_bridge import cds_set, cds_get as _cg
                old = _cg(self._cds_class_id, self._cds_field_id, self._cds_type_code, slot)
                cds_set(self._cds_class_id, self._cds_field_id, self._cds_type_code, slot, value)
                if not getattr(instance, '_inf_deserializing', False):
                    if old != value and _on_field_did_change is not None:
                        _on_field_did_change(instance, self.metadata.name, old, value)
                return

        inst_id = id(instance)

        # --- Undo integration via injectable callback ---
        if not getattr(instance, '_inf_deserializing', False) and _on_field_will_change is not None:
            with self._lock:
                old_value = self._values.get(inst_id, self.metadata.default)
            if old_value != value:
                if _on_field_will_change(instance, self.metadata.name, old_value, value):
                    return  # callback handled the write (undo recorded)

        # Normal set path
        with self._lock:
            old = self._values.get(inst_id, self.metadata.default)
            self._values[inst_id] = value
            # Track instance with weak reference for cleanup
            if inst_id not in self._weak_refs:
                self._weak_refs[inst_id] = weakref.ref(instance, self._make_ref_callback(inst_id))

        # Mark scene dirty via injectable callback
        if not getattr(instance, '_inf_deserializing', False):
            if old != value and _on_field_did_change is not None:
                _on_field_did_change(instance, self.metadata.name, old, value)

        # Periodic batch cleanup as a safety net for ref-cycle GC edge cases.
        self._set_count += 1
        if self._set_count >= 128:
            self._set_count = 0
            self._cleanup_dead_refs()
    
    def __delete__(self, instance: 'InxComponent'):
        inst_id = id(instance)
        with self._lock:
            self._values.pop(inst_id, None)
            self._weak_refs.pop(inst_id, None)


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-wrap / raw-access helpers for COMPONENT fields
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_component_ref(value):
    """Wrap a component instance into ComponentRef; pass through if already one."""
    from .ref_wrappers import ComponentRef
    if isinstance(value, ComponentRef):
        return value
    if value is None:
        return ComponentRef()
    # Live component → wrap by go_id + type_name
    go = getattr(value, 'game_object', None)
    go_id = int(go.id) if go is not None else 0
    type_name = getattr(value, 'type_name', type(value).__name__)
    return ComponentRef(go_id=go_id, component_type=type_name)


def _get_asset_db():
    try:
        from Infernux.core.asset_ref import _get_asset_database
        return _get_asset_database()
    except ImportError:
        # AssetManager is unavailable in standalone player builds where the
        # editor-only asset database is excluded from the bundle.
        return None


def _guid_from_path(path: str) -> str:
    if not path:
        return ""
    db = _get_asset_db()
    if not db:
        return ""
    try:
        guid = db.get_guid_from_path(path)
        return guid or ""
    except Exception as exc:
        Debug.log_suppressed("serialized_field._guid_for_path", exc)
        return ""


def _extract_guid_and_path(value, path_attrs: tuple[str, ...]) -> tuple[str, str]:
    if value is None:
        return "", ""

    if isinstance(value, str):
        guid = _guid_from_path(value)
        if guid:
            return guid, value
        return value, ""

    guid = getattr(value, 'guid', '') or getattr(getattr(value, 'native', None), 'guid', '') or ''
    path_hint = ''
    for attr in path_attrs:
        path_hint = getattr(value, attr, '') or getattr(getattr(value, 'native', None), attr, '') or ''
        if path_hint:
            break

    if not guid and path_hint:
        guid = _guid_from_path(path_hint)
    return guid, path_hint


def _ensure_game_object_ref(value):
    from .ref_wrappers import GameObjectRef, PrefabRef
    if isinstance(value, (GameObjectRef, PrefabRef)):
        return value
    if value is None:
        return GameObjectRef(persistent_id=0)
    return GameObjectRef(value)


def _ensure_material_ref(value):
    from .ref_wrappers import MaterialRef
    if isinstance(value, MaterialRef):
        return value
    if value is None:
        return MaterialRef(guid="")
    return MaterialRef(value)


def _ensure_texture_ref(value):
    from Infernux.core.asset_ref import TextureRef
    if isinstance(value, TextureRef):
        return value
    ref = TextureRef()
    if value is None:
        return ref
    guid, path_hint = _extract_guid_and_path(value, ('source_path', 'file_path'))
    ref.guid = guid
    ref.path_hint = path_hint
    ref._cached = value
    return ref


def _ensure_shader_ref(value):
    from Infernux.core.asset_ref import ShaderRef
    if isinstance(value, ShaderRef):
        return value
    ref = ShaderRef()
    if value is None:
        return ref
    guid, path_hint = _extract_guid_and_path(value, ('source_path', 'file_path'))
    ref.guid = guid
    ref.path_hint = path_hint
    ref._cached = value
    return ref


def _ensure_audio_clip_ref(value):
    from Infernux.core.asset_ref import AudioClipRef
    if isinstance(value, AudioClipRef):
        return value
    ref = AudioClipRef()
    if value is None:
        return ref
    guid, path_hint = _extract_guid_and_path(value, ('file_path', 'source_path'))
    ref.guid = guid
    ref.path_hint = path_hint
    ref._cached = value
    return ref


def _ensure_asset_ref(value, asset_type: str = "AudioClip"):
    """Wrap *value* in the appropriate AssetRefBase subclass for *asset_type*."""
    from Infernux.core.asset_ref import AssetRefBase, get_asset_type_config
    cfg = get_asset_type_config(asset_type)
    if cfg is None:
        return _ensure_audio_clip_ref(value)
    ref_class = cfg["ref_class"]
    if isinstance(value, ref_class):
        return value
    # Accept any AssetRefBase — transfer guid/path_hint
    if isinstance(value, AssetRefBase):
        return ref_class(guid=value.guid, path_hint=value.path_hint)
    ref = ref_class()
    if value is None:
        return ref
    guid, path_hint = _extract_guid_and_path(value, ('file_path', 'source_path'))
    ref.guid = guid
    ref.path_hint = path_hint
    ref._cached = value
    return ref


def _resolve_single_reference(value: Any, field_type: FieldType) -> Any:
    if value is None:
        return None
    if field_type == FieldType.GAME_OBJECT:
        from .ref_wrappers import GameObjectRef
        return value.resolve() if isinstance(value, GameObjectRef) else value
    if field_type == FieldType.COMPONENT:
        from .ref_wrappers import ComponentRef
        return value.resolve() if isinstance(value, ComponentRef) else value
    if field_type == FieldType.MATERIAL:
        from .ref_wrappers import MaterialRef
        return value.resolve() if isinstance(value, MaterialRef) else value
    if field_type == FieldType.TEXTURE:
        from Infernux.core.asset_ref import TextureRef
        return value.resolve() if isinstance(value, TextureRef) else value
    if field_type == FieldType.SHADER:
        from Infernux.core.asset_ref import ShaderRef
        return value.resolve() if isinstance(value, ShaderRef) else value
    if field_type == FieldType.ASSET:
        from Infernux.core.asset_ref import AssetRefBase
        return value.resolve() if isinstance(value, AssetRefBase) else value
    return value


def resolve_runtime_field_value(value: Any, field_meta_or_type) -> Any:
    if hasattr(field_meta_or_type, 'field_type'):
        field_type = field_meta_or_type.field_type
        element_type = getattr(field_meta_or_type, 'element_type', None)
    else:
        field_type = field_meta_or_type
        element_type = None

    if field_type in {
        FieldType.GAME_OBJECT,
        FieldType.COMPONENT,
        FieldType.MATERIAL,
        FieldType.TEXTURE,
        FieldType.SHADER,
        FieldType.ASSET,
    }:
        return _resolve_single_reference(value, field_type)

    if field_type == FieldType.LIST and isinstance(value, list):
        if element_type in {
            FieldType.GAME_OBJECT,
            FieldType.COMPONENT,
            FieldType.MATERIAL,
            FieldType.TEXTURE,
            FieldType.SHADER,
            FieldType.ASSET,
        }:
            return [_resolve_single_reference(item, element_type) for item in value]
    return value


def normalize_runtime_field_value(value: Any, field_meta_or_type) -> Any:
    if hasattr(field_meta_or_type, 'field_type'):
        field_type = field_meta_or_type.field_type
        element_type = getattr(field_meta_or_type, 'element_type', None)
        asset_type = getattr(field_meta_or_type, 'asset_type', None)
    else:
        field_type = field_meta_or_type
        element_type = None
        asset_type = None

    if field_type == FieldType.COMPONENT:
        return _ensure_component_ref(value)
    if field_type == FieldType.GAME_OBJECT:
        return _ensure_game_object_ref(value)
    if field_type == FieldType.MATERIAL:
        return _ensure_material_ref(value)
    if field_type == FieldType.TEXTURE:
        return _ensure_texture_ref(value)
    if field_type == FieldType.SHADER:
        return _ensure_shader_ref(value)
    if field_type == FieldType.ASSET:
        return _ensure_asset_ref(value, asset_type or "AudioClip")
    if field_type == FieldType.LIST and isinstance(value, list):
        if element_type == FieldType.COMPONENT:
            return [_ensure_component_ref(v) for v in value]
        if element_type == FieldType.GAME_OBJECT:
            return [_ensure_game_object_ref(v) for v in value]
        if element_type == FieldType.MATERIAL:
            return [_ensure_material_ref(v) for v in value]
        if element_type == FieldType.TEXTURE:
            return [_ensure_texture_ref(v) for v in value]
        if element_type == FieldType.SHADER:
            return [_ensure_shader_ref(v) for v in value]
        if element_type == FieldType.ASSET:
            return [_ensure_audio_clip_ref(v) for v in value]
    return value


def get_raw_field_value(component: 'InxComponent', field_name: str) -> Any:
    """Get the raw stored value of a serialized field (bypasses auto-resolve).

    For COMPONENT fields this returns the underlying ``ComponentRef``
    instead of the resolved component.  Used by serialization, Inspector,
    and undo internals.
    """
    for cls in type(component).__mro__:
        desc = cls.__dict__.get(field_name)
        if isinstance(desc, SerializedFieldDescriptor):
            return desc.get_raw(component)
    fields = getattr(type(component), '_serialized_fields_', {})
    if field_name in fields and hasattr(component, '__dict__'):
        return component.__dict__.get(field_name, fields[field_name].default)
    return getattr(component, field_name)


# ── Type → FieldType dispatch tables (used by _infer_field_type) ──

_DIRECT_TYPE_TO_FIELD: dict = {
    int:   FieldType.INT,
    float: FieldType.FLOAT,
    bool:  FieldType.BOOL,
    str:   FieldType.STRING,
}

_TYPE_NAME_TO_FIELD: dict = {
    'Vec2': FieldType.VEC2,    'Vector2': FieldType.VEC2,    'vector2': FieldType.VEC2,
    'Vec3': FieldType.VEC3,    'Vector3': FieldType.VEC3,    'vector3': FieldType.VEC3,
    'vec4f': FieldType.VEC4,   'Vec4': FieldType.VEC4,
    'Vector4': FieldType.VEC4, 'vector4': FieldType.VEC4,
    'GameObject': FieldType.GAME_OBJECT,
    'Material': FieldType.MATERIAL,
    'Texture': FieldType.TEXTURE,    'TextureRef': FieldType.TEXTURE,
    'Shader': FieldType.SHADER,     'ShaderRef': FieldType.SHADER,
    'ShaderAssetInfo': FieldType.SHADER,
    'AudioClip': FieldType.ASSET,   'AudioClipRef': FieldType.ASSET,
    'ComponentRef': FieldType.COMPONENT,
}

_DEFAULT_TYPE_NAME_TO_FIELD: dict = {
    'Material': FieldType.MATERIAL,     'InxMaterial': FieldType.MATERIAL,
    'Texture': FieldType.TEXTURE,       'TextureRef': FieldType.TEXTURE,
    'Shader': FieldType.SHADER,         'ShaderRef': FieldType.SHADER,
    'ShaderAssetInfo': FieldType.SHADER,
    'AudioClip': FieldType.ASSET,       'AudioClipRef': FieldType.ASSET,
}

_VEC_ANNOTATION_MAP: dict = {
    'Vec2': FieldType.VEC2,    'Vector2': FieldType.VEC2,    'vector2': FieldType.VEC2,
    'Vec3': FieldType.VEC3,    'Vector3': FieldType.VEC3,    'vector3': FieldType.VEC3,
    'vec4f': FieldType.VEC4,   'Vec4': FieldType.VEC4,
    'Vector4': FieldType.VEC4, 'vector4': FieldType.VEC4,
}


def _make_vec_default(ft: FieldType):
    from Infernux.lib._Infernux import Vector2, Vector3, vec4f
    if ft == FieldType.VEC2:
        return Vector2(0, 0)
    if ft == FieldType.VEC3:
        return Vector3(0, 0, 0)
    if ft == FieldType.VEC4:
        return vec4f(0, 0, 0, 0)
    return None


def _infer_field_type(python_type: Optional[Type], default: Any) -> FieldType:
    """Infer FieldType from Python type annotation or default value."""
    if python_type is not None:
        # Direct type match (int, float, bool, str)
        result = _DIRECT_TYPE_TO_FIELD.get(python_type)
        if result is not None:
            return result

        # Name-based match (vectors, asset refs, etc.)
        type_name = getattr(python_type, '__name__', str(python_type))
        result = _TYPE_NAME_TO_FIELD.get(type_name)
        if result is not None:
            return result

        if isinstance(python_type, type) and issubclass(python_type, Enum):
            return FieldType.ENUM
        if hasattr(python_type, '__origin__') and python_type.__origin__ in (list, tuple):
            return FieldType.LIST

        # SerializableObject subclass detection.
        # ImportError is benign: SerializableObject may not be importable
        # yet during early module load — the field will be classified on
        # the next inspection pass.
        try:
            from .serializable_object import SerializableObject as _SO
            if isinstance(python_type, type) and issubclass(python_type, _SO):
                return FieldType.SERIALIZABLE_OBJECT
        except ImportError:
            pass

        # InxComponent subclass → COMPONENT (e.g. ``text: UIText``)
        try:
            from .component import InxComponent as _IC
            if isinstance(python_type, type) and issubclass(python_type, _IC) and python_type is not _IC:
                return FieldType.COMPONENT
        except ImportError:
            pass

    # Infer from default value
    if default is not None:
        # SerializableObject instance
        try:
            from .serializable_object import SerializableObject as _SO
            if isinstance(default, _SO):
                return FieldType.SERIALIZABLE_OBJECT
        except ImportError:
            pass

        # ComponentRef instance
        try:
            from .ref_wrappers import ComponentRef
            if isinstance(default, ComponentRef):
                return FieldType.COMPONENT
        except ImportError:
            pass

        # Order matters: Enum before int (IntEnum is both), bool before int
        if isinstance(default, Enum):
            return FieldType.ENUM
        if isinstance(default, bool):
            return FieldType.BOOL
        if isinstance(default, int):
            return FieldType.INT
        if isinstance(default, float):
            return FieldType.FLOAT
        if isinstance(default, str):
            return FieldType.STRING
        if hasattr(default, 'x') and hasattr(default, 'y'):
            if hasattr(default, 'z') and hasattr(default, 'w'):
                return FieldType.VEC4
            if hasattr(default, 'z'):
                return FieldType.VEC3
            return FieldType.VEC2
        if isinstance(default, (list, tuple)):
            return FieldType.LIST

        # Asset ref types by class name (avoids circular import)
        result = _DEFAULT_TYPE_NAME_TO_FIELD.get(type(default).__name__)
        if result is not None:
            return result

    return FieldType.UNKNOWN


def _infer_list_element_type(default: Any) -> Optional[FieldType]:
    if not isinstance(default, (list, tuple)):
        return None

    for item in default:
        if item is None:
            continue
        inferred = infer_field_type_from_value(item)
        if inferred != FieldType.UNKNOWN:
            return inferred

    return None


def infer_field_type_from_value(value: Any) -> FieldType:
    """Infer FieldType from a runtime value (for auto-serialized fields)."""
    if value is None:
        return FieldType.UNKNOWN
    return _infer_field_type(type(value), value)


def resolve_annotation(annotation) -> Optional['FieldMetadata']:
    """Resolve a type annotation to a FieldMetadata with an appropriate default.

    Used by ``InxComponent.__init_subclass__`` to support bare annotation
    syntax like ``text: UIText`` or ``mat: Material = None``.

    Also handles ``list[Camera]`` / ``list[GameObject]`` generics,
    producing a ``LIST`` field with the appropriate ``element_type``.

    Returns ``None`` if the annotation is not a recognised supported type.
    """
    if isinstance(annotation, str):
        text = annotation.strip()
        if text.startswith(('list[', 'List[')) and text.endswith(']'):
            inner = text[text.find('[') + 1:-1].strip()
            inner_meta = resolve_annotation(inner)
            if inner_meta is not None:
                return FieldMetadata(
                    name="",
                    field_type=FieldType.LIST,
                    default=[],
                    element_type=inner_meta.field_type,
                    component_type=inner_meta.component_type,
                )
            return None

        simple_name = text.split('.')[-1]
        _vec_ft = _VEC_ANNOTATION_MAP.get(simple_name)
        if _vec_ft is not None:
            return FieldMetadata(name="", field_type=_vec_ft, default=_make_vec_default(_vec_ft))
        if simple_name in {
            'GameObject', 'Material', 'Texture', 'TextureRef',
            'Shader', 'ShaderRef', 'AudioClip', 'AudioClipRef', 'ComponentRef'
        }:
            return resolve_annotation(type(simple_name, (), {'__name__': simple_name}))

        try:
            from .registry import get_type
            resolved = get_type(simple_name)
            if resolved is not None:
                return resolve_annotation(resolved)
        except Exception as exc:
            Debug.log_suppressed(
                f"serialized_field.resolve_annotation.registry_lookup[{simple_name}]",
                exc,
            )
        return None

    # ── list[X] / List[X] generics ──
    import typing as _typing
    _origin = _typing.get_origin(annotation)
    if _origin in (list, tuple):
        _args = _typing.get_args(annotation)
        if _args:
            inner_meta = resolve_annotation(_args[0])
            if inner_meta is not None:
                return FieldMetadata(
                    name="",
                    field_type=FieldType.LIST,
                    default=[],
                    element_type=inner_meta.field_type,
                    component_type=inner_meta.component_type,
                )
        return None

    if annotation is None or not isinstance(annotation, type):
        return None

    type_name = annotation.__name__

    # ── Basic value types ──
    if annotation is int:
        return FieldMetadata(name="", field_type=FieldType.INT, default=0)
    if annotation is float:
        return FieldMetadata(name="", field_type=FieldType.FLOAT, default=0.0)
    if annotation is bool:
        return FieldMetadata(name="", field_type=FieldType.BOOL, default=False)
    if annotation is str:
        return FieldMetadata(name="", field_type=FieldType.STRING, default="")

    # ── Vector types ──
    _vec_ft = _VEC_ANNOTATION_MAP.get(type_name)
    if _vec_ft is not None:
        return FieldMetadata(name="", field_type=_vec_ft, default=_make_vec_default(_vec_ft))

    # ── InxComponent subclass → ComponentRef ──
    try:
        from .component import InxComponent as _IC
        if issubclass(annotation, _IC) and annotation is not _IC:
            from .ref_wrappers import ComponentRef
            return FieldMetadata(
                name="",
                field_type=FieldType.COMPONENT,
                default=ComponentRef(component_type=type_name),
                component_type=type_name,
            )
    except ImportError:
        # InxComponent not yet importable during early module load — the
        # annotation will be reclassified on the next inspection pass.
        pass

    # ── Known reference / asset types ──
    _MAP = {
        'GameObject':   (FieldType.GAME_OBJECT, '_go_ref'),
        'Material':     (FieldType.MATERIAL,    '_mat_ref'),
        'Texture':      (FieldType.TEXTURE,     '_tex_ref'),
        'TextureRef':   (FieldType.TEXTURE,     '_tex_ref'),
        'Shader':       (FieldType.SHADER,      '_shader_ref'),
        'ShaderRef':    (FieldType.SHADER,      '_shader_ref'),
        'AudioClip':    (FieldType.ASSET,       '_audio_ref'),
        'AudioClipRef': (FieldType.ASSET,       '_audio_ref'),
        'ComponentRef': (FieldType.COMPONENT,   '_comp_ref'),
    }
    entry = _MAP.get(type_name)
    if entry is not None:
        field_type, _tag = entry
        default = _make_ref_default(type_name)
        return FieldMetadata(
            name="",
            field_type=field_type,
            default=default,
            component_type="" if field_type == FieldType.COMPONENT else None,
        )

    return None


def get_annotation_default(annotation) -> Any:
    """Best-effort default value for annotation-only fields.

    This is used for private, non-serialized annotation-only fields such as
    ``_counter: int`` so they behave like initialized instance fields.
    """
    meta = resolve_annotation(annotation)
    if meta is not None:
        return meta.default

    if isinstance(annotation, str):
        text = annotation.strip()
        if text.startswith(('list[', 'List[')) and text.endswith(']'):
            return []
        return None

    import typing as _typing
    origin = _typing.get_origin(annotation)
    if origin in (list, tuple):
        return []

    return None


def _make_ref_default(type_name: str):
    """Create an empty default instance for a known reference type name."""
    if type_name == 'GameObject':
        from .ref_wrappers import GameObjectRef
        return GameObjectRef()
    if type_name == 'Material':
        from .ref_wrappers import MaterialRef
        return MaterialRef()
    if type_name in ('Texture', 'TextureRef'):
        from ..core.asset_ref import TextureRef
        return TextureRef()
    if type_name in ('Shader', 'ShaderRef'):
        from ..core.asset_ref import ShaderRef
        return ShaderRef()
    if type_name in ('AudioClip', 'AudioClipRef'):
        from ..core.asset_ref import AudioClipRef
        return AudioClipRef()
    if type_name == 'ComponentRef':
        from .ref_wrappers import ComponentRef
        return ComponentRef()
    return None


def serialized_field(
    default: Any = None,
    *,
    field_type: Optional[FieldType] = None,
    element_type: Optional[FieldType] = None,
    element_class: Optional[Type] = None,
    serializable_class: Optional[Type] = None,
    component_type: Optional[str] = None,
    asset_type: Optional[str] = None,
    range: Optional[Tuple[float, float]] = None,
    tooltip: str = "",
    readonly: bool = False,
    header: str = "",
    space: float = 0.0,
    group: str = "",
    info_text: str = "",
    multiline: bool = False,
    slider: bool = True,
    drag_speed: Optional[float] = None,
    required_component: Optional[str] = None,
    visible_when: Optional[Callable] = None,
    hdr: bool = False,
) -> Any:
    """
    Decorator/descriptor for marking a field as serialized and inspector-visible.
    
    Args:
        default: Default value for the field
        field_type: Explicit field type (auto-detected if not provided)
        range: (min, max) tuple for numeric sliders / bounded drag
        tooltip: Hover text shown in inspector
        readonly: If True, field cannot be modified in inspector
        header: Group header text shown above this field
        space: Vertical spacing before this field in inspector
        group: Collapsible group name.  All consecutive fields with the
            same *group* value are wrapped inside a single
            ``collapsing_header`` section.
        info_text: Non-editable description line rendered after the field
            widget in dimmed text.  Useful for hints and explanations.
        multiline: If True and the field is STRING, render a multiline
            text input widget instead of a single-line one.
        slider: When ``range`` is set, controls the widget style.
            ``True`` (default) = slider, ``False`` = bounded drag.
        drag_speed: Override the default drag speed for numeric fields.
            ``None`` means use the type default (0.1 for float, 1.0 for int).
        required_component: For GAME_OBJECT fields only.  If set, only
            GameObjects that have a C++ component with this type name
            (e.g. ``"MeshRenderer"``) will be accepted when dragged from
            the Hierarchy panel.
        hdr: For COLOR fields only.  If True, allow HDR values (> 1.0)
            in the colour picker.
    
    Returns:
        A descriptor that manages the field value and metadata
    
    Example:
        class MyComponent(InxComponent):
            speed: float = serialized_field(default=5.0, range=(0, 100))
            name: str = serialized_field(default="Player", tooltip="Object name")
            debug: bool = serialized_field(default=False, header="Debug Options")
            text: str = serialized_field(default="Hi", group="Content")
    """
    # Infer field type if not provided
    if field_type is None and component_type:
        inferred_type = FieldType.COMPONENT
    elif field_type is None and asset_type:
        inferred_type = FieldType.ASSET
    else:
        inferred_type = field_type or _infer_field_type(None, default)

    # Auto-default for reference fields: store empty refs internally
    if default is None:
        if inferred_type == FieldType.COMPONENT:
            from .ref_wrappers import ComponentRef
            default = ComponentRef(component_type=component_type or "")
        elif inferred_type == FieldType.GAME_OBJECT:
            default = _ensure_game_object_ref(None)
        elif inferred_type == FieldType.MATERIAL:
            default = _ensure_material_ref(None)
        elif inferred_type == FieldType.TEXTURE:
            default = _ensure_texture_ref(None)
        elif inferred_type == FieldType.SHADER:
            default = _ensure_shader_ref(None)
        elif inferred_type == FieldType.ASSET:
            default = _ensure_asset_ref(None, asset_type or "AudioClip")

    inferred_element_type = element_type
    if inferred_type == FieldType.LIST and inferred_element_type is None:
        inferred_element_type = _infer_list_element_type(default)
    
    # Auto-detect enum_type from default
    enum_type = None
    if isinstance(default, Enum):
        enum_type = type(default)
    
    metadata = FieldMetadata(
        name="",  # Will be set by __set_name__
        field_type=inferred_type,
        default=default,
        range=range,
        tooltip=tooltip,
        readonly=readonly,
        header=header,
        space=space,
        enum_type=enum_type,
        element_type=inferred_element_type,
        group=group,
        info_text=info_text,
        multiline=multiline,
        slider=slider,
        drag_speed=drag_speed,
        required_component=required_component,
        visible_when=visible_when,
        element_class=element_class,
        serializable_class=serializable_class,
        component_type=component_type,
        hdr=hdr,
        asset_type=asset_type,
    )
    
    return SerializedFieldDescriptor(metadata)


_SERIALIZED_FIELDS_CACHE: dict = {}  # component_class -> Dict[str, FieldMetadata]


def clear_serialized_fields_cache(component_class=None):
    """Clear cached field metadata.

    If *component_class* is given, only that entry is removed.
    Otherwise the entire cache is flushed (used after hot-reload).
    """
    if component_class is not None:
        _SERIALIZED_FIELDS_CACHE.pop(component_class, None)
    else:
        _SERIALIZED_FIELDS_CACHE.clear()


def get_serialized_fields(component_class: Type['InxComponent']) -> Dict[str, FieldMetadata]:
    """
    Get all serialized fields from a component class.  Results are cached.
    """
    cached = _SERIALIZED_FIELDS_CACHE.get(component_class)
    if cached is not None:
        return cached
    fields = {}
    # Use cls.__dict__ directly so each class in the MRO contributes
    # only its OWN fields (avoids inheriting a parent's empty dict).
    for cls in reversed(component_class.__mro__):
        own = cls.__dict__.get('_serialized_fields_')
        if own:
            fields.update(own)
    # Fallback: if _serialized_fields_ was cleared (e.g. by a script reload),
    # rediscover descriptors directly from the class hierarchy.
    if not fields:
        for cls in reversed(component_class.__mro__):
            for attr_name, attr in cls.__dict__.items():
                if attr_name.startswith('_'):
                    continue
                if isinstance(attr, SerializedFieldDescriptor):
                    fields[attr_name] = attr.metadata
                    continue
                if getattr(attr, '_is_cpp_property', False) and hasattr(attr, 'metadata'):
                    fields[attr_name] = attr.metadata
    _SERIALIZED_FIELDS_CACHE[component_class] = fields
    return fields


def get_field_value(component: 'InxComponent', field_name: str) -> Any:
    """Get the value of a serialized field."""
    return getattr(component, field_name)


def set_field_value(component: 'InxComponent', field_name: str, value: Any):
    """Set the value of a serialized field."""
    setattr(component, field_name, value)


class HiddenField:
    """
    Marker class for fields that should not be serialized or shown in Inspector.
    
    Use hide_field() to create instances of this class.
    """
    def __init__(self, default: Any = None):
        self.default = default
    
    def __set_name__(self, owner, name):
        self._name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        hidden_name = f'_hidden_{self._name}'
        if not hasattr(obj, hidden_name):
            try:
                value = copy.deepcopy(self.default)
            except Exception:
                value = self.default
            setattr(obj, hidden_name, value)
        return getattr(obj, hidden_name)
    
    def __set__(self, obj, value):
        setattr(obj, f'_hidden_{self._name}', value)


def hide_field(default: Any = None) -> Any:
    """
    Mark a class-level field as hidden (not serialized, not shown in Inspector).
    
    Use this for internal state that shouldn't be exposed to the editor.
    
    Args:
        default: Default value for the field
    
    Example:
        class MyComponent(InxComponent):
            speed = 5.0           # Serialized, shown in Inspector
            _internal = 0         # Not serialized (private, starts with _)
            cache = hide_field()  # Not serialized, but public API
    """
    return HiddenField(default)


def int_field(
    default: int = 0,
    *,
    range: Optional[Tuple[float, float]] = None,
    tooltip: str = "",
    readonly: bool = False,
    header: str = "",
    space: float = 0.0,
    group: str = "",
    info_text: str = "",
    slider: bool = True,
    drag_speed: Optional[float] = None,
) -> Any:
    """
    Shortcut for creating an integer serialized field.
    
    Equivalent to: serialized_field(default=..., field_type=FieldType.INT, ...)
    
    Args:
        default: Default integer value
        range: (min, max) tuple for slider / bounded drag
        tooltip: Hover text in inspector
        readonly: If True, field cannot be modified
        header: Group header text
        space: Vertical spacing before field
        group: Collapsible group name
        info_text: Non-editable description line (dimmed)
        slider: Widget style when range is set (True = slider, False = drag)
        drag_speed: Override default drag speed
    
    Example:
        class MyComponent(InxComponent):
            count = int_field(default=5, range=(0, 100))
    """
    return serialized_field(
        default=default,
        field_type=FieldType.INT,
        range=range,
        tooltip=tooltip,
        readonly=readonly,
        header=header,
        space=space,
        group=group,
        info_text=info_text,
        slider=slider,
        drag_speed=drag_speed,
    )


# ── Unified list helper ──────────────────────────────────────────────
def list_field(
    *,
    element_type: FieldType,
    element_class: Optional[Type] = None,
    component_type: Optional[str] = None,
    default: Optional[list] = None,
    tooltip: str = "",
    readonly: bool = False,
    header: str = "",
    space: float = 0.0,
    group: str = "",
    info_text: str = "",
) -> Any:
    return serialized_field(
        default=list(default) if default is not None else [],
        field_type=FieldType.LIST,
        element_type=element_type,
        element_class=element_class,
        component_type=component_type,
        tooltip=tooltip,
        readonly=readonly,
        header=header,
        space=space,
        group=group,
        info_text=info_text,
    )


# ── Layer 2 helper: ComponentRef field ────────────────────────────────

def component_field(
    component_type: str = "",
    default=None,
    **kwargs,
) -> Any:
    """Create a ComponentRef field.

    Args:
        component_type: Optional filter — only accept components of this
            type name in the Inspector drag-drop slot.
        default: Default ComponentRef (auto-created if ``None``).

    Example::

        class Follower(InxComponent):
            target = component_field(component_type="PlayerController")
    """
    from .ref_wrappers import ComponentRef

    if default is None:
        default = ComponentRef(component_type=component_type)
    return serialized_field(
        default=default,
        field_type=FieldType.COMPONENT,
        component_type=component_type,
        **kwargs,
    )


def component_list_field(
    component_type: str = "",
    default: Optional[list] = None,
    **kwargs,
) -> Any:
    """Create a list field whose elements are ComponentRef instances.

    Args:
        component_type: Optional type filter shown in the Inspector.
        default: Optional default list.

    Example::

        class TeamManager(InxComponent):
            members = component_list_field(component_type="CharacterStats")
    """
    return list_field(
        element_type=FieldType.COMPONENT,
        component_type=component_type,
        default=default,
        **kwargs,
    )
