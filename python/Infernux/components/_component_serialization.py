"""ComponentSerializationMixin — extracted from InxComponent."""
from __future__ import annotations

"""
InxComponent - Base class for all Python-defined components.

Provides Unity-style lifecycle methods and property injection.
Users inherit from this class to create custom game logic.

Example:
    from Infernux.components import InxComponent, serialized_field
    
    class PlayerController(InxComponent):
        speed: float = serialized_field(default=5.0)
        
        def start(self):
            print("Player started!")
        
        def update(self, delta_time: float):
            pos = self.transform.position
            self.transform.position = Vector3(pos.x + self.speed * delta_time, pos.y, pos.z)
"""

from typing import Optional, Dict, Any, Type, TYPE_CHECKING, List
import copy
import threading
import weakref

from Infernux.lib import GameObject


class ComponentSerializationMixin:
    """ComponentSerializationMixin method group for InxComponent."""

    def _serialize_fields(self) -> str:
        """
        Serialize all serialized fields to JSON string.
        Called by C++ PyComponentProxy::Serialize().
        
        Returns:
            JSON string of field values
        """
        import json
        from .serialized_field import get_serialized_fields
        
        # Call on_before_serialize hook
        self._call_on_before_serialize()
        
        fields = get_serialized_fields(self.__class__)
        data = {
            "__schema_version__": getattr(self, "__schema_version__", 1),
            "__type_name__": self.__class__.__name__,
            "__component_id__": self._component_id,
        }
        from .serialized_field import get_raw_field_value
        for name, meta in fields.items():
            try:
                value = get_raw_field_value(self, name)
                data[name] = self._serialize_value(value)
            except Exception as exc:
                import logging
                logging.getLogger("Infernux.serialize").error(
                    "Failed to serialize field '%s' on %s: %s",
                    name, self.__class__.__name__, exc,
                )
                data[name] = None
        
        return json.dumps(data)

    def _deserialize_fields(self, json_str: str, *, _skip_on_after_deserialize: bool = False):
        """
        Restore serialized field values from JSON string.
        Calls on_after_deserialize() after restoration unless suppressed.
        
        Args:
            json_str: JSON string of field values
            _skip_on_after_deserialize: If True, suppress the automatic
                on_after_deserialize callback.  Used during batch component
                restoration where the caller issues the callback explicitly
                after all components are attached to the scene.
        """
        import json
        from .serialized_field import get_serialized_fields
        
        data = json.loads(json_str)
        schema_version = data.get("__schema_version__", None)
        current_version = getattr(self, "__schema_version__", 1)

        if schema_version is not None and schema_version != current_version:
            # Python-side schema migration for user scripts.
            # NOTE: C++ components have a *separate* schema_version tracked
            # in Component::Serialize/Deserialize (Component.cpp).  That
            # version covers the base wire format; this Python version
            # covers per-script field layout changes.  The two systems
            # are independent — keep both in mind when adding new base fields.
            migrate = getattr(self.__class__, '__migrate__', None)
            if migrate is not None:
                try:
                    data = migrate(data, schema_version)
                except Exception as exc:
                    from Infernux.debug import Debug
                    Debug.log_error(
                        f"Schema migration failed for {self.__class__.__name__} "
                        f"(v{schema_version}→v{current_version}): {exc}"
                    )
            else:
                from Infernux.debug import Debug
                Debug.log_warning(
                    f"Component schema mismatch: {self.__class__.__name__} "
                    f"(saved={schema_version}, current={current_version}). "
                    f"Define a __migrate__(data, from_version) classmethod to handle this."
                )

        # Restore component ID if present
        saved_id = data.get("__component_id__")
        if saved_id is not None:
            self._component_id = int(saved_id)
            # Ensure ID generator is ahead of restored ID
            from Infernux.components.component import InxComponent as _InxComp
            with _InxComp._id_lock:
                if _InxComp._next_component_id <= self._component_id:
                    _InxComp._next_component_id = self._component_id + 1

        fields = get_serialized_fields(self.__class__)

        self._inf_deserializing = True
        try:
            for name, value in data.items():
                if name.startswith("__"):
                    continue
                if name in fields:
                    meta = fields[name]
                    value = self._deserialize_value(value, meta)
                    setattr(self, name, value)
        finally:
            self._inf_deserializing = False

        # Call on_after_deserialize hook
        if not _skip_on_after_deserialize:
            self._call_on_after_deserialize()

    def _serialize_value(self, value: Any):
        """Serialize a value into JSON-friendly format."""
        if isinstance(value, (bool, int, float, str, type(None))):
            return value

        # Recursively serialize list/dict so that nested refs/enums survive
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Enum support: store type name + member name
        from enum import Enum as _Enum
        if isinstance(value, _Enum):
            return {"__enum__": type(value).__qualname__, "name": value.name}

        # SerializableObject — nested data objects
        from Infernux.components.serializable_object import SerializableObject
        if isinstance(value, SerializableObject):
            return value._serialize()

        # ComponentRef — component reference
        from Infernux.components.ref_wrappers import GameObjectRef, ComponentRef
        from Infernux.core.asset_ref import MaterialRef

        if isinstance(value, ComponentRef):
            return value._serialize()

        # GameObjectRef (null-safe wrapper) — store persistent ID
        if isinstance(value, GameObjectRef):
            return {"__game_object__": value.persistent_id}

        # PrefabRef (asset reference) — store GUID + path hint
        from Infernux.components.ref_wrappers import PrefabRef
        if isinstance(value, PrefabRef):
            return value._serialize()

        # MaterialRef (null-safe wrapper) — store GUID + path_hint
        if isinstance(value, MaterialRef):
            d = {"__material_ref__": value.guid}
            if value._path_hint:
                d["__path_hint__"] = value._path_hint
            return d

        # TextureRef — store GUID + path_hint
        from Infernux.core.asset_ref import TextureRef, ShaderRef, AudioClipRef
        if isinstance(value, TextureRef):
            d: dict = {"__texture_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # ShaderRef — store GUID + path_hint
        if isinstance(value, ShaderRef):
            d = {"__shader_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # AudioClipRef — store GUID + path_hint
        if isinstance(value, AudioClipRef):
            d = {"__audio_clip_ref__": value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

        # Generic AssetRefBase (AnimStateMachineRef, AnimationClipRef, etc.)
        # — delegate to the shared registry-aware helper.
        from Infernux.core.asset_ref import AssetRefBase
        if isinstance(value, AssetRefBase):
            from ._serialize_helpers import _serialize_asset_ref
            ref_dict = _serialize_asset_ref(value)
            if ref_dict is not None:
                return ref_dict

        # Raw GameObject reference — store persistent ID (scene-stable)
        if hasattr(value, 'id') and hasattr(value, 'name') and hasattr(value, 'transform'):
            return {"__game_object__": int(value.id)}

        # Raw Material reference — store GUID via AssetDatabase
        try:
            from Infernux.core.material import Material
            if isinstance(value, Material):
                guid = MaterialRef._extract_guid(value)
                if guid:
                    return {"__material_ref__": guid}
                # Fallback: store path if no GUID available
                path = getattr(value.native, 'file_path', '') if value.native else ''
                return {"__material_ref__": path or value.name}
        except ImportError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        # Vec2/3/4 support (pybind types expose x/y/z/w)
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and hasattr(value, "w"):
            return [float(value.x), float(value.y), float(value.z), float(value.w)]
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            return [float(value.x), float(value.y), float(value.z)]
        if hasattr(value, "x") and hasattr(value, "y"):
            return [float(value.x), float(value.y)]

        # Fallback: warn and return None (do NOT silently degrade to str)
        import logging
        logging.getLogger("Infernux.serialize").warning(
            "Cannot serialize value of type %s — returning None. "
            "Define a SerializableObject or register a custom adapter.",
            type(value).__name__,
        )
        return None

    def _deserialize_value(self, value: Any, field_meta_or_type):
        """Deserialize a value based on FieldType."""
        from Infernux.components.serialized_field import FieldType
        from Infernux.components._serialize_helpers import make_null_ref, deserialize_dict_ref
        from Infernux.math import Vector2, Vector3, vec4f

        if hasattr(field_meta_or_type, 'field_type'):
            field_type = field_meta_or_type.field_type
            element_type = getattr(field_meta_or_type, 'element_type', None)
        else:
            field_type = field_meta_or_type
            element_type = None

        # Null values for ref types → return a null ref wrapper (not raw None)
        if value is None:
            return make_null_ref(field_type, field_meta_or_type)

        if field_type == FieldType.SERIALIZABLE_OBJECT:
            if isinstance(value, dict):
                from Infernux.components.serializable_object import SerializableObject
                if '__serializable_type__' in value:
                    return SerializableObject._deserialize(value)
                so_cls = getattr(field_meta_or_type, 'serializable_class', None)
                if so_cls:
                    return so_cls._deserialize(value)
            return value

        if field_type == FieldType.COMPONENT:
            if isinstance(value, dict) and '__component_ref__' in value:
                from Infernux.components.ref_wrappers import ComponentRef
                return ComponentRef._from_dict(value['__component_ref__'])
            return value

        if field_type == FieldType.LIST:
            if not isinstance(value, list):
                return []
            if element_type == FieldType.SERIALIZABLE_OBJECT:
                from Infernux.components.serializable_object import SerializableObject
                elem_cls = getattr(field_meta_or_type, 'element_class', None)
                result = []
                for item in value:
                    if isinstance(item, dict):
                        if '__serializable_type__' in item:
                            result.append(SerializableObject._deserialize(item))
                        elif elem_cls:
                            result.append(elem_cls._deserialize(item))
                        else:
                            result.append(item)
                    else:
                        result.append(item)
                return result
            return [self._deserialize_value(item, element_type or FieldType.UNKNOWN) for item in value]

        if field_type == FieldType.VEC2:
            return self._to_vec(value, 2, Vector2)
        if field_type == FieldType.VEC3:
            return self._to_vec(value, 3, Vector3)
        if field_type == FieldType.VEC4:
            return self._to_vec(value, 4, vec4f)
        if field_type == FieldType.ENUM and isinstance(value, dict) and "__enum__" in value:
            return self._deserialize_enum(value)

        if isinstance(value, dict):
            return deserialize_dict_ref(value)

        return value

    def _deserialize_enum(self, value: dict):
        """Reconstruct an enum member from {__enum__, name}."""
        from .serialized_field import get_serialized_fields
        fields = get_serialized_fields(self.__class__)
        for meta in fields.values():
            if meta.enum_type is not None and meta.enum_type.__qualname__ == value["__enum__"]:
                return meta.enum_type[value["name"]]
        return value  # fallback: return dict as-is

    def _to_vec(self, value: Any, n: int, ctor):
        """Convert list/tuple/string to vec type if possible."""
        if isinstance(value, (list, tuple)) and len(value) >= n:
            return ctor(*[float(value[i]) for i in range(n)])
        if isinstance(value, str):
            cleaned = value.strip().replace("<", "").replace(">", "").replace("(", "").replace(")", "")
            parts = [p for p in cleaned.split(",") if p.strip()]
            if len(parts) >= n:
                nums = [float(p.strip()) for p in parts[:n]]
                return ctor(*nums)
        return value

