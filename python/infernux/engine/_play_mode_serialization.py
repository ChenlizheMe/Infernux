"""PlayModeSerializationMixin — extracted from PlayModeManager."""
from __future__ import annotations


from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.components.component import InxComponent


class PlayModeSerializationMixin:
    """PlayModeSerializationMixin method group for PlayModeManager."""

    def _serialize_py_component(self, component: 'InxComponent') -> Dict[str, Any]:
        """Serialize Python component fields and metadata.

        Uses the component's ``_serialize_value`` so that ref wrappers
        (GameObjectRef, MaterialRef) are converted to JSON-safe dicts.
        """
        from infernux.components.missing_script import MissingScript
        from infernux.components.fields import get_raw_field_value, get_serialized_fields

        field_ids = {}
        if isinstance(component, MissingScript):
            preserved = component._serialize_fields_document()
            data = {
                name: value for name, value in preserved.items()
                if name not in {"__type_name__", "__component_id__"}
            }
        else:
            fields = get_serialized_fields(component.__class__)
            data = {}
            for name, meta in fields.items():
                raw = get_raw_field_value(component, name)
                data[name] = component._serialize_value(raw)
                field_ids[name] = meta.field_id or name

        script_guid = getattr(component, "_script_guid", None)
        type_guid = component.__class__._get_type_guid()

        return {
            "type_name": getattr(component, "type_name", component.__class__.__name__),
            "script_guid": script_guid,
            "type_guid": type_guid,
            "component_id": int(getattr(component, "component_id", 0) or 0),
            "module_name": component.__class__.__module__,
            "qualified_name": component.__class__.__qualname__,
            "enabled": getattr(component, "enabled", True),
            "fields": data,
            "field_ids": field_ids,
        }

    def _apply_py_component_state(self, component: 'InxComponent', state: Dict[str, Any]):
        """Apply serialized field values to a Python component instance.

        Uses ``_deserialize_value`` so that JSON dicts produced by
        ``_serialize_py_component`` are correctly reconstructed into
        GameObjectRef / MaterialRef / enum values.
        """
        if not state or component is None:
            return
        state_script_guid = state.get("script_guid") or ""
        live_script_guid = getattr(component, "_script_guid", "") or ""
        if state_script_guid and live_script_guid and state_script_guid != live_script_guid:
            raise ValueError("Play Mode component script GUID changed during state restore")
        from infernux.components.fields import get_serialized_fields
        from infernux.components.missing_script import MissingScript

        if isinstance(component, MissingScript):
            document = dict(state.get("fields", {}))
            document["__type_name__"] = state["type_name"]
        else:
            fields_by_id = {
                meta.field_id or name: name
                for name, meta in get_serialized_fields(type(component)).items()
            }
            saved_ids = state.get("field_ids", {})
            document = {
                fields_by_id[saved_ids.get(name, name)]: value
                for name, value in state.get("fields", {}).items()
                if saved_ids.get(name, name) in fields_by_id
            }
        component_id = state.get("component_id")
        if type(component_id) is int and component_id > 0:
            document["__component_id__"] = component_id
        component._deserialize_fields_document(document, _skip_on_after_deserialize=True)
        component.enabled = bool(state.get("enabled", True))

        if state.get("script_guid"):
            component._script_guid = state.get("script_guid")

