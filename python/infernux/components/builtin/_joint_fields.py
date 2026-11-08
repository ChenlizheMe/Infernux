"""Common native joint fields, retaining each concrete type's field metadata."""
from infernux.components.builtin_component import CppProperty


def _to_native_rigidbody(value):
    if value is None:
        return None
    getter = getattr(value, "_require_cpp_component", None)
    return getter() if getter is not None else value


def _from_native_rigidbody(value):
    if value is None:
        return None
    from .rigidbody import Rigidbody
    return Rigidbody._get_or_create_wrapper(value, value.game_object)


def joint_fields(native_type):
    anchor = CppProperty.from_native(native_type, "anchor")
    connected = CppProperty.from_native(native_type, "connected_body",
        get_converter=_from_native_rigidbody, set_converter=_to_native_rigidbody)
    connected.metadata.component_type = "Rigidbody"
    return anchor, connected, CppProperty.from_native(native_type, "enable_collision")
