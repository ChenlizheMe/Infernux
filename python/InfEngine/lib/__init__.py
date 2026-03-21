import os
import sys
from functools import wraps

lib_dir = os.path.join(os.path.dirname(__file__))
lib_dir = os.path.abspath(lib_dir)

sys.path.insert(0, lib_dir)

if sys.platform == "win32":
    os.add_dll_directory(lib_dir)
    os.environ["PATH"] = lib_dir + ";" + os.environ["PATH"]
else:
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from ._InfEngine import *


_INVALID_NATIVE_LIFETIME_MARKERS = (
    "access violation",
    "rtti",
    "null pointer",
    "instance is null",
    "has been destroyed",
    "use after free",
)


def _is_native_lifetime_error(exc) -> bool:
    """Return True when *exc* looks like a stale native-object access."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).strip().lower()
    return any(marker in message for marker in _INVALID_NATIVE_LIFETIME_MARKERS)


def _zero_vec3():
    return Vector3(0.0, 0.0, 0.0)


def _one_vec3():
    return Vector3(1.0, 1.0, 1.0)


def _identity_quat():
    return quatf(0.0, 0.0, 0.0, 1.0)


def _identity_matrix4x4():
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _native_safe_default(obj, name: str):
    """Return a conservative fallback for invalid native-object access."""
    if name in {"id", "component_id", "game_object_id", "child_count", "get_child_count"}:
        return 0
    if name in {"active", "enabled", "has_changed", "is_trigger", "is_active_in_hierarchy", "is_child_of"}:
        return False
    if name in {"name", "type_name"}:
        return ""
    if name in {"transform", "get_transform", "game_object", "get_parent", "parent", "root", "get_component",
                "get_cpp_component", "get_py_component", "get_child", "find", "collider"}:
        return None
    if name in {"get_components", "get_cpp_components", "get_py_components", "get_children"}:
        return []
    if name in {"serialize"}:
        return "{}"
    if name in {"deserialize", "remove_component", "remove_py_component"}:
        return False
    if name in {"position", "local_position", "euler_angles", "local_euler_angles", "forward", "up", "right",
                "local_forward", "local_up", "local_right", "contact_point", "contact_normal", "relative_velocity",
                "point", "normal"}:
        return _zero_vec3()
    if name in {"local_scale", "lossy_scale"}:
        return _one_vec3()
    if name in {"rotation", "local_rotation"}:
        return _identity_quat()
    if name in {"local_to_world_matrix", "world_to_local_matrix"}:
        return _identity_matrix4x4()
    if name in {"distance", "impulse"}:
        return 0.0

    if name.startswith("get_") and name.endswith("s"):
        return []
    if name.startswith("get_"):
        return None
    if name.startswith(("is_", "has_")):
        return False
    if name.startswith(("set_", "add_", "move_", "wake_", "sleep", "look_", "translate", "rotate", "detach_", "clear")):
        return None
    if name.startswith("remove_"):
        return False
    return None


def _wrap_native_callable(obj, name: str, func):
    @wraps(func)
    def _guarded(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            if _is_native_lifetime_error(exc):
                return _native_safe_default(obj, name)
            raise

    setattr(_guarded, "_infengine_native_guarded", True)
    return _guarded


def _install_native_lifetime_guard(cls) -> None:
    """Patch a pybind class so stale native pointers fail safely in Python."""
    if getattr(cls, "_infengine_native_lifetime_guard_installed", False):
        return

    original_getattribute = cls.__getattribute__
    original_setattr = cls.__setattr__

    def _guarded_getattribute(self, name):
        try:
            value = original_getattribute(self, name)
        except RuntimeError as exc:
            if _is_native_lifetime_error(exc):
                return _native_safe_default(self, name)
            raise

        if name.startswith("__"):
            return value
        if callable(value) and not getattr(value, "_infengine_native_guarded", False):
            return _wrap_native_callable(self, name, value)
        return value

    def _guarded_setattr(self, name, value):
        try:
            return original_setattr(self, name, value)
        except RuntimeError as exc:
            if _is_native_lifetime_error(exc):
                return None
            raise

    def _guarded_bool(self):
        try:
            identifier = _guarded_getattribute(self, "id")
        except AttributeError:
            identifier = 0
        if identifier is None or identifier == 0:
            try:
                identifier = _guarded_getattribute(self, "component_id")
            except AttributeError:
                identifier = 0
        return bool(identifier)

    cls.__getattribute__ = _guarded_getattribute
    cls.__setattr__ = _guarded_setattr
    cls.__bool__ = _guarded_bool
    cls._infengine_native_lifetime_guard_installed = True


for _native_cls in (GameObject, Component, Transform, RaycastHit, CollisionInfo):
    _install_native_lifetime_guard(_native_cls)