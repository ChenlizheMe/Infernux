import ctypes
import glob
import os
import sys
from functools import wraps

lib_dir = os.path.join(os.path.dirname(__file__))
lib_dir = os.path.abspath(lib_dir)

_dll_dir_handles = []


def _register_native_search_dir(path: str) -> None:
    if not path or not os.path.isdir(path):
        return

    norm = os.path.abspath(path)
    if norm not in sys.path:
        sys.path.insert(0, norm)

    if sys.platform == "win32":
        handle = os.add_dll_directory(norm)
        _dll_dir_handles.append(handle)
        path_entries = os.environ.get("PATH", "").split(";") if os.environ.get("PATH") else []
        if norm not in path_entries:
            os.environ["PATH"] = norm + (";" + os.environ["PATH"] if os.environ.get("PATH") else "")
    elif sys.platform == "darwin":
        dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
        parts = dyld_path.split(":") if dyld_path else []
        if norm not in parts:
            os.environ["DYLD_LIBRARY_PATH"] = norm + ((":" + dyld_path) if dyld_path else "")
    else:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        parts = ld_path.split(":") if ld_path else []
        if norm not in parts:
            os.environ["LD_LIBRARY_PATH"] = norm + ((":" + ld_path) if ld_path else "")


def _iter_dev_native_search_dirs():
    repo_root = os.path.abspath(os.path.join(lib_dir, "..", "..", ".."))
    build_root = os.path.join(repo_root, "out", "build")
    configs = ("RelWithDebInfo", "Release", "Debug")

    for config in configs:
        yield os.path.join(build_root, config)

    externals = (
        ("external", "assimp", "bin"),
        ("external", "glslang", "glslang"),
        ("external", "JoltPhysics"),
        ("external", "SDL"),
    )
    for prefix in externals:
        for config in configs:
            yield os.path.join(build_root, *prefix, config)


def _collect_windows_native_load_hints():
    if sys.platform != "win32":
        return []

    hints = []
    checks = (
        ("MSVCP140.dll", "Install or repair the Microsoft Visual C++ Redistributable."),
        ("VCRUNTIME140.dll", "Install or repair the Microsoft Visual C++ Redistributable."),
        ("VCRUNTIME140_1.dll", "Install or repair the Microsoft Visual C++ Redistributable."),
        ("vulkan-1.dll", "Install a current GPU driver or the Vulkan Runtime."),
    )

    for dll_name, remedy in checks:
        try:
            ctypes.WinDLL(dll_name)
        except OSError:
            hints.append(f"Missing {dll_name}. {remedy}")

    if not glob.glob(os.path.join(lib_dir, "_Infernux*.pyd")):
        hints.append(f"Missing _Infernux*.pyd under {lib_dir}. Reinstall the Infernux wheel.")

    return hints


def _raise_native_import_error(exc):
    lines = [
        "Failed to load the Infernux native module.",
        f"Library directory: {lib_dir}",
        f"Original error: {exc}",
    ]

    hints = _collect_windows_native_load_hints()
    if hints:
        lines.append("Likely causes:")
        lines.extend(f"- {hint}" for hint in hints)
    elif sys.platform == "darwin":
        if not glob.glob(os.path.join(lib_dir, "_Infernux*.so")):
            lines.append(f"Missing _Infernux*.so under {lib_dir}. Build the native module first.")
        lines.append(
            "Likely causes: missing Vulkan SDK (MoltenVK), or the native module was not built for this architecture."
        )
    elif sys.platform == "win32":
        lines.append(
            "Likely causes: a missing Vulkan runtime or missing Microsoft Visual C++ runtime DLLs."
        )

    raise ImportError("\n".join(lines)) from exc


def _preload_bundled_crt_dlls() -> None:
    """Pre-load MSVC CRT DLLs bundled alongside ``_Infernux.pyd``.

    On machines without a system-wide Visual C++ Redistributable install,
    ``os.add_dll_directory()`` alone is not always sufficient —
    ``_Infernux.pyd`` (and the engine DLLs it depends on) may still fail
    to resolve ``vcruntime140.dll`` / ``msvcp140.dll`` at load time.

    Explicitly loading them via ``ctypes.WinDLL`` before the ``from
    ._Infernux import *`` guarantees they are resident in the process
    and the dynamic linker can satisfy the dependency.
    """
    if sys.platform != "win32":
        return

    # Order matters: vcruntime first, then msvcp / concrt (they depend
    # on vcruntime).
    _CRT_LOAD_ORDER = (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "concrt140.dll",
    )

    for name in _CRT_LOAD_ORDER:
        full = os.path.join(lib_dir, name)
        if os.path.isfile(full):
            try:
                ctypes.WinDLL(full)
            except OSError:
                pass  # Best-effort; the import below will give a clear error.


_register_native_search_dir(lib_dir)
_preload_bundled_crt_dlls()

try:
    from ._Infernux import *
except (ModuleNotFoundError, ImportError):
    for candidate in _iter_dev_native_search_dirs():
        _register_native_search_dir(candidate)
    try:
        from ._Infernux import *
    except (ModuleNotFoundError, ImportError) as exc:
        _raise_native_import_error(exc)


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

    setattr(_guarded, "_infernux_native_guarded", True)
    return _guarded


def _install_native_lifetime_guard(cls) -> None:
    """Patch a pybind class so stale native pointers fail safely in Python."""
    if getattr(cls, "_infernux_native_lifetime_guard_installed", False):
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
        if callable(value) and not getattr(value, "_infernux_native_guarded", False):
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
    cls._infernux_native_lifetime_guard_installed = True


for _native_cls in (GameObject, Component, Transform, RaycastHit, CollisionInfo):
    _install_native_lifetime_guard(_native_cls)


_native_game_object_add_component = GameObject.add_component
_native_game_object_remove_component = GameObject.remove_component
_native_game_object_can_remove_component = GameObject.can_remove_component
_native_game_object_get_remove_component_blockers = GameObject.get_remove_component_blockers
_native_game_object_get_component = GameObject.get_component
_native_game_object_get_components = GameObject.get_components
_native_game_object_get_component_in_children = GameObject.get_component_in_children
_native_game_object_get_component_in_parent = GameObject.get_component_in_parent
_native_game_object_instantiate = GameObject.instantiate


def _call_native_game_object(method_name: str, native_method, game_object, *args):
    try:
        return native_method(game_object, *args)
    except RuntimeError as exc:
        if _is_native_lifetime_error(exc):
            return _native_safe_default(game_object, method_name)
        raise


def _is_vector3_like(value) -> bool:
    return value is not None and hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and not hasattr(value, "w")


def _is_quat_like(value) -> bool:
    return value is not None and hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and hasattr(value, "w")


def _resolve_game_object_instantiate_source(original):
    if isinstance(original, GameObject):
        return "game_object", original

    try:
        from Infernux.components.ref_wrappers import GameObjectRef, PrefabRef
        if isinstance(original, PrefabRef):
            return "prefab", original
        if isinstance(original, GameObjectRef):
            return "game_object", original.resolve()
    except Exception:
        pass

    resolver = getattr(original, "resolve", None)
    if callable(resolver):
        try:
            resolved = resolver()
        except Exception:
            resolved = None
        if isinstance(resolved, GameObject):
            return "game_object", resolved

    return "game_object", None


def _coerce_parent_game_object(parent):
    if parent is None:
        return None
    if isinstance(parent, GameObject):
        return parent

    try:
        from Infernux.components.ref_wrappers import GameObjectRef
        if isinstance(parent, GameObjectRef):
            return parent.resolve()
    except Exception:
        pass

    game_object = getattr(parent, "game_object", None)
    if isinstance(game_object, GameObject):
        return game_object

    raise TypeError(
        "instantiate(): parent must be a GameObject, Transform, GameObjectRef, or None"
    )


def _instantiate_prefab_reference(prefab_ref):
    current_path = getattr(prefab_ref, "path_hint", "")
    guid = getattr(prefab_ref, "guid", "")
    if not guid and not current_path:
        return None

    from Infernux.engine.prefab_manager import instantiate_prefab

    if guid:
        adb = None
        registry = AssetRegistry.instance()
        if registry:
            adb = registry.get_asset_database()
        result = instantiate_prefab(guid=guid, asset_database=adb)
        if result is not None:
            return result

    if current_path and os.path.isfile(current_path):
        return instantiate_prefab(file_path=current_path)

    return None


def _capture_local_transform(game_object):
    if game_object is None:
        return None
    try:
        transform = game_object.transform
        return transform.local_position, transform.local_rotation, transform.local_scale
    except Exception:
        return None


def _restore_local_transform(game_object, local_transform):
    if game_object is None or local_transform is None:
        return
    local_position, local_rotation, local_scale = local_transform
    try:
        transform = game_object.transform
        transform.local_position = local_position
        transform.local_rotation = local_rotation
        transform.local_scale = local_scale
    except Exception:
        return


def _parse_instantiate_arguments(args, kwargs):
    if len(args) > 3:
        raise TypeError("instantiate(): expected at most 4 arguments including original")

    position = kwargs.pop("position", None)
    rotation = kwargs.pop("rotation", None)
    parent = kwargs.pop("parent", None)
    instantiate_in_world_space = kwargs.pop("instantiate_in_world_space", kwargs.pop("instantiateInWorldSpace", None))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"instantiate(): unexpected keyword arguments: {unexpected}")

    if len(args) == 1:
        parent = args[0]
        if instantiate_in_world_space is None:
            instantiate_in_world_space = False
    elif len(args) == 2:
        if _is_vector3_like(args[0]) and _is_quat_like(args[1]):
            position, rotation = args
        else:
            parent = args[0]
            instantiate_in_world_space = args[1]
    elif len(args) == 3:
        position, rotation, parent = args
        if instantiate_in_world_space is None:
            instantiate_in_world_space = True

    if position is not None and not _is_vector3_like(position):
        raise TypeError("instantiate(): position must be a Vector3")
    if rotation is not None and not _is_quat_like(rotation):
        raise TypeError("instantiate(): rotation must be a quatf")
    if instantiate_in_world_space is None:
        instantiate_in_world_space = True
    if not isinstance(instantiate_in_world_space, bool):
        raise TypeError("instantiate(): instantiate_in_world_space must be a bool")

    return position, rotation, parent, instantiate_in_world_space


def _game_object_instantiate(original, *args, **kwargs):
    position, rotation, parent_arg, instantiate_in_world_space = _parse_instantiate_arguments(args, kwargs)
    parent = _coerce_parent_game_object(parent_arg) if parent_arg is not None else None

    source_kind, source = _resolve_game_object_instantiate_source(original)
    if source_kind == "prefab":
        instance = _instantiate_prefab_reference(source)
        source_local_transform = None
    else:
        if source is None:
            return None
        source_local_transform = _capture_local_transform(source)
        instance = _call_native_game_object("instantiate", _native_game_object_instantiate, source, None)

    if instance is None:
        return None

    if parent is not None:
        instance.set_parent(parent, instantiate_in_world_space)
        if not instantiate_in_world_space and source_local_transform is not None:
            _restore_local_transform(instance, source_local_transform)

    if position is not None:
        instance.transform.position = position
    if rotation is not None:
        instance.transform.rotation = rotation

    return instance


def _resolve_component_api_types():
    from Infernux.components.component import InxComponent
    from Infernux.components.builtin_component import BuiltinComponent
    from Infernux.components.registry import get_type

    return InxComponent, BuiltinComponent, get_type


def _resolve_builtin_wrapper(component_type):
    _, BuiltinComponent, _ = _resolve_component_api_types()

    if isinstance(component_type, type) and issubclass(component_type, BuiltinComponent):
        return component_type
    if isinstance(component_type, str):
        return BuiltinComponent._builtin_registry.get(component_type)

    cpp_type_name = getattr(component_type, "_cpp_type_name", "")
    if cpp_type_name:
        return BuiltinComponent._builtin_registry.get(cpp_type_name)

    type_name = getattr(component_type, "__name__", "")
    if type_name:
        return BuiltinComponent._builtin_registry.get(type_name)

    return None


def _resolve_python_component_class(component_type):
    InxComponent, BuiltinComponent, get_type = _resolve_component_api_types()

    if isinstance(component_type, type) and issubclass(component_type, InxComponent):
        if issubclass(component_type, BuiltinComponent):
            return None
        return component_type

    if isinstance(component_type, str):
        component_cls = get_type(component_type)
        if component_cls is not None and not issubclass(component_cls, BuiltinComponent):
            return component_cls

    return None


def _find_python_component_by_name(game_object, type_name: str):
    for component in game_object.get_py_components() or []:
        component_name = getattr(component.__class__, "__name__", "")
        if component_name == type_name or getattr(component, "type_name", "") == type_name:
            return component
    return None


def _find_python_components_by_name(game_object, type_name: str):
    return [
        component
        for component in (game_object.get_py_components() or [])
        if getattr(component.__class__, "__name__", "") == type_name
        or getattr(component, "type_name", "") == type_name
    ]


def _wrap_builtin_component(game_object, wrapper_cls, cpp_component):
    if cpp_component is None:
        return None
    return wrapper_cls._get_or_create_wrapper(cpp_component, game_object)


def _wrap_builtin_component_list(game_object, wrapper_cls, cpp_components):
    return [wrapper_cls._get_or_create_wrapper(component, game_object) for component in cpp_components]


def _resolve_public_component(component):
    if component is None:
        return None

    py_component_getter = getattr(component, "get_py_component", None)
    if callable(py_component_getter):
        try:
            return py_component_getter()
        except RuntimeError as exc:
            if _is_native_lifetime_error(exc):
                return None
            raise

    return component


def _unwrap_component_argument(game_object, component):
    native_getter = getattr(component, "_get_bound_native_component", None)
    if callable(native_getter):
        native_component = native_getter()
        if native_component is not None:
            return (native_component, False)

    for py_component in game_object.get_py_components() or []:
        if py_component is component:
            return (component, True)

    return (component, False)


def _game_object_add_component(self, component_type):
    python_component_cls = _resolve_python_component_class(component_type)
    if python_component_cls is not None:
        return self.add_py_component(python_component_cls())

    builtin_wrapper_cls = _resolve_builtin_wrapper(component_type)
    if builtin_wrapper_cls is not None:
        cpp_type_name = getattr(builtin_wrapper_cls, "_cpp_type_name", builtin_wrapper_cls.__name__)
        cpp_component = _call_native_game_object(
            "add_component", _native_game_object_add_component, self, cpp_type_name
        )
        return _wrap_builtin_component(self, builtin_wrapper_cls, cpp_component)

    return _call_native_game_object("add_component", _native_game_object_add_component, self, component_type)


def _game_object_get_component(self, component_type):
    builtin_wrapper_cls = _resolve_builtin_wrapper(component_type)
    if builtin_wrapper_cls is not None:
        cpp_type_name = getattr(builtin_wrapper_cls, "_cpp_type_name", builtin_wrapper_cls.__name__)
        cpp_component = self.get_cpp_component(cpp_type_name)
        return _wrap_builtin_component(self, builtin_wrapper_cls, cpp_component)

    python_component_cls = _resolve_python_component_class(component_type)
    if python_component_cls is not None:
        return self.get_py_component(python_component_cls)

    if isinstance(component_type, str):
        python_component = _find_python_component_by_name(self, component_type)
        if python_component is not None:
            return python_component

    resolved_type_name = getattr(component_type, "_cpp_type_name", "") or getattr(component_type, "__name__", "")
    if resolved_type_name:
        return _call_native_game_object("get_component", _native_game_object_get_component, self, resolved_type_name)

    return _call_native_game_object("get_component", _native_game_object_get_component, self, component_type)


def _game_object_get_components(self, component_type=None):
    if component_type is None:
        raw_components = _call_native_game_object("get_components", _native_game_object_get_components, self)
        public_components = []
        for component in raw_components or []:
            public_component = _resolve_public_component(component)
            if public_component is not None:
                public_components.append(public_component)
        return public_components

    builtin_wrapper_cls = _resolve_builtin_wrapper(component_type)
    if builtin_wrapper_cls is not None:
        cpp_type_name = getattr(builtin_wrapper_cls, "_cpp_type_name", builtin_wrapper_cls.__name__)
        cpp_components = self.get_cpp_components(cpp_type_name)
        return _wrap_builtin_component_list(self, builtin_wrapper_cls, cpp_components)

    python_component_cls = _resolve_python_component_class(component_type)
    if python_component_cls is not None:
        return [component for component in (self.get_py_components() or []) if isinstance(component, python_component_cls)]

    if isinstance(component_type, str):
        python_components = _find_python_components_by_name(self, component_type)
        if python_components:
            return python_components
        return self.get_cpp_components(component_type)

    type_name = getattr(component_type, "__name__", "")
    if type_name:
        return self.get_cpp_components(type_name)
    return []


def _game_object_get_component_in_children(self, component_type, include_inactive=False):
    result = _call_native_game_object(
        "get_component_in_children",
        _native_game_object_get_component_in_children,
        self,
        component_type,
        include_inactive,
    )
    builtin_wrapper_cls = _resolve_builtin_wrapper(component_type)
    if builtin_wrapper_cls is None:
        return result
    result_game_object = getattr(result, "game_object", self)
    return _wrap_builtin_component(result_game_object, builtin_wrapper_cls, result)


def _game_object_get_component_in_parent(self, component_type, include_inactive=False):
    result = _call_native_game_object(
        "get_component_in_parent",
        _native_game_object_get_component_in_parent,
        self,
        component_type,
        include_inactive,
    )
    builtin_wrapper_cls = _resolve_builtin_wrapper(component_type)
    if builtin_wrapper_cls is None:
        return result
    result_game_object = getattr(result, "game_object", self)
    return _wrap_builtin_component(result_game_object, builtin_wrapper_cls, result)


def _game_object_remove_component(self, component):
    unwrapped_component, is_python_component = _unwrap_component_argument(self, component)
    if is_python_component:
        return self.remove_py_component(unwrapped_component)
    return _call_native_game_object("remove_component", _native_game_object_remove_component, self, unwrapped_component)


def _game_object_can_remove_component(self, component):
    unwrapped_component, is_python_component = _unwrap_component_argument(self, component)
    if is_python_component:
        return True
    return _call_native_game_object(
        "can_remove_component", _native_game_object_can_remove_component, self, unwrapped_component
    )


def _game_object_get_remove_component_blockers(self, component):
    unwrapped_component, is_python_component = _unwrap_component_argument(self, component)
    if is_python_component:
        return []
    return _call_native_game_object(
        "get_remove_component_blockers",
        _native_game_object_get_remove_component_blockers,
        self,
        unwrapped_component,
    )


GameObject.add_component = _game_object_add_component
GameObject.remove_component = _game_object_remove_component
GameObject.can_remove_component = _game_object_can_remove_component
GameObject.get_remove_component_blockers = _game_object_get_remove_component_blockers
GameObject.get_component = _game_object_get_component
GameObject.get_components = _game_object_get_components
GameObject.get_component_in_children = _game_object_get_component_in_children
GameObject.get_component_in_parent = _game_object_get_component_in_parent
GameObject.instantiate = staticmethod(_game_object_instantiate)


def _game_object_self_alias(self):
    object_id = int(getattr(self, "id", 0) or 0)
    if object_id <= 0:
        return None
    return self


GameObject.game_object = property(
    _game_object_self_alias,
    doc="Unity-style self alias so GameObject fields can be accessed via .game_object.",
)