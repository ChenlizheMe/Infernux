"""
Null-safe reference wrappers for GameObject and Component.

These wrappers track the validity of references so that accessing a
destroyed/missing object returns ``None`` instead of crashing with a
C++ exception.

``GameObjectRef`` stores a persistent scene-ID and lazily resolves the
live object via ``Scene.find_by_id``.  If the object has been destroyed
the wrapper evaluates to falsy and all attribute access returns ``None``.

``MaterialRef`` is defined in ``InfEngine.core.asset_ref`` and re-exported
here for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# Re-export MaterialRef from core so existing callers still work.
from InfEngine.core.asset_ref import MaterialRef  # noqa: F401

_log = logging.getLogger("InfEngine.ref")


# ============================================================================
# GameObjectRef — Null-safe, persistent-ID based reference
# ============================================================================

class GameObjectRef:
    """Null-safe wrapper around a scene GameObject.

    Stores the persistent ``id`` (uint64, written into the .scene file) and
    lazily resolves the live C++ object each time it is accessed.  If the
    object has been destroyed or the scene was reloaded, the wrapper simply
    returns ``None`` instead of raising a pybind11 segfault.

    Supports truthiness check::

        if self.target:   # False when target is None or destroyed
            self.target.name
    """

    __slots__ = ("_persistent_id", "_cached_obj")

    def __init__(self, game_object=None, *, persistent_id: int = 0):
        if game_object is not None:
            self._persistent_id: int = int(game_object.id)
            self._cached_obj = game_object
        else:
            self._persistent_id = int(persistent_id)
            self._cached_obj = None

    # -- resolution --------------------------------------------------------

    def _resolve(self):
        """Try to resolve the live object from the current scene."""
        if self._persistent_id == 0:
            self._cached_obj = None
            return None
        try:
            from InfEngine.lib import SceneManager as _SM
            scene = _SM.instance().get_active_scene()
            if scene is not None:
                obj = scene.find_by_id(self._persistent_id)
                self._cached_obj = obj
                return obj
        except (ImportError, RuntimeError):
            pass  # pybind11 raises RuntimeError when C++ object is destroyed
        except Exception as exc:
            _log.warning("GameObjectRef._resolve failed: %s", exc)
        self._cached_obj = None
        return None

    # -- public API --------------------------------------------------------

    @property
    def persistent_id(self) -> int:
        """The persistent ID stored in the scene file."""
        return self._persistent_id

    def resolve(self):
        """Return the live GameObject, or ``None`` if destroyed/missing."""
        obj = self._cached_obj
        # Quick validity check: the C++ side exposes `.id`; if it throws
        # the wrapper has been invalidated.
        if obj is not None:
            try:
                _ = obj.id
                return obj
            except RuntimeError:
                self._cached_obj = None
        return self._resolve()

    def __copy__(self):
        return type(self)(persistent_id=self._persistent_id)

    def __deepcopy__(self, memo):
        copied = type(self)(persistent_id=self._persistent_id)
        memo[id(self)] = copied
        return copied

    # -- convenience attribute forwarding ----------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying GameObject."""
        # Avoid infinite recursion for our own slots
        if name.startswith("_"):
            raise AttributeError(name)
        obj = self.resolve()
        if obj is None:
            return None
        return getattr(obj, name)

    def __bool__(self) -> bool:
        return self.resolve() is not None

    def __eq__(self, other):
        if other is None:
            return self._persistent_id == 0
        if isinstance(other, GameObjectRef):
            return self._persistent_id == other._persistent_id
        # Compare to raw GameObject
        if hasattr(other, "id"):
            return self._persistent_id == other.id
        return NotImplemented

    def __hash__(self):
        return hash(self._persistent_id)

    def __repr__(self):
        obj = self.resolve()
        if obj is not None:
            return f"GameObjectRef('{obj.name}', id={self._persistent_id})"
        return f"GameObjectRef(None, id={self._persistent_id})"


# ============================================================================
# ComponentRef — Null-safe, persistent-ID based component reference
# ============================================================================

class ComponentRef:
    """Null-safe reference to a component on a specific GameObject.

    Stores the target GameObject's persistent ID and the component type
    name.  Lazily resolves the live component instance at access time.

    Usage::

        class Follower(InfComponent):
            target = component_field(component_type="PlayerController")

            def update(self, dt):
                ctrl = self.target.resolve()
                if ctrl:
                    ctrl.do_something()

    Serialization format::

        {"__component_ref__": {"go_id": 12345, "type_name": "PlayerController"}}
    """

    __slots__ = ("_go_id", "_component_type", "_cached")

    def __init__(self, *, go_id: int = 0, component_type: str = ""):
        self._go_id: int = int(go_id)
        self._component_type: str = component_type
        self._cached = None

    # -- resolution --------------------------------------------------------

    def _resolve(self):
        """Try to resolve the live component from the current scene."""
        if self._go_id == 0:
            self._cached = None
            return None

        try:
            from InfEngine.lib import SceneManager as _SM
            scene = _SM.instance().get_active_scene()
            if scene is None:
                self._cached = None
                return None

            go = scene.find_by_id(self._go_id)
            if go is None:
                self._cached = None
                return None

            # Search via type map (O(1) hash lookup)
            if self._component_type:
                from .component import InfComponent
                tmap = InfComponent._type_map.get(self._go_id, {})
                found = tmap.get(self._component_type)
                if found is not None and not getattr(found, '_is_destroyed', False):
                    self._cached = found
                    return found

                # Fallback: lazy-create BuiltinComponent wrapper if C++ has
                # the component but no Python wrapper exists yet.
                from .builtin_component import BuiltinComponent
                builtin_cls = BuiltinComponent._builtin_registry.get(self._component_type)
                if builtin_cls is not None:
                    cpp_type = getattr(builtin_cls, '_cpp_type_name', self._component_type)
                    cpp_comp = go.get_cpp_component(cpp_type)
                    if cpp_comp is not None:
                        wrapper = builtin_cls._get_or_create_wrapper(cpp_comp, go)
                        self._cached = wrapper
                        return wrapper
            else:
                # No type filter — return the first Python component on this GO
                from .component import InfComponent
                py_comps = InfComponent._active_instances.get(self._go_id, [])
                if py_comps:
                    self._cached = py_comps[0]
                    return py_comps[0]

            self._cached = None
            return None
        except (ImportError, RuntimeError) as exc:
            _log.warning("ComponentRef._resolve failed: %s", exc)
            self._cached = None
            return None

    def resolve(self):
        """Return the live component instance, or ``None`` if unavailable."""
        # Quick validity check on cached value
        if self._cached is not None:
            try:
                if hasattr(self._cached, '_is_destroyed') and self._cached._is_destroyed:
                    self._cached = None
                else:
                    return self._cached
            except (RuntimeError, AttributeError):
                self._cached = None
        return self._resolve()

    def __copy__(self):
        return type(self)(go_id=self._go_id, component_type=self._component_type)

    def __deepcopy__(self, memo):
        copied = type(self)(go_id=self._go_id, component_type=self._component_type)
        memo[id(self)] = copied
        return copied

    # -- public properties -------------------------------------------------

    @property
    def go_id(self) -> int:
        return self._go_id

    @property
    def component_type(self) -> str:
        return self._component_type

    @property
    def display_name(self) -> str:
        """Human-readable label for Inspector display."""
        comp = self.resolve()
        if comp is None:
            return "None"
        # Try to get the GO name
        go_name = ""
        if hasattr(comp, 'game_object') and comp.game_object:
            go_name = getattr(comp.game_object, 'name', '')
        elif hasattr(comp, 'name'):
            go_name = comp.name or ""
        type_name = self._component_type or type(comp).__name__
        if go_name:
            return f"{type_name} ({go_name})"
        return type_name

    # -- serialization -----------------------------------------------------

    def _serialize(self) -> dict:
        return {
            "__component_ref__": {
                "go_id": self._go_id,
                "type_name": self._component_type,
            }
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "ComponentRef":
        return cls(
            go_id=int(data.get("go_id", 0)),
            component_type=str(data.get("type_name", "")),
        )

    # -- dunder helpers ----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying component.

        This makes ``.resolve()`` unnecessary for most use cases::

            # Before: ctrl = self.target.resolve(); ctrl.do_something()
            # After:  self.target.do_something()
        """
        if name.startswith("_"):
            raise AttributeError(name)
        comp = self.resolve()
        if comp is None:
            return None
        return getattr(comp, name)

    def __bool__(self) -> bool:
        return self.resolve() is not None

    def __eq__(self, other):
        if other is None:
            return self._go_id == 0
        if isinstance(other, ComponentRef):
            return (self._go_id == other._go_id
                    and self._component_type == other._component_type)
        return NotImplemented

    def __hash__(self):
        return hash((self._go_id, self._component_type))

    def __repr__(self):
        comp = self.resolve()
        if comp is not None:
            return f"ComponentRef({self._component_type}, go_id={self._go_id})"
        return f"ComponentRef(None, type={self._component_type}, go_id={self._go_id})"
