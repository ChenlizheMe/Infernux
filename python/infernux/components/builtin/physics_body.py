"""Public base of built-in components that own a physical backend."""

from abc import ABC

from infernux.components.builtin_component import BuiltinComponent


class PhysicsBody(BuiltinComponent, ABC):
    """A physical component simulated by its Scene's physics backend."""

    _cpp_type_name = "PhysicsBody"
    _component_user_addable_ = False
