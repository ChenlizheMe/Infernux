"""Public base of built-in components that own a physical backend."""

from abc import ABC, abstractmethod

from infernux.components.builtin_component import BuiltinComponent


class PhysicsBody(BuiltinComponent, ABC):
    """A component with its own CPU or GPU physics implementation."""

    _cpp_type_name = "PhysicsBody"
    _component_user_addable_ = False

    @property
    @abstractmethod
    def gpu(self) -> bool:
        """Whether this component selects GPU physics. The default is CPU."""

    @gpu.setter
    @abstractmethod
    def gpu(self, value: bool) -> None:
        """Select the backend. A live Rigidbody applies the selection at a fixed-step boundary."""
