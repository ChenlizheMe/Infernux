"""Type stubs for infernux.components.decorators."""

from __future__ import annotations

from typing import Callable, Type, Union


def require_component(*component_types: Type) -> Callable:
    """Declare that a component requires other component types.

    Example::

        @require_component(Rigidbody, Collider)
        class PhysicsController(InxComponent): ...
    """
    ...


def disallow_multiple(cls: Type = ...) -> Union[Type, Callable]:
    """Prevent multiple instances of this component on a GameObject.

    Usable with or without parentheses::

        @disallow_multiple
        class MySingleton(InxComponent): ...

        @disallow_multiple()
        class MySingleton(InxComponent): ...
    """
    ...


def execute_in_edit_mode(cls: Type = ...) -> Union[Type, Callable]:
    """Allow a component's ``update()`` to run in edit mode.

    Example::

        @execute_in_edit_mode
        class PreviewComponent(InxComponent): ...
    """
    ...


def add_component_menu(path: str) -> Callable:
    """Specify where this component appears in the Add Component menu.

    Args:
        path: Menu path like ``"Physics/Character Controller"``.
    """
    ...


def icon(icon_path: str) -> Callable:
    """Specify a custom icon for this component in the inspector.

    Args:
        icon_path: Path to the icon image (relative to project assets).
    """
    ...


def help_url(url: str) -> Callable:
    """Specify a help URL for this component.

    Args:
        url: URL to documentation.
    """
    ...


# Unity-style aliases
RequireComponent = require_component
DisallowMultipleComponent = disallow_multiple
ExecuteInEditMode = execute_in_edit_mode
AddComponentMenu = add_component_menu
HelpURL = help_url
Icon = icon
