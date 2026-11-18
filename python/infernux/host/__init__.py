"""Transport-neutral automation host contracts."""

from infernux.engine.path_utils import resolve_project_path

from .commands import CommandFuture, MainThreadCommandQueue
from .operations import (
    Operation,
    OperationError,
    OperationJobRegistry,
    OperationKind,
    OperationRegistry,
    OperationSchema,
    capability_granted,
)


_EDITOR_EXPORTS = {
    "EditorAutomationHost": (".editor", "EditorAutomationHost"),
    "build_editor_operations": (".editor_operations", "build_editor_operations"),
    "install_editor_operations": (".editor_operations", "install_editor_operations"),
}


def __getattr__(name):
    if name not in _EDITOR_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module, attribute = _EDITOR_EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "CommandFuture",
    "EditorAutomationHost",
    "MainThreadCommandQueue",
    "Operation",
    "OperationError",
    "OperationJobRegistry",
    "OperationKind",
    "OperationRegistry",
    "OperationSchema",
    "capability_granted",
    "build_editor_operations",
    "install_editor_operations",
    "resolve_project_path",
]
