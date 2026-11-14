"""Transport-neutral automation host contracts."""

from infernux.engine.path_utils import resolve_project_path

from .commands import CommandFuture, MainThreadCommandQueue

# Runtime command admission does not import Editor operations or their schema
# dependencies. Editor callers resolve the same exports when they use them.
_EDITOR_EXPORTS = {
    "EditorAutomationHost": ".editor",
    "build_editor_operations": ".editor_operations",
    "install_editor_operations": ".editor_operations",
    **dict.fromkeys(("Operation", "OperationError", "OperationJobRegistry",
                     "OperationKind", "OperationRegistry", "OperationSchema",
                     "capability_granted"), ".operations"),
}


def __getattr__(name):
    module = _EDITOR_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(module, __name__), name)
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
