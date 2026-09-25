"""Shared source contract for Infernux GPU kernel declarations.

The Editor compiler, native Player AOT cook, and platform exporters all need
to identify a kernel the same way.  Keep this module deliberately independent
of the runtime/compiler so source-only build checks can use it too.
"""

from __future__ import annotations

import ast
from pathlib import Path


def attribute_name(node: ast.expr) -> str:
    """Return a dotted decorator name, or an empty string for expressions."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    return any(
        attribute_name(item.func if isinstance(item, ast.Call) else item) == name
        for item in node.decorator_list
    )


def source_module_name(source_path: str | Path) -> str:
    """Derive the runtime module identity from an Assets source path.

    Build exporters receive source paths but not always a project context.  A
    project script under ``Assets/Scripts/Jelly.py`` therefore has the same
    stable identity as the normal module loader: ``Scripts.Jelly``.
    """

    path = Path(source_path)
    parts = list(path.with_suffix("").parts)
    lowered = [part.casefold() for part in parts]
    if "assets" in lowered:
        parts = parts[lowered.index("assets") + 1 :]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(part for part in parts if part not in {".", ""}) or path.stem


def qualified_kernel_name(
    module_name: str,
    scope: tuple[str, ...],
    function_name: str,
) -> str:
    return ".".join((module_name, *scope, function_name))


def implicit_receiver_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    in_class: bool,
) -> str | None:
    """Return a conventionally implicit receiver for a class kernel.

    ``inx.compute.kernel`` creates a non-descriptor ``Kernel`` object, so a
    class nesting alone does not inject an argument.  Only the conventional
    ``self``/``cls`` names claim a receiver and violate the source-less GPU
    ABI; other first parameters remain explicit kernel arguments.
    """

    if not in_class or has_decorator(node, "staticmethod"):
        return None
    first = node.args.args[0].arg if node.args.args else "<missing>"
    # A Kernel object is intentionally not a descriptor.  A class function
    # whose first argument is the actual domain therefore does not receive an
    # implicit Python receiver.  Reserve the diagnostic for conventional
    # receiver names, which is the source form that can silently inject self.
    return first if first in {"self", "cls"} else None


def kernel_diagnostic(
    source_path: str | Path,
    qualified: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    target: str,
    reason: str,
    advice: str,
) -> str:
    return (
        f"GPU kernel '{qualified}' at {source_path}:{node.lineno}:"
        f"{node.col_offset + 1} is invalid for target '{target}': {reason}. "
        f"Rewrite: {advice}"
    )


__all__ = [
    "attribute_name",
    "has_decorator",
    "implicit_receiver_name",
    "kernel_diagnostic",
    "qualified_kernel_name",
    "source_module_name",
]
