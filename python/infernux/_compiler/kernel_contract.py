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


_INDEX_NAMES = frozenset({"index", "inx.compute.index", "infernux.compute.index", "compute.index"})


def index_declaration(statement: ast.stmt, names=_INDEX_NAMES) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target, call = statement.targets[0], statement.value
    if not isinstance(target, ast.Name) or not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        return None
    if attribute_name(call.func) not in names:
        return None
    if not isinstance(call.args[0], ast.Name):
        raise TypeError("inx.compute.index requires a buffer parameter name")
    return target.id, call.args[0].id


def execution_domain(definition: ast.FunctionDef, names=_INDEX_NAMES) -> tuple[int, str, str, int]:
    """Return declaration position, work index, domain name and parameter slot."""
    declarations = [(position, found) for position, statement in enumerate(definition.body)
                    if (found := index_declaration(statement, names)) is not None]
    if len(declarations) != 1:
        raise TypeError("GPU kernel must declare exactly one execution domain with inx.compute.index(buffer)")
    position, (index_name, domain_name) = declarations[0]
    parameters = tuple(argument.arg for argument in (*definition.args.posonlyargs, *definition.args.args))
    if domain_name not in parameters:
        raise TypeError("inx.compute.index must refer to an inx.buffer parameter")
    return position, index_name, domain_name, parameters.index(domain_name)


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

    Only conventional ``self``/``cls`` names request descriptor binding;
    other first parameters remain explicit kernel arguments.
    """

    if not in_class or has_decorator(node, "staticmethod"):
        return None
    positional = (*node.args.posonlyargs, *node.args.args)
    first = positional[0].arg if positional else "<missing>"
    return first if first in {"self", "cls"} else None


def implicit_receiver_attribute(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ast.Attribute] | None:
    """Return the first access through an unbindable ``self``/``cls`` name.

    A class kernel may use an explicit buffer/scalar parameter as its first
    argument, but it must not smuggle an engine object through a free
    ``self.foo``/``cls.foo`` access.  Keeping this check in the shared source
    contract makes Editor compilation and source-only platform cooks report
    the same failure instead of diverging during lowering.
    """

    for item in sorted(ast.walk(node), key=lambda item: (
        getattr(item, "lineno", 0), getattr(item, "col_offset", 0)
    )):
        if not isinstance(item, ast.Attribute) or not isinstance(item.value, ast.Name):
            continue
        if item.value.id in {"self", "cls"}:
            return item.value.id, item
    return None


def receiver_field_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    """Return stable, source-order fields read through ``self``/``cls``.

    Receiver lowering is deliberately narrow: the receiver itself never enters
    the GPU ABI.  Only fields that are explicitly read through the conventional
    receiver name become ordinary buffer/scalar parameters.  The caller still
    validates their runtime values before compilation.
    """

    positional = (*node.args.posonlyargs, *node.args.args)
    receiver = positional[0].arg if positional and positional[0].arg in {"self", "cls"} else None
    if receiver is None:
        return ()
    names: list[str] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Attribute) or not isinstance(item.value, ast.Name):
            continue
        if item.value.id == receiver and item.attr not in names:
            names.append(item.attr)
    return tuple(names)


def receiver_issue(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Locate receiver uses that cannot be expressed as value parameters."""
    receiver = implicit_receiver_name(node, in_class=True)
    if receiver is None:
        return None
    parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id == receiver:
            parent = parents.get(item)
            if not isinstance(parent, ast.Attribute) or parent.value is not item:
                return item, "the receiver object cannot be passed, assigned or captured by GPU code"
            if not isinstance(parent.ctx, ast.Load):
                return parent, "GPU kernels cannot assign receiver fields; write an inx.buffer element instead"
            grandparent = parents.get(parent)
            if isinstance(grandparent, ast.Call) and grandparent.func is parent:
                return parent, "receiver method calls are not GPU helpers; use @inx.compute.function"
    return None


def kernel_diagnostic(
    source_path: str | Path,
    qualified: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    target: str,
    reason: str,
    advice: str,
    line: int | None = None,
    column: int | None = None,
) -> str:
    location_line = int(line if line is not None else node.lineno)
    location_column = int(column if column is not None else node.col_offset + 1)
    return (
        f"GPU kernel '{qualified}' at {source_path}:{location_line}:"
        f"{location_column} is invalid for target '{target}': {reason}. "
        f"Rewrite: {advice}"
    )


__all__ = [
    "attribute_name",
    "has_decorator",
    "implicit_receiver_attribute",
    "implicit_receiver_name",
    "receiver_field_names",
    "receiver_issue",
    "kernel_diagnostic",
    "qualified_kernel_name",
    "source_module_name",
]
