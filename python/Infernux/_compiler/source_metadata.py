"""Build-time source metadata for source-less GPU kernel modules."""

from __future__ import annotations

import ast
import textwrap


_MARKER = "# <infernux-compute-source-metadata>"
_GLOBAL_NAME = "__infernux_compute_sources__"


def _attribute_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _decorator_names(tree: ast.Module) -> set[str]:
    names = {
        # Engine-owned compute modules commonly define the decorator in the
        # same module and use the concise ``@kernel``/``@function`` spelling.
        # Preserve those sources too; Player modules are intentionally
        # source-less, so inspect.getsource cannot recover them later.
        "kernel",
        "function",
        "compute.kernel",
        "compute.function",
        "inx.compute.kernel",
        "inx.compute.function",
        "Infernux.compute.kernel",
        "Infernux.compute.function",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"Infernux", "infernux"}:
                    root = alias.asname or alias.name
                    names.update({f"{root}.compute.kernel", f"{root}.compute.function"})
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"Infernux", "infernux"}:
                for alias in node.names:
                    if alias.name == "compute":
                        root = alias.asname or alias.name
                        names.update({f"{root}.kernel", f"{root}.function"})
            elif node.module in {"Infernux.compute", "infernux.compute"}:
                for alias in node.names:
                    if alias.name in {"kernel", "function"}:
                        names.add(alias.asname or alias.name)
    return names


def embed_compute_sources(source: str) -> str:
    """Embed only decorated GPU functions for a source-less Player module."""

    had_marker = _MARKER in source
    base = source.split(_MARKER, 1)[0].rstrip() + "\n"
    tree = ast.parse(base)
    decorator_names = _decorator_names(tree)
    records: dict[str, str] = {}

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if any(
                _attribute_name(item.func if isinstance(item, ast.Call) else item)
                in decorator_names
                for item in node.decorator_list
            ):
                segment = ast.get_source_segment(base, node)
                if segment is None:
                    raise ValueError(f"Cannot preserve GPU source for {node.name}")
                records[".".join((*self.classes, node.name))] = textwrap.dedent(segment)

        visit_AsyncFunctionDef = visit_FunctionDef

    Collector().visit(tree)
    if not records:
        return base if had_marker else source
    ordered = {name: records[name] for name in sorted(records)}
    return f"{base}\n{_MARKER}\n{_GLOBAL_NAME} = {ordered!r}\n"


__all__ = ["embed_compute_sources"]
