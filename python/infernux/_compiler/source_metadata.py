"""Build-time source metadata for source-less GPU kernel modules."""

from __future__ import annotations

import ast
from functools import lru_cache

from .kernel_contract import attribute_name as _attribute_name


_MARKER = "# <infernux-compute-source-metadata>"
_GLOBAL_NAME = "__infernux_compute_sources__"


def compute_decorator_names(
    tree: ast.Module, *, kinds: tuple[str, ...] = ("kernel", "function"),
    include_local: bool = False,
) -> set[str]:
    """Use one decorator vocabulary for closure discovery and cooked sources."""
    names = {f"{root}.{kind}" for root in ("compute", "inx.compute", "infernux.compute")
             for kind in kinds}
    if include_local:
        # Engine-owned modules define these decorators in their own namespace.
        names.update(kinds)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "infernux":
                    root = alias.asname or alias.name
                    names.update(f"{root}.compute.{kind}" for kind in kinds)
                elif alias.name == "infernux.compute":
                    root = alias.asname or alias.name
                    names.update(f"{root}.{kind}" for kind in kinds)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "infernux":
                for alias in node.names:
                    if alias.name == "compute":
                        root = alias.asname or alias.name
                        names.update(f"{root}.{kind}" for kind in kinds)
            elif node.module == "infernux.compute":
                for alias in node.names:
                    if alias.name in kinds:
                        names.add(alias.asname or alias.name)
    return names


@lru_cache(maxsize=32)
def function_source_spans(source: str):
    """Shared immutable source index for live declarations and cooked modules."""
    tree = ast.parse(source)
    decorator_names = compute_decorator_names(tree, include_local=True)
    spans = []
    lines = source.splitlines(keepends=True)

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            gpu = any(
                _attribute_name(item.func if isinstance(item, ast.Call) else item)
                in decorator_names
                for item in node.decorator_list
            )
            first = node.lineno
            if node.decorator_list:
                first = min(decorator.lineno for decorator in node.decorator_list)
                # A parenthesized decorator may start before its expression.
                while not lines[first - 1].lstrip().startswith('@'):
                    first -= 1
            spans.append((".".join((*self.scope, node.name)), first, node.end_lineno, gpu))
            self.scope.extend((node.name, "<locals>"))
            self.generic_visit(node)
            del self.scope[-2:]

        visit_AsyncFunctionDef = visit_FunctionDef

    Collector().visit(tree)
    return tuple(spans)


def embed_compute_sources(source: str) -> str:
    """Embed only decorated GPU functions for a source-less Player module."""
    had_marker = _MARKER in source
    base = source.split(_MARKER, 1)[0].rstrip() + "\n"
    lines = base.splitlines(keepends=True)
    records = {name: (''.join(lines[first - 1:last]), first)
               for name, first, last, gpu in function_source_spans(base) if gpu}
    if not records:
        return base if had_marker else source
    ordered = {name: records[name] for name in sorted(records)}
    return f"{base}\n{_MARKER}\n{_GLOBAL_NAME} = {ordered!r}\n"


__all__ = ["embed_compute_sources"]
