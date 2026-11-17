"""Authored GPU source records for accepted scripts and cooked Player modules."""

from __future__ import annotations

import ast
from functools import lru_cache

from .kernel_contract import attribute_name as _attribute_name, index_declaration


_RECORDS_PREFIX = "\n# <infernux-compute-source-metadata>\n__infernux_compute_sources__ = "


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
def function_source_spans(source: str, *, include_work_items: bool = False):
    """Return qualified name, code start, source range, and whether to retain the declaration."""
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
            code_line = first = node.lineno
            if node.decorator_list:
                code_line = first = min(decorator.lineno for decorator in node.decorator_list)
                # A parenthesized decorator may start before its expression.
                while not lines[first - 1].lstrip().startswith('@'):
                    first -= 1
            work_item = include_work_items and any(
                index_declaration(statement) is not None for statement in node.body
            )
            spans.append((".".join((*self.scope, node.name)), code_line, first, node.end_lineno, gpu or work_item))
            self.scope.extend((node.name, "<locals>"))
            self.generic_visit(node)
            del self.scope[-2:]

        visit_AsyncFunctionDef = visit_FunctionDef

    Collector().visit(tree)
    return tuple(spans)


def compute_source_records(source: str, *, include_work_items: bool = False) -> dict[tuple[str, int], tuple[str, int]]:
    """Index GPU declarations by qualified name and Python's code start line."""
    source = source.partition(_RECORDS_PREFIX)[0]
    if "kernel" not in source and "function" not in source and not include_work_items:
        return {}
    lines = source.splitlines(keepends=True)
    return {(name, code_line): (''.join(lines[first - 1:last]), first)
            for name, code_line, first, last, retain in function_source_spans(
                source, include_work_items=include_work_items) if retain}


def embed_compute_sources(source: str, *, include_work_items: bool = False) -> str:
    """Embed compute definitions consumed by the resident compiler in a source-less Player."""
    base, marker, _ = source.partition(_RECORDS_PREFIX)
    base = base.rstrip() + "\n"
    records = compute_source_records(base, include_work_items=include_work_items)
    if not records:
        return base if marker else source
    ordered = {name: records[name] for name in sorted(records)}
    return f"{base}{_RECORDS_PREFIX}{ordered!r}\n"


__all__ = ["embed_compute_sources"]
