"""Snapshot the compile-time values actually read by a work-item source plan."""
from __future__ import annotations

import ast
import types

import numpy as np


def _snapshot(value):
    """Return an exact, JSON-compatible identity and the value given to the compiler."""
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value], value
    if type(value) is float:
        return ['float', value.hex()], value
    if isinstance(value, np.generic):
        copied = value.copy()
        return ['numpy-scalar', value.dtype.str, copied.tobytes().hex()], copied
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('compile-time arrays cannot contain Python objects')
        copied = value.copy()
        return ['numpy-array', copied.dtype.str, list(copied.shape), copied.tobytes().hex()], copied
    if type(value) in (tuple, list):
        values = [_snapshot(item) for item in value]
        return [type(value).__name__, [item[0] for item in values]], type(value)(item[1] for item in values)
    if type(value) is dict:
        pairs = [(_snapshot(key), _snapshot(item)) for key, item in value.items()]
        return ['dict', [[key[0], item[0]] for key, item in pairs]], {key[1]: item[1] for key, item in pairs}
    raise TypeError(f'compile-time value has unsupported type {type(value).__name__}')


class _CompilationCapture(ast.NodeTransformer):
    def __init__(self, definitions, globals_map):
        self.globals_map = globals_map
        self.identities = {}
        self.slots = {}
        self.reserved = {node.id if isinstance(node, ast.Name) else node.arg
                         for definition in definitions for node in ast.walk(definition)
                         if isinstance(node, (ast.Name, ast.arg))}

    def capture(self, definition):
        self.locals = {arg.arg for arg in (*definition.args.posonlyargs, *definition.args.args,
                                           *definition.args.kwonlyargs)}
        self.locals.update(node.id for node in ast.walk(definition)
                           if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        definition.body = [self.visit(statement) for statement in definition.body]

    def bind(self, node, name, value):
        if isinstance(value, (types.ModuleType, type)) or callable(value):
            return None
        if name not in self.slots:
            identity, frozen = _snapshot(value)
            slot = f'_infernux_compile_value_{len(self.slots)}'
            if slot in self.globals_map or slot in self.reserved:
                raise TypeError('work-item source uses a reserved _infernux_compile_value_ name')
            self.identities[name] = identity
            self.slots[name] = slot
            self.globals_map[slot] = frozen
        # Only this actual capture owns the synthetic slot. Integer syntax
        # lets both backends prove static bounds without guessing by name.
        frozen = self.globals_map[self.slots[name]]
        if type(frozen) is int:
            return ast.copy_location(ast.Constant(value=frozen), node)
        return ast.copy_location(ast.Name(id=self.slots[name], ctx=ast.Load()), node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.locals and node.id in self.globals_map:
            replacement = self.bind(node, node.id, self.globals_map[node.id])
            if replacement is not None:
                return replacement
        return node

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            parts = []
            base = node
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id not in self.locals and base.id in self.globals_map:
                value = self.globals_map[base.id]
                for part in reversed(parts):
                    value = getattr(value, part)
                name = '.'.join([base.id, *reversed(parts)])
                replacement = self.bind(node, name, value)
                if replacement is not None:
                    return replacement
        return self.generic_visit(node)


def snapshot_compilation_values(definitions: list[ast.FunctionDef], globals_map: dict) -> dict:
    """Freeze only accessed nonlocal values for both artifact identity and compilation."""
    capture = _CompilationCapture(definitions, globals_map)
    for definition in definitions:
        capture.capture(definition)
    return capture.identities
