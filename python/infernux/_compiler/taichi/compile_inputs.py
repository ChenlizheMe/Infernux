"""Snapshot the compile-time values actually read by a GPU source plan."""
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
            raise TypeError('GPU compile-time arrays cannot contain Python objects')
        copied = value.copy()
        return ['numpy-array', copied.dtype.str, list(copied.shape), copied.tobytes().hex()], copied
    if type(value) in (tuple, list):
        values = [_snapshot(item) for item in value]
        return [type(value).__name__, [item[0] for item in values]], type(value)(item[1] for item in values)
    if type(value) is dict:
        pairs = [(_snapshot(key), _snapshot(item)) for key, item in value.items()]
        return ['dict', [[key[0], item[0]] for key, item in pairs]], {key[1]: item[1] for key, item in pairs}
    raise TypeError(f'GPU compile-time value has unsupported type {type(value).__name__}')


def snapshot_compilation_values(definitions: list[ast.FunctionDef], globals_map: dict) -> dict:
    """Bind globals/attributes to immutable request-local slots before keying and compiling.

    Only loaded nonlocal values participate: unrelated module state and names
    shadowed by arguments/locals do not invalidate the artifact. Both the key
    and the compiler use this same snapshot, including mutable arrays/containers.
    Modules, types and callable intrinsics/helpers retain their source identity;
    their accessed scalar/container attributes are captured individually.
    """
    identities: dict[str, object] = {}
    slots: dict[str, str] = {}

    class Capture(ast.NodeTransformer):
        def __init__(self, definition):
            self.locals = {arg.arg for arg in (*definition.args.posonlyargs, *definition.args.args,
                                               *definition.args.kwonlyargs)}
            self.locals.update(node.id for node in ast.walk(definition)
                               if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))

        def bind(self, node, name, value):
            if isinstance(value, (types.ModuleType, type)) or callable(value):
                return None
            if name not in slots:
                identity, frozen = _snapshot(value)
                slot = f'_infernux_compile_value_{len(slots)}'
                if slot in globals_map or any(slot in capture for capture in reserved):
                    raise TypeError('GPU source uses a reserved _infernux_compile_value_ name')
                identities[name] = identity
                slots[name] = slot
                globals_map[slot] = frozen
            return ast.copy_location(ast.Name(id=slots[name], ctx=ast.Load()), node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id not in self.locals and node.id in globals_map:
                replacement = self.bind(node, node.id, globals_map[node.id])
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
                if isinstance(base, ast.Name) and base.id not in self.locals and base.id in globals_map:
                    value = globals_map[base.id]
                    for part in reversed(parts):
                        value = getattr(value, part)
                    name = '.'.join([base.id, *reversed(parts)])
                    replacement = self.bind(node, name, value)
                    if replacement is not None:
                        return replacement
            return self.generic_visit(node)

    reserved = [{node.id for node in ast.walk(definition) if isinstance(node, ast.Name)}
                for definition in definitions]
    for definition in definitions:
        capture = Capture(definition)
        definition.body = [capture.visit(statement) for statement in definition.body]
    return identities
