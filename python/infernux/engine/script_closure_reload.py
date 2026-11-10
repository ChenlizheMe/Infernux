"""Match retained closures by their lexical definition and captured layout."""

from __future__ import annotations

import gc
import inspect
from types import CodeType, FunctionType
from weakref import WeakKeyDictionary


_module_codes = WeakKeyDictionary()
_SUSPENDED = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


class ClosureBody:
    __slots__ = ("function", "__weakref__")

    def __init__(self, function):
        self.function = function

    @property
    def __wrapped__(self):
        return self.function


def remember_closure_code(module, code=None):
    if code is None:
        if module in _module_codes:
            return
        # A helper may already have been imported by CPython. Its live
        # declarations still identify the earlier lexical code objects.
        namespace = vars(module)
        declarations = list(namespace.values())
        for value in tuple(declarations):
            if isinstance(value, type) and value.__module__ == module.__name__:
                declarations.extend(vars(value).values())
        roots = []
        for value in declarations:
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if type(value) is FunctionType and value.__globals__ is namespace:
                roots.append(value.__code__)
    else:
        roots = [code]

    definitions = {}
    visited = set()
    while roots:
        body = roots.pop()
        if id(body) in visited:
            continue
        visited.add(id(body))
        roots.extend(child for child in body.co_consts if type(child) is CodeType)
        if ((body.co_name == "<lambda>" or "<locals>" in body.co_qualname)
                and not body.co_flags & _SUSPENDED):
            definitions.setdefault(body.co_qualname, []).append(body)
    _module_codes[module] = {name: matches[0] for name, matches in definitions.items() if len(matches) == 1}


def _closure_layout(code):
    flags = code.co_flags & _VARIADIC
    argument_count = (code.co_argcount + code.co_kwonlyargcount
                      + bool(flags & inspect.CO_VARARGS) + bool(flags & inspect.CO_VARKEYWORDS))
    return (code.co_freevars, code.co_posonlyargcount, code.co_argcount,
            code.co_kwonlyargcount, flags, code.co_varnames[:argument_count])


def closure_updates(module_pairs):
    """Yield existing closures and replacement code without executing a factory."""
    replacements = {}
    for previous, current in module_pairs:
        old_namespace, new_namespace = vars(previous), vars(current)
        current_code = _module_codes.get(current, {})
        for name, old_code in _module_codes.get(previous, {}).items():
            new_code = current_code.get(name)
            if new_code is not None and _closure_layout(old_code) == _closure_layout(new_code):
                replacements[id(old_namespace), id(old_code)] = new_code, new_namespace
    if not replacements:
        return

    for retained in gc.get_objects():
        if type(retained) is ClosureBody:
            function = retained.function
        elif type(retained) is FunctionType:
            function = retained
        else:
            continue
        replacement = replacements.get((id(function.__globals__), id(function.__code__)))
        if (replacement is not None and "__wrapped__" not in function.__dict__
                and (type(retained) is ClosureBody or not function.__dict__.get("__infernux_closure_body__"))):
            yield retained, function, replacement
