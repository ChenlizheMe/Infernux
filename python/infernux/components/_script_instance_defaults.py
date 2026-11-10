"""Prepare literal defaults for state fields added to a live Python class."""

import ast
import gc
import inspect
from types import FunctionType


_MISSING = object()


def _literal_fields(definition, initializer):
    method = next(
        (item for item in definition.body
         if isinstance(item, ast.FunctionDef) and item.name == initializer),
        None,
    )
    if method is None or method.decorator_list:
        return
    parameters = method.args.posonlyargs + method.args.args
    if not parameters:
        return
    receiver = parameters[0].arg
    references = {}
    for item in ast.walk(method):
        if (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
                and item.value.id == receiver):
            references[item.attr] = references.get(item.attr, 0) + 1

    for statement in method.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        elif isinstance(statement, (ast.Expr, ast.Assign, ast.Pass)):
            continue
        else:
            # Code after a branch or early exit does not initialize every object.
            break
        if (not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name)
                or target.value.id != receiver or references.get(target.attr) != 1 or value is None):
            continue
        if any(isinstance(node, ast.Call) for node in ast.walk(value)):
            continue
        try:
            default = ast.literal_eval(value)
        except ValueError:
            continue
        yield target.attr, default


def _previous_initializer_names(target, initializer):
    names = set()
    for base in target.__mro__:
        method = vars(base).get(initializer)
        if type(method) is FunctionType:
            names.update(method.__code__.co_names)
    return names


def _live_instance_dicts(target, initializer):
    classes = [target]
    visited = set()
    implementation = inspect.getattr_static(target, initializer, None)
    while classes:
        cls = classes.pop()
        identity = id(cls)
        if identity in visited:
            continue
        visited.add(identity)
        classes.extend(type.__subclasses__(cls))
        if inspect.getattr_static(cls, initializer, None) is not implementation:
            continue
        for instance in gc.get_referrers(cls):
            if type(instance) is not cls:
                continue
            if initializer == "awake" and not instance._awake_called:
                continue
            try:
                fields = object.__getattribute__(instance, "__dict__")
            except AttributeError:
                # A slotted object has no storage for a new plain attribute.
                continue
            if type(fields) is dict:
                yield fields


def plan_instance_defaults(pairs, sources, component_base):
    """Return (instance dictionary, name, literal) for newly declared fields."""
    declarations = {}
    operations = []
    for target, candidate in pairs:
        source = sources.get(candidate.__module__)
        if source is None:
            continue
        initializers = ("__init__", "awake") if issubclass(target, component_base) else ("__init__",)
        new_names = {}
        for initializer in initializers:
            method = vars(candidate).get(initializer)
            if type(method) is FunctionType:
                added = set(method.__code__.co_names) - _previous_initializer_names(target, initializer)
                if added:
                    new_names[initializer] = added
        if not new_names:
            continue
        module = candidate.__module__
        if module not in declarations:
            declarations[module] = {
                node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
            }
        definition = declarations[module].get(candidate.__name__)
        if definition is None:
            continue
        for initializer, added in new_names.items():
            defaults = tuple(
                (name, value) for name, value in _literal_fields(definition, initializer)
                if name in added and inspect.getattr_static(candidate, name, _MISSING) is _MISSING
            )
            if defaults:
                for fields in _live_instance_dicts(target, initializer):
                    operations.extend(
                        (fields, name, value) for name, value in defaults if name not in fields
                    )
    return tuple(operations)
