"""Shared single-work-item source normalization for CPU and GPU compilation.

Only source and request-local values live here; no device/executor is created.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
import inspect
import os
import symtable
import textwrap

import numpy as np

from .kernel_contract import (execution_domain, implicit_receiver_attribute,
    implicit_receiver_name, receiver_field_names, receiver_issue, ARRAY_DTYPES)
from .compile_inputs import snapshot_compilation_values


def _kernel_qualified_name(function) -> str:
    """Return the stable source identity shown in kernel diagnostics."""
    module = str(getattr(function, "__module__", "") or "<module>")
    qualified = str(
        getattr(function, "__qualname__", None)
        or getattr(function, "__name__", "<kernel>")
    )
    return f"{module}.{qualified}"


def _kernel_target() -> str:
    """Describe the active compiler target without importing the engine eagerly."""
    explicit = os.environ.get("INFERNUX_COMPUTE_TARGET", "").strip()
    if explicit:
        return explicit
    try:
        from infernux.application import Application

        return "Player/Desktop" if Application.is_player() else "Editor/Desktop"
    except Exception:
        return "Editor/Desktop"


def _kernel_diagnostic(
    function,
    reason: str,
    advice: str,
    *,
    target: str | None = None,
    line: int | None = None,
    column: int = 1,
) -> str:
    code = getattr(function, "__code__", None)
    path = str(getattr(code, "co_filename", "<unknown>"))
    line = int(line if line is not None else getattr(code, "co_firstlineno", 0) or 0)
    qualified = _kernel_qualified_name(function)
    compiler_target = target or _kernel_target()
    return (
        f"compute kernel '{qualified}' at {path}:{line}:{column} is invalid for "
        f"target '{compiler_target}': {reason}. Rewrite: {advice}"
    )


def _is_class_qualified(function) -> bool:
    qualified = str(getattr(function, "__qualname__", ""))
    parts = qualified.split(".")
    # ``test.<locals>.Kernel.step`` is still a class method; only a plain
    # nested function has ``<locals>`` as the component immediately before
    # the function name.
    return len(parts) >= 2 and parts[-2] != "<locals>"


def _validate_kernel_method(
    function,
    definition: ast.FunctionDef,
    *,
    target: str | None = None,
    receiver_fields: tuple[str, ...] | None = None,
) -> None:
    """Validate class identity and keep Python objects out of the work-item ABI."""
    if not _is_class_qualified(function):
        return
    first = implicit_receiver_name(definition, in_class=True)
    if first is not None and receiver_fields is None:
        raise TypeError(_kernel_diagnostic(
            function,
            f"class-contained kernels cannot use an implicit instance receiver '{first}'",
            "put @staticmethod above @inx.compute.kernel (decorator order: @staticmethod, then @inx.compute.kernel) or move the kernel to module scope",
            target=target,
            line=definition.lineno,
            column=definition.col_offset + 1,
        ))
    issue = receiver_issue(definition)
    if issue is not None:
        node, reason = issue
        raise TypeError(_kernel_diagnostic(
            function, reason, "read numeric fields or mutate inx.buffer elements only",
            target=target, line=node.lineno, column=node.col_offset + 1,
        ))
    access = implicit_receiver_attribute(definition)
    if access is not None and access[1].attr not in (receiver_fields or ()):
        receiver, attribute = access
        raise TypeError(_kernel_diagnostic(
            function,
            f"class-contained kernels cannot access unbound receiver field '{receiver}.{attribute.attr}'",
            "pass the required scalar or inx.buffer explicitly, or put @staticmethod above @inx.compute.kernel",
            target=target,
            line=attribute.lineno,
            column=attribute.col_offset + 1,
        ))


class _ReceiverLowering(ast.NodeTransformer):
    """Replace the small, explicit instance receiver ABI with parameters."""

    def __init__(self, receiver: str, fields: tuple[str, ...]):
        self._receiver = receiver
        self._fields = frozenset(fields)

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self._receiver
            and node.attr in self._fields
        ):
            return ast.copy_location(
                ast.Name(id=f"_infernux_receiver_{node.attr}", ctx=node.ctx),
                node,
            )
        return node


def _function_definition(value, *, kind: str) -> ast.FunctionDef:
    if value.__closure__:
        raise TypeError(f"{kind} cannot capture Python closure values")
    source = _function_source(value)
    tree = ast.parse(source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if not isinstance(definition, ast.FunctionDef):
        raise TypeError(f"{kind} must be a synchronous Python function")
    definition.decorator_list = []
    return definition


def _function_source(value) -> str:
    records = value.__globals__.get("__infernux_compute_sources__", {})
    embedded = records.get(value.__qualname__) if isinstance(records, dict) else None
    if embedded is not None:
        if not isinstance(embedded, str) or not embedded.strip():
            raise TypeError(f"compute source metadata is invalid for {value.__qualname__}")
        return textwrap.dedent(embedded)
    return textwrap.dedent(inspect.getsource(value))


def receiver_fields(function) -> tuple[str, ...]:
    """Analyze a declaration once; no receiver or live values are cached."""

    source = _function_source(function)
    tree = ast.parse(source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if not isinstance(definition, ast.FunctionDef):
        raise TypeError("compute kernel must be a synchronous Python function")
    positional = (*definition.args.posonlyargs, *definition.args.args)
    if not positional or positional[0].arg not in {"self", "cls"}:
        raise TypeError("instance kernel receiver is not a conventional self/cls parameter")
    names = receiver_field_names(definition)
    _validate_kernel_method(function, definition, receiver_fields=names)
    return names


def receiver_bindings(function, receiver, fields: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    """Snapshot numeric fields/buffer owners once at the launch boundary."""
    from infernux.compute import Buffer

    bindings: list[tuple[str, object]] = []
    for name in fields:
        try:
            value = getattr(receiver, name)
        except AttributeError as exception:
            raise TypeError(_kernel_diagnostic(
                function, f"receiver field '{name}' is not initialized",
                "initialize the scalar or inx.buffer before prepare/launch",
            )) from exception
        if not isinstance(value, (Buffer, bool, int, float, np.integer, np.floating)):
            raise TypeError(_kernel_diagnostic(
                function, f"receiver field '{name}' has unsupported type {type(value).__name__}",
                "pass a buffer or numeric scalar; Python objects cannot enter the work-item ABI",
            ))
        bindings.append((name, value))
    return tuple(bindings)


def _helper_definitions(kernel_definition: ast.FunctionDef, globals_map: dict) -> list[ast.FunctionDef]:
    """Collect explicitly declared helpers and reject recursive call graphs."""
    from infernux.compute import Function

    ordered: list[ast.FunctionDef] = []
    states: dict[str, str] = {}

    def visit(definition: ast.FunctionDef) -> None:
        for call in (node for node in ast.walk(definition) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Name):
                continue
            name = call.func.id
            helper = globals_map.get(name)
            if not isinstance(helper, Function):
                continue
            state = states.get(name)
            if state == "visiting":
                raise TypeError(f"compute helper recursion is not supported: {name}")
            if state == "done":
                continue
            states[name] = "visiting"
            helper_definition = _function_definition(helper.function, kind="compute helper")
            helper_definition.name = name
            visit(helper_definition)
            helper_definition.decorator_list = [ast.Name(id="_infernux_gpu_function", ctx=ast.Load())]
            ordered.append(helper_definition)
            states[name] = "done"

    try:
        visit(kernel_definition)
        return ordered
    finally:
        # The recursive closure otherwise retains the unfiltered project
        # globals until cyclic GC, including after a compilation failure.
        visit = None


@dataclass(slots=True)
class WorkItem:
    definition: ast.FunctionDef
    helpers: list[ast.FunctionDef]
    globals_map: dict
    compile_values: dict
    index_name: str
    domain_name: str
    domain_parameter: int

    def loop_definition(self, definition=None, *, bounds: tuple[str, str] | None = None):
        """Wrap the one author body; backends never recover it independently."""
        definition = copy.deepcopy(self.definition) if definition is None else definition
        arguments = [ast.Subscript(
            value=ast.Attribute(value=ast.Name(id=self.domain_name, ctx=ast.Load()),
                                attr="shape", ctx=ast.Load()),
            slice=ast.Constant(value=0), ctx=ast.Load())]
        if bounds is not None:
            arguments = [ast.Name(id=name, ctx=ast.Load()) for name in bounds]
        definition.body = [ast.For(
            target=ast.Name(id=self.index_name, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=arguments, keywords=[]),
            body=definition.body, orelse=[])]
        return ast.fix_missing_locations(definition)

    def analyze(self, *, argument_types=None):
        from infernux.jit_hir import TypeRef, ValueType, analyze_source

        types = {name: TypeRef(ValueType.ARRAY) for name, value in self.globals_map.items()
                 if isinstance(value, (np.ndarray, list, tuple, dict))}
        types.update(argument_types or {})
        return analyze_source(ast.unparse(self.loop_definition()), call_bindings=self.globals_map,
                              argument_types=types if argument_types is not None else None)

    def reduction_variants(self, plan, parameter_kinds):
        """Finite backend-independent programs from one ordered region."""
        plan.require()
        region = plan.ordered_reduction
        if region is None or not region.eligible:
            raise TypeError("work item has no ordered inner cut: " + (
                "; ".join(region.diagnostics) if region is not None else "no local reduction region"))
        result = {"whole": LoweredWorkItem(self, plan, tuple(parameter_kinds))}
        if region.count > 1:
            for kind in ("initial", "resume", "final"):
                if kind != "resume" or region.count > 2:
                    result[kind] = self.lower_reduction(plan, parameter_kinds, kind)
        return result

    def lower_reduction(self, plan, parameter_kinds, kind):
        """Derive a finite inner variant from this body's existing region proof."""
        from infernux.jit_hir import InnerRangeContract, ResourceExtent

        plan.require()
        region = plan.ordered_reduction
        if region is None or not region.eligible:
            raise TypeError("work item has no ordered inner cut: " + (
                "; ".join(region.diagnostics) if region is not None else "no local reduction region"))
        if kind not in {"initial", "resume", "final"} or region.count < (3 if kind == "resume" else 2):
            raise ValueError("reduction variant has no nonempty legal inner range")
        body = ast.Module(body=self.definition.body, type_ignores=[])
        identity = ast.dump(ast.parse(ast.unparse(body)), include_attributes=False)
        if identity != region.body_identity:
            raise ValueError("reduction proof does not belong to this normalized body")
        names = tuple(arg.arg for arg in (*self.definition.args.posonlyargs, *self.definition.args.args))
        if len(parameter_kinds) != len(names) or any(kind not in {*ARRAY_DTYPES, "float32", "int32"} for kind in parameter_kinds):
            raise TypeError("ordered reduction requires typed POD arrays and float32/int32 scalars")
        hidden = ("_infernux_carry_in", "_infernux_carry_out", "_infernux_inner_begin", "_infernux_inner_end")
        used = {node.id if isinstance(node, ast.Name) else node.arg for node in ast.walk(self.definition)
                if isinstance(node, (ast.Name, ast.arg))} | set(self.globals_map)
        if any(name in used for name in hidden):
            raise TypeError("work item uses a reserved ordered-reduction parameter")
        definition = copy.deepcopy(self.definition)
        prefix = definition.body[:region.loop_statement]
        loop = definition.body[region.loop_statement]
        epilogue = definition.body[region.loop_statement + 1:]
        if kind != "initial":
            prefix[region.initializer_statement].value = ast.Subscript(
                value=ast.Name(id=hidden[0], ctx=ast.Load()),
                slice=ast.Name(id=self.index_name, ctx=ast.Load()), ctx=ast.Load())
        loop.iter.args = [ast.Name(id=name, ctx=ast.Load()) for name in hidden[2:]]
        if kind != "final":
            epilogue = [ast.Assign(targets=[ast.Subscript(value=ast.Name(id=hidden[1], ctx=ast.Load()),
                slice=ast.Name(id=self.index_name, ctx=ast.Load()), ctx=ast.Store())],
                value=ast.Name(id=region.accumulator, ctx=ast.Load()))]
        definition.body = [*prefix, loop, *epilogue]
        definition.args.posonlyargs = []
        definition.args.args = [*(ast.arg(arg=name) for name in names), *(ast.arg(arg=name) for name in hidden)]
        definition.name += "_" + kind
        definition = ast.fix_missing_locations(definition)
        all_names = (*names, *hidden)
        arrays = {name for name, layout in zip(all_names, (*parameter_kinds, "array_f32", "array_f32", "int32", "int32"))
                  if layout in ARRAY_DTYPES}
        reads = tuple(sorted({node.value.id for node in ast.walk(definition) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id in arrays and isinstance(node.ctx, ast.Load)}))
        stores = tuple(node.value.id for node in ast.walk(definition) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id in arrays and isinstance(node.ctx, ast.Store))
        writes = tuple(sorted(set(stores)))
        expected = plan.outputs if kind == "final" else (hidden[1],)
        if writes != expected or set(reads) & set(writes):
            raise ValueError("derived reduction variant changed its proved read/write roles")
        row_upper = ("sub", ("n",), ("c", 1))
        extents = tuple(replace(extent, exact_domain=True) if names[extent.parameter] in plan.outputs else extent
                        for extent in plan.resource_extents)
        extents += tuple(ResourceExtent(len(names) + index, True, ((("c", 0), row_upper),), (row_upper,))
                         for index in range(2))
        control = InnerRangeContract(kind, region.count, len(names) + 2, len(names) + 3)
        derived = replace(plan, inputs=reads, outputs=writes, fixed_parameters=all_names,
                          fully_written_outputs=writes, resource_extents=extents, inner_range=control,
                          operation_cost=region.operation_cost(kind),
                          read_elements_per_iteration=(region.prefix_reads + region.term_reads * region.count
                              + (region.epilogue_reads if kind == "final" else 0) + (kind != "initial")),
                          write_elements_per_iteration=len(stores))
        item = WorkItem(definition, copy.deepcopy(self.helpers), dict(self.globals_map), dict(self.compile_values),
                        self.index_name, self.domain_name, self.domain_parameter)
        return LoweredWorkItem(item, derived, (*parameter_kinds, "array_f32", "array_f32", "int32", "int32"))


@dataclass(frozen=True)
class LoweredWorkItem:
    item: WorkItem
    continuation: object
    parameter_kinds: tuple[str, ...]


def normalize_work_item(function, parameter_count: int, *, receiver_fields=None, target=None) -> WorkItem:
    """Normalize source/receiver/helpers once, independently of storage device."""
    signature = inspect.signature(function)
    if any(parameter.kind not in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
           or parameter.default is not parameter.empty for parameter in signature.parameters.values()):
        raise TypeError("compute kernels currently require positional parameters without defaults")
    bound_receiver = receiver_fields is not None
    field_names = tuple(name for name, _ in (receiver_fields or ()))
    positional = tuple(signature.parameters.values())
    receiver_name = positional[0].name if positional and positional[0].name in {"self", "cls"} else None
    expected = len(positional) - (1 if receiver_name and bound_receiver else 0) + len(field_names)
    if parameter_count != expected:
        raise TypeError(f"compute kernel requires {expected} parameters, got {parameter_count}")
    if function.__closure__:
        captured = ", ".join(str(name) for name in getattr(function.__code__, "co_freevars", ()))
        raise TypeError(_kernel_diagnostic(
            function, f"arbitrary Python closure capture is not part of the work-item ABI ({captured or 'unknown value'})",
            "pass buffers or numeric scalars as explicit parameters; do not close over Python locals", target=target))
    definition = next((node for node in ast.parse(_function_source(function)).body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not isinstance(definition, ast.FunctionDef):
        raise TypeError("compute kernel must be a synchronous Python function")
    _validate_kernel_method(function, definition, target=target,
                            receiver_fields=field_names if bound_receiver else None)
    if receiver_name and bound_receiver:
        declared = receiver_field_names(definition)
        if declared != field_names:
            raise TypeError(_kernel_diagnostic(
                function, "receiver field binding does not match the source receiver fields "
                f"({', '.join(declared) or 'none'})",
                "bind every self/cls field explicitly or remove the receiver access", target=target,
                line=definition.lineno, column=definition.col_offset + 1))
        if definition.args.posonlyargs and definition.args.posonlyargs[0].arg == receiver_name:
            definition.args.posonlyargs = definition.args.posonlyargs[1:]
        elif definition.args.args and definition.args.args[0].arg == receiver_name:
            definition.args.args = definition.args.args[1:]
        remaining = [*definition.args.posonlyargs, *definition.args.args]
        reserved = {node.id if isinstance(node, ast.Name) else node.arg
                    for node in ast.walk(definition) if isinstance(node, (ast.Name, ast.arg))}
        if any(f"_infernux_receiver_{name}" in reserved for name in field_names):
            raise TypeError("compute kernel source uses a reserved _infernux_receiver_ name")
        definition.args.posonlyargs = []
        definition.args.args = [*(ast.arg(arg=f"_infernux_receiver_{name}") for name in field_names), *remaining]
        definition = _ReceiverLowering(receiver_name, field_names).visit(definition)
        ast.fix_missing_locations(definition)
    definition.decorator_list = []
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == function.__name__ for node in ast.walk(definition)):
        raise TypeError(f"compute kernel recursion is not supported: {function.__name__}")
    if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await)) for node in ast.walk(definition)):
        raise TypeError("compute kernels write output buffer values in place and cannot return or yield")
    position, index_name, domain_name, domain_parameter = execution_domain(definition)
    body = list(definition.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
        position -= 1
    body.pop(position)
    definition.body = body
    globals_map = dict(function.__globals__)
    helpers = _helper_definitions(definition, globals_map)
    compile_values = snapshot_compilation_values([*helpers, definition], globals_map)
    # Retained source plans own only actual dependencies, never an old snapshot
    # of unrelated worlds, models or a previous generation of declarations.
    scopes = [symtable.symtable(ast.unparse(ast.Module(body=[*helpers, definition], type_ignores=[])),
                                "<infernux-work-item-globals>", "exec")]
    required = {"__name__"}
    while scopes:
        scope = scopes.pop()
        required.update(symbol.get_name() for symbol in scope.get_symbols()
                        if symbol.is_global() and symbol.is_referenced())
        scopes.extend(scope.get_children())
    globals_map = {name: value for name, value in globals_map.items() if name in required}
    return WorkItem(definition, helpers, globals_map, compile_values, index_name, domain_name, domain_parameter)
