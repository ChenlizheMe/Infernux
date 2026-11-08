"""Small, conservative Typed HIR front-end for ``auto_parallel``.

This module is intentionally independent from :mod:`infernux.jit` and from
the runtime JIT implementation.  It answers one question only: can a loop be
lowered to an independent range loop without changing Python semantics?

The implementation is deliberately fail-closed.  It does not try to prove
arbitrary Python code safe; unknown calls, indirect indexing, container
mutation, and control-flow which changes loop lifetime are diagnostics rather
than optimization opportunities.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import math
import operator
import sys
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ValueType(_StrEnum):
    UNKNOWN = "unknown"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    SCALAR = "scalar"
    ARRAY = "array"
    VOID = "void"


class BufferAccessKind(_StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class EffectKind(_StrEnum):
    PURE = "pure"
    BUFFER_READ = "buffer_read"
    BUFFER_WRITE = "buffer_write"
    REDUCTION = "reduction"
    PURE_CALL = "pure_call"
    UNKNOWN_CALL = "unknown_call"
    CONTAINER_MUTATION = "container_mutation"
    CONTROL_FLOW = "control_flow"
    ALIAS_RISK = "alias_risk"


class AliasRiskKind(_StrEnum):
    NONE = "none"
    POSSIBLE = "possible"
    HIGH = "high"
    UNKNOWN = "unknown"


class DiagnosticSeverity(_StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticCode(_StrEnum):
    PARSE_ERROR = "parse_error"
    FUNCTION_NOT_FOUND = "function_not_found"
    ASYNC_FUNCTION = "async_function"
    UNSUPPORTED_CONTROL_FLOW = "unsupported_control_flow"
    UNSUPPORTED_NESTED_LOOP = "unsupported_nested_loop"
    UNSUPPORTED_RANGE = "unsupported_range"
    NON_AFFINE_INDEX = "non_affine_index"
    INDIRECT_WRITE = "indirect_write"
    LOOP_CARRIED_READ = "loop_carried_read"
    LOOP_CARRIED_WRITE = "loop_carried_write"
    REDUCTION_FEEDBACK = "reduction_feedback"
    LOOP_CARRIED_SCALAR = "loop_carried_scalar"
    UNKNOWN_CALL = "unknown_call"
    CONTAINER_MUTATION = "container_mutation"
    UNSUPPORTED_STATEMENT = "unsupported_statement"
    UNSUPPORTED_EXPRESSION = "unsupported_expression"
    ALIAS_RISK = "alias_risk"
    INVALID_REDUCTION = "invalid_reduction"
    INVALID_ARGUMENT = "invalid_argument"


@dataclass(frozen=True)
class SourceLocation:
    """A source span suitable for editor diagnostics and source rewriting."""

    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def from_node(cls, node: ast.AST) -> "SourceLocation":
        return cls(
            line=getattr(node, "lineno", 0),
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
        )


@dataclass(frozen=True)
class TypeRef:
    value_type: ValueType
    shape: tuple[str, ...] = ()
    mutable: bool = False

    @property
    def is_scalar(self) -> bool:
        return self.value_type in {
            ValueType.BOOL,
            ValueType.INT,
            ValueType.FLOAT,
            ValueType.SCALAR,
        }


@dataclass(frozen=True)
class HIRArgument:
    name: str
    type_ref: TypeRef
    annotation: str | None = None
    location: SourceLocation | None = None


@dataclass(frozen=True)
class AffineExpr:
    """A small affine expression represented as ``sum(coeffs*x) + constant``."""

    coefficients: tuple[tuple[str, int], ...] = ()
    constant: int = 0

    @classmethod
    def variable(cls, name: str) -> "AffineExpr":
        return cls(((name, 1),), 0)

    @classmethod
    def literal(cls, value: int) -> "AffineExpr":
        return cls((), value)

    def coefficient(self, name: str) -> int:
        return dict(self.coefficients).get(name, 0)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.coefficients)

    def format(self) -> str:
        parts: list[str] = []
        for name, coefficient in self.coefficients:
            if coefficient == 1:
                parts.append(name)
            elif coefficient == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"{coefficient}*{name}")
        if self.constant or not parts:
            parts.append(str(self.constant))
        return " + ".join(parts).replace("+ -", "- ")


@dataclass(frozen=True)
class RangeSpec:
    start: ast.AST
    stop: ast.AST
    step: ast.AST
    start_affine: AffineExpr | None
    stop_affine: AffineExpr | None
    step_value: int | None
    location: SourceLocation

    @property
    def is_affine(self) -> bool:
        return self.start_affine is not None and self.stop_affine is not None and self.step_value not in (None, 0)


@dataclass(frozen=True)
class BufferAccess:
    buffer: str
    index: AffineExpr | None
    kind: BufferAccessKind
    location: SourceLocation
    syntax: str = ""
    same_iteration: bool = False
    unique: bool = False
    axes: tuple[AffineExpr | None, ...] = ()
    bounds: tuple = ()
    multiplicity: int = 1


@dataclass(frozen=True)
class Reduction:
    target: str
    operator: str
    value_source: str
    identity: str | None
    location: SourceLocation
    commutative: bool = True
    associative: bool = True


@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    detail: str
    location: SourceLocation
    target: str | None = None


@dataclass(frozen=True)
class AliasRisk:
    kind: AliasRiskKind
    buffers: tuple[str, ...]
    detail: str
    location: SourceLocation | None = None


@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    code: DiagnosticCode
    message: str
    location: SourceLocation | None = None
    loop_id: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == DiagnosticSeverity.ERROR

    def format(self) -> str:
        prefix = self.code.value
        if self.location is not None:
            prefix += f" at {self.location.line}:{self.location.column}"
        return f"{prefix}: {self.message}"


class HIRStatementKind(_StrEnum):
    ASSIGN = "assign"
    AUG_ASSIGN = "aug_assign"
    EXPRESSION = "expression"
    IF = "if"
    CONTINUE = "continue"
    PASS = "pass"


class BasicBlockKind(_StrEnum):
    ENTRY = "entry"
    LINEAR = "linear"
    LOOP_HEADER = "loop_header"
    LOOP_BODY = "loop_body"
    EXIT = "exit"


@dataclass(frozen=True)
class HIRStatement:
    kind: HIRStatementKind
    target: str | None
    expression: str
    value_type: TypeRef
    location: SourceLocation


@dataclass(frozen=True)
class BasicBlock:
    """A stable structured CFG block used by lowering and diagnostics."""

    stable_id: str
    kind: BasicBlockKind
    source: str
    location: SourceLocation | None
    successors: tuple[str, ...]
    loop_id: str | None = None


def _operation_cost(tree):
    return sum(2 if isinstance(item, ast.Subscript) else 8 if isinstance(item, ast.Call) else 1 if isinstance(
        item, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Assign, ast.AugAssign, ast.If)) else 0
        for item in ast.walk(tree))


@dataclass(frozen=True)
class OrderedReductionRegion:
    inner_index: str
    count: int
    accumulator: str
    initializer_statement: int
    loop_statement: int
    body_identity: str
    prefix_cost: int
    term_cost: int
    epilogue_cost: int
    control_cost: int
    term_reads: int
    prefix_reads: int
    epilogue_reads: int
    diagnostics: tuple[str, ...] = ()

    @property
    def eligible(self):
        return not self.diagnostics

    def operation_cost(self, kind="whole", inner_count=None):
        count = self.count if inner_count is None else inner_count
        return (self.prefix_cost + count * self.term_cost + self.control_cost
                + (self.epilogue_cost if kind in ("whole", "final") else 3)
                + (2 if kind in ("resume", "final") else 0))


@dataclass(frozen=True)
class InnerRangeContract:
    kind: str
    count: int
    begin_parameter: int
    end_parameter: int

    def __post_init__(self):
        if (self.kind not in {"initial", "resume", "final"}
                or type(self.count) is not int or not (3 if self.kind == "resume" else 2) <= self.count <= 0x7fffffff
                or type(self.begin_parameter) is not int or type(self.end_parameter) is not int
                or self.begin_parameter < 0 or self.end_parameter != self.begin_parameter + 1):
            raise ValueError("invalid ordered inner range contract")

    def guards(self, begin, end):
        conditions = [f"not (0 <= {begin} < {end} <= {self.count})"]
        if self.kind == "initial":
            conditions.append(f"{begin} != 0 or {end} >= {self.count}")
        elif self.kind == "resume":
            conditions.append(f"{begin} <= 0 or {end} >= {self.count}")
        elif self.kind == "final":
            conditions.append(f"{begin} <= 0 or {end} != {self.count}")
        else:
            raise ValueError("unknown ordered reduction variant")
        return tuple(conditions)

    def validate(self, begin, end):
        from numbers import Integral

        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or not -0x80000000 <= int(value) <= 0x7fffffff for value in (begin, end)):
            raise TypeError("ordered inner controls require exact signed int32 values")
        valid = 0 <= begin < end <= self.count
        valid = valid and ({"initial": begin == 0 and end < self.count,
                           "resume": begin > 0 and end < self.count,
                           "final": begin > 0 and end == self.count}[self.kind])
        if not valid:
            raise ValueError(f"invalid {self.kind} inner range [{begin},{end}) for [0,{self.count})")


@dataclass(frozen=True)
class RangeLoopHIR:
    """Typed summary of one candidate range loop."""

    stable_id: str
    function_name: str
    index_name: str
    range_spec: RangeSpec | None
    source_location: SourceLocation
    source: str
    statements: tuple[HIRStatement, ...]
    buffer_reads: tuple[BufferAccess, ...]
    buffer_writes: tuple[BufferAccess, ...]
    reductions: tuple[Reduction, ...]
    effects: tuple[Effect, ...]
    alias_risks: tuple[AliasRisk, ...]
    diagnostics: tuple[Diagnostic, ...]
    parallel_eligible: bool
    reason: str
    operation_cost: int = 1
    local_reductions: tuple[OrderedReductionRegion, ...] = ()
    fully_written_outputs: tuple[str, ...] = ()

    @property
    def loop_id(self) -> str:
        return self.stable_id

    @property
    def eligible(self) -> bool:
        return self.parallel_eligible

    @property
    def parallel_reason(self) -> str:
        return self.reason


_EXTENT_OPERATORS = {"add": (operator.add, "+"), "sub": (operator.sub, "-"),
                     "mul": (operator.mul, "*"), "div": (operator.floordiv, "//"),
                     "min": (min, "min"), "max": (max, "max")}


def _extent_expression(value):
    """Validate structured compiler expressions, never evaluate source strings."""
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("invalid resource extent expression")
    if value[0] == "n" and len(value) == 1:
        return ("n",)
    if value[0] == "c" and len(value) == 2 and type(value[1]) is int and 0 <= value[1] <= 0x7fffffff:
        return ("c", value[1])
    if value[0] in _EXTENT_OPERATORS and len(value) == 3:
        left, right = _extent_expression(value[1]), _extent_expression(value[2])
        if value[0] == "div" and (right[0] != "c" or right[1] <= 0):
            raise ValueError("resource extent divisor must be a positive constant")
        return (value[0], left, right)
    raise ValueError("invalid resource extent expression")


def _extent_value(expression, count):
    if expression[0] == "c":
        return expression[1]
    if expression[0] == "n":
        return count
    return _EXTENT_OPERATORS[expression[0]][0](_extent_value(expression[1], count),
                                             _extent_value(expression[2], count))


def _extent_source(expression, count):
    if expression[0] == "c":
        return str(expression[1])
    if expression[0] == "n":
        return count
    if expression[0] in ("min", "max"):
        return f"{expression[0]}({_extent_source(expression[1], count)}, {_extent_source(expression[2], count)})"
    return f"({_extent_source(expression[1], count)} {_EXTENT_OPERATORS[expression[0]][1]} {_extent_source(expression[2], count)})"


def _extent_has_domain(expression):
    return expression[0] == "n" or (len(expression) == 3 and
        (_extent_has_domain(expression[1]) or _extent_has_domain(expression[2])))


@dataclass(frozen=True)
class ResourceExtent:
    """Per-parameter address proof consumed by CPU, Vulkan and C ABI lowering.

    Bounds are in evaluation order, including intermediate integer operations;
    guards prevent i32 overflow before a shader or native row reads any data.
    """
    parameter: int
    exact_domain: bool
    bounds: tuple
    upper_indices: tuple

    @classmethod
    def from_record(cls, value):
        if (not isinstance(value, dict) or set(value) != {"parameter", "exact_domain", "bounds", "upper_indices"}
                or type(value["parameter"]) is not int or value["parameter"] < 0
                or type(value["exact_domain"]) is not bool):
            raise ValueError("invalid resource extent contract")
        pairs = tuple(tuple(_extent_expression(part) for part in pair) for pair in value["bounds"])
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("resource extent requires lower/upper bounds")
        upper = tuple(_extent_expression(item) for item in value["upper_indices"])
        if any(item not in {pair[1] for pair in pairs} for item in upper):
            raise ValueError("resource extent omitted an index proof")
        return cls(value["parameter"], value["exact_domain"], pairs, upper)

    def validate(self, count, length):
        if type(length) is not int or not 0 <= length <= 0x7fffffff:
            raise ValueError("continuation resource length exceeds the i32 ABI")
        if self.exact_domain and length != count:
            raise ValueError("pointwise continuation requires same-length buffers in its domain")
        if count:
            for low, high in self.bounds:
                lo, hi = _extent_value(low, count), _extent_value(high, count)
                if not 0 <= lo <= hi <= 0x7fffffff:
                    raise ValueError("continuation resource index is negative or exceeds i32")
        for upper in self.upper_indices:
            required = 0 if count == 0 and _extent_has_domain(upper) else _extent_value(upper, count) + 1
            if length < required:
                raise ValueError(f"continuation resource {self.parameter} requires at least {required} elements, got {length}")

    def guards(self, count, length):
        """Emit this same proof for a no-GIL native wrapper; count is int64."""
        result = [f"{length} < 0"]
        if self.exact_domain:
            result.append(f"{length} != {count}")
        for low, high in self.bounds:
            lo, hi = _extent_source(low, count), _extent_source(high, count)
            result.append(f"{count} > 0 and not (0 <= {lo} <= {hi} <= 2147483647)")
        for upper in self.upper_indices:
            required = f"({_extent_source(upper, count)} + 1)"
            prefix = f"{count} > 0 and " if _extent_has_domain(upper) else ""
            result.append(f"{prefix}{length} < {required}")
        return result


def _bounded_index(node, index_name, local_bounds, local_indices=None):
    """Conservative nonnegative intervals, with exact runtime domain N."""
    bounds = []
    zero = ("c", 0)

    def clipped_float(value):
        # A dynamic f32 address is admitted only after explicit NaN handling
        # and clipping. Every integer up to 2**24-1 is exact in f32.
        def call(part, name):
            return (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                    and part.func.id == name and len(part.args) == 2 and not part.keywords)
        if not call(value, "min"):
            raise ValueError
        upper, lower = value.args
        maximum = upper.value if isinstance(upper, ast.Constant) and type(upper.value) in (int, float) else None
        if (isinstance(upper, ast.Call) and isinstance(upper.func, ast.Name) and upper.func.id == "float"
                and len(upper.args) == 1 and not upper.keywords):
            maximum = _constant_int(upper.args[0])
        if maximum is None or not 0 <= maximum <= 0xffffff or int(maximum) != maximum or not call(lower, "max"):
            raise ValueError
        floor, clean = lower.args
        if (not isinstance(floor, ast.Constant) or floor.value != 0
                or not isinstance(clean, ast.IfExp) or not isinstance(clean.orelse, ast.Constant)
                or clean.orelse.value != 0 or not isinstance(clean.test, ast.Compare)
                or len(clean.test.ops) != 1 or not isinstance(clean.test.ops[0], ast.Eq)
                or len(clean.test.comparators) != 1):
            raise ValueError
        expressions = (clean.body, clean.test.left, clean.test.comparators[0])
        if len({ast.dump(part, include_attributes=False) for part in expressions}) != 1:
            raise ValueError
        return zero, ("c", int(maximum))

    def visit(value):
        literal = _constant_int(value)
        if literal is not None and 0 <= literal <= 0x7fffffff:
            pair = (("c", literal), ("c", literal))
        elif isinstance(value, ast.Name) and value.id == index_name:
            pair = (zero, ("sub", ("n",), ("c", 1)))
        elif isinstance(value, ast.Name) and value.id in local_bounds:
            pair = (zero, ("c", local_bounds[value.id] - 1))
        elif isinstance(value, ast.Name) and local_indices is not None and value.id in local_indices:
            bounds.extend(local_indices[value.id][:-1])
            pair = local_indices[value.id][-1]
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "int"
              and len(value.args) == 1 and not value.keywords):
            pair = clipped_float(value.args[0])
        elif isinstance(value, ast.IfExp):
            left, right = visit(value.body), visit(value.orelse)
            pair = (("min", left[0], right[0]), ("max", left[1], right[1]))
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
              and value.func.id in ("min", "max") and len(value.args) == 2 and not value.keywords):
            left, right = (visit(arg) for arg in value.args)
            pair = ((value.func.id, left[0], right[0]), (value.func.id, left[1], right[1]))
        elif isinstance(value, ast.BinOp):
            left = visit(value.left)
            if isinstance(value.op, (ast.FloorDiv, ast.Mod)):
                divisor = _constant_int(value.right)
                if divisor is None or not 0 < divisor <= 0x7fffffff:
                    raise ValueError
                pair = (("div", left[0], ("c", divisor)), ("div", left[1], ("c", divisor))) \
                    if isinstance(value.op, ast.FloorDiv) else (zero, ("c", divisor - 1))
            else:
                right = visit(value.right)
                if isinstance(value.op, ast.Add):
                    pair = (("add", left[0], right[0]), ("add", left[1], right[1]))
                elif isinstance(value.op, ast.Sub):
                    pair = (("sub", left[0], right[1]), ("sub", left[1], right[0]))
                elif isinstance(value.op, ast.Mult) and (
                        _constant_int(value.left) is not None or _constant_int(value.right) is not None):
                    pair = (("mul", left[0], right[0]), ("mul", left[1], right[1]))
                else:
                    raise ValueError
        else:
            raise ValueError
        bounds.append(pair)
        return pair

    try:
        visit(node)
        if any(not 0 <= _extent_value(low, 1) <= _extent_value(high, 1) <= 0x7fffffff for low, high in bounds):
            return ()
        return tuple(bounds)
    except ValueError:
        return ()


@dataclass(frozen=True)
class ContinuationPlan:
    """A row-boundary proof, independent of parallel selection or a scheduler."""

    eligible: bool
    diagnostics: tuple[Diagnostic, ...]
    index_name: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    fixed_parameters: tuple[str, ...] = ()
    operation_cost: int = 0
    read_elements_per_iteration: int = 0
    write_elements_per_iteration: int = 0
    cost_source: str = "hir_ast_weight"
    granularity: str = "complete_logical_iteration"
    cross_slice_state: tuple = ()
    resource_extents: tuple[ResourceExtent, ...] = ()
    gathered: bool = False
    fully_written_outputs: tuple[str, ...] = ()
    ordered_reduction: OrderedReductionRegion | None = None
    inner_range: InnerRangeContract | None = None

    def require(self) -> None:
        if not self.eligible:
            raise TypeError("work item cannot continue: " + "; ".join(item.format() for item in self.diagnostics))


@dataclass(frozen=True)
class FunctionHIR:
    name: str
    source: str
    arguments: tuple[HIRArgument, ...]
    loops: tuple[RangeLoopHIR, ...]
    diagnostics: tuple[Diagnostic, ...]
    statements: tuple[HIRStatement, ...] = ()
    blocks: tuple[BasicBlock, ...] = ()

    @property
    def eligible_loops(self) -> tuple[RangeLoopHIR, ...]:
        if any(diagnostic.is_error for diagnostic in self.diagnostics if diagnostic.loop_id is None):
            return ()
        return tuple(loop for loop in self.loops if loop.parallel_eligible)

    @property
    def parallel_eligible(self) -> bool:
        return bool(self.eligible_loops)

    @property
    def operation_cost(self) -> int:
        return sum(loop.operation_cost for loop in self.eligible_loops)

    @property
    def diagnostics_by_loop(self) -> Mapping[str, tuple[Diagnostic, ...]]:
        return {
            loop.stable_id: loop.diagnostics
            for loop in self.loops
        }

    @property
    def entry_block(self) -> BasicBlock:
        return self.blocks[0]

    @property
    def exit_block(self) -> BasicBlock:
        return self.blocks[-1]

    @property
    def continuation_plan(self) -> ContinuationPlan:
        # Continuation rejects every overlapping output at actual binding.
        # Parallel lowering retains its separate conservative alias diagnostic.
        diagnostics = list(item for item in self.diagnostics if item.is_error and item.code != DiagnosticCode.ALIAS_RISK)

        def reject(code, reason, location=None):
            diagnostics.append(Diagnostic(DiagnosticSeverity.ERROR, code, reason, location))

        if len(self.loops) != 1:
            reject(DiagnosticCode.UNSUPPORTED_RANGE, "continuation requires exactly one logical range loop")
            return ContinuationPlan(False, tuple(diagnostics))
        loop = self.loops[0]
        gathered = bool(loop.local_reductions) or any(
            access.index != AffineExpr.variable(loop.index_name) for access in loop.buffer_reads)
        if loop.reductions:
            reject(DiagnosticCode.INVALID_REDUCTION, "continuation has no cross-slice reduction state or merge",
                   loop.reductions[0].location)
        if any(effect.kind == EffectKind.CONTROL_FLOW for effect in loop.effects):
            reject(DiagnosticCode.UNSUPPORTED_CONTROL_FLOW,
                   "loop finalizer/control flow is not part of row continuation", loop.source_location)
        if any(block.kind == BasicBlockKind.LINEAR for block in self.blocks):
            reject(DiagnosticCode.UNSUPPORTED_STATEMENT, "preheader/finalizer state is not part of row continuation")
        spec = loop.range_spec
        if spec is None or spec.start_affine != AffineExpr.literal(0) or spec.step_value != 1:
            reject(DiagnosticCode.UNSUPPORTED_RANGE, "continuation requires a zero-based, unit-step logical domain",
                   loop.source_location)
        arguments = tuple(argument.name for argument in self.arguments)
        argument_types = {argument.name: argument.type_ref.value_type for argument in self.arguments}
        if any(argument.type_ref.value_type == ValueType.UNKNOWN for argument in self.arguments):
            reject(DiagnosticCode.INVALID_ARGUMENT, "continuation requires bound array/scalar type facts")
        for access in (*loop.buffer_reads, *loop.buffer_writes):
            if access.buffer in argument_types and argument_types[access.buffer] != ValueType.ARRAY:
                reject(DiagnosticCode.INVALID_ARGUMENT, f"buffer '{access.buffer}' requires an array binding", access.location)
            if (access.buffer not in arguments or len(access.axes) != 1 or not access.bounds or
                    (access.kind != BufferAccessKind.READ and access.index != AffineExpr.variable(loop.index_name))):
                reject(DiagnosticCode.NON_AFFINE_INDEX,
                       f"continuation requires a proven one-dimensional parameter index and direct output[{loop.index_name}]: {access.syntax}",
                       access.location)
        inputs = tuple(sorted({access.buffer for access in loop.buffer_reads}))
        outputs = tuple(sorted({access.buffer for access in loop.buffer_writes}))
        if not outputs:
            reject(DiagnosticCode.INDIRECT_WRITE, "continuation requires an output row")
        if set(inputs) & set(outputs):
            reject(DiagnosticCode.ALIAS_RISK, "continuation inputs must be immutable and outputs write-only")
        if gathered and (len(loop.buffer_writes) != 1 or outputs != loop.fully_written_outputs):
            reject(DiagnosticCode.INDIRECT_WRITE, "gather/reduce requires one unconditional write-only output row")
        extents = []
        for parameter, name in enumerate(arguments):
            if argument_types[name] != ValueType.ARRAY:
                continue
            accesses = [item for item in (*loop.buffer_reads, *loop.buffer_writes) if item.buffer == name]
            bounds = tuple(dict.fromkeys(pair for access in accesses for pair in access.bounds))
            upper = tuple(dict.fromkeys(access.bounds[-1][1] for access in accesses if access.bounds))
            extents.append(ResourceExtent(parameter, not gathered, bounds, upper))
        return ContinuationPlan(not diagnostics, tuple(diagnostics), loop.index_name, inputs, outputs, arguments,
                                loop.operation_cost, sum(access.multiplicity for access in loop.buffer_reads),
                                len(loop.buffer_writes), resource_extents=tuple(extents), gathered=gathered,
                                fully_written_outputs=loop.fully_written_outputs if not diagnostics else (),
                                ordered_reduction=loop.local_reductions[0]
                                if not diagnostics and len(loop.local_reductions) == 1 else None)

    @property
    def continuable(self) -> bool:
        return self.continuation_plan.eligible


# Familiar aliases for callers that prefer the long names.
HIRFunction = FunctionHIR
HIRRangeLoop = RangeLoopHIR
BufferReadWrite = BufferAccess
ReductionSpec = Reduction


_PURE_BUILTINS = frozenset({"abs", "min", "max", "round", "float", "int"})
_PURE_MATH = frozenset({
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil",
    "copysign", "cos", "cosh", "degrees", "exp", "expm1", "fabs", "floor",
    "fmod", "frexp", "hypot", "isfinite", "isinf", "isnan", "ldexp", "log",
    "log10", "log1p", "log2", "pow", "radians", "sin", "sinh", "sqrt",
    "tan", "tanh", "trunc",
})
_PURE_NUMPY = frozenset({
    "abs", "absolute", "ceil", "clip", "cos", "exp", "fabs", "floor", "log",
    "log10", "log1p", "maximum", "minimum", "power", "sin", "sqrt", "tan",
    "tanh",
})
_MUTATING_METHODS = frozenset({
    "append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort",
    "update", "setdefault", "discard", "add",
})
_REDUCTION_OPERATORS = frozenset({"+", "*", "min", "max"})


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _location(node: ast.AST) -> SourceLocation:
    return SourceLocation.from_node(node)


def _constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _constant_int(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _constant_int(node.left), _constant_int(node.right)
        if left is not None and right is not None:
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            operation = operations.get(type(node.op))
            if operation is not None and (right != 0 or not isinstance(node.op, (ast.FloorDiv, ast.Mod))):
                return operation(left, right)
    return None


def _combine_affine(left: AffineExpr, right: AffineExpr, sign: int) -> AffineExpr:
    coefficients = dict(left.coefficients)
    for name, value in right.coefficients:
        coefficients[name] = coefficients.get(name, 0) + sign * value
    return AffineExpr(
        tuple(sorted((name, value) for name, value in coefficients.items() if value)),
        left.constant + sign * right.constant,
    )


def _scale_affine(value: AffineExpr, factor: int) -> AffineExpr:
    return AffineExpr(
        tuple((name, coefficient * factor) for name, coefficient in value.coefficients if coefficient * factor),
        value.constant * factor,
    )


def _affine(node: ast.AST, *, known_calls: Mapping[str, str] | None = None) -> AffineExpr | None:
    if isinstance(node, ast.Name):
        return AffineExpr.variable(node.id)
    value = _constant_int(node)
    if value is not None:
        return AffineExpr.literal(value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _affine(node.left, known_calls=known_calls)
        right = _affine(node.right, known_calls=known_calls)
        if left is not None and right is not None:
            return _combine_affine(left, right, 1 if isinstance(node.op, ast.Add) else -1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_constant = _constant_int(node.left)
        right_constant = _constant_int(node.right)
        if left_constant is not None:
            value = _affine(node.right, known_calls=known_calls)
            return _scale_affine(value, left_constant) if value is not None else None
        if right_constant is not None:
            value = _affine(node.left, known_calls=known_calls)
            return _scale_affine(value, right_constant) if value is not None else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
        argument = node.args[0]
        if isinstance(argument, ast.Name):
            return AffineExpr(((f"len({argument.id})", 1),), 0)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr in {"shape", "strides"}:
        if isinstance(node.value.value, ast.Name):
            index = node.slice
            index_value = _constant_int(index)
            if index_value is not None:
                return AffineExpr(((f"{node.value.value.id}.{node.value.attr}[{index_value}]", 1),), 0)
    return None


def _annotation_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return _unparse(node)


def _type_from_annotation(annotation: ast.AST | None) -> TypeRef:
    name = (_annotation_name(annotation) or "").lower()
    if name in {"int", "int32", "int64", "np.int32", "np.int64"}:
        return TypeRef(ValueType.INT)
    if name in {"float", "float32", "float64", "np.float32", "np.float64"}:
        return TypeRef(ValueType.FLOAT)
    if name in {"bool", "np.bool_"}:
        return TypeRef(ValueType.BOOL)
    if "ndarray" in name or "array" in name or "buffer" in name:
        return TypeRef(ValueType.ARRAY, mutable=True)
    return TypeRef(ValueType.UNKNOWN)


def _index_node(node: ast.Subscript) -> ast.AST:
    # Python 3.8+ stores the subscript expression directly in ``slice``.
    return node.slice


def _is_same_iteration_index(index: AffineExpr | None, loop_name: str) -> bool:
    return index is not None and index.coefficients == ((loop_name, 1),) and index.constant == 0


def _index_is_unique(index: AffineExpr | None, loop_name: str) -> bool:
    if index is None:
        return False
    coefficient = index.coefficient(loop_name)
    return coefficient != 0 and all(name == loop_name for name in index.variables)


def _may_cross_iterations(left: AffineExpr | None, right: AffineExpr | None, loop_name: str) -> bool:
    if left is None or right is None:
        return True
    if left == right:
        return left.coefficient(loop_name) == 0
    left_uniform = tuple(item for item in left.coefficients if item[0] != loop_name)
    right_uniform = tuple(item for item in right.coefficients if item[0] != loop_name)
    if left_uniform != right_uniform:
        return True
    divisor = math.gcd(left.coefficient(loop_name), right.coefficient(loop_name))
    # Distinct residue classes (e.g. 2*i and 2*i+1) cannot share an element.
    return not divisor or (left.constant - right.constant) % divisor == 0


def _call_name(node: ast.Call) -> tuple[str | None, str | None]:
    if isinstance(node.func, ast.Name):
        return node.func.id, None
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return node.func.value.id, node.func.attr
    return None, None


def _access_axes(node, imports):
    index = _index_node(node)
    return tuple(_affine(axis, known_calls=imports) for axis in
                 (index.elts if isinstance(index, ast.Tuple) else (index,)))


def _row_local(axes, loop):
    """One or more unshifted induction axes, with literal remaining axes.

    Restrict this proof to nonnegative increasing ranges: crossing from -1
    to positive indices can revisit a NumPy row after negative-index wrapping.
    Dependence analysis still checks that different accesses share an
    independent axis; x[i, 0] and x[0, i] are not automatically independent.
    """
    if len(axes) < 2 or not isinstance(loop.target, ast.Name):
        return False
    call = loop.iter
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "range":
        return False
    start = 0 if len(call.args) == 1 else _constant_int(call.args[0]) if call.args else None
    step = _constant_int(call.args[2]) if len(call.args) == 3 else 1
    if start is None or start < 0 or step is None or step <= 0:
        return False
    same = lambda axis: _is_same_iteration_index(axis, loop.target.id)
    return any(same(axis) for axis in axes) and all(
        axis is not None and (not axis.coefficients or same(axis)) for axis in axes)


def _buffer_dependences(reads, writes, index_name, loop_id, aliases):
    """One dependence rule for symbolic buffers and admitted runtime aliases."""
    def same_buffer(left, right):
        return aliases.get(left, left) == aliases.get(right, right)

    def may_cross(left, right):
        # Multidimensional arrays intersect only if every axis intersects.
        # Equal induction axes separate row-local work even when other
        # coordinates differ. Mixed axes/shifted rows remain conservative.
        if left.axes and len(left.axes) == len(right.axes):
            return all(_may_cross_iterations(a, b, index_name)
                       for a, b in zip(left.axes, right.axes))
        return _may_cross_iterations(left.index, right.index, index_name)

    result = []
    for access in reads:
        if any(same_buffer(write.buffer, access.buffer) and
               may_cross(access, write)
               for write in writes):
            result.append(Diagnostic(
                DiagnosticSeverity.ERROR, DiagnosticCode.LOOP_CARRIED_READ,
                f"read {access.syntax} depends on another loop iteration",
                access.location, loop_id))
    for index, write in enumerate(writes):
        if any(same_buffer(other.buffer, write.buffer) and
               may_cross(write, other)
               for other in writes[index + 1:]):
            result.append(Diagnostic(
                DiagnosticSeverity.ERROR, DiagnosticCode.LOOP_CARRIED_WRITE,
                f"different writes to '{write.buffer}' may overlap between iterations",
                write.location, loop_id))
    return tuple(result)


def parallel_alias_pairs(hir: FunctionHIR) -> tuple[tuple[str, str], ...]:
    """Equal-layout parameter pairs proven safe for this parallel publication.

    Restrict this first proof to one loop. Multiple loops can be fused by the
    backend, so per-loop dependence checks alone do not prove their ordering.
    The small result also travels with cooked bytecode; Players need no source
    parsing or second alias analysis.
    """
    if len(hir.loops) != 1 or not hir.eligible_loops:
        return ()
    function = next(node for node in _parse(hir.source).body
                    if isinstance(node, ast.FunctionDef) and node.name == hir.name)
    for statement in function.body:
        if isinstance(statement, ast.For):
            continue
        if any(isinstance(node, (ast.Call, ast.Subscript, ast.Attribute))
               for node in ast.walk(statement)):
            return ()
    loop = hir.loops[0]
    names = sorted({access.buffer for access in (*loop.buffer_reads, *loop.buffer_writes)})
    return tuple((left, right) for index, left in enumerate(names) for right in names[index + 1:]
                 if not analyze_buffer_aliases(hir, ((left, right),)))


def analyze_buffer_aliases(hir: FunctionHIR, groups: tuple[tuple[str, ...], ...]) -> tuple[Diagnostic, ...]:
    """Check equal-layout shared buffers once per prepared argument alias pattern.

    Groups contain parameter names whose arrays have exactly the same memory
    span, dtype and shape. Offset/reshaped views must be rejected by the backend
    before this analysis: symbolic indices would not denote the same bytes.
    """
    if not groups:
        return ()
    # The minimal HIR cannot prove alias safety for expressions it did not
    # inspect. Do not mistake an absent access record for independent buffers.
    # Affine buffer index disjointness is valid for any subset of range values;
    # a non-affine trip count (e.g. shape[0] // 2) does not hide body accesses.
    ranges = {loop.stable_id for loop in hir.loops
              if loop.range_spec is not None and loop.range_spec.step_value not in (None, 0)}
    incomplete = next((item for item in hir.diagnostics
                       if item.is_error and item.code != DiagnosticCode.ALIAS_RISK
                       and not (item.code == DiagnosticCode.UNSUPPORTED_RANGE and item.loop_id in ranges)), None)
    if incomplete is not None:
        return (Diagnostic(
            DiagnosticSeverity.ERROR, DiagnosticCode.ALIAS_RISK,
            f"shared-buffer parallel access is not proven: {incomplete.message}",
            incomplete.location, incomplete.loop_id),)
    arguments = {argument.name for argument in hir.arguments}
    for loop in hir.loops:
        for access in (*loop.buffer_reads, *loop.buffer_writes):
            if access.buffer not in arguments:
                return (Diagnostic(
                    DiagnosticSeverity.ERROR, DiagnosticCode.ALIAS_RISK,
                    f"shared-buffer analysis requires direct array parameters; '{access.buffer}' is an indirect binding",
                    access.location, loop.stable_id),)
    aliases = {name: group[0] for group in groups for name in group}
    return tuple(diagnostic for loop in hir.loops for diagnostic in _buffer_dependences(
        loop.buffer_reads, loop.buffer_writes, loop.index_name, loop.stable_id, aliases))


class _Analyzer:
    def __init__(self, tree: ast.Module, source: str, function: ast.FunctionDef | ast.AsyncFunctionDef,
                 call_bindings: Mapping[str, object] | None = None,
                 argument_types: Mapping[str, TypeRef] | None = None):
        self.tree = tree
        self.source = source
        self.function = function
        self.call_bindings = call_bindings
        self.bound_arguments = argument_types is not None
        self.local_names = {node.id for node in ast.walk(function)
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.diagnostics: list[Diagnostic] = []
        self.imports: dict[str, str] = {"math": "math", "np": "numpy", "numpy": "numpy"}
        self._collect_imports()
        self.arg_types: dict[str, TypeRef] = {}
        self.initialized_scalars: dict[str, str] = {}
        self._collect_arguments()
        self.arg_types.update(argument_types or {})
        self._collect_scalar_initializers()
        self._local_bounds = {}
        self._local_reductions = []
        self._local_loop_extra_cost = 0

    def _collect_imports(self) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    self.imports[item.asname or item.name.split(".")[0]] = item.name
            elif isinstance(node, ast.ImportFrom) and node.module in {"math", "numpy"}:
                for item in node.names:
                    self.imports[item.asname or item.name] = f"{node.module}.{item.name}"

    def _collect_arguments(self) -> None:
        arguments = [*self.function.args.posonlyargs, *self.function.args.args, *self.function.args.kwonlyargs]
        for argument in arguments:
            self.arg_types[argument.arg] = _type_from_annotation(argument.annotation)

    def _collect_scalar_initializers(self) -> None:
        for node in self.function.body:
            if isinstance(node, ast.For):
                break
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if self._expr_type(node.value).is_scalar:
                    self.initialized_scalars[node.targets[0].id] = _unparse(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                if self._expr_type(node.value).is_scalar:
                    self.initialized_scalars[node.target.id] = _unparse(node.value)

    def diagnostic(
        self,
        code: DiagnosticCode,
        message: str,
        node: ast.AST,
        *,
        loop_id: str | None = None,
    ) -> Diagnostic:
        value = Diagnostic(DiagnosticSeverity.ERROR, code, message, _location(node), loop_id)
        self.diagnostics.append(value)
        return value

    def _loop_id(self, loop: ast.For, ordinal: int) -> str:
        # The ordinal disambiguates identical loops while the AST dump makes
        # the ID independent of whitespace and source line shifts.
        payload = "|".join((self.function.name, str(ordinal), ast.dump(loop, include_attributes=False)))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.function.name}:loop:{digest}"

    def _range_spec(self, loop: ast.For, loop_id: str) -> RangeSpec | None:
        call = loop.iter
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "range":
            self.diagnostic(DiagnosticCode.UNSUPPORTED_RANGE, "only range(...) loops are eligible", loop.iter, loop_id=loop_id)
            return None
        if not 1 <= len(call.args) <= 3 or call.keywords:
            self.diagnostic(DiagnosticCode.UNSUPPORTED_RANGE, "range must have one to three positional arguments", call, loop_id=loop_id)
            return None
        if len(call.args) == 1:
            start, stop, step = ast.Constant(0), call.args[0], ast.Constant(1)
        elif len(call.args) == 2:
            start, stop, step = call.args[0], call.args[1], ast.Constant(1)
        else:
            start, stop, step = call.args
        start_affine = _affine(start, known_calls=self.imports)
        stop_affine = _affine(stop, known_calls=self.imports)
        step_value = _constant_int(step)
        if start_affine is None or stop_affine is None or step_value in (None, 0):
            self.diagnostic(
                DiagnosticCode.UNSUPPORTED_RANGE,
                "range bounds must be affine and step must be a known non-zero integer",
                call,
                loop_id=loop_id,
            )
        return RangeSpec(start, stop, step, start_affine, stop_affine, step_value, _location(call))

    def _expr_type(self, node: ast.AST) -> TypeRef:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return TypeRef(ValueType.BOOL)
            if isinstance(node.value, int):
                return TypeRef(ValueType.INT)
            if isinstance(node.value, float):
                return TypeRef(ValueType.FLOAT)
            return TypeRef(ValueType.UNKNOWN)
        if isinstance(node, ast.Name):
            return self.arg_types.get(node.id, TypeRef(ValueType.SCALAR))
        if isinstance(node, ast.Subscript):
            return TypeRef(ValueType.SCALAR)
        if isinstance(node, ast.UnaryOp):
            return self._expr_type(node.operand)
        if isinstance(node, ast.IfExp):
            left, right = self._expr_type(node.body), self._expr_type(node.orelse)
            return left if left == right else TypeRef(ValueType.SCALAR if left.is_scalar and right.is_scalar else ValueType.UNKNOWN)
        if isinstance(node, ast.BinOp):
            left, right = self._expr_type(node.left), self._expr_type(node.right)
            if ValueType.FLOAT in {left.value_type, right.value_type}:
                return TypeRef(ValueType.FLOAT)
            if left.is_scalar and right.is_scalar:
                return TypeRef(ValueType.INT)
            return TypeRef(ValueType.UNKNOWN)
        if isinstance(node, ast.Compare) or isinstance(node, ast.BoolOp):
            return TypeRef(ValueType.BOOL)
        if isinstance(node, ast.Attribute):
            if node.attr in {"shape", "ndim", "size", "strides"}:
                return TypeRef(ValueType.INT)
            return TypeRef(ValueType.UNKNOWN)
        if isinstance(node, ast.Call):
            name, attr = _call_name(node)
            if name in _PURE_BUILTINS and attr is None:
                return TypeRef(ValueType.SCALAR)
            if attr in _PURE_MATH | _PURE_NUMPY:
                return TypeRef(ValueType.SCALAR)
            if name == "len" and attr is None:
                return TypeRef(ValueType.INT)
        return TypeRef(ValueType.UNKNOWN)

    def _is_pure_call(self, node: ast.Call) -> bool:
        name, attr = _call_name(node)
        if self.call_bindings is not None:
            if name in self.arg_types or name in self.local_names:
                return False
            if attr is None and name in _PURE_BUILTINS | {"len"}:
                return self.call_bindings.get(name, getattr(builtins, name)) is getattr(builtins, name)
            module = self.imports.get(name or "")
            if module in {"math", "numpy", "np"}:
                expected = math if module == "math" else sys.modules.get("numpy")
                if expected is None or self.call_bindings.get(name) is not expected:
                    return False
        if attr is None and name in _PURE_BUILTINS:
            return True
        module = self.imports.get(name or "")
        if attr is not None and (module == "math" or module == "numpy" or module == "np"):
            return attr in (_PURE_MATH if module == "math" else _PURE_NUMPY)
        if attr is None and name in self.imports and self.imports[name].startswith("math."):
            return name.split(".")[-1] in _PURE_MATH
        return name == "len" and len(node.args) == 1 and not node.keywords

    def _check_shape_access(self, node: ast.Attribute | ast.Subscript) -> bool:
        if isinstance(node, ast.Attribute) and node.attr in {"ndim", "size"} and isinstance(node.value, ast.Name):
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            return node.value.attr == "shape" and isinstance(node.value.value, ast.Name) and _constant_int(node.slice) is not None
        return False

    def _visit_expression(
        self,
        node: ast.AST,
        loop: ast.For,
        loop_id: str,
        reads: list[BufferAccess],
        effects: list[Effect],
    ) -> None:
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            self.diagnostic(
                DiagnosticCode.UNSUPPORTED_CONTROL_FLOW,
                f"{type(node).__name__} is not supported by the minimal HIR",
                node,
                loop_id=loop_id,
            )
            effects.append(Effect(EffectKind.CONTROL_FLOW, type(node).__name__, _location(node)))
            return
        if isinstance(node, ast.Subscript):
            if self._check_shape_access(node):
                return
            if not isinstance(node.value, ast.Name):
                self.diagnostic(DiagnosticCode.INDIRECT_WRITE, "buffer access base must be a named buffer", node, loop_id=loop_id)
            else:
                index = _affine(_index_node(node), known_calls=self.imports)
                axes = _access_axes(node, self.imports)
                bounds = _bounded_index(_index_node(node), loop.target.id, self._local_bounds, self._local_indices) \
                    if isinstance(loop.target, ast.Name) else ()
                row_local = _row_local(axes, loop)
                loop_name = loop.target.id if isinstance(loop.target, ast.Name) else ""
                if not row_local and not bounds and (index is None or any(name != loop_name for name in index.variables)):
                    self.diagnostic(
                        DiagnosticCode.NON_AFFINE_INDEX,
                        f"read from {node.value.id}[...] is not indexed by the loop induction variable",
                        node,
                        loop_id=loop_id,
                    )
                same = row_local or _is_same_iteration_index(index, loop_name=loop_name)
                reads.append(BufferAccess(node.value.id, index, BufferAccessKind.READ, _location(node), _unparse(node),
                    same, row_local or _index_is_unique(index, loop_name), axes, bounds,
                    next(iter(self._local_bounds.values()), 1)))
                effects.append(Effect(EffectKind.BUFFER_READ, f"read {node.value.id}[{_unparse(_index_node(node))}]", _location(node), node.value.id))
            for child in ast.iter_child_nodes(node):
                if child is not node.value:
                    self._visit_expression(child, loop, loop_id, reads, effects)
            return
        if (self.bound_arguments and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and
                self.arg_types.get(node.id, TypeRef(ValueType.UNKNOWN)).value_type == ValueType.ARRAY):
            self.diagnostic(DiagnosticCode.UNSUPPORTED_EXPRESSION,
                            f"array '{node.id}' must use an indexed row access, not escape into a scalar call/value",
                            node, loop_id=loop_id)
            return
        if isinstance(node, ast.Call):
            name, attr = _call_name(node)
            if attr in _MUTATING_METHODS:
                self.diagnostic(DiagnosticCode.CONTAINER_MUTATION, f"container mutation '{attr}()' is not parallel-safe", node, loop_id=loop_id)
                effects.append(Effect(EffectKind.CONTAINER_MUTATION, f"{attr}()", _location(node), name))
            elif not self._is_pure_call(node):
                self.diagnostic(DiagnosticCode.UNKNOWN_CALL, f"call '{_unparse(node.func)}' is not in the pure scalar allowlist", node, loop_id=loop_id)
                effects.append(Effect(EffectKind.UNKNOWN_CALL, f"unknown call {_unparse(node.func)}", _location(node)))
            else:
                effects.append(Effect(EffectKind.PURE_CALL, f"pure call {_unparse(node.func)}", _location(node)))
            for child in ast.iter_child_nodes(node):
                self._visit_expression(child, loop, loop_id, reads, effects)
            return
        if isinstance(node, ast.Attribute) and self._check_shape_access(node):
            return
        for child in ast.iter_child_nodes(node):
            self._visit_expression(child, loop, loop_id, reads, effects)

    def _write_access(
        self,
        target: ast.AST,
        loop: ast.For,
        loop_id: str,
        reads: list[BufferAccess],
        writes: list[BufferAccess],
        effects: list[Effect],
        *,
        aug: bool,
    ) -> None:
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            if isinstance(target, (ast.Name, ast.Attribute)):
                return
            self.diagnostic(DiagnosticCode.INDIRECT_WRITE, "only direct buffer[i] writes are eligible", target, loop_id=loop_id)
            effects.append(Effect(EffectKind.BUFFER_WRITE, "indirect buffer write", _location(target)))
            return
        buffer_name = target.value.id
        index = _affine(_index_node(target), known_calls=self.imports)
        axes = _access_axes(target, self.imports)
        row_local = _row_local(axes, loop)
        unique = row_local or _index_is_unique(index, loop.target.id if isinstance(loop.target, ast.Name) else "")
        same = row_local or _is_same_iteration_index(index, loop.target.id if isinstance(loop.target, ast.Name) else "")
        if not unique:
            self.diagnostic(
                DiagnosticCode.INDIRECT_WRITE,
                f"write to {buffer_name}[...] is not provably a unique affine index",
                target,
                loop_id=loop_id,
            )
        access_kind = BufferAccessKind.READ_WRITE if aug else BufferAccessKind.WRITE
        bounds = _bounded_index(_index_node(target), loop.target.id, self._local_bounds, self._local_indices) \
            if isinstance(loop.target, ast.Name) else ()
        access = BufferAccess(buffer_name, index, access_kind, _location(target), _unparse(target), same, unique, axes, bounds)
        writes.append(access)
        effects.append(Effect(EffectKind.BUFFER_WRITE, f"write {buffer_name}[{_unparse(_index_node(target))}]", _location(target), buffer_name))
        if aug:
            reads.append(BufferAccess(buffer_name, index, BufferAccessKind.READ, _location(target), _unparse(target), same, unique, axes, bounds))
            effects.append(Effect(EffectKind.BUFFER_READ, f"read for update {buffer_name}[{_unparse(_index_node(target))}]", _location(target), buffer_name))

    def _is_scalar_reduction(
        self,
        node: ast.AugAssign,
        loop: ast.For,
        loop_id: str,
        statements: list[HIRStatement],
        reductions: list[Reduction],
        effects: list[Effect],
    ) -> bool:
        if not isinstance(node.target, ast.Name):
            return False
        operator_map = {ast.Add: "+", ast.Mult: "*"}
        operator = next((symbol for klass, symbol in operator_map.items() if isinstance(node.op, klass)), None)
        if operator is None:
            return False
        if operator not in _REDUCTION_OPERATORS:
            return False
        target = node.target.id
        # A reduction variable must not be a mutable array argument.
        if self.arg_types.get(target, TypeRef(ValueType.SCALAR)).value_type == ValueType.ARRAY:
            return False
        if target not in self.arg_types and target not in self.initialized_scalars:
            self.diagnostic(
                DiagnosticCode.INVALID_REDUCTION,
                f"scalar reduction '{target}' has no initializer before the loop",
                node,
                loop_id=loop_id,
            )
            return False
        reduction = Reduction(target, operator, _unparse(node.value), self.initialized_scalars.get(target), _location(node))
        reductions.append(reduction)
        effects.append(Effect(EffectKind.REDUCTION, f"{target} {operator}= ...", _location(node), target))
        statements.append(HIRStatement(HIRStatementKind.AUG_ASSIGN, target, _unparse(node.value), self._expr_type(node.value), _location(node)))
        return True

    def _statement(
        self,
        node: ast.stmt,
        loop: ast.For,
        loop_id: str,
        reads: list[BufferAccess],
        writes: list[BufferAccess],
        reductions: list[Reduction],
        effects: list[Effect],
        statements: list[HIRStatement],
    ) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                self.diagnostic(DiagnosticCode.UNSUPPORTED_STATEMENT, "multiple assignment targets are not supported", node, loop_id=loop_id)
                return
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                self._write_access(target, loop, loop_id, reads, writes, effects, aug=False)
            elif isinstance(target, ast.Name):
                statements.append(HIRStatement(HIRStatementKind.ASSIGN, target.id, _unparse(node.value), self._expr_type(node.value), _location(node)))
            else:
                self.diagnostic(DiagnosticCode.INDIRECT_WRITE, "attribute or computed targets are not writable in a parallel loop", target, loop_id=loop_id)
            self._visit_expression(node.value, loop, loop_id, reads, effects)
            if isinstance(target, ast.Name):
                bounds = _bounded_index(node.value, loop.target.id, self._local_bounds, self._local_indices)
                if node in loop.body and bounds:
                    self._local_indices[target.id] = bounds
                else:
                    self._local_indices.pop(target.id, None)
            return
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                self._local_indices.pop(node.target.id, None)
            if self._is_scalar_reduction(node, loop, loop_id, statements, reductions, effects):
                self._visit_expression(node.value, loop, loop_id, reads, effects)
                return
            if isinstance(node.target, ast.Subscript):
                self._write_access(node.target, loop, loop_id, reads, writes, effects, aug=True)
                self._visit_expression(node.value, loop, loop_id, reads, effects)
                return
            self.diagnostic(DiagnosticCode.INVALID_REDUCTION, "augmented assignment is only valid for scalar reductions or direct buffer[i]", node, loop_id=loop_id)
            return
        if isinstance(node, ast.Expr):
            self._visit_expression(node.value, loop, loop_id, reads, effects)
            statements.append(HIRStatement(HIRStatementKind.EXPRESSION, None, _unparse(node.value), self._expr_type(node.value), _location(node)))
            return
        if isinstance(node, ast.If):
            self._visit_expression(node.test, loop, loop_id, reads, effects)
            for child in (*node.body, *node.orelse):
                self._statement(child, loop, loop_id, reads, writes, reductions, effects, statements)
            statements.append(HIRStatement(HIRStatementKind.IF, None, _unparse(node.test), TypeRef(ValueType.BOOL), _location(node)))
            return
        if isinstance(node, ast.Continue):
            statements.append(HIRStatement(HIRStatementKind.CONTINUE, None, "continue", TypeRef(ValueType.VOID), _location(node)))
            return
        if isinstance(node, ast.Pass):
            statements.append(HIRStatement(HIRStatementKind.PASS, None, "pass", TypeRef(ValueType.VOID), _location(node)))
            return
        if isinstance(node, ast.For) and (self._local_reduce(node, loop, loop_id, reads, effects, statements)
                                        or self._local_scan(node, loop, loop_id, reads, effects, statements)):
            return
        if isinstance(node, (ast.For, ast.While)):
            self.diagnostic(DiagnosticCode.UNSUPPORTED_NESTED_LOOP, "nested loops are outside the minimal HIR", node, loop_id=loop_id)
            return
        if isinstance(node, (ast.Break, ast.Return, ast.Try, ast.Yield, ast.YieldFrom, ast.Await, ast.With, ast.AsyncWith, ast.Raise)):
            self.diagnostic(DiagnosticCode.UNSUPPORTED_CONTROL_FLOW, f"{type(node).__name__} is not allowed in a parallel loop", node, loop_id=loop_id)
            effects.append(Effect(EffectKind.CONTROL_FLOW, type(node).__name__, _location(node)))
            return
        self.diagnostic(DiagnosticCode.UNSUPPORTED_STATEMENT, f"statement '{type(node).__name__}' is not supported", node, loop_id=loop_id)

    def _local_loop_count(self, node, loop):
        call = node.iter
        if (node not in loop.body or node.orelse or not node.body
                or not isinstance(node.target, ast.Name) or node.target.id == loop.target.id
                or node.target.id in self.arg_types or not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name) or call.func.id != "range"
                or len(call.args) != 1 or call.keywords
                or "range" in self.arg_types or "range" in self.local_names
                or (self.call_bindings is not None and self.call_bindings.get("range", builtins.range) is not builtins.range)):
            return None
        count = _constant_int(call.args[0])
        if count is None or not 0 < count <= 0x7fffffff:
            return None
        return count

    def _local_scan(self, node, loop, loop_id, reads, effects, statements):
        """A bounded pure scalar fold stays indivisible within one output row."""
        count = self._local_loop_count(node, loop)
        if count is None:
            return False
        assigned = set()
        for item in node.body:
            if (not isinstance(item, ast.Assign) or len(item.targets) != 1
                    or not isinstance(item.targets[0], ast.Name) or not self._expr_type(item.value).is_scalar):
                return False
            name = item.targets[0].id
            if name in self.arg_types or name in (loop.target.id, node.target.id):
                return False
            if any((isinstance(part, ast.Call) and not self._is_pure_call(part))
                   or isinstance(part, (ast.NamedExpr, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
                   for part in ast.walk(item.value)):
                return False
            assigned.add(name)
        # A carried value can change on every term; its old address proof is
        # invalid both inside this fold and after it. Unchanged row locals keep
        # their facts. Runtime code generation still uses the authored loop.
        for name in (*assigned, node.target.id):
            self._local_indices.pop(name, None)
        outer_indices = dict(self._local_indices)
        self._local_bounds = {node.target.id: count}
        try:
            for item in node.body:
                self._statement(item, loop, loop_id, reads, [], [], effects, statements)
                bound = _bounded_index(item.value, loop.target.id, self._local_bounds, self._local_indices)
                if bound:
                    self._local_indices[item.targets[0].id] = bound
        finally:
            self._local_bounds = {}
            self._local_indices = outer_indices
        self._local_loop_extra_cost += (count - 1) * _operation_cost(ast.Module(body=node.body, type_ignores=[]))
        return True

    def _local_reduce(self, node, loop, loop_id, reads, effects, statements):
        count = self._local_loop_count(node, loop)
        if count is None:
            return False
        update = node.body[-1]
        term_prefix = node.body[:-1]
        operation = "add"
        if isinstance(update, ast.AugAssign) and isinstance(update.op, ast.Add) and isinstance(update.target, ast.Name):
            target, value = update.target.id, update.value
        elif (isinstance(update, ast.Assign) and len(update.targets) == 1 and isinstance(update.targets[0], ast.Name)
                and isinstance(update.value, ast.BinOp) and isinstance(update.value.op, ast.Add)
                and isinstance(update.value.left, ast.Name) and update.value.left.id == update.targets[0].id):
            target, value = update.targets[0].id, update.value.right
        elif (isinstance(update, ast.Assign) and len(update.targets) == 1 and isinstance(update.targets[0], ast.Name)
                and isinstance(update.value, ast.Call) and isinstance(update.value.func, ast.Name)
                and update.value.func.id in ("max", "min") and self._is_pure_call(update.value)
                and len(update.value.args) == 2 and not update.value.keywords
                and isinstance(update.value.args[0], ast.Name)
                and update.value.args[0].id == update.targets[0].id):
            target, value, operation = update.targets[0].id, update.value.args[1], update.value.func.id
        else:
            return False
        # Per-term temporaries must be pure, single-assignment values confined
        # to this inner loop. They are recomputed on every resumed term; only
        # the reduction accumulator crosses a cut.
        temporaries = []
        for item in term_prefix:
            if (not isinstance(item, ast.Assign) or len(item.targets) != 1
                    or not isinstance(item.targets[0], ast.Name) or not self._expr_type(item.value).is_scalar):
                return False
            name = item.targets[0].id
            if name in {*self.arg_types, target, loop.target.id, node.target.id, *temporaries}:
                return False
            temporaries.append(name)
        if any(isinstance(item, ast.Name) and item.id in temporaries
               for statement in loop.body if statement is not node for item in ast.walk(statement)):
            return False
        defined = set()
        for item in term_prefix:
            if any(isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)
                   and (part.id == target or part.id in set(temporaries) - defined) for part in ast.walk(item.value)):
                return False
            defined.add(item.targets[0].id)
        initializers = [item for item in loop.body[:loop.body.index(node)] if isinstance(item, ast.Assign)
                        and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id == target]
        if (target in self.arg_types or target in {node.target.id, loop.target.id} or not initializers
                or not self._expr_type(initializers[-1].value).is_scalar
                or any(isinstance(item, ast.Name) and item.id == target for item in ast.walk(value))):
            return False
        self._local_indices.pop(target, None)
        self._local_indices.pop(node.target.id, None)
        outer_indices = dict(self._local_indices)
        self._local_bounds = {node.target.id: count}
        for item in term_prefix:
            self._statement(item, loop, loop_id, reads, [], [], effects, statements)
            bound = _bounded_index(item.value, loop.target.id, self._local_bounds, self._local_indices)
            if bound:
                self._local_indices[item.targets[0].id] = bound
        self._visit_expression(value, loop, loop_id, reads, effects)
        self._local_bounds = {}
        self._local_indices = outer_indices
        position = loop.body.index(node)
        initializer = loop.body.index(initializers[-1])
        prefix = ast.Module(body=loop.body[:position], type_ignores=[])
        epilogue = ast.Module(body=loop.body[position + 1:], type_ignores=[])
        diagnostics = []
        if operation != "add":
            diagnostics.append("non-additive local reductions continue at row boundaries")
        initial = initializers[-1].value
        if len(initializers) != 1 or not (isinstance(initial, ast.Constant) and type(initial.value) is float and initial.value == 0.0):
            diagnostics.append("inner continuation requires one dominating floating zero initializer")
        if len([item for item in ast.walk(loop) if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store) and item.id == target]) != 2:
            diagnostics.append("inner continuation cannot carry other accumulator assignments")
        if any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == target
               for item in ast.walk(prefix)):
            diagnostics.append("inner continuation prefix cannot depend on the evolving accumulator")
        if any(not isinstance(item, ast.Assign) or len(item.targets) != 1
               or not isinstance(item.targets[0], ast.Name) or not self._expr_type(item.value).is_scalar
               or item.targets[0].id in {loop.target.id, node.target.id} for item in prefix.body):
            diagnostics.append("inner continuation prefix requires pure scalar assignments")
        if any(not isinstance(item, ast.Assign) or len(item.targets) != 1 for item in epilogue.body):
            diagnostics.append("inner continuation epilogue requires unconditional assignments")
        term = ast.Module(body=node.body, type_ignores=[])
        if any((isinstance(item, ast.Call) and not self._is_pure_call(item))
               or isinstance(item, (ast.NamedExpr, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
               for part in (prefix, term, epilogue) for item in ast.walk(part)):
            diagnostics.append("inner continuation requires directly proved pure arithmetic")
        self._local_reductions.append(OrderedReductionRegion(
            node.target.id, count, target, initializer, position,
            ast.dump(ast.Module(body=loop.body, type_ignores=[]), include_attributes=False),
            _operation_cost(prefix), _operation_cost(term), _operation_cost(epilogue),
            _operation_cost(loop) - _operation_cost(prefix) - _operation_cost(term) - _operation_cost(epilogue),
            sum(isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load) for item in ast.walk(term)),
            sum(isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load) for item in ast.walk(prefix)),
            sum(isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load) for item in ast.walk(epilogue)),
            tuple(diagnostics)))
        statements.append(HIRStatement(HIRStatementKind.AUG_ASSIGN, target, _unparse(value), self._expr_type(value), _location(node)))
        self._local_loop_extra_cost += (count - 1) * _operation_cost(ast.Module(body=node.body, type_ignores=[]))
        return True

    def _inspect_function_controls(self) -> None:
        if isinstance(self.function, ast.AsyncFunctionDef):
            self.diagnostic(DiagnosticCode.ASYNC_FUNCTION, "async functions cannot be lowered to this synchronous HIR", self.function)

    @staticmethod
    def _scalar_flow(
        nodes: Sequence[ast.stmt],
        defined: set,
        on_read: Callable[[ast.Name, set], None],
        *, on_write: Callable[[ast.expr, set], None] | None = None,
        on_exit: Callable[[ast.stmt, set], None] | None = None,
    ) -> set | None:
        """Visit structured paths with a must-defined set, without another IR.

        A terminating branch does not reach its sibling's merge. Unsupported
        compound statements conservatively expose reads and kill no names.
        """
        defined = set(defined)

        def read(expression):
            for child in ast.walk(expression):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    on_read(child, defined)

        for node in nodes:
            if isinstance(node, ast.If):
                read(node.test)
                branches = [
                    _Analyzer._scalar_flow(branch, defined, on_read, on_write=on_write, on_exit=on_exit)
                    for branch in (node.body, node.orelse)
                ]
                reachable = [branch for branch in branches if branch is not None]
                if not reachable:
                    return None
                defined = set.intersection(*reachable)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value is not None:
                    read(node.value)
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        read(target)
                        defined.update(child.id for child in ast.walk(target)
                                       if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store))
                        if on_write is not None:
                            on_write(target, defined)
            elif isinstance(node, ast.AugAssign):
                read(node.value)
                read(node.target)
                if isinstance(node.target, ast.Name):
                    on_read(node.target, defined)
                    defined.add(node.target.id)
                if on_write is not None:
                    on_write(node.target, defined)
            elif (isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse
                    and isinstance(node.iter, ast.Call) and len(node.iter.args) == 1
                    and (_constant_int(node.iter.args[0]) or 0) > 0):
                read(node.iter)
                # Its exits belong to the local loop, not the logical row.
                inner = _Analyzer._scalar_flow(node.body, defined | {node.target.id}, on_read,
                                               on_write=on_write)
                if inner is not None:
                    defined.update(inner)
            else:
                read(node)
                if isinstance(node, (ast.Continue, ast.Break, ast.Return, ast.Raise)):
                    if on_exit is not None:
                        on_exit(node, defined)
                    return None
        return defined

    def _row_output_coverage(self, loop, writes):
        """Must-write facts at normal row exits, using the existing path walk."""
        index = AffineExpr.variable(loop.target.id)
        direct = {access.buffer for access in writes if access.index == index and len(access.axes) == 1}
        tag, exits = object(), []

        def output_facts(defined):
            return {value[1] for value in defined if isinstance(value, tuple)
                    and len(value) == 2 and value[0] is tag}

        def write(target, defined):
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id in direct and len(_access_axes(target, self.imports)) == 1
                    and _affine(_index_node(target), known_calls=self.imports) == index):
                defined.add((tag, target.value.id))

        def row_exit(statement, defined):
            exits.append(output_facts(defined) if isinstance(statement, ast.Continue) else set())

        fallthrough = self._scalar_flow(loop.body, set(), lambda *_: None,
                                        on_write=write, on_exit=row_exit)
        if fallthrough is not None:
            exits.append(output_facts(fallthrough))
        return tuple(sorted(set.intersection(*exits))) if exits else ()

    def _scalar_dependencies(self, node: ast.For, loop_id: str, reductions: Sequence[Reduction]) -> None:
        assigned = {child.id for statement in node.body for child in ast.walk(statement)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}
        index = node.target.id if isinstance(node.target, ast.Name) else ""
        reduction_names = {reduction.target for reduction in reductions}
        carried_candidates = assigned - reduction_names
        reported = set()

        def reject(name, location, reason):
            if name not in reported:
                reported.add(name)
                self.diagnostic(DiagnosticCode.LOOP_CARRIED_SCALAR, reason, location, loop_id=loop_id)

        if index in assigned:
            reject(index, node.target, f"loop induction variable '{index}' is reassigned")

        def read_iteration(child, defined):
            name = child.id
            if name in carried_candidates and name not in defined:
                reject(name, child, f"scalar '{name}' may read a previous iteration before being defined")

        self._scalar_flow(node.body, {index}, read_iteration)

        # A non-reduction scalar's last sequential value cannot be obtained
        # from unordered worker-private assignments. A definite overwrite
        # before a later read makes that loop output dead and remains legal.
        live_out_candidates = (assigned | {index}) - reduction_names

        def read_after(child, defined):
            if child.id in live_out_candidates and child.id not in defined:
                reject(child.id, child, f"scalar '{child.id}' is read after the loop")

        position = self.function.body.index(node)
        self._scalar_flow([*node.orelse, *self.function.body[position + 1:]], set(), read_after)

    def _loop(self, node: ast.For, ordinal: int) -> RangeLoopHIR:
        self._local_bounds = {}
        self._local_indices = {}
        self._local_reductions = []
        self._local_loop_extra_cost = 0
        loop_id = self._loop_id(node, ordinal)
        range_spec = self._range_spec(node, loop_id)
        index_name = node.target.id if isinstance(node.target, ast.Name) else "<non-name>"
        if not isinstance(node.target, ast.Name):
            self.diagnostic(DiagnosticCode.UNSUPPORTED_RANGE, "range induction target must be a single name", node.target, loop_id=loop_id)
        reads: list[BufferAccess] = []
        writes: list[BufferAccess] = []
        reductions: list[Reduction] = []
        effects: list[Effect] = []
        statements: list[HIRStatement] = []
        if node.orelse:
            effects.append(Effect(EffectKind.CONTROL_FLOW, "loop else/finalizer",
                                  _location(node.orelse[0])))
        for statement in node.body:
            self._statement(statement, node, loop_id, reads, writes, reductions, effects, statements)
        self._scalar_dependencies(node, loop_id, reductions)

        # An access to a previous/future element of a buffer written in this
        # loop is a loop-carried dependence.  Same-element read/modify/write
        # remains safe because the write is unique for the current induction
        # value.
        self.diagnostics.extend(_buffer_dependences(reads, writes, index_name, loop_id, {}))

        # An accumulator may be returned after its loop, but consuming its
        # intermediate value inside the loop turns a reduction into a scan.
        reduction_targets = {reduction.target for reduction in reductions}
        for target in reduction_targets:
            if len({item.operator for item in reductions if item.target == target}) > 1:
                self.diagnostic(
                    DiagnosticCode.INVALID_REDUCTION,
                    f"reduction '{target}' mixes different operators",
                    node, loop_id=loop_id,
                )
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    if child.value is not None:
                        for target in targets:
                            if isinstance(target, ast.Name) and target.id in reduction_targets:
                                self.diagnostic(
                                    DiagnosticCode.INVALID_REDUCTION,
                                    f"reduction '{target.id}' is overwritten inside the loop",
                                    child, loop_id=loop_id,
                                )
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in reduction_targets:
                    self.diagnostic(
                        DiagnosticCode.REDUCTION_FEEDBACK,
                        f"reduction '{child.id}' is consumed before the parallel loop completes",
                        child, loop_id=loop_id,
                    )

        # Multiple symbolic buffers may alias at runtime.  Elementwise
        # same-index access is safe; retaining the risk in HIR lets a future
        # transformer require explicit no-alias metadata for more aggressive
        # lowerings.
        buffer_names = sorted({access.buffer for access in (*reads, *writes)})
        alias_risks: list[AliasRisk] = []
        if len(buffer_names) > 1:
            risk = AliasRisk(
                AliasRiskKind.POSSIBLE,
                tuple(buffer_names),
                "distinct Python buffer arguments may alias; only same-index elementwise access is admitted",
                _location(node),
            )
            alias_risks.append(risk)
            effects.append(Effect(EffectKind.ALIAS_RISK, risk.detail, _location(node)))
            if any(not access.same_iteration for access in (*reads, *writes)):
                self.diagnostic(
                    DiagnosticCode.ALIAS_RISK,
                    "different buffers have a non-elementwise access and may alias at runtime",
                    node,
                    loop_id=loop_id,
                )

        diagnostics = tuple(item for item in self.diagnostics if item.loop_id == loop_id)
        error_codes = {item.code for item in diagnostics if item.is_error}
        if range_spec is None or not range_spec.is_affine:
            reason = "rejected: range bounds or step are not a provable affine range"
        elif error_codes:
            first = diagnostics[0]
            reason = f"rejected: {first.message}"
        elif not writes and not reductions:
            reason = "rejected: loop has no analyzable output write or scalar reduction"
        else:
            reason = "eligible: independent affine iterations with unique writes and/or standard scalar reductions"
        eligible = not error_codes and range_spec is not None and range_spec.is_affine and bool(writes or reductions)
        source = ast.get_source_segment(self.source, node) or _unparse(node)
        operation_cost = _operation_cost(node) + self._local_loop_extra_cost
        return RangeLoopHIR(
            stable_id=loop_id,
            function_name=self.function.name,
            index_name=index_name,
            range_spec=range_spec,
            source_location=_location(node),
            source=source,
            statements=tuple(statements),
            buffer_reads=tuple(reads),
            buffer_writes=tuple(writes),
            reductions=tuple(reductions),
            effects=tuple(effects),
            alias_risks=tuple(alias_risks),
            diagnostics=diagnostics,
            parallel_eligible=eligible,
            reason=reason,
            operation_cost=max(1, operation_cost),
            local_reductions=tuple(self._local_reductions),
            fully_written_outputs=self._row_output_coverage(node, writes)
                if not (error_codes - {DiagnosticCode.ALIAS_RISK}) else (),
        )

    def _block_id(self, kind: BasicBlockKind, node: ast.AST | None, ordinal: int) -> str:
        syntax = ast.dump(node, include_attributes=False) if node is not None else kind.value
        payload = f"{self.function.name}|{kind.value}|{ordinal}|{syntax}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.function.name}:block:{digest}"

    def _build_cfg(
        self,
        loop_nodes: Sequence[ast.For],
        loop_hirs: Sequence[RangeLoopHIR],
    ) -> tuple[BasicBlock, ...]:
        """Build a deterministic structured CFG for the admitted frontend.

        R7 intentionally admits top-level affine loops only. The CFG therefore
        represents normal statements as linear blocks and each loop as a
        header/body back-edge pair. Unsupported nested control remains inside
        the body block and is already rejected by effect analysis.
        """

        entry_id = f"{self.function.name}:block:entry"
        exit_id = f"{self.function.name}:block:exit"
        loop_by_node = {
            id(node): loop_hirs[index]
            for index, node in enumerate(loop_nodes)
        }
        drafts: list[tuple[str, BasicBlockKind, str, SourceLocation | None, str | None]] = []
        ordinal = 0
        for node in self.function.body:
            if isinstance(node, ast.For):
                loop_hir = loop_by_node[id(node)]
                header_id = f"{loop_hir.stable_id}:header"
                body_id = f"{loop_hir.stable_id}:body"
                drafts.append(
                    (
                        header_id,
                        BasicBlockKind.LOOP_HEADER,
                        _unparse(node.iter),
                        _location(node),
                        loop_hir.stable_id,
                    )
                )
                drafts.append(
                    (
                        body_id,
                        BasicBlockKind.LOOP_BODY,
                        "\n".join(_unparse(statement) for statement in node.body),
                        _location(node.body[0]) if node.body else _location(node),
                        loop_hir.stable_id,
                    )
                )
                ordinal += 2
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            drafts.append(
                (
                    self._block_id(BasicBlockKind.LINEAR, node, ordinal),
                    BasicBlockKind.LINEAR,
                    _unparse(node),
                    _location(node),
                    None,
                )
            )
            ordinal += 1

        blocks: list[BasicBlock] = []
        first_id = drafts[0][0] if drafts else exit_id
        blocks.append(BasicBlock(entry_id, BasicBlockKind.ENTRY, "", None, (first_id,)))
        for index, (stable_id, kind, source, location, loop_id) in enumerate(drafts):
            next_id = drafts[index + 1][0] if index + 1 < len(drafts) else exit_id
            if kind == BasicBlockKind.LOOP_HEADER:
                body_id = drafts[index + 1][0]
                after_body = drafts[index + 2][0] if index + 2 < len(drafts) else exit_id
                successors = (body_id, after_body)
            elif kind == BasicBlockKind.LOOP_BODY:
                successors = (drafts[index - 1][0],)
            else:
                successors = (next_id,)
            blocks.append(BasicBlock(stable_id, kind, source, location, successors, loop_id))
        blocks.append(BasicBlock(exit_id, BasicBlockKind.EXIT, "", None, ()))
        return tuple(blocks)

    def build(self) -> FunctionHIR:
        self._inspect_function_controls()
        # R7 intentionally supports one-dimensional top-level affine loops.
        # Nested loops are diagnosed by the owning top-level loop and never
        # become independent lowering candidates of their own.
        loops = tuple(node for node in self.function.body if isinstance(node, ast.For))
        loop_hirs = tuple(self._loop(node, ordinal) for ordinal, node in enumerate(loops))
        function_statements: list[HIRStatement] = []
        for node in self.function.body:
            if isinstance(node, ast.For):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                function_statements.append(HIRStatement(HIRStatementKind.ASSIGN, node.targets[0].id, _unparse(node.value), self._expr_type(node.value), _location(node)))
        arguments = tuple(
            HIRArgument(argument.arg, self.arg_types.get(argument.arg, TypeRef(ValueType.UNKNOWN)), _annotation_name(argument.annotation), _location(argument))
            for argument in [*self.function.args.posonlyargs, *self.function.args.args, *self.function.args.kwonlyargs]
        )
        blocks = self._build_cfg(loops, loop_hirs)
        return FunctionHIR(
            self.function.name,
            self.source,
            arguments,
            loop_hirs,
            tuple(self.diagnostics),
            tuple(function_statements),
            blocks,
        )


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(textwrap.dedent(source), mode="exec")
    except SyntaxError as exc:
        location = SourceLocation(exc.lineno or 0, exc.offset or 0, exc.lineno or 0, exc.offset or 0)
        raise HIRParseError(f"{DiagnosticCode.PARSE_ERROR.value} at {location.line}:{location.column}: {exc.msg}") from exc


class HIRParseError(ValueError):
    """Raised only when Python source cannot be parsed into an AST."""


def _find_function(tree: ast.Module, function_name: str | None) -> ast.FunctionDef | ast.AsyncFunctionDef:
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_name is not None:
        for function in functions:
            if function.name == function_name:
                return function
        raise LookupError(f"{DiagnosticCode.FUNCTION_NOT_FOUND.value}: {function_name}")
    if not functions:
        raise LookupError(f"{DiagnosticCode.FUNCTION_NOT_FOUND.value}: no top-level function")
    return functions[0]


def build_hir(source: str, function_name: str | None = None, *,
              call_bindings: Mapping[str, object] | None = None,
              argument_types: Mapping[str, TypeRef] | None = None) -> FunctionHIR:
    """Parse source and build the minimal typed HIR for one function."""

    tree = _parse(source)
    function = _find_function(tree, function_name)
    return _Analyzer(tree, textwrap.dedent(source), function, call_bindings, argument_types).build()


def analyze_source(source: str, function_name: str | None = None, *,
                   call_bindings: Mapping[str, object] | None = None,
                   argument_types: Mapping[str, TypeRef] | None = None) -> FunctionHIR:
    """Alias for :func:`build_hir`, named for analysis-oriented callers."""

    return build_hir(source, function_name, call_bindings=call_bindings, argument_types=argument_types)


def analyze_function(function: Any, *, function_name: str | None = None) -> FunctionHIR:
    """Build HIR from a Python function or source string.

    ``inspect.getsource`` is used only to obtain source; the function is never
    executed.  Passing a source string is useful for generated kernels and
    keeps tests independent from inspect's file lookup rules.
    """

    if isinstance(function, str):
        return build_hir(function, function_name)
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise TypeError("analyze_function needs a source string or an inspectable Python function") from exc
    return build_hir(source, function_name or getattr(function, "__name__", None))


def eligible_loops(function_or_source: Any, *, function_name: str | None = None) -> tuple[RangeLoopHIR, ...]:
    """Return only loops that the conservative analyzer can parallelize."""

    return analyze_function(function_or_source, function_name=function_name).eligible_loops


def hir_fingerprint(hir: FunctionHIR) -> str:
    """Return a stable identity for the normalized legality result."""

    payload = {
        "revision": 1,
        "name": hir.name,
        "source_ast": ast.dump(_parse(hir.source), include_attributes=False),
        "arguments": [
            (argument.name, argument.type_ref.value_type.value, argument.type_ref.shape)
            for argument in hir.arguments
        ],
        "loops": [
            {
                "id": loop.stable_id,
                "index": loop.index_name,
                "range": (
                    loop.range_spec.start_affine.format() if loop.range_spec and loop.range_spec.start_affine else None,
                    loop.range_spec.stop_affine.format() if loop.range_spec and loop.range_spec.stop_affine else None,
                    loop.range_spec.step_value if loop.range_spec else None,
                ),
                "reads": [
                    (item.buffer, item.index.format() if item.index else None, item.kind.value)
                    for item in loop.buffer_reads
                ],
                "writes": [
                    (item.buffer, item.index.format() if item.index else None, item.kind.value)
                    for item in loop.buffer_writes
                ],
                "reductions": [
                    (item.target, item.operator, item.identity)
                    for item in loop.reductions
                ],
                "eligible": loop.parallel_eligible,
                "operation_cost": loop.operation_cost,
                "diagnostics": [item.code.value for item in loop.diagnostics],
            }
            for loop in hir.loops
        ],
        "cfg": [
            (block.stable_id, block.kind.value, block.successors, block.loop_id)
            for block in hir.blocks
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AffineExpr",
    "AliasRisk",
    "AliasRiskKind",
    "BufferAccess",
    "BufferAccessKind",
    "BufferReadWrite",
    "BasicBlock",
    "BasicBlockKind",
    "Diagnostic",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "ContinuationPlan",
    "InnerRangeContract",
    "OrderedReductionRegion",
    "Effect",
    "EffectKind",
    "FunctionHIR",
    "HIRArgument",
    "HIRFunction",
    "HIRParseError",
    "HIRRangeLoop",
    "HIRStatement",
    "HIRStatementKind",
    "RangeLoopHIR",
    "RangeSpec",
    "Reduction",
    "ReductionSpec",
    "SourceLocation",
    "TypeRef",
    "ValueType",
    "analyze_function",
    "analyze_buffer_aliases",
    "analyze_source",
    "build_hir",
    "eligible_loops",
    "hir_fingerprint",
    "parallel_alias_pairs",
]
