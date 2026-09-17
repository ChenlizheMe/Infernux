"""Runtime support for Infernux JIT compilation and adaptive dispatch.

The frontend owns legality while this module owns stable identities, bounded
process caches, runtime signature buckets, and side-effect-safe comparison.
It deliberately has no dependency on Numba so it is also usable by build
tools and by the pure-Python fallback.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ast
import dis
import hashlib
import inspect
import json
import os
import platform
import sys
from typing import Any, Generic, Iterator, MutableMapping, TypeVar


_K = TypeVar("_K")
_V = TypeVar("_V")

class BoundedLRU(Generic[_K, _V]):
    """Small LRU used for compiled dispatchers and per-signature decisions."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("LRU capacity must be positive")
        self.capacity = int(capacity)
        self._values: OrderedDict[_K, _V] = OrderedDict()

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def get(self, key: _K, default: Any = None) -> _V | Any:
        try:
            value = self._values.pop(key)
        except KeyError:
            return default
        self._values[key] = value
        return value

    def __getitem__(self, key: _K) -> _V:
        missing = object()
        value = self.get(key, missing)
        if value is missing:
            raise KeyError(key)
        return value

    def __setitem__(self, key: _K, value: _V) -> None:
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def clear(self) -> None:
        self._values.clear()

    def items(self) -> Iterator[tuple[_K, _V]]:
        return iter(tuple(self._values.items()))


def _stable_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return {"type": type(value).__qualname__, "truncated": True}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value:
            return {"float": "nan"}
        if value == float("inf"):
            return {"float": "+inf"}
        if value == float("-inf"):
            return {"float": "-inf"}
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.ndarray):
        # Numba embeds captured arrays as constants. This is a compile-boundary
        # identity, never a hash of invocation arguments or a per-frame check.
        return {
            "array_dtype": value.dtype.descr,
            "shape": value.shape,
            "strides": value.strides,
            "data": _stable_value(value.tobytes(order="A"), depth=depth + 1),
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item, depth=depth + 1) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=repr))
    if isinstance(value, dict):
        items = [
            (
                _stable_value(key, depth=depth + 1),
                _stable_value(item, depth=depth + 1),
            )
            for key, item in value.items()
        ]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, default=repr))
        return {"mapping": items}
    code = getattr(value, "__code__", None)
    if code is not None:
        return {
            "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}",
            "code": hashlib.sha256(code.co_code).hexdigest(),
            "consts": _stable_value(code.co_consts, depth=depth + 1),
            "names": tuple(code.co_names),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _compiler_environment(fn: Any) -> list[dict[str, Any]]:
    """Describe authored helper dependencies once, including recursive graphs."""
    functions = [fn]
    indices = {id(fn): 0}
    nodes = []

    def capture(value):
        if inspect.isfunction(value):
            identity = id(value)
            if identity not in indices:
                indices[identity] = len(functions)
                functions.append(value)
            return {"function_ref": indices[identity]}
        return _stable_value(value)

    for function in functions:
        code = getattr(function, "__code__", None)
        closure = {}
        cells = {}
        if code is not None:
            for name, cell in zip(code.co_freevars, function.__closure__ or ()):
                try:
                    value = cell.cell_contents
                except ValueError:
                    closure[name] = {"empty_cell": True}
                else:
                    cells[name] = value
                    closure[name] = capture(value)
        globals_map = getattr(function, "__globals__", {})
        # Module identity alone misses e.g. settings.GRAVITY. Track only static
        # attribute chains the function reads, without walking entire modules
        # or invoking user-defined attribute hooks during publication.
        attributes = {}
        value = None
        path = ""
        if code is not None:
            for instruction in dis.get_instructions(code):
                if instruction.opname == "LOAD_GLOBAL":
                    path = instruction.argval
                    value = globals_map.get(path)
                elif instruction.opname == "LOAD_DEREF":
                    path = instruction.argval
                    value = cells.get(path)
                elif instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"} and inspect.ismodule(value):
                    path += "." + instruction.argval
                    value = vars(value).get(instruction.argval)
                    attributes[path] = capture(value)
                else:
                    value = None
        nodes.append({
            "function": _stable_value(function),
            "defaults": _stable_value(getattr(function, "__defaults__", None)),
            "kwdefaults": _stable_value(getattr(function, "__kwdefaults__", None)),
            "closure": closure,
            "module_attributes": attributes,
            "globals": {
                name: capture(globals_map[name])
                for name in (code.co_names if code is not None else ())
                if name in globals_map
            },
        })
    return nodes


def compiler_fingerprint(fn: Any, options: MutableMapping[str, Any] | None = None) -> str:
    """Return a stable compiler/cache identity for a Python kernel."""

    code = getattr(fn, "__code__", None)
    source_ast: str | None = None
    try:
        source = inspect.getsource(fn)
        source_ast = ast.dump(ast.parse(inspect.cleandoc(source)), include_attributes=False)
    except (OSError, TypeError, IndentationError, SyntaxError):
        source_ast = None

    runtime_versions: dict[str, Any] = {
        "python": tuple(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME"),
        "numba_cpu_features": os.environ.get("NUMBA_CPU_FEATURES"),
        "numba_threading_layer_env": os.environ.get("NUMBA_THREADING_LAYER"),
    }
    for module_name in ("numpy", "numba", "llvmlite"):
        try:
            module = __import__(module_name)
            runtime_versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            runtime_versions[module_name] = None
    try:
        import numba

        runtime_versions["numba_threading_layer"] = numba.threading_layer()
    except Exception:
        runtime_versions["numba_threading_layer"] = "uninitialized"

    payload = {
        # Revision 3 makes recursive cross-specialization dependencies explicit
        # in each independently owned code library, including cached objects.
        "compiler_revision": 3,
        "module": getattr(fn, "__module__", ""),
        "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", "")),
        "source_ast": source_ast,
        "bytecode": hashlib.sha256(code.co_code).hexdigest() if code is not None else None,
        "constants": _stable_value(code.co_consts if code is not None else ()),
        "names": tuple(code.co_names) if code is not None else (),
        "defaults": _stable_value(getattr(fn, "__defaults__", None)),
        "kwdefaults": _stable_value(getattr(fn, "__kwdefaults__", None)),
        "environment": _compiler_environment(fn),
        "annotations": _stable_value(getattr(fn, "__annotations__", {})),
        "options": _stable_value(dict(options or {})),
        "runtime": runtime_versions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trip_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    for upper in (31, 127, 511, 2047, 8191, 65535):
        if value <= upper:
            return f"<={upper}"
    return ">=65536"


def _argument_signature(value: Any) -> tuple[Any, ...]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    ndim = getattr(value, "ndim", None)
    flags = getattr(value, "flags", None)
    if shape is not None and dtype is not None and ndim is not None:
        try:
            shape_tuple = tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            shape_tuple = ()
        c_contiguous = bool(getattr(flags, "c_contiguous", False)) if flags is not None else False
        f_contiguous = bool(getattr(flags, "f_contiguous", False)) if flags is not None else False
        leading = shape_tuple[0] if shape_tuple else 0
        return (
            "array",
            str(dtype),
            int(ndim),
            "C" if c_contiguous else "F" if f_contiguous else "strided",
            _trip_bucket(leading),
            tuple(_trip_bucket(item) for item in shape_tuple[1:]),
        )
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, int):
        return ("int", _trip_bucket(abs(value)))
    if isinstance(value, float):
        return ("float",)
    return ("object", type(value).__module__, type(value).__qualname__)


def runtime_signature(args: tuple[Any, ...], kwargs: dict[str, Any], *, thread_count: int) -> tuple[Any, ...]:
    return (
        tuple(_argument_signature(value) for value in args),
        tuple((name, _argument_signature(value)) for name, value in sorted(kwargs.items())),
        max(1, int(thread_count)),
        platform.machine(),
    )


@dataclass(frozen=True)
class StaticCostDecision:
    mode: str
    confidence: str
    work_units: int
    reason: str


def static_cost_decision(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    operation_cost: int,
    thread_count: int,
) -> StaticCostDecision:
    """Estimate one-shot kernel value before any timing samples are taken.

    The estimate is intentionally coarse and conservative. The leading array
    extent models loop trips while trailing extents model vector lanes touched
    by ``buffer[i]``. Integer-only kernels use the largest non-negative integer
    argument as their trip estimate.
    """

    trip_count = 0
    lane_count = 1
    values = (*args, *kwargs.values())
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        try:
            dimensions = tuple(max(0, int(item)) for item in shape)
        except (TypeError, ValueError):
            continue
        if not dimensions:
            continue
        candidate_trip = dimensions[0]
        candidate_lanes = 1
        for dimension in dimensions[1:]:
            candidate_lanes *= max(1, dimension)
        if candidate_trip * candidate_lanes > trip_count * lane_count:
            trip_count = candidate_trip
            lane_count = candidate_lanes
    if trip_count == 0:
        integers = [
            int(value)
            for value in values
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        trip_count = max(integers, default=0)

    if trip_count == 0:
        return StaticCostDecision(
            "serial",
            "gray",
            0,
            "static HIR cost has no runtime trip-count evidence; serial until measured",
        )

    work_units = max(0, trip_count) * max(1, lane_count) * max(1, int(operation_cost))
    threads = max(1, int(thread_count))
    serial_limit = 250_000
    parallel_limit = max(4_000_000, threads * 250_000)
    if threads <= 1 or work_units <= serial_limit:
        mode = "serial"
        confidence = "high"
        reason = f"static HIR cost selected serial ({work_units} work units, {threads} thread(s))"
    elif work_units >= parallel_limit:
        mode = "parallel"
        confidence = "high"
        reason = f"static HIR cost selected parallel ({work_units} work units, {threads} threads)"
    else:
        mode = "serial"
        confidence = "gray"
        reason = f"static HIR cost is inconclusive ({work_units} work units); serial until measured"
    return StaticCostDecision(mode, confidence, work_units, reason)


def array_arguments_alias(args: tuple[Any, ...], kwargs: dict[str, Any], *,
                          parameter_names=(), safe_pairs=frozenset()) -> bool:
    """Whether storage overlap invalidates the publication's parallel proof.

    This inspects layout, never array contents. Equal-layout aliases may use
    HIR-proven parameter pairs; shifted, reinterpreted or internally overlapping
    views cannot use that proof. Ordinary strided and reversed arrays are valid
    when each logical element occupies distinct bytes.
    """
    import numpy as np

    arrays = [(parameter_names[index] if index < len(parameter_names) else index, value)
              for index, value in enumerate(args) if isinstance(value, np.ndarray)]
    arrays.extend((name, value) for name, value in kwargs.items() if isinstance(value, np.ndarray))
    for _name, value in arrays:
        if value.size == 0 or value.flags.c_contiguous or value.flags.f_contiguous:
            continue
        span = value.itemsize
        for stride, extent in sorted((abs(stride), extent)
                                     for stride, extent in zip(value.strides, value.shape) if extent > 1):
            if stride < span:
                return True
            span += stride * (extent - 1)
    for index, (left_name, left) in enumerate(arrays):
        for right_name, right in arrays[index + 1:]:
            if not np.may_share_memory(left, right):
                continue
            if ((left_name, right_name) not in safe_pairs or left.dtype != right.dtype
                    or left.shape != right.shape or left.strides != right.strides
                    or left.ctypes.data != right.ctypes.data):
                return True
    return False


def clone_call_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Isolate preparation data, preserving shared identities and ndarray views."""
    import numpy as np

    memo: dict[int, Any] = {}
    array_roots: dict[int, Any] = {}

    def clone(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
            return value
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("cannot isolate object-array contents for compute preparation")
            if not value.size:
                # Empty reversed views may have an offset outside their empty
                # owner. There are no bytes to alias; retain the exact layout.
                copied = np.ndarray(value.shape, dtype=value.dtype, buffer=bytearray(), strides=value.strides)
                memo[identity] = copied
                return copied
            root = value
            owner = value
            # NumPy stride-trick views insert a non-ndarray owner between the
            # view and its allocation. Preserve their strides and aliases too.
            while (owner := getattr(owner, "base", None)) is not None:
                if isinstance(owner, np.ndarray):
                    root = owner
            if id(root) not in array_roots:
                if any(np.may_share_memory(root, other) for other in array_roots.values()):
                    raise TypeError("cannot isolate overlapping arrays with independent storage owners")
                array_roots[id(root)] = root
            if not (root.flags.c_contiguous or root.flags.f_contiguous):
                raise TypeError("cannot isolate ndarray views of a non-contiguous storage owner")
            if root is value:
                copied = value.copy(order="K")
            else:
                copied = np.ndarray(
                    value.shape, dtype=value.dtype, buffer=clone(root),
                    offset=value.ctypes.data - root.ctypes.data, strides=value.strides,
                )
            copied.flags.writeable = value.flags.writeable
            memo[identity] = copied
            return copied
        if isinstance(value, tuple):
            copied = tuple(clone(item) for item in value)
            memo[identity] = copied
            return copied
        if isinstance(value, list):
            copied = []
            memo[identity] = copied
            copied.extend(clone(item) for item in value)
            return copied
        if isinstance(value, dict):
            copied = {}
            memo[identity] = copied
            copied.update((clone(key), clone(item)) for key, item in value.items())
            return copied
        copier = getattr(value, "copy", None)
        if callable(copier):
            copied = copier()
            if copied is value:
                raise TypeError(f"cannot isolate mutable argument {type(value).__qualname__}")
            memo[identity] = copied
            return copied
        raise TypeError(f"cannot isolate argument {type(value).__qualname__}")

    try:
        return tuple(clone(value) for value in args), {name: clone(value) for name, value in kwargs.items()}
    finally:
        # The recursive closure can await cyclic GC. It must not keep original
        # sample arrays or returned clones alive past this preparation call.
        array_roots.clear()
        memo.clear()


def values_equivalent(left: Any, right: Any, *, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        if left != left and right != right:
            return True
        return abs(left - right) <= atol + rtol * abs(right)
    if left is None or isinstance(left, (bool, int, complex, str, bytes)):
        return left == right
    if isinstance(left, tuple):
        return len(left) == len(right) and all(values_equivalent(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right))
    if isinstance(left, list):
        return len(left) == len(right) and all(values_equivalent(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            values_equivalent(left[key], right[key], rtol=rtol, atol=atol) for key in left
        )

    shape = getattr(left, "shape", None)
    dtype = getattr(left, "dtype", None)
    if shape is not None and dtype is not None:
        if tuple(shape) != tuple(getattr(right, "shape", ())) or str(dtype) != str(getattr(right, "dtype", "")):
            return False
        try:
            import numpy as np

            if np.issubdtype(dtype, np.inexact):
                return bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    try:
        result = left == right
        return bool(result)
    except Exception:
        return False


def calls_equivalent(
    left_result: Any,
    left_args: tuple[Any, ...],
    left_kwargs: dict[str, Any],
    right_result: Any,
    right_args: tuple[Any, ...],
    right_kwargs: dict[str, Any],
) -> bool:
    return (
        values_equivalent(left_result, right_result)
        and values_equivalent(left_args, right_args)
        and values_equivalent(left_kwargs, right_kwargs)
    )


@dataclass(frozen=True)
class DispatchDecision:
    mode: str
    reason: str
    serial_seconds: float | None = None
    parallel_seconds: float | None = None
    samples: int = 0


@dataclass(frozen=True, slots=True)
class CpuPassTiming:
    pipeline: str
    name: str
    total_ms: float


@dataclass(frozen=True, slots=True)
class CpuSpecializationStatistics:
    implementation: str
    signature: str
    preparation_ms: float
    preparation_succeeded: bool
    cache_hit: bool
    optimization_level: str
    object_mode: bool
    pipeline_timings: tuple[CpuPassTiming, ...]
    mapped_bytes: int | None
    peak_mapped_bytes: int | None


@dataclass(frozen=True, slots=True)
class CpuCompilationStatistics:
    function_name: str
    selected_mode: str
    last_diagnostic: str
    decisions: tuple[tuple[str, DispatchDecision], ...]
    specializations: tuple[CpuSpecializationStatistics, ...]
    specialization_limit_per_implementation: int
    memory_statistics_available: bool
    owned_engine_count: int
    reachable_engine_count: int
    owned_mapped_bytes: int | None
    reachable_mapped_bytes: int | None


__all__ = [
    "BoundedLRU",
    "DispatchDecision",
    "StaticCostDecision",
    "calls_equivalent",
    "clone_call_arguments",
    "compiler_fingerprint",
    "runtime_signature",
    "static_cost_decision",
    "values_equivalent",
]
