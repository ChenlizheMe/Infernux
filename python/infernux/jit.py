"""Engine-owned CPU compilation and warmup.

Use ``@inx.jit.compile`` for ordinary CPU functions. The compiler chooses a
legal serial or parallel implementation from the existing typed-HIR analysis;
``warmup`` prepares the selected implementation on isolated inputs before
gameplay, and decisions and diagnostics remain queryable.

GPU kernels are a separate ``inx.compute.kernel`` execution model and never
route through Numba CUDA. CPU compilation does not install, repair, or select a
GPU backend at runtime.

Numba remains an internal CPU code-generation backend; its decorator and
``prange`` are not a second public authoring surface.
"""

from infernux.cpu_execution import _CpuArguments, CpuRangeProgram

from infernux._compiler.kernel_contract import WORK_ITEM_FAST_MATH, CPU_WORK_ITEM_ERROR_MODEL
import os
import sys

from infernux._jit_kernels import (
    JIT_AVAILABLE as _BACKEND_JIT_AVAILABLE,
    njit as _njit,
    warmup as _native_warmup,
)
from infernux.jit_runtime import (
    CpuCompilationStatistics as Statistics,
    CpuOptimizationReport as OptimizationReport,
    CpuPassTiming as PassTiming,
    CpuSpecializationStatistics as SpecializationStatistics,
)

def _is_web_runtime() -> bool:
    return os.environ.get("INFERNUX_WEB_RUNTIME") == "1" or sys.platform == "emscripten"

def _jit_runtime_available() -> bool:
    """Return whether this process may execute the native CPU JIT."""
    # A host Python can have Numba installed while assembling a Web Player.
    # The target capability, not merely the host backend, controls execution.
    return _BACKEND_JIT_AVAILABLE and not _is_web_runtime()

# Describe the active Player. A host Python used while assembling Web content
# may have Numba installed, but the Web Player deliberately does not.
JIT_AVAILABLE = _jit_runtime_available()

class _CompiledCpuFunction(_CpuArguments):
    """Thin public adapter from engine CPU buffers to native JIT storage."""

    def __init__(self, compiled):
        self._compiled = compiled
        self.__name__ = getattr(compiled, "__name__", type(compiled).__name__)
        self.__qualname__ = getattr(compiled, "__qualname__", self.__name__)
        self.__doc__ = getattr(compiled, "__doc__", None)
        self.__module__ = getattr(compiled, "__module__", __name__)

    @staticmethod
    def _restore(value, owners):
        import numpy as np

        owner = owners.get(id(value))
        if owner is not None:
            return owner
        if isinstance(value, np.void):
            vector = owners.get(("vector_type", value.dtype.str))
            if vector is not None:
                vector_type, fields = vector
                return vector_type(*(float(value[name]) for name in fields))
        if isinstance(value, tuple):
            return tuple(_CompiledCpuFunction._restore(item, owners) for item in value)
        if value is None or isinstance(value, (bool, int, float, complex, np.number, np.ndarray)):
            if isinstance(value, np.ndarray) and value.dtype.hasobject:
                raise TypeError("inx.jit.compile cannot return a NumPy object array")
            return value
        raise TypeError(
            "inx.jit.compile must return a numeric scalar, NumPy array, CPU inx.buffer, "
            "Infernux vector, tuple of supported values, or None"
        )

    @staticmethod
    def _commit_vectors(owners):
        for record, original, _vector_type, fields in owners.get("vectors", ()):
            for name in fields:
                setattr(original, name, float(record[name]))

    def __call__(self, *args, **kwargs):
        native_args, native_kwargs, owners = self._arguments(args, kwargs)
        try:
            result = self._compiled(*native_args, **native_kwargs)
        finally:
            # As with ndarray writes, an authored exception does not roll back
            # writes already performed by native code. Never replay the call.
            self._commit_vectors(owners)
        return self._restore(result, owners)

    def _infernux_warmup(self, *args, **kwargs):
        native_args, native_kwargs, _ = self._arguments(args, kwargs)
        prepare = getattr(self._compiled, "_infernux_warmup", None)
        if prepare is not None:
            return prepare(*native_args, **native_kwargs)
        from infernux.jit_runtime import clone_call_arguments
        prepared_args, prepared_kwargs = clone_call_arguments(native_args, native_kwargs)
        self._compiled(*prepared_args, **prepared_kwargs)

    def __getattr__(self, name):
        return getattr(self._compiled, name)

def compile(fn=None, **options):
    """Compile a CPU function with Infernux's serial/parallel analysis.

    ``auto_parallel`` defaults to true. Callers may set it explicitly when a
    strictly serial native function is required. The returned dispatcher keeps
    the existing diagnostics, warmup, bounded specialization, and no-replay
    execution rules.

    Referenced globals and closures are compile-time values for a publication;
    publish a new function revision to change them. ``cache=True`` also keys
    disk entries by those dependencies, not just the source file timestamp.
    Editor artifacts live under ``Library/Artifacts/Compute/CPU``; Players use
    their writable application data root. Standalone compiler tools requesting
    disk caching must supply an explicit ``NUMBA_CACHE_DIR``.

    Each compiled implementation admits up to 64 input-type specializations.
    Array lengths do not create new specializations. Beyond capacity, new
    signatures are rejected before user code runs; existing ones remain valid.

    Automatic parallel selection also respects actual array storage. Shared
    parameters need a proven independent access pattern and identical layouts;
    offset aliases or overlapping elements select serial before execution.
    ``parallel_policy="required"`` rejects layouts without that proof. No
    partially executed call is replayed through another implementation.
    """
    if not _jit_runtime_available():
        raise RuntimeError(
            "inx.jit.compile requires the bundled Numba/llvmlite CPU JIT runtime"
        )
    options.setdefault("auto_parallel", True)
    if fn is None:
        def decorate(function):
            return _CompiledCpuFunction(_njit(**options)(function))
        return decorate
    if not callable(fn):
        raise TypeError("inx.jit.compile expects a callable")
    return _CompiledCpuFunction(_njit(fn, **options))

class _CompiledCpuRange(CpuRangeProgram, _CompiledCpuFunction):
    """One prepared native row function; ownership and scheduling stay with its caller."""

    def __init__(self, compiled, parameter_names, parameter_kinds, domain_parameter, plan):
        _CompiledCpuFunction.__init__(self, compiled)
        CpuRangeProgram.__init__(self, parameter_names, parameter_kinds, domain_parameter, plan)
        self._c_entry = None

    def _native_entry(self):
        """Retain a C wrapper for this exact, already-proved publication."""
        if self._c_entry is None:
            from infernux._jit_backend import compile_range_entry
            self._c_entry = compile_range_entry(self._compiled, self.parameter_kinds, self.continuation_plan)
        return self._c_entry

    def __call__(self, *params, begin, end):
        import numpy as np

        values, owners, _, begin, end = self._range_arguments(params, begin, end)
        if begin != end:
            self._compiled(*values, np.int32(begin), np.int32(end))

    def _infernux_warmup(self, *params):
        import numpy as np

        values, owners, _ = self._bind(params)
        # The proven function contains only this loop. An empty native call
        # specializes exact readonly/layout types without reading/writing rows.
        self._compiled(*values, np.int32(0), np.int32(0))

def _prepare_cpu_work_item(function, params, *, receiver_fields=None):
    import numpy as np
    from infernux._compiler.kernel_contract import ARRAY_KINDS
    from infernux._compiler.work_item import normalize_work_item
    from infernux.jit_hir import TypeRef, ValueType

    if not _jit_runtime_available():
        raise RuntimeError("native CPU work-item continuation is unavailable on this target")
    item = normalize_work_item(function, len(params), receiver_fields=receiver_fields, target="CPU/Native")
    names = tuple(argument.arg for argument in (*item.definition.args.posonlyargs, *item.definition.args.args))
    native_params, _, owners = _CompiledCpuFunction._arguments(tuple(params), {})
    array_parameters = tuple(index for index, value in enumerate(native_params) if isinstance(value, np.ndarray))
    if any(str(native_params[index].dtype) not in ARRAY_KINDS for index in array_parameters):
        raise TypeError("CPU work-item arrays require float32, int32, uint32 or int64 storage")
    kinds = tuple(ARRAY_KINDS[str(native_params[index].dtype)] if index in array_parameters else
                  "int32" if isinstance(native_params[index], (int, np.integer)) else "float32"
                  for index in range(len(names)))
    types = {name: TypeRef(ValueType.ARRAY, ("domain",), True) if index in array_parameters
             else TypeRef(ValueType.INT if kinds[index] == "int32" else ValueType.FLOAT)
             for index, name in enumerate(names)}
    hir = item.analyze(argument_types=types)
    plan = hir.continuation_plan
    plan.require()
    return item, native_params, hir, plan, kinds

def _compile_work_item(function, params, *, receiver_fields=None):
    """One author body, HIR proof and typed native [begin,end) entry."""
    return _compile_cpu_work_item(function, *_prepare_cpu_work_item(function, params, receiver_fields=receiver_fields))

def _compile_cpu_work_item(function, item, params, hir, plan, parameter_kinds):
    import ast
    import builtins
    import hashlib
    import json
    import numpy as np

    names = tuple(argument.arg for argument in (*item.definition.args.posonlyargs, *item.definition.args.args))
    hidden = ("_infernux_range_begin", "_infernux_range_end", "_infernux_range_f32", "_infernux_range_iter")
    used = {node.id for node in ast.walk(item.definition) if isinstance(node, ast.Name)} | set(item.globals_map) | set(names)
    if any(name in used for name in hidden):
        raise TypeError("work item source uses a reserved _infernux_range_ name")

    class Float32Constants(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "float":
                node.func.id = hidden[2]
            return node

        def visit_Constant(self, node):
            if type(node.value) is float:
                return ast.copy_location(ast.Call(func=ast.Name(id=hidden[2], ctx=ast.Load()),
                                                  args=[node], keywords=[]), node)
            return node

    definition = item.loop_definition(bounds=hidden[:2])
    definition.body[0].iter.func.id = hidden[3]
    # A C entry freezes its dispatcher. Distinct typed ABIs must not share that
    # dispatcher even when the authored row body (for example a copy) is equal.
    definition.name = "_infernux_range_" + item.definition.name + "__" + "_".join(parameter_kinds)
    definition.args.posonlyargs = []
    definition.args.args = [*(ast.arg(arg=name) for name in names), *(ast.arg(arg=name) for name in hidden[:2])]
    definition.returns = None
    definition = ast.fix_missing_locations(Float32Constants().visit(definition))
    namespace = dict(item.globals_map)
    namespace.update({"__name__": function.__module__, hidden[2]: np.float32, hidden[3]: builtins.range})
    for name, value in tuple(namespace.items()):
        if isinstance(value, (float, np.floating)):
            namespace[name] = np.float32(value)
    filename = "<infernux-native-range:" + function.__qualname__ + ">"
    exec(builtins.compile(ast.Module(body=[definition], type_ignores=[]), filename, "exec"), namespace)
    generated = namespace[definition.name]
    # Retain only the existing backend's actual global references, not the
    # author module's scene/editor objects or a new source-text cache.
    from infernux._jit_backend import _global_names
    required = set(_global_names(generated.__code__)) | {"__builtins__"}
    for name in tuple(namespace):
        if name not in required:
            del namespace[name]
    compiled = compile(generated, auto_parallel=False, fastmath=WORK_ITEM_FAST_MATH,
                       error_model=CPU_WORK_ITEM_ERROR_MODEL)
    result = _CompiledCpuRange(compiled._compiled, names, parameter_kinds, item.domain_parameter, plan)
    # Frozen code/constant identity; cost metadata never keys on live buffers or
    # the changing parameter generation of a trained model.
    result.cost_identity = hashlib.sha256(json.dumps(dict(
        source=ast.dump(ast.Module(body=[*item.helpers, item.definition], type_ignores=[]), include_attributes=False),
        constants=item.compile_values, kinds=parameter_kinds, fast_math=WORK_ITEM_FAST_MATH,
        error_model=CPU_WORK_ITEM_ERROR_MODEL,
    ), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    warmup(result, *params)
    result.hir = hir
    result.arithmetic_types = tuple(sorted({str(signature.return_type)
        for overload in result._compiled.overloads.values()
        for expression, signature in overload.type_annotation.calltypes.items()
        if getattr(expression, "op", None) in {"binop", "inplace_binop", "unary", "call"}
        and str(signature.return_type) in {"float32", "float64"}}))
    if "float64" in result.arithmetic_types:
        raise TypeError("CPU continuation requires float32 arithmetic; use floating literals/float32 operands instead of implicit wider arithmetic")
    return result

def _compile_reduction_work_item(function, params, *, receiver_fields=None, _prepared=None, native_entries=True):
    """Cold, finite CPU programs from one ordered region; no execution/scheduling."""
    import numpy as np

    programs = {}
    carry = lowered = samples = program = whole = variants = None
    item = values = hir = plan = kinds = None
    try:
        item, values, hir, plan, kinds = (_prepare_cpu_work_item(function, params, receiver_fields=receiver_fields)
                                       if _prepared is None else _prepared)
        variants = item.reduction_variants(plan, kinds)
        region = plan.ordered_reduction
        whole = _compile_cpu_work_item(function, item, values, hir, plan, kinds)
        programs["whole"] = whole
        if native_entries:
            whole._native_entry()
        count = len(whole._bind(values)[0][whole.domain_parameter])
        if region.count > 1:
            carry = np.empty((2, count), dtype=np.float32)
            for kind, lowered in variants.items():
                if kind == "whole":
                    continue
                begin, end = (0, 1) if kind == "initial" else (1, 2) if kind == "resume" else (region.count - 1, region.count)
                samples = (*values, carry[0], carry[1], np.int32(begin), np.int32(end))
                program = _compile_cpu_work_item(function, lowered.item, samples, hir,
                                                lowered.continuation, lowered.parameter_kinds)
                if native_entries:
                    program._native_entry()
                programs[kind] = program
        whole._inner_variants = {kind: program for kind, program in programs.items() if kind != "whole"}
        result, programs = programs, None
        return result
    except BaseException:
        programs.clear()
        raise
    finally:
        function = params = item = values = hir = plan = kinds = carry = lowered = samples = program = whole = variants = _prepared = None

def warmup(fn, *args, **kwargs):
    """Prepare native CPU code without executing authored gameplay state.

    Targets without a CPU compiler remove public warmup calls while cooking
    project scripts. Reaching this runtime API therefore always requires the
    bundled native JIT instead of silently changing execution semantics.
    """
    if not _jit_runtime_available():
        raise RuntimeError(
            "inx.jit.warmup requires the bundled Numba/llvmlite CPU JIT runtime"
        )
    return _native_warmup(fn, *args, **kwargs)

def statistics(fn) -> Statistics:
    """Snapshot a CPU compilation without compiling or executing the function.

    Preparation time includes dependency compilation/cache loading, excluding
    waiting for the compiler lock; it is not steady-state execution time.
    Pass timings describe cold compilation only. The report owns no native
    code and remains readable after the function has retired.

    Memory counts require the Infernux llvmlite fork, otherwise they are None.
    They measure mapped code/data including allocator padding, not IR or RSS.
    Reachable memory includes shared linking dependencies and cannot be summed
    across function reports. This reports usage, not a memory-budget guarantee.
    """
    if not isinstance(fn, _CompiledCpuFunction):
        raise TypeError("jit.statistics expects a function returned by jit.compile")
    from infernux._jit_backend import compilation_statistics

    compiled = fn._compiled
    if getattr(compiled, "auto_parallel", False):
        implementations = [("serial", compiled.serial)]
        if compiled.parallel is not compiled.serial:
            implementations.append(("parallel", compiled.parallel))
        mode = compiled.selected_mode
        diagnostic = compiled.last_diagnostic
        decisions = tuple((repr(key), value) for key, value in compiled.decisions.items())
    else:
        mode = "parallel" if compiled.targetoptions.get("parallel") else "serial"
        implementations = [(mode, compiled)]
        diagnostic = "explicit CPU implementation"
        decisions = ()
    return compilation_statistics(implementations, function_name=fn.__qualname__,
                                  selected_mode=mode, last_diagnostic=diagnostic, decisions=decisions)

__all__ = [
    "JIT_AVAILABLE", "compile", "warmup", "statistics", "Statistics",
    "PassTiming", "SpecializationStatistics", "OptimizationReport",
]
