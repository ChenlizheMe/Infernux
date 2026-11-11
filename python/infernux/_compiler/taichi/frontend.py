"""Private Python-to-SPIR-V lowering used by :mod:`infernux.compute`.

This module is deliberately not a Taichi compatibility layer. It translates
the Infernux single-work-item kernel contract into the temporary internal IR
frontend and returns compiler artifacts. All imports stay inside the engine's
private compiler namespace; unrelated packages keep their own module identity.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import linecache
import os
from pathlib import Path
import struct
import sys
import threading

import numpy as np

from . import CompilerInstallationError, _vendor_dir, load_native
from ..kernel_contract import WORK_ITEM_FAST_MATH, attribute_name
from ..work_item import WorkItem, _function_source, normalize_work_item, receiver_bindings, receiver_fields
from ..cache import compiler_cache_root, prune_cache_files
from ..gpu_artifact import (
    _CACHE_ABI, _CACHE_MAGIC, CompilerArtifact, RangeContract, ArtifactAbiMismatchError,
    parameter_key, _specialization_identity, _compilation_request, _read_artifact, _decode_artifact, _encode_artifact,
)


_VENDOR_NAME = "infernux._compiler.taichi._vendor.taichi"
_lock = threading.RLock()
_vendor = None
_CACHE_FILE_LIMIT = 128
_CACHE_BYTE_LIMIT = 256 * 1024 * 1024


class _IntrinsicLowering(ast.NodeTransformer):
    """Map the small public kernel intrinsic surface to compiler intrinsics."""

    _names = {
        "acos", "asin", "atan", "atan2", "ceil", "cos", "exp", "floor",
        "log", "sin", "sqrt", "tan", "tanh",
    }

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in self._names):
            node.func = ast.Attribute(
                value=ast.Name(id="_infernux_gpu_intrinsics", ctx=ast.Load()),
                attr=node.func.attr,
                ctx=ast.Load(),
            )
        elif _attribute_name(node.func) in {
            "atomic_add", "compute.atomic_add", "inx.compute.atomic_add",
            "infernux.compute.atomic_add",
        }:
            node.func = ast.Attribute(
                value=ast.Name(id="_infernux_gpu_intrinsics", ctx=ast.Load()),
                attr="atomic_add",
                ctx=ast.Load(),
            )
        return node


def _constant_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)):
        value = int(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


class _OriginalLoopControl(ast.NodeTransformer):
    """Retarget control statements from one logical iteration.

    The iteration is placed inside a one-shot constant loop. ``continue`` then
    naturally advances to the next logical iteration. ``break`` additionally
    records that the surrounding author loop must stop. Nested author loops
    have already been lowered and own their control flow, so they are opaque.
    """

    def __init__(self, break_name: str):
        self._break_name = break_name

    def visit_Break(self, node: ast.Break):
        assignment = ast.Assign(
            targets=[ast.Name(id=self._break_name, ctx=ast.Store())],
            value=ast.Constant(value=True),
        )
        return [ast.copy_location(assignment, node), node]

    def visit_For(self, node: ast.For):
        return node

    def visit_While(self, node: ast.While):
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node


class _SerialLoopPairLowering(ast.NodeTransformer):
    """Canonicalize dynamic serial loops before SPIR-V generation.

    Some Vulkan drivers miscompile a canonical signed-integer induction
    reduction during pipeline creation. Executing at most two *logical*
    iterations per physical loop is a standard partial-unroll form and avoids
    that faulty recognizer. It does not change author types, dispatch size, or
    the single-work-item contract. Constant ranges remain available to the
    ordinary compiler unroller, and the generated dispatch range is added only
    after this pass, so it can never be serialized here.
    """

    def __init__(self, definition: ast.FunctionDef):
        self._counter = 0
        self._used_names = {
            node.id if isinstance(node, ast.Name) else node.arg for node in ast.walk(definition)
            if isinstance(node, (ast.Name, ast.arg))
        }

    def _fresh(self, role: str) -> str:
        while True:
            name = f"__inx_serial_{role}_{self._counter}"
            self._counter += 1
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    def _statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            visited = self.visit(statement)
            if visited is None:
                continue
            if isinstance(visited, list):
                result.extend(visited)
            else:
                result.append(visited)
        return result

    @staticmethod
    def _name(name: str, context: ast.expr_context) -> ast.Name:
        return ast.Name(id=name, ctx=context)

    def _logical_iteration(self, body: list[ast.stmt], break_name: str) -> ast.For:
        once_name = self._fresh("once")
        copied = copy.deepcopy(body)
        rewritten: list[ast.stmt] = []
        control = _OriginalLoopControl(break_name)
        for statement in copied:
            value = control.visit(statement)
            if isinstance(value, list):
                rewritten.extend(value)
            elif value is not None:
                rewritten.append(value)
        return ast.For(
            target=self._name(once_name, ast.Store()),
            iter=ast.Call(func=self._name("range", ast.Load()), args=[ast.Constant(value=1)], keywords=[]),
            body=rewritten,
            orelse=[],
        )

    def _guarded_second_iteration(
        self, condition: ast.expr, body: list[ast.stmt], break_name: str,
    ) -> ast.If:
        return ast.If(
            test=ast.BoolOp(op=ast.And(), values=[
                condition,
                ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            ]),
            body=[self._logical_iteration(body, break_name)],
            orelse=[],
        )

    def visit_For(self, node: ast.For):
        node.body = self._statements(node.body)
        node.orelse = self._statements(node.orelse)
        if (not isinstance(node.iter, ast.Call)
                or not isinstance(node.iter.func, ast.Name)
                or node.iter.func.id != "range"
                or node.iter.keywords
                or not 1 <= len(node.iter.args) <= 3):
            return node

        values = tuple(_constant_integer(argument) for argument in node.iter.args)
        if all(value is not None for value in values):
            return node

        if len(node.iter.args) == 1:
            start_expression = ast.Constant(value=0)
            stop_expression = node.iter.args[0]
            step = 1
        else:
            start_expression, stop_expression = node.iter.args[:2]
            step = 1 if len(node.iter.args) == 2 else _constant_integer(node.iter.args[2])
            if step is None:
                raise TypeError("GPU dynamic range step must be a compile-time integer")
            if step == 0:
                raise TypeError("GPU range step cannot be zero")

        start_name = self._fresh("start")
        stop_name = self._fresh("stop")
        cursor_name = self._fresh("cursor")
        break_name = self._fresh("break")
        prefix = [
            ast.Assign(targets=[self._name(start_name, ast.Store())], value=start_expression),
            ast.Assign(targets=[self._name(stop_name, ast.Store())], value=stop_expression),
            ast.Assign(
                targets=[self._name(cursor_name, ast.Store())],
                value=self._name(start_name, ast.Load()),
            ),
            ast.Assign(targets=[self._name(break_name, ast.Store())], value=ast.Constant(value=False)),
        ]

        def condition() -> ast.expr:
            return ast.Compare(
                left=self._name(cursor_name, ast.Load()),
                ops=[ast.Lt() if step > 0 else ast.Gt()],
                comparators=[self._name(stop_name, ast.Load())],
            )

        def prepare_iteration() -> list[ast.stmt]:
            return [
                ast.Assign(targets=[copy.deepcopy(node.target)], value=self._name(cursor_name, ast.Load())),
                ast.AugAssign(
                    target=self._name(cursor_name, ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(value=step),
                ),
            ]

        first = [*prepare_iteration(), self._logical_iteration(node.body, break_name)]
        second = ast.If(
            test=ast.BoolOp(op=ast.And(), values=[
                condition(),
                ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            ]),
            body=[*prepare_iteration(), self._logical_iteration(node.body, break_name)],
            orelse=[],
        )
        physical_loop = ast.While(
            test=ast.BoolOp(op=ast.And(), values=[
                condition(),
                ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            ]),
            body=[*first, second],
            orelse=[],
        )
        suffix = [ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            body=node.orelse,
            orelse=[],
        )] if node.orelse else []
        return [*(ast.copy_location(statement, node) for statement in prefix),
                ast.copy_location(physical_loop, node), *suffix]

    def visit_While(self, node: ast.While):
        node.body = self._statements(node.body)
        node.orelse = self._statements(node.orelse)
        break_name = self._fresh("break")
        prefix = ast.Assign(
            targets=[self._name(break_name, ast.Store())],
            value=ast.Constant(value=False),
        )
        first = self._logical_iteration(node.body, break_name)
        second = self._guarded_second_iteration(copy.deepcopy(node.test), node.body, break_name)
        physical_loop = ast.While(
            test=ast.BoolOp(op=ast.And(), values=[
                node.test,
                ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            ]),
            body=[first, second],
            orelse=[],
        )
        suffix = [ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=self._name(break_name, ast.Load())),
            body=node.orelse,
            orelse=[],
        )] if node.orelse else []
        return [ast.copy_location(prefix, node), ast.copy_location(physical_loop, node), *suffix]


def _lower_serial_loops(definition: ast.FunctionDef) -> None:
    definition.body = _SerialLoopPairLowering(definition)._statements(definition.body)


def _load_vendor():
    global _vendor
    if _vendor is not None:
        return _vendor
    with _lock:
        if _vendor is not None:
            return _vendor
        package_dir = _vendor_dir()
        # Player runtimes are source-less; the vendor package is shipped as
        # optimized bytecode alongside its native binding.
        init_file = package_dir / "__init__.py"
        if not init_file.is_file():
            init_file = package_dir / "__init__.pyc"
        if not init_file.is_file():
            raise CompilerInstallationError(
                "The Infernux installation is missing its private GPU kernel lowering sources"
            )
        spec = importlib.util.spec_from_file_location(
            _VENDOR_NAME, init_file, submodule_search_locations=[str(package_dir)]
        )
        if spec is None or spec.loader is None:
            raise CompilerInstallationError(
                f"The Infernux GPU kernel frontend cannot be loaded: {init_file}"
            )
        module = importlib.util.module_from_spec(spec)
        native = load_native()
        sys.modules[_VENDOR_NAME] = module
        sys.modules[_VENDOR_NAME + "._lib.core._infernux_gpu_compiler"] = native
        try:
            spec.loader.exec_module(module)
            _vendor = module
        except BaseException:
            sys.modules.pop(_VENDOR_NAME, None)
            raise
        return _vendor


def _attribute_name(node: ast.expr) -> str | None:
    value = attribute_name(node)
    return value or None


def _buffer_annotation(ti, value, *, needs_grad=False):
    scalar = getattr(ti.types.primitive_types, value._dtype.scalar.name)
    dtype = ti.types.vector(value._dtype.lanes, scalar) if value._dtype.lanes > 1 else scalar
    return ti.types.external_buffer(dtype=dtype, ndim=len(value.shape), needs_grad=needs_grad)










def _cache_root() -> Path:
    return compiler_cache_root()


def _artifact_key(function, definition, helpers, params, compile_values, *, autodiff_mode="none", gradient_parameters=(),
                  ordered_body=None) -> str:
    source_tree = ast.Module(body=[*helpers, definition], type_ignores=[])
    payload = {
        "abi": _CACHE_ABI,
        "fast_math": WORK_ITEM_FAST_MATH,
        "float_not_equal": "unordered",
        "float32_semantics": "signed-zero-inf-nan-preserve",
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source": ast.dump(source_tree, include_attributes=False),
        "parameters": parameter_key(params),
        "compile_values": compile_values,
        "autodiff_mode": autodiff_mode,
        "gradient_parameters": gradient_parameters,
        "target": "vulkan-1.2-spirv-1.5",
    }
    if ordered_body is not None:
        # Lowering removes the original loop bound and epilogue from some
        # stages. Their source proof must still distinguish cached contracts.
        payload["ordered_body"] = ordered_body
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()












def _load_artifact(key: str, params, *, autodiff_mode="none", gradient_parameters=()) -> CompilerArtifact | None:
    path = _cache_root() / f"{key}.inxgpu"
    return _decode_artifact(path, specialization=_specialization_identity(
        params, autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters))


def _load_player_artifact(identity: str, params, artifact_key: str, *, autodiff_mode="none", gradient_parameters=()) -> CompilerArtifact:
    root = _cache_root()
    manifest_path = root / "AotManifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = document["artifacts"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Player GPU AOT artifact is missing: {manifest_path}") from exc
    specialization = _specialization_identity(params, autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters)
    matches = [
        record for record in records
        if isinstance(record, dict)
        and record.get("function") == identity
        and record.get("specialization") == specialization
        and record.get("artifact") == f"{artifact_key}.inxgpu"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Player GPU AOT specialization is missing or ambiguous for " + identity
        )
    filename = str(matches[0].get("artifact", ""))
    if not filename or Path(filename).name != filename or not filename.endswith(".inxgpu"):
        raise RuntimeError(f"Player GPU AOT artifact path is invalid for {identity}")
    artifact = _decode_artifact(root / filename, specialization=specialization)
    if artifact is None:
        raise RuntimeError(f"Player GPU AOT artifact is unavailable for {identity}")
    return artifact


def _store_artifact(key: str, artifact: CompilerArtifact, params) -> None:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.inxgpu"
    payload = _encode_artifact(artifact, _specialization_identity(
        params, autodiff_mode=artifact.autodiff_mode, gradient_parameters=artifact.gradient_parameters))
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)

    prune_cache_files(root, "*.inxgpu", file_limit=_CACHE_FILE_LIMIT, byte_limit=_CACHE_BYTE_LIMIT)


@dataclass(slots=True)
class KernelCompilationPlan:
    artifact_key: str
    definition: ast.FunctionDef
    helpers: list[ast.FunctionDef]
    globals_map: dict
    domain_parameter: int
    autodiff_mode: str
    gradient_parameters: tuple[int, ...]
    work_item: WorkItem
    continuation: RangeContract
    fast_math: bool


def _work_item_proof(work_item, params):
    from infernux.compute import Buffer
    from infernux.jit_hir import TypeRef, ValueType

    names = tuple(argument.arg for argument in (*work_item.definition.args.posonlyargs, *work_item.definition.args.args))
    types = {name: TypeRef(ValueType.ARRAY if isinstance(value, Buffer) else
                           ValueType.FLOAT if isinstance(value, (float, np.floating)) else ValueType.INT)
             for name, value in zip(names, params)}
    return work_item.analyze(argument_types=types).continuation_plan


def _continuation_contract(work_item, params, *, autodiff_mode, gradient_parameters, proof=None):
    from infernux.compute import Buffer
    from infernux._compiler.kernel_contract import ARRAY_KINDS

    names = tuple(argument.arg for argument in (*work_item.definition.args.posonlyargs, *work_item.definition.args.args))
    proof = _work_item_proof(work_item, params) if proof is None else proof
    diagnostics = [item.format() for item in proof.diagnostics]
    if autodiff_mode != "none" or gradient_parameters:
        diagnostics.append("local VJP parameters do not have a proven range ABI")
    inner = proof.inner_range
    if inner is not None:
        inner.validate(params[inner.begin_parameter], params[inner.end_parameter])
    if any((len(value.shape) != 1 or str(value._dtype.scalar) not in ARRAY_KINDS or value._dtype.lanes != 1)
           if isinstance(value, Buffer) else not isinstance(value, (float, np.floating, int, np.integer))
           for value in params):
        diagnostics.append("continuation requires typed POD buffers and float32/int32 scalars")
    count = len(params)
    return RangeContract(count if not diagnostics else None, count + 1 if not diagnostics else None,
        tuple(sorted(names.index(name) for name in proof.inputs if name in names)),
        tuple(sorted(names.index(name) for name in proof.outputs if name in names)),
        proof.operation_cost, tuple(diagnostics),
        tuple(sorted(names.index(name) for name in proof.fully_written_outputs)) if not diagnostics else (),
        resource_extents=proof.resource_extents if not diagnostics else (),
        gathered=proof.gathered if not diagnostics else False,
        inner_range=inner if not diagnostics else None)


def _lower_continuation(work_item, definition):
    hidden = ("_infernux_range_begin", "_infernux_range_end", "_infernux_range_relative")
    used = {node.id if isinstance(node, ast.Name) else node.arg for node in ast.walk(definition)
            if isinstance(node, (ast.Name, ast.arg))} | set(work_item.globals_map)
    if any(name in used for name in hidden):
        raise TypeError("work item source uses a reserved _infernux_range_ name")
    definition.args.args = [*definition.args.posonlyargs, *definition.args.args, *(ast.arg(arg=name) for name in hidden[:2])]
    definition.args.posonlyargs = []
    assign_index = ast.Assign(targets=[ast.Name(id=work_item.index_name, ctx=ast.Store())],
        value=ast.BinOp(left=ast.Name(id=hidden[0], ctx=ast.Load()), op=ast.Add(),
                        right=ast.Name(id=hidden[2], ctx=ast.Load())))
    definition.body = [ast.For(target=ast.Name(id=hidden[2], ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.BinOp(
            left=ast.Name(id=hidden[1], ctx=ast.Load()), op=ast.Sub(), right=ast.Name(id=hidden[0], ctx=ast.Load()))], keywords=[]),
        body=[assign_index, *definition.body], orelse=[])]
    return ast.fix_missing_locations(definition)


def prepare_kernel(function, params, *, receiver_fields: tuple[tuple[str, object], ...] | None = None,
                   autodiff_mode="none", gradient_parameters=(), _lowered=None) -> KernelCompilationPlan:
    """Use the shared author body before GPU-specific storage/lowering checks."""
    from infernux.compute import Buffer

    gradient_parameters = _compilation_request(params, autodiff_mode, gradient_parameters)
    if _lowered is not None and (autodiff_mode != "none" or gradient_parameters or receiver_fields is not None):
        raise TypeError("lowered work items use their frozen receiver and forward arithmetic roles")
    work_item = normalize_work_item(function, len(params), receiver_fields=receiver_fields) if _lowered is None else _lowered.item
    domain = params[work_item.domain_parameter]
    if not isinstance(domain, Buffer) or domain.device != "gpu" or len(domain.shape) != 1:
        raise TypeError("inx.compute.index currently requires a one-dimensional GPU inx.buffer")
    for value in params:
        if isinstance(value, Buffer) and value.device != "gpu":
            raise TypeError("GPU kernels require GPU inx.buffer parameters")
    definition = copy.deepcopy(work_item.definition)
    helpers = copy.deepcopy(work_item.helpers)
    continuation = _continuation_contract(work_item, params, autodiff_mode=autodiff_mode,
                                         gradient_parameters=gradient_parameters,
                                         proof=None if _lowered is None else _lowered.continuation)
    _lower_serial_loops(definition)
    for helper in helpers:
        _lower_serial_loops(helper)
    definition = _lower_continuation(work_item, definition) if continuation.eligible else work_item.loop_definition(definition)
    artifact_key = _artifact_key(function, definition, helpers, params, work_item.compile_values,
                                 autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters,
                                 ordered_body=None if continuation.inner_range is None else
                                 _lowered.continuation.ordered_reduction.body_identity)
    return KernelCompilationPlan(artifact_key, definition, helpers, dict(work_item.globals_map),
                                 work_item.domain_parameter, autodiff_mode, gradient_parameters, work_item, continuation,
                                 WORK_ITEM_FAST_MATH)


def compile_kernel(function, params, *, receiver_fields: tuple[tuple[str, object], ...] | None = None,
                   autodiff_mode="none", gradient_parameters=(), _lowered=None) -> CompilerArtifact:
    """Lower one Infernux kernel specialization without creating a GPU."""
    from infernux.compute import Buffer

    plan = prepare_kernel(function, params, receiver_fields=receiver_fields,
                          autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters, _lowered=_lowered)
    request = {"autodiff_mode": plan.autodiff_mode, "gradient_parameters": plan.gradient_parameters}
    artifact_key = plan.artifact_key
    identity = f"{function.__module__}.{function.__qualname__}"
    if plan.continuation.inner_range is not None:
        identity += "#" + plan.continuation.inner_range.kind
    if (_cache_root() / "AotOnly").is_file():
        return _load_player_artifact(identity, params, artifact_key, **request)
    cached = _load_artifact(artifact_key, params, **request)
    if cached is not None:
        return cached

    definition, helpers, globals_map = plan.definition, plan.helpers, plan.globals_map
    generated_name = f"_infernux_kernel_{function.__name__}_{id(function):x}"
    definition.name = generated_name
    tree = ast.Module(body=[*helpers, definition], type_ignores=[])
    ti = _load_vendor()
    signature = []
    with _lock:
        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is None:
            runtime.create_program()
        program = runtime.prog
        config = program.config()
        # Set the actual Program configuration before ensure_compiled creates IR.
        config.fast_math = plan.fast_math
        runtime.short_circuit_operators = True
        runtime.print_full_traceback = False
        runtime.unrolling_limit = 32
        globals_map["_infernux_gpu_intrinsics"] = importlib.import_module(_VENDOR_NAME + ".lang.ops")
        globals_map["_infernux_gpu_function"] = importlib.import_module(
            _VENDOR_NAME + ".lang.kernel_impl"
        ).func
        internal_params = (*params, 0, 0) if plan.continuation.eligible else params
        for index, (argument, value) in enumerate(zip(definition.args.args, internal_params)):
            annotation_name = f"_infernux_parameter_{index}"
            if isinstance(value, Buffer):
                annotation = _buffer_annotation(ti, value, needs_grad=index in plan.gradient_parameters)
                globals_map[annotation_name] = annotation
                signature.append(annotation)
            elif isinstance(value, (bool, int, np.integer)):
                globals_map[annotation_name] = ti.i32
                signature.append(None)
            else:
                globals_map[annotation_name] = ti.f32
                signature.append(None)
            argument.annotation = ast.Name(id=annotation_name, ctx=ast.Load())
        tree = _IntrinsicLowering().visit(tree)
        ast.fix_missing_locations(tree)
        generated_source = ast.unparse(tree)
        filename = f"<infernux-kernel:{function.__module__}.{function.__qualname__}>"
        linecache.cache[filename] = (
            len(generated_source), None, generated_source.splitlines(keepends=True), filename
        )
        namespace = globals_map
        try:
            exec(compile(generated_source, filename, "exec", dont_inherit=True), namespace)
            compiler = ti.kernel(namespace[generated_name], autodiff_mode=plan.autodiff_mode)
            key = compiler.ensure_compiled(*signature)
            kernel = compiler.compiled_kernels[key]
            target_caps = program.make_target_caps(any(isinstance(value, Buffer) and value._dtype.scalar == np.dtype(np.int64)
                                                       for value in params))
            compiled = program.compile_kernel(config, target_caps, kernel)
            artifact = CompilerArtifact(
                spirv_tasks=tuple(compiled._infernux_spirv_tasks),
                task_metadata=tuple(dict(item) for item in compiled._infernux_task_metadata),
                domain_parameter=plan.domain_parameter,
                argument_layout=dict(kernel._infernux_argument_layout),
                required_capabilities=dict(compiled._infernux_required_capabilities),
                diagnostic_locations=tuple({
                    "entry_point": str(item["entry_point"]),
                    "path": str(function.__code__.co_filename),
                    "line": int(function.__code__.co_firstlineno),
                    "column": 0,
                    "function": identity,
                } for item in compiled._infernux_task_metadata),
                continuation=plan.continuation,
                fast_math=config.fast_math,
                **request,
            )
        finally:
            # exec-created functions refer back to this request-local globals
            # dict. Break that cycle without waiting for the editor's GC pass.
            namespace.clear()
            # Source text is needed only while lowering this request. Keeping
            # every hot-reloaded function here would bypass the artifact cache.
            del linecache.cache[filename]
        _store_artifact(artifact_key, artifact, params)
        return artifact


__all__ = ["CompilerArtifact", "compile_kernel", "parameter_key", "receiver_fields", "receiver_bindings"]
