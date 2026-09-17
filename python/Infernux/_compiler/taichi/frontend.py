"""Private Python-to-SPIR-V lowering used by :mod:`Infernux.compute`.

This module is deliberately not a Taichi compatibility layer. It translates
the Infernux single-work-item kernel contract into the temporary internal IR
frontend and returns compiler artifacts. All imports stay inside the engine's
private compiler namespace; unrelated packages keep their own module identity.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import linecache
import os
from pathlib import Path
import struct
import sys
import textwrap
import threading

import numpy as np

from . import CompilerInstallationError, _vendor_dir, load_native
from ..cache import compiler_cache_root, prune_cache_files


_VENDOR_NAME = "Infernux._compiler.taichi._vendor.taichi"
_lock = threading.RLock()
_vendor = None
_CACHE_MAGIC = b"INXGPU\x01"
_CACHE_ABI = "infernux-gpu-kernel-v1"
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
            "Infernux.compute.atomic_add",
        }:
            node.func = ast.Attribute(
                value=ast.Name(id="_infernux_gpu_intrinsics", ctx=ast.Load()),
                attr="atomic_add",
                ctx=ast.Load(),
            )
        return node


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
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _index_declaration(statement: ast.stmt) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    call = statement.value
    if not isinstance(target, ast.Name) or not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        return None
    name = _attribute_name(call.func)
    if name not in {"index", "inx.compute.index", "Infernux.compute.index", "compute.index"}:
        return None
    if not isinstance(call.args[0], ast.Name):
        raise TypeError("inx.compute.index requires a buffer parameter name")
    return target.id, call.args[0].id


def _buffer_annotation(ti, value):
    scalar = {
        np.dtype(np.float32): ti.f32,
        np.dtype(np.int32): ti.i32,
        np.dtype(np.uint32): ti.u32,
    }[value._dtype.scalar]
    dtype = ti.types.vector(value._dtype.lanes, scalar) if value._dtype.lanes > 1 else scalar
    return ti.types.external_buffer(dtype=dtype, ndim=len(value.shape))


def parameter_key(params) -> tuple:
    """Return the bounded specialization key; runtime sizes are not compiled in."""
    from Infernux.compute import Buffer

    key = []
    for value in params:
        if isinstance(value, Buffer):
            key.append(("buffer", value.dtype, len(value.shape)))
        elif isinstance(value, (bool, int, np.integer)):
            key.append(("int32",))
        elif isinstance(value, (float, np.floating)):
            key.append(("float32",))
        else:
            raise TypeError("GPU kernel parameters must be inx.buffer values or numeric scalars")
    return tuple(key)


def _function_definition(value, *, kind: str) -> ast.FunctionDef:
    if value.__closure__:
        raise TypeError(f"GPU {kind} cannot capture Python closure values")
    source = _function_source(value)
    tree = ast.parse(source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if not isinstance(definition, ast.FunctionDef):
        raise TypeError(f"GPU {kind} must be a synchronous Python function")
    definition.decorator_list = []
    return definition


def _function_source(value) -> str:
    records = value.__globals__.get("__infernux_compute_sources__", {})
    embedded = records.get(value.__qualname__) if isinstance(records, dict) else None
    if embedded is not None:
        if not isinstance(embedded, str) or not embedded.strip():
            raise TypeError(f"GPU source metadata is invalid for {value.__qualname__}")
        return textwrap.dedent(embedded)
    return textwrap.dedent(inspect.getsource(value))


def _helper_definitions(kernel_definition: ast.FunctionDef, globals_map: dict) -> list[ast.FunctionDef]:
    """Collect explicitly declared helpers and reject recursive call graphs."""
    from Infernux.compute import Function

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
                raise TypeError(f"GPU compute helper recursion is not supported: {name}")
            if state == "done":
                continue
            states[name] = "visiting"
            helper_definition = _function_definition(helper.function, kind="compute helper")
            helper_definition.name = name
            visit(helper_definition)
            helper_definition.decorator_list = [ast.Name(id="_infernux_gpu_function", ctx=ast.Load())]
            ordered.append(helper_definition)
            states[name] = "done"

    visit(kernel_definition)
    return ordered


@dataclass(slots=True)
class CompilerArtifact:
    spirv_tasks: tuple[bytes, ...]
    task_metadata: tuple[dict, ...]
    domain_parameter: int
    argument_layout: dict

    def argument_bytes(self, params) -> bytes:
        from Infernux.compute import Buffer

        layouts = tuple(self.argument_layout["parameters"])
        if len(layouts) != len(params):
            raise RuntimeError("GPU compiler argument layout does not match the kernel parameters")
        payload = bytearray(int(self.argument_layout["size"]))
        for layout, value in zip(layouts, params):
            kind = layout["kind"]
            if kind == "external_buffer":
                if not isinstance(value, Buffer):
                    raise TypeError("Compiled GPU buffer layout does not match the launch parameters")
                offsets = tuple(layout["shape_offsets"])
                if len(offsets) != len(value.shape):
                    raise RuntimeError("GPU compiler buffer rank does not match the launch parameters")
                for offset, size in zip(offsets, value.shape):
                    struct.pack_into("<i", payload, int(offset), int(size))
                struct.pack_into(
                    "<Q", payload, int(layout["byte_offset_offset"]),
                    int(value.description.byte_offset),
                )
            elif kind == "int32":
                struct.pack_into("<i", payload, int(layout["offset"]), int(value))
            elif kind == "float32":
                struct.pack_into("<f", payload, int(layout["offset"]), float(value))
            else:
                raise RuntimeError(f"Unsupported GPU compiler argument layout kind: {kind}")
        return bytes(payload)


def _cache_root() -> Path:
    return compiler_cache_root()


def _artifact_key(function, definition, helpers, params) -> str:
    source_tree = ast.Module(body=[*helpers, definition], type_ignores=[])
    payload = {
        "abi": _CACHE_ABI,
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source": ast.dump(source_tree, include_attributes=False),
        "parameters": parameter_key(params),
        "target": "vulkan-1.2-spirv-1.5",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_artifact(key: str) -> CompilerArtifact | None:
    path = _cache_root() / f"{key}.inxgpu"
    if not path.is_file():
        return None
    payload = path.read_bytes()
    if not payload.startswith(_CACHE_MAGIC):
        raise RuntimeError(f"Invalid Infernux GPU compiler cache artifact: {path}")
    header_size = struct.unpack_from("<I", payload, len(_CACHE_MAGIC))[0]
    header_begin = len(_CACHE_MAGIC) + 4
    header_end = header_begin + header_size
    header = json.loads(payload[header_begin:header_end].decode("utf-8"))
    tasks = []
    cursor = header_end
    for size in header["task_sizes"]:
        end = cursor + int(size)
        tasks.append(payload[cursor:end])
        cursor = end
    if cursor != len(payload):
        raise RuntimeError(f"Invalid Infernux GPU compiler cache artifact size: {path}")
    return CompilerArtifact(
        spirv_tasks=tuple(tasks),
        task_metadata=tuple(dict(item) for item in header["task_metadata"]),
        domain_parameter=int(header["domain_parameter"]),
        argument_layout=dict(header["argument_layout"]),
    )


def _store_artifact(key: str, artifact: CompilerArtifact) -> None:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.inxgpu"
    header = json.dumps({
        "task_sizes": [len(task) for task in artifact.spirv_tasks],
        "task_metadata": artifact.task_metadata,
        "domain_parameter": artifact.domain_parameter,
        "argument_layout": artifact.argument_layout,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = _CACHE_MAGIC + struct.pack("<I", len(header)) + header + b"".join(artifact.spirv_tasks)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)

    prune_cache_files(root, "*.inxgpu", file_limit=_CACHE_FILE_LIMIT, byte_limit=_CACHE_BYTE_LIMIT)


def compile_kernel(function, params) -> CompilerArtifact:
    """Lower one Infernux kernel specialization without creating a GPU."""
    from Infernux.compute import Buffer

    signature = inspect.signature(function)
    if any(parameter.kind not in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
           or parameter.default is not parameter.empty for parameter in signature.parameters.values()):
        raise TypeError("GPU kernels currently require positional parameters without defaults")
    if len(signature.parameters) != len(params):
        raise TypeError(f"GPU kernel requires {len(signature.parameters)} parameters, got {len(params)}")

    if function.__closure__:
        raise TypeError("GPU kernel cannot capture Python closure values")
    source = _function_source(function)
    tree = ast.parse(source)
    definition = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not isinstance(definition, ast.FunctionDef):
        raise TypeError("GPU kernel must be a synchronous Python function")
    definition.decorator_list = []
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == function.__name__ for node in ast.walk(definition)):
        raise TypeError(f"GPU kernel recursion is not supported: {function.__name__}")
    if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await)) for node in ast.walk(definition)):
        raise TypeError("GPU kernels write inx.buffer values in place and cannot return or yield")

    declarations = [(position, found) for position, statement in enumerate(definition.body)
                    if (found := _index_declaration(statement)) is not None]
    if len(declarations) != 1:
        raise TypeError("GPU kernel must declare exactly one execution domain with inx.compute.index(buffer)")
    declaration_position, (index_name, domain_name) = declarations[0]
    parameter_names = tuple(signature.parameters)
    if domain_name not in parameter_names:
        raise TypeError("inx.compute.index must refer to an inx.buffer parameter")
    domain_parameter = parameter_names.index(domain_name)
    domain = params[domain_parameter]
    if not isinstance(domain, Buffer) or domain.device != "gpu" or len(domain.shape) != 1:
        raise TypeError("inx.compute.index currently requires a one-dimensional GPU inx.buffer")
    for value in params:
        if isinstance(value, Buffer) and value.device != "gpu":
            raise TypeError("GPU kernels require GPU inx.buffer parameters")

    body = list(definition.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
        declaration_position -= 1
    body.pop(declaration_position)
    definition.body = [ast.For(
        target=ast.Name(id=index_name, ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=domain_name, ctx=ast.Load()), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=0), ctx=ast.Load())],
            keywords=[]),
        body=body,
        orelse=[],
    )]
    globals_map = dict(function.__globals__)
    helpers = _helper_definitions(definition, globals_map)
    artifact_key = _artifact_key(function, definition, helpers, params)
    cached = _load_artifact(artifact_key)
    if cached is not None:
        return cached

    generated_name = f"_infernux_kernel_{function.__name__}_{id(function):x}"
    definition.name = generated_name
    tree.body = [*helpers, definition]
    ti = _load_vendor()
    signature = []
    with _lock:
        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is None:
            runtime.create_program()
        runtime.short_circuit_operators = True
        runtime.print_full_traceback = False
        runtime.unrolling_limit = 32
        globals_map["_infernux_gpu_intrinsics"] = importlib.import_module(_VENDOR_NAME + ".lang.ops")
        globals_map["_infernux_gpu_function"] = importlib.import_module(
            _VENDOR_NAME + ".lang.kernel_impl"
        ).func
        for index, (argument, value) in enumerate(zip(definition.args.args, params)):
            annotation_name = f"_infernux_parameter_{index}"
            if isinstance(value, Buffer):
                annotation = _buffer_annotation(ti, value)
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
            compiler = ti.kernel(namespace[generated_name])
            key = compiler.ensure_compiled(*signature)
            kernel = compiler.compiled_kernels[key]
            program = runtime.prog
            compiled = program.compile_kernel(program.config(), program.get_device_caps(), kernel)
            artifact = CompilerArtifact(
                spirv_tasks=tuple(compiled._infernux_spirv_tasks),
                task_metadata=tuple(dict(item) for item in compiled._infernux_task_metadata),
                domain_parameter=domain_parameter,
                argument_layout=dict(kernel._infernux_argument_layout),
            )
        finally:
            # exec-created functions refer back to this request-local globals
            # dict. Break that cycle without waiting for the editor's GC pass.
            namespace.clear()
            # Source text is needed only while lowering this request. Keeping
            # every hot-reloaded function here would bypass the artifact cache.
            del linecache.cache[filename]
        _store_artifact(artifact_key, artifact)
        return artifact


__all__ = ["CompilerArtifact", "compile_kernel", "parameter_key"]
