"""Numerical WorkItem lowering through the engine's resident Slang compiler."""
from __future__ import annotations

import ast
import base64
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np

from .slang_ast import SlangEmitter, common_type, qualified
from .shader_library import Function, LibraryGraph, project_surface_sources, typed_function_call
from .value_types import SCALAR_STORAGE_TYPES
from .work_item import WorkItem, normalize_work_item
from .cache import compiler_cache_root, prune_cache_files
from .kernel_contract import WORK_ITEM_FAST_MATH
from .gpu_artifact import (_CACHE_ABI, CompilerArtifact, RangeContract, parameter_key, _specialization_identity,
                           _compilation_request, _decode_artifact, _encode_artifact, _validate_device_count)


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
    target: str
    fast_math: bool = WORK_ITEM_FAST_MATH
    generated_source: str = ""
    lowering: object = None
    source_locations: tuple[dict, ...] = ()
    libraries: tuple[tuple[str, str, str], ...] = ()
    library_functions: dict = field(default_factory=dict)
    author_schema: str = ""


def _work_item_proof(work_item, params, *, value_calls=()):
    from infernux.compute import Buffer
    from infernux.jit_hir import TypeRef, ValueType

    names = tuple(argument.arg for argument in (*work_item.definition.args.posonlyargs, *work_item.definition.args.args))
    types = {name: TypeRef(ValueType.ARRAY if isinstance(value, Buffer) else
                           ValueType.FLOAT if isinstance(value, (float, np.floating)) else ValueType.INT)
             for name, value in zip(names, params)}
    return work_item.analyze(argument_types=types, value_calls=value_calls).continuation_plan


def _continuation_contract(work_item, params, *, autodiff_mode, gradient_parameters, proof=None):
    from infernux.compute import Buffer
    from infernux._compiler.kernel_contract import ARRAY_KINDS

    names = tuple(argument.arg for argument in (*work_item.definition.args.posonlyargs, *work_item.definition.args.args))
    proof = _work_item_proof(work_item, params) if proof is None else proof
    diagnostics = [item.format() for item in proof.diagnostics]
    if work_item.library_calls:
        diagnostics.append("ShaderLib calls require a proven operation bound for continuation")
    if autodiff_mode != "none" or gradient_parameters:
        role = "JVP" if autodiff_mode == "forward" else "VJP"
        diagnostics.append(f"local {role} parameters do not have a proven range ABI")
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


def _graph_memory_contract(item, params, continuation):
    """Prove row storage separately from the unknown cost of a World query.

    ShaderLib exports accept POD values and reflected World inputs are read-only.
    An opaque pure call therefore preserves argument reads without exposing its
    result as an affine index or pretending traversal has a fixed row cost.
    """
    proof = _work_item_proof(item, params, value_calls=item.library_calls)
    if not proof.eligible:
        raise TypeError("Graph compute requires an independent row memory contract: " +
                        "; ".join(diagnostic.format() for diagnostic in proof.diagnostics))
    names = tuple(argument.arg for argument in (*item.definition.args.posonlyargs, *item.definition.args.args))
    return replace(continuation,
                   input_parameters=tuple(sorted(names.index(name) for name in proof.inputs if name in names)),
                   output_parameters=tuple(sorted(names.index(name) for name in proof.outputs if name in names)),
                   fully_written_output_parameters=tuple(sorted(names.index(name) for name in proof.fully_written_outputs)),
                   resource_extents=proof.resource_extents, gathered=proof.gathered, inner_range=proof.inner_range)


def prepare_kernel(function, params, *, target="spirv", receiver_fields=None, autodiff_mode="none", gradient_parameters=(),
                   _lowered=None, _scalar_slang=None, _scalar_slang_jvp=False):
    from infernux.compute import Buffer
    from infernux.lib import _Infernux

    gradients = _compilation_request(params, autodiff_mode, gradient_parameters)
    item = (normalize_work_item(function, len(params), receiver_fields=receiver_fields, allow_device_count=True)
            if _lowered is None else _lowered.item)
    domain = params[item.domain_parameter]
    if not isinstance(domain, Buffer) or domain.device != "gpu" or len(domain.shape) != 1:
        raise TypeError("inx.compute.index requires a one-dimensional GPU buffer")
    continuation = _continuation_contract(item, params, autodiff_mode=autodiff_mode, gradient_parameters=gradients,
                                          proof=None if _lowered is None else _lowered.continuation)
    if item.count_parameter is not None:
        _validate_device_count(params[item.count_parameter])
        if (not continuation.eligible or continuation.gathered or continuation.inner_range is not None
                or item.count_parameter in (*continuation.input_parameters, *continuation.output_parameters)):
            raise TypeError("Device-counted compute work requires independent same-row accesses and a separate count")
        reads = item.domain_parameter in continuation.input_parameters
        writes = item.domain_parameter in continuation.output_parameters
        if reads == writes or (writes and item.domain_parameter not in continuation.fully_written_output_parameters):
            raise TypeError("A counted domain must be read-only or written completely at every active row")
    plan = KernelCompilationPlan("", copy.deepcopy(item.definition), copy.deepcopy(item.helpers), dict(item.globals_map),
                                 item.domain_parameter, autodiff_mode, gradients, item, continuation, target)
    entry = _source_location(item.origin, item.definition, role="entry")
    if continuation.inner_range is not None:
        entry["function"] += "#" + continuation.inner_range.kind
    plan.source_locations = (entry, *(_source_location(item.helper_origins[helper.name], helper, role="helper")
                                     for helper in item.helpers))
    if item.library_calls:
        graph, plan.library_functions = _link_libraries(item, autodiff_mode)
        plan.libraries = graph.sources()
        if graph.ray_lit is not None:
            plan.continuation = _graph_memory_contract(item, params, continuation)
            continuation = plan.continuation
            plan.author_schema = json.dumps({"domain": "compute", "inputs": [], "parameters": [],
                                            "ray_surfaces": graph.ray_lit.manifests,
                                            "ray_surface_catalog": [guid for guid, _ in graph.ray_surfaces]})
        plan.source_locations += tuple({"role": "library", "path": module.path, "line": 1, "column": 1}
                                       for module in graph.ordered)
    if _scalar_slang is not None:
        _link_scalar_row(plan, _scalar_slang, jvp=_scalar_slang_jvp)
    plan.generated_source, plan.lowering = _source(plan, params)
    # Cache the exact compiler input, including its generated ABI and work
    # proof. Runtime scalar values and buffer identities are not specializations.
    identity = {
        "abi": _CACHE_ABI, "compiler": _Infernux._slang_version(), "source": plan.generated_source, "target": target,
        "declarations": plan.source_locations,
        "parameters": parameter_key(params), "continuation": asdict(continuation),
        "autodiff": autodiff_mode, "gradients": gradients, "precise": True,
    }
    if plan.libraries:
        identity["libraries"] = plan.libraries
    if plan.author_schema:
        identity["author_schema"] = plan.author_schema
    if item.library_calls:
        plan.lowering.source_ranges.extend(graph.ranges())
    plan.artifact_key = hashlib.blake2b(json.dumps(identity, sort_keys=True).encode(), digest_size=16).hexdigest()
    plan.lowering.plan = None
    return plan


def _link_scalar_row(plan, definition, *, jvp):
    """Substitute a scalar call while retaining the row wrapper's buffer accesses."""
    entry_point, source, *dependencies = definition
    (assignment,) = plan.definition.body
    if not isinstance(assignment, ast.Assign):
        raise ValueError("A linked Slang scalar requires a direct row-copy wrapper")
    module = "inx_nn_scalar"
    path = "NN/SlangUnary.slang"
    symbol = "_infernux_nn_scalar"
    value = assignment.value
    function = f"{module}::{entry_point}"
    library = f'namespace {module} {{\n#line 1 "{path}"\n{source}\n}}'
    arguments = [value]
    if jvp:
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            raise ValueError("A linked Slang JVP requires a two-input pointwise row wrapper")
        arguments = [value.left, value.right]
        module = "inx_nn_scalar_jvp"
        library += f'''
namespace {module} {{
#line 1 "NN/SlangUnaryJVP.slang"
public float directional(float x, float dx) {{
    let result = fwd_diff({function})(diffPair(x, dx));
    return result.d;
}}
}}
'''
        function = f"{module}::directional"
    if not all(isinstance(argument, ast.Subscript) for argument in arguments):
        raise ValueError("A linked Slang scalar requires direct buffer row inputs")
    assignment.value = ast.copy_location(ast.Call(ast.Name(symbol, ast.Load()), arguments, []), value)
    plan.library_functions[symbol] = Function(function, "float", ("float",) * len(arguments))
    modules = dependencies[0] if dependencies else ()
    plan.libraries = ((module, path, library), *modules)
    plan.source_locations += ({"role": "library", "path": path, "function": entry_point, "line": 1, "column": 1},)
    plan.source_locations += tuple({"role": "library", "path": origin, "line": 1, "column": 1}
                                   for _, origin, _ in modules)
    # The buffers still obey the row-address contract. The Slang body has no
    # bounded service-cost estimate for managed scheduling.
    plan.continuation = replace(plan.continuation, operation_cost=0, cost_source="authored_scalar_slang")


def _source_location(origin, definition, *, role):
    return {"function": origin.function, "path": origin.path, "role": role,
            "line": origin.line_offset + definition.lineno, "column": definition.col_offset + 1}


def _read_library_source(path):
    from infernux.application import Application
    from infernux.resources import resources_path

    source = Path(path)
    if Application.is_player() and not source.is_relative_to(Path(resources_path).resolve()):
        # The cooked asset catalog maps the authored path to the shipped
        # shader source. Keep the original path in Slang's source map and key.
        root = Path(Application.data_path())
        parts = source.relative_to(root).parts if source.is_relative_to(root) else source.parts
        assets = next((index for index, part in enumerate(parts) if part.casefold() == "assets"), None)
        if assets is not None:
            source = Path(Application.asset_path("/".join(parts[assets:])))
    return source.read_text(encoding="utf-8-sig")


def _load_library(graph, reference, owner, location):
    from infernux.engine.project_context import get_project_root

    declaring_path = Path(reference.declaring_path)
    project_root = get_project_root()
    if project_root and not declaring_path.is_absolute():
        declaring_path = Path(project_root) / declaring_path
    return graph.load_reference(reference.module, reference.level, owner, location, declaring_path=declaring_path)


def scalar_library_definition(reference, entry_point):
    """Freeze one typed ShaderLib function for the existing scalar row ABI."""
    graph = LibraryGraph(_read_library_source, compute_autodiff="reverse")
    owner = SlangEmitter("", reference.declaring_path)
    location = ast.Constant(value=None, lineno=1, col_offset=0)
    module = _load_library(graph, reference, owner, location)
    function = graph.scalar_function(module, entry_point, owner, location)
    source = (f"import {module.name};\n[Differentiable]\n"
              f"public float {entry_point}(float value) {{ return {function.symbol}(value); }}\n")
    return source, graph.sources()


def _link_libraries(item, autodiff_mode):
    graph = LibraryGraph(_read_library_source, compute_autodiff=autodiff_mode,
                         allow_world_rays=autodiff_mode == "none",
                         ray_surfaces=project_surface_sources)
    functions = {}
    definitions = ((item.definition, item.origin),
                   *((helper, item.helper_origins[helper.name]) for helper in item.helpers))
    for definition, origin in definitions:
        owner = SlangEmitter(origin.source, origin.path)
        owner.line_offset = origin.line_offset
        for node in ast.walk(definition):
            if not isinstance(node, ast.Call):
                continue
            name = qualified(node.func)
            if name in functions or name not in item.library_calls:
                continue
            reference, export = item.library_calls[name]
            module = _load_library(graph, reference, owner, node)
            if export not in module.functions:
                owner.error(node, f"ShaderLib {reference.module!r} has no export {export!r}",
                            code="author.reference")
            functions[name] = module.functions[export]
    if autodiff_mode != "none":
        graph.prepare_derivatives(functions.values())
    return graph, functions


_SCALAR_TYPES = SCALAR_STORAGE_TYPES


def _element_type(value):
    scalar = _SCALAR_TYPES[value._dtype.scalar.name]
    return scalar if value._dtype.lanes == 1 else scalar + str(value._dtype.lanes)


@dataclass(frozen=True)
class _BufferSchema:
    itemsize: int
    rank: int
    scalar_type: str
    scalar_bytes: int


def _constant(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not np.isfinite(value):
        bits = np.float32(value).view(np.uint32).item()
        return f"asfloat({bits}u)"
    return json.dumps(value)


class _NumericalEmitter(SlangEmitter):
    """Keep Python math's integer results in kernels and imported helpers."""
    def __init__(self, *args, math_aliases=(), math_symbols=None, library_functions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.math_aliases = math_aliases
        self.math_symbols = dict(math_symbols or {})
        self.library_functions = {} if library_functions is None else library_functions

    def symbol(self, name):
        return self.math_symbols.get(name, super().symbol(name))

    def _integer_math_call(self, node):
        name = qualified(node.func)
        module, _, function = name.rpartition(".")
        matched = (self.math_symbols.get(name) in {"ceil", "floor", "trunc"}
                   or module in self.math_aliases and function in {"ceil", "floor", "trunc"})
        if matched and (len(node.args) != 1 or node.keywords):
            self.error(node, "Python math rounding takes one positional scalar", code="author.call")
        return matched

    def infer(self, node):
        if isinstance(node, ast.Call):
            function = self.library_functions.get(qualified(node.func))
            if function is not None:
                return function.result
            if self._integer_math_call(node):
                kind = self.infer(node.args[0])
                return kind if kind in {"int", "uint", "int64_t"} else "int"
        return super().infer(node)

    def visit_Call(self, node):
        function = self.library_functions.get(qualified(node.func))
        if function is not None:
            return typed_function_call(self, node, function)
        if self._integer_math_call(node):
            if self.infer(node.args[0]) in {"int", "uint", "int64_t"}:
                return self.visit(node.args[0])
            return "int(" + super().visit_Call(node) + ")"
        return super().visit_Call(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            function = "inx_floor_div" if isinstance(node.op, ast.FloorDiv) else "inx_mod"
            return f"{function}({self.visit(node.left)}, {self.visit(node.right)})"
        return super().visit_BinOp(node)


class _KernelEmitter(_NumericalEmitter):
    def __init__(self, plan, params, path, source):
        module_aliases = []
        for name, value in plan.globals_map.items():
            if isinstance(value, ModuleType):
                if value.__name__ in {"math", "infernux.shader", "infernux.compute"}:
                    module_aliases.append(name)
                elif value.__name__ == "infernux":
                    module_aliases.extend((name + ".shader", name + ".compute"))
        super().__init__(source, path, aliases=("math", "shader", "inx.compute", "compute", "infernux.compute",
                                               *module_aliases),
                         math_aliases=tuple(name for name, value in plan.globals_map.items()
                                            if isinstance(value, ModuleType) and value.__name__ == "math"),
                         math_symbols={name: value.__name__ for name, value in plan.globals_map.items()
                                       if getattr(value, "__module__", None) == "math"},
                         library_functions=plan.library_functions)
        from infernux.compute import Buffer

        self.plan = plan
        self.buffers = {}
        self.accesses = {}
        self.gradient_accesses = {}
        self.argument_fields = []
        self.constants = []
        self.constant_arrays = {}
        self.constant_symbols = {}
        self.helpers = {function.name: function for function in plan.helpers}
        self.helper_sources = {}
        self.helper_return_types = {}
        names = [arg.arg for arg in (*plan.definition.args.posonlyargs, *plan.definition.args.args)]
        self.names = names
        # Reflection is positional. Keep Python arguments out of the Slang resource namespace.
        self.argument_symbols = {name: f"inx_arg_{i}" for i, name in enumerate(names)}
        for index, (name, value) in enumerate(zip(names, params)):
            symbol = self.argument_symbols[name]
            fields = []
            if isinstance(value, Buffer):
                self.buffers[name] = (index, _BufferSchema(value._dtype.itemsize, len(value.shape),
                                                         _SCALAR_TYPES[value._dtype.scalar.name], value._dtype.scalar.itemsize))
                self.types[name] = _element_type(value)
                fields.extend(("int", f"{symbol}_shape_{axis}") for axis in range(len(value.shape)))
                fields.append(("uint", symbol + "_byte_offset"))
                if index in plan.gradient_parameters:
                    fields.append(("uint", symbol + "_grad_byte_offset"))
            else:
                kind = "int" if isinstance(value, (bool, int, np.integer)) else "float"
                self.types[name] = kind
                fields.append((kind, symbol))
            self.argument_fields.extend(fields)
        if plan.continuation.eligible:
            self.argument_fields.extend([("int", "inx_range_begin"), ("int", "inx_range_end")])
        self.types[plan.work_item.index_name] = "int"
        self.bind_value(plan.work_item.index_name)
        self.atomic_buffers = {
            node.args[0].value.id
            for node in ast.walk(plan.definition)
            if isinstance(node, ast.Call) and self.symbol(qualified(node.func)) == "atomic_add"
            and node.args and isinstance(node.args[0], ast.Subscript)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id in self.buffers
        }
        for name, value in plan.globals_map.items():
            if isinstance(value, (bool, int, float, np.number)):
                kind = "bool" if isinstance(value, (bool, np.bool_)) else "int" if isinstance(value, (int, np.integer)) else "float"
                self.types[name] = kind
                symbol = self.constant_symbols[name] = self.bind_value(name)
                self.constants.append(f"static const {kind} {symbol} = {_constant(value)};")
            elif isinstance(value, (tuple, list, np.ndarray)):
                values = np.asarray(value)
                kind = "float" if values.dtype.kind == "f" else "int"
                self.types[name] = kind
                self.constant_arrays[name] = values.shape
                symbol = self.constant_symbols[name] = self.bind_value(name)
                payload = ", ".join(_constant(item) for item in values.ravel())
                self.constants.append(f"static const {kind} {symbol}[{values.size}] = {{{payload}}};")

    def infer(self, node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in self.buffers or node.value.id in self.constant_arrays:
                return self.types[node.value.id]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return "int"
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if self.symbol(name) == "atomic_add":
                return self.types[self._atomic_target(node)[0]]
            if name == "len":
                return "int"
            if name in self.helpers:
                return self._helper(node)[1]
        return super().infer(node)

    def visit_Name(self, node):
        name = node.id
        if name in self.argument_symbols and name not in self.buffers:
            return "inx_args." + self.argument_symbols[name]
        return super().visit_Name(node)

    def _address(self, name, indices, *, derivative=False):
        _, value = self.buffers[name]
        symbol = self.argument_symbols[name]
        indices = indices.elts if isinstance(indices, ast.Tuple) else [indices]
        linear = "0"
        for axis, index in enumerate(indices):
            if self.infer(index) not in {"int", "uint", "int64_t"}:
                self.error(index, "GPU buffer indices must be integers")
            linear = f"(({linear}) * inx_args.{symbol}_shape_{axis} + ({self.visit(index)}))"
        field = symbol + ("_grad_byte_offset" if derivative else "_byte_offset")
        return f"(inx_args.{field} + uint({linear}) * {value.itemsize}u)"

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.buffers:
            name = node.value.id
            symbol = self.argument_symbols[name]
            address = self._address(name, node.slice)
            self.accesses[name] = self.accesses.get(name, 0) | 1
            if name in self.atomic_buffers:
                return f"inx_atomic_load_{self.types[name]}({symbol}, {address})"
            return f"{symbol}.Load<{self.types[name]}>({address})"
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape" and isinstance(node.value.value, ast.Name):
            name = node.value.value.id
            axis = ast.literal_eval(node.slice)
            return f"inx_args.{self.argument_symbols[name]}_shape_{axis}"
        return super().visit_Subscript(node)

    def assign(self, target, value):
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript):
            element = target.value
            if isinstance(element.value, ast.Name) and element.value.id in self.buffers:
                name = element.value.id
                schema = self.buffers[name][1]
                address = self._address(name, element.slice)
                self.accesses[name] = self.accesses.get(name, 0) | 2
                self.line(f"{self.argument_symbols[name]}.Store<{schema.scalar_type}>({address} + uint({self.visit(target.slice)}) * {schema.scalar_bytes}u, {schema.scalar_type}({value}));")
                return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.buffers:
            name = target.value.id
            symbol = self.argument_symbols[name]
            address = self._address(name, target.slice)
            self.accesses[name] = self.accesses.get(name, 0) | 2
            if name in self.atomic_buffers:
                self.line(f"inx_atomic_store_{self.types[name]}({symbol}, {address}, {self.types[name]}({value}));")
            else:
                self.line(f"{symbol}.Store<{self.types[name]}>({address}, {self.types[name]}({value}));")
        else:
            super().assign(target, value)

    def _helper(self, call, caller=None):
        caller = self if caller is None else caller
        function = self.helpers[qualified(call.func)]
        arguments = list(call.args)
        parameters = [*function.args.posonlyargs, *function.args.args]
        defaults = dict(zip([arg.arg for arg in parameters[-len(function.args.defaults):]], function.args.defaults))
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        keyword_names = {argument.arg for argument in function.args.args}
        supplied_names = {argument.arg for argument in parameters[:len(arguments)]}
        if keywords.keys() - keyword_names or keywords.keys() & supplied_names:
            caller.error(call, "GPU helper keywords do not match its declaration")
        arguments.extend(keywords.get(arg.arg, defaults.get(arg.arg)) for arg in parameters[len(arguments):])
        if any(argument is None for argument in arguments) or len(arguments) != len(parameters):
            caller.error(call, "GPU helper arguments do not match its declaration")
        types = tuple(caller.infer(argument) for argument in arguments)
        key = function.name, types
        if key not in self.helper_sources:
            emitter = _HelperEmitter(self, function)
            signature = list(zip(parameters, types))
            emitter.types.update({argument.arg: kind for argument, kind in signature})
            parameter_list = ", ".join(kind + " " + emitter.bind_value(arg.arg) for arg, kind in signature)
            emitter.declare_locals(function.body, parameters=(argument.arg for argument, _ in signature))
            results = [emitter.infer(node.value) for node in ast.walk(function) if isinstance(node, ast.Return)]
            emitter.local_declarations()
            emitter.block(function.body)
            if not results or any(kind is None for kind in results):
                emitter.error(function, "GPU helpers return a numeric value")
            result = results[0]
            for kind in results[1:]:
                result = common_type(result, kind)
            name = "inx_helper_" + str(len(self.helper_sources))
            attributes = ["[Differentiable]"] if self.plan.autodiff_mode != "none" else []
            self.helper_sources[key] = "\n".join([*attributes, emitter.source_directive(function),
                                                  f"{result} {name}({parameter_list}) {{", *emitter.lines, "}"])
            self.helper_return_types[key] = name, result
            self.source_ranges.extend(emitter.source_ranges)
            self.required_modules.update(emitter.required_modules)
        return (*self.helper_return_types[key], arguments)

    def _atomic_target(self, node):
        if len(node.args) != 2 or node.keywords:
            self.error(node, "atomic_add takes a buffer element and an increment")
        target = node.args[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in self.buffers:
            self.error(target, "atomic_add writes one buffer element")
        buffer = target.value.id
        if self.types[buffer] not in {"int", "uint", "float"}:
            self.error(target, "atomic_add requires a scalar int32, uint32 or float32 buffer element")
        return buffer, target.slice

    def visit_Call(self, node):
        name = qualified(node.func)
        if self.symbol(name) == "atomic_add":
            buffer, indices = self._atomic_target(node)
            address = self._address(buffer, indices)
            self.accesses[buffer] = 3
            return f"inx_atomic_add({self.argument_symbols[buffer]}, {address}, {self.types[buffer]}({self.visit(node.args[1])}))"
        if name == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id in self.buffers:
            return "inx_args." + self.argument_symbols[node.args[0].id] + "_shape_0"
        if name in self.helpers:
            symbol, _, arguments = self._helper(node)
            return symbol + "(" + ", ".join(self.visit(arg) for arg in arguments) + ")"
        return super().visit_Call(node)

    def declarations(self):
        fields = "\n".join("    " + kind + " " + name + ";" for kind, name in self.argument_fields)
        result = ["struct Arguments {\n" + fields + "\n};", "ConstantBuffer<Arguments> inx_args;", *self.constants]
        for name, access in self.accesses.items():
            symbol = self.argument_symbols[name]
            if name in self.atomic_buffers:
                result.append(f"RWStructuredBuffer<Atomic<uint>> {symbol};")
            else:
                prefix = "RW" if access & 2 else ""
                result.append(f"{prefix}ByteAddressBuffer {symbol};")
        for name, access in self.gradient_accesses.items():
            prefix = "RW" if access & 2 else ""
            result.append(f"{prefix}ByteAddressBuffer inx_grad_{self.argument_symbols[name]};")
        return result


class _HelperEmitter(_NumericalEmitter):
    def __init__(self, parent, function):
        origin = parent.plan.work_item.helper_origins[function.name]
        super().__init__(origin.source, origin.path, aliases=parent.aliases, types=parent.types,
                         math_aliases=parent.math_aliases, math_symbols=parent.math_symbols,
                         library_functions=parent.library_functions)
        self.line_offset = origin.line_offset
        self.parent = parent
        self.value_names.update(parent.constant_symbols)

    def infer(self, node):
        if isinstance(node, ast.Call) and qualified(node.func) in self.parent.helpers:
            return self.parent._helper(node, self)[1]
        return super().infer(node)

    def visit_Call(self, node):
        if qualified(node.func) in self.parent.helpers:
            symbol, _, arguments = self.parent._helper(node, self)
            return symbol + "(" + ", ".join(self.visit(arg) for arg in arguments) + ")"
        return super().visit_Call(node)


class _LocalEmitter(_KernelEmitter):
    """The local differentiable function receives values, not GPU resources."""
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.buffers:
            if not isinstance(node.slice, ast.Name) or node.slice.id != self.plan.work_item.index_name:
                self.error(node, "Local differentiation reads the current row", code="author.derivative")
            return "inx_local_" + self.argument_symbols[node.value.id]
        return super().visit_Subscript(node)

    def assign(self, target, value):
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.buffers:
            if not isinstance(target.slice, ast.Name) or target.slice.id != self.plan.work_item.index_name:
                self.error(target, "Local differentiation writes the current row", code="author.derivative")
            self.line("inx_local_" + self.argument_symbols[target.value.id] + " = " + value + ";")
        else:
            base = target
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id in self.buffers:
                self.error(target, "Local differentiation writes complete buffer rows", code="author.derivative")
            super().assign(target, value)


def _local_derivative_source(plan, params, primal):
    inputs = [name for name in primal.names if primal.accesses.get(name, 0) & 1]
    outputs = [name for name in primal.names if primal.accesses.get(name, 0) & 2]
    index = plan.work_item.index_name
    selected = {primal.names[position] for position in plan.gradient_parameters}
    forward = plan.autodiff_mode == "forward"
    if set(inputs) & set(outputs) or not outputs:
        primal.error(plan.definition, "Local differentiation requires separate input and output buffers",
                     code="author.derivative")
    if forward:
        from infernux.jit_hir import _definite_row_writes

        if not selected.intersection(outputs):
            primal.error(plan.definition, "Local JVP requires an output tangent buffer", code="author.derivative")
        if selected.difference(inputs, outputs):
            primal.error(plan.definition, "Local JVP tangent buffers require a corresponding row input or output",
                         code="author.derivative")
        complete = _definite_row_writes(plan.definition.body, index, outputs)
        if missing := sorted(set(outputs).difference(complete)):
            primal.error(plan.definition, "Local JVP requires every row to write " + ", ".join(missing),
                         code="author.derivative")
    for node in ast.walk(plan.definition):
        if isinstance(node, ast.Name) and node.id == index and isinstance(node.ctx, ast.Store):
            primal.error(node, "Local differentiation keeps the execution-domain index fixed", code="author.derivative")
    local = _LocalEmitter(plan, params, primal.path, plan.work_item.origin.source)
    local.line_offset = primal.line_offset
    local.declare_locals(plan.definition.body)
    local.local_declarations()
    types = primal.types
    symbols = primal.argument_symbols
    for name in outputs:
        local.line(f"{types[name]} inx_local_{symbols[name]};")
    local.block(plan.definition.body)
    local.line("LocalOutputs result;")
    for name in outputs:
        local.line(f"result.{symbols[name]} = inx_local_{symbols[name]};")
    local.line("return result;")
    # The discrete row index has no differential.
    parameters = ["no_diff int " + local.value_names[index]]
    for name in inputs:
        qualifier = "" if name in selected else "no_diff "
        parameters.append(f"{qualifier}{types[name]} inx_local_{symbols[name]}")
    fields = " ".join(f"{types[name]} {symbols[name]};" for name in outputs)
    local_function = "struct LocalOutputs : IDifferentiable { " + fields + " };\n[Differentiable]\nLocalOutputs inx_primal(" + ", ".join(parameters) + ") {\n" + "\n".join(local.lines) + "\n}"
    primal.lines.clear()
    primal.accesses.clear()
    index_node = ast.Name(id=index, ctx=ast.Load())
    arguments = [primal.value_names[index]]
    for name in inputs:
        symbol = symbols[name]
        kind = types[name]
        address = primal._address(name, index_node)
        primal.accesses[name] = 1
        if name in selected:
            tangent = f"{kind}(0.0)"
            if forward:
                tangent_address = primal._address(name, index_node, derivative=True)
                tangent = f"inx_grad_{symbol}.Load<{kind}>({tangent_address})"
                primal.gradient_accesses[name] = 1
            primal.line(f"DifferentialPair<{kind}> inx_pair_{symbol} = diffPair({symbol}.Load<{kind}>({address}), {tangent});")
            arguments.append("inx_pair_" + symbol)
        else:
            arguments.append(f"{symbol}.Load<{kind}>({address})")
    if forward:
        primal.line("let inx_result = fwd_diff(inx_primal)(" + ", ".join(arguments) + ");")
        for name in outputs:
            symbol = symbols[name]
            kind = types[name]
            address = primal._address(name, index_node)
            primal.accesses[name] = 2
            primal.line(f"{symbol}.Store<{kind}>({address}, inx_result.p.{symbol});")
            if name in selected:
                tangent_address = primal._address(name, index_node, derivative=True)
                primal.gradient_accesses[name] = 2
                primal.line(f"inx_grad_{symbol}.Store<{kind}>({tangent_address}, inx_result.d.{symbol});")
    else:
        primal.line("LocalOutputs.Differential seed;")
        for name in outputs:
            symbol = symbols[name]
            kind = types[name]
            seed = f"{kind}(0.0)"
            if name in selected:
                address = primal._address(name, index_node, derivative=True)
                primal.gradient_accesses[name] = 1
                seed = f"inx_grad_{symbol}.Load<{kind}>({address})"
            primal.line(f"seed.{symbol} = {seed};")
        primal.line("bwd_diff(inx_primal)(" + ", ".join([*arguments, "seed"]) + ");")
        for name in inputs:
            if name in selected:
                symbol = symbols[name]
                kind = types[name]
                address = primal._address(name, index_node, derivative=True)
                primal.gradient_accesses[name] = 3
                primal.line(f"inx_grad_{symbol}.Store<{kind}>({address}, inx_grad_{symbol}.Load<{kind}>({address}) + inx_pair_{symbol}.d);")
    primal.required_modules.update(local.required_modules)
    primal.helper_sources.update(local.helper_sources)
    primal.source_ranges.extend(local.source_ranges)
    return local_function


_ARITHMETIC = """
int inx_floor_div(int a, int b) { int q = a / b; int r = a - q * b; return q - int(r != 0 && (r < 0) != (b < 0)); }
uint inx_floor_div(uint a, uint b) { return a / b; }
float inx_floor_div(float a, float b) { return floor(a / b); }
int inx_mod(int a, int b) { int r = a - (a / b) * b; return r != 0 && (r < 0) != (b < 0) ? r + b : r; }
uint inx_mod(uint a, uint b) { return a % b; }
float inx_mod(float a, float b) { return a - floor(a / b) * b; }
int inx_atomic_load_int(RWStructuredBuffer<Atomic<uint>> buffer, uint address) {
    return asint(buffer[address / 4u].load());
}
uint inx_atomic_load_uint(RWStructuredBuffer<Atomic<uint>> buffer, uint address) {
    return buffer[address / 4u].load();
}
float inx_atomic_load_float(RWStructuredBuffer<Atomic<uint>> buffer, uint address) {
    return asfloat(buffer[address / 4u].load());
}
void inx_atomic_store_int(RWStructuredBuffer<Atomic<uint>> buffer, uint address, int value) {
    buffer[address / 4u].store(asuint(value));
}
void inx_atomic_store_uint(RWStructuredBuffer<Atomic<uint>> buffer, uint address, uint value) {
    buffer[address / 4u].store(value);
}
void inx_atomic_store_float(RWStructuredBuffer<Atomic<uint>> buffer, uint address, float value) {
    buffer[address / 4u].store(asuint(value));
}
int inx_atomic_add(RWStructuredBuffer<Atomic<uint>> buffer, uint address, int value) {
    return asint(buffer[address / 4u].add(asuint(value)));
}
uint inx_atomic_add(RWStructuredBuffer<Atomic<uint>> buffer, uint address, uint value) {
    return buffer[address / 4u].add(value);
}
float inx_atomic_add(RWStructuredBuffer<Atomic<uint>> buffer, uint address, float value) {
    uint original = buffer[address / 4u].load();
    for (;;) {
        uint assumed = original;
        original = buffer[address / 4u].compareExchange(assumed, asuint(asfloat(assumed) + value));
        if (original == assumed) return asfloat(assumed);
    }
}
"""


def _source(plan, params):
    origin = plan.work_item.origin
    emitter = _KernelEmitter(plan, params, origin.path, origin.source)
    emitter.line_offset = origin.line_offset
    emitter.declare_locals(plan.definition.body)
    emitter.local_declarations()
    emitter.block(plan.definition.body)
    derivative = _local_derivative_source(plan, params, emitter) if plan.autodiff_mode != "none" else ""
    index = emitter.value_names[plan.work_item.index_name]
    domain = emitter.argument_symbols[plan.work_item.domain_name]
    first = "inx_args.inx_range_begin" if plan.continuation.eligible else "0"
    last = "inx_args.inx_range_end" if plan.continuation.eligible else f"inx_args.{domain}_shape_0"
    extent = f"uint(({last}) - ({first}))"
    if plan.work_item.count_parameter is not None:
        count = emitter.names[plan.work_item.count_parameter]
        count_value = emitter.visit(ast.Subscript(ast.Name(count, ast.Load()), ast.Constant(0), ast.Load()))
        extent = f"min({extent}, {count_value})"
    entry = f'''{emitter.source_directive(plan.definition)}
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
    if (tid.x >= {extent}) return;
    int {index} = int(tid.x) + ({first});
'''
    imports = dict.fromkeys("import " + function.symbol.split("::", 1)[0] + ";"
                            for function in plan.library_functions.values())
    code = "\n".join([
        *imports, *(f"import {module};" for module in sorted(emitter.required_modules)),
        *emitter.declarations(), _ARITHMETIC, *emitter.helper_sources.values(),
        derivative, entry, *emitter.lines, "}",
    ])
    return code, emitter


def _artifact(compiled, emitter, plan, params):
    from infernux.compute import Buffer

    resources = {item["name"]: item for item in compiled["resources"]}
    arguments = resources["inx_args"]["value"]
    fields = {field["name"]: field for field in arguments["fields"]}
    layouts = []
    for index, (name, value) in enumerate(zip(emitter.names, params)):
        symbol = emitter.argument_symbols[name]
        if isinstance(value, Buffer):
            layout = {"kind": "external_buffer", "shape_offsets": [fields[f"{symbol}_shape_{axis}"]["offset"] for axis in range(len(value.shape))],
                            "byte_offset_offset": fields[symbol + "_byte_offset"]["offset"], "byte_offset_width": 4,
                            "needs_grad": index in plan.gradient_parameters}
            if index in plan.gradient_parameters:
                layout["grad_byte_offset_offset"] = fields[symbol + "_grad_byte_offset"]["offset"]
            layouts.append(layout)
        else:
            layouts.append({"kind": "int32" if emitter.types[name] == "int" else "float32", "offset": fields[symbol]["offset"]})
    if plan.continuation.eligible:
        layouts.extend({"kind": "int32", "offset": fields[name]["offset"]} for name in ("inx_range_begin", "inx_range_end"))
    bindings = [{"binding": resources["inx_args"]["binding"], "binding_type": "uniform", "resource_kind": "arguments", "access": 1}]
    for name, access in emitter.accesses.items():
        bindings.append({"binding": resources[emitter.argument_symbols[name]]["binding"], "binding_type": "storage" if access & 2 else "read_only_storage",
                         "resource_kind": "external_buffer", "argument_indices": [emitter.names.index(name)], "is_grad": False, "access": access})
    for name, access in emitter.gradient_accesses.items():
        bindings.append({"binding": resources["inx_grad_" + emitter.argument_symbols[name]]["binding"], "binding_type": "storage" if access & 2 else "read_only_storage",
                         "resource_kind": "external_buffer", "argument_indices": [emitter.names.index(name)], "is_grad": True, "access": access})
    entry = compiled["entries"][0]
    metadata = {"entry_point": entry["name"], "threads_per_group": entry["workgroup_size"][0], "buffer_bindings": bindings,
                "task_type": "range_for", "range": {"begin_source": "constant", "begin": 0, "const_begin": True,
                                                       "const_end": False, "end_source": "arguments"}}
    active = {tuple(address) for address in entry["bindings"]}
    numerical = {(0, binding["binding"]) for binding in bindings}
    if active - numerical:
        metadata["graph_metadata"] = base64.b64encode(compiled["graph_metadata"]).decode("ascii")
    if plan.work_item.count_parameter is not None:
        metadata.update(count_parameter=plan.work_item.count_parameter, workgroup_size=entry["workgroup_size"])
    locations = tuple({**location, "entry_point": entry["name"]} for location in plan.source_locations)
    return CompilerArtifact(target=compiled["target"], tasks=(entry["code"],), task_metadata=(metadata,), domain_parameter=plan.domain_parameter,
        argument_layout={"size": arguments["byte_size"], "parameters": layouts},
        # SPIR-V carries SignedZeroInfNanPreserve execution modes. WGSL uses
        # its native finite-f32 contract; it cannot promise those IEEE values.
        required_capabilities={
            "shader_int64": any(isinstance(value, Buffer) and value._dtype.scalar.name == "int64" for value in params),
            "shader_float32_preserve": compiled["target"] == "spirv",
        },
        diagnostic_locations=locations,
        continuation=plan.continuation, fast_math=WORK_ITEM_FAST_MATH, autodiff_mode=plan.autodiff_mode,
        gradient_parameters=plan.gradient_parameters, identity=plan.artifact_key)


@dataclass(frozen=True, slots=True)
class _ArtifactRequest:
    source: str
    path: str
    target: str
    libraries: tuple
    finalize: Callable[[dict], CompilerArtifact]
    source_ranges: tuple = ()
    entry_point: str | None = None
    author_schema: str = ""

    def _invoke(self, compiler):
        author = {"author_schema": self.author_schema} if self.author_schema else {}
        return compiler(self.source, self.path, "inx_kernel", libraries=self.libraries, targets=[self.target],
                        precise=True, source_ranges=self.source_ranges, entry_point=self.entry_point, **author)

    def compile(self):
        from infernux.lib import _Infernux

        return self.finalize(self._invoke(_Infernux._compile_shader_program)["targets"][0])

    def begin(self):
        from infernux.lib import _Infernux

        return self._invoke(_Infernux._begin_compile_shader_program)


@dataclass(frozen=True, slots=True)
class _CachedArtifactRequest:
    request: _ArtifactRequest
    specialization: str
    path: Path

    def load(self):
        return _decode_artifact(self.path, specialization=self.specialization)

    def store(self, artifact):
        root = self.path.parent
        root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(_encode_artifact(artifact, self.specialization))
        os.replace(temporary, self.path)
        prune_cache_files(root, "*.inxgpu", file_limit=128, byte_limit=256 * 1024 * 1024)
        return artifact

    def compile(self):
        artifact = self.load()
        return artifact if artifact is not None else self.store(self.request.compile())

    def begin(self):
        return self.request.begin()

    def finish(self, result):
        return self.store(self.request.finalize(result["targets"][0]))


def _prepare_kernel_artifact(function, params, *, target="spirv", receiver_fields=None, autodiff_mode="none",
                             gradient_parameters=(), _lowered=None, _scalar_slang=None, _scalar_slang_jvp=False):
    plan = prepare_kernel(function, params, target=target, receiver_fields=receiver_fields, autodiff_mode=autodiff_mode,
                          gradient_parameters=gradient_parameters, _lowered=_lowered, _scalar_slang=_scalar_slang,
                          _scalar_slang_jvp=_scalar_slang_jvp)
    specialization = _specialization_identity(params, target=target, autodiff_mode=autodiff_mode, gradient_parameters=plan.gradient_parameters)

    def finalize(code):
        artifact = _artifact(code, plan.lowering, plan, params)
        if _scalar_slang is not None:
            actual = {tuple(binding) for binding in code["entries"][0]["bindings"]}
            expected = {(0, row["binding"]) for row in artifact.task_metadata[0]["buffer_bindings"]}
            if actual != expected:
                raise TypeError("NN Slang functions must be resource-free scalar operations")
        return artifact

    request = _ArtifactRequest(plan.generated_source, plan.work_item.origin.path, target,
                               plan.libraries, finalize, tuple(plan.lowering.source_ranges),
                               author_schema=plan.author_schema)
    return _CachedArtifactRequest(request, specialization, compiler_cache_root() / (plan.artifact_key + ".inxgpu"))


def compile_kernel(function, params, **options):
    return _prepare_kernel_artifact(function, params, **options).compile()


_SLANG_CONSTANTS_MODULE = "inx_specialization"


@dataclass(frozen=True, slots=True)
class _SlangKernelSource:
    path: str
    entry_point: str
    domain: str
    source: str
    modules: tuple[tuple[str, str, str], ...]
    constants: tuple[tuple[str, int], ...]

    @property
    def _linked_modules(self):
        if not self.constants:
            return self.modules
        name = _SLANG_CONSTANTS_MODULE
        text = "\n".join(f"export static const int {key} = {value};" for key, value in self.constants)
        return (*self.modules, (name, f"{name}.slang", text))

    def artifact_key(self, specialization):
        from infernux.lib import _Infernux

        identity = (_CACHE_ABI, _Infernux._slang_version(), self.path, self.entry_point, self.domain,
                    self.source, self._linked_modules, specialization, True)
        return hashlib.blake2b(json.dumps(identity).encode(), digest_size=16).hexdigest()

    def _request(self, target, key):
        def finalize(result):
            return _slang_dispatch_artifact(result, self.path, self.entry_point, self.domain, self.modules, key)

        return _ArtifactRequest(self.source, self.path, target, self._linked_modules, finalize,
                                entry_point=self.entry_point)

    def compile(self, target, key):
        return self._request(target, key).compile()


def _read_slang_kernel_source(path, entry_point, domain, libraries, constants, *, read_source):
    if constants and any(name == _SLANG_CONSTANTS_MODULE for name, _ in libraries):
        raise ValueError(f"Slang library name '{_SLANG_CONSTANTS_MODULE}' is used for link constants")
    modules = read_slang_libraries(libraries, read_source=read_source)
    return _SlangKernelSource(path, entry_point, domain, read_source(path), modules, constants)


def _read_slang_asset(asset):
    from infernux.application import Application

    return Path(Application.asset_path(asset)).read_text(encoding="utf-8-sig")


def read_slang_libraries(libraries, *, read_source=_read_slang_asset):
    return tuple((name, asset, read_source(asset)) for name, asset in libraries)


def _prepare_slang_artifact(path, entry_point, domain, libraries, constants, params, *, target):
    source = _read_slang_kernel_source(path, entry_point, domain, libraries, constants, read_source=_read_slang_asset)
    specialization = _specialization_identity(params, target=target)
    key = source.artifact_key(specialization)
    return _CachedArtifactRequest(source._request(target, key), specialization, compiler_cache_root() / (key + ".inxgpu"))


def compile_slang_kernel(path, entry_point, domain, libraries, constants, params, *, target):
    return _prepare_slang_artifact(path, entry_point, domain, libraries, constants, params, target=target).compile()


def _slang_dispatch_artifact(compiled, path, entry_point, domain, modules, key):
    entry, = compiled["entries"]
    if entry["stage"] != "compute":
        raise ValueError(f"Entry '{entry_point}' must be declared with [shader(\"compute\")]")
    width, height, depth = entry["workgroup_size"]
    if width <= 0 or height != 1 or depth != 1:
        raise ValueError(f"Slang compute entry '{entry_point}' requires [numthreads(x, 1, 1)]")
    if entry["root_constants"] is not None:
        raise ValueError("Slang compute entry resources must use structured storage buffers")

    resources = {(item["set"], item["binding"]): item for item in compiled["resources"]}
    # ShaderProgramLayout.h supplies the value kind, scalar kind and vector width.
    dtypes = {(0, 2, 1): "int32", (0, 3, 1): "uint32", (0, 7, 1): "float32",
              (1, 7, 2): "vector2", (1, 7, 4): "vector4"}
    bindings, layouts, writes = [], [], []
    for group, binding in sorted(entry["bindings"]):
        resource = resources[group, binding]
        value = resource["value"]
        dtype = dtypes.get((value["kind"], value["scalar"], value["columns"]))
        name = resource["name"]
        if group != 0 or resource["type"] != 1 or resource["count"] != 1 or dtype is None:
            raise ValueError(f"Slang binding '{name}' requires a float32, int32, uint32, float2 or float4 "
                             "structured buffer in group 0")
        stride = resource["element_stride"]
        packed_stride = 4 * value["columns"]
        if stride != packed_stride:
            raise ValueError(f"Slang binding '{name}' uses {stride}-byte elements on {compiled['target']}; "
                             f"inx.buffer {dtype} requires {packed_stride}-byte elements")
        index = len(layouts)
        read_only = resource["read_only"]
        layouts.append({"kind": "external_buffer", "name": name, "dtype": dtype})
        bindings.append({"binding": binding, "binding_type": "read_only_storage" if read_only else "storage",
                         "resource_kind": "external_buffer", "argument_indices": [index],
                         "access": 1 if read_only else 3})
        if not read_only:
            writes.append(index)
    domain_index = next((index for index, layout in enumerate(layouts) if layout["name"] == domain), None)
    if domain_index is None:
        raise ValueError(f"Slang dispatch domain '{domain}' must name a reflected buffer of entry '{entry_point}'")

    # An authored dispatch can use group barriers and arbitrary storage
    # indexing. Reflect its accesses and require complete incoming contents.
    contract = RangeContract(None, None, tuple(range(len(layouts))), tuple(writes), 0,
        (f"Slang entry '{entry_point}' executes as a whole dispatch",), ())
    metadata = {"entry_point": entry["name"], "threads_per_group": width,
                "buffer_bindings": bindings, "task_type": "native_dispatch"}
    locations = ({"role": "entry", "path": path, "function": entry_point, "entry_point": entry_point, "line": 1, "column": 1},
                 *({"role": "library", "path": asset, "line": 1, "column": 1} for _, asset, _ in modules))
    return CompilerArtifact(compiled["target"], (entry["code"],), (metadata,), domain_index,
        {"size": 0, "parameters": layouts}, {"shader_float32_preserve": compiled["target"] == "spirv"},
        locations, contract, WORK_ITEM_FAST_MATH, identity=key)

