"""Numerical WorkItem lowering through the engine's resident Slang compiler."""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os

import numpy as np

from .slang_ast import SlangEmitter, common_type, floating_type, qualified
from .work_item import WorkItem, normalize_work_item, _function_source
from .cache import compiler_cache_root, prune_cache_files
from .kernel_contract import WORK_ITEM_FAST_MATH
from .gpu_artifact import (_CACHE_ABI, CompilerArtifact, RangeContract, parameter_key, _specialization_identity,
                           _compilation_request, _decode_artifact, _encode_artifact)


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
    fast_math: bool = WORK_ITEM_FAST_MATH
    generated_source: str = ""
    lowering: object = None


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


def prepare_kernel(function, params, *, receiver_fields=None, autodiff_mode="none", gradient_parameters=(), _lowered=None):
    from infernux.compute import Buffer
    from infernux.lib import _Infernux

    gradients = _compilation_request(params, autodiff_mode, gradient_parameters)
    item = normalize_work_item(function, len(params), receiver_fields=receiver_fields) if _lowered is None else _lowered.item
    domain = params[item.domain_parameter]
    if not isinstance(domain, Buffer) or domain.device != "gpu" or len(domain.shape) != 1:
        raise TypeError("inx.compute.index requires a one-dimensional GPU buffer")
    continuation = _continuation_contract(item, params, autodiff_mode=autodiff_mode, gradient_parameters=gradients,
                                          proof=None if _lowered is None else _lowered.continuation)
    plan = KernelCompilationPlan("", copy.deepcopy(item.definition), copy.deepcopy(item.helpers), dict(item.globals_map),
                                 item.domain_parameter, autodiff_mode, gradients, item, continuation)
    plan.generated_source, plan.lowering = _source(plan, params, function)
    # Cache the exact compiler input, including its generated ABI and work
    # proof. Runtime scalar values and buffer identities are not specializations.
    identity = {
        "abi": _CACHE_ABI, "compiler": _Infernux._slang_version(), "source": plan.generated_source,
        "parameters": parameter_key(params), "continuation": asdict(continuation),
        "autodiff": autodiff_mode, "gradients": gradients, "precise": True,
    }
    plan.artifact_key = hashlib.blake2b(json.dumps(identity, sort_keys=True).encode(), digest_size=16).hexdigest()
    plan.lowering.plan = None
    return plan


_SCALAR_TYPES = {"float32": "float", "int32": "int", "uint32": "uint", "int64": "int64_t"}


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


class _KernelEmitter(SlangEmitter):
    def __init__(self, plan, params, path, source):
        super().__init__(source, path, aliases=("math", "shader", "inx.compute", "compute", "infernux.compute"))
        from infernux.compute import Buffer

        self.plan = plan
        self.buffers = {}
        self.accesses = {}
        self.gradient_accesses = {}
        self.argument_fields = []
        self.argument_names = []
        self.constants = []
        self.constant_arrays = {}
        self.helpers = {function.name: function for function in plan.helpers}
        self.helper_sources = {}
        self.helper_return_types = {}
        names = [arg.arg for arg in (*plan.definition.args.posonlyargs, *plan.definition.args.args)]
        self.names = names
        for index, (name, value) in enumerate(zip(names, params)):
            fields = []
            if isinstance(value, Buffer):
                self.buffers[name] = (index, _BufferSchema(value._dtype.itemsize, len(value.shape),
                                                         _SCALAR_TYPES[value._dtype.scalar.name], value._dtype.scalar.itemsize))
                self.types[name] = _element_type(value)
                fields.extend(("int", f"{name}_shape_{axis}") for axis in range(len(value.shape)))
                fields.append(("uint", name + "_byte_offset"))
                if index in plan.gradient_parameters:
                    fields.append(("uint", name + "_grad_byte_offset"))
            else:
                kind = "int" if isinstance(value, (bool, int, np.integer)) else "float"
                self.types[name] = kind
                fields.append((kind, name))
            self.argument_fields.extend(fields)
            self.argument_names.append(name)
        if plan.continuation.eligible:
            self.argument_fields.extend([("int", "inx_range_begin"), ("int", "inx_range_end")])
        self.types[plan.work_item.index_name] = "int"
        for name, value in plan.globals_map.items():
            if isinstance(value, (bool, int, float, np.number)):
                kind = "bool" if isinstance(value, (bool, np.bool_)) else "int" if isinstance(value, (int, np.integer)) else "float"
                self.types[name] = kind
                self.constants.append(f"static const {kind} {name} = {_constant(value)};")
            elif isinstance(value, (tuple, list, np.ndarray)):
                values = np.asarray(value)
                kind = "float" if values.dtype.kind == "f" else "int"
                self.types[name] = kind
                self.constant_arrays[name] = values.shape
                payload = ", ".join(_constant(item) for item in values.ravel())
                self.constants.append(f"static const {kind} {name}[{values.size}] = {{{payload}}};")

    def infer(self, node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in self.buffers or node.value.id in self.constant_arrays:
                return self.types[node.value.id]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return "int"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            kind = common_type(self.infer(node.left), self.infer(node.right))
            return floating_type(kind)
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if name in {"math.ceil", "math.floor", "math.trunc"}:
                return "int"
            if name == "len":
                return "int"
            if name in self.helpers:
                return self._helper(node)[1]
        return super().infer(node)

    def visit_Name(self, node):
        name = node.id
        if name in self.argument_names and name not in self.buffers:
            return "inx_args." + name
        return super().visit_Name(node)

    def _address(self, name, indices):
        _, value = self.buffers[name]
        indices = indices.elts if isinstance(indices, ast.Tuple) else [indices]
        linear = "0"
        for axis, index in enumerate(indices):
            if self.infer(index) not in {"int", "uint", "int64_t"}:
                self.error(index, "GPU buffer indices must be integers")
            linear = f"(({linear}) * inx_args.{name}_shape_{axis} + ({self.visit(index)}))"
        return f"(inx_args.{name}_byte_offset + uint({linear}) * {value.itemsize}u)"

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.buffers:
            name = node.value.id
            address = self._address(name, node.slice)
            self.accesses[name] = self.accesses.get(name, 0) | 1
            return f"{name}.Load<{self.types[name]}>({address})"
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape" and isinstance(node.value.value, ast.Name):
            name = node.value.value.id
            axis = ast.literal_eval(node.slice)
            return f"inx_args.{name}_shape_{axis}"
        return super().visit_Subscript(node)

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.FloorDiv):
            return f"inx_floor_div({left}, {right})"
        if isinstance(node.op, ast.Mod):
            return f"inx_mod({left}, {right})"
        if isinstance(node.op, ast.Div):
            if self.infer(node.left).startswith("float") or self.infer(node.right).startswith("float"):
                return f"(({left}) / ({right}))"
            return f"(float({left}) / float({right}))"
        return super().visit_BinOp(node)

    def assign(self, target, value):
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript):
            element = target.value
            if isinstance(element.value, ast.Name) and element.value.id in self.buffers:
                name = element.value.id
                schema = self.buffers[name][1]
                address = self._address(name, element.slice)
                self.accesses[name] = self.accesses.get(name, 0) | 2
                self.line(f"{name}.Store<{schema.scalar_type}>({address} + uint({self.visit(target.slice)}) * {schema.scalar_bytes}u, {schema.scalar_type}({value}));")
                return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.buffers:
            name = target.value.id
            address = self._address(name, target.slice)
            self.accesses[name] = self.accesses.get(name, 0) | 2
            self.line(f"{name}.Store<{self.types[name]}>({address}, {self.types[name]}({value}));")
        else:
            super().assign(target, value)

    def _helper(self, call, caller=None):
        caller = self if caller is None else caller
        function = self.helpers[qualified(call.func)]
        arguments = list(call.args)
        parameters = function.args.args
        defaults = dict(zip([arg.arg for arg in parameters[-len(function.args.defaults):]], function.args.defaults))
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        arguments.extend(keywords.get(arg.arg, defaults.get(arg.arg)) for arg in parameters[len(arguments):])
        if any(argument is None for argument in arguments) or len(arguments) != len(parameters):
            self.error(call, "GPU helper arguments do not match its declaration")
        types = tuple(caller.infer(argument) for argument in arguments)
        key = function.name, types
        if key not in self.helper_sources:
            emitter = _HelperEmitter(self, function)
            signature = list(zip(function.args.args, types))
            emitter.types.update({argument.arg: kind for argument, kind in signature})
            emitter.declare_locals(function.body)
            results = [emitter.infer(node.value) for node in ast.walk(function) if isinstance(node, ast.Return)]
            result = results[0]
            for kind in results[1:]:
                result = common_type(result, kind)
            emitter.local_declarations()
            emitter.block(function.body)
            name = function.name + "_" + str(len(self.helper_sources))
            parameters = ", ".join(kind + " " + arg.arg for arg, kind in signature)
            self.helper_sources[key] = f"{result} {name}({parameters}) {{\n" + "\n".join(emitter.lines) + "\n}"
            if self.plan.autodiff_mode == "reverse":
                self.helper_sources[key] = "[Differentiable]\n" + self.helper_sources[key]
            self.helper_return_types[key] = name, result
        return (*self.helper_return_types[key], arguments)

    def visit_Call(self, node):
        name = qualified(node.func)
        if self.symbol(name) == "atomic_add":
            target, value = node.args
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in self.buffers:
                self.error(target, "atomic_add writes one buffer element")
            buffer = target.value.id
            address = self._address(buffer, target.slice)
            self.accesses[buffer] = 3
            return f"inx_atomic_add({buffer}, {address}, {self.types[buffer]}({self.visit(value)}))"
        if name == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id in self.buffers:
            return "inx_args." + node.args[0].id + "_shape_0"
        if name in self.helpers:
            symbol, _, arguments = self._helper(node)
            return symbol + "(" + ", ".join(self.visit(arg) for arg in arguments) + ")"
        if name in {"math.ceil", "math.floor", "math.trunc"}:
            return "int(" + super().visit_Call(node) + ")"
        return super().visit_Call(node)

    def declarations(self):
        fields = "\n".join("    " + kind + " " + name + ";" for kind, name in self.argument_fields)
        result = ["struct Arguments {\n" + fields + "\n};", "ConstantBuffer<Arguments> inx_args;", *self.constants]
        for name, access in self.accesses.items():
            prefix = "RW" if access & 2 else ""
            result.append(f"{prefix}ByteAddressBuffer {name};")
        for name, access in self.gradient_accesses.items():
            prefix = "RW" if access & 2 else ""
            result.append(f"{prefix}ByteAddressBuffer inx_grad_{name};")
        return result


class _HelperEmitter(SlangEmitter):
    def __init__(self, parent, function):
        super().__init__(ast.unparse(function), parent.path, aliases=parent.aliases, types=parent.types)
        self.parent = parent

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
                self.error(node, "Local VJP reads the current row; use an explicit tensor derivative for gathered work")
            return "inx_local_" + node.value.id
        return super().visit_Subscript(node)

    def assign(self, target, value):
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.buffers:
            if not isinstance(target.slice, ast.Name) or target.slice.id != self.plan.work_item.index_name:
                self.error(target, "Local VJP writes the current row")
            self.line("inx_local_" + target.value.id + " = " + value + ";")
        else:
            super().assign(target, value)


def _reverse_source(plan, params, function, primal):
    inputs = [name for name in primal.names if primal.accesses.get(name, 0) & 1]
    outputs = [name for name in primal.names if primal.accesses.get(name, 0) & 2]
    if set(inputs) & set(outputs) or not outputs:
        raise TypeError("Local VJP requires separate input and output buffers")
    local = _LocalEmitter(plan, params, primal.path, _function_source(function))
    local.line_offset = primal.line_offset
    local.declare_locals(plan.definition.body)
    local.local_declarations()
    for name in outputs:
        local.line("float inx_local_" + name + ";")
    local.block(plan.definition.body)
    local.line("LocalOutputs result;")
    for name in outputs:
        local.line("result." + name + " = inx_local_" + name + ";")
    local.line("return result;")
    index = plan.work_item.index_name
    # The row index is discrete; the scalar inputs and outputs carry tangents.
    parameters = [f"no_diff int {index}"]
    differentiated = []
    for name in inputs:
        position = primal.names.index(name)
        qualifier = "" if position in plan.gradient_parameters else "no_diff "
        parameters.append(qualifier + "float inx_local_" + name)
        if not qualifier:
            differentiated.append(name)
    fields = " ".join("float " + name + ";" for name in outputs)
    local_function = "struct LocalOutputs : IDifferentiable { " + fields + " };\n[Differentiable]\nLocalOutputs inx_primal(" + ", ".join(parameters) + ") {\n" + "\n".join(local.lines) + "\n}"
    primal.lines.clear()
    primal.accesses.clear()
    primal.types.update({"inx_local_" + name: "float" for name in inputs})
    index_node = ast.Name(id=index, ctx=ast.Load())
    arguments = [index]
    for name in inputs:
        address = primal._address(name, index_node)
        primal.accesses[name] = 1
        if name in differentiated:
            primal.line(f"DifferentialPair<float> inx_pair_{name} = diffPair({name}.Load<float>({address}), 0.0);")
            arguments.append("inx_pair_" + name)
        else:
            arguments.append(f"{name}.Load<float>({address})")
    primal.line("LocalOutputs.Differential seed;")
    for name in outputs:
        position = primal.names.index(name)
        seed = "0.0"
        if position in plan.gradient_parameters:
            address = primal._address(name, index_node).replace("_byte_offset", "_grad_byte_offset")
            primal.gradient_accesses[name] = 1
            seed = f"inx_grad_{name}.Load<float>({address})"
        primal.line(f"seed.{name} = {seed};")
    primal.line("bwd_diff(inx_primal)(" + ", ".join([*arguments, "seed"]) + ");")
    for name in differentiated:
        address = primal._address(name, index_node).replace("_byte_offset", "_grad_byte_offset")
        primal.gradient_accesses[name] = 3
        primal.line(f"inx_grad_{name}.Store<float>({address}, inx_grad_{name}.Load<float>({address}) + inx_pair_{name}.d);")
    primal.helper_sources.update(local.helper_sources)
    return local_function


_ARITHMETIC = """
int inx_floor_div(int a, int b) { int q = a / b; int r = a % b; return q - int(r != 0 && (r < 0) != (b < 0)); }
uint inx_floor_div(uint a, uint b) { return a / b; }
float inx_floor_div(float a, float b) { return floor(a / b); }
int inx_mod(int a, int b) { int r = a % b; return r != 0 && (r < 0) != (b < 0) ? r + b : r; }
uint inx_mod(uint a, uint b) { return a % b; }
float inx_mod(float a, float b) { return a - floor(a / b) * b; }
int inx_atomic_add(RWByteAddressBuffer buffer, uint address, int value) {
    int original; buffer.InterlockedAdd(address, value, original); return original;
}
uint inx_atomic_add(RWByteAddressBuffer buffer, uint address, uint value) {
    uint original; buffer.InterlockedAdd(address, value, original); return original;
}
float inx_atomic_add(RWByteAddressBuffer buffer, uint address, float value) {
    uint original = buffer.Load<uint>(address);
    for (;;) {
        uint assumed = original;
        buffer.InterlockedCompareExchange(address, assumed, asuint(asfloat(assumed) + value), original);
        if (original == assumed) return asfloat(assumed);
    }
}
"""


def _source(plan, params, function):
    emitter = _KernelEmitter(plan, params, str(function.__code__.co_filename), _function_source(function))
    emitter.line_offset = function.__code__.co_firstlineno - 1
    emitter.declare_locals(plan.definition.body)
    emitter.local_declarations()
    emitter.block(plan.definition.body)
    derivative = _reverse_source(plan, params, function, emitter) if plan.autodiff_mode == "reverse" else ""
    index = plan.work_item.index_name
    domain = plan.work_item.domain_name
    first = "inx_args.inx_range_begin" if plan.continuation.eligible else "0"
    last = "inx_args.inx_range_end" if plan.continuation.eligible else f"inx_args.{domain}_shape_0"
    entry = f'''[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
    if (tid.x >= uint(({last}) - ({first}))) return;
    int {index} = int(tid.x) + ({first});
'''
    code = "\n".join([*emitter.declarations(), _ARITHMETIC, *emitter.helper_sources.values(), derivative, entry, *emitter.lines, "}"])
    return code, emitter


def _artifact(compiled, emitter, plan, params, function):
    from infernux.compute import Buffer

    resources = {item["name"]: item for item in compiled["resources"]}
    arguments = resources["inx_args"]["value"]
    fields = {field["name"]: field for field in arguments["fields"]}
    layouts = []
    for index, (name, value) in enumerate(zip(emitter.names, params)):
        if isinstance(value, Buffer):
            layout = {"kind": "external_buffer", "shape_offsets": [fields[f"{name}_shape_{axis}"]["offset"] for axis in range(len(value.shape))],
                            "byte_offset_offset": fields[name + "_byte_offset"]["offset"], "byte_offset_width": 4,
                            "needs_grad": index in plan.gradient_parameters}
            if index in plan.gradient_parameters:
                layout["grad_byte_offset_offset"] = fields[name + "_grad_byte_offset"]["offset"]
            layouts.append(layout)
        else:
            layouts.append({"kind": "int32" if emitter.types[name] == "int" else "float32", "offset": fields[name]["offset"]})
    if plan.continuation.eligible:
        layouts.extend({"kind": "int32", "offset": fields[name]["offset"]} for name in ("inx_range_begin", "inx_range_end"))
    bindings = [{"binding": resources["inx_args"]["binding"], "binding_type": "uniform", "resource_kind": "arguments", "access": 1}]
    for name, access in emitter.accesses.items():
        bindings.append({"binding": resources[name]["binding"], "binding_type": "storage" if access & 2 else "read_only_storage",
                         "resource_kind": "external_buffer", "argument_indices": [emitter.names.index(name)], "is_grad": False, "access": access})
    for name, access in emitter.gradient_accesses.items():
        bindings.append({"binding": resources["inx_grad_" + name]["binding"], "binding_type": "storage" if access & 2 else "read_only_storage",
                         "resource_kind": "external_buffer", "argument_indices": [emitter.names.index(name)], "is_grad": True, "access": access})
    entry = compiled["entries"][0]
    metadata = {"entry_point": entry["name"], "threads_per_group": entry["workgroup_size"][0], "buffer_bindings": bindings,
                "task_type": "range_for", "range": {"begin_source": "constant", "begin": 0, "const_begin": True,
                                                       "const_end": False, "end_source": "arguments"}}
    identity = f"{function.__module__}.{function.__qualname__}"
    if plan.continuation.inner_range is not None:
        identity += "#" + plan.continuation.inner_range.kind
    return CompilerArtifact(spirv_tasks=(entry["code"],), task_metadata=(metadata,), domain_parameter=plan.domain_parameter,
        argument_layout={"size": arguments["byte_size"], "parameters": layouts},
        required_capabilities={"shader_int64": any(isinstance(value, Buffer) and value._dtype.scalar.name == "int64" for value in params)},
        diagnostic_locations=({"entry_point": entry["name"], "path": str(function.__code__.co_filename),
                               "line": function.__code__.co_firstlineno, "column": 0,
                               "function": identity},),
        continuation=plan.continuation, fast_math=WORK_ITEM_FAST_MATH, autodiff_mode=plan.autodiff_mode,
        gradient_parameters=plan.gradient_parameters, identity=plan.artifact_key)


def compile_kernel(function, params, *, receiver_fields=None, autodiff_mode="none", gradient_parameters=(), _lowered=None):
    from infernux.lib import _Infernux

    plan = prepare_kernel(function, params, receiver_fields=receiver_fields, autodiff_mode=autodiff_mode,
                          gradient_parameters=gradient_parameters, _lowered=_lowered)
    root = compiler_cache_root()
    path = root / (plan.artifact_key + ".inxgpu")
    specialization = _specialization_identity(params, autodiff_mode=autodiff_mode, gradient_parameters=plan.gradient_parameters)
    cached = _decode_artifact(path, specialization=specialization)
    if cached is not None:
        return cached
    compiled = _Infernux._compile_shader_program(plan.generated_source, str(function.__code__.co_filename), "inx_kernel", targets=["spirv"], precise=True)
    artifact = _artifact(compiled["targets"][0], plan.lowering, plan, params, function)
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_encode_artifact(artifact, specialization))
    os.replace(temporary, path)
    prune_cache_files(root, "*.inxgpu", file_limit=128, byte_limit=256 * 1024 * 1024)
    return artifact


