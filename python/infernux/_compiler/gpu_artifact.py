"""GPU program contracts and bytes, independent of the source compiler."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct

import numpy as np
from .kernel_contract import WORK_ITEM_FAST_MATH, ARRAY_KINDS

_CACHE_MAGIC = b"INXGPU\x02"
_CACHE_ABI = "infernux-gpu-ordered-inner-f32"

def parameter_key(params) -> tuple:
    """Return the bounded specialization key; runtime sizes are not compiled in."""
    from infernux.compute import Buffer

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

def _require_separate_f32_ops(payload):
    """Check final compiler bytes once, including cache/AOT, outside execution.

    This finite arithmetic contract is not a general SPIR-V validator. Numeric
    opcodes/decorations are from the vendored Khronos unified1 headers.
    """
    if len(payload) < 20 or len(payload) % 4:
        raise ValueError("Invalid GPU artifact SPIR-V word stream")
    words = struct.unpack(f"<{len(payload) // 4}I", payload)
    if words[0] != 0x07230203:
        raise ValueError("Invalid GPU artifact SPIR-V magic")
    instructions = []
    cursor = 5
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        if not size or cursor + size > len(words):
            raise ValueError("Truncated GPU artifact SPIR-V instruction")
        instructions.append((opcode, words[cursor + 1:cursor + size]))
        cursor += size
    f32 = {args[0] for op, args in instructions if op == 22 and args[1] == 32}  # OpTypeFloat
    if f32 and not any(op == 16 and args[1:] == (4461, 32) for op, args in instructions):
        raise ValueError("GPU f32 artifact must preserve signed zero, infinity and NaN; recompile this model")
    f32.update(args[0] for op, args in instructions if op == 23 and args[1] in f32)  # OpTypeVector
    glsl = {args[0] for op, args in instructions if op == 11 and struct.pack(
        f"<{len(args) - 1}I", *args[1:]).split(b"\0", 1)[0] == b"GLSL.std.450"}
    precise = {args[0] for op, args in instructions if op == 71 and args[1] == 42}  # NoContraction
    for op, args in instructions:
        if op == 74 and args[0] in precise:  # OpGroupDecorate
            precise.update(args[1:])
    for op, args in instructions:
        if op in (129, 131, 133, 136) and args[0] in f32 and args[1] not in precise:
            raise ValueError(f"GPU f32 no-contraction contract missing on arithmetic result %{args[1]}")
        if op == 12 and args[0] in f32 and args[2] in glsl and args[3] == 50:  # GLSL Fma
            raise ValueError("GPU f32 no-contraction contract cannot contain fused multiply-add")
        if ((op == 71 and (args[1] == 0 or (args[1] == 40 and args[2] != 0)))
                or (op == 72 and args[2] == 0)):
            raise ValueError("GPU f32 no-contraction contract cannot relax arithmetic precision")

@dataclass(frozen=True, slots=True)
class RangeContract:
    begin_parameter: int | None
    end_parameter: int | None
    input_parameters: tuple[int, ...]
    output_parameters: tuple[int, ...]
    operation_cost: int
    diagnostics: tuple[str, ...]
    # Every executed row writes these logical output views. This is not proof
    # that a partial dispatch initialized the whole view or its allocation.
    fully_written_output_parameters: tuple[int, ...]
    cost_source: str = "hir_ast_weight"
    resource_extents: tuple = ()
    gathered: bool = False
    inner_range: object = None

    @property
    def eligible(self) -> bool:
        return self.begin_parameter is not None

    @classmethod
    def from_record(cls, record):
        from infernux.jit_hir import InnerRangeContract, ResourceExtent

        expected = {"begin_parameter", "end_parameter", "input_parameters", "output_parameters", "operation_cost", "diagnostics", "fully_written_output_parameters", "cost_source", "resource_extents", "gathered", "inner_range"}
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("GPU artifact omitted its complete range contract")
        return cls(record["begin_parameter"], record["end_parameter"], tuple(record["input_parameters"]),
                   tuple(record["output_parameters"]), record["operation_cost"], tuple(record["diagnostics"]),
                   tuple(record["fully_written_output_parameters"]), record["cost_source"],
                   tuple(ResourceExtent.from_record(item) for item in record["resource_extents"]), record["gathered"],
                   None if record["inner_range"] is None else InnerRangeContract(**record["inner_range"]))

@dataclass(slots=True)
class CompilerArtifact:
    spirv_tasks: tuple[bytes, ...]
    task_metadata: tuple[dict, ...]
    domain_parameter: int
    argument_layout: dict
    required_capabilities: dict
    diagnostic_locations: tuple[dict, ...]
    continuation: RangeContract
    fast_math: bool
    autodiff_mode: str = "none"
    gradient_parameters: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.fast_math) is not bool or self.fast_math is not WORK_ITEM_FAST_MATH:
            raise ValueError("GPU artifact disagrees with the fixed WorkItem floating-point policy")
        for task in self.spirv_tasks:
            _require_separate_f32_ops(task)
        layouts = self.argument_layout["parameters"]
        contract = self.continuation
        if (not isinstance(contract, RangeContract) or type(contract.operation_cost) is not int
                or contract.operation_cost < 0 or contract.cost_source != "hir_ast_weight"
                or type(contract.gathered) is not bool
                or any(not isinstance(value, str) or not value for value in contract.diagnostics)):
            raise ValueError("Invalid GPU artifact range contract")
        if contract.eligible:
            count = len(layouts) - 2
            if (contract.diagnostics or type(contract.begin_parameter) is not int or type(contract.end_parameter) is not int
                    or (contract.begin_parameter, contract.end_parameter) != (count, count + 1)
                    or [row["kind"] for row in layouts[-2:]] != ["int32", "int32"]
                    or not contract.output_parameters or contract.operation_cost <= 0
                    or self.autodiff_mode != "none" or self.gradient_parameters):
                raise ValueError("GPU artifact range ABI disagrees with its request/layout")
        else:
            count = len(layouts)
            if contract.end_parameter is not None or not contract.diagnostics:
                raise ValueError("GPU artifact requires an explicit continuation rejection")
        if type(self.domain_parameter) is not int or not 0 <= self.domain_parameter < count or layouts[self.domain_parameter]["kind"] != "external_buffer":
            raise ValueError("GPU artifact has an invalid execution domain parameter")
        inner = contract.inner_range
        if inner is not None:
            from infernux.jit_hir import InnerRangeContract

            if (not isinstance(inner, InnerRangeContract) or not contract.eligible or not contract.gathered
                    or count < 5 or self.domain_parameter >= count - 4
                    or (inner.begin_parameter, inner.end_parameter) != (count - 2, count - 1)
                    or [row["kind"] for row in layouts[count - 4:count]] != ["external_buffer", "external_buffer", "int32", "int32"]):
                raise ValueError("GPU artifact inner range disagrees with its parameter layout")
            carry_in, carry_out = count - 4, count - 3
            if (carry_out in contract.input_parameters or carry_in in contract.output_parameters
                    or (inner.kind == "initial" and (carry_in in contract.input_parameters
                        or contract.output_parameters != (carry_out,)))
                    or (inner.kind == "resume" and (carry_in not in contract.input_parameters
                        or contract.output_parameters != (carry_out,)))
                    or (inner.kind == "final" and (carry_in not in contract.input_parameters
                        or any(index >= count - 4 for index in contract.output_parameters)))
                    or contract.fully_written_output_parameters != contract.output_parameters
                    or any(row["function"].rsplit("#", 1)[-1] != inner.kind for row in self.diagnostic_locations)):
                raise ValueError("GPU artifact inner stage disagrees with its read/write roles")
        for indices in (contract.input_parameters, contract.output_parameters, contract.fully_written_output_parameters):
            if (tuple(sorted(set(indices))) != indices or
                    any(type(index) is not int or not 0 <= index < count or layouts[index]["kind"] != "external_buffer"
                        for index in indices)):
                raise ValueError("GPU artifact has invalid range resource parameters")
        if (not set(contract.fully_written_output_parameters).issubset(contract.output_parameters)
                or (not contract.eligible and contract.fully_written_output_parameters)):
            raise ValueError("GPU artifact has invalid fully written range outputs")
        if contract.eligible and set(contract.input_parameters) & set(contract.output_parameters):
            raise ValueError("GPU continuation outputs cannot be inputs")
        if contract.eligible:
            from infernux.jit_hir import ResourceExtent

            expected_parameters = tuple(index for index, layout in enumerate(layouts[:count]) if layout["kind"] == "external_buffer")
            if (tuple(item.parameter for item in contract.resource_extents) != expected_parameters
                    or any(ResourceExtent.from_record(asdict(item)) != item for item in contract.resource_extents)
                    or any(not item.bounds or not item.upper_indices for item in contract.resource_extents
                           if item.parameter in (*contract.input_parameters, *contract.output_parameters))
                    or (inner is None and any(item.exact_domain == contract.gathered for item in contract.resource_extents))
                    or (inner is not None and any(not item.exact_domain for item in contract.resource_extents
                        if item.parameter in (*contract.output_parameters, count - 4, count - 3)))
                    or (contract.gathered and len(contract.output_parameters) != 1)):
                raise ValueError("GPU artifact omitted or changed its complete resource extent proof")
        elif contract.resource_extents or contract.gathered:
            raise ValueError("rejected GPU continuation cannot carry resource extents")
        if (self.autodiff_mode not in {"none", "reverse"}
                or any(type(index) is not int or not 0 <= index < len(layouts)
                       for index in self.gradient_parameters)
                or tuple(sorted(set(self.gradient_parameters))) != self.gradient_parameters):
            raise ValueError("Invalid GPU artifact differentiation request")
        gradient_layouts = []
        for index, layout in enumerate(layouts):
            if layout["kind"] != "external_buffer":
                continue
            if layout.get("byte_offset_width") != 4 or type(layout.get("needs_grad")) is not bool:
                raise ValueError("GPU artifact requires the current 32-bit buffer ABI")
            if layout["needs_grad"]:
                if "grad_byte_offset_offset" not in layout:
                    raise ValueError("GPU artifact omitted the gradient byte offset")
                gradient_layouts.append(index)
        if tuple(gradient_layouts) != self.gradient_parameters or (self.autodiff_mode == "reverse" and not gradient_layouts):
            raise ValueError("GPU artifact gradient layout disagrees with its differentiation request")
        if len(self.spirv_tasks) != len(self.task_metadata):
            raise ValueError("GPU artifact task metadata does not match its SPIR-V")
        if contract.eligible:
            if (len(self.task_metadata) != 1 or self.task_metadata[0]["task_type"] != "range_for"
                    or self.task_metadata[0]["range"]["begin_source"] != "constant"
                    or self.task_metadata[0]["range"]["begin"] != 0
                    or not self.task_metadata[0]["range"]["const_begin"]
                    or self.task_metadata[0]["range"]["const_end"]
                    or self.task_metadata[0]["range"]["end_source"] != "arguments"):
                raise ValueError("GPU continuation requires one constant-zero/arguments range task")
        for task in self.task_metadata:
            for binding in task["buffer_bindings"]:
                if contract.eligible and binding["resource_kind"] not in {"arguments", "external_buffer"}:
                    raise ValueError("GPU continuation cannot depend on temporary/global resources")
                if binding["resource_kind"] == "external_buffer":
                    if type(binding.get("is_grad")) is not bool:
                        raise ValueError("GPU artifact omitted external buffer gradient role")
                    indices = binding["argument_indices"]
                    if len(indices) != 1 or not 0 <= indices[0] < len(layouts):
                        raise ValueError("GPU artifact has an invalid external buffer parameter")
                    if binding["is_grad"] and indices[0] not in self.gradient_parameters:
                        raise ValueError("GPU artifact binds an undeclared gradient resource")
                    if contract.eligible:
                        expected_access = 1 if indices[0] in contract.input_parameters else 2 if indices[0] in contract.output_parameters else None
                        if expected_access is None or binding["access"] != expected_access or binding["is_grad"]:
                            raise ValueError("GPU continuation resource access disagrees with its row proof")

    def range_bounds(self, params, *, begin=None, end=None, host=None, _validate_resources=True):
        from infernux.compute import Buffer

        contract = self.continuation
        count = len(self.argument_layout["parameters"]) - (2 if contract.eligible else 0)
        if len(params) != count:
            raise RuntimeError("GPU compiler argument layout does not match the kernel parameters")
        inner = contract.inner_range
        if inner is not None:
            inner.validate(params[inner.begin_parameter], params[inner.end_parameter])
        requested = begin is not None or end is not None
        if not contract.eligible:
            if requested:
                raise TypeError("GPU kernel cannot continue: " + "; ".join(contract.diagnostics))
            return None
        domain = params[self.domain_parameter]
        if not isinstance(domain, Buffer):
            raise TypeError("GPU continuation requires a buffer execution domain")
        size = domain.element_count
        if size > 0x7fffffff:
            raise ValueError("GPU continuation exceeds the i32 logical index range")
        if not requested:
            begin, end = 0, size
        for name, value in (("begin", begin), ("end", end)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"GPU continuation {name} must be an integer, not bool")
        begin, end = int(begin), int(end)
        if not 0 <= begin <= end <= size:
            raise ValueError(f"GPU continuation range [{begin},{end}) is outside [0,{size})")
        if not _validate_resources:
            return begin, end
        self.validate_resources(params, size, host=host)
        return begin, end

    def validate_resources(self, params, size, *, host):
        from infernux.compute import Buffer

        contract = self.continuation
        if not contract.eligible:
            return
        inner = contract.inner_range
        domain = params[self.domain_parameter]
        buffers = {}
        owner = domain._host if host is None else host
        owner_identity = None if owner is None else owner.identity
        for index, value in enumerate(params):
            if isinstance(value, Buffer):
                if index in contract.output_parameters:
                    value._require_writable()
                else:
                    value._require_open()
                if (value.device != "gpu" or len(value.shape) != 1 or str(value._dtype.scalar) not in ARRAY_KINDS
                        or value._dtype.lanes != 1):
                    raise ValueError("GPU continuation requires one-dimensional float32/int32/uint32/int64 buffers")
                if value._host is None or value._host.identity != owner_identity:
                    raise ValueError("GPU continuation buffers must belong to the executable's host")
                native = value._native
                if native is None:
                    raise ValueError("GPU continuation requires a native resource identity")
                offset = value._byte_offset
                buffers[index] = ((native.resource_index, native.resource_generation), offset, offset + value.nbytes)
            elif inner is not None and index in (inner.begin_parameter, inner.end_parameter):
                pass  # Exact i32 and stage bounds were checked before any early return/cache lookup.
            elif self.argument_layout["parameters"][index]["kind"] == "int32":
                if (not isinstance(value, (int, np.integer))
                        or not -0x80000000 <= int(value) <= 0x7fffffff):
                    raise TypeError("GPU continuation integer scalar requires an exact signed int32")
            elif not isinstance(value, (float, np.floating)):
                raise TypeError("GPU continuation scalar parameters must be floating values")
        for extent in contract.resource_extents:
            extent.validate(size, params[extent.parameter].element_count)
        for index in contract.output_parameters:
            resource, left, right = buffers[index]
            for other_index, (other_resource, other_left, other_right) in buffers.items():
                if index != other_index and resource == other_resource and left < other_right and other_left < right:
                    raise ValueError("GPU continuation output overlaps another buffer range")

    def argument_bytes(self, params, *, bounds, gradients=None, host=None) -> bytes:
        from infernux.compute import Buffer

        _compilation_request(params, self.autodiff_mode, self.gradient_parameters)
        gradients = {} if gradients is None else dict(gradients)
        if any(type(index) is not int for index in gradients) or set(gradients) != set(self.gradient_parameters):
            raise ValueError("Gradient bindings must exactly match the compiled gradient parameters")
        layouts = tuple(self.argument_layout["parameters"])
        if bounds is not None:
            params = (*params, *bounds)
        if len(layouts) != len(params):
            raise RuntimeError("GPU compiler argument layout does not match the kernel parameters")
        payload = bytearray(int(self.argument_layout["size"]))
        resources = []
        owner = host
        if self.gradient_parameters and owner is None:
            owner = params[self.domain_parameter]._host
        owner_identity = None if owner is None else owner.identity

        def buffer_offset(value):
            value._require_open()
            if value.device != "gpu":
                raise TypeError("Compiled buffer bindings require GPU buffers")
            offset = value._byte_offset
            end = offset + value.nbytes
            # The logical-pointer shader ABI performs signed i32 byte indexing.
            if offset < 0 or offset % value._dtype.scalar.itemsize or end > 0x7fffffff:
                raise ValueError("GPU buffer range exceeds the aligned signed-32-bit byte-offset ABI")
            if self.gradient_parameters:
                native = value._native
                if native is None or value._host is None or value._host.identity != owner_identity:
                    raise ValueError("Local VJP primal and gradient buffers must belong to the executable's host")
                resources.append(((native.resource_index, native.resource_generation), offset, end))
            return offset

        for index, (layout, value) in enumerate(zip(layouts, params)):
            kind = layout["kind"]
            if kind == "external_buffer":
                if not isinstance(value, Buffer):
                    raise TypeError("Compiled GPU buffer layout does not match the launch parameters")
                offsets = tuple(layout["shape_offsets"])
                if len(offsets) != len(value.shape):
                    raise RuntimeError("GPU compiler buffer rank does not match the launch parameters")
                for offset, size in zip(offsets, value.shape):
                    struct.pack_into("<i", payload, int(offset), int(size))
                if layout["byte_offset_width"] != 4:
                    raise RuntimeError("GPU compiler requires a 32-bit logical buffer offset")
                struct.pack_into(
                    "<I", payload, int(layout["byte_offset_offset"]), buffer_offset(value),
                )
                if layout["needs_grad"]:
                    gradient = gradients[index]
                    if not isinstance(gradient, Buffer) or gradient.shape != value.shape or gradient._dtype != value._dtype:
                        raise ValueError("Gradient buffers must match their primal buffer shape and dtype")
                    gradient._require_writable()
                    struct.pack_into("<I", payload, int(layout["grad_byte_offset_offset"]), buffer_offset(gradient))
            elif kind == "int32":
                struct.pack_into("<i", payload, int(layout["offset"]), int(value))
            elif kind == "float32":
                struct.pack_into("<f", payload, int(layout["offset"]), float(value))
            else:
                raise RuntimeError(f"Unsupported GPU compiler argument layout kind: {kind}")
        if self.autodiff_mode == "reverse":
            if any(value.shape != params[self.domain_parameter].shape for value in params):
                raise ValueError("Local VJP buffers must have the same one-dimensional domain")
            for position, (storage, begin, end) in enumerate(resources):
                if any(storage == other and begin < other_end and other_begin < end
                       for other, other_begin, other_end in resources[:position]):
                    raise ValueError("Local VJP primal and gradient buffer ranges must not overlap")
        return bytes(payload)

def _specialization_identity(params, *, autodiff_mode="none", gradient_parameters=()) -> str:
    return json.dumps({"parameters": parameter_key(params), "autodiff_mode": autodiff_mode,
                       "gradient_parameters": gradient_parameters, "fast_math": WORK_ITEM_FAST_MATH},
                      sort_keys=True, separators=(",", ":"))

def _compilation_request(params, autodiff_mode, gradient_parameters):
    from infernux.compute import Buffer

    if autodiff_mode not in {"none", "reverse"}:
        raise ValueError("GPU compiler autodiff_mode must be 'none' or 'reverse'")
    indices = tuple(gradient_parameters)
    if any(type(index) is not int or index < 0 or index >= len(params) for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Gradient parameters must be unique valid positional buffer indices")
    indices = tuple(sorted(indices))
    if autodiff_mode == "reverse" and not indices:
        raise ValueError("Local VJP requires explicit input-gradient and output-seed parameters")
    checked = range(len(params)) if autodiff_mode == "reverse" else indices
    for index in checked:
        value = params[index]
        if (not isinstance(value, Buffer) or value.device != "gpu" or len(value.shape) != 1
                or value._dtype.scalar != np.dtype(np.float32) or value._dtype.lanes != 1):
            raise TypeError("Local VJP parameters must be one-dimensional float32 GPU buffers")
    return indices

class ArtifactAbiMismatchError(RuntimeError):
    """An explicitly identified artifact ABI is not executable by this reader."""

def _artifact_header_bytes(payload, *, path="<memory>"):
    payload = memoryview(payload).cast("B")
    if len(payload) < len(_CACHE_MAGIC) + 4 or payload[:len(_CACHE_MAGIC)] != _CACHE_MAGIC:
        raise RuntimeError(f"Invalid Infernux GPU compiler cache artifact: {path}")
    header_size = struct.unpack_from("<I", payload, len(_CACHE_MAGIC))[0]
    header_begin = len(_CACHE_MAGIC) + 4
    header_end = header_begin + header_size
    if header_end > len(payload):
        raise RuntimeError(f"Truncated GPU compiler artifact header: {path}")
    return header_size


def _read_artifact_payload(payload, *, specialization: str | None = None, path="<memory>"):
    """Decode the same immutable tasks from cache bytes or private IO storage."""
    payload = memoryview(payload).cast("B")
    header_size = _artifact_header_bytes(payload, path=path)
    header_begin = len(_CACHE_MAGIC) + 4
    header_end = header_begin + header_size
    header = json.loads(payload[header_begin:header_end].tobytes())
    abi = header.get("abi")
    if not isinstance(abi, str) or not abi:
        raise RuntimeError(f"GPU compiler artifact omitted a valid ABI: {path}")
    if abi != _CACHE_ABI:
        raise ArtifactAbiMismatchError(f"Unsupported GPU compiler artifact ABI {abi}; recompile: {path}")
    fast_math = header.get("fast_math")
    if type(fast_math) is not bool or fast_math is not WORK_ITEM_FAST_MATH:
        raise RuntimeError(f"GPU artifact omitted or changed its floating-point policy: {path}")
    identity = json.loads(header["specialization"])
    if (identity["autodiff_mode"] != header["autodiff_mode"]
            or identity["gradient_parameters"] != header["gradient_parameters"]
            or type(identity.get("fast_math")) is not bool or identity["fast_math"] is not fast_math):
        raise RuntimeError(f"GPU artifact compilation identity mismatch: {path}")
    if specialization is not None and header.get("specialization") != specialization:
        raise RuntimeError(f"GPU compiler artifact specialization mismatch: {path}")
    tasks = []
    cursor = header_end
    for size in header["task_sizes"]:
        end = cursor + int(size)
        tasks.append(payload[cursor:end].tobytes())
        cursor = end
    if cursor != len(payload):
        raise RuntimeError(f"Invalid Infernux GPU compiler cache artifact size: {path}")
    artifact = CompilerArtifact(
        spirv_tasks=tuple(tasks),
        task_metadata=tuple(dict(item) for item in header["task_metadata"]),
        domain_parameter=int(header["domain_parameter"]),
        argument_layout=dict(header["argument_layout"]),
        required_capabilities=dict(header["required_capabilities"]),
        diagnostic_locations=tuple(
            dict(item) for item in header["diagnostic_locations"]
        ),
        continuation=RangeContract.from_record(header.get("continuation")),
        fast_math=fast_math,
        autodiff_mode=str(header["autodiff_mode"]),
        gradient_parameters=tuple(header["gradient_parameters"]),
    )
    return header, artifact


def _read_artifact(path: Path, *, specialization: str | None = None) -> tuple[dict, CompilerArtifact] | None:
    if not path.is_file():
        return None
    return _read_artifact_payload(path.read_bytes(), specialization=specialization, path=str(path))


def _decode_artifact(path: Path, *, specialization: str | None = None) -> CompilerArtifact | None:
    result = _read_artifact(path, specialization=specialization)
    return None if result is None else result[1]


def _encode_artifact(artifact, specialization):
    """One encoding for compiler cache, model assets and cold re-publication."""
    header = json.dumps({
        "abi": _CACHE_ABI,
        "fast_math": artifact.fast_math,
        "autodiff_mode": artifact.autodiff_mode,
        "gradient_parameters": artifact.gradient_parameters,
        "task_sizes": [len(task) for task in artifact.spirv_tasks],
        "task_metadata": artifact.task_metadata,
        "domain_parameter": artifact.domain_parameter,
        "argument_layout": artifact.argument_layout,
        "required_capabilities": artifact.required_capabilities,
        "diagnostic_locations": artifact.diagnostic_locations,
        "continuation": asdict(artifact.continuation),
        "specialization": specialization,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _CACHE_MAGIC + struct.pack("<I", len(header)) + header + b"".join(artifact.spirv_tasks)
