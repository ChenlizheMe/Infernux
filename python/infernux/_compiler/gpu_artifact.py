"""GPU program contracts and bytes, independent of the source compiler."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct

import numpy as np
from .kernel_contract import WORK_ITEM_FAST_MATH, ARRAY_KINDS

_CACHE_MAGIC = b"INXGPU\x02"
_CACHE_ABI = "infernux-slang-target-work-item"

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
    target: str
    tasks: tuple[bytes, ...]
    task_metadata: tuple[dict, ...]
    domain_parameter: int
    argument_layout: dict
    required_capabilities: dict
    diagnostic_locations: tuple[dict, ...]
    continuation: RangeContract
    fast_math: bool
    autodiff_mode: str = "none"
    gradient_parameters: tuple[int, ...] = ()
    identity: str = ""

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

def _specialization_identity(params, *, target="spirv", autodiff_mode="none", gradient_parameters=()) -> str:
    return json.dumps({"parameters": parameter_key(params), "autodiff_mode": autodiff_mode,
                       "gradient_parameters": gradient_parameters, "fast_math": WORK_ITEM_FAST_MATH, "target": target},
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
            or identity["target"] != header["target"]
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
        target=header["target"],
        tasks=tuple(tasks),
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
        identity=header["identity"],
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
        "identity": artifact.identity,
        "target": artifact.target,
        "task_sizes": [len(task) for task in artifact.tasks],
        "task_metadata": artifact.task_metadata,
        "domain_parameter": artifact.domain_parameter,
        "argument_layout": artifact.argument_layout,
        "required_capabilities": artifact.required_capabilities,
        "diagnostic_locations": artifact.diagnostic_locations,
        "continuation": asdict(artifact.continuation),
        "specialization": specialization,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _CACHE_MAGIC + struct.pack("<I", len(header)) + header + b"".join(artifact.tasks)
