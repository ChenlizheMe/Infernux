"""Authoring-space semantics and Vulkan execution of the actual lowered conversion."""
import struct

import numpy as np
import pytest

from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.graph.types import CoordinateSpace, TypeRef, ValueType
from infernux.lib import _Infernux as native
from infernux.particle import (
    EmitterSettings, GpuParticleGlslLowerer, ParticleEmitterAsset, ParticleGraphAsset,
    ParticleGraphCompiler, ParticleKernelLowerer, ParticleParameter, SimulationSpace,
    pack_gpu_particle_parameters,
)
from infernux.particle import gpu_glsl_backend as backend
from infernux.particle.kernel_ir import ParticleKernelFunction


CASES = [
    ("particle.collision.plane", "point", "position"),
    ("particle.collision.sphere", "center", "position"),
    ("particle.collision.plane", "normal", "direction"),
    ("particle.attribute.position", "value", "position"),
    ("particle.motion.target_position", "target", "position"),
]
WORLD_VALUE = [12., 3., 4.]


def _spatial_program(node_type, port):
    update = GraphDocument("particle.update", nodes=(
        GraphNodeRecord("root.update", "particle.root.update"),
        GraphNodeRecord("world", "particle.parameter", properties={"parameter": "world"}),
        GraphNodeRecord("operation", node_type),
    ), links=(
        GraphLinkRecord("exec", "root.update", "out", "operation", "in", PortKind.EXEC),
        GraphLinkRecord("value", "world", "value", "operation", port, PortKind.VALUE),
    ))
    asset = ParticleGraphAsset(
        parameters=(ParticleParameter(
            "world", "World", TypeRef(ValueType.VEC3, CoordinateSpace.WORLD), WORLD_VALUE,
        ),),
        emitters=(ParticleEmitterAsset(
            settings=EmitterSettings(simulation_space=SimulationSpace.LOCAL), update=update,
        ),),
    )
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(asset))
    conversion, = (item for item in kernel.emitters[0].update.instructions
                   if item.opcode == "convert_space" and item.source.node_uid == "operation")
    return kernel, conversion


@pytest.mark.parametrize("node_type,port,semantic", CASES)
def test_spatial_inputs_compile_with_their_declared_semantics(node_type, port, semantic):
    kernel, conversion = _spatial_program(node_type, port)
    assert conversion.immediate_dict()["semantic"] == semantic
    stages = GpuParticleGlslLowerer().lower(kernel).emitters[0].stages()
    compiled = native._compile_compute_glsl_batch(stages, "particle-collision-space")
    assert all(bytes(value)[:4] == b"\x03\x02\x23\x07" for value in compiled.values())


def _conversion_probe(kernel, conversion):
    # Slice only the input dependencies of the production collision operand.
    # No reimplementation of conversion semantics or generated expressions.
    emitter = kernel.emitters[0]
    wanted = {conversion.result_id}
    selected = []
    for instruction in reversed(emitter.update.instructions):
        if instruction.result_id in wanted:
            selected.append(instruction)
            wanted.update(operand.value_id for operand in instruction.operands)
    function = ParticleKernelFunction(emitter.update.stage, tuple(reversed(selected)), (), ())
    slots, _ = backend._parameter_slot_layout(kernel.parameters)
    compiler = backend._StageCompiler(
        emitter, backend._attribute_fields(emitter), {}, kernel.events, 0, parameter_slots=slots,
    )
    body, _ = compiler.compile(function)
    result = compiler._values[conversion.result_id]
    return f"""#version 450
layout(local_size_x=1) in;
layout(std430, set=0, binding=0) buffer Result {{ vec4 result; }};
layout(std430, set=0, binding=1) readonly buffer Parameters {{ uvec4 parameter_words[]; }};
layout(std430, set=0, binding=2) readonly buffer Transforms {{
    mat4 emitter_to_world;
    mat4 world_to_emitter;
    mat4 simulation_to_world;
    mat4 world_to_simulation;
}} transforms;
void main() {{
{body}
    result = vec4({result}, 1.0);
}}
"""


def _transform(case):
    rotation = np.eye(3, dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    if case in {"rotation", "combined"}:
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
    if case in {"translation", "combined"}:
        translation = np.array([10., -2., 5.], dtype=np.float32)
    if case == "scale":
        scale = np.array([2., 3., 4.], dtype=np.float32)
    if case == "combined":
        scale *= 2.
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix, rotation


@pytest.mark.parametrize("case", ["identity", "translation", "rotation", "scale", "combined"])
@pytest.mark.parametrize("node_type,port,semantic", CASES)
def test_collision_operand_conversion_executes_on_gpu(engine, node_type, port, semantic, case):
    kernel, conversion = _spatial_program(node_type, port)
    source = _conversion_probe(kernel, conversion)
    spirv = native._compile_compute_glsl_batch({"conversion": source}, "particle-space-readback")["conversion"]
    matrix, rotation = _transform(case)
    inverse = np.linalg.inv(matrix).astype(np.float32)
    expected = ((inverse @ np.array([*WORLD_VALUE, 1.], dtype=np.float32))[:3]
                if semantic == "position" else rotation.T @ np.asarray(WORLD_VALUE, dtype=np.float32))
    host = engine._acquire_compute_host()
    result = host.create_buffer(4, "float32", 1)
    words = pack_gpu_particle_parameters(kernel.parameters)
    parameters = host.create_buffer(len(words), "uint32", 1)
    transforms = host.create_buffer(64, "float32", 1)
    program = host.create_kernel(spirv, buffer_binding_count=3)
    parameters.set_bytes(struct.pack(f"<{len(words)}I", *words))
    transforms.set_bytes(b"".join(m.tobytes(order="F") for m in (matrix, inverse, matrix, inverse)))
    try:
        program.dispatch([result, parameters, transforms])
        observed = struct.unpack("<4f", bytes(result.get_bytes(16)))
        np.testing.assert_allclose(observed[:3], expected, rtol=1e-6, atol=1e-6)
        assert observed[3] == 1.0
    finally:
        program.wait()
        program = None
        result = parameters = transforms = host = None
