"""Compile authored event graphs and execute their generated queue body on Vulkan."""
from dataclasses import replace
import re
import struct

import pytest

from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.graph.types import TypeRef, ValueType
from infernux.lib import _Infernux as native
from infernux.particle import (
    GpuParticleGlslLowerer, ParticleEmitterAsset, ParticleEventField,
    ParticleEventFlow, ParticleEventType, ParticleGraphAsset, ParticleGraphCompiler,
    ParticleKernelLowerer, default_event_graph,
)
from infernux.particle import gpu_glsl_backend as backend
from infernux.particle.asset import particle_attribute_cache_id
from infernux.particle.nodes import particle_event_payload_port_id as payload_port


def _event_program(value, default, capacity=3):
    fields = (
        ParticleEventField("flag", "Flag", TypeRef(ValueType.BOOL), default),
        ParticleEventField("size", "Size", TypeRef(ValueType.F32), 2.5),
        ParticleEventField("strip", "Strip", TypeRef(ValueType.U32), 7),
        ParticleEventField("signed", "Signed", TypeRef(ValueType.I32), -5),
        ParticleEventField("velocity", "Velocity", TypeRef(ValueType.VEC3), [1., 2., 3.]),
    )
    properties = {"event": "event"}
    if value is not None:
        properties[payload_port("flag")] = value
    update = GraphDocument("particle.update", nodes=(
        GraphNodeRecord("root.update", "particle.root.update"),
        GraphNodeRecord("trigger", "particle.event.trigger", properties=properties),
    ), links=(GraphLinkRecord("trigger", "root.update", "out", "trigger", "in", PortKind.EXEC),))
    event = default_event_graph("event")
    nodes, links = list(event.nodes), list(event.links)
    previous = "root.event"
    for payload_field in fields:
        field = payload_field.stable_id
        nodes.append(GraphNodeRecord(field, "particle.attribute.cache", properties={
            "name": payload_field.name, "value_type": payload_field.value_type.value_type.value,
        }))
        links.extend((
            GraphLinkRecord(field + ".exec", previous, "out", field, "in", PortKind.EXEC),
            GraphLinkRecord(field + ".value", "root.event", payload_port(field), field, "value", PortKind.VALUE),
        ))
        previous = field
    event = replace(event, nodes=tuple(nodes), links=tuple(links))
    asset = ParticleGraphAsset(
        event_types=(ParticleEventType("event", "Event", capacity, fields),),
        emitters=(ParticleEmitterAsset(
            stable_id="emitter", update=update,
            event_flows=(ParticleEventFlow("consume", event),),
        ),),
    )
    assert ParticleGraphAsset.from_json(asset.canonical_json()) == asset
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(asset))
    return kernel, GpuParticleGlslLowerer().lower(kernel).emitters[0]


@pytest.mark.parametrize("value,default", [(True, False), (False, True), (None, True), (None, False)])
@pytest.mark.parametrize("capacity", [1, 3])
def test_authored_bool_event_compiles_every_stage(value, default, capacity):
    _, gpu = _event_program(value, default, capacity)
    compiled = native._compile_compute_glsl_batch(gpu.stages(), "event-payload-storage")
    assert set(compiled) == set(gpu.stages())
    assert all(bytes(spirv)[:4] == b"\x03\x02\x23\x07" for spirv in compiled.values())


def _queue_probe_source(kernel, gpu):
    """Run unmodified production-generated Update instructions in a small SSBO harness.

    This isolates FIFO storage/consumption, not the full particle scheduler or
    drawing path. Full production stages are compiled independently above.
    """
    emitter = kernel.emitters[0]
    fields = backend._attribute_fields(emitter)
    body, _ = backend._StageCompiler(emitter, fields, {}, kernel.events, 0).compile(emitter.update)
    assert body in gpu.update
    declaration = re.search(r"struct ParticleState \{.*?\};", gpu.update, re.S)
    assert declaration is not None
    return f"""#version 450
layout(local_size_x=1) in;
{declaration.group()}
layout(std430, set=0, binding=0) buffer State {{ ParticleState state; }};
layout(std430, set=0, binding=1) buffer Counters {{ uint event_counters[3]; }} counters;
struct Frame {{ float delta_time; }};
const Frame pc = Frame(0.0);
void main() {{
    bool particle_alive = true;
{body}
}}
""", {stable_id: (name, kind, offset, size) for stable_id, name, kind, offset, size in gpu.attribute_fields}


@pytest.mark.parametrize("value,default", [(True, False), (False, True), (None, True), (None, False)])
def test_generated_bool_event_queue_consumes_and_wraps_on_gpu(engine, value, default):
    kernel, gpu = _event_program(value, default)
    source, layout = _queue_probe_source(kernel, gpu)
    spirv = native._compile_compute_glsl_batch({"probe": source}, "event-payload-readback")["probe"]
    host = engine._acquire_compute_host()
    state = host.create_buffer(gpu.state_stride // 4, "uint32", 1)
    counters = host.create_buffer(3, "uint32", 1)
    program = host.create_kernel(spirv, buffer_binding_count=2)
    initial = bytearray(gpu.state_stride)
    expected = default if value is None else value
    # Opposite sentinel proves false is explicitly consumed, not merely zeroed.
    def cache(field):
        return particle_attribute_cache_id("event.consume", field)
    struct.pack_into("<f", initial, layout["builtin.lifetime"][2], 5.0)
    struct.pack_into("<I", initial, layout[cache("flag")][2], int(not expected))
    state.set_bytes(bytes(initial))
    counters.set_bytes(bytes(12))
    try:
        for step in range(1, 8):
            program.dispatch([state, counters])
            data = bytes(state.get_bytes(gpu.state_stride))
            def scalar(stable_id, kind="I"):
                return struct.unpack_from("<" + kind, data, layout[stable_id][2])[0]
            # Event begin takes its snapshot before this update's producer;
            # events queued in this step become consumable on the next step.
            assert scalar(cache("flag")) == int(not expected if step == 1 else expected)
            if step > 1:
                assert scalar(cache("size"), "f") == 2.5
                assert scalar(cache("strip")) == 7
                assert scalar(cache("signed"), "i") == -5
                assert struct.unpack_from("<3f", data, layout[cache("velocity")][2]) == (1., 2., 3.)
            assert scalar("internal.event.0.head") == (step - 1) % 3
            assert scalar("internal.event.0.tail") == step % 3
            assert scalar("internal.event.0.count") == 1
            assert scalar("internal.event.0.active") == 0
            assert struct.unpack("<3I", bytes(counters.get_bytes(12))) == (0, step, step - 1)
    finally:
        program.wait()
        program = None
        state = counters = host = None
