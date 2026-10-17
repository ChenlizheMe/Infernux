from __future__ import annotations
import copy
import json as json
from dataclasses import replace
from pathlib import Path as Path
import threading as threading
import pytest
from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.particle import (
    EmitterSettings,
    EmitterShape as EmitterShape,
    GpuParticleGlslLowerer,
    KernelCompileError,
    MeshEmissionMode as MeshEmissionMode,
    ParticleCompileError,
    ParticleEmitterAsset,
    ParticleAttribute,
    ParticleEventField,
    ParticleEventFlow,
    ParticleEventType,
    ParticleGraphAsset,
    ParticleGraphCompiler,
    ParticleGraphSchemaError,
    ParticleParameter,
    ParticleKernelLowerer,
    ParticleStage,
    ParticleScriptCompiler,
    ParticleScriptError,
    ParticleArtifactError as ParticleArtifactError,
    ParticleArtifactSuperseded as ParticleArtifactSuperseded,
    ParticleArtifactRegistry as ParticleArtifactRegistry,
    ParticleBurst as ParticleBurst,
    ParticleRuntimeMetadataError,
    SdfVolume,
    SimulationSpace,
    VectorField,
    decode_particle_runtime_metadata,
    compile_gpu_particle_spirv,
    validate_gpu_particle_spirv,
    default_event_graph,
)
from infernux.graph import AssetReference, CoordinateSpace, TypeRef, ValueType
from infernux.particle.nodes import (
    PARTICLE_EVENT_ACTIVE_TYPE_ID,
    PARTICLE_EVENT_TRIGGER_TYPE_ID,
    particle_graph_node_definitions,
)
from infernux.particle.asset import (
    particle_attribute_cache_id,
    particle_attribute_capture_id,
    particle_cache_attributes,
)


def _operations(stage):
    return tuple(stage.flow.iter_operations())


def _collision_lifecycle_graph(stage: str, size: float):
    operation_uid = f"{stage.removeprefix('collision_')}.size"
    return GraphDocument(
        f"particle.{stage}",
        nodes=(
            GraphNodeRecord(f"root.{stage}", f"particle.root.{stage}"),
            GraphNodeRecord(
                operation_uid,
                "particle.attribute.size",
                properties={"value": size},
            ),
        ),
        links=(
            GraphLinkRecord(
                f"{stage}.exec",
                f"root.{stage}",
                "out",
                operation_uid,
                "in",
                PortKind.EXEC,
            ),
        ),
    )


def _collision_lifecycle_stages():
    return {
        "collision_enter": _collision_lifecycle_graph("collision_enter", 2.0),
        "collision_stay": _collision_lifecycle_graph("collision_stay", 3.0),
        "collision_exit": _collision_lifecycle_graph("collision_exit", 1.0),
    }


def _event_trigger_stage(event_id: str) -> GraphDocument:
    return GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "event.trigger",
                PARTICLE_EVENT_TRIGGER_TYPE_ID,
                properties={"event": event_id, "condition": True},
            ),
        ),
        links=(
            GraphLinkRecord(
                "event.stream",
                "root.update",
                "out",
                "event.trigger",
                "in",
                PortKind.EXEC,
            ),
        ),
    )


PARTICLE_SCRIPT_SOURCE = """\
from infernux.particle import AssetReference, ParticleScript, ParticleEmitter, EmitterSettings, VectorField

class SmokeGraph(ParticleScript):
    stable_id = "smoke-graph"

    class Smoke(ParticleEmitter):
        stable_id = "smoke"
        settings = EmitterSettings(
            capacity=100000,
            simulation_space="world",
            spawn_rate=20000.0,
            spawn_rate_over_distance=3.0,
            duration=4.0,
            loop=False,
            start_delay=0.25,
        )
        data_interfaces = (
            VectorField(
                stable_id="wind-field",
                name="Wind",
                texture=AssetReference(path_hint="Assets/Fields/Wind.vectorfield"),
                space="world",
                boundary="repeat",
            ),
        )

        def init(self, ctx, particles):
            particles.set_velocity((0.0, 1.0, 0.0))
            particles.set_lifetime(6.0)
            particles.set_rotation(0.25)

        def update(self, ctx, particles):
            particles .add_velocity((0.0, -0.2, 0.0))
            particles .add_rotation(180.0)

        def rendering(self, ctx, particles):
            particles.set_lifetime(8.0)
            particles.set_flipbook_frame(particles.normalized_age * 63.0)
            particles.sprite(
                shader="Particle Six-Way Smoke",
                receive_scene_lighting=True,
                receive_shadows=True,
                sort="back_to_front",
                alignment="velocity",
            )
"""


PARTICLE_SCRIPT_EVENT_SOURCE = """\
from infernux.particle import (
    ParticleScript, ParticleEmitter, EmitterSettings,
    EventField, EventType, event,
)

class ImpactGraph(ParticleScript):
    stable_id = "impact-graph"
    event_types = (
        EventType(
            stable_id="impact",
            name="Impact",
            queue_capacity=32,
            fields=(
                EventField("energy", "Energy", "f32", 1.0),
                EventField("tint", "Tint", "color", (1.0, 1.0, 1.0, 1.0)),
            ),
        ),
    )
    class Source(ParticleEmitter):
        stable_id = "source"
        settings = EmitterSettings(capacity=1024)

        def init(self, ctx, particles):
            particles.set_size(2.0)

        def update(self, ctx, particles):
            particles.trigger_event(
                event="impact",
                condition=particles.age >= particles.lifetime,
                payload={
                    "energy": particles.size,
                    "tint": particles.color,
                },
            )

        def rendering(self, ctx, particles):
            particles.sprite()

        @event("impact")
        def on_impact(self, ctx, particles):
            particles.set_size(ctx.event_payload(
                field="energy"
            ))
            particles.set_color(ctx.event_payload(
                field="tint"
            ))
"""


def test_particle_exec_parallel_writes_require_explicit_order_or_merge():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("first", "particle.attribute.size"),
            GraphNodeRecord("second", "particle.attribute.size"),
        ),
        links=(
            GraphLinkRecord(
                "exec-first",
                "root.update",
                "out",
                "first",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "exec-second",
                "root.update",
                "out",
                "second",
                "in",
                PortKind.EXEC,
            ),
        ),
    )

    with pytest.raises(
        ParticleCompileError,
        match="parallel particle Exec branches have an unordered state dependency.*builtin.size",
    ):
        ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(ParticleEmitterAsset(update=update),)))


def test_particle_exec_parallel_read_write_dependency_requires_order_or_join():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("write-position", "particle.attribute.position"),
            GraphNodeRecord(
                "read-position",
                "particle.attribute.get",
                properties={"attribute": "builtin.position"},
            ),
            GraphNodeRecord("accelerate", "particle.attribute.velocity"),
        ),
        links=(
            GraphLinkRecord(
                "exec-write",
                "root.update",
                "out",
                "write-position",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "exec-read",
                "root.update",
                "out",
                "accelerate",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "value-read",
                "read-position",
                "value",
                "accelerate",
                "value",
                PortKind.VALUE,
            ),
        ),
    )

    with pytest.raises(
        ParticleCompileError,
        match="unordered state dependency.*builtin.position",
    ):
        ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(ParticleEmitterAsset(update=update),)))


def test_particle_exec_join_all_continues_after_every_parallel_branch():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("left", "particle.attribute.size"),
            GraphNodeRecord("right", "particle.attribute.position"),
            GraphNodeRecord("join", "particle.control.join_all"),
            GraphNodeRecord("tail", "particle.lifecycle.kill_if"),
        ),
        links=(
            GraphLinkRecord("root-left", "root.update", "out", "left", "in", PortKind.EXEC),
            GraphLinkRecord("root-right", "root.update", "out", "right", "in", PortKind.EXEC),
            GraphLinkRecord("left-join", "left", "out", "join", "in0", PortKind.EXEC),
            GraphLinkRecord("right-join", "right", "out", "join", "in1", PortKind.EXEC),
            GraphLinkRecord("join-tail", "join", "out", "tail", "in", PortKind.EXEC),
        ),
    )

    program = ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(ParticleEmitterAsset(update=update),)))
    stage = program.emitters[0].update
    schedule = [block.node_uid for block in stage.flow.blocks if block.node_uid != stage.flow.entry_node_uid]

    assert schedule.index("join") > schedule.index("left")
    assert schedule.index("join") > schedule.index("right")
    assert schedule.index("tail") > schedule.index("join")
    assert len(stage.flow.joins) == 1
    join = stage.flow.joins[0]
    assert join.node_uid == "join"
    assert len(join.input_lane_indices) == 2
    assert join.output_lane_index not in join.input_lane_indices
    kernel_program = ParticleKernelLowerer().lower(program)
    kernel_emitter = kernel_program.emitters[0]
    kernel = kernel_emitter.update
    assert all(instruction.source.node_uid != "join" for instruction in kernel.instructions)
    update_flow = next(flow for flow in kernel_emitter.flows if flow.lifecycle_stage is ParticleStage.UPDATE)
    assert [lane.index for lane in update_flow.lanes] == list(range(len(update_flow.lanes)))
    assert update_flow.lanes[0].parent_index == -1
    assert all(0 <= lane.parent_index < lane.index for lane in update_flow.lanes[1:])
    blocks = {block.source_node_uid: block for block in update_flow.blocks}
    assert blocks["left"].lane_index != blocks["right"].lane_index
    assert blocks["join"].instruction_begin == blocks["join"].instruction_end
    assert blocks["tail"].instruction_begin >= blocks["join"].instruction_end
    assert [block.source_node_uid for block in update_flow.blocks] == [
        "root.update",
        "left",
        "right",
        "join",
        "tail",
    ]
    assert len(update_flow.joins) == 1
    kernel_join = update_flow.joins[0]
    assert kernel_join.source_node_uid == "join"
    assert set(kernel_join.input_lane_indices) == {
        blocks["left"].lane_index,
        blocks["right"].lane_index,
    }
    assert kernel_join.output_lane_index == blocks["join"].lane_index

    restored = type(kernel_program).from_dict(kernel_program.to_dict())
    assert restored == kernel_program
    assert restored.kernel_hash == kernel_program.kernel_hash

    stale = kernel_program.to_dict()
    stale["emitters"][0].pop("flows")
    with pytest.raises(KernelCompileError, match="kernel emitter keys"):
        type(kernel_program).from_dict(stale)

    changed_topology = kernel_program.to_dict()
    serialized_update_flow = next(
        flow for flow in changed_topology["emitters"][0]["flows"] if flow["lifecycle_stage"] == "update"
    )
    branch_lane_indices = sorted((blocks["left"].lane_index, blocks["right"].lane_index))
    serialized_update_flow["lanes"][branch_lane_indices[1]]["parent_index"] = branch_lane_indices[0]
    with pytest.raises(KernelCompileError, match="kernel hash"):
        type(kernel_program).from_dict(changed_topology)


def test_particle_if_activates_only_the_selected_exec_branch():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("condition", "common.constant.bool", properties={"value": True}),
            GraphNodeRecord("if", "particle.control.if"),
            GraphNodeRecord("true-size", "particle.attribute.size", properties={"value": 2.0}),
            GraphNodeRecord("false-size", "particle.attribute.size", properties={"value": 0.5}),
        ),
        links=(
            GraphLinkRecord("root-if", "root.update", "out", "if", "in", PortKind.EXEC),
            GraphLinkRecord("condition-if", "condition", "value", "if", "condition"),
            GraphLinkRecord("if-true", "if", "true", "true-size", "in", PortKind.EXEC),
            GraphLinkRecord("if-false", "if", "false", "false-size", "in", PortKind.EXEC),
        ),
    )

    program = ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(ParticleEmitterAsset(update=update),)))
    stage = program.emitters[0].update
    by_uid = {operation.source_node_uid: operation for operation in stage.flow.iter_operations()}

    assert by_uid["if"].opcode == "control.if"
    assert by_uid["true-size"].execution_predicates[0].expected is True
    assert by_uid["false-size"].execution_predicates[0].expected is False
    assert next(edge for edge in stage.flow.edges if edge.link_uid == "if-true").predicate_expected is True
    assert next(edge for edge in stage.flow.edges if edge.link_uid == "if-false").predicate_expected is False

    kernel = ParticleKernelLowerer().lower(program).emitters[0].update
    opcodes = [instruction.opcode for instruction in kernel.instructions]
    assert opcodes.count("begin_if") == 2
    assert opcodes.count("end_if") == 2
    assert "logical_not" in opcodes


def test_particle_event_flows_compile_to_stable_typed_queue_abi():
    impact = ParticleEventType(
        "impact",
        "Impact",
        32,
        (
            ParticleEventField(
                "position",
                "Position",
                TypeRef(ValueType.VEC3, CoordinateSpace.WORLD),
                [0.0, 0.0, 0.0],
            ),
            ParticleEventField(
                "color",
                "Color",
                TypeRef(ValueType.COLOR),
                [1.0, 1.0, 1.0, 1.0],
            ),
        ),
    )
    emitter = ParticleEmitterAsset(
        stable_id="source",
        name="Source",
        event_flows=(ParticleEventFlow("impact", default_event_graph("impact")),),
    )
    asset = ParticleGraphAsset(
        stable_id="event-graph",
        emitters=(emitter,),
        event_types=(impact,),
    )

    restored = ParticleGraphAsset.from_json(asset.canonical_json())
    assert restored == asset
    program = ParticleGraphCompiler().compile(asset)
    assert len(program.events.event_types) == 1
    event_type = program.events.event_types[0]
    assert event_type.payload_stride_words == 7
    assert [(field.word_offset, field.word_count) for field in event_type.fields] == [
        (0, 3),
        (3, 4),
    ]
    assert event_type.stable_type_hash != 0
    assert program.events.event_abi_u64 != 0
    assert program.emitters[0].event_flows[0].flow_id == "impact"
    kernel = ParticleKernelLowerer().lower(program)
    opcodes = [item.opcode for item in kernel.emitters[0].update.instructions]
    assert "event_begin" in opcodes
    assert "event_complete" in opcodes

    changed = replace(
        asset,
        event_types=(replace(impact, queue_capacity=64),),
    )
    assert ParticleGraphCompiler().compile(changed).events.event_abi_hash != program.events.event_abi_hash


def test_integer_emitter_shape_preserves_saved_aot_signature(tmp_path):
    asset = ParticleGraphAsset(
        stable_id="integer-shape",
        emitters=(ParticleEmitterAsset(settings=EmitterSettings(
            shape=EmitterShape(kind="cone", radius=0, angle_degrees=35,
                               dimensions=(1, 2, 3)),
        )),),
    )
    path = str(tmp_path / "Shape.particlegraph")
    asset.save(path)
    artifact = ParticleArtifactRegistry.get(path)
    restored = ParticleGraphAsset.load(path)
    program = ParticleGraphCompiler().compile(restored)
    gpu = GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(program))

    assert program.behavior_hash == artifact.behavior_hash
    assert gpu.kernel_hash == artifact.gpu_spirv["kernel_hash"]
    validate_gpu_particle_spirv(artifact.gpu_spirv, gpu)
    shape = asset.emitters[0].settings.shape
    assert type(shape.radius) is float
    assert type(shape.angle_degrees) is float
    assert shape == restored.emitters[0].settings.shape


def test_particle_data_interfaces_round_trip_with_stable_identity_and_space():
    emitter = ParticleEmitterAsset(
        stable_id="data-emitter",
        data_interfaces=(
            VectorField(
                stable_id="wind-field",
                name="Wind",
                texture=AssetReference(path_hint="Assets/Fields/Wind.vectorfield"),
                space="world",
                boundary="repeat",
                filtering="linear",
                vector_scale=2.5,
            ),
            SdfVolume(
                stable_id="collision-field",
                name="Collision",
                texture=AssetReference(path_hint="Assets/Fields/Collision.inxsdf"),
                distance_scale=1.5,
            ),
        ),
    )
    asset = ParticleGraphAsset(stable_id="data-graph", emitters=(emitter,))

    restored = ParticleGraphAsset.from_json(asset.canonical_json())
    hir = ParticleGraphCompiler().compile(restored)

    assert restored == asset
    assert [interface.stable_id for interface in hir.emitters[0].data_interfaces] == [
        "wind-field",
        "collision-field",
    ]
    assert hir.emitters[0].data_interfaces[0].boundary.value == "repeat"

    with pytest.raises(ParticleGraphSchemaError, match="stable ids must be unique"):
        ParticleEmitterAsset(
            data_interfaces=(
                VectorField(stable_id="duplicate"),
                SdfVolume(stable_id="duplicate"),
            )
        )


def test_particle_graph_compiler_builds_multi_emitter_schedule_and_render_plan():
    first = ParticleEmitterAsset(
        stable_id="smoke",
        name="Smoke",
        settings=EmitterSettings(
            capacity=100_000,
            simulation_space=SimulationSpace.WORLD,
            spawn_rate=20_000.0,
        ),
    )
    rendering = first.rendering
    sprite = rendering.nodes[1]
    first = ParticleEmitterAsset(
        stable_id=first.stable_id,
        name=first.name,
        settings=first.settings,
        attributes=first.attributes,
        init=first.init,
        update=first.update,
        rendering=GraphDocument(
            rendering.domain,
            (
                rendering.nodes[0],
                GraphNodeRecord(
                    sprite.uid,
                    sprite.type_id,
                    properties={
                        "shader": "Particle Six-Way Smoke",
                        "receive_scene_lighting": True,
                        "receive_shadows": True,
                        "soft_particles": True,
                        "soft_distance": 0.35,
                        "sort": "back_to_front",
                    },
                ),
            ),
            rendering.links,
        ),
    )
    sparks = ParticleEmitterAsset(stable_id="sparks", name="Sparks")
    program = ParticleGraphCompiler().compile(
        ParticleGraphAsset(stable_id="fire", name="Fire", emitters=(first, sparks))
    )

    assert program.schedule.emitter_ids == ("smoke", "sparks")
    smoke = program.emitters[0]
    assert smoke.init.stage is ParticleStage.INIT
    assert _operations(smoke.init)[0].opcode == "emitter.sample_shape"
    assert _operations(smoke.update)[0].opcode == "attribute.modify_velocity"
    assert smoke.render_plan.outputs[0].shader == "Particle Six-Way Smoke"
    assert {item.name for item in smoke.render_plan.outputs[0].shader_properties} >= {
        "baseColor",
        "positiveAxesMap",
        "negativeAxesMap",
    }
    assert smoke.render_plan.outputs[0].receive_scene_lighting is True
    assert smoke.render_plan.outputs[0].receive_shadows is True
    assert smoke.render_plan.outputs[0].cast_shadows is False
    assert smoke.render_plan.outputs[0].soft_particles is True
    assert smoke.render_plan.outputs[0].soft_distance == pytest.approx(0.35)
