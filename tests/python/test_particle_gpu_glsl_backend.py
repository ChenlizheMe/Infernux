from __future__ import annotations

import copy
import re
import struct

import pytest

import infernux.particle.gpu_glsl_backend as gpu_backend

from infernux.lib import _Infernux as native
from infernux.particle import (
    EmitterSettings,
    EmitterShape,
    GpuParticleGlslLowerer,
    GpuParticleCompileError,
    ParticleAttribute,
    ParticleEmitterAsset,
    ParticleEventField,
    ParticleEventFlow,
    ParticleEventType,
    ParticleGraphAsset,
    ParticleGraphCompiler,
    ParticleScriptCompiler,
    ParticleParameter,
    ParticleKernelLowerer,
    MeshEmissionMode,
    SdfVolume,
    VectorField,
    build_gpu_particle_migration,
    compile_gpu_particle_spirv,
    default_event_graph,
    standard_particle_attributes,
    validate_gpu_particle_spirv,
    pack_gpu_particle_parameters,
    pack_gpu_particle_parameter_updates,
)
from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.graph.types import AssetReference, CoordinateSpace, TypeRef, ValueType
from infernux.graph.ramp import AnimationCurve, Gradient, GradientKey, Keyframe
from infernux.particle.nodes import (
    PARTICLE_EVENT_ACTIVE_TYPE_ID,
    PARTICLE_EVENT_TRIGGER_TYPE_ID,
    particle_event_payload_port_id,
)
from infernux.particle.asset import particle_attribute_cache_id


def _gpu_source():
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(stable_id="gpu-particle"))
    kernel = ParticleKernelLowerer().lower(hir)
    return GpuParticleGlslLowerer().lower(kernel)


def _assert_generated_values_are_declared(glsl: str) -> None:
    value_names = set(re.findall(r"\bv\d+\b", glsl))
    declarations = set(
        re.findall(
            r"\b(?:bool|int|uint|float|[biu]?vec[234]|mat[234])\s+(v\d+)\b",
            glsl,
        )
    )
    assert value_names <= declarations


def _assert_all_generated_values_are_declared(source) -> None:
    for emitter in source.emitters:
        for glsl in emitter.stages().values():
            _assert_generated_values_are_declared(glsl)
        if emitter.continuation is not None:
            for glsl in emitter.continuation.stages().values():
                _assert_generated_values_are_declared(glsl)


def _scene_collision_asset():
    return ParticleGraphAsset(
        stable_id="scene-collision-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(collision_enabled=True),
            ),
        ),
    )


def _if_asset():
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
    return ParticleGraphAsset(
        stable_id="if-gpu",
        emitters=(ParticleEmitterAsset(update=update),),
    )


def _collision_lifecycle_asset():
    collision_enter = GraphDocument(
        "particle.collision_enter",
        nodes=(
            GraphNodeRecord("root.collision_enter", "particle.root.collision_enter"),
            GraphNodeRecord(
                "enter-size",
                "particle.attribute.size",
                properties={"value": 2.0},
            ),
        ),
        links=(
            GraphLinkRecord(
                "enter-exec",
                "root.collision_enter",
                "out",
                "enter-size",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    return ParticleGraphAsset(
        stable_id="collision-lifecycle-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(collision_enabled=True),
                collision_enter=collision_enter,
            ),
        ),
    )


def _collision_wait_asset():
    collision_enter = GraphDocument(
        "particle.collision_enter",
        nodes=(
            GraphNodeRecord("root.collision_enter", "particle.root.collision_enter"),
            GraphNodeRecord("frames", "common.constant.i32", properties={"value": 3}),
            GraphNodeRecord("wait", "particle.control.wait_frames"),
            GraphNodeRecord(
                "enter-size",
                "particle.attribute.size",
                properties={"value": 2.0},
            ),
        ),
        links=(
            GraphLinkRecord(
                "enter-wait",
                "root.collision_enter",
                "out",
                "wait",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "frames-wait",
                "frames",
                "value",
                "wait",
                "frames",
                PortKind.VALUE,
            ),
            GraphLinkRecord(
                "wait-size",
                "wait",
                "out",
                "enter-size",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    return ParticleGraphAsset(
        stable_id="collision-wait-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(collision_enabled=True),
                collision_enter=collision_enter,
            ),
        ),
    )


def _collision_wait_join_asset():
    collision_enter = GraphDocument(
        "particle.collision_enter",
        nodes=(
            GraphNodeRecord("root.collision_enter", "particle.root.collision_enter"),
            GraphNodeRecord("frames", "common.constant.i32", properties={"value": 3}),
            GraphNodeRecord("wait", "particle.control.wait_frames"),
            GraphNodeRecord(
                "delayed-size",
                "particle.attribute.size",
                properties={"value": 2.0},
            ),
            GraphNodeRecord(
                "immediate-position",
                "particle.attribute.position",
                properties={"value": [1.0, 2.0, 3.0]},
            ),
            GraphNodeRecord("join", "particle.control.join_all"),
            GraphNodeRecord(
                "tail-color",
                "particle.attribute.color",
                properties={"value": [0.25, 0.5, 0.75, 1.0]},
            ),
        ),
        links=(
            GraphLinkRecord("root-wait", "root.collision_enter", "out", "wait", "in", PortKind.EXEC),
            GraphLinkRecord("frames-wait", "frames", "value", "wait", "frames", PortKind.VALUE),
            GraphLinkRecord("wait-size", "wait", "out", "delayed-size", "in", PortKind.EXEC),
            GraphLinkRecord("size-join", "delayed-size", "out", "join", "in0", PortKind.EXEC),
            GraphLinkRecord(
                "root-position",
                "root.collision_enter",
                "out",
                "immediate-position",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "position-join",
                "immediate-position",
                "out",
                "join",
                "in1",
                PortKind.EXEC,
            ),
            GraphLinkRecord("join-tail", "join", "out", "tail-color", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="collision-wait-join-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(collision_enabled=True),
                collision_enter=collision_enter,
            ),
        ),
    )


def _wait_asset(*, fork: bool = False):
    nodes = [
        GraphNodeRecord("root.update", "particle.root.update"),
        GraphNodeRecord("frames", "common.constant.i32", properties={"value": 3}),
        GraphNodeRecord("wait", "particle.control.wait_frames"),
        GraphNodeRecord("tail", "particle.attribute.size", properties={"value": 2.0}),
    ]
    links = [
        GraphLinkRecord("root-wait", "root.update", "out", "wait", "in", PortKind.EXEC),
        GraphLinkRecord("frames-wait", "frames", "value", "wait", "frames", PortKind.VALUE),
        GraphLinkRecord("wait-tail", "wait", "out", "tail", "in", PortKind.EXEC),
    ]
    if fork:
        nodes.append(
            GraphNodeRecord(
                "sibling",
                "particle.attribute.position",
                properties={"value": [1.0, 2.0, 3.0]},
            )
        )
        links.append(
            GraphLinkRecord(
                "root-sibling",
                "root.update",
                "out",
                "sibling",
                "in",
                PortKind.EXEC,
            )
        )
    return ParticleGraphAsset(
        stable_id="wait-gpu-fork" if fork else "wait-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(capacity=512),
                update=GraphDocument("particle.update", nodes=tuple(nodes), links=tuple(links)),
            ),
        ),
    )


def _wait_acceleration_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("wait", "particle.control.wait_frames", properties={"frames": 1}),
            GraphNodeRecord(
                "accelerate",
                "particle.attribute.velocity",
                properties={"value": [0.0, -9.81, 0.0]},
            ),
        ),
        links=(
            GraphLinkRecord("root-wait", "root.update", "out", "wait", "in", PortKind.EXEC),
            GraphLinkRecord("wait-accelerate", "wait", "out", "accelerate", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="wait-acceleration-gpu",
        emitters=(ParticleEmitterAsset(update=update),),
    )


def _two_wait_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("frames", "common.constant.i32", properties={"value": 2}),
            GraphNodeRecord("seconds", "common.constant.f32", properties={"value": 0.25}),
            GraphNodeRecord("wait.frames", "particle.control.wait_frames"),
            GraphNodeRecord("wait.seconds", "particle.control.wait_seconds"),
            GraphNodeRecord("tail", "particle.attribute.size"),
        ),
        links=(
            GraphLinkRecord(
                "root-frames",
                "root.update",
                "out",
                "wait.frames",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "frames-value",
                "frames",
                "value",
                "wait.frames",
                "frames",
                PortKind.VALUE,
            ),
            GraphLinkRecord(
                "frames-seconds",
                "wait.frames",
                "out",
                "wait.seconds",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "seconds-value",
                "seconds",
                "value",
                "wait.seconds",
                "seconds",
                PortKind.VALUE,
            ),
            GraphLinkRecord(
                "seconds-tail",
                "wait.seconds",
                "out",
                "tail",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    return ParticleGraphAsset(
        stable_id="two-waits-gpu",
        emitters=(ParticleEmitterAsset(update=update),),
    )


def _terminal_wait_acceleration_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "accelerate.down",
                "particle.attribute.velocity",
                properties={"value": [0.0, -9.8, 0.0]},
            ),
            GraphNodeRecord(
                "wait.three",
                "particle.control.wait_seconds",
                properties={"seconds": 3.0},
            ),
            GraphNodeRecord(
                "accelerate.up_right",
                "particle.attribute.velocity",
                properties={"value": [9.8, 9.8, 0.0]},
            ),
            GraphNodeRecord(
                "wait.five",
                "particle.control.wait_seconds",
                properties={"seconds": 5.0},
            ),
        ),
        links=(
            GraphLinkRecord(
                "root-down",
                "root.update",
                "out",
                "accelerate.down",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "down-wait",
                "accelerate.down",
                "out",
                "wait.three",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "wait-up",
                "wait.three",
                "out",
                "accelerate.up_right",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "up-terminal",
                "accelerate.up_right",
                "out",
                "wait.five",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    return ParticleGraphAsset(
        stable_id="terminal-wait-acceleration-gpu",
        emitters=(ParticleEmitterAsset(update=update),),
    )


def _until_velocity_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "velocity.down",
                "particle.attribute.velocity",
                properties={"composition": "add", "value": [0.0, -1.0, 0.0]},
            ),
            GraphNodeRecord(
                "until.down",
                "particle.control.until_seconds",
                properties={"seconds": 3.0},
            ),
            GraphNodeRecord(
                "velocity.side",
                "particle.attribute.velocity",
                properties={"composition": "add", "value": [1.0, 0.0, 0.0]},
            ),
            GraphNodeRecord(
                "until.side",
                "particle.control.until_frames",
                properties={"frames": 5},
            ),
        ),
        links=(
            GraphLinkRecord("root-down", "root.update", "out", "velocity.down", "in", PortKind.EXEC),
            GraphLinkRecord("down-until", "velocity.down", "out", "until.down", "in", PortKind.EXEC),
            GraphLinkRecord("until-side", "until.down", "out", "velocity.side", "in", PortKind.EXEC),
            GraphLinkRecord("side-until", "velocity.side", "out", "until.side", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="until-velocity-gpu",
        emitters=(ParticleEmitterAsset(update=update),),
    )


def _wait_join_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("frames", "common.constant.i32", properties={"value": 2}),
            GraphNodeRecord("wait", "particle.control.wait_frames"),
            GraphNodeRecord(
                "left",
                "particle.attribute.size",
                properties={"value": 2.0},
            ),
            GraphNodeRecord(
                "right",
                "particle.attribute.position",
                properties={"value": [1.0, 2.0, 3.0]},
            ),
            GraphNodeRecord("join", "particle.control.join_all"),
            GraphNodeRecord(
                "tail",
                "particle.attribute.color",
                properties={"value": [0.25, 0.5, 0.75, 1.0]},
            ),
        ),
        links=(
            GraphLinkRecord("root-wait", "root.update", "out", "wait", "in", PortKind.EXEC),
            GraphLinkRecord("frames-wait", "frames", "value", "wait", "frames", PortKind.VALUE),
            GraphLinkRecord("wait-left", "wait", "out", "left", "in", PortKind.EXEC),
            GraphLinkRecord("left-join", "left", "out", "join", "in0", PortKind.EXEC),
            GraphLinkRecord("root-right", "root.update", "out", "right", "in", PortKind.EXEC),
            GraphLinkRecord("right-join", "right", "out", "join", "in1", PortKind.EXEC),
            GraphLinkRecord("join-tail", "join", "out", "tail", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="wait-join-gpu",
        emitters=(ParticleEmitterAsset(settings=EmitterSettings(capacity=512), update=update),),
    )


def _dual_wait_join_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("frames", "common.constant.i32", properties={"value": 2}),
            GraphNodeRecord("seconds", "common.constant.f32", properties={"value": 0.25}),
            GraphNodeRecord("wait.left", "particle.control.wait_frames"),
            GraphNodeRecord("wait.right", "particle.control.wait_seconds"),
            GraphNodeRecord("left", "particle.attribute.size", properties={"value": 2.0}),
            GraphNodeRecord(
                "right",
                "particle.attribute.position",
                properties={"value": [1.0, 2.0, 3.0]},
            ),
            GraphNodeRecord("join", "particle.control.join_all"),
            GraphNodeRecord(
                "tail",
                "particle.attribute.color",
                properties={"value": [0.25, 0.5, 0.75, 1.0]},
            ),
        ),
        links=(
            GraphLinkRecord("root-left", "root.update", "out", "wait.left", "in", PortKind.EXEC),
            GraphLinkRecord("frames-left", "frames", "value", "wait.left", "frames", PortKind.VALUE),
            GraphLinkRecord("left-tail", "wait.left", "out", "left", "in", PortKind.EXEC),
            GraphLinkRecord("left-join", "left", "out", "join", "in0", PortKind.EXEC),
            GraphLinkRecord("root-right", "root.update", "out", "wait.right", "in", PortKind.EXEC),
            GraphLinkRecord("seconds-right", "seconds", "value", "wait.right", "seconds", PortKind.VALUE),
            GraphLinkRecord("right-tail", "wait.right", "out", "right", "in", PortKind.EXEC),
            GraphLinkRecord("right-join", "right", "out", "join", "in1", PortKind.EXEC),
            GraphLinkRecord("join-tail", "join", "out", "tail", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="dual-wait-join-gpu",
        emitters=(ParticleEmitterAsset(settings=EmitterSettings(capacity=512), update=update),),
    )


def _dual_init_wait_asset():
    init = GraphDocument(
        "particle.init",
        nodes=(
            GraphNodeRecord("root.init", "particle.root.init"),
            GraphNodeRecord(
                "wait.left",
                "particle.control.wait_frames",
                properties={"frames": 2},
            ),
            GraphNodeRecord(
                "wait.right",
                "particle.control.wait_frames",
                properties={"frames": 2},
            ),
        ),
        links=(
            GraphLinkRecord("root-left", "root.init", "out", "wait.left", "in", PortKind.EXEC),
            GraphLinkRecord("root-right", "root.init", "out", "wait.right", "in", PortKind.EXEC),
        ),
    )
    return ParticleGraphAsset(
        stable_id="dual-init-wait-gpu",
        emitters=(ParticleEmitterAsset(init=init),),
    )


def _nested_wait_join_asset():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("frames.first", "common.constant.i32", properties={"value": 2}),
            GraphNodeRecord("frames.second", "common.constant.i32", properties={"value": 3}),
            GraphNodeRecord("wait.first", "particle.control.wait_frames"),
            GraphNodeRecord("first.left", "particle.attribute.size", properties={"value": 2.0}),
            GraphNodeRecord(
                "first.right",
                "particle.attribute.position",
                properties={"value": [1.0, 2.0, 3.0]},
            ),
            GraphNodeRecord("join.first", "particle.control.join_all"),
            GraphNodeRecord("wait.second", "particle.control.wait_frames"),
            GraphNodeRecord(
                "second.left",
                "particle.attribute.velocity",
                properties={"value": [0.0, 4.0, 0.0]},
            ),
            GraphNodeRecord(
                "second.right",
                "particle.attribute.color",
                properties={"value": [0.2, 0.4, 0.8, 1.0]},
            ),
            GraphNodeRecord("join.second", "particle.control.join_all"),
            GraphNodeRecord(
                "tail",
                "particle.attribute.rotation",
                properties={"value": 0.75},
            ),
        ),
        links=(
            GraphLinkRecord(
                "root-first-wait",
                "root.update",
                "out",
                "wait.first",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "first-frames-value",
                "frames.first",
                "value",
                "wait.first",
                "frames",
                PortKind.VALUE,
            ),
            GraphLinkRecord(
                "first-wait-left",
                "wait.first",
                "out",
                "first.left",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "first-left-join",
                "first.left",
                "out",
                "join.first",
                "in0",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "root-first-right",
                "root.update",
                "out",
                "first.right",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "first-right-join",
                "first.right",
                "out",
                "join.first",
                "in1",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "first-join-second-wait",
                "join.first",
                "out",
                "wait.second",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "second-frames-value",
                "frames.second",
                "value",
                "wait.second",
                "frames",
                PortKind.VALUE,
            ),
            GraphLinkRecord(
                "second-wait-left",
                "wait.second",
                "out",
                "second.left",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "second-left-join",
                "second.left",
                "out",
                "join.second",
                "in0",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "first-join-second-right",
                "join.first",
                "out",
                "second.right",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "second-right-join",
                "second.right",
                "out",
                "join.second",
                "in1",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "second-join-tail",
                "join.second",
                "out",
                "tail",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    return ParticleGraphAsset(
        stable_id="nested-wait-join-gpu",
        emitters=(
            ParticleEmitterAsset(
                settings=EmitterSettings(capacity=512),
                update=update,
            ),
        ),
    )


@pytest.mark.parametrize("fork", [False, True])
def test_gpu_wait_emits_bounded_continuation_program_and_valid_spirv(fork):
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(_wait_asset(fork=fork)))
    source = GpuParticleGlslLowerer().lower(kernel)
    emitter = source.emitters[0]
    continuation = emitter.continuation

    assert continuation is not None
    assert continuation.record_stride == 64
    assert continuation.lane_count == 1
    assert continuation.join_count == 0
    assert "layout(std430, set = 5, binding = 0)" in emitter.update
    assert "inx_suspend_frames(" in emitter.update
    assert "case 1u:" in continuation.dispatch
    assert "state.a_builtin_size =" in continuation.dispatch
    if fork:
        assert "state.a_builtin_position =" in emitter.update
        assert emitter.update.count("_active") >= 2

    compiled = compile_gpu_particle_spirv(source)
    validate_gpu_particle_spirv(compiled, source)
    encoded = compiled["emitters"][0]["continuation"]
    assert encoded is not None
    assert set(encoded["stages"]) == {"prepare", "classify", "dispatch"}
    decoded = gpu_backend.decode_gpu_particle_spirv(compiled, 0)["continuation"]
    assert decoded is not None
    assert decoded["record_stride"] == 64
    assert all(decoded["stages"][stage] for stage in encoded["stages"])


def test_gpu_until_repeats_preceding_attribute_operation_and_compiles_valid_spirv():
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(_until_velocity_asset()))
    emitter = kernel.emitters[0]
    assert [item.resume_node_uid for item in emitter.suspensions] == [
        "velocity.down",
        "velocity.side",
    ]

    source = GpuParticleGlslLowerer().lower(kernel)
    continuation = source.emitters[0].continuation
    assert continuation is not None
    assert "inx_until_seconds(" in source.emitters[0].update
    assert "inx_until_frames(" in source.emitters[0].update
    assert "INX_CONTINUATION_FLAG_UNTIL_SECONDS" in source.emitters[0].update
    assert "INX_CONTINUATION_FLAG_UNTIL_FRAMES" in source.emitters[0].update
    assert continuation.dispatch.count("case ") == 2
    assert continuation.dispatch.count("state.a_builtin_velocity =") >= 2
    compiled = compile_gpu_particle_spirv(source)
    validate_gpu_particle_spirv(compiled, source)


def test_gpu_nested_wait_joins_move_the_continuation_to_the_next_join():
    source = GpuParticleGlslLowerer().lower(
        ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(_nested_wait_join_asset()))
    )
    emitter = source.emitters[0]
    continuation = emitter.continuation

    assert continuation is not None
    assert continuation.lane_count == 2
    assert continuation.join_count == 2
    assert continuation.dispatch.count("case ") == 2
    assert "state.a_builtin_velocity =" in continuation.dispatch
    assert "state.a_builtin_rotation =" in continuation.dispatch
    assert "inx_continuation_record_join_index == 0u" in continuation.dispatch
    assert "inx_continuation_record_join_index == 1u" in continuation.dispatch

    compiled = compile_gpu_particle_spirv(source)
    validate_gpu_particle_spirv(compiled, source)


def test_contact_wait_join_uses_full_contact_identity_and_valid_spirv():
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(_collision_wait_join_asset()))
    source = GpuParticleGlslLowerer().lower(kernel)
    emitter = source.emitters[0]
    continuation = emitter.continuation

    assert continuation is not None
    assert continuation.join_count == 1
    assert "ParticleContactContinuationJoinStates" in emitter.contact_dispatch
    assert "inx_contact_continuation_join_begin" in emitter.contact_dispatch
    assert "inx_continuation_context_identity_low = contact.identity.z" in emitter.contact_dispatch
    assert "inx_continuation_context_identity_high = contact.identity.w" in emitter.contact_dispatch
    assert "inx_contact_continuation_join_arrive" in continuation.dispatch
    assert "contact_continuation_snapshot_words[snapshot_base + 2u]" in continuation.dispatch
    assert "contact_continuation_snapshot_words[snapshot_base + 3u]" in continuation.dispatch
    assert "contact Wait/Until branches cannot Join" not in continuation.dispatch

    compiled = compile_gpu_particle_spirv(source)
    validate_gpu_particle_spirv(compiled, source)


def test_scene_collision_uses_shared_grid_abi_and_compiles_to_spirv():
    hir = ParticleGraphCompiler().compile(_scene_collision_asset())
    assert not any(operation.opcode == "collision.scene" for operation in hir.emitters[0].update.flow.iter_operations())

    kernel = ParticleKernelLowerer().lower(hir)
    collision_instructions = [
        value for value in kernel.emitters[0].update.instructions if value.opcode == "collide_scene"
    ]
    assert len(collision_instructions) == 1
    assert not any(value.opcode == "record_collision_diagnostics" for value in kernel.emitters[0].update.instructions)
    instruction = collision_instructions[0]
    assert instruction.result_id == ""
    assert [operand.value_type.value_type for operand in instruction.operands] == [
        ValueType.VEC3,
        ValueType.VEC3,
        ValueType.F32,
        ValueType.U32,
        ValueType.BOOL,
        ValueType.F32,
        ValueType.F32,
    ]
    assert instruction.immediate_dict() == {
        "collider_id_high_attribute": "builtin.collision_collider_id_high",
        "collider_id_low_attribute": "builtin.collision_collider_id_low",
        "hit_attribute": "builtin.collision_hit",
        "material_attribute": "builtin.collision_material",
        "normal_attribute": "builtin.collision_normal",
        "penetration_attribute": "builtin.collision_penetration",
        "point_attribute": "builtin.collision_point",
        "position_attribute": "builtin.position",
        "relative_velocity_attribute": "builtin.collision_relative_velocity",
        "trigger_attribute": "builtin.collision_is_trigger",
        "velocity_attribute": "builtin.velocity",
    }

    source = GpuParticleGlslLowerer().lower(kernel)
    assert source.emitters[0].collision_enabled is True
    update_source = source.emitters[0].update
    assert "struct InxParticleAffine" in update_source
    assert "vec4 row0;" in update_source
    assert "vec4 row1;" in update_source
    assert "vec4 row2;" in update_source
    assert "InxParticleAffine collider_to_world;" in update_source
    assert "InxParticleAffine world_to_collider;" in update_source
    assert "InxParticleAffine previous_world_to_collider;" in update_source
    assert "uvec4 geometry;" in update_source
    assert "mat4 collider_to_world;" not in update_source
    assert "mat4 world_to_collider;" not in update_source
    assert "inx_particle_affine_point" in update_source
    assert "inx_particle_affine_linear" in update_source
    collider_fields = (
        "vec4 material;",
        "vec4 world_aabb_min;",
        "vec4 world_aabb_max;",
        "uvec4 metadata;",
        "uvec4 identity;",
    )
    assert [update_source.index(field) for field in collider_fields] == sorted(
        update_source.index(field) for field in collider_fields
    )
    for binding in range(7):
        assert f"set = 6, binding = {binding}" in update_source
    assert "inx_collide_scene(" in update_source
    assert "bool inx_collide_scene(" in update_source
    assert "out vec3 simulation_collision_normal" in update_source
    assert "out vec3 simulation_contact_point" in update_source
    assert "out vec3 simulation_relative_velocity" in update_source
    assert "out float simulation_penetration" in update_source
    assert "out bool collision_is_trigger" in update_source
    assert "out vec4 collision_material" in update_source
    assert "out uvec2 collision_collider_id" in update_source
    assert "primary_collider_id = collider_id" in update_source
    assert "if (is_trigger) continue;" in update_source
    assert update_source.index("atomicAdd(counters.collision_trigger_count, 1u);") < update_source.index(
        "if (is_trigger) continue;"
    )
    assert "state.a_builtin_collision_collider_id_low" in update_source
    assert "state.a_builtin_collision_collider_id_high" in update_source
    assert "uint collision_hit_count;" in update_source
    assert "uint collision_exit_count;" in update_source
    assert "uint collision_max_outward_speed_bits;" in update_source
    assert "uint collision_max_tangent_speed_bits;" in update_source
    assert "uint collision_candidate_overflow_count;" in update_source
    assert "uint candidate_indices[16];" in update_source
    assert "particle_colliders[existing_index].identity.xy" in update_source
    assert "atomicAdd(counters.collision_candidate_overflow_count, 1u);" in update_source
    assert update_source.index("uint candidate_indices[16];") < update_source.index(
        "for (uint candidate = 0u; candidate < candidate_count; ++candidate)"
    )
    render_reset_source = source.emitters[0].render_reset
    assert "(pc.diagnostic_flags & 2u) != 0u" in render_reset_source
    assert "counters.collision_hit_count = 0u;" in render_reset_source
    assert "counters.collision_max_outward_speed_bits = 0u;" in render_reset_source
    assert "counters.collision_max_tangent_speed_bits = 0u;" in render_reset_source
    assert "counters.collision_candidate_overflow_count = 0u;" in render_reset_source
    assert render_reset_source.index("counters.collision_hit_count = 0u;") < render_reset_source.index(
        "simulation_control.simulation_allowed == 0u"
    )
    assert "particle_collision_grid_offsets[cell_index + 1u]" in update_source
    assert "bool inx_sweep_box(" in update_source
    assert "bool inx_sweep_sphere(" in update_source
    assert "bool inx_sweep_capsule(" in update_source
    assert "bool inx_sweep_triangle(" in update_source
    assert "bool inx_collision_mesh(" in update_source
    assert "particle_collision_mesh_bvh[node_index]" in update_source
    assert "collider.metadata.x == 3u" in update_source
    assert "vec3 collider_swept_min" in update_source
    assert "collider.previous_world_aabb_min.xyz" in update_source
    assert "collider.previous_world_aabb_max.xyz" in update_source
    assert "vec3 particle_swept_min" in update_source
    assert "collider.previous_world_to_collider" in update_source
    assert "previous_world_position + collider_displacement" not in update_source
    assert update_source.index("inx_sweep_box(local_previous") < update_source.index(
        "else if (all(lessThanEqual(abs(local_position), half_extent)))"
    )

    contact_prepare_source = source.emitters[0].contact_prepare
    for binding in range(8):
        assert f"set = 7, binding = {binding}" in contact_prepare_source
    assert "const uint INX_CONTACTS_PER_PARTICLE = 8u;" in contact_prepare_source
    assert "uint current_page = pc.simulation_step & 1u;" in contact_prepare_source
    assert "previous_state.y + 1u == pc.simulation_step" in contact_prepare_source
    assert "contact_particle_record_indices[current_base + slot]" in contact_prepare_source
    assert "contact_counters.contact_current_record_count = 0u;" in contact_prepare_source
    assert "contact_counters.contact_max_per_particle = 0u;" in contact_prepare_source
    assert "contact_counters.multi_contact_particle_count = 0u;" in contact_prepare_source
    assert "contact_counters.contact_retained_order_hash = 0u;" in contact_prepare_source
    assert "contact_counters.contact_dropped_order_hash = 0u;" in contact_prepare_source
    assert "contact_counters.contact_min_particle_index = INX_CONTACT_INVALID_INDEX;" in contact_prepare_source
    assert "contact_counters.contact_max_particle_index = 0u;" in contact_prepare_source
    assert "contact_hash_slots[hash_slot].key = uvec4(INX_CONTACT_INVALID_INDEX);" in contact_prepare_source
    assert "bool reset_all = (pc.diagnostic_flags & 4u) != 0u;" in contact_prepare_source
    assert "for (uint page = 0u; page < 2u; ++page)" in contact_prepare_source
    assert (
        "contact_particle_states[particle_index] =\n                uvec4(INX_CONTACT_INVALID_INDEX"
        in contact_prepare_source
    )
    assert "(pc.capacity + 255u) / 256u" in contact_prepare_source
    assert "particle_index * INX_CONTACTS_PER_PARTICLE * 2u" not in contact_prepare_source
    assert "inx_store_contact(" in update_source
    assert "atomicAdd(contact_counters.contact_current_record_count, 1u)" in update_source
    assert "particle_index * INX_CONTACTS_PER_PARTICLE + contact_slot" not in update_source
    assert "bounded global sparse pool" in update_source
    assert "privileges particle indices" in update_source
    assert "first eight actual" in update_source
    assert "particle_state = uvec4(generation, pc.simulation_step, 0u, 0u);" in update_source
    assert "atomicAdd(contact_counters.contact_current_record_count, 1u);" in update_source
    assert "(pc.diagnostic_flags & 1u) != 0u" in update_source
    assert "atomicMax(contact_counters.contact_max_per_particle, contact_slot + 1u);" in update_source
    assert "atomicAdd(contact_counters.multi_contact_particle_count, 1u);" in update_source
    assert "atomicAdd(contact_counters.contact_retained_order_hash" in update_source
    assert "atomicAdd(contact_counters.contact_dropped_order_hash" in update_source
    assert "uvec4(collider_id, contact_slot, 0u)" in update_source
    assert "uvec4(collider_id, contact_order, 1u)" in update_source
    assert "atomicMin(contact_counters.contact_min_particle_index, particle_index);" in update_source
    assert "atomicMax(contact_counters.contact_max_particle_index, particle_index);" in update_source
    assert "inx_collide_scene(particle_index, " in update_source
    assert "particle_index, contact_hit_order, collider_id" in update_source
    assert "struct InxParticleContactRecord" in update_source

    contact_solve_source = source.emitters[0].contact_solve
    assert "uint hash_slot = inx_contact_hash(key) & hash_mask;" in contact_solve_source
    assert "atomicCompSwap(" in contact_solve_source
    assert "contact_hash_slots[physical_hash_slot].value = uvec4(" in contact_solve_source
    assert "uint inx_find_contact_record(uint page, uvec4 key)" in contact_solve_source
    assert "for (uint lifecycle = 0u; lifecycle < 2u; ++lifecycle)" in contact_solve_source
    assert "contact_work_items[work_index].dispatch = uvec4(" in contact_solve_source
    assert "2u, record_index, pc.simulation_step," in contact_solve_source
    assert "contact_records[record_index].metadata.w" in contact_solve_source
    assert "atomicAdd(contact_counters.contact_work_item_count, work_count);" in contact_solve_source
    assert "uint work_capacity = record_capacity * 2u;" in contact_solve_source
    assert "one contiguous" in contact_solve_source
    assert "particle_index * INX_CONTACTS_PER_PARTICLE * 2u" not in contact_solve_source

    contact_dispatch_source = source.emitters[0].contact_dispatch
    assert "packed_work_range >> INX_CONTACT_WORK_COUNT_BITS" in contact_dispatch_source
    assert "local_work < work_count" in contact_dispatch_source

    payload = compile_gpu_particle_spirv(source)
    assert validate_gpu_particle_spirv(payload, source) is payload


def test_gpu_texture2d_parameter_lowers_to_rhi_resource_and_sample():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("set-color", "particle.attribute.color"),
            GraphNodeRecord(
                "texture",
                "particle.parameter",
                properties={"parameter": "smoke-texture"},
            ),
            GraphNodeRecord(
                "uv",
                "common.constant.vec2",
                properties={"value": [0.25, 0.75]},
            ),
            GraphNodeRecord("sample", "common.texture.sample2d"),
        ),
        links=(
            GraphLinkRecord("stream", "root.update", "out", "set-color", "in", PortKind.EXEC),
            GraphLinkRecord("texture-value", "texture", "value", "sample", "texture", PortKind.VALUE),
            GraphLinkRecord("uv-value", "uv", "value", "sample", "uv", PortKind.VALUE),
            GraphLinkRecord("sample-color", "sample", "color", "set-color", "value", PortKind.VALUE),
        ),
    )
    default = AssetReference("smoke-guid", "Assets/VFX/Smoke.png")
    asset = ParticleGraphAsset(
        parameters=(
            ParticleParameter(
                "smoke-texture",
                "Smoke Texture",
                TypeRef(ValueType.TEXTURE2D),
                default.to_dict(),
            ),
        ),
        emitters=(ParticleEmitterAsset(update=update),),
    )
    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(asset))
    source = GpuParticleGlslLowerer().lower(kernel)
    emitter = source.emitters[0]

    assert pack_gpu_particle_parameters(kernel.parameters) == (0, 0, 0, 0)
    assert emitter.data_interface_layout["texture2d_parameters"] == [
        {
            "stable_id": "smoke-texture",
            "name": "Smoke Texture",
            "parameter_slot": 0,
            "resource_index": 0,
            "texture_binding": 1,
            "default": default.to_dict(),
        }
    ]
    assert "layout(set = 2, binding = 1) uniform sampler2D inx_parameter_texture_0;" in emitter.update
    texture_handle = gpu_backend._texture_resource_handle("smoke-texture")
    assert f"inx_sample_parameter_texture({texture_handle}u," in emitter.update
    payload = compile_gpu_particle_spirv(source)
    assert validate_gpu_particle_spirv(payload, source) is payload


def _kill_if_gpu_source():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("kill", "particle.lifecycle.kill_if"),
            GraphNodeRecord(
                "age",
                "particle.attribute.get",
                properties={"attribute": "builtin.age"},
            ),
            GraphNodeRecord("limit", "common.constant.f32", properties={"value": 0.5}),
            GraphNodeRecord("older", "common.compare.greater_than"),
        ),
        links=(
            GraphLinkRecord("stream", "root.update", "out", "kill", "in", PortKind.EXEC),
            GraphLinkRecord("a", "age", "value", "older", "a", PortKind.VALUE),
            GraphLinkRecord("b", "limit", "value", "older", "b", PortKind.VALUE),
            GraphLinkRecord("condition", "older", "result", "kill", "condition", PortKind.VALUE),
        ),
    )
    emitter = ParticleEmitterAsset(stable_id="kill", update=update)
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def _event_queue_program():
    impact = ParticleEventType(
        "impact",
        "Impact",
        4,
        (
            ParticleEventField("position", "Position", TypeRef(ValueType.VEC3), [1.0, 2.0, 3.0]),
            ParticleEventField("kind", "Kind", TypeRef(ValueType.U32), 7),
            ParticleEventField("weight", "Weight", TypeRef(ValueType.F32), 2.5),
        ),
    )
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "trigger",
                PARTICLE_EVENT_TRIGGER_TYPE_ID,
                properties={
                    "event": "impact",
                    "condition": True,
                    particle_event_payload_port_id("position"): [4.0, 5.0, 6.0],
                },
            ),
        ),
        links=(GraphLinkRecord("trigger.exec", "root.update", "out", "trigger", "in", PortKind.EXEC),),
    )
    event_graph = GraphDocument(
        "particle.event",
        nodes=(
            GraphNodeRecord(
                "root.event",
                PARTICLE_EVENT_ACTIVE_TYPE_ID,
                properties={"event": "impact"},
            ),
            GraphNodeRecord("set.size", "particle.attribute.size"),
        ),
        links=(
            GraphLinkRecord("event.exec", "root.event", "out", "set.size", "in", PortKind.EXEC),
            GraphLinkRecord(
                "event.weight",
                "root.event",
                particle_event_payload_port_id("weight"),
                "set.size",
                "value",
                PortKind.VALUE,
            ),
        ),
    )
    emitter = ParticleEmitterAsset(
        stable_id="source",
        update=update,
        event_flows=(ParticleEventFlow("impact", event_graph),),
    )
    kernel = ParticleKernelLowerer().lower(
        ParticleGraphCompiler().compile(
            ParticleGraphAsset(stable_id="event-queue-graph", emitters=(emitter,), event_types=(impact,))
        )
    )
    return kernel, GpuParticleGlslLowerer().lower(kernel)


def test_gpu_event_fanout_stays_non_reentrant_until_every_waiting_branch_finishes():
    event_type = ParticleEventType("pulse", "Pulse", queue_capacity=8)
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "trigger",
                PARTICLE_EVENT_TRIGGER_TYPE_ID,
                properties={"event": "pulse"},
            ),
        ),
        links=(
            GraphLinkRecord(
                "trigger.exec",
                "root.update",
                "out",
                "trigger",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    event_graph = GraphDocument(
        "particle.event",
        nodes=(
            GraphNodeRecord(
                "root.event",
                PARTICLE_EVENT_ACTIVE_TYPE_ID,
                properties={"event": "pulse"},
            ),
            GraphNodeRecord(
                "wait.left",
                "particle.control.wait_frames",
                properties={"frames": 2},
            ),
            GraphNodeRecord(
                "wait.right",
                "particle.control.wait_frames",
                properties={"frames": 3},
            ),
            GraphNodeRecord(
                "set.size",
                "particle.attribute.size",
                properties={"value": 2.0},
            ),
            GraphNodeRecord(
                "set.color",
                "particle.attribute.color",
                properties={"value": [1.0, 0.0, 1.0, 1.0]},
            ),
        ),
        links=(
            GraphLinkRecord(
                "event.left",
                "root.event",
                "out",
                "wait.left",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "event.right",
                "root.event",
                "out",
                "wait.right",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "left.tail",
                "wait.left",
                "out",
                "set.size",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "right.tail",
                "wait.right",
                "out",
                "set.color",
                "in",
                PortKind.EXEC,
            ),
        ),
    )
    emitter = ParticleEmitterAsset(
        stable_id="event-fanout",
        update=update,
        event_flows=(ParticleEventFlow("pulse", event_graph),),
    )
    kernel = ParticleKernelLowerer().lower(
        ParticleGraphCompiler().compile(
            ParticleGraphAsset(
                stable_id="event-fanout-graph",
                emitters=(emitter,),
                event_types=(event_type,),
            )
        )
    )

    event_suspensions = [
        item
        for item in kernel.emitters[0].suspensions
        if item.lifecycle_stage.value == "event" and item.flow_id == "pulse"
    ]
    assert len(event_suspensions) == 2
    assert len({item.lane_stable_id for item in event_suspensions}) == 2

    gpu = GpuParticleGlslLowerer().lower(kernel)
    update_source = gpu.emitters[0].update
    continuation_source = gpu.emitters[0].continuation.stages()["dispatch"]
    assert update_source.count("!inx_continuation_lane_pending(particle_index") >= 2
    assert continuation_source.count("if (!inx_continuation_resuspended && !inx_continuation_lane_pending") == 2
    assert continuation_source.count("a_internal_event_0_active = 0u") == 2
    assert validate_gpu_particle_spirv(compile_gpu_particle_spirv(gpu), gpu)


def _noise_gpu_source():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("acceleration", "particle.attribute.velocity"),
            GraphNodeRecord(
                "position",
                "particle.attribute.get",
                properties={"attribute": "builtin.position"},
            ),
            GraphNodeRecord(
                "noise",
                "common.noise.vector3d",
                properties={"frequency": 2.0, "seed": 17},
            ),
        ),
        links=(
            GraphLinkRecord("stream", "root.update", "out", "acceleration", "in", PortKind.EXEC),
            GraphLinkRecord("position", "position", "value", "noise", "position"),
            GraphLinkRecord("noise", "noise", "value", "acceleration", "value"),
        ),
    )
    emitter = ParticleEmitterAsset(stable_id="noise", update=update)
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def _vector_field_gpu_source(*, boundary="zero", filtering="linear"):
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord("acceleration", "particle.attribute.velocity"),
            GraphNodeRecord(
                "position",
                "particle.attribute.get",
                properties={"attribute": "builtin.position"},
            ),
            GraphNodeRecord(
                "sample",
                "particle.vector_field.sample",
                properties={"interface": "wind"},
            ),
        ),
        links=(
            GraphLinkRecord("stream", "root.update", "out", "acceleration", "in", PortKind.EXEC),
            GraphLinkRecord("position", "position", "value", "sample", "position"),
            GraphLinkRecord("value", "sample", "value", "acceleration", "value"),
        ),
    )
    emitter = ParticleEmitterAsset(
        stable_id="vector-field-emitter",
        update=update,
        data_interfaces=(
            VectorField(
                stable_id="wind",
                texture=AssetReference(guid="wind-texture-guid"),
                boundary=boundary,
                filtering=filtering,
            ),
        ),
    )
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(stable_id="vector-field-gpu", emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def _sdf_gpu_source():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "collision",
                "particle.collision.sdf",
                properties={
                    "interface": "collision-field",
                    "particle_radius": 0.05,
                    "restitution": 0.5,
                    "friction": 0.2,
                },
            ),
        ),
        links=(GraphLinkRecord("stream", "root.update", "out", "collision", "in", PortKind.EXEC),),
    )
    emitter = ParticleEmitterAsset(
        stable_id="sdf-emitter",
        update=update,
        data_interfaces=(
            SdfVolume(
                stable_id="collision-field",
                texture=AssetReference(guid="sdf-texture-guid"),
            ),
        ),
    )
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(stable_id="sdf-gpu", emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def _sdf_sample_gpu_source():
    update = GraphDocument(
        "particle.update",
        nodes=(
            GraphNodeRecord("root.update", "particle.root.update"),
            GraphNodeRecord(
                "position",
                "particle.attribute.get",
                properties={"attribute": "builtin.position"},
            ),
            GraphNodeRecord(
                "distance",
                "particle.sdf.sample_distance",
                properties={"interface": "sample-field"},
            ),
            GraphNodeRecord(
                "gradient",
                "particle.sdf.sample_gradient",
                properties={"interface": "sample-field"},
            ),
            GraphNodeRecord("size", "particle.attribute.size"),
            GraphNodeRecord("velocity", "particle.attribute.velocity"),
        ),
        links=(
            GraphLinkRecord("root-size", "root.update", "out", "size", "in", PortKind.EXEC),
            GraphLinkRecord("size-velocity", "size", "out", "velocity", "in", PortKind.EXEC),
            GraphLinkRecord("position-distance", "position", "value", "distance", "position"),
            GraphLinkRecord("position-gradient", "position", "value", "gradient", "position"),
            GraphLinkRecord("distance-size", "distance", "distance", "size", "value"),
            GraphLinkRecord("gradient-velocity", "gradient", "gradient", "velocity", "value"),
        ),
    )
    emitter = ParticleEmitterAsset(
        stable_id="sdf-sample-emitter",
        update=update,
        data_interfaces=(
            SdfVolume(
                stable_id="sample-field",
                texture=AssetReference(guid="sdf-texture-guid"),
            ),
        ),
    )
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(stable_id="sdf-sample-gpu", emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def _sdf_shape_gpu_source(mode: str):
    interface = SdfVolume(
        stable_id="spawn-field",
        texture=AssetReference(guid="sdf-texture-guid"),
    )
    emitter = ParticleEmitterAsset(
        stable_id=f"sdf-{mode}-emitter",
        settings=EmitterSettings(
            shape=EmitterShape(
                kind="sdf",
                sdf_interface=interface.stable_id,
                sdf_mode=mode,
            )
        ),
        data_interfaces=(interface,),
    )
    hir = ParticleGraphCompiler().compile(ParticleGraphAsset(stable_id=f"sdf-{mode}-gpu", emitters=(emitter,)))
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir))


def test_gpu_layout_migration_descriptor_copies_stable_fields_and_packs_defaults():
    stable_id = "layout-emitter"
    cache_node_uid = "cache.temperature"
    cache_id = particle_attribute_cache_id("init", cache_node_uid)
    previous_asset = ParticleGraphAsset(
        stable_id="previous-layout",
        emitters=(ParticleEmitterAsset(stable_id=stable_id),),
    )
    next_asset = ParticleGraphAsset(
        stable_id="next-layout",
        emitters=(
            ParticleEmitterAsset(
                stable_id=stable_id,
                init=GraphDocument(
                    "particle.init",
                    nodes=(
                        GraphNodeRecord("root.init", "particle.root.init"),
                        GraphNodeRecord(
                            cache_node_uid,
                            "particle.attribute.cache",
                            properties={
                                "name": "Temperature",
                                "value_type": "f32",
                                "value_space": "none",
                                "composition": "set",
                                "value": 3.5,
                            },
                        ),
                    ),
                    links=(
                        GraphLinkRecord(
                            "root-cache",
                            "root.init",
                            "out",
                            cache_node_uid,
                            "in",
                            PortKind.EXEC,
                        ),
                    ),
                ),
            ),
        ),
    )
    previous_kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(previous_asset))
    next_kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(next_asset))
    previous_layout = GpuParticleGlslLowerer().lower(previous_kernel).emitters[0].to_dict()
    next_layout = GpuParticleGlslLowerer().lower(next_kernel).emitters[0].to_dict()

    migration = build_gpu_particle_migration(
        previous_layout,
        next_layout,
        next_kernel.emitters[0],
    )

    assert migration["source_stride"] == previous_layout["state_stride"]
    assert migration["destination_stride"] == next_layout["state_stride"]
    assert len(migration["copy_ranges"]) == len(standard_particle_attributes())
    temperature = next(field for field in next_layout["attribute_fields"] if field["stable_id"] == cache_id)
    default_word = migration["default_state_words"][temperature["offset"] // 4]
    assert struct.unpack("<f", struct.pack("<I", default_word))[0] == pytest.approx(0.0)
    assert all(
        item["source_offset"] % 4 == 0 and item["destination_offset"] % 4 == 0 and item["byte_size"] % 4 == 0
        for item in migration["copy_ranges"]
    )

    stale_layout = copy.deepcopy(previous_layout)
    stale_layout["attribute_fields"][0].pop("offset")
    with pytest.raises(GpuParticleCompileError, match="layout entry"):
        build_gpu_particle_migration(stale_layout, next_layout, next_kernel.emitters[0])


def test_gpu_mesh_parameter_exposes_all_static_mesh_attributes_and_compiles():
    source = """\
from infernux.particle import AssetReference, EmitterSettings, Parameter, ParticleEmitter, ParticleScript

class MeshSampling(ParticleScript):
    parameters = (
        Parameter(
            "surface-mesh",
            "Surface Mesh",
            "mesh",
            AssetReference(path_hint="Assets/Models/surface.fbx"),
        ),
    )

    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings()

        def init(self, ctx, particles):
            particles.set_position(ctx.sample_mesh_position(ctx.parameter("Surface Mesh"), (0.1, 0.2, 0.3)))
            particles.set_velocity(ctx.sample_mesh_normal(ctx.parameter("Surface Mesh"), (0.2, 0.3, 0.4), mode="edge"))
            particles.set_attribute("Mesh UV", ctx.sample_mesh_uv(ctx.parameter("Surface Mesh"), (0.3, 0.4, 0.5), mode="vertex"))
            particles.set_attribute("Mesh Barycentric", ctx.sample_mesh_barycentric(ctx.parameter("Surface Mesh"), (0.4, 0.5, 0.6)))

        def update(self, ctx, particles):
            particles.set_velocity(ctx.sample_mesh_tangent(ctx.parameter("Surface Mesh"), (0.5, 0.6, 0.7)))

        def rendering(self, ctx, particles):
            particles.sprite()
"""
    hir = ParticleScriptCompiler().compile(source, source_name="MeshSampling.particle.py")
    emitter = GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir)).emitters[0]

    mesh_interfaces = emitter.data_interface_layout["mesh_interfaces"]
    assert len(mesh_interfaces) == 1
    assert mesh_interfaces[0]["stable_id"].startswith("sample.mesh.")
    assert mesh_interfaces[0]["mesh_parameter"] == "surface-mesh"
    assert mesh_interfaces[0]["mesh"] == AssetReference(path_hint="Assets/Models/surface.fbx").to_dict()
    combined = emitter.init + emitter.update
    assert ".position" in combined
    assert ".normal" in combined
    assert ".tangent" in combined
    assert ".uv" in combined
    assert ".barycentric" in combined
    assert "INX_MESH_SAMPLE_VERTEX" in combined
    assert "INX_MESH_SAMPLE_EDGE" in combined
    assert "INX_MESH_SAMPLE_SURFACE" in combined
    assert "edge_count" in combined
    assert "triangle_count +" in combined
    assert "transformed_tangent -= result.normal" in combined
    assert "inx_mesh_vertex_0" in combined
    assert "inx_mesh_palette_0" in combined
    assert "vec4 tangent" in combined
    assert "tangent_handedness" in combined
    assert "determinant(mat3(mesh_to_simulation))" in combined
    assert combined.count("readonly buffer InxMeshVertices0") == 2
    assert "InxMeshVertices1" not in combined
    compiled = native._compile_compute_glsl_batch(emitter.stages(), "particle-mesh-data-interface")
    assert set(compiled) == set(emitter.stages())


@pytest.mark.parametrize("mode,surface_literal", (("surface", "true"), ("volume", "false")))
def test_gpu_sdf_emitter_shape_uses_bounded_spawn_only_sampling_and_valid_spirv(mode, surface_literal):
    emitter = _sdf_shape_gpu_source(mode).emitters[0]

    assert emitter.data_interface_layout["volume_interfaces"] == [
        {
            "kind": "sdf",
            "stable_id": "spawn-field",
            "interface_index": 0,
            "texture_binding": 1,
            "filtering": "linear",
        }
    ]
    assert "inx_sample_sdf_shape_position_0" in emitter.init
    assert f", {surface_literal})" in emitter.init
    assert "attempt < 16" in emitter.init
    assert "iteration < 8" in emitter.init
    assert emitter.init.count("inx_sample_sdf_shape_position_0") == 2
    assert emitter.update.count("inx_sample_sdf_shape_position_0") == 1

    compiled = native._compile_compute_glsl_batch(emitter.stages(), f"particle-sdf-{mode}-shape-test")
    assert set(compiled) == set(emitter.stages())
