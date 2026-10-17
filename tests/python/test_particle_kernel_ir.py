from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.graph.types import AssetReference, CoordinateSpace, TypeRef, ValueType
from infernux.particle import (
    EmitterSettings,
    EmitterShape,
    KernelCompileError,
    KernelInstruction,
    KernelOperand,
    KernelStage,
    ParticleEmitterAsset,
    ParticleEventFlow,
    ParticleEventType,
    ParticleGraphAsset,
    ParticleGraphCompiler,
    ParticleKernelFunction,
    ParticleKernelLowerer,
    ParticleKernelProgram,
    ParticleRuntimeCompatibility,
    SdfVolume,
    VectorField,
    classify_emitter_update,
    default_event_graph,
    particle_random_f32,
    particle_random_u32,
)
from infernux.particle.kernel_ir import (
    KernelSourceRef,
    _PURE_KERNEL_VALUE_OPCODES,
    _KernelBuilder,
)
from infernux.particle.kernel_semantics import KERNEL_OPCODE_SPECS


def _lower(asset: ParticleGraphAsset):
    return ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(asset))


def test_data_interface_abi_round_trips_and_resource_rebind_preserves_state():
    first_emitter = ParticleEmitterAsset(
        stable_id="data-emitter",
        data_interfaces=(
            VectorField(
                stable_id="wind",
                texture=AssetReference(path_hint="Assets/WindA.vectorfield"),
            ),
            SdfVolume(
                stable_id="collision",
                texture=AssetReference(path_hint="Assets/Collision.inxsdf"),
            ),
        ),
    )
    first = _lower(ParticleGraphAsset(stable_id="data-graph", emitters=(first_emitter,)))
    restored = ParticleKernelProgram.from_dict(first.to_dict())

    assert restored == first
    assert [value.stable_id for value in restored.emitters[0].data_interfaces] == [
        "collision",
        "wind",
    ]

    rebound_emitter = replace(
        first_emitter,
        data_interfaces=(
            replace(
                first_emitter.data_interfaces[0],
                texture=AssetReference(path_hint="Assets/WindB.vectorfield"),
            ),
            first_emitter.data_interfaces[1],
        ),
    )
    rebound = _lower(ParticleGraphAsset(stable_id="data-graph", emitters=(rebound_emitter,)))

    assert rebound.kernel_hash != first.kernel_hash
    assert (
        classify_emitter_update(
            first.emitters[0],
            rebound.emitters[0],
            first_emitter.settings,
            rebound_emitter.settings,
        )
        is ParticleRuntimeCompatibility.PARAMETER_ONLY
    )

    reordered = _lower(
        ParticleGraphAsset(
            stable_id="data-graph",
            emitters=(
                replace(
                    first_emitter,
                    data_interfaces=tuple(reversed(first_emitter.data_interfaces)),
                ),
            ),
        )
    )
    assert reordered.kernel_hash == first.kernel_hash
    assert [interface.stable_id for interface in reordered.emitters[0].data_interfaces] == ["collision", "wind"]

    extended_emitter = replace(
        first_emitter,
        data_interfaces=first_emitter.data_interfaces + (VectorField(stable_id="turbulence"),),
    )
    extended = _lower(ParticleGraphAsset(stable_id="data-graph", emitters=(extended_emitter,)))
    assert (
        classify_emitter_update(
            first.emitters[0],
            extended.emitters[0],
            first_emitter.settings,
            extended_emitter.settings,
        )
        is ParticleRuntimeCompatibility.KERNEL_COMPATIBLE
    )


def test_enabling_collision_is_a_layout_migratable_kernel_change():
    previous_emitter = ParticleEmitterAsset(
        stable_id="collision-toggle",
        settings=EmitterSettings(collision_enabled=False),
    )
    next_emitter = replace(
        previous_emitter,
        settings=replace(previous_emitter.settings, collision_enabled=True),
    )
    previous = _lower(ParticleGraphAsset(emitters=(previous_emitter,))).emitters[0]
    current = _lower(ParticleGraphAsset(emitters=(next_emitter,))).emitters[0]

    assert (
        classify_emitter_update(
            previous,
            current,
            previous_emitter.settings,
            next_emitter.settings,
        )
        is ParticleRuntimeCompatibility.LAYOUT_MIGRATABLE
    )


def _attribute_cache_emitter(*fields):
    nodes = [GraphNodeRecord("root.init", "particle.root.init")]
    links = []
    for stable_id, name, value_type, value in fields:
        nodes.append(
            GraphNodeRecord(
                stable_id,
                "particle.attribute.cache",
                properties={
                    "name": name,
                    "value_type": value_type,
                    "value_space": "none",
                    "composition": "set",
                    "value": value,
                },
            )
        )
        links.append(
            GraphLinkRecord(
                f"root-{stable_id}",
                "root.init",
                "out",
                stable_id,
                "in",
                PortKind.EXEC,
            )
        )
    return ParticleEmitterAsset(
        stable_id="cache-hot-reload",
        init=GraphDocument("particle.init", nodes=nodes, links=links),
    )


def test_attribute_cache_addition_migrates_but_storage_type_change_restarts():
    previous_emitter = _attribute_cache_emitter(
        ("phase", "Phase", "f32", 0.5),
    )
    extended_emitter = _attribute_cache_emitter(
        ("phase", "Phase", "f32", 0.5),
        ("heat", "Heat", "f32", 0.25),
    )
    changed_type_emitter = _attribute_cache_emitter(
        ("phase", "Phase", "f32", 0.5),
        ("heat", "Heat", "vec3", [0.25, 0.5, 0.75]),
    )
    previous = _lower(ParticleGraphAsset(emitters=(previous_emitter,))).emitters[0]
    extended = _lower(ParticleGraphAsset(emitters=(extended_emitter,))).emitters[0]
    changed_type = _lower(ParticleGraphAsset(emitters=(changed_type_emitter,))).emitters[0]

    assert (
        classify_emitter_update(
            previous,
            extended,
            previous_emitter.settings,
            extended_emitter.settings,
        )
        is ParticleRuntimeCompatibility.LAYOUT_MIGRATABLE
    )
    assert (
        classify_emitter_update(
            extended,
            changed_type,
            extended_emitter.settings,
            changed_type_emitter.settings,
        )
        is ParticleRuntimeCompatibility.EMITTER_RESTART
    )


def test_event_queue_abi_change_restarts_the_owning_emitter():
    event_flow = ParticleEventFlow("impact", default_event_graph("impact"))
    previous_emitter = ParticleEmitterAsset(
        stable_id="event-abi",
        event_flows=(event_flow,),
    )
    next_emitter = replace(previous_emitter)
    previous = _lower(
        ParticleGraphAsset(
            emitters=(previous_emitter,),
            event_types=(ParticleEventType("impact", "Impact", 4),),
        )
    ).emitters[0]
    current = _lower(
        ParticleGraphAsset(
            emitters=(next_emitter,),
            event_types=(ParticleEventType("impact", "Impact", 8),),
        )
    ).emitters[0]

    assert (
        classify_emitter_update(
            previous,
            current,
            previous_emitter.settings,
            next_emitter.settings,
        )
        is ParticleRuntimeCompatibility.EMITTER_RESTART
    )


def test_emitter_seed_change_rebuilds_kernel_without_discarding_state():
    previous_emitter = ParticleEmitterAsset(
        stable_id="seeded-emitter",
        settings=EmitterSettings(seed=11),
    )
    next_emitter = replace(
        previous_emitter,
        settings=replace(previous_emitter.settings, seed=17),
    )
    previous = _lower(ParticleGraphAsset(emitters=(previous_emitter,))).emitters[0]
    current = _lower(ParticleGraphAsset(emitters=(next_emitter,))).emitters[0]

    assert previous.random_seed != current.random_seed
    assert (
        classify_emitter_update(
            previous,
            current,
            previous_emitter.settings,
            next_emitter.settings,
        )
        is ParticleRuntimeCompatibility.KERNEL_COMPATIBLE
    )
