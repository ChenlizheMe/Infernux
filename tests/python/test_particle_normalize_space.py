"""Normalize preserves its input space through authoring, lowering and GPU execution."""
import struct

import numpy as np
import pytest

from infernux.graph import GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind
from infernux.graph.types import CoordinateSpace, TypeRef, ValueType
from infernux.lib import _Infernux as native
from infernux.particle import (
    EmitterSettings, GpuParticleGlslLowerer, ParticleArtifactRegistry,
    ParticleEmitterAsset, ParticleGraphAsset, ParticleGraphCompiler,
    ParticleKernelLowerer, ParticleParameter, SimulationSpace,
    pack_gpu_particle_parameters,
)
from infernux.particle.nodes import particle_graph_node_definitions

SPACES = (CoordinateSpace.NONE, CoordinateSpace.WORLD,
          CoordinateSpace.SIMULATION, CoordinateSpace.EMITTER_LOCAL)
WIND = [2.0, 3.0, 4.0]


def _asset(space, *, normalized=True, chained=False):
    nodes = [
        GraphNodeRecord("root.update", "particle.root.update"),
        GraphNodeRecord("wind", "particle.parameter", properties={"parameter": "wind"}),
        GraphNodeRecord("store", "particle.attribute.velocity"),
    ]
    links = [GraphLinkRecord("exec", "root.update", "out", "store", "in", PortKind.EXEC)]
    source, pin = "wind", "value"
    if space in (CoordinateSpace.SIMULATION, CoordinateSpace.EMITTER_LOCAL):
        nodes.append(GraphNodeRecord("transform", "common.space.transform_direction",
                                     properties={"target_space": space.value}))
        links.append(GraphLinkRecord("transform-in", source, pin, "transform", "input"))
        source = "transform"
    if normalized:
        nodes.append(GraphNodeRecord("normal", "common.vector.normalize"))
        links.append(GraphLinkRecord("normal-in", source, pin, "normal", "value"))
        source, pin = "normal", "result"
    if chained:
        nodes.extend((GraphNodeRecord("sum", "common.math.add", properties={"b": 0.0}),
                      GraphNodeRecord("normal2", "common.vector.normalize")))
        links.extend((GraphLinkRecord("sum-in", source, pin, "sum", "a"),
                      GraphLinkRecord("normal2-in", "sum", "result", "normal2", "value")))
        source, pin = "normal2", "result"
    links.append(GraphLinkRecord("store-in", source, pin, "store", "value"))
    return ParticleGraphAsset(
        parameters=(ParticleParameter("wind", "Wind", TypeRef(ValueType.VEC3,
                    CoordinateSpace.NONE if space is CoordinateSpace.NONE else CoordinateSpace.WORLD), WIND),),
        emitters=(ParticleEmitterAsset(
            settings=EmitterSettings(simulation_space=SimulationSpace.LOCAL),
            update=GraphDocument("particle.update", tuple(nodes), tuple(links)),
        ),),
    )


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("normalized,chained", [(False, False), (True, False), (True, True)])
def test_normalize_space_survives_expression_kernel_and_spirv(space, normalized, chained):
    asset = _asset(space, normalized=normalized, chained=chained)
    hir = ParticleGraphCompiler().compile(asset)
    normals = [i for i in hir.emitters[0].update.expressions.instructions if i.opcode == "normalize"]
    assert len(normals) == int(normalized) + int(chained)
    for instruction in normals:
        assert instruction.result_type == instruction.operands[0].value_type == TypeRef(ValueType.VEC3, space)
    kernel = ParticleKernelLowerer().lower(hir)
    stages = GpuParticleGlslLowerer().lower(kernel).emitters[0].stages()
    compiled = native._compile_compute_glsl_batch(stages, "normalize-space")
    assert all(bytes(value)[:4] == b"\x03\x02\x23\x07" for value in compiled.values())


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("chained", [False, True])
def test_editor_infers_normalize_output_and_cache_space(space, chained):
    from infernux.engine.ui.graph_document_authoring import ParticleEmitterGraphAuthoringModel

    asset = _asset(space, chained=chained)
    model = ParticleEmitterGraphAuthoringModel(asset.emitters[0], definition_set=particle_graph_node_definitions(asset))
    source = model.find_node("update::normal2" if chained else "update::normal")
    port = model.definition_for_type(source.type_id).port("result")
    assert model._effective_port_type(source, port) == TypeRef(ValueType.VEC3, space)
    model.set_authoring_stage("update")
    model.prepare_node_creation("update")
    cache = model.add_node("particle.attribute.cache")
    assert model.add_link(source.uid, "result", cache.uid, "value") is not None
    assert (cache.data["value_type"], cache.data["value_space"]) == ("vec3", space.value)


@pytest.mark.parametrize("case", ["identity", "rotation", "combined"])
@pytest.mark.parametrize("space", SPACES)
def test_normalized_velocity_direction_executes_on_gpu(engine, space, case):
    # Reuse the dependency slicer from the collision tests: it executes production
    # lowered instructions, including Normalize and all inserted conversions.
    from test_particle_collision_space import _conversion_probe, _transform

    kernel = ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(_asset(space)))
    instructions = kernel.emitters[0].update.instructions
    if space is CoordinateSpace.SIMULATION:
        conversion = next(i for i in instructions if i.opcode == "normalize")
    else:
        conversion = next(i for i in reversed(instructions)
                          if i.opcode == "convert_space" and i.source.node_uid == "store")
    source = _conversion_probe(kernel, conversion)
    production_source = GpuParticleGlslLowerer().lower(kernel).emitters[0].update
    helpers = "\n".join(line for line in production_source.splitlines()
                        if line.startswith(("vec2 inx_safe_normalize(", "vec3 inx_safe_normalize(", "vec4 inx_safe_normalize(")))
    assert len(helpers.splitlines()) == 3
    source = source.replace("void main() {", helpers + "\nvoid main() {")
    spirv = native._compile_compute_glsl_batch({"probe": source}, "normalize-readback")["probe"]
    matrix, rotation = _transform(case)
    inverse = np.linalg.inv(matrix).astype(np.float32)
    direction = np.asarray(WIND, dtype=np.float32) / np.linalg.norm(WIND)
    expected = direction if space is CoordinateSpace.NONE else rotation.T @ direction
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


def test_spatial_normalize_asset_saves_and_reloads_compiled_artifacts(tmp_path, monkeypatch):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Wind.particlegraph"
    asset = _asset(CoordinateSpace.WORLD, chained=True)
    ParticleArtifactRegistry.clear()
    try:
        published = ParticleArtifactRegistry.save_graph_asset(asset, str(path))
        assert ParticleGraphAsset.load(str(path)) == asset
        assert ParticleArtifactRegistry.get(str(path)) is published
        assert published.hir["name"] == asset.name
        restored = ParticleGraphAsset.load(str(path))
        ParticleKernelLowerer().lower(ParticleGraphCompiler().compile(restored))
    finally:
        ParticleArtifactRegistry.clear()


def test_editor_inference_tracks_replaced_links_without_stale_space():
    from infernux.engine.ui.graph_document_authoring import ParticleEmitterGraphAuthoringModel

    asset = _asset(CoordinateSpace.WORLD)
    model = ParticleEmitterGraphAuthoringModel(asset.emitters[0], definition_set=particle_graph_node_definitions(asset))
    normal = model.find_node("update::normal")
    port = model.definition_for_type(normal.type_id).port("result")
    assert model._effective_port_type(normal, port).space is CoordinateSpace.WORLD
    assert model.remove_link("update::normal-in")
    assert model._effective_port_type(normal, port) == TypeRef(ValueType.VEC3)
    model.prepare_node_creation("update")
    scalar = model.add_node("common.constant.f32", value=2.0)
    assert model.add_link(scalar.uid, "value", normal.uid, "value")
    assert model._effective_port_type(normal, port) == TypeRef(ValueType.VEC3)


@pytest.mark.parametrize("kind", ["incomplete", "cycle", "wrong_type"])
def test_shared_editor_can_inspect_unfinished_expression_without_inventing_a_type(kind):
    from infernux.engine.ui.graph_document_authoring import GraphDocumentAuthoringModel

    nodes = [GraphNodeRecord("normal", "common.vector.normalize")]
    if kind == "cycle":
        nodes.append(GraphNodeRecord("source", "common.math.add"))
        links = [GraphLinkRecord("back", "normal", "result", "source", "a"),
                 GraphLinkRecord("forward", "source", "result", "normal", "value")]
    else:
        nodes.append(GraphNodeRecord("source", "common.vector.cross" if kind == "incomplete" else "common.constant.bool"))
        links = [GraphLinkRecord("forward", "source", "result" if kind == "incomplete" else "value", "normal", "value")]
    model = GraphDocumentAuthoringModel(GraphDocument("particle.expression", tuple(nodes), tuple(links)))
    normal = model.find_node("normal")
    port = model.definition_for_type(normal.type_id).port("result")
    assert model._effective_port_type(normal, port) is None
