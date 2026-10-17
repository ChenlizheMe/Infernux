from __future__ import annotations

from test_particle_graph_hir import (
    AssetReference,
    EmitterSettings,
    EmitterShape,
    GraphDocument,
    GraphLinkRecord,
    GraphNodeRecord,
    KernelCompileError,
    MeshEmissionMode,
    PARTICLE_SCRIPT_EVENT_SOURCE,
    PARTICLE_SCRIPT_SOURCE,
    ParticleArtifactError,
    ParticleArtifactRegistry,
    ParticleArtifactSuperseded,
    ParticleAttribute,
    ParticleBurst,
    ParticleCompileError,
    ParticleEmitterAsset,
    ParticleGraphAsset,
    ParticleGraphCompiler,
    ParticleGraphSchemaError,
    ParticleKernelLowerer,
    ParticleParameter,
    ParticleRuntimeMetadataError,
    ParticleScriptCompiler,
    ParticleScriptError,
    Path,
    PortKind,
    TypeRef,
    ValueType,
    VectorField,
    _operations,
    copy,
    decode_particle_runtime_metadata,
    json,
    particle_cache_attributes,
    pytest,
    replace,
    threading,
)


def test_particle_script_compiles_without_execution_to_same_hir_contract():
    compiler = ParticleScriptCompiler()
    asset = compiler.parse(PARTICLE_SCRIPT_SOURCE, source_name="Smoke.particle.py")
    program = compiler.compile(PARTICLE_SCRIPT_SOURCE, source_name="Smoke.particle.py")
    emitter = program.emitters[0]

    assert asset.stable_id == "smoke-graph"
    assert program.schedule.emitter_ids == ("smoke",)
    assert emitter.settings.spawn_rate_over_distance == 3.0
    assert emitter.settings.duration == 4.0
    assert emitter.settings.loop is False
    assert emitter.settings.start_delay == 0.25
    assert not hasattr(emitter, "enabled")
    assert not hasattr(emitter, "play_on_start")
    assert [operation.opcode for operation in _operations(emitter.init)] == [
        "emitter.sample_shape",
        "attribute.modify_velocity",
        "attribute.modify_lifetime",
        "attribute.modify_rotation",
    ]
    assert [operation.opcode for operation in _operations(emitter.update)[-2:]] == [
        "attribute.modify_velocity",
        "attribute.modify_rotation",
    ]
    assert [operation.opcode for operation in _operations(emitter.rendering)[:2]] == [
        "attribute.modify_lifetime",
        "attribute.modify_flipbook_frame",
    ]
    assert any(instruction.opcode == "normalized_age" for instruction in emitter.rendering.expressions.instructions)
    assert emitter.render_plan.outputs[0].receive_scene_lighting is True
    assert emitter.render_plan.outputs[0].receive_shadows is True
    assert emitter.render_plan.outputs[0].sprite_alignment == "velocity"
    assert [interface.stable_id for interface in emitter.data_interfaces] == [
        "wind-field",
    ]
    assert program.behavior_hash == ParticleGraphCompiler().compile(asset).behavior_hash


def test_particle_script_typed_events_lower_through_the_graph_event_abi():
    compiler = ParticleScriptCompiler()
    asset = compiler.parse(PARTICLE_SCRIPT_EVENT_SOURCE, source_name="Impact.particle.py")
    hir = ParticleGraphCompiler().compile(asset)
    kernel = ParticleKernelLowerer().lower(hir)

    assert [value.stable_id for value in asset.event_types] == ["impact"]
    assert [value.event_id for value in asset.emitters[0].event_flows] == ["impact"]
    assert [value.flow_id for value in hir.emitters[0].event_flows] == ["impact"]
    assert any(instruction.opcode == "event_enqueue" for instruction in kernel.emitters[0].update.instructions)
    payloads = [
        instruction for instruction in kernel.emitters[0].update.instructions if instruction.opcode == "event_payload"
    ]
    assert [value.result_type.value_type.value for value in payloads] == [
        "f32",
        "color",
    ]


def test_ribbon_output_has_stable_topology_attributes_and_script_parity():
    source = """
from infernux.particle import AssetReference, ParticleScript, ParticleEmitter, EmitterSettings

class TrailGraph(ParticleScript):
    class Trail(ParticleEmitter):
        stable_id = "trail"
        settings = EmitterSettings(capacity=4096)

        def init(self, ctx, particles):
            particles.set_strip_id(7)
            particles.set_ribbon_order(11)
            particles.break_ribbon(False)

        def update(self, ctx, particles):
            particles.set_ribbon_order(12)

        def rendering(self, ctx, particles):
            particles.ribbon(
                shader="Particle Unlit",
                uv_mode="repeat",
                uv_scale=2.5,
            )
"""
    compiler = ParticleScriptCompiler()
    asset = compiler.parse(source, source_name="Trail.particle.py")
    program = compiler.compile(source, source_name="Trail.particle.py")
    graph_program = ParticleGraphCompiler().compile(asset)
    emitter = program.emitters[0]
    output = emitter.render_plan.outputs[0]

    assert program.behavior_hash == graph_program.behavior_hash
    assert output.output_type == "ribbon"
    assert output.ribbon_uv_mode == "repeat"
    assert output.ribbon_uv_scale == pytest.approx(2.5)
    assert output.sort_mode == "none"
    assert [operation.opcode for operation in _operations(emitter.init)[-3:]] == [
        "attribute.modify_strip_id",
        "attribute.modify_ribbon_order",
        "attribute.modify_ribbon_break",
    ]
    assert {
        "builtin.ribbon_strip_id",
        "builtin.ribbon_order",
        "builtin.ribbon_break",
    }.issubset({attribute.stable_id for attribute in emitter.attributes})


def test_particle_graph_and_script_save_to_equivalent_aot_artifacts(tmp_path, monkeypatch):
    from infernux.engine import project_context

    ParticleArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    graph_path = tmp_path / "Assets" / "Smoke.particlegraph"
    script_path = tmp_path / "Assets" / "Smoke.particle.py"
    graph_path.parent.mkdir()
    script_path.write_text(PARTICLE_SCRIPT_SOURCE, encoding="utf-8")
    graph_asset = ParticleScriptCompiler().parse(
        PARTICLE_SCRIPT_SOURCE,
        source_name=str(script_path),
    )
    graph_asset.save(str(graph_path))

    graph_artifact = ParticleArtifactRegistry.get(str(graph_path))
    script_artifact = ParticleArtifactRegistry.compile_path(str(script_path))

    assert graph_artifact is not None
    assert graph_artifact.behavior_hash == script_artifact.behavior_hash
    assert graph_artifact.hir["schedule"] == ["smoke"]
    assert graph_artifact.kernel_ir["$schema"] == "infernux.particle_kernel_ir"
    assert graph_artifact.kernel_ir["source_behavior_hash"] == graph_artifact.behavior_hash
    assert graph_artifact.kernel_ir["kernel_hash"] == script_artifact.kernel_ir["kernel_hash"]
    assert graph_artifact.gpu_glsl["$schema"] == "infernux.particle_gpu_glsl"
    assert graph_artifact.gpu_glsl["kernel_hash"] == graph_artifact.kernel_ir["kernel_hash"]
    assert [value["stable_id"] for value in graph_artifact.gpu_glsl["emitters"][0]["data_interfaces"]] == ["wind-field"]
    assert set(graph_artifact.gpu_glsl["emitters"][0]["stages"]) == {
        "bootstrap",
        "init",
        "update",
        "contact_prepare",
        "contact_solve",
        "contact_dispatch",
        "render_reset",
        "rendering",
    }
    assert graph_artifact.gpu_spirv["target"] == "vulkan1.2-spirv1.5"
    assert graph_artifact.gpu_spirv["kernel_hash"] == graph_artifact.kernel_ir["kernel_hash"]
    assert set(graph_artifact.gpu_spirv["emitters"][0]["stages"]) == set(
        graph_artifact.gpu_glsl["emitters"][0]["stages"]
    )
    assert set(graph_artifact.gpu_spirv["billboard"]) == {
        "vertex",
        "picking_fragment",
        "motion_vertex",
        "motion_fragment",
    }
    assert set(graph_artifact.gpu_spirv["mesh"]) == {
        "vertex",
        "shadow_fragment",
        "picking_fragment",
        "motion_vertex",
        "motion_fragment",
    }
    from infernux.particle import decode_gpu_particle_spirv

    decoded = decode_gpu_particle_spirv(graph_artifact.gpu_spirv, 0)
    assert decoded["stable_id"] == "smoke"
    assert set(decoded["stages"]) == set(graph_artifact.gpu_glsl["emitters"][0]["stages"])
    assert set(decoded["billboard"]) == {
        "vertex",
        "picking_fragment",
        "motion_vertex",
        "motion_fragment",
    }
    assert set(decoded["mesh"]) == {
        "vertex",
        "shadow_fragment",
        "picking_fragment",
        "motion_vertex",
        "motion_fragment",
    }
    assert all(binary[:4] == b"\x03\x02#\x07" for binary in decoded["stages"].values())
    assert all(binary[:4] == b"\x03\x02#\x07" for binary in decoded["billboard"].values())
    assert all(binary[:4] == b"\x03\x02#\x07" for binary in decoded["mesh"].values())
    assert script_artifact.hir["emitters"][0]["render_plan"][0]["receive_scene_lighting"] is True
    assert script_artifact.hir["emitters"][0]["render_plan"][0]["cast_shadows"] is False
    runtime_metadata = decode_particle_runtime_metadata(script_artifact.hir)
    assert runtime_metadata.emitters[0].outputs[0].cast_shadows is False
    stale_hir = copy.deepcopy(script_artifact.hir)
    stale_hir["emitters"][0]["render_plan"][0].pop("sprite_alignment")
    with pytest.raises(ParticleRuntimeMetadataError, match="is invalid"):
        decode_particle_runtime_metadata(stale_hir)
    assert graph_artifact.artifact_path.endswith("smoke-graph.inxparticle")


def test_particle_runtime_index_loads_aot_without_authoring_source(tmp_path, monkeypatch):
    from infernux.engine import project_context

    ParticleArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    source_path = tmp_path / "Assets" / "Smoke.particle.py"
    source_path.parent.mkdir()
    source_path.write_text(PARTICLE_SCRIPT_SOURCE, encoding="utf-8")
    compiled = ParticleArtifactRegistry.compile_path(str(source_path), guid="smoke-guid")
    assert Path(compiled.artifact_path).is_file()

    runtime_index = Path(compiled.artifact_path).parent / "RuntimeIndex.json"
    runtime_index.write_text(
        json.dumps(
            {
                "$schema": "infernux.particle_runtime_index",
                "entries": [
                    {
                        "guid": "smoke-guid",
                        "path_hint": "Assets/Smoke.particle.py",
                        "stable_id": Path(compiled.artifact_path).stem,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_path.unlink()
    ParticleArtifactRegistry.clear()

    restored = ParticleArtifactRegistry.load_runtime_reference(guid="smoke-guid")

    assert restored is not None
    assert restored.source_kind == "script"
    assert restored.behavior_hash == compiled.behavior_hash
    assert restored.artifact_path == compiled.artifact_path


def test_particle_graph_rebuilds_deterministically_after_library_cleanup(tmp_path, monkeypatch):
    from infernux.engine import project_context

    ParticleArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Rebuild.particlegraph"
    asset = ParticleGraphAsset(stable_id="deterministic-rebuild")
    asset.save(str(path))
    first = ParticleArtifactRegistry.get(str(path))
    assert first is not None

    artifact_path = Path(first.artifact_path)
    artifact_path.unlink()
    runtime_index = artifact_path.parent / "RuntimeIndex.json"
    runtime_index.unlink()
    ParticleArtifactRegistry.clear()

    rebuilt = ParticleArtifactRegistry.compile_path(str(path))

    assert Path(rebuilt.artifact_path).is_file()
    assert runtime_index.is_file()
    assert rebuilt.source_hash == first.source_hash
    assert rebuilt.semantic_hash == first.semantic_hash
    assert rebuilt.behavior_hash == first.behavior_hash
    assert rebuilt.hir == first.hir
    assert rebuilt.kernel_ir == first.kernel_ir
    assert rebuilt.gpu_glsl == first.gpu_glsl
    assert rebuilt.gpu_spirv == first.gpu_spirv


def test_particle_graph_latest_request_wins_out_of_order_compilation(tmp_path, monkeypatch):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = str(tmp_path / "Assets" / "LatestWins.particlegraph")
    older = ParticleGraphAsset(stable_id="latest-wins", name="Older")
    newer = replace(older, name="Newer")
    old_started = threading.Event()
    release_old = threading.Event()
    old_results = []
    original_compile = ParticleArtifactRegistry._compile_graph_asset

    def controlled_compile(cls, asset, source_path):
        if asset.name == "Older":
            old_started.set()
            assert release_old.wait(timeout=5.0)
        return original_compile(asset, source_path)

    monkeypatch.setattr(
        ParticleArtifactRegistry,
        "_compile_graph_asset",
        classmethod(controlled_compile),
    )
    ParticleArtifactRegistry.clear()

    def save_older():
        with pytest.raises(ParticleArtifactSuperseded, match="superseded") as rejected:
            ParticleArtifactRegistry.save_graph_asset(older, path)
        old_results.append(rejected.value)

    old_thread = threading.Thread(target=save_older)
    old_thread.start()
    assert old_started.wait(timeout=5.0)
    latest = ParticleArtifactRegistry.save_graph_asset(newer, path)
    release_old.set()
    old_thread.join(timeout=5.0)

    assert not old_thread.is_alive()
    assert len(old_results) == 1
    assert ParticleArtifactRegistry.get(path) is latest
    assert latest.hir["name"] == "Newer"
    assert ParticleGraphAsset.load(path).name == "Newer"
    assert json.loads(Path(latest.artifact_path).read_text(encoding="utf-8"))["hir"]["name"] == "Newer"


def test_failed_newer_particle_compile_invalidates_older_in_flight_result(tmp_path, monkeypatch):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = str(tmp_path / "Assets" / "FailedLatest.particlegraph")
    baseline = ParticleGraphAsset(stable_id="failed-latest", name="Baseline")
    older = replace(baseline, name="Older In Flight")
    broken = replace(baseline, name="Broken Newer")
    ParticleArtifactRegistry.clear()
    published = ParticleArtifactRegistry.save_graph_asset(baseline, path)
    old_started = threading.Event()
    release_old = threading.Event()
    old_results = []
    original_compile = ParticleArtifactRegistry._compile_graph_asset

    def controlled_compile(cls, asset, source_path):
        if asset.name == "Older In Flight":
            old_started.set()
            assert release_old.wait(timeout=5.0)
        if asset.name == "Broken Newer":
            raise ValueError("intentional latest revision failure")
        return original_compile(asset, source_path)

    monkeypatch.setattr(
        ParticleArtifactRegistry,
        "_compile_graph_asset",
        classmethod(controlled_compile),
    )

    def save_older():
        with pytest.raises(ParticleArtifactSuperseded, match="superseded") as rejected:
            ParticleArtifactRegistry.save_graph_asset(older, path)
        old_results.append(rejected.value)

    old_thread = threading.Thread(target=save_older)
    old_thread.start()
    assert old_started.wait(timeout=5.0)
    with pytest.raises(ParticleArtifactError, match="intentional latest revision failure"):
        ParticleArtifactRegistry.save_graph_asset(broken, path)
    release_old.set()
    old_thread.join(timeout=5.0)

    assert not old_thread.is_alive()
    assert len(old_results) == 1
    assert ParticleArtifactRegistry.get(path) is published
    assert published.hir["name"] == "Baseline"
    assert ParticleGraphAsset.load(path).name == "Baseline"
    assert json.loads(Path(published.artifact_path).read_text(encoding="utf-8"))["hir"]["name"] == "Baseline"


def test_particle_aot_failure_preserves_last_known_good_and_cache_hit(tmp_path, monkeypatch):
    from infernux.engine import project_context
    from infernux.particle import artifact as artifact_module

    ParticleArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Smoke.particlegraph"
    path.parent.mkdir()
    asset = ParticleGraphAsset(stable_id="smoke-cache")
    asset.save(str(path))
    published = ParticleArtifactRegistry.get(str(path))
    assert published is not None

    path.write_text('{"broken": true}', encoding="utf-8")
    with pytest.raises(ParticleArtifactError, match="AOT compile failed"):
        ParticleArtifactRegistry.compile_path(str(path))
    assert ParticleArtifactRegistry.get(str(path)) == published

    asset.save(str(path))
    current = ParticleArtifactRegistry.get(str(path))
    ParticleArtifactRegistry.clear()

    def fail_compile(_self, _asset):
        raise AssertionError("matching particle artifact should bypass HIR compilation")

    monkeypatch.setattr(artifact_module.ParticleGraphCompiler, "compile", fail_compile)
    restored = ParticleArtifactRegistry.compile_path(str(path))

    assert restored.source_hash == current.source_hash
    assert restored.behavior_hash == current.behavior_hash


def test_particle_graph_save_does_not_replace_valid_source_with_invalid_draft(tmp_path):
    path = tmp_path / "Smoke.particlegraph"
    valid = ParticleGraphAsset(stable_id="atomic-particle-save")
    valid.save(str(path))
    source_before = path.read_bytes()

    emitter = valid.emitters[0]
    update = emitter.update
    invalid_update = GraphDocument(
        update.domain,
        update.nodes
        + (
            GraphNodeRecord("noise", "common.noise.vector3d", properties={}),
            GraphNodeRecord("invalid.acceleration", "particle.attribute.velocity"),
        ),
        update.links
        + (
            GraphLinkRecord(
                "root-to-invalid-acceleration",
                "root.update",
                "out",
                "invalid.acceleration",
                "in",
                PortKind.EXEC,
            ),
            GraphLinkRecord(
                "noise-to-acceleration",
                "noise",
                "value",
                "invalid.acceleration",
                "value",
                PortKind.VALUE,
            ),
        ),
        update.metadata,
    )
    invalid = replace(valid, emitters=(replace(emitter, update=invalid_update),))

    with pytest.raises(ParticleArtifactError, match="required input noise.position"):
        invalid.save(str(path))

    assert path.read_bytes() == source_before
