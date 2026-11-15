"""Full material lighting after ordinary mesh instance and UV transformations."""

import json

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel
from tests.gpu.mesh_normal_map_case import MeshNormalMapCase, NORMAL, TANGENT, reference_basis


ARITHMETIC_SCALES = [
    (1.0, 1.0, 1.0),
    (2.0, 2.0, 2.0),
    (2.0, 0.5, 1.0),
    (-2.0, 0.5, 1.0),
    (2.0, -0.5, 1.0),
    (-2.0, -0.5, 1.0),
]


@pytest.fixture(scope="module")
def mesh_basis_arithmetic(engine, tmp_path_factory):
    from infernux.lib import _Infernux as native

    source = """import inx_geometry;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> values;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x * 6;
    InxVertexInput input = {};
    input.normal = values[i].xyz;
    input.tangent = values[i + 1];
    float3 scale = values[i + 2].xyz;
    float4x4 model = float4x4(scale.x,0,0,0, 0,scale.y,0,0, 0,0,scale.z,0, 0,0,0,1);
    float4x4 identity = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    InxInstanceData instance = {};
    InxMeshInput vertex = inx_transform_vertex(input, model, identity, identity, instance);
    values[i + 3] = float4(vertex.normal_ws, 0.0);
    values[i + 4] = vertex.tangent_ws;
    values[i + 5] = float4(cross(vertex.normal_ws, vertex.tangent_ws.xyz) * vertex.tangent_ws.w, 0.0);
}
"""
    program = native._compile_shader_program(source, "<mesh-basis>", "mesh_basis", targets=["spirv", "wgsl"])
    entry = program["targets"][0]["entries"][0]
    inputs = np.zeros((len(ARITHMETIC_SCALES) * 2, 6, 4), dtype=np.float32)
    for index, scale in enumerate(ARITHMETIC_SCALES):
        for mirror in (False, True):
            row = inputs[index * 2 + mirror]
            row[0, :3] = NORMAL
            row[1] = (*TANGENT, -1.0 if mirror else 1.0)
            row[2, :3] = scale
    host = engine._acquire_compute_host()
    kernel = host._create_kernel_from_compiler(bytes(entry["code"]), [(0, "storage")], 0, entry["name"])
    buffer = host.create_buffer(inputs.size // 4, "float32", 4)
    try:
        buffer.set_bytes(inputs.tobytes())
        kernel.dispatch([buffer], b"", len(inputs))
        kernel.wait()
        output = np.frombuffer(buffer.get_bytes(inputs.nbytes), dtype=np.float32).reshape(inputs.shape)
        results = output[:, 3:].copy()
    finally:
        kernel.wait()
        del buffer, kernel
        host._release_lease()
    (tmp_path_factory.mktemp("mesh-basis-arithmetic") / "results.json").write_text(
        json.dumps(results.tolist()), encoding="utf-8"
    )
    return results


@pytest.mark.parametrize("index,mirror_uv", [(index, mirror) for index in range(len(ARITHMETIC_SCALES)) for mirror in (False, True)])
def test_mesh_basis_matches_geometry_derivatives(mesh_basis_arithmetic, mirror_uv, index):
    normal, tangent = reference_basis(ARITHMETIC_SCALES[index], mirror_uv)
    bitangent = np.cross(normal, tangent[:3]) * tangent[3]
    expected = np.array([(*normal, 0), tangent, (*bitangent, 0)])
    actual = mesh_basis_arithmetic[index * 2 + mirror_uv]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(actual[:, :3], axis=1), 1.0, atol=2e-5, rtol=0.0)
    assert abs(np.dot(actual[0, :3], actual[1, :3])) < 2e-5


@pytest.mark.parametrize(
    "mirror_uv,pipeline,samples", [(False, "forward", 4), (False, "forward_plus", 1), (True, "deferred", 1)]
)
def test_uv0_normal_mapping_matches_cpu_baked_geometry(
    engine, scene, tmp_path, monkeypatch, pipeline, samples, mirror_uv
):
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    case = None
    console = ConsolePanel()
    previous_rendering = engine.is_play_mode_rendering()
    try:
        case = MeshNormalMapCase(engine, scene, tmp_path, pipeline, samples, mirror_uv)
        scene.main_camera = case.camera
        engine.resize_game_render_target(320, 240)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(case.pipeline)
        engine.set_play_mode_rendering(True)
        phase = frame = 0
        mesh_ready_execution = None
        ticket = None
        failures = []

        def after_draw():
            nonlocal phase, frame, mesh_ready_execution, ticket
            try:
                frame += 1
                assert frame < 400, case.results
                if ticket is None:
                    execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                    if engine.pending_mesh_gpu_upload_count or engine.pending_texture_gpu_upload_count:
                        mesh_ready_execution = None
                    elif mesh_ready_execution is None:
                        try:
                            engine.get_object_mesh_index_gpu_info(case.owner.id)
                        except IndexError:
                            return
                        mesh_ready_execution = execution
                    elif engine.game_presentation_ready and execution > mesh_ready_execution:
                        ticket = engine.request_render_target_readback(True)
                elif ticket.done:
                    case.observe(ticket.result_numpy())
                    (tmp_path / "normal-map-results.json").write_text(json.dumps(case.results), encoding="utf-8")
                    issues = [
                        item
                        for item in console._get_visible_log_snapshot(1000)
                        if item["level"] in ("ERROR", "FATAL", "WARN", "WARNING")
                    ]
                    assert not issues, issues
                    phase += 1
                    if phase == case.phases:
                        engine.exit()
                    else:
                        case.change(phase)
                        mesh_ready_execution, ticket = None, None
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert phase == case.phases
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(previous_rendering)
        if case:
            case.close()
