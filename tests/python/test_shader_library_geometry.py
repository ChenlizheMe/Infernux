"""Public typed GPU helpers through real authored-material import and Scene draws."""

import json

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel
from tests.gpu.library_compile import compile_library_compute
from tests.gpu.shader_library_case import ROTATIONS, ShaderLibraryCase, quaternion_reference


@pytest.fixture(scope="module")
def rotation_arithmetic(engine, tmp_path_factory):
    entry = compile_library_compute("geometry", """
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> values;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x * 3;
    values[i + 2] = float4(LIBRARY::rotate_axis(values[i].xyz, values[i + 1].xyz, values[i + 1].w), 0.0);
}
""", "library-rotation")
    host = engine._acquire_compute_host()
    kernel = host._create_kernel_from_compiler(bytes(entry["code"]), [(0, "storage")], 0, entry["name"])
    inputs = np.zeros((len(ROTATIONS) * 3, 4), dtype=np.float32)
    for index, (value, axis, angle) in enumerate(ROTATIONS):
        inputs[index * 3, :3] = value
        inputs[index * 3 + 1] = (*axis, angle)
    buffer = host.create_buffer(len(inputs), "float32", 4)
    try:
        buffer.set_bytes(inputs.tobytes())
        kernel.dispatch([buffer], b"", len(ROTATIONS))
        kernel.wait()
        values = np.frombuffer(buffer.get_bytes(inputs.nbytes), dtype=np.float32).reshape(-1, 4)
        results = values[2::3, :3].copy()
    finally:
        kernel.wait()
        del buffer, kernel
        host._release_lease()
    (tmp_path_factory.mktemp("rotation-arithmetic") / "results.json").write_text(
        json.dumps(results.tolist()), encoding="utf-8"
    )
    return results


@pytest.mark.parametrize("index", range(len(ROTATIONS)))
def test_rotation_direction_and_axis_scale(rotation_arithmetic, index):
    actual = rotation_arithmetic[index]
    expected = quaternion_reference(*ROTATIONS[index])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(ROTATIONS[index][0]), atol=2e-5, rtol=0.0)


@pytest.mark.parametrize("kind", ["rotation", "cross", "depth"])
def test_authored_material_library_geometry(engine, scene, tmp_path, monkeypatch, kind):
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    case = None
    console = ConsolePanel()
    previous_rendering = engine.is_play_mode_rendering()
    try:
        case = ShaderLibraryCase(engine, scene, tmp_path, kind)
        scene.main_camera = case.camera
        engine.resize_game_render_target(128, 128)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(case.pipeline)
        engine.set_play_mode_rendering(True)
        phase = frame = changed = 0
        ticket = None
        failures = []

        def after_draw():
            nonlocal phase, frame, changed, ticket
            try:
                frame += 1
                assert frame < 400, case.results
                execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                if ticket is None and engine.game_presentation_ready and execution >= changed + 6:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    case.observe(ticket.result_numpy())
                    (tmp_path / "library-results.json").write_text(json.dumps(case.results), encoding="utf-8")
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
                        ticket, changed = None, execution
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


def test_standalone_material_draws_use_resident_program_until_explicit_reload(engine, scene, tmp_path, monkeypatch):
    from infernux.core.shader import Shader
    from infernux.lib import Vector3

    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    case = ShaderLibraryCase(engine, scene, tmp_path, "cross")
    source = tmp_path / "library.frag.py"
    parked = tmp_path / "library.offline"
    red = '''import infernux.shader as shader
class Resident(shader.frag):
    shader_id = "tests/library/resident"
    cull_mode = "none"
    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        return shader.vec4(1.0, 0.0, 0.0, 1.0)
'''
    console = ConsolePanel()
    previous_rendering = engine.is_play_mode_rendering()
    phase = frame = changed = 0
    ticket = None
    failures = []

    def after_draw():
        nonlocal phase, frame, changed, ticket
        try:
            frame += 1
            assert frame < 100
            # Keep submitting changed camera data instead of reusing a static list.
            case.camera.game_object.transform.position = Vector3(0.001 * (frame % 2), 0, -3)
            execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
            if ticket is None and engine.game_presentation_ready and execution >= changed + 6:
                ticket = engine.request_render_target_readback(True)
            elif ticket is not None and ticket.done:
                pixel = ticket.result_numpy()[64, 64]
                np.testing.assert_allclose(pixel, (0, 1, 0, 1) if phase == 2 else (1, 0, 0, 1), atol=.002)
                if phase == 0:
                    source.rename(parked)
                elif phase == 1:
                    parked.rename(source)
                    source.write_text(red.replace("vec4(1.0, 0.0, 0.0, 1.0)", "vec4(0.0, 1.0, 0.0, 1.0)"), encoding="utf-8")
                    assert Shader.reload(str(source))
                else:
                    engine.exit()
                phase += 1
                ticket, changed = None, execution
        except BaseException as error:
            failures.append(error)
            engine.exit()

    try:
        source.write_text(red, encoding="utf-8")
        assert AssetManager.reimport_asset(str(source))
        case.material.frag_shader_name = "tests/library/resident"
        scene.main_camera = case.camera
        engine.resize_game_render_target(128, 128)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(case.pipeline)
        engine.set_play_mode_rendering(True)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert phase == 3
        assert console.get_error_count() == console.get_warning_count() == 0
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(previous_rendering)
        if parked.exists():
            parked.rename(source)
        case.close()
