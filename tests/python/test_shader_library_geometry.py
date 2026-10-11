"""Public GLSL helpers through real authored-material import and Scene draws."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager
from tests.gpu.shader_library_case import ROTATIONS, ShaderLibraryCase, quaternion_reference


@pytest.fixture(scope='module')
def rotation_arithmetic(engine, tmp_path_factory):
    from infernux.lib import _Infernux as native
    root = Path(__file__).parents[2] / 'python/infernux/resources/shaders/lib'
    functions = []
    for name, signature in [('common.glsl','vec3 rotateAboutAxis('), ('vertex_utils.glsl','vec3 rotateAroundAxis(')]:
        source = (root / name).read_text(encoding='utf-8')
        start = source.index(signature)
        opening = source.index('{', start)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        functions.append(source[start:end])
    source = '''#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Values { vec4 values[]; };
layout(push_constant) uniform Params { int mode; } pc;
''' + '\n'.join(functions) + '''
void main() {
    uint i = gl_GlobalInvocationID.x * 3;
    vec3 value = values[i].xyz;
    vec4 axisAngle = values[i+1];
    values[i+2] = vec4(pc.mode == 0 ? rotateAboutAxis(value, axisAngle.xyz, axisAngle.w)
                                  : rotateAroundAxis(value, axisAngle.xyz, axisAngle.w), 0);
}
'''
    spirv = native._compile_compute_glsl_batch({'rotation':source}, 'library-rotation')['rotation']
    host = engine._acquire_compute_host()
    kernel = host.create_kernel(spirv, 1, 4)
    inputs = np.zeros((len(ROTATIONS)*3, 4), dtype=np.float32)
    for index, (value, axis, angle) in enumerate(ROTATIONS):
        inputs[index*3,:3] = value
        inputs[index*3+1] = (*axis, angle)
    buffer = host.create_buffer(len(inputs), 'float32', 4)
    results = []
    try:
        for mode in (0,1):
            buffer.set_bytes(inputs.tobytes())
            kernel.dispatch([buffer], struct.pack('i', mode), len(ROTATIONS))
            kernel.wait()
            values = np.frombuffer(buffer.get_bytes(inputs.nbytes), dtype=np.float32).reshape(-1,4)
            results.append(values[2::3,:3].copy())
    finally:
        kernel.wait()
        del buffer, kernel
        host._release_lease()
    (tmp_path_factory.mktemp('rotation-arithmetic') / 'results.json').write_text(
        json.dumps([values.tolist() for values in results]), encoding='utf-8')
    return results


@pytest.mark.parametrize('mode', [0,1], ids=['common','vertex-utils'])
@pytest.mark.parametrize('index', range(len(ROTATIONS)))
def test_rotation_direction_and_axis_scale(rotation_arithmetic, mode, index):
    actual = rotation_arithmetic[mode][index]
    expected = quaternion_reference(*ROTATIONS[index])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0.)
    np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(ROTATIONS[index][0]), atol=2e-5, rtol=0.)


@pytest.mark.parametrize('kind', ['rotation', 'cross'])
def test_authored_material_library_geometry(engine, scene, tmp_path, monkeypatch, kind):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = None
    manager, console = SceneManager.instance(), ConsolePanel()
    previous_rendering = engine.is_play_mode_rendering()
    try:
        case = ShaderLibraryCase(engine, scene, tmp_path, kind)
        scene.main_camera = case.camera
        engine.resize_game_render_target(128,128)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(case.pipeline)
        manager.play()
        manager.pause()
        engine.set_play_mode_rendering(True)
        phase = frame = changed = 0
        ticket = None
        failures = []
        def after_draw():
            nonlocal phase, frame, changed, ticket
            try:
                frame += 1
                assert frame < 400, case.results
                if ticket is None and frame >= changed + 6:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    case.observe(ticket.result_numpy())
                    (tmp_path / 'library-results.json').write_text(json.dumps(case.results), encoding='utf-8')
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item['level'] in ('ERROR','FATAL','WARN','WARNING')]
                    assert not issues, issues
                    phase += 1
                    if phase == case.phases:
                        engine.exit()
                    else:
                        case.change(phase)
                        ticket, changed = None, frame
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
        manager.stop()
        engine.set_play_mode_rendering(previous_rendering)
        if case:
            case.close()
