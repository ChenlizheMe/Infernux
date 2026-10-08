"""WhiteBalanceEffect must preserve neutral HDR color and use the library math."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from tests.gpu.white_balance_case import (
    COLORS, INPUT_SHADER, LIBRARY_SHADER, MONITOR_SHADER, PHASES,
    WhiteBalancePipeline, check_pixels, expected_linear,
)


PARAMETERS = sorted({(t, tint) for _, t, tint in PHASES} |
                    {(t, tint) for t in (-100., -50., 0., 50., 100.) for tint in (-100., -50., 0., 50., 100.)})


@pytest.fixture(scope='module')
def white_balance_arithmetic(engine, tmp_path_factory):
    from infernux.lib import _Infernux as native
    source = (Path(__file__).parents[2] / 'python/infernux/resources/shaders/lib/color.glsl').read_text(encoding='utf-8')
    functions = source[source.index('vec3 _whiteBalanceWhiteLMS'):source.index('// Channel Mixer')]
    shader = '''#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Values { vec4 values[]; };
layout(push_constant) uniform Params { float temperature; float tint; } pc;
''' + functions + '''
void main() {
    uint i = gl_GlobalInvocationID.x;
    values[i].rgb = whiteBalance(values[i].rgb, pc.temperature, pc.tint);
}
'''
    spirv = native._compile_compute_glsl_batch({'white_balance':shader}, 'white-balance-arithmetic')['white_balance']
    host = engine._acquire_compute_host()
    kernel = host.create_kernel(spirv, 1, 8)
    buffer = host.create_buffer(16, 'float32', 4)
    inputs = np.tile(COLORS.astype(np.float16).astype(np.float32), (2, 1))
    inputs[8:, :3] *= .5
    results = {}
    try:
        for temperature, tint in PARAMETERS:
            buffer.set_bytes(inputs.tobytes())
            kernel.dispatch([buffer], struct.pack('ff', temperature, tint), 16)
            kernel.wait()
            results[temperature, tint] = np.frombuffer(buffer.get_bytes(inputs.nbytes), dtype=np.float32).copy().reshape(16, 4)
    finally:
        kernel.wait()
        del kernel, buffer
        host._release_lease()
    directory = tmp_path_factory.mktemp('white-balance-arithmetic')
    (directory / 'results.json').write_text(json.dumps([
        dict(temperature=t, tint=tint, values=values.tolist()) for (t,tint),values in results.items()
    ]), encoding='utf-8')
    return results


@pytest.mark.parametrize('temperature,tint', PARAMETERS)
def test_white_balance_gpu_math(white_balance_arithmetic, temperature, tint):
    values = white_balance_arithmetic[temperature, tint]
    assert np.isfinite(values).all() and (values >= 0).all()
    np.testing.assert_array_equal(values[0], np.zeros(4))
    np.testing.assert_array_equal(values[:, 3], np.tile(COLORS[:, 3], 2))
    np.testing.assert_allclose(values[8:, :3], values[:8, :3] * .5, rtol=1e-5, atol=1e-6)
    if temperature == tint == 0:
        np.testing.assert_allclose(values[:8], COLORS.astype(np.float16).astype(np.float32), rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(values[:8], expected_linear(temperature, tint), rtol=3e-4, atol=4e-4)
        if max(abs(temperature), abs(tint)) <= .001:
            np.testing.assert_allclose(values, white_balance_arithmetic[0., 0.], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('library', [False, True], ids=['postprocess', 'color-library'])
def test_white_balance_fullscreen_parameter_domain(engine, scene, tmp_path, monkeypatch, library):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    assets = []
    pipeline = WhiteBalancePipeline(library=library)
    console = ConsolePanel()
    manager = SceneManager.instance()
    try:
        for name, source in [('Input', INPUT_SHADER), ('Library', LIBRARY_SHADER), ('Monitor', MONITOR_SHADER)]:
            path = tmp_path / (name + '.frag')
            path.write_text(source, encoding='utf-8')
            result = AssetManager.import_asset(str(path), database=database)
            assert result.succeeded, result.error
            assets.append(path)
        camera = scene.create_game_object('White Balance Camera')
        camera.transform.position = Vector3(0, 0, -5)
        camera.add_component('Camera')
        engine.resize_game_render_target(64, 32)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(pipeline)
        manager.play()
        manager.pause()
        phase = frame = changed = 0
        ticket = None
        samples, failures = [], []

        def after_draw():
            nonlocal frame, changed, phase, ticket
            try:
                frame += 1
                assert frame < 500, samples
                if ticket is None and frame >= changed + 6:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy()
                    values = check_pixels(pixels, PHASES[phase])
                    samples.append(dict(phase=PHASES[phase], values=values))
                    (tmp_path / 'white-balance-pixels.json').write_text(json.dumps(samples), encoding='utf-8')
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                    assert not issues, issues
                    phase += 1
                    if phase == len(PHASES):
                        np.testing.assert_array_equal(samples[0]['values'], samples[-1]['values'])
                        np.testing.assert_allclose(samples[0]['values'], samples[1]['values'], atol=.001, rtol=.001)
                        engine.exit()
                    else:
                        pipeline.change(PHASES[phase])
                        changed, ticket = frame, None
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert phase == len(PHASES)
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        pipeline.dispose()
        manager.stop()
        for path in assets:
            assert database.delete_asset(str(path))
