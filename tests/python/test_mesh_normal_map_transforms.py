"""Full material lighting after ordinary mesh instance and UV transformations."""
import json
from pathlib import Path

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager
from tests.gpu.mesh_normal_map_case import MeshNormalMapCase, NORMAL, TANGENT, reference_basis


ARITHMETIC_SCALES = [(1.,1.,1.), (2.,2.,2.), (2.,.5,1.), (-2.,.5,1.), (2.,-.5,1.), (-2.,-.5,1.)]


@pytest.fixture(scope='module')
def mesh_basis_arithmetic(engine, tmp_path_factory):
    from infernux.lib import _Infernux as native
    root = Path(__file__).parents[2] / 'python/infernux/resources/shaders/_templates'
    sources = {}
    for name in ('vertex_main', 'shadow_vertex_main'):
        template = (root / (name + '.glsl')).read_text(encoding='utf-8')
        start = template.rindex('mat3 modelMatrix =')
        end = template.index(';', template.index('worldTangent =', start)) + 1
        sources[name] = '''#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Values { vec4 values[]; };
struct VertexInput { vec3 normal; vec4 tangent; };
void main() {
    uint i = gl_GlobalInvocationID.x * 6;
    VertexInput v;
    v.normal = values[i].xyz;
    v.tangent = values[i+1];
    vec3 scale = values[i+2].xyz;
    mat4 instModel = mat4(scale.x,0,0,0, 0,scale.y,0,0, 0,0,scale.z,0, 0,0,0,1);
    vec3 worldNormal;
    vec4 worldTangent;
''' + template[start:end] + '''
    values[i+3] = vec4(worldNormal,0);
    values[i+4] = worldTangent;
    values[i+5] = vec4(cross(worldNormal,worldTangent.xyz)*worldTangent.w,0);
}
'''
    binaries = native._compile_compute_glsl_batch(sources, 'mesh-basis-arithmetic')
    inputs = np.zeros((len(ARITHMETIC_SCALES)*2,6,4), dtype=np.float32)
    for index, scale in enumerate(ARITHMETIC_SCALES):
        for mirror in (False,True):
            row = inputs[index*2+mirror]
            row[0,:3] = NORMAL
            row[1] = (*TANGENT, -1. if mirror else 1.)
            row[2,:3] = scale
    host = engine._acquire_compute_host()
    results = {}
    try:
        for name, spirv in binaries.items():
            kernel = host.create_kernel(spirv, 1, 0)
            buffer = host.create_buffer(inputs.size//4, 'float32', 4)
            try:
                buffer.set_bytes(inputs.tobytes())
                kernel.dispatch([buffer], b'', len(inputs))
                kernel.wait()
                output = np.frombuffer(buffer.get_bytes(inputs.nbytes), dtype=np.float32).reshape(inputs.shape)
                results[name] = output[:,3:].copy()
            finally:
                kernel.wait()
                del buffer, kernel
    finally:
        host._release_lease()
    (tmp_path_factory.mktemp('mesh-basis-arithmetic') / 'results.json').write_text(
        json.dumps({name:value.tolist() for name,value in results.items()}), encoding='utf-8')
    return results


@pytest.mark.parametrize('template', ['vertex_main','shadow_vertex_main'])
@pytest.mark.parametrize('mirror_uv', [False,True])
@pytest.mark.parametrize('index', range(len(ARITHMETIC_SCALES)))
def test_mesh_basis_matches_geometry_derivatives(mesh_basis_arithmetic, template, mirror_uv, index):
    normal, tangent = reference_basis(ARITHMETIC_SCALES[index], mirror_uv)
    bitangent = np.cross(normal,tangent[:3])*tangent[3]
    expected = np.array([(*normal,0), tangent, (*bitangent,0)])
    actual = mesh_basis_arithmetic[template][index*2+mirror_uv]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0.)
    np.testing.assert_allclose(np.linalg.norm(actual[:,:3],axis=1), 1., atol=2e-5, rtol=0.)
    assert abs(np.dot(actual[0,:3],actual[1,:3])) < 2e-5


@pytest.mark.parametrize('pipeline,samples', [('forward',1),('forward',4),('forward_plus',1),
                                            ('forward_plus',4),('deferred',1)])
@pytest.mark.parametrize('mirror_uv', [False,True])
def test_uv0_normal_mapping_matches_cpu_baked_geometry(engine, scene, tmp_path, monkeypatch, pipeline, samples, mirror_uv):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = None
    manager, console = SceneManager.instance(), ConsolePanel()
    previous_rendering = engine.is_play_mode_rendering()
    try:
        case = MeshNormalMapCase(engine, scene, tmp_path, pipeline, samples, mirror_uv)
        scene.main_camera = case.camera
        engine.resize_game_render_target(320,240)
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
                if ticket is None and frame >= changed + 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    case.observe(ticket.result_numpy())
                    (tmp_path / 'normal-map-results.json').write_text(json.dumps(case.results), encoding='utf-8')
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
