"""Actual engine UI draws consume immutable dependency reload candidates."""
import json

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.debug import DebugConsole, LogType
from infernux.lib import InxMaterial, ScreenUIList
from infernux.engine.runtime_screen_ui_pipeline import RuntimeScreenUIRenderPipeline
from infernux.renderstack import RenderStackPipeline
from infernux.ui.ui_render_dispatch import _bind_runtime_material
from test_shader_dependency_publication import dependency_sources


VERTEX = '''#version 450
layout(location=0) in vec2 aPosition;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aColor;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location=0) out vec4 outColor;
layout(location=1) out vec2 outUV;
void main() {
    gl_Position = vec4(aPosition * pc.scale + pc.translate, 0.0, 1.0);
    outColor = aColor;
    outUV = aUV;
}
'''

FRAGMENT = '''#version 450
layout(location=0) in vec4 inColor;
layout(location=1) in vec2 inUV;
layout(set=0,binding=0) uniform sampler2D uiTexture;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location=0) out vec4 outColor;
void main() { outColor = dependencyColor() * texture(uiTexture, inUV) * inColor * pc.materialColor; }
'''


@pytest.mark.parametrize('draw_list', (ScreenUIList.Overlay, ScreenUIList.Camera), ids=('overlay', 'camera'))
@pytest.mark.parametrize('material_properties', (False, True), ids=('plain', 'material-buffer'))
def test_actual_ui_draw_consumes_prepared_dependency_and_rejects_invalid_saves(
        engine, scene, dependency_sources, capfd, draw_list, material_properties):
    prefix, create, _ = dependency_sources
    library_name = prefix + ' Library'
    red = (f'ShaderInfo {{ Name "{library_name}" }}\n'
           'vec4 dependencyColor(){[[flatten]] if(true) return vec4(1,0,0,1); return vec4(0);}\n')
    green = red.replace('vec4(1,0,0,1)', 'vec4(0,1,0,1)')
    invalid = red.replace('return vec4(1,0,0,1)', 'return missingUiDependencyFunction()')
    library, _ = create('Library', '.glsl', red)
    vertex_name, fragment_name = prefix + ' Vertex', prefix + ' Fragment'
    vertex, vertex_guid = create('Vertex', '.vert',
        f'#version 450\nShaderInfo {{ Name "{vertex_name}" Capabilities [ScreenUI] }}\n' + VERTEX.removeprefix('#version 450\n'))
    properties = ' Properties { Color baseColor = [1,1,1,1] }' if material_properties else ''
    fragment_body = FRAGMENT.removeprefix('#version 450\n')
    if material_properties:
        fragment_body = fragment_body.replace('pc.materialColor;', 'pc.materialColor * material.baseColor;')
    fragment, fragment_guid = create('Fragment', '.frag',
        '#version 450\n#extension GL_EXT_control_flow_attributes : require\n'
        f'ShaderInfo {{ Name "{fragment_name}" Capabilities [ScreenUI] Imports ["{library_name}"]{properties} }}\n' + fragment_body)
    roots = {vertex: vertex.read_bytes(), fragment: fragment.read_bytes()}
    document = InxMaterial.create_default_lit().serialize_document()
    document['builtin'] = False
    document['name'] = prefix + ' UI material'
    document['shaders'] = {
        'vertex': {'guid': vertex_guid, 'shader_id': vertex_name},
        'fragment': {'guid': fragment_guid, 'shader_id': fragment_name},
    }
    _, material_guid = create('Material', '.mat', json.dumps(document))
    material = AssetManager.load(material_guid, Material)
    assert engine.refresh_material_pipeline(material._native)
    assert not engine.is_shader_loaded(vertex_name, 'vertex'), 'Pre-draw refresh published an ownerless UI artifact'
    scene.create_game_object('UI dependency camera').add_component('Camera')
    # Target creation may replace the native UI backend. Borrow the renderer
    # only after the final target signature is established.
    engine.resize_game_render_target(64, 64)
    renderer = engine.get_screen_ui_renderer()
    state = {'frames': 0, 'phase': 0, 'ticket': None, 'submissions': 0}
    samples, errors = [], []

    class Submission:
        def submit(self):
            nonlocal renderer
            renderer = engine.get_screen_ui_renderer()
            state['submissions'] += 1
            renderer.begin_frame(64, 64)
            renderer.begin_command_packet()
            try:
                _bind_runtime_material(renderer, draw_list, {'_native': material._native, 'color': (1, 1, 1, 1)})
                renderer.add_filled_rect(draw_list, 8, 8, 56, 56)
                packet = renderer.end_command_packet()
            except BaseException:
                renderer.abort_command_packet()
                raise
            renderer.append_command_packets([packet])

    def after_draw():
        nonlocal renderer
        try:
            state['frames'] += 1
            assert state['frames'] <= 100, 'The same submitted readback ticket never completed'
            observation = {
                'frames': state['frames'], 'phase': state['phase'], 'submissions': state['submissions'],
                'commands': renderer.has_commands(draw_list), 'enabled': renderer.is_enabled(),
                'shader_loaded': engine.is_shader_loaded(vertex_name, 'vertex'),
                'pending_textures': engine.pending_imgui_texture_upload_count,
                'submitted_textures': engine.submitted_imgui_texture_upload_count,
                'completed_textures': engine.completed_imgui_texture_upload_count,
                'msaa_samples': engine.get_msaa_samples(),
                'native_frame': {key: value for key, value in engine.renderer_frame_snapshot.items()
                                 if key in ('frame', 'game_camera_enabled', 'game_camera_available',
                                            'game_target_ready', 'scene_view_visible', 'game_draw_call_count',
                                            'game_render_graph_name', 'game_render_graph_pass_names')}}
            assert renderer.last_submitted_draw_count(draw_list) >= 1, json.dumps(observation)
            assert engine.is_shader_loaded(vertex_name, 'vertex')
            assert all(path.read_bytes() == contents for path, contents in roots.items())
            if state['ticket'] is None:
                state['ticket'] = engine.request_render_target_readback(True)
                return
            ticket = state['ticket']
            if not ticket.done:
                return
            assert not ticket.error, ticket.error
            pixels = ticket.result_numpy()
            assert pixels.shape == (64, 64, 4) and str(pixels.dtype) == 'float16'
            sample = pixels[32, 32].tolist()
            expected = (0, 1, 1, 0)[state['phase']]
            assert sample[expected] >= .95 and sample[1-expected] <= .02 and sample[2] <= .02, sample
            samples.append(sample)
            state['ticket'] = None
            if state['phase'] == 0:
                library.write_text(green, encoding='utf-8')
                prepared = AssetManager.reimport_asset(str(library))
                assert prepared, prepared.error
                # This later unreported write must not mutate the accepted,
                # prepared artifact. No watcher is running in this fixture.
                # The next real draw must use green without compiling again.
                library.write_text(invalid, encoding='utf-8')
            elif state['phase'] == 1:
                before = engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']
                rejected = AssetManager.reimport_asset(str(library))
                assert not rejected and 'missingUiDependencyFunction' in rejected.error
                assert engine.gpu_residency_snapshot['shader_hot_reload_retirement_count'] == before
                own = [entry for entry in DebugConsole.instance().get_entries()
                       if entry.log_type == LogType.ERROR and entry.source_file == str(library)]
                assert len(own) == 1
            elif state['phase'] == 2:
                library.write_text(red, encoding='utf-8')
                restored = AssetManager.reimport_asset(str(library))
                assert restored, restored.error
                assert not [entry for entry in DebugConsole.instance().get_entries()
                            if entry.log_type == LogType.ERROR and entry.source_file == str(library)]
            else:
                engine.exit()
            state['phase'] += 1
        except BaseException as error:
            errors.append(error)
            engine.exit()

    try:
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(RuntimeScreenUIRenderPipeline(Submission(), RenderStackPipeline()))
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert state['phase'] == 4
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        renderer.begin_frame(64, 64)
        library.write_text(red, encoding='utf-8')
    captured = capfd.readouterr()
    assert not any(token in captured.out + captured.err for token in (
        'VUID-', 'SYNC-HAZARD', 'Validation Error', 'Screen UI packet outlived',
        'UI material shader publication failed',
        'Validation layers requested but not available',
    )), captured.out + captured.err
    print(json.dumps({'draw_list': str(draw_list), 'material_properties': material_properties, 'samples': samples,
                      'frames': state['frames'],
                      'prepared_immutable': True, 'rejected_candidate_keeps_valid_program': True}))
