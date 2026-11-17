"""Actual engine UI draws consume immutable dependency reload candidates."""
import json

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.debug import DebugConsole, LogType
from infernux.lib import ConsolePanel, InxMaterial, ScreenUIList, Vector3
from infernux.engine.runtime_screen_ui_pipeline import RuntimeScreenUIRenderPipeline
from infernux.renderstack import RenderStackPipeline
from infernux.ui.ui_render_dispatch import _bind_runtime_material
from test_shader_dependency_publication import dependency_sources


_FRAME_COLORS = ((1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0))


def ui_shader_source(shader_id, domain, color, *, library=None, material_properties=False):
    context = 'WorldUIContext' if domain == 'world_ui' else 'UIContext'
    lines = ['import infernux.shader as shader']
    if library is not None:
        lines.append(f'from .{library.name.removesuffix(".shaderlib.py")} import dependency_color')
    lines.extend(('', f'class DependencyUI(shader.{domain}):', f'    shader_id = {shader_id!r}'))
    if material_properties:
        lines.append('    base_color = shader.material.color()')
    lines.append(f'    def main(self, ctx: shader.{context}) -> shader.vec4:')
    expression = color + ' * ctx.texture_color * ctx.color * ctx.material_color'
    if material_properties:
        expression += ' * self.base_color'
    lines.append(f'        return {expression}')
    return '\n'.join(lines) + '\n'


def test_missing_ui_shader_skips_draw_and_recovers_on_asset_publication(engine, scene, dependency_sources):
    prefix, create, _ = dependency_sources
    fragment_name = prefix.lower() + '/ui'
    fragment, fragment_guid = create('Fragment', '.frag.py', ui_shader_source(
        fragment_name, 'screen_ui', 'shader.vec4(0.0, 0.0, 1.0, 1.0)'))
    document = InxMaterial.create_default_lit().serialize_document()
    document.update(builtin=False, name=prefix + ' Missing UI material')
    document['shaders'] = {
        'vertex': {'guid': '', 'shader_id': ''},
        'fragment': {'guid': fragment_guid, 'shader_id': fragment_name},
    }
    _, material_guid = create('Material', '.mat', json.dumps(document))
    material = AssetManager.load(material_guid, Material)
    metadata = fragment.with_name(fragment.name + '.meta')
    source_bytes, metadata_bytes = fragment.read_bytes(), metadata.read_bytes()

    def remove_shader():
        fragment.unlink()
        removed = AssetManager.delete_asset(str(fragment))
        assert removed, removed.error
        assert not engine.get_asset_database().get_path_from_guid(fragment_guid)

    def restore_shader():
        metadata.write_bytes(metadata_bytes)
        fragment.write_bytes(source_bytes)
        imported = AssetManager.import_asset(str(fragment))
        assert imported and imported.guid == fragment_guid, imported.error

    console = ConsolePanel()
    remove_shader()
    camera = scene.create_game_object('Missing UI shader camera')
    camera.transform.position = Vector3(0, 0, -2)
    camera.add_component('Camera')
    previous_samples = engine.get_msaa_samples()
    engine.set_msaa_samples(4)
    engine.resize_game_render_target(64, 64)
    state = {'phase': 0, 'ticket': None, 'frames': 0, 'phase_execution': None}
    errors = []

    class Submission:
        def submit(self, **_capture):
            renderer = engine.get_screen_ui_renderer()
            renderer.begin_frame(64, 64)
            renderer.begin_command_packet()
            try:
                _bind_runtime_material(renderer, ScreenUIList.Overlay, {})
                renderer.add_filled_rect(ScreenUIList.Overlay, 4, 4, 16, 16, *_FRAME_COLORS[state['phase']])
                _bind_runtime_material(renderer, ScreenUIList.Overlay, {'_native': material._native})
                renderer.add_filled_rect(ScreenUIList.Overlay, 28, 28, 60, 60)
                packet = renderer.end_command_packet()
            except BaseException:
                renderer.abort_command_packet()
                raise
            renderer.append_command_packets([packet])

    def after_draw():
        try:
            state['frames'] += 1
            assert state['frames'] <= 100, f'UI shader recovery readback never completed: {state}'
            ticket = state['ticket']
            if ticket is not None and not ticket.done:
                return
            if ticket is None and not engine.game_presentation_ready:
                return
            frame = engine.renderer_frame_snapshot
            execution = (frame['game_render_view_id'], frame['game_render_graph_execution_count'])
            if ticket is None:
                if not frame['game_render_graph_current_executed'] or execution == state['phase_execution']:
                    return
                state['ticket'] = engine.request_render_target_readback(True)
                return
            assert not ticket.error, ticket.error
            pixels = ticket.result_numpy()
            state['ticket'] = None
            state['phase_execution'] = execution
            # The Game readback can complete on a frame submitted before the
            # asset edit. The default-material rectangle labels each phase
            # independently of the custom material being tested.
            if tuple(pixels[8, 8, :3]) != _FRAME_COLORS[state['phase']]:
                return
            sample = pixels[42, 42, :3].tolist()
            draws = engine.get_screen_ui_renderer().last_submitted_draw_count(ScreenUIList.Overlay)
            expect_present = state['phase'] % 2 == 1
            blue = sample[2] >= .95 and max(sample[:2]) <= .02
            expected_draws = 2 if expect_present else 1
            assert draws == expected_draws
            assert blue == expect_present, sample
            if not expect_present:
                restore_shader()
            else:
                if state['phase'] == 1:
                    remove_shader()
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
        engine.get_screen_ui_renderer().begin_frame(64, 64)
        engine.set_msaa_samples(previous_samples)
        if not fragment.exists():
            restore_shader()
    rejected = [entry for entry in console._get_visible_log_snapshot(1000)
                if fragment_guid in entry['message']
                and 'This UI draw is disabled' in entry['message']]
    assert len(rejected) == 2, rejected


@pytest.mark.parametrize('draw_list', (ScreenUIList.Overlay, ScreenUIList.Camera, ScreenUIList.World),
                         ids=('overlay', 'camera', 'world'))
@pytest.mark.parametrize('material_properties', (False, True), ids=('plain', 'material-buffer'))
@pytest.mark.parametrize('initial_samples', (1, 4), ids=('initial-1x', 'initial-4x'))
def test_actual_ui_draw_consumes_prepared_dependency_and_rejects_invalid_saves(
        engine, scene, dependency_sources, capfd, draw_list, material_properties, initial_samples):
    prefix, create, _ = dependency_sources
    red = ('import infernux.shader as shader\n'
           'def dependency_color() -> shader.vec4:\n'
           '    return shader.vec4(1.0, 0.0, 0.0, 1.0)\n')
    green = red.replace('1.0, 0.0, 0.0', '0.0, 1.0, 0.0')
    invalid = red.replace('shader.vec4(1.0, 0.0, 0.0, 1.0)', 'missing_ui_dependency_function()')
    library, _ = create('Library', '.shaderlib.py', red)
    fragment_name = prefix.lower() + '/ui'
    world = draw_list == ScreenUIList.World
    domain = 'world_ui' if world else 'screen_ui'
    fragment, fragment_guid = create('Fragment', '.frag.py', ui_shader_source(
        fragment_name, domain, 'dependency_color()', library=library, material_properties=material_properties))
    fragment_source = fragment.read_bytes()
    document = InxMaterial.create_default_lit().serialize_document()
    document['builtin'] = False
    document['name'] = prefix + ' UI material'
    document['shaders'] = {
        'vertex': {'guid': '', 'shader_id': ''},
        'fragment': {'guid': fragment_guid, 'shader_id': fragment_name},
    }
    _, material_guid = create('Material', '.mat', json.dumps(document))
    material = AssetManager.load(material_guid, Material)
    before_refresh = engine.renderer_frame_snapshot['graphics_pipeline_create_attempt_count']
    assert engine.refresh_material_pipeline(material._native)
    assert engine.renderer_frame_snapshot['graphics_pipeline_create_attempt_count'] == before_refresh
    camera = scene.create_game_object('UI dependency camera')
    camera.transform.position = Vector3(0, 0, -2)
    camera.add_component('Camera')
    world_object = scene.create_game_object('World UI dependency quad') if world else None
    # Target creation may replace the native UI backend. Borrow the renderer
    # only after the final target signature is established.
    previous_samples = engine.get_msaa_samples()
    engine.set_msaa_samples(initial_samples)
    assert engine.get_msaa_samples() == initial_samples
    engine.resize_game_render_target(64, 64)
    renderer = engine.get_screen_ui_renderer()
    renderer_samples = engine.get_msaa_samples()
    state = {'frames': 0, 'phase': 0, 'ticket': None, 'submissions': 0, 'msaa_rebuilds': 0, 'phase_execution': None}
    samples, errors = [], []

    class Submission:
        def submit(self, **_capture):
            nonlocal renderer, renderer_samples
            renderer = engine.get_screen_ui_renderer()
            renderer_samples = engine.get_msaa_samples()
            state['submissions'] += 1
            renderer.begin_frame(64, 64)
            renderer.begin_command_packet()
            try:
                _bind_runtime_material(renderer, ScreenUIList.Overlay, {})
                renderer.add_filled_rect(ScreenUIList.Overlay, 2, 2, 7, 7, *_FRAME_COLORS[state['phase']])
                if world:
                    renderer.begin_world_object(world_object, 32, 32)
                _bind_runtime_material(renderer, draw_list, {'_native': material._native, 'color': (1, 1, 1, 1)})
                renderer.add_filled_rect(draw_list, 8, 8, 56, 56)
                if world:
                    renderer.end_world_element()
                packet = renderer.end_command_packet()
            except BaseException:
                renderer.abort_command_packet()
                raise
            renderer.append_command_packets([packet])

    def after_draw():
        nonlocal renderer, renderer_samples
        try:
            state['frames'] += 1
            assert state['frames'] <= 100, 'The same submitted readback ticket never completed'
            current_renderer = engine.get_screen_ui_renderer()
            if current_renderer is not renderer:
                # MSAA replaces the backend and abandons this frame before GPU
                # recording. The callback still advances deferred owner work;
                # never inspect the retired borrow as an executed UI draw.
                assert state['phase'] == 0 and state['ticket'] is None
                assert engine.get_msaa_samples() != renderer_samples
                assert not engine.renderer_frame_snapshot['game_render_graph_pass_names']
                state['msaa_rebuilds'] += 1
                assert state['msaa_rebuilds'] <= 2
                renderer = current_renderer
                renderer_samples = engine.get_msaa_samples()
                return
            ticket = state['ticket']
            if ticket is None and not engine.game_presentation_ready:
                return
            frame = engine.renderer_frame_snapshot
            execution = (frame['game_render_view_id'], frame['game_render_graph_execution_count'])
            if ticket is None and (
                not frame['game_render_graph_current_executed'] or execution == state['phase_execution']
            ):
                return
            observation = {
                'frames': state['frames'], 'phase': state['phase'], 'submissions': state['submissions'],
                'commands': renderer.has_commands(draw_list), 'enabled': renderer.is_enabled(),
                'shader_loaded': engine.is_shader_loaded(fragment_guid, 'fragment'),
                'pending_textures': engine.pending_imgui_texture_upload_count,
                'submitted_textures': engine.submitted_imgui_texture_upload_count,
                'completed_textures': engine.completed_imgui_texture_upload_count,
                'msaa_samples': engine.get_msaa_samples(),
                'native_frame': {key: value for key, value in frame.items()
                                 if key in ('frame', 'game_camera_enabled', 'game_camera_available',
                                            'game_target_ready', 'scene_view_visible', 'game_draw_call_count',
                                            'game_render_graph_name', 'game_render_graph_pass_names')}}
            assert renderer.last_submitted_draw_count(draw_list) >= 1, json.dumps(observation)
            assert observation['shader_loaded']
            assert fragment.read_bytes() == fragment_source
            if ticket is None:
                state['ticket'] = engine.request_render_target_readback(True)
                return
            if not ticket.done:
                return
            assert not ticket.error, ticket.error
            pixels = ticket.result_numpy()
            assert pixels.shape == (64, 64, 4) and str(pixels.dtype) == 'float16'
            state['ticket'] = None
            state['phase_execution'] = execution
            if tuple(pixels[4, 4, :3]) != _FRAME_COLORS[state['phase']]:
                return
            sample = pixels[32, 32].tolist()
            expected = (0, 1, 1, 0)[state['phase']]
            assert sample[expected] >= .95 and sample[1-expected] <= .02 and sample[2] <= .02, sample
            samples.append(sample)
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
                assert not rejected and 'author.syntax' in rejected.error, rejected.error
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
        if initial_samples == 1:
            assert state['msaa_rebuilds'] >= 1, 'The 1x-to-default-4x startup transition was not exercised'
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        engine.get_screen_ui_renderer().begin_frame(64, 64)
        engine.set_msaa_samples(previous_samples)
        library.write_text(red, encoding='utf-8')
    captured = capfd.readouterr()
    assert not any(token in captured.out + captured.err for token in (
        'VUID-', 'SYNC-HAZARD', 'Validation Error', 'Screen UI packet outlived',
        'UI material shader publication failed',
        'Validation layers requested but not available',
    )), captured.out + captured.err
    print(json.dumps({'draw_list': str(draw_list), 'material_properties': material_properties, 'samples': samples,
                      'frames': state['frames'],
                      'initial_samples': initial_samples, 'msaa_rebuilds': state['msaa_rebuilds'],
                      'prepared_immutable': True, 'rejected_candidate_keeps_valid_program': True}))
