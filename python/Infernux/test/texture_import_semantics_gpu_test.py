"""Tutorial swatches: live sRGB/Data/Normal Map import and alpha preservation."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from Infernux import Engine
from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import TextureImportSettings, TextureType, TextureCompression, read_texture_import_settings
from Infernux.lib import AssetRegistry, ConsolePanel, SceneManager, Vector3
from Infernux.renderstack import RenderPipeline
from Infernux.rendergraph import Format

SHADER = '''#version 450
ShaderInfo {
    Name "Tutorial Texture Swatch"
    Hidden On
    Capabilities [Fullscreen]
    Resources { Texture2D _SourceTex }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() { outColor = texture(_SourceTex, inUV); }
'''

class SwatchPipeline(RenderPipeline):
    name = "Tutorial Texture Swatch"

    def define_topology(self, graph):
        self.builds += 1
        sampled = graph.import_texture("swatch", self.texture)
        color = graph.create_texture("color", camera_target=True)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT)
        with graph.add_pass("EmptyDepth") as render_pass:
            render_pass.write_depth(depth)
            render_pass.set_clear(depth=1.0)
            render_pass.draw_renderers(material_pass="depth")
        with graph.add_pass("SampleSwatch") as render_pass:
            render_pass.set_texture("_SourceTex", sampled)
            render_pass.write_color(color)
            render_pass.fullscreen_quad("Tutorial Texture Swatch")
        graph.screen_ui_section(resources={"color"})
        graph.set_output(color)

def encode(linear):
    return np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)

def main():
    with tempfile.TemporaryDirectory(prefix='infernux-texture-swatches-') as root:
        project = Path(root)
        for directory in ('Assets', 'Packages', 'ProjectSettings'):
            (project / directory).mkdir()
        shader = project / 'Assets/Swatch.frag'
        shader.write_text(SHADER, encoding='ascii')
        texture_path = project / 'Assets/Gray.png'
        Image.new('RGBA', (8,8), (128,128,128,64)).save(texture_path)
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = SwatchPipeline()
        failures, completed = [], False
        try:
            frontend.init_renderer(64,48,str(project))
            frontend.resize_game_render_target(64,48)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            engine.set_scene_view_visible(False)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Swatch Camera')
            camera.transform.position = Vector3(0,0,-5)
            camera.add_component('Camera')
            engine.set_game_camera_enabled(True)
            database = frontend.get_asset_database()
            assert AssetManager.import_asset(str(shader), database=database)
            imported = AssetManager.import_asset(str(texture_path), database=database)
            assert imported, imported.error
            settings = TextureImportSettings(generate_mipmaps=False, compression=TextureCompression.NONE)
            assert AssetManager.apply_import_settings('texture', str(texture_path), settings)
            pipeline.texture = AssetRegistry.instance().load_texture_by_guid(imported.guid)
            assert pipeline.texture is not None
            assert pipeline.texture.srgb
            pipeline.builds = 0
            frontend.set_render_pipeline(pipeline)
            phases = ('srgb', 'linear', 'data', 'normal')
            phase, frame, changed, ticket, builds = 0, 0, 0, None, None
            def after_draw():
                nonlocal phase, frame, changed, ticket, completed, builds
                try:
                    frame += 1
                    # This standalone loop replaces the Editor post-draw callback;
                    # retain its authoritative queued GPU publication service.
                    AssetManager.flush_pending_gpu_texture_reloads()
                    pipeline.texture = AssetRegistry.instance().load_texture_by_guid(imported.guid)
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        center = pixels[24,32]
                        print('Swatch',phases[phase],center.tolist(), 'builds',pipeline.builds,
                              'format',pipeline.texture.pixel_format,'sRGB',pipeline.texture.srgb,flush=True)
                        if builds is not None:
                            assert pipeline.builds == builds, 'Import settings rebuilt authored topology'
                        errors = [entry for entry in console._get_visible_log_snapshot(1000)
                                  if entry['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors, errors
                        if phase < 3:
                            expected = 128/255 if phase == 0 else float(encode(np.array(128/255)))
                            np.testing.assert_allclose(center[:3], expected, atol=.003)
                            np.testing.assert_allclose(center[3], 64/255, atol=.001)
                        else:
                            # The flat normal's RG remain linear and AUTO stores
                            # them in BC5; the shader helper reconstructs Z.
                            np.testing.assert_allclose(center[:2], encode(np.array([128/255]*2)), atol=.006)
                            assert not pipeline.texture.srgb
                            assert 'bc5' in pipeline.texture.pixel_format.lower(), pipeline.texture.pixel_format
                            completed = True
                            print('PASS live sRGB/Data/Normal Map swatches and alpha preservation', flush=True)
                            engine.exit()
                            return
                        if builds is None:
                            builds = pipeline.builds
                        assert pipeline.builds == builds, 'Import settings rebuilt authored topology'
                        phase += 1
                        if phase == 1:
                            settings.srgb = False
                        elif phase == 2:
                            settings.texture_type = TextureType.DATA
                            settings.compression = TextureCompression.AUTO
                            settings.srgb = True
                            settings._sync_derived_fields()
                        else:
                            Image.new('RGBA',(8,8),(128,128,255,255)).save(texture_path)
                            settings.texture_type = TextureType.NORMAL_MAP
                            settings.compression = TextureCompression.AUTO
                            settings.srgb = True
                            settings._sync_derived_fields()
                        assert AssetManager.apply_import_settings('texture', str(texture_path), settings)
                        stored = read_texture_import_settings(str(texture_path))
                        print('Applied', phases[phase], 'stored', stored.to_dict(), flush=True)
                        assert stored.srgb == settings.srgb
                        assert database.get_guid_from_path(str(texture_path)) == imported.guid
                        pipeline.texture = AssetRegistry.instance().load_texture_by_guid(imported.guid)
                        ticket, changed = None, frame
                    if frame > 120:
                        raise AssertionError('Texture swatch readback timed out')
                except BaseException as error:
                    failures.append(error)
                    engine.exit()
            engine.set_post_draw_callback(after_draw)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            engine.cleanup()

if __name__ == '__main__':
    main()
