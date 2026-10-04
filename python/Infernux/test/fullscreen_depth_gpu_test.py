"""Real tutorial depth copy, full-screen blending, and subsequent depth tests."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from Infernux import Engine
from Infernux.core.assets import AssetManager
from Infernux.lib import ConsolePanel, SceneManager, Vector3
from Infernux.renderstack import RenderPipeline
from Infernux.rendergraph import DepthCompare, Format

def shader(name, body, resources=''):
    return f'''#version 450
ShaderInfo {{
    Name "{name}"
    Hidden On
    Capabilities [Fullscreen]
    {resources}
    Inputs {{ Float2 inUV }}
    Outputs {{ Float4 outColor }}
}}
void main() {{ {body} }}
'''

class DepthAudit(RenderPipeline):
    name = 'Tutorial Fullscreen Depth Audit'
    def define_topology(self,graph):
        samples = graph.set_msaa_samples(self.samples)
        color = graph.create_texture('color',camera_target=True)
        scene_color = graph.create_texture('scene_color',format=Format.RGBA16_SFLOAT,samples=samples)
        scene_depth = graph.create_texture('depth',format=Format.D32_SFLOAT,samples=samples)
        previous_depth = graph.create_texture('previous_depth',format=Format.D32_SFLOAT,samples=samples)
        resolved = graph.create_texture('resolved',format=Format.RGBA16_SFLOAT,samples=1) if samples>1 else scene_color
        with graph.add_pass('Initialize') as p:
            p.write_color(scene_color).write_depth(scene_depth)
            p.set_clear(color=(0,0,0,0),depth=1.0)
            p.fullscreen_quad('Tutorial Initialize',depth_test=DepthCompare.ALWAYS,depth_write=True)
        graph.add_copy_pass('PreserveSceneDepth').copy_texture(scene_depth,previous_depth)
        with graph.add_pass('DepthAwareOverlay') as p:
            p.write_color(scene_color).write_depth(scene_depth)
            p.set_texture('sceneDepthTex',previous_depth)
            if self.clear:
                p.set_clear(color=(1,0,0,0))
            p.fullscreen_quad('Tutorial Overlay',depth_test=DepthCompare.ALWAYS,
                              depth_write=self.write,alpha_blend=self.blend)
        with graph.add_pass('DepthProbe') as p:
            p.write_color(scene_color).write_depth(scene_depth)
            if samples>1:
                p.write_resolve(resolved)
            p.fullscreen_quad('Tutorial Probe',depth_test=DepthCompare.LESS_EQUAL)
        with graph.add_pass('Commit') as p:
            p.write_color(color)
            p.set_texture('_SourceTex',resolved)
            p.fullscreen_quad('Fullscreen Blit')
        graph.screen_ui_section(resources={'color'})
        graph.set_output(color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,choices=(1,2,4,8),default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-depth-audit-') as root:
        project = Path(root)
        for directory in ('Assets','Packages','ProjectSettings'):
            (project/directory).mkdir()
        sources = {
            'Initialize':shader('Tutorial Initialize','outColor=vec4(0,0,1,1); gl_FragDepth=0.8;'),
            'Overlay':shader('Tutorial Overlay','''float d = texture(sceneDepthTex,inUV).r;
                outColor = abs(d-0.8)<0.001 ? vec4(0,1,0,0.5) : vec4(1,0,1,1);
                gl_FragDepth=0.4;''','Resources { Texture2D sceneDepthTex }'),
            'Probe':shader('Tutorial Probe','outColor=vec4(1,0,0,1); gl_FragDepth=0.6;'),
        }
        for name,content in sources.items():
            (project/f'Assets/{name}.frag').write_text(content,encoding='ascii')
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = DepthAudit()
        pipeline.samples=args.samples
        pipeline.write,pipeline.blend,pipeline.clear = True,True,False
        completed,failures = False,[]
        try:
            frontend.init_renderer(64,48,str(project))
            frontend.resize_game_render_target(64,48)
            engine.set_scene_view_visible(False)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Depth Camera')
            camera.transform.position = Vector3(0,0,-5)
            camera.add_component('Camera')
            for name in sources:
                imported = AssetManager.import_asset(str(project/f'Assets/{name}.frag'),database=frontend.get_asset_database())
                assert imported,imported.error
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)
            phases=('depth_written','depth_not_written','blend_off','explicit_clear','restored')
            expectations=((0,.5,.5,1),(1,0,0,1),(0,1,0,.5),(.5,.5,0,.5),(0,.5,.5,1))
            phase,frame,changed,ticket=0,0,0,None
            images={}
            def after_draw():
                nonlocal phase,frame,changed,ticket,completed
                try:
                    frame+=1
                    if ticket is None and frame>=changed+8:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        images[phases[phase]]=pixels
                        print('Fullscreen',args.samples,phases[phase],pixels[24,32].tolist(),flush=True)
                        errors=[e for e in console._get_visible_log_snapshot(2000)
                                if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors,errors
                        expected=np.array(expectations[phase],dtype=np.float32)
                        expected[:3]=np.where(expected[:3]<=.0031308,expected[:3]*12.92,
                                               1.055*expected[:3]**(1/2.4)-.055)
                        np.testing.assert_allclose(pixels,np.broadcast_to(expected,pixels.shape),atol=.005,
                                                   err_msg=f'{args.samples}: {phases[phase]}')
                        if phase==len(phases)-1:
                            np.testing.assert_array_equal(images['restored'],images['depth_written'])
                            print('PASS depth copy/blend/load/gl_FragDepth/subsequent depth test',flush=True)
                            completed=True
                            engine.exit()
                            return
                        phase+=1
                        pipeline.write=phase!=1
                        pipeline.blend=phase!=2
                        pipeline.clear=phase==3
                        pipeline.dispose()
                        ticket,changed=None,frame
                    if frame>130:
                        raise AssertionError('Fullscreen depth audit timeout')
                except BaseException as exc:
                    failures.append(exc)
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

if __name__=='__main__':
    main()
