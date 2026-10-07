"""Pixel proof that bone-only movement invalidates cached camera visibility."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3
from skinned_mesh_gpu_fixture import write_skinned_quad


SURFACE = '''#version 450
ShaderInfo { Name "Skin Culling Probe" ShadingModel Unlit Cull Off }
void surface(out SurfaceData s) {
    s=InitSurfaceData();
    s.albedo=vec3(1,0,0);
}
'''


class CullingPipeline(inx.renderstack.RenderPipeline):
    name = 'Bone-only Culling'

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color = graph.create_texture('color', camera_target=True)
        depth = graph.create_texture('depth', format=inx.rendergraph.Format.D32_SFLOAT, samples=1)
        graph.add_pass('Surface').write_color(color).write_depth(depth).set_clear(
            color=(0,0,0,0), depth=1.,
        ).draw_renderers(queue_range=(0,2500))
        graph.screen_ui_overlay_section(resources={'color'})
        graph.set_output(color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    proof = {'phases': [], 'scope': 'Actual Vulkan color readback; fixed camera and object transforms, bone translation only.'}
    with tempfile.TemporaryDirectory(prefix='infernux-skinned-culling-') as folder:
        project = Path(folder)
        for directory in ('Assets', 'Packages', 'ProjectSettings'):
            (project / directory).mkdir()
        model = project / 'Assets/Translated.gltf'
        write_skinned_quad(model, translations=[[100,0,0],[0,0,0],[100,0,0],[0,0,0]])
        shader = project / 'Assets/Culling.frag'
        shader.write_text(SURFACE, encoding='utf-8')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        pipeline = CullingPipeline()
        pipeline.builds = 0
        errors = []
        completed = False
        console = ConsolePanel()
        try:
            frontend.init_renderer(320,240,str(project))
            frontend.resize_game_render_target(320,240)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(model), database=database)
            assert imported, imported.error
            shader_import = AssetManager.import_asset(str(shader), database=database)
            assert shader_import, shader_import.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Fixed Camera')
            camera.transform.position = Vector3(0,0,-5)
            camera.add_component('Camera')
            obj = scene.create_game_object('Bone Moved Quad')
            renderer = obj.add_component('SkinnedMeshRenderer')
            renderer.set_source_model_guid(imported.guid)
            renderer.submit_animation_pose('Translate',0.,0.,loop=False)
            material = InxMaterial.create_default_unlit()
            material.frag_shader_name = 'Skin Culling Probe'
            renderer.set_material(0, material)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            phases = [('outside',0.,False), ('entered',1.,True), ('held_inside',None,True),
                      ('left',2.,False), ('held_outside',None,False), ('reentered',3.,True)]
            phase = frames = changed = 0
            ticket = None

            def after_draw():
                nonlocal phase, frames, changed, ticket, completed
                try:
                    frames += 1
                    if ticket is None and frames >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        count = int(((pixels[...,0] > .5) & (pixels[...,1] < .1)).sum())
                        name, _, visible = phases[phase]
                        record = dict(phase=name, pixels=count, bounds=list(renderer.get_world_bounds()), builds=pipeline.builds)
                        assert count > 500 if visible else count == 0, record
                        assert pipeline.builds == 1, record
                        assert not [entry for entry in console._get_visible_log_snapshot(2000)
                                    if entry['level'] in ('ERROR','FATAL')]
                        proof['phases'].append(record)
                        print('PASS ' + json.dumps(record), flush=True)
                        if phase == len(phases)-1:
                            proof['passed'] = True
                            if args.proof:
                                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')
                            completed = True
                            native.exit()
                            return
                        phase += 1
                        seconds = phases[phase][1]
                        if seconds is not None:
                            renderer.submit_animation_pose('Translate',seconds,seconds/3.,loop=False)
                        changed = frames
                        ticket = None
                    if frames > 220:
                        raise AssertionError('Skinned visibility readback timed out')
                except BaseException as error:
                    errors.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if errors:
                raise errors[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__ == '__main__':
    main()
