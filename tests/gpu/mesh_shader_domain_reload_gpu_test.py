"""Change shader domains through reimport while retaining GUIDs and Names."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry, ConsolePanel, InxMaterial, LightShadows, SceneManager, Vector3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--start-mesh', action='store_true')
    args = parser.parse_args()
    proof = {'package': inx.__file__, 'phases': [], 'passed': False}
    failures = []
    with tempfile.TemporaryDirectory(prefix='infernux-domain-source-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        resources = Path(inx.__file__).parent / 'resources/shaders'
        full_vertex = (resources / 'fullscreen_triangle.vert').read_text(encoding='utf-8').replace('Fullscreen Triangle', 'Domain Transition Vertex')
        full_fragment = (resources / 'fullscreen_blit.frag').read_text(encoding='utf-8').replace('Fullscreen Blit', 'Domain Transition Fragment')
        mesh_vertex = '#version 450\nShaderInfo { Name "Domain Transition Vertex" }\nvoid vertex(inout VertexInput v) {}\n'
        mesh_fragment = '#version 450\nShaderInfo { Name "Domain Transition Fragment" ShadingModel Unlit }\nvoid surface(out SurfaceData s) { s=InitSurfaceData(); s.albedo=vec3(1.0,0.1,0.05); }\n'
        paths = (project / 'Assets/Transition.vert', project / 'Assets/Transition.frag')
        initial = (mesh_vertex, mesh_fragment) if args.start_mesh else (full_vertex, full_fragment)
        for path, text in zip(paths, initial):
            path.write_text(text, encoding='utf-8')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        try:
            frontend.init_renderer(160, 120, str(project))
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database = frontend.get_asset_database()
            guids = []
            for path in paths:
                result = AssetManager.import_asset(str(path), database=database)
                assert result, result.error
                guids.append(result.guid)
            document = InxMaterial.create_default_unlit().serialize_document()
            document.update({'name': 'Domain Transition', 'shaders': {
                'vertex': {'guid': guids[0], 'shader_id': 'Domain Transition Vertex'},
                'fragment': {'guid': guids[1], 'shader_id': 'Domain Transition Fragment'}}})
            material_path = project / 'Assets/Transition.mat'
            material_path.write_text(json.dumps(document), encoding='utf-8')
            result = AssetManager.import_asset(str(material_path), database=database)
            assert result, result.error
            material = AssetRegistry.instance().load_material_by_guid(result.guid)
            assert material
            proof['material_guid'] = result.guid
            proof['shader_guids'] = guids
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Camera')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component('Camera')
            sun = scene.create_game_object('Domain Sun')
            sun.transform.euler_angles = Vector3(35, 25, 0)
            sun.add_component('Light').shadows = LightShadows.Hard
            obj = scene.create_game_object('Quad')
            renderer = obj.add_component('MeshRenderer')
            renderer.set_inline_mesh_data(
                np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]],dtype=np.float32),
                np.array([[0,0,-1]]*4,dtype=np.float32),
                np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32),
                np.array([0,2,1,0,3,2],dtype=np.uint32),'Quad')
            renderer.set_material(0, material)
            native.set_game_camera_enabled(True)
            phase = frames = changed = 0
            ticket = None
            def after_draw():
                nonlocal phase, frames, changed, ticket
                try:
                    frames += 1
                    if ticket is None and frames >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        assert material.guid == proof['material_guid']
                        assert material.vert_shader_reference['guid'] == guids[0]
                        assert material.frag_shader_reference['guid'] == guids[1]
                        assert material.vert_shader_name == 'Domain Transition Vertex'
                        assert material.frag_shader_name == 'Domain Transition Fragment'
                        pixels = ticket.result_numpy().astype(np.float32)
                        red = (pixels[...,0]>.8) & (pixels[...,1]<.4) & (pixels[...,2]<.4)
                        diagnostics = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        proof['phases'].append({'phase': phase, 'red_pixels': int(red.sum()), 'diagnostics': diagnostics[:8]})
                        assert not [e for e in diagnostics if 'Vulkan Validation Error:' in e['message']]
                        if phase == 0 or (args.start_mesh and phase == 1):
                            if args.start_mesh and phase == 0:
                                assert red.sum() > 100 and not diagnostics, proof['phases'][-1]
                            else:
                                assert not red.any() and any("uses domain 'Fullscreen'" in e['message'] for e in diagnostics), proof['phases'][-1]
                            # Publish both source files before reimporting either;
                            # keep GUIDs, ShaderInfo Names and material selection fixed.
                            next_sources = (full_vertex, full_fragment) if args.start_mesh and phase == 0 else (mesh_vertex, mesh_fragment)
                            for path, text in zip(paths, next_sources):
                                path.write_text(text, encoding='utf-8')
                            for path, guid in zip(paths, guids):
                                imported = database.reimport_asset(str(path))
                                assert imported and imported.guid == guid, imported.error
                                error = native.reload_shader_runtime(str(path), 'Domain Transition Vertex' if path.suffix=='.vert' else 'Domain Transition Fragment')
                                assert not error, error
                            phase += 1
                            changed = frames
                            ticket = None
                        else:
                            assert red.sum()>100, proof['phases'][-1]
                            proof['passed'] = True
                            print('PASS rejected shader source becomes Mesh without changing its GUID or Name', flush=True)
                            native.exit()
                    if frames>200:
                        raise AssertionError('Domain source transition timed out')
                except BaseException as error:
                    failures.append(error)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert proof['passed']
        finally:
            native.cleanup()
            proof['failures'] = [str(error)[:3000] for error in failures]
            if args.proof:
                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
