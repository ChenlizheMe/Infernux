"""Change shader domains through reimport while retaining GUIDs and shader IDs."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.debug import DebugConsole
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
        mesh_vertex = """import infernux.shader as shader
class TransitionVertex(shader.vert):
    shader_id = "test/domain/vertex"
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex
"""
        full_vertex = mesh_vertex
        full_fragment = """import infernux.shader as shader
class TransitionFragment(shader.fullscreen):
    shader_id = "test/domain/fragment"
    hidden = True
    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        return shader.vec4(0.0, 0.0, 1.0, 1.0)
"""
        mesh_fragment = """import infernux.shader as shader
class TransitionFragment(shader.surface):
    shader_id = "test/domain/fragment"
    shading_model = shader.ref("infernux/shading/unlit")
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(1.0, 0.1, 0.05))
"""
        paths = (project / 'Assets/transition.vert.py', project / 'Assets/transition.frag.py')
        initial = (mesh_vertex, mesh_fragment) if args.start_mesh else (full_vertex, full_fragment)
        for path, text in zip(paths, initial):
            path.write_text(text, encoding='utf-8')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        try:
            frontend.init_renderer(160, 120, str(project))
            DebugConsole.instance().set_native_console(console)
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
                'vertex': {'guid': guids[0], 'shader_id': 'test/domain/vertex'},
                'fragment': {'guid': guids[1], 'shader_id': 'test/domain/fragment'}}})
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
            deadline = time.monotonic() + 30

            def diagnostics():
                return [entry for entry in console._get_visible_log_snapshot(2000)
                        if entry['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]

            def complete_phase(pixels):
                nonlocal phase, changed
                assert material.guid == proof['material_guid']
                assert material.vert_shader_reference['guid'] == guids[0]
                assert material.frag_shader_reference['guid'] == guids[1]
                assert material.vert_shader_name == 'test/domain/vertex'
                assert material.frag_shader_name == 'test/domain/fragment'
                red = None if pixels is None else ((pixels[..., 0] > .8) & (pixels[..., 1] < .4) & (pixels[..., 2] < .4))
                entries = diagnostics()
                rejected = phase == 0 and not args.start_mesh or phase == 1 and args.start_mesh
                owned = [entry for entry in entries if entry.get('diagnostic_owner')]
                if (rejected and not owned) or (not rejected and (red.sum() <= 100 or entries)):
                    return False
                proof['phases'].append({'phase': phase, 'red_pixels': None if red is None else int(red.sum()),
                                        'diagnostics': entries[:8]})
                assert not [entry for entry in entries if 'Vulkan Validation Error:' in entry['message']]
                if phase == 0 or (args.start_mesh and phase == 1):
                    if args.start_mesh and phase == 0:
                        assert red.sum() > 100 and not entries, proof['phases'][-1]
                    else:
                        # A rejected first publication has no image. An already
                        # completed World retains its actual prior red image.
                        assert (red.sum() > 100 if args.start_mesh else red is None) and len(owned) == 1
                        if not args.start_mesh:
                            assert owned[0]['diagnostic_owner'] == 'material.shader-domain', proof['phases'][-1]
                        assert 'mesh' in owned[0]['message'].lower()
                        assert Path(owned[0]['source_file']).name in {'Transition.mat', 'transition.frag.py'}, proof['phases'][-1]
                    # Keep both source GUIDs, shader IDs and the material selection.
                    next_sources = (full_vertex, full_fragment) if args.start_mesh and phase == 0 else (mesh_vertex, mesh_fragment)
                    paths[1].write_text(next_sources[1], encoding='utf-8')
                    imported = AssetManager.reimport_asset(str(paths[1]), database=database)
                    assert imported.guid == guids[1] and imported.database_committed, imported.error
                    if args.start_mesh and phase == 0:
                        assert not imported and 'mesh shader' in imported.error.lower(), imported.error
                    else:
                        assert imported, imported.error
                    phase += 1
                    changed = frames
                else:
                    assert red.sum() > 100 and not entries, proof['phases'][-1]
                    proof['passed'] = True
                    print('PASS rejected shader source becomes Mesh without changing its GUID or shader ID', flush=True)
                    native.exit()
                return True

            def after_draw():
                nonlocal phase, frames, changed, ticket
                try:
                    frames += 1
                    assert time.monotonic() < deadline, proof['phases']
                    if AssetManager.refresh_pending():
                        return
                    if ticket is None and frames >= changed + 10:
                        world_id = SceneManager.instance()._current_runtime_world_id
                        if phase == 0 and not args.start_mesh:
                            if not diagnostics():
                                return
                            assert frontend._acquire_game_screen(world_id) is None
                            complete_phase(None)
                            return
                        if frontend._acquire_game_screen(world_id) is None:
                            return
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().astype(np.float32)
                        complete_phase(pixels)
                        ticket = None
                except BaseException as error:
                    failures.append(error)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert proof['passed']
        finally:
            DebugConsole.instance().set_native_console(None)
            native.cleanup()
            proof['failures'] = [str(error)[:3000] for error in failures]
            if args.proof:
                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
