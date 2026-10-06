"""Verify deformed tangent frames under nonuniform and mirrored model scales."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3

VERTEX = '''#version 450
ShaderInfo { Name "Deformed Tangent Probe" Properties { Float handedness=1.0 } }
void vertex(inout VertexInput v) {
    mat3 rotation = mat3(vec3(0.8, 0.48, 0.36), vec3(-0.6, 0.64, 0.48), vec3(0.0, -0.6, 0.8));
    v.position = rotation * v.position;
    v.normal = rotation * v.normal;
    v.tangent.xyz = rotation * v.tangent.xyz;
    v.tangent.w = material.handedness;
}
'''
FRAGMENT = '''#version 450
ShaderInfo {
    Name "Deformed Tangent Consumer" ShadingModel Unlit Cull Off
    Properties { Float scaleX=1.0 Float scaleY=1.0 Float scaleZ=1.0 Float handedness=1.0 }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    vec3 scale = vec3(material.scaleX, material.scaleY, material.scaleZ);
    vec3 N = normalize(vec3(0.0, 0.6, -0.8) / scale);
    vec3 T = normalize(vec3(0.8, 0.48, 0.36) * scale);
    float signW = material.handedness * sign(scale.x * scale.y * scale.z);
    vec3 B = cross(N, T) * signW;
    vec3 tangentNormal = normalize(vec3(0.35, -0.25, 0.9));
    vec3 expectedMapped = normalize(mat3(T, B, N) * tangentNormal);
    vec4 actualT = getWorldTangent();
    bool normalOK = length(getWorldNormal() - N) < 0.002;
    bool tangentOK = length(actualT.xyz - T) < 0.002 && abs(actualT.w - signW) < 0.002;
    bool basisOK = length(getWorldBitangent() - B) < 0.002
        && abs(dot(getWorldNormal(), actualT.xyz)) < 0.002;
    bool mappedOK = length(normalFromTangentSpace(tangentNormal, getWorldNormal(), actualT) - expectedMapped) < 0.002;
    s.albedo = normalOK && tangentOK && basisOK && mappedOK ? vec3(0,1,0) : vec3(1,0,1);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    plans = [(inx.renderstack.DefaultForwardPipeline, 1), (inx.renderstack.DefaultForwardPipeline, 4),
             (inx.renderstack.DefaultForwardPlusPipeline, 1), (inx.renderstack.DefaultForwardPlusPipeline, 4),
             (inx.renderstack.DefaultDeferredPipeline, 1)]
    phases = [('unit', (1., 1., 1.), 1.), ('nonuniform', (2., .5, 1.), 1.),
              ('mirror', (-2., .5, 1.), 1.), ('uv_mirror', (2., .5, 1.), -1.), ('restored', (1., 1., 1.), 1.)]
    proof = {'package': inx.__file__, 'scope': 'Public vertex hook rotates positions, normals and tangents together. Shader checks analytical world normal/tangent, orientation, orthogonality, bitangent and tangent-space normal mapping. Two shared mesh/material renderers, five render-path/MSAA configurations, nonuniform/mirrored scales and both UV handedness signs.', 'phases': []}
    with tempfile.TemporaryDirectory(prefix='infernux-tangent-frame-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        for name, source in (('Probe.vert', VERTEX), ('Probe.frag', FRAGMENT)):
            (project / 'Assets' / name).write_text(source, encoding='utf-8')
        mesh = project / 'Assets/Quad.obj'
        mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n', encoding='ascii')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = None
        failures = []
        complete = False
        try:
            frontend.init_renderer(320, 240, str(project))
            frontend.resize_game_render_target(320, 240)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            imported_mesh = AssetManager.import_asset(str(mesh), database=database)
            assert imported_mesh, imported_mesh.error
            for name in ('Probe.vert', 'Probe.frag'):
                imported = AssetManager.import_asset(str(project / 'Assets' / name), database=database)
                assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Tangent Camera')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component('Camera')
            material = InxMaterial.create_default_unlit()
            material.vert_shader_name = 'Deformed Tangent Probe'
            material.frag_shader_name = 'Deformed Tangent Consumer'
            objects = []
            for x in (-1., 1.):
                obj = scene.create_game_object(f'Tangent Quad {x}')
                obj.transform.position = Vector3(x, 0, 0)
                renderer = obj.add_component('MeshRenderer')
                renderer.set_mesh_asset_guid(imported_mesh.guid)
                renderer.set_material(0, material)
                objects.append(obj)
            plan = phase = frame = changed = 0
            ticket = baseline = None

            def apply_phase():
                _, scale, handedness = phases[phase]
                for key, value in zip(('scaleX', 'scaleY', 'scaleZ'), scale):
                    material.set_float(key, value)
                material.set_float('handedness', handedness)
                # Alter UV orientation in the authored hook without replacing
                # the mesh, so both renderers continue sharing the same asset.
                for obj in objects:
                    obj.transform.local_scale = Vector3(*scale)

            def set_pipeline():
                nonlocal pipeline
                if pipeline is not None:
                    pipeline.dispose()
                base, samples = plans[plan]
                pipeline = base()
                pipeline.shadow_resolution = 256
                if base is not inx.renderstack.DefaultDeferredPipeline:
                    pipeline.msaa_samples = samples
                frontend.set_render_pipeline(pipeline)

            set_pipeline()
            apply_phase()
            native.set_game_camera_enabled(True)

            def after_draw():
                nonlocal plan, phase, frame, changed, ticket, baseline, complete
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().astype(np.float32)
                        rgb = pixels[..., :3]
                        green = (rgb[..., 1] > .9) & (rgb[..., 0] < .05) & (rgb[..., 2] < .05)
                        bad = (rgb[..., 0] > .9) & (rgb[..., 2] > .9)
                        record = {'pipeline': plans[plan][0].name, 'samples': plans[plan][1], 'phase': phases[phase][0],
                                  'green': int(green.sum()), 'bad': int(bad.sum())}
                        assert green.sum() > 1000 and not bad.any(), record
                        assert green[:, :160].sum() > 400 and green[:, 160:].sum() > 400, record
                        issues = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                        assert not issues, issues
                        if phase == 0:
                            baseline = pixels
                        if phase == len(phases) - 1:
                            np.testing.assert_array_equal(pixels, baseline)
                        proof['phases'].append(record)
                        print('PASS ' + json.dumps(record), flush=True)
                        phase += 1
                        if phase == len(phases):
                            plan += 1
                            if plan == len(plans):
                                complete = True
                                native.exit()
                                return
                            phase = 0
                            set_pipeline()
                        apply_phase()
                        changed = frame
                        ticket = None
                    if frame > 600:
                        raise AssertionError('Tangent frame GPU audit timed out')
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None)
            if pipeline is not None:
                pipeline.dispose()
            native.cleanup()
    proof['passed'] = True
    if args.proof:
        args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print('PASS 25 deformed tangent-frame GPU phases', flush=True)


if __name__ == '__main__':
    main()
