"""Exact tutorial time-driven hooks: static shadow control versus animated mask."""
import argparse
import json
from pathlib import Path
import re
import tempfile
import time
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel,InxMaterial,LightShadows,SceneManager,Vector3

ROOT=Path(__file__).resolve().parents[3]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    guide=(ROOT/'docs/learn/vertex-stage.md').read_text(encoding='utf-8').split('<!-- language:zh -->',1)[0]
    blocks=re.findall(r'```glsl\n(.*?)```',guide,re.S)
    wave=next(b for b in blocks if 'Name "Wave"' in b)
    wind=next(b for b in blocks if 'Name "Wind Bent"' in b)
    wind_fragment=next(b for b in blocks if 'Name "Wind Surface"' in b)
    control=wind.replace('Name "Wind Bent"','Name "Wind Static Control"').replace('float bend = sin(phase) * 0.1;', 'float bend = 0.0;')
    assert control!=wind
    sources={'Wave.vert':wave,'Wind.vert':wind,'Control.vert':control,'Wind.frag':wind_fragment,
        'Red.frag':'''#version 450
ShaderInfo { Name "Wave Shadow Red" ShadingModel Unlit Cull Off }
void surface(out SurfaceData s) { s=InitSurfaceData(); s.albedo=vec3(1,0,0); }
''', 'Receiver.frag':'''#version 450
ShaderInfo { Name "Wave Shadow Receiver" ShadingModel PBR Cull Off }
void surface(out SurfaceData s) { s=InitSurfaceData(); s.albedo=vec3(.5); s.smoothness=0.; }
'''}
    plans=[inx.renderstack.DefaultForwardPipeline,inx.renderstack.DefaultForwardPlusPipeline,inx.renderstack.DefaultDeferredPipeline]
    stages=[('wave','Standard','Wave','Wave Shadow Red'),('wind','Wind Static Control','Wind Bent','Wind Surface')]
    proof={'package':inx.__file__,'scope':'Exact current Wave and Wind sources. Fixed camera/light/model/CPU bounds. Four static shadow masks must be identical before four animated masks change, outside a conservative caster screen region. Forward, Forward+ and Deferred. Independent temporary project GPU readbacks; no live Editor pixels.','cases':[]}
    with tempfile.TemporaryDirectory(prefix='infernux-exact-shadow-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):
            (project/name).mkdir()
        for name,source in sources.items():
            (project/'Assets'/name).write_text(source,encoding='utf-8')
        quad=project/'Assets/Quad.obj'
        quad.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n',encoding='ascii')
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        console=ConsolePanel()
        pipeline=None
        failures=[]
        complete=False
        try:
            frontend.init_renderer(320,240,str(project))
            frontend.resize_game_render_target(320,240)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(60.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            mesh=AssetManager.import_asset(str(quad),database=database)
            assert mesh,mesh.error
            for name in sources:
                result=AssetManager.import_asset(str(project/'Assets'/name),database=database)
                assert result,result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera=scene.create_game_object('Camera')
            camera.transform.position=Vector3(0,0,-6)
            camera.add_component('Camera')
            sun=scene.create_game_object('Sun')
            sun.transform.euler_angles=Vector3(0,45,0)
            light=sun.add_component('Light')
            light.shadows=LightShadows.Hard
            light.intensity=1.
            receiver_obj=scene.create_game_object('Receiver')
            receiver_obj.transform.local_scale=Vector3(6,4,1)
            receiver=receiver_obj.add_component('MeshRenderer')
            receiver.set_mesh_asset_guid(mesh.guid)
            receiver_mat=InxMaterial.create_default_lit()
            receiver_mat.frag_shader_name='Wave Shadow Receiver'
            receiver.set_material(0,receiver_mat)
            caster_obj=scene.create_game_object('Caster')
            caster_obj.transform.position=Vector3(-.5,0,-1.5)
            caster=caster_obj.add_component('MeshRenderer')
            caster.set_mesh_asset_guid(mesh.guid)
            material=InxMaterial.create_default_unlit()
            caster.set_material(0,material)
            expected_bounds=caster.get_world_bounds()
            plan=stage=phase=frame=requested=0
            ticket=baseline=None
            started=time.perf_counter()
            last_sample_time=float(native.get_shader_time_seconds())
            masks=[]
            samples=[]
            def apply():
                nonlocal pipeline,baseline,masks,samples
                if phase==0:
                    if pipeline is not None:
                        pipeline.dispose()
                    pipeline=plans[plan]()
                    pipeline.shadow_resolution=1024
                    if plans[plan] is not inx.renderstack.DefaultDeferredPipeline:
                        pipeline.msaa_samples=1
                    frontend.set_render_pipeline(pipeline)
                    material.frag_shader_name=stages[stage][3]
                    material.vert_shader_name=stages[stage][1]
                    caster.enabled=False
                    baseline=None
                    masks=[]
                    samples=[]
                elif phase==1:
                    caster.enabled=True
                elif phase==5:
                    material.vert_shader_name=stages[stage][2]
            apply()
            native.set_game_camera_enabled(True)
            def after_draw():
                nonlocal plan,stage,phase,frame,requested,ticket,baseline,complete,last_sample_time
                try:
                    frame+=1
                    if ticket is None and frame>=requested+12 and native.get_shader_time_seconds()>=last_sample_time+.3:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        image=ticket.result_numpy().astype(np.float32)
                        last_sample_time=float(native.get_shader_time_seconds())
                        issues=[e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not issues,issues
                        assert caster.get_world_bounds()==expected_bounds
                        if phase==0:
                            baseline=image
                        else:
                            shadow=(baseline[...,0]-image[...,0])>.08
                            shadow[60:180,80:170]=False
                            assert shadow.sum()>100,(plans[plan].name,stages[stage][0],phase,int(shadow.sum()))
                            masks.append(shadow.copy())
                            samples.append({'phase':phase,'time':float(native.get_shader_time_seconds()),'shadow_pixels':int(shadow.sum())})
                            proof['pending_samples']=samples.copy()
                        if phase==8:
                            assert all(np.array_equal(masks[0],m) for m in masks[1:4]),'Static shadow control changed over time'
                            differences=[int((masks[4]!=m).sum()) for m in masks[5:]]
                            assert max(differences)>10,('Animated shadow stayed static',stages[stage][0],differences)
                            record={'pipeline':plans[plan].name,'stage':stages[stage][0],'samples':samples.copy(),'animated_shadow_mask_changes':differences}
                            proof['cases'].append(record)
                            print('PASS '+json.dumps(record),flush=True)
                            phase=0
                            stage+=1
                            if stage==len(stages):
                                stage=0
                                plan+=1
                                if plan==len(plans):
                                    complete=True
                                    native.exit()
                                    return
                        else:
                            phase+=1
                        apply()
                        requested,ticket=frame,None
                    if time.perf_counter()-started>45:
                        raise AssertionError('Exact animated shadow audit timeout')
                except BaseException as exc:
                    failures.append(exc)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
            proof['passed']=True
        finally:
            frontend.set_render_pipeline(None)
            if pipeline is not None:
                pipeline.dispose()
            native.cleanup()
            if args.proof:
                args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
