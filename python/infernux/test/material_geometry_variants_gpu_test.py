"""Independent numeric GPU proof of generated deformation/cutout variants."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3

VERTEX = '''#version 450
ShaderInfo { Name "Tutorial Variant Shift" Properties { Float shiftX=0.4 Float shiftY=0.2 } }
void vertex(inout VertexInput v) { v.position.xy += vec2(material.shiftX, material.shiftY); }
'''
SURFACE = '''#version 450
ShaderInfo {
    Name "Tutorial Variant Cutout"
    ShadingModel Unlit
    Cull Off
    AlphaClip 0.5
    DepthTest Less
    Properties { Float clipEdge=0.5 }
}
void surface(out SurfaceData s) {
    s=InitSurfaceData();
    s.albedo=vec3(1,0,0);
    s.alpha=step(material.clipEdge,v_TexCoord.x);
}
'''
CONSUMER = '''#version 450
ShaderInfo {
    Name "Tutorial Variant Coverage Consumer"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D colorTex
        Texture2D normalTex
        Texture2D baseTex
        Texture2D depthTex
        Texture2D motionTex
        Texture2DUInt pickingTex
    }
    PushConstants pc { Float expected_object Float secondary_object }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    ivec2 p=ivec2(gl_FragCoord.xy);
    bool color=texelFetch(colorTex,p,0).a>0.9;
    bool normal=texelFetch(normalTex,p,0).a>0.9;
    bool base=texelFetch(baseTex,p,0).a>0.9;
    bool depth=texelFetch(depthTex,p,0).r<0.99999;
    uvec2 id=texelFetch(pickingTex,p,0).rg;
    bool picking=id==uvec2(uint(pc.expected_object),0u) ||
        (pc.secondary_object>0.0 && id==uvec2(uint(pc.secondary_object),0u));
    vec2 motion=texelFetch(motionTex,p,0).rg;
    bool moving=length(motion)>0.00001;
    if(color!=normal || color!=base || color!=depth || color!=picking || (!picking && id!=uvec2(0u)) || (!color && moving))
        outColor=vec4(1,0,1,1);
    else if(!color) outColor=vec4(0,0,0,1);
    else outColor=moving ? vec4(0,1,0,1) : vec4(1,0,0,1);
}
'''


class VariantsPipeline(inx.renderstack.RenderPipeline):
    name='Tutorial Vertex Surface Variant GPU'

    def define_topology(self, graph):
        self.builds+=1
        graph.set_msaa_samples(1)
        fmt=inx.rendergraph.Format
        depth=graph.create_texture('depth',format=fmt.D32_SFLOAT,samples=1)
        color=graph.create_texture('color_data',format=fmt.RGBA16_SFLOAT,samples=1)
        graph.add_pass('Color').write_color(color).write_depth(depth).set_clear(color=(0,0,0,0),depth=1.).draw_renderers(queue_range=(0,2500))
        normal=graph.create_texture('normal',format=fmt.RGBA16_SFLOAT,samples=1)
        base=graph.create_texture('base',format=fmt.RGBA16_SFLOAT,samples=1)
        motion=graph.create_texture('motion',format=fmt.RG16_SFLOAT,samples=1)
        for name,texture,material_pass in (('Normal',normal,'normal'),('BaseColor',base,'base_color'),('Motion',motion,'motion')):
            graph.add_pass(name).read(depth).write_color(texture).set_clear(color=(0,0,0,0)).draw_renderers(queue_range=(0,2500),material_pass=material_pass)
        picking=graph.create_texture('picking',format=fmt.RG32_UINT,samples=1)
        picking_depth=graph.create_texture('picking_depth',format=fmt.D32_SFLOAT,samples=1)
        graph.add_pass('Picking').write_color(picking).write_depth(picking_depth).set_clear(color=(0,0,0,0),depth=1.).draw_renderers(queue_range=(0,2500),material_pass='picking')
        target=graph.create_texture('color',camera_target=True)
        p=graph.add_pass('CompareVariants').write_color(target)
        for key,texture in (('colorTex',color),('normalTex',normal),('baseTex',base),('depthTex',depth),('motionTex',motion),('pickingTex',picking)):
            p.set_texture(key,texture)
        p.set_param('expected_object',float(self.object_id)).fullscreen_quad('Tutorial Variant Coverage Consumer')
        p.set_param('secondary_object',float(self.secondary_id))
        graph.screen_ui_overlay_section(resources={'color'})
        graph.set_output(target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instances',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-tutorial-variants-') as root:
        project=Path(root)
        for directory in ('Assets','Packages','ProjectSettings'):
            (project/directory).mkdir()
        mesh_path=project/'Assets/Probe.obj'
        mesh_path.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n',encoding='ascii')
        for file,content in (('Shift.vert',VERTEX),('Cutout.frag',SURFACE),('Consumer.frag',CONSUMER)):
            (project/'Assets'/file).write_text(content,encoding='ascii')
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        console=ConsolePanel()
        pipeline=VariantsPipeline()
        pipeline.builds=0
        failures,completed=[],False
        proof={'scope':'Independent real GPU masks compare color, normal, base-color, depth and integer picking coverage for material vertex deformation and surface alpha clipping; motion vectors distinguish transform motion from post-hook local changes. No MCP Editor pixel access.', 'instances':args.instances, 'phases':[]}
        try:
            frontend.init_renderer(160,120,str(project))
            frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            mesh=AssetManager.import_asset(str(mesh_path),database=database)
            assert mesh,mesh.error
            for file in ('Shift.vert','Cutout.frag','Consumer.frag'):
                result=AssetManager.import_asset(str(project/'Assets'/file),database=database)
                assert result,result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera=scene.create_game_object('Variant Camera')
            camera.transform.position=Vector3(0,0,-5)
            camera.add_component('Camera')
            material=InxMaterial.create_default_unlit()
            material.vert_shader_name='Tutorial Variant Shift'
            material.frag_shader_name='Tutorial Variant Cutout'
            for key,value in (('shiftX',.4),('shiftY',.2),('clipEdge',.5)):
                material.set_float(key,value)
            objects=[]
            positions=(0,) if args.instances==1 else (-1,1)
            for index,x in enumerate(positions):
                obj=scene.create_game_object(f'Deformed Cutout {index}')
                obj.transform.position=Vector3(x,0,0)
                renderer=obj.add_component('MeshRenderer')
                renderer.set_mesh_asset_guid(mesh.guid)
                renderer.set_material(0,material)
                objects.append(obj)
            pipeline.object_id=objects[0].id
            pipeline.secondary_id=objects[1].id if args.instances==2 else 0
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            phases=('cutout','solid','cutout_restored','vertex_property_changed','object_motion','procedural_local_motion','static_restored')
            phase,frames,changed,ticket=0,0,0,None
            initial=None
            initial_pixels=None
            last_image=None

            def after_draw():
                nonlocal phase,frames,changed,ticket,completed,initial,initial_pixels,last_image
                try:
                    frames+=1
                    if ticket is None and frames>=changed+8:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        rgb=pixels[...,:3]
                        red=(rgb[...,0]>.9)&(rgb[...,1]<.05)&(rgb[...,2]<.05)
                        green=(rgb[...,1]>.9)&(rgb[...,0]<.05)&(rgb[...,2]<.05)
                        bad=(rgb[...,0]>.9)&(rgb[...,2]>.9)
                        coverage=red|green
                        record={'phase':phases[phase],'covered':int(coverage.sum()),'moving':int(green.sum()),'mismatch':int(bad.sum()),'builds':pipeline.builds}
                        record['game_draw_calls']=native.renderer_frame_snapshot['game_draw_call_count']
                        if coverage.any():
                            record['centroid_x']=float(np.where(coverage)[1].mean())
                        print(record,flush=True)
                        assert not bad.any() and coverage.sum()>100,record
                        assert pipeline.builds==1,record
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        if phase==0:
                            initial=record
                            initial_pixels=pixels
                            assert record['centroid_x']>90,record
                        elif phase==1:
                            assert record['covered']>initial['covered']*1.8,record
                        elif phase in (2,6):
                            np.testing.assert_array_equal(pixels,initial_pixels)
                        elif phase==3:
                            assert record['centroid_x']<initial['centroid_x']-10,record
                        if phase==4:
                            assert green.sum()>coverage.sum()*.9,record
                        else:
                            assert not green.any(),record
                        if phase==5:
                            assert last_image is not None and not np.array_equal(last_image,pixels),record
                        proof['phases'].append(record)
                        last_image=pixels
                        if phase==6:
                            proof['passed']=True
                            print(json.dumps(proof,indent=2),flush=True)
                            completed=True
                            native.exit()
                            return
                        phase+=1
                        if phase==1:
                            material.set_float('clipEdge',0.)
                        elif phase==2:
                            material.set_float('clipEdge',.5)
                        elif phase==3:
                            material.set_float('shiftX',-.4)
                        elif phase==6:
                            for obj,x in zip(objects,positions):
                                obj.transform.position=Vector3(x,0,0)
                            material.set_float('shiftX',.4)
                        changed,ticket=frames,None
                    if phase==4:
                        for index,obj in enumerate(objects):
                            position=obj.transform.position
                            obj.transform.position=Vector3(position.x+(.03 if index==0 else -.03),0,0)
                    elif phase==5:
                        material.set_float('shiftX',material.get_float('shiftX')+.03)
                    if frames>250:
                        raise AssertionError('Variant GPU audit timed out')
                except BaseException as error:
                    failures.append(error)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__=='__main__':
    main()
