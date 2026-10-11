"""Independent shared-topology multi-camera GPU audit in a temporary project."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags,ConsolePanel,SceneManager,Vector3,vec4f,LightType


MONITOR='''#version 450
ShaderInfo { Name "Tutorial View Lighting Monitor" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D leftTex Texture2D rightTex }
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() {
 outColor=inUV.x<0.5 ? texture(leftTex,vec2(inUV.x*2,inUV.y))
                    : texture(rightTex,vec2(inUV.x*2-1,inUV.y));
}
'''

class Monitor(inx.renderstack.RenderPipeline):
    name='Tutorial View Lighting Monitor'
    def define_topology(self,g):
        self.builds+=1;g.set_msaa_samples(1)
        a=g.import_texture('left',self.targets[0]);b=g.import_texture('right',self.targets[1])
        color=g.create_texture('color',camera_target=True)
        g.add_pass('CompareViews').write_color(color).set_texture('leftTex',a).set_texture('rightTex',b).fullscreen_quad('Tutorial View Lighting Monitor')
        g.screen_ui_overlay_section(resources={'color'});g.set_output(color)
    def render_camera(self,context,camera,culling):
        self.native_id=int(context.graph_instance_id)
        assert culling.visible_object_count==0
        super().render_camera(context,camera,culling)

class Router(inx.renderstack.RenderPipeline):
    name='Tutorial View Lighting Router'
    def render(self,context,camera):
        (self.monitor if camera.component_id==self.monitor_id else self.shared).render(context,camera)
    def dispose(self):
        self.shared.dispose();self.monitor.dispose();super().dispose()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pipeline',choices=('forward','forward_plus','deferred'),required=True);args=parser.parse_args()
    base={'forward':inx.renderstack.DefaultForwardPipeline,'forward_plus':inx.renderstack.DefaultForwardPlusPipeline,'deferred':inx.renderstack.DefaultDeferredPipeline}[args.pipeline]
    class Shared(base):
        name='Tutorial View Lighting '+args.pipeline
        def define_topology(self,g):
            self.builds+=1;super().define_topology(g)
        def render_camera(self,context,camera,culling):
            self.views[camera.component_id]={'light_count':int(culling.visible_light_count),'native_id':int(context.graph_instance_id),'source_revision':int(self._standalone_desc.source_revision),'visible':int(culling.visible_object_count),'output_samples':int(context.output_samples)}
            super().render_camera(context,camera,culling)
    with tempfile.TemporaryDirectory(prefix='infernux-view-lighting-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):(project/name).mkdir()
        mesh=project/'Assets/Quad.obj';mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n',encoding='ascii')
        shader=project/'Assets/Monitor.frag';shader.write_text(MONITOR,encoding='ascii')
        frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();router=Router()
        router.shared=Shared();router.shared.builds=0;router.shared.views={};router.shared.msaa_samples=1;router.shared.shadow_resolution=256
        router.monitor=Monitor();router.monitor.builds=0
        proof={'scope':'Two spatially separated cameras share one built-in pipeline with differently colored point lights and material layer masks. Change one light independently, move cameras/geometry, reverse order, resize one target and exactly restore. GPU lighting evidence, not a shadow/history claim.','pipeline':args.pipeline,'phases':[]};failures=[];complete=False
        try:
            frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
            db=frontend.get_asset_database()
            imported=AssetManager.import_asset(str(mesh),database=db);assert imported,imported.error;mesh_guid=imported.guid
            imported=AssetManager.import_asset(str(shader),database=db);assert imported,imported.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():scene.destroy_game_object(obj)
            cameras=[];owners=[];targets=[];objects=[];lights=[]
            for i,(width,height,color,x) in enumerate(((53,29,(1,0,0,1),-.4),(97,61,(0,0,1,1),.4))):
                path=project/f'Assets/View{i}.rendertexture'
                path.write_text(json.dumps({'$type':'render_texture','size':{'width':width,'height':height},'format':'rgba16_sfloat','depth_format':'d32_sfloat','samples':1,'filter':'nearest','storage':False,'sampled_depth':args.pipeline=='deferred'}),encoding='utf-8')
                imported=AssetManager.import_asset(str(path),database=db);assert imported,imported.error
                target=inx.RenderTexture.load_by_guid(imported.guid);targets.append(target)
                owner=scene.create_game_object('View'+str(i));owner.transform.position=Vector3(i*100,0,-5);owners.append(owner)
                camera=owner.add_component(inx.Camera);camera.culling_mask=1<<i;camera.depth=i+1;camera.target_texture=target;camera.background_color=vec4f(0,0,0,1);camera.clear_flags=CameraClearFlags.SolidColor;cameras.append(camera)
                obj=scene.create_game_object('Layer'+str(i));obj.layer=i;obj.transform.position=Vector3(i*100+x,0,0);objects.append(obj)
                renderer=obj.add_component(inx.MeshRenderer);renderer.set_mesh_asset_guid(mesh_guid)
                material=inx.Material.create_lit();material.set_color('baseColor',1,1,1,1);material.set_float('metallic',0);material.set_float('smoothness',0.3);renderer.set_material(0,material)
                light_owner=scene.create_game_object('Light'+str(i));light_owner.transform.position=Vector3(i*100,0,-2)
                light=light_owner.add_component(inx.Light);light.light_type=LightType.Point;light.color=color[:3];light.intensity=4;light.range=12;light.culling_mask=1<<i;lights.append(light)
            owner=scene.create_game_object('Monitor');owner.transform.position=Vector3(0,0,-5);monitor=owner.add_component(inx.Camera);monitor.culling_mask=0;monitor.depth=-1;scene.main_camera=monitor
            router.monitor_id=monitor.component_id;router.monitor.targets=targets
            frontend.set_render_pipeline(router);native.set_game_camera_enabled(True)
            phases=('initial','red_dimmed','red_restored','camera_a_moved','object_b_moved','order_reversed','target_a_resized','restored')
            frame=0;changed=0;phase=0;ticket=None;initial=None;previous=None
            def after_draw():
                nonlocal frame,changed,phase,ticket,initial,previous,complete
                try:
                    frame+=1
                    if ticket is None and frame>=changed+8:ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32);left=pixels[:,:80,:3];right=pixels[:,80:,:3]
                        red=np.any(left>.1,axis=-1)
                        blue=np.any(right>.1,axis=-1)
                        if red.sum()<=50 or blue.sum()<=50:
                            print('Unexpected view colors '+json.dumps({'left_max':left.max(axis=(0,1)).tolist(),'right_max':right.max(axis=(0,1)).tolist(),'left_unique':np.unique(left.reshape(-1,3),axis=0).tolist()[:15],'right_unique':np.unique(right.reshape(-1,3),axis=0).tolist()[:15],'console':console._get_visible_log_snapshot(2000)}),flush=True)
                        assert red.sum()>50 and blue.sum()>50,(red.sum(),blue.sum())
                        if phase in (0,2):assert float(left[...,0][red].mean())>float(left[...,2][red].mean())+.15 and float(right[...,2][blue].mean())>float(right[...,0][blue].mean())+.15
                        record={'phase':phases[phase],'left_red_mean':float(left[...,0][red].mean()),'right_blue_mean':float(right[...,2][blue].mean()),'left_geometry_pixels':int(red.sum()),'right_geometry_pixels':int(blue.sum()),'left_centroid':float(np.where(red)[1].mean()),'right_centroid':float(np.where(blue)[1].mean()),'shared_builds':router.shared.builds,'views':router.shared.views.copy(),'monitor_id':router.monitor.native_id,'target_sizes':[[t.width,t.height] for t in targets]}
                        print(record,flush=True)
                        assert router.shared.builds==1 and router.monitor.builds==1,record
                        assert len(record['views'])==2 and all(v['visible']==1 and v['output_samples']==1 for v in record['views'].values()),record
                        assert len({v['native_id'] for v in record['views'].values()}|{record['monitor_id']})==3,record
                        assert len({v['source_revision'] for v in record['views'].values()})==1,record
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        if phase==0:initial=pixels
                        else:
                            assert record['views']==proof['phases'][0]['views'],record
                            if phase==1:
                                np.testing.assert_array_equal(pixels[:,80:],previous[:,80:]);assert record['left_red_mean']<proof['phases'][0]['left_red_mean']-.05
                            elif phase==2:np.testing.assert_array_equal(pixels,initial)
                            elif phase==3:
                                np.testing.assert_array_equal(pixels[:,80:],previous[:,80:]);assert record['left_centroid']<proof['phases'][0]['left_centroid']-3
                            elif phase==4:
                                np.testing.assert_array_equal(pixels[:,:80],previous[:,:80]);assert record['right_centroid']<proof['phases'][0]['right_centroid']-3
                            elif phase==5:np.testing.assert_array_equal(pixels,previous)
                            elif phase==6:np.testing.assert_array_equal(pixels[:,80:],previous[:,80:])
                            else:np.testing.assert_array_equal(pixels,initial)
                        proof['phases'].append(record);previous=pixels
                        if phase==len(phases)-1:complete=True;proof['passed']=True;native.exit();return
                        phase+=1
                        if phase==1:lights[0].intensity=2
                        elif phase==2:lights[0].intensity=4
                        elif phase==3:owners[0].transform.position=Vector3(.8,0,-5)
                        elif phase==4:objects[1].transform.position=Vector3(99.6,0,0)
                        elif phase==5:cameras[0].depth=4;cameras[1].depth=3;monitor.depth=5
                        elif phase==6:assert targets[0].resize(79,47)
                        else:
                            owners[0].transform.position=Vector3(0,0,-5);objects[1].transform.position=Vector3(100.4,0,0)
                            cameras[0].depth=1;cameras[1].depth=2;monitor.depth=-1;assert targets[0].resize(53,29)
                        changed=frame;ticket=None
                    if frame>180:raise AssertionError('Shared camera audit timed out')
                except BaseException as exc:failures.append(exc);native.exit()
            native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
            if failures:raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None);router.dispose();native.cleanup()
        print(json.dumps(proof,indent=2),flush=True)
        print('PASS view-lighting '+args.pipeline,flush=True)

if __name__=='__main__':main()
