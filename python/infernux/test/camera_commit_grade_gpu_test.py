"""Verify Camera commit, current-frame consumption and nearest RenderTexture sampling."""

import argparse,json,re,tempfile
from pathlib import Path
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel,SceneManager,Vector3
from infernux.engine.path_utils import resolved_path
ROOT=Path(resolved_path(__file__)).parents[3]

blocks=re.findall(r'```python\n(.*?)```',(ROOT/'docs/learn/rendergraph-advanced.md').read_text(encoding='utf-8'),re.S)
commit=[block for block in blocks if block.startswith('with graph.add_pass("CommitGrade")')]
assert len(commit)==2 and commit[0]==commit[1]
GRADE='''#version 450
ShaderInfo { Name "Tutorial Commit Grade Source" Hidden On Capabilities [Fullscreen]
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() { outColor=vec4(0.25,0.5,0.75,1); }
'''

class GradePipeline(inx.renderstack.RenderPipeline):
    name='Tutorial Camera Commit Grade'
    def define_topology(self,g):
        self.builds+=1;g.set_msaa_samples(1);f=inx.rendergraph.Format
        color=g.create_texture('color',camera_target=True)
        graded=g.create_texture('graded',format=f.RGBA16_SFLOAT,samples=1)
        g.add_pass('ProduceGrade').write_color(graded).fullscreen_quad('Tutorial Commit Grade Source')
        exec(compile(commit[0],str(ROOT/'docs/learn/rendergraph-advanced.md'),'exec'),{'graph':g,'camera_color':color,'graded':graded})
        depth=g.create_texture('depth',format=f.D32_SFLOAT,samples=1)
        g.add_pass('DrawAfterCommit').write_color(color).write_depth(depth).set_clear(depth=1).draw_renderers(queue_range=(0,2500))
        g.screen_ui_overlay_section(resources={'color'});g.set_output(color)
    def render_camera(self,context,camera,culling):
        self.views[camera.component_id]={'graph_instance_id':int(context.graph_instance_id),'source_revision':int(self._standalone_desc.source_revision),'output_samples':int(context.output_samples),'visible_objects':int(culling.visible_object_count)}
        super().render_camera(context,camera,culling)

class MonitorPipeline(inx.renderstack.RenderPipeline):
    name='Tutorial Saved Camera Monitor'
    def define_topology(self,g):
        self.builds+=1;g.set_msaa_samples(1)
        sampled=g.import_texture('producer',self.target)
        color=g.create_texture('color',camera_target=True)
        g.add_pass('ReadCurrentCameraOutput').write_color(color).set_texture('_SourceTex',sampled).fullscreen_quad('Fullscreen Blit')
        g.screen_ui_overlay_section(resources={'color'});g.set_output(color)
    def render_camera(self,context,camera,culling):
        self.views[camera.component_id]={'graph_instance_id':int(context.graph_instance_id),'source_revision':int(self._standalone_desc.source_revision),'output_samples':int(context.output_samples),'visible_objects':int(culling.visible_object_count)}
        super().render_camera(context,camera,culling)

class CameraRouter(inx.renderstack.RenderPipeline):
    name='Tutorial Camera Router'
    def render(self,context,camera):
        if camera.component_id==self.producer_id:self.grade.render(context,camera)
        else:
            assert camera.component_id==self.monitor_id,camera.component_id
            self.monitor.render(context,camera)
    def dispose(self):
        self.grade.dispose()
        if self.monitor is not None:self.monitor.dispose()
        super().dispose()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=('screen','asset'),required=True);args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-camera-commit-grade-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):(project/name).mkdir()
        (project/'Assets/Grade.frag').write_text(GRADE,encoding='ascii')
        mesh=project/'Assets/Quad.obj';mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n',encoding='ascii')
        rt_path=project/'Assets/CameraOutput.rendertexture'
        rt_path.write_text(json.dumps({'$type':'render_texture','size':{'width':53,'height':29},'format':'rgba16_sfloat','depth_format':'d32_sfloat','samples':1,'filter':'nearest','storage':False,'sampled_depth':False}),encoding='utf-8')
        frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();router=CameraRouter();router.grade=GradePipeline();router.grade.builds=0;router.grade.views={};router.monitor=None
        proof={'scope':'Exact bilingual CommitGrade snippet renders processed color then geometry into a Camera; saved target adds an input-only same-frame consumer with reversed Camera depth and resize. Independent GPU readback.','target':args.target,'phases':[]};failures=[];complete=False;target=None
        try:
            frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
            database=frontend.get_asset_database()
            for path in (mesh,project/'Assets/Grade.frag'):
                imported=AssetManager.import_asset(str(path),database=database);assert imported,imported.error
                if path==mesh:mesh_guid=imported.guid
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():scene.destroy_game_object(obj)
            owner=scene.create_game_object('Commit Camera');owner.transform.position=Vector3(0,0,-5);producer=owner.add_component(inx.Camera);producer.culling_mask=1;producer.depth=1
            router.producer_id=producer.component_id
            obj=scene.create_game_object('Draw After Commit');obj.transform.position=Vector3(-.4,0,0)
            renderer=obj.add_component(inx.MeshRenderer);renderer.set_mesh_asset_guid(mesh_guid)
            material=inx.Material.create_unlit();material.set_color('baseColor',1,0,0,1);renderer.set_material(0,material)
            if args.target=='asset':
                imported=AssetManager.import_asset(str(rt_path),database=database);assert imported,imported.error
                target=inx.RenderTexture.load_by_guid(imported.guid);producer.target_texture=target
                assert producer.target_texture is target
                owner=scene.create_game_object('Monitor Camera');owner.transform.position=Vector3(0,0,-5);monitor=owner.add_component(inx.Camera);monitor.culling_mask=0;monitor.depth=-1
                scene.main_camera=monitor;router.monitor_id=monitor.component_id
                router.monitor=MonitorPipeline();router.monitor.target=target;router.monitor.builds=0;router.monitor.views={}
            else:scene.main_camera=producer
            frontend.set_render_pipeline(router);native.set_game_camera_enabled(True)
            phases=('initial','moved_right','restored') if args.target=='screen' else ('consumer_first','producer_moved','producer_first','resized_target','restored')
            frame=0;changed=0;phase=0;ticket=None;initial=None
            def after_draw():
                nonlocal frame,changed,phase,ticket,initial,complete
                try:
                    frame+=1
                    if ticket is None and frame>=changed+8:ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32);rgb=pixels[...,:3]
                        red=(rgb[...,0]>.9)&(rgb[...,1]<.02)&(rgb[...,2]<.02)
                        expected=np.where(np.array([.25,.5,.75])<=.0031308,np.array([.25,.5,.75])*12.92,1.055*np.array([.25,.5,.75])**(1/2.4)-.055)
                        background=np.all(np.abs(rgb-expected)<.004,axis=-1)
                        bad=~(red|background)
                        record={'phase':phases[phase],'red_pixels':int(red.sum()),'background_pixels':int(background.sum()),'unexpected_pixels':int(bad.sum()),'centroid_x':float(np.where(red)[1].mean()) if red.any() else None,'grade_builds':router.grade.builds,'grade_views':router.grade.views.copy()}
                        if target is not None:record.update(target_size=[target.width,target.height],monitor_builds=router.monitor.builds,monitor_views=router.monitor.views.copy())
                        print(record,flush=True)
                        if bad.any():
                            values,counts=np.unique(rgb[bad],axis=0,return_counts=True)
                            largest=np.argsort(counts)[-12:]
                            print('Unexpected color samples '+json.dumps([{'rgb':values[i].tolist(),'count':int(counts[i])} for i in largest]),flush=True)
                        assert red.sum()>100 and background.sum()>1000 and not bad.any(),record
                        assert router.grade.builds==1 and (router.monitor is None or router.monitor.builds==1),record
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        if phase==0:initial=pixels;assert record['centroid_x']<75,record
                        elif phase==len(phases)-1:np.testing.assert_array_equal(pixels,initial)
                        else:assert record['centroid_x']>85,record
                        if router.monitor is not None:
                            a=router.grade.views[router.producer_id]['graph_instance_id'];b=router.monitor.views[router.monitor_id]['graph_instance_id'];assert a!=b and a>0 and b>0,(a,b)
                        if proof['phases']:
                            first=proof['phases'][0]
                            assert record['grade_views']==first['grade_views'],record
                            if router.monitor is not None:assert record['monitor_views']==first['monitor_views'],record
                        proof['phases'].append(record)
                        if phase==len(phases)-1:complete=True;proof['passed']=True;native.exit();return
                        phase+=1
                        if phase==1:obj.transform.position=Vector3(.4,0,0)
                        elif args.target=='asset' and phase==2:monitor.depth=3
                        elif args.target=='asset' and phase==3:assert target.resize(97,61)
                        else:
                            obj.transform.position=Vector3(-.4,0,0)
                            if target is not None:assert target.resize(53,29);monitor.depth=-1
                        changed=frame;ticket=None
                    if frame>150:raise AssertionError('Camera commit GPU audit did not complete')
                except BaseException as exc:failures.append(exc);native.exit()
            native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
            if failures:raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None);router.dispose();native.cleanup()
        print(json.dumps(proof,indent=2),flush=True)
        print('PASS exact Camera commit, later geometry and target '+args.target,flush=True)

if __name__=='__main__':main()
