"""Check each view's previous-frame marker and first-frame validity on the GPU."""
import json,tempfile
from pathlib import Path
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel,SceneManager,Vector3,GraphParameterBlockUpdate
from shared_camera_gpu_test import Monitor,Router,MONITOR


SOURCE='''#version 450
ShaderInfo { Name "Tutorial View History Source" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D historyTex }
 PushConstants pc { Float red Float blue Float oldRed Float oldBlue Float expectedValid Float _InfernuxHistoryValid }
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() {
 vec4 old=texture(historyTex,inUV);
 bool valid=pc._InfernuxHistoryValid<0.5
     ? all(lessThan(abs(old),vec4(0.001)))
     : pc.expectedValid>0.5 && abs(old.r-pc.oldRed)<0.001 && abs(old.b-pc.oldBlue)<0.001
       && abs(old.g-1)<0.001 && abs(old.a-1)<0.001;
 // Preserve an earlier failure until history is explicitly invalidated.
 outColor=vec4(pc.red,valid ? 1 : 2,pc.blue,1);
}
'''
CHECK='''#version 450
ShaderInfo { Name "Tutorial View History Check" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D historyTex Texture2D currentTex }
 PushConstants pc { Float oldRed Float oldBlue Float expectedValid Float _InfernuxHistoryValid }
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() {
 vec4 old=texture(historyTex,inUV);
 bool valid=pc._InfernuxHistoryValid<0.5
     ? all(lessThan(abs(old),vec4(0.001)))
     : pc.expectedValid>0.5 && abs(old.r-pc.oldRed)<0.001 && abs(old.b-pc.oldBlue)<0.001;
 valid=valid && abs(texture(currentTex,inUV).g-1)<0.001;
 outColor=valid ? vec4(0,1,pc._InfernuxHistoryValid<0.5 ? 1 : 0,1) : vec4(1,0,0,1);
}
'''
def update(context,block,revision,values):
    item=GraphParameterBlockUpdate();item.id=block;item.revision=revision;item.values=list(values.items())
    context.update_parameter_blocks([item])

class History(inx.renderstack.RenderPipeline):
    name='Tutorial Per View History'
    def define_topology(self,g):
        self.builds+=1;g.set_msaa_samples(1);f=inx.rendergraph.Format
        raw=g.create_texture('raw',format=f.RGBA16_SFLOAT,samples=1)
        old,new=g.create_temporal_history('view_history',format=f.RGBA16_SFLOAT)
        color=g.create_texture('color',camera_target=True)
        g.add_pass('CurrentMarker').write_color(raw).set_texture('historyTex',old).bind_parameter_block('marker',{'red':0,'blue':0,'oldRed':0,'oldBlue':0,'expectedValid':0,'_InfernuxHistoryValid':0}).fullscreen_quad('Tutorial View History Source')
        g.add_pass('CheckPreviousView').write_color(color).set_texture('historyTex',old).set_texture('currentTex',raw).bind_parameter_block('expected',{'oldRed':0,'oldBlue':0,'expectedValid':0,'_InfernuxHistoryValid':0}).fullscreen_quad('Tutorial View History Check')
        g.add_copy_pass('CommitMarker').copy_texture(raw,new).set_side_effect()
        g.screen_ui_overlay_section(resources={'color'});g.set_output(color)
    def render_camera(self,context,camera,culling):
        key=camera.component_id
        if not context.is_graph_revision_current(self._standalone_desc.source_revision):
            context.apply_graph(self._standalone_desc)
        count=self.counts.get(key,0)+1;self.counts[key]=count
        reset=key not in self.previous or key in self.reset_next
        if key in self.reset_next:self.reset_next.remove(key);self.resets[key]=self.resets.get(key,0)+1
        old=(0,0) if reset else self.previous[key]
        marker=(.25 if count%2 else .5,0) if key==self.left_id else (0,.625 if count%2 else .875)
        update(context,'marker',count,{'red':marker[0],'blue':marker[1],'oldRed':old[0],'oldBlue':old[1],'expectedValid':0 if reset else 1,'_InfernuxHistoryValid':0})
        update(context,'expected',count,{'oldRed':old[0],'oldBlue':old[1],'expectedValid':0 if reset else 1,'_InfernuxHistoryValid':0})
        context.submit_culling(culling)
        self.previous[key]=marker
        self.views[key]={'native_id':int(context.graph_instance_id),'source_revision':int(self._standalone_desc.source_revision),'output_samples':int(context.output_samples)}

def main():
    with tempfile.TemporaryDirectory(prefix='infernux-view-history-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):(project/name).mkdir()
        for filename,shader in [('Source.frag',SOURCE),('Check.frag',CHECK),('Monitor.frag',MONITOR)]: (project/'Assets'/filename).write_text(shader,encoding='ascii')
        frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();router=Router()
        router.shared=History();router.shared.builds=0;router.shared.counts={};router.shared.previous={};router.shared.reset_next=set();router.shared.resets={};router.shared.views={}
        router.monitor=Monitor();router.monitor.builds=0;failures=[];complete=False
        proof={'scope':'Distinct per-camera previous-frame markers in one shared topology; first history read and target resize validity checked inside a GPU shader. Exact current-frame monitor consumption under reversed scheduling and target replacement.','phases':[]}
        try:
            frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
            db=frontend.get_asset_database()
            for path in (project/'Assets').glob('*.frag'):
                imported=AssetManager.import_asset(str(path),database=db);assert imported,imported.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():scene.destroy_game_object(obj)
            cameras=[];targets=[]
            for i,(width,height) in enumerate(((53,29),(97,61))):
                path=project/f'Assets/View{i}.rendertexture';path.write_text(json.dumps({'$type':'render_texture','size':{'width':width,'height':height},'format':'rgba16_sfloat','depth_format':'d32_sfloat','samples':1,'filter':'nearest','storage':False,'sampled_depth':False}),encoding='utf-8')
                imported=AssetManager.import_asset(str(path),database=db);assert imported,imported.error
                target=inx.RenderTexture.load_by_guid(imported.guid);targets.append(target)
                owner=scene.create_game_object('History'+str(i));owner.transform.position=Vector3(0,0,-5)
                camera=owner.add_component(inx.Camera);camera.depth=i+1;camera.target_texture=target;cameras.append(camera)
            router.shared.left_id=cameras[0].component_id
            owner=scene.create_game_object('Monitor');owner.transform.position=Vector3(0,0,-5);monitor=owner.add_component(inx.Camera);monitor.culling_mask=0;monitor.depth=-1;scene.main_camera=monitor
            router.monitor_id=monitor.component_id;router.monitor.targets=targets
            frontend.set_render_pipeline(router);native.set_game_camera_enabled(True)
            phases=('initialized','steady','order_reversed','left_resized_first_read','left_resized_steady','restored_first_read','restored_steady')
            phase=0;frame=0;changed=0;ticket=None
            def after_draw():
                nonlocal phase,frame,changed,ticket,complete
                try:
                    frame+=1
                    if ticket is None and frame>=changed+(1 if phase in (3,5) else 8):ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32);rgb=pixels[...,:3]
                        green=(rgb[...,0]<.01)&(rgb[...,1]>.99)&(rgb[...,2]<.01)
                        cyan=(rgb[...,0]<.01)&(rgb[...,1]>.99)&(rgb[...,2]>.99)
                        record={'phase':phases[phase],'valid_history_pixels':int(green.sum()),'zero_history_pixels':int(cyan.sum()),'bad_pixels':int((~(green|cyan)).sum()),'frame_counts':router.shared.counts.copy(),'shared_builds':router.shared.builds,'views':router.shared.views.copy(),'resets':router.shared.resets.copy()}
                        print(record,flush=True)
                        assert green.all(),record
                        assert router.shared.builds==1 and router.monitor.builds==1
                        assert len(record['views'])==2 and len({v['native_id'] for v in record['views'].values()})==2
                        assert len({v['source_revision'] for v in record['views'].values()})==1
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        if proof['phases']:assert record['views']==proof['phases'][0]['views']
                        proof['phases'].append(record)
                        if phase==len(phases)-1:complete=True;proof['passed']=True;native.exit();return
                        phase+=1
                        if phase==2:cameras[0].depth=4;cameras[1].depth=3;monitor.depth=5
                        elif phase==3:assert targets[0].resize(79,47);router.shared.reset_next.add(cameras[0].component_id)
                        elif phase==5:
                            assert targets[0].resize(53,29);router.shared.reset_next.add(cameras[0].component_id)
                            cameras[0].depth=1;cameras[1].depth=2;monitor.depth=-1
                        changed=frame
                        ticket=frontend.request_render_target_readback(True) if phase in (3,5) else None
                    if frame>160:raise AssertionError('Per-view history audit timed out')
                except BaseException as exc:failures.append(exc);native.exit()
            native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
            if failures:raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None);router.dispose();native.cleanup()
        print(json.dumps(proof,indent=2),flush=True)
        print('PASS per-view GPU previous markers, invalidation and shared topology',flush=True)

if __name__=='__main__':main()
