"""Independent GPU proof of tutorial Mesh material raster state and shader stencil."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, RenderPipelineCallback, SceneManager, Vector3
from infernux.rendergraph import Format, RenderGraph

SURFACE = '''#version 450
ShaderInfo {
 Name "Raster Near" ShadingModel Unlit Cull Back DepthWrite On DepthTest Less
 Properties { Color baseColor=[1,1,1,1] }
}
void surface(out SurfaceData s) { s=InitSurfaceData(); s.albedo=material.baseColor.rgb; }
'''


class RasterPipeline(RenderPipelineCallback):
    def __init__(self, stencil=False):
        super().__init__()
        graph = RenderGraph('Tutorial Raster State')
        graph.set_msaa_samples(1)
        color = graph.create_texture('color', camera_target=True)
        depth = graph.create_texture('raster_depth', format=Format.D24_UNORM_S8_UINT if stencil else Format.D32_SFLOAT)
        graph.add_pass('Clear').write_color(color).write_depth(depth).set_clear(color=(0,0,0,1), depth=1.)
        for name, queue in (('NearOrStencilWriter',1000), ('FarOrStencilReader',2000)):
            graph.add_pass(name).write_color(color).write_depth(depth).draw_renderers(queue_range=(queue,queue), material_pass='forward')
        graph.set_output(color)
        self.description=graph.build()

    def render(self, context, camera):
        if not context.render_compiled(camera, self.description.source_revision):
            context.render_with_graph(camera, self.description)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    proof={'passed':False,'package':inx.__file__,'phases':[],'scope':'Independent temporary project; actual GPU culling/depth override/reload and shader stencil with and without a stencil attachment. No live Editor pixels.'}
    with tempfile.TemporaryDirectory(prefix='infernux-raster-state-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):
            (project/name).mkdir()
        path=project/'Assets/Near.frag'
        path.write_text(SURFACE,encoding='utf-8')
        (project/'Assets/Writer.frag').write_text(SURFACE.replace('Raster Near','Stencil Writer').replace('DepthTest Less','DepthTest Always Stencil "always,1,replace,keep,keep"'),encoding='utf-8')
        (project/'Assets/Reader.frag').write_text(SURFACE.replace('Raster Near','Stencil Reader').replace('DepthTest Less','DepthTest Always Stencil "equal,1,keep,keep,keep"'),encoding='utf-8')
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        console=ConsolePanel()
        failures,complete=[],False
        pipelines=[]
        try:
            frontend.init_renderer(160,120,folder)
            frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            for shader in project.glob('Assets/*.frag'):
                result=AssetManager.import_asset(str(shader),database=database)
                assert result,result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera=scene.create_game_object('Raster Camera')
            camera.transform.position=Vector3(0,0,-5)
            camera.add_component(inx.Camera)
            objects,materials,renderers=[],[],[]
            for name,z,color,queue in (('Near',-.5,(1,0,0,1),1000),('Far',.5,(0,1,0,1),2000),('Writer',0,(0,0,1,1),1000),('Reader',.2,(0,1,0,1),2000)):
                obj=scene.create_game_object(name)
                obj.transform.position=Vector3(0,0,z)
                renderer=obj.add_component(inx.MeshRenderer)
                renderer.set_inline_mesh_data(np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=np.float32),np.array([[0,0,-1]]*4,dtype=np.float32),np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32),np.array([0,2,1,0,3,2],dtype=np.uint32),'State Quad')
                material=inx.Material.create_unlit()
                material.frag_shader_name='Stencil Writer' if name=='Writer' else 'Stencil Reader' if name=='Reader' else 'Raster Near' if name=='Near' else 'Unlit'
                material.set_color('baseColor',*color)
                material.render_queue=queue
                material.cull_mode=2
                renderer.set_material(0,material)
                renderer.enabled=name in ('Near','Far')
                objects.append(obj);materials.append(material);renderers.append(renderer)
            objects[2].transform.position=Vector3(-.5,0,0)
            objects[2].transform.local_scale=Vector3(.5,1,1)
            native.set_game_camera_enabled(True)
            pipeline=RasterPipeline()
            pipelines.append(pipeline)
            frontend.set_render_pipeline(pipeline)
            labels=('baseline','front_culled','back_restored','depth_never','depth_write_off','far_always','far_test_off','restored','shader_reload_overrides','stencil_clear','stencil_written','no_stencil_attachment','stencil_clear_restored','stencil_not_equal_clear','stencil_not_equal_written','plain_restored')
            phase=frames=changed=0
            ticket=None
            near,far=materials[:2]
            baseline=None

            def after_draw():
                nonlocal phase,frames,changed,ticket,complete,baseline,pipeline
                try:
                    frames+=1
                    if ticket is None and frames>=changed+10:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        label=labels[phase]
                        center=pixels[54:66,72:88,:3].mean(axis=(0,1))
                        expected=(0,1,0) if phase in (1,3,4,5,6,11,13) else (0,0,0) if phase in (9,12) else (1,0,0)
                        if phase in (10,14):
                            np.testing.assert_allclose(pixels[54:66,68:76,:3],np.broadcast_to((0,1,0) if phase==10 else (0,0,1),(12,8,3)),atol=.005)
                            np.testing.assert_allclose(pixels[54:66,84:92,:3],np.broadcast_to((0,0,0) if phase==10 else (0,1,0),(12,8,3)),atol=.005)
                        else:
                            np.testing.assert_allclose(center,expected,atol=.005,err_msg=label)
                        errors=[e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors,errors[:5]
                        if phase==0:
                            baseline=pixels
                        if phase in (2,7,8,15):
                            np.testing.assert_allclose(pixels,baseline,atol=.005)
                        proof['phases'].append({'phase':label,'center_rgb':center.tolist(),'near_render_state':near.serialize_document()['renderState'],'near_override_bits':near.render_state_overrides})
                        print('PASS',label,center.tolist(),flush=True)
                        if phase==len(labels)-1:
                            complete=True;native.exit();return
                        phase+=1
                        near.cull_mode=1 if phase==1 else 2
                        near.depth_compare_op=0 if phase==3 else 1
                        near.depth_write_enable=phase!=4
                        far.depth_compare_op=7 if phase==5 else 1
                        far.depth_test_enable=phase!=6
                        if phase==8:
                            path.write_text(SURFACE.replace('Cull Back','Cull Front').replace('DepthTest Less','DepthTest Never').replace('DepthWrite On','DepthWrite Off'),encoding='utf-8')
                            result=AssetManager.reimport_asset(str(path),database=database)
                            assert result,result.error
                            error=native.reload_shader_runtime(str(path),'Raster Near')
                            assert not error,error
                        if phase==13:
                            reader_path=project/'Assets/Reader.frag'
                            reader_path.write_text(reader_path.read_text(encoding='utf-8').replace('equal,1','not_equal,1'),encoding='utf-8')
                            result=AssetManager.reimport_asset(str(reader_path),database=database)
                            assert result,result.error
                            error=native.reload_shader_runtime(str(reader_path),'Stencil Reader')
                            assert not error,error
                        if phase in (9,11,12,15):
                            pipeline=RasterPipeline(stencil=phase in (9,12))
                            pipelines.append(pipeline);frontend.set_render_pipeline(pipeline)
                        if phase>=9:
                            for i,renderer in enumerate(renderers):
                                renderer.enabled=(i==3 or i==2 and phase in (10,14)) if phase<15 else i<2
                        changed,ticket=frames,None
                    if frames>350:
                        raise AssertionError(('Raster state timeout',labels[phase]))
                except BaseException as error:
                    failures.append(error);native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None)
            native.cleanup()
    proof['passed']=True
    if args.proof:
        args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('PASS material raster state and Equal/NotEqual stencil coverage',flush=True)


if __name__=='__main__':
    main()
