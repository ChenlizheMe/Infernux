import json,re,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.core.asset_types import TextureImportSettings,TextureType,TextureCompression,read_mesh_import_settings
from infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3
ROOT=Path(__file__).resolve().parents[3]

text=(ROOT/'docs/learn/fragment-materials.md').read_text(encoding='utf-8')
functions=[block for block in re.findall(r'```glsl\n(.*?)```',text,re.S) if block.startswith('void surface(out SurfaceData s)')]
assert len(functions)==2 and functions[0]==functions[1],functions
exact=functions[0]
header=(Path(inx.__file__).parent/'resources/shaders/lit.frag').read_text(encoding='utf-8').split('void surface(')[0]
header=header.replace('Name "Lit"','Name "Exact Tutorial Lit Surface"')
assert 'Name "Exact Tutorial Lit Surface"' in header
CHECK='''#version 450
ShaderInfo { Name "Tutorial Lit Contract Consumer" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D colorTex Texture2D normalTex Texture2D baseTex }
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() {
 ivec2 p=ivec2(gl_FragCoord.xy);
 vec4 color=texelFetch(colorTex,p,0), normal=texelFetch(normalTex,p,0), base=texelFetch(baseTex,p,0);
 bool a=color.a>0.9,b=normal.a>0.9,c=base.a>0.9;
 if(a!=b || a!=c || (a && any(greaterThan(abs(base.rgb-vec3(1)),vec3(0.002))))) outColor=vec4(1,0,1,1);
 else outColor=a ? vec4(normal.rgb,1) : vec4(0,0,0,0);
}
'''
class ContractPipeline(inx.renderstack.RenderPipeline):
    name='Exact Lit Surface Contract'
    def define_topology(self,g):
        self.builds+=1
        g.set_msaa_samples(1)
        f=inx.rendergraph.Format
        depth=g.create_texture('depth',format=f.D32_SFLOAT,samples=1)
        color=g.create_texture('surface',format=f.RGBA16_SFLOAT,samples=1)
        normal=g.create_texture('normal',format=f.RGBA16_SFLOAT,samples=1)
        base=g.create_texture('base',format=f.RGBA16_SFLOAT,samples=1)
        g.add_pass('LitSurface').write_color(color).write_depth(depth).set_clear(color=(0,0,0,0),depth=1).draw_renderers(queue_range=(0,2500))
        for name,texture,material_pass in [('Normal',normal,'normal'),('Base',base,'base_color')]:
            g.add_pass(name).read(depth).write_color(texture).set_clear(color=(0,0,0,0)).draw_renderers(queue_range=(0,2500),material_pass=material_pass)
        target=g.create_texture('color',camera_target=True)
        p=g.add_pass('CompareLitContract').write_color(target)
        for key,texture in [('colorTex',color),('normalTex',normal),('baseTex',base)]:p.set_texture(key,texture)
        p.fullscreen_quad('Tutorial Lit Contract Consumer')
        g.screen_ui_overlay_section(resources={'color'})
        g.set_output(target)

with tempfile.TemporaryDirectory(prefix='infernux-exact-lit-surface-') as folder:
    project=Path(folder)
    for directory in ('Assets','Packages','ProjectSettings'):(project/directory).mkdir()
    mesh=project/'Assets/Probe.obj'
    mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n',encoding='ascii')
    mirrored=project/'Assets/Mirrored.obj'
    mirrored.write_text(mesh.read_text(encoding='ascii').replace('vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1','vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0'),encoding='ascii')
    normal_path=project/'Assets/TiltedNormal.png'
    Image.new('RGBA',(8,8),(128,218,218,255)).save(normal_path)
    fragment=project/'Assets/LitContract.frag'
    fragment.write_text(header+exact,encoding='utf-8')
    consumer=project/'Assets/Check.frag'
    consumer.write_text(CHECK,encoding='ascii')
    frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();pipeline=ContractPipeline();pipeline.builds=0
    proof={'scope':'Independent installed GPU exact bilingual Lit function: flat and non-unit world normal, tangent normal map, mirrored UV handedness, rotated world TBN and exact restore. Measured normal buffer values compared with analytical world normals. No Editor image access.','phases':[]}
    failures=[];complete=False
    try:
        frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
        native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
        database=frontend.get_asset_database()
        for path in (mesh,mirrored,fragment,consumer,normal_path):
            imported=AssetManager.import_asset(str(path),database=database);assert imported,imported.error
            if path==mesh:mesh_guid=imported.guid
            if path==mirrored:mirrored_guid=imported.guid
            if path==normal_path:normal_guid=imported.guid
        settings=TextureImportSettings(texture_type=TextureType.NORMAL_MAP,compression=TextureCompression.NONE,generate_mipmaps=False)
        assert AssetManager.apply_import_settings('texture',str(normal_path),settings)
        assert read_mesh_import_settings(str(mesh)).flip_uvs
        assert read_mesh_import_settings(str(mirrored)).flip_uvs
        proof['import_flip_uvs']=True
        scene=SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():scene.destroy_game_object(obj)
        camera=scene.create_game_object('Lit Contract Camera');camera.transform.position=Vector3(0,0,-5);camera.add_component('Camera')
        obj=scene.create_game_object('Lit Contract Quad');renderer=obj.add_component('MeshRenderer');renderer.set_mesh_asset_guid(mesh_guid)
        material=InxMaterial.create_default_lit();material.vert_shader_name='Standard';material.frag_shader_name='Exact Tutorial Lit Surface';renderer.set_material(0,material)
        frontend.set_render_pipeline(pipeline);native.set_game_camera_enabled(True)
        frame=0;changed=0;phase=0;ticket=None;initial=None
        def after_draw():
            global frame,changed,phase,ticket,initial,complete
            try:
                frame+=1
                if ticket is None and frame>=changed+8:ticket=frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels=ticket.result_numpy().copy().astype(np.float32);rgb=pixels[...,:3]
                    covered=pixels[...,3]>.9;bad=(rgb[...,0]>.9)&(rgb[...,2]>.9)
                    record={'phase':('exact_lit','nonunit_authored_normal','tangent_normal_map','mirrored_uv','rotated_world_tbn','exact_restored')[phase],'covered':int(covered.sum()),'mismatch':int(bad.sum()),'builds':pipeline.builds}
                    assert covered.sum()>100 and not bad.any() and pipeline.builds==1,record
                    x=128/255*2-1;y=218/255*2-1;z=float(np.sqrt(1-x*x-y*y))
                    if phase in (0,1,5):expected=np.array([.5,.5,0])
                    # The importer defaults to flip_uvs=True. Thus the
                    # ordinary OBJ's bitangent is -Y, while the authored
                    # mirrored OBJ produces +Y after the same import.
                    elif phase==2:expected=np.array([x,-y,-z])*.5+.5
                    elif phase==3:expected=np.array([x,y,-z])*.5+.5
                    else:expected=np.array([(x-y)/np.sqrt(2),(x+y)/np.sqrt(2),-z])*.5+.5
                    measured=rgb[covered]
                    # The normal buffer is diagnostic data, but this consumer
                    # presents it as camera color. Readback includes the normal
                    # linear-to-sRGB display conversion.
                    encoded=np.where(expected<=.0031308,expected*12.92,1.055*expected**(1/2.4)-.055)
                    np.testing.assert_allclose(measured,np.broadcast_to(encoded,measured.shape),atol=.003)
                    record.update(measured_display=measured.mean(axis=0).tolist(),expected_normal=expected.tolist(),expected_display=encoded.tolist())
                    assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                    if phase==0:initial=pixels
                    elif phase==5:np.testing.assert_array_equal(pixels,initial)
                    proof['phases'].append(record);print(record,flush=True)
                    if phase==5:complete=True;proof['passed']=True;native.exit();return
                    phase+=1
                    body=exact.replace('s.normalWS = sampleNormal(normalMap, material.normalScale);','s.normalWS = vec3(0,0,-4);') if phase==1 else exact
                    fragment.write_text(header+body,encoding='utf-8')
                    imported=AssetManager.reimport_asset(str(fragment),database=database);assert imported,imported.error
                    if phase==2:
                        material.set_texture_guid('normalMap',normal_guid)
                    elif phase==3:
                        renderer.set_mesh_asset_guid(mirrored_guid)
                    elif phase==4:
                        obj.transform.euler_angles=Vector3(0,0,45)
                    elif phase==5:
                        renderer.set_mesh_asset_guid(mesh_guid)
                        obj.transform.euler_angles=Vector3(0,0,0)
                        material.set_texture_guid('normalMap','')
                    changed=frame;ticket=None
                if frame>160:raise AssertionError('Lit normal mapping GPU audit did not finish')
            except BaseException as exc:failures.append(exc);native.exit()
        native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
        if failures:raise failures[0]
        assert complete
    finally:
        frontend.set_render_pipeline(None);pipeline.dispose();native.cleanup()
    print(json.dumps(proof,indent=2),flush=True)
    print('PASS exact Lit function, tangent/handed/world normal mapping, normalization and restore',flush=True)
