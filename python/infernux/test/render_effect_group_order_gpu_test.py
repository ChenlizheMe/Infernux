"""Independent GPU proof of stage/slot/nested-group/feature-pass ordering."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.core.asset_ref import RenderEffectRef
from infernux.lib import ConsolePanel, RenderPipelineCallback, SceneManager
from infernux.renderstack import EffectSlot, RenderStack
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_compiler import publish_live_effect_group_document

class StackHost(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack = RenderStack()

    def render(self, context, camera):
        self.stack.render(context, camera)


def display(value):
    return np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)


BASE=np.array([.08,.12,.2])

SHADERS={
 'Baseline': '''#version 450
ShaderInfo { Name "Group Order Baseline" Hidden On Capabilities [Fullscreen]
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() { outColor=vec4(0.08,0.12,0.2,1); }
''',
 'Scale': '''#version 450
ShaderInfo { Name "Group Order Scale" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D _SourceTex } Inputs { Float2 inUV } Outputs { Float4 outColor }
 PushConstants pc { Float multiplier } }
void main() { vec4 c=texture(_SourceTex,inUV); outColor=vec4(c.rgb*pc.multiplier,c.a); }
''',
 'Offset': '''#version 450
ShaderInfo { Name "Group Order Offset" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D _SourceTex } Inputs { Float2 inUV } Outputs { Float4 outColor }
 PushConstants pc { Float bias } }
void main() { vec4 c=texture(_SourceTex,inUV); outColor=vec4(c.rgb+pc.bias,c.a); }
''',
}


@inx.renderstack.render_effect_feature('audit.group_order.affine')
class AffineEffect(inx.renderstack.FullScreenEffect):
    name='Group Order Affine'
    injection_point='before_post_process'
    default_order=1000
    modifies={'color'}
    multiplier: float=inx.serialized_field(default=1.)
    bias: float=inx.serialized_field(default=0.)

    def setup_passes(self,graph,bus):
        self.apply_single_source_effect(graph,bus,output_name='scaled',pass_name='Scale',shader_name='Group Order Scale',format=inx.rendergraph.Format.RGBA16_SFLOAT,params={'multiplier':float(self.multiplier)})
        self.apply_single_source_effect(graph,bus,output_name='offset',pass_name='Offset',shader_name='Group Order Offset',format=inx.rendergraph.Format.RGBA16_SFLOAT,params={'bias':float(self.bias)})


class OrderingPipeline(inx.renderstack.RenderPipeline):
    name='Group Order Pipeline'
    builds=0

    def define_topology(self,graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color=graph.create_texture('color',camera_target=True)
        graph.add_pass('Baseline').write_color(color).fullscreen_quad('Group Order Baseline')
        result=graph.publish_pass_result('initial',{'color':color})
        with graph.pass_result(result):
            graph.effects('first',scope='composite',inputs={'color'},outputs={'color'})
            graph.effects('second',scope='composite',inputs={'color'},outputs={'color'})
        graph.set_output(color)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    proof={'passed':False,'package':inx.__file__,'phases':[],'scope':'Independent temporary renderer project. Noncommutative scale/offset operations verify actual stage, slot, nested group and feature-pass order, shared-source versus group overrides, enabled state and parameter-only publication. No live Editor pixel access.'}
    with tempfile.TemporaryDirectory(prefix='infernux-effect-group-order-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):
            (project/name).mkdir()
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        host=StackHost()
        console=ConsolePanel()
        failures,complete=[],False
        try:
            frontend.init_renderer(96,64,folder)
            frontend.resize_game_render_target(96,64)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            dependencies=[]
            for name,source in SHADERS.items():
                path=project/'Assets'/(name+'.frag');path.write_text(source,encoding='utf-8')
                result=AssetManager.import_asset(str(path),database=database)
                assert result,result.error
                if name!='Baseline':
                    dependencies.append({'guid':result.guid})
            effects={}
            for name,multiplier,bias in (('Scale',2.,0.),('Offset',1.,.05),('Affine',2.,.05)):
                path=project/'Assets'/(name+'.effect')
                path.write_text(json.dumps({'$schema':'infernux.render_effect','feature_type':'audit.group_order.affine','parameters':{'multiplier':multiplier,'bias':bias},'dependencies':dependencies}),encoding='utf-8')
                result=AssetManager.import_asset(str(path),database=database)
                assert result,result.error
                effects[name]=result.guid
            def entry(name,guid,overrides=None,enabled=True):
                return {'entry_id':name,'asset':{'guid':guid},'enabled':enabled,'overrides':overrides or {}}
            inner_path=project/'Assets/Inner.effectgroup'
            inner={'$schema':'infernux.render_effect_group','entries':[entry('scale',effects['Scale']),entry('offset',effects['Offset'])]}
            inner_path.write_text(json.dumps(inner),encoding='utf-8')
            inner_result=AssetManager.import_asset(str(inner_path),database=database)
            assert inner_result,inner_result.error
            outer_path=project/'Assets/Outer.effectgroup'
            outer={'$schema':'infernux.render_effect_group','entries':[entry('inner',inner_result.guid),entry('outer-offset',effects['Offset'],{'bias':.1})]}
            outer_path.write_text(json.dumps(outer),encoding='utf-8')
            outer_result=AssetManager.import_asset(str(outer_path),database=database)
            assert outer_result,outer_result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            scene.create_game_object('Order Camera').add_component(inx.Camera)
            host.stack._pipeline=OrderingPipeline()
            frontend.set_render_pipeline(host)
            native.set_game_camera_enabled(True)
            source=AssetManager.load_by_guid(effects['Scale'])
            assert isinstance(source,RenderEffect)
            def slots(first, second=()):
                return [
                    EffectSlot(stage_id=stage, slot_id=f'{stage}-{index}',
                               effect=RenderEffectRef(guid=guid))
                    for stage, values in (('first', first), ('second', second))
                    for index, guid in enumerate(values)
                ]

            plans = (
                ('baseline', BASE, 0),
                ('one_feature_two_passes', BASE*2+.05, 1),
                ('slot_scale_offset', BASE*2+.05, 2),
                ('slot_offset_scale', (BASE+.05)*2, 2),
                ('nested_group', BASE*2+.15, 3),
                ('nested_group_reordered', (BASE+.1)*2+.05, 3),
                ('interleaved_stages', BASE*2+.05, 2),
                ('group_override', BASE*3+.05, 2),
                ('group_override_parameter_publication', BASE*4.5+.05, 2),
                ('shared_source_changed_override_kept', BASE*4.5+.05, 2),
                ('direct_source_changed', BASE*4+.05, 2),
                ('group_entry_disabled', BASE+.05, 1),
                ('group_slot_disabled', BASE, 0),
                ('restored', BASE*2+.05, 2),
            )
            phase=frames=changed=0
            ticket=None
            images={}
            def mount(values):
                host.stack.effect_slots=values
                host.stack.invalidate_graph()
            mount([])
            def after_draw():
                nonlocal phase,frames,changed,ticket,complete
                try:
                    frames+=1
                    if ticket is None and frames>=changed+8:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        label,expected,count=plans[phase]
                        np.testing.assert_allclose(pixels[...,:3],np.broadcast_to(display(expected),pixels[...,:3].shape),atol=.005,err_msg=label)
                        np.testing.assert_allclose(pixels[...,3],1.,atol=.005)
                        assert not host.stack.effect_compile_errors,host.stack.effect_compile_errors
                        bindings=host.stack._graph_state.bindings
                        assert len(bindings)==count,(label,len(bindings),count)
                        record = {
                            'phase': label,
                            'rgb': pixels[32,48,:3].tolist(),
                            'expected_linear_rgb': expected.tolist(),
                            'builds': host.stack._pipeline.builds,
                            'bindings': [
                                {'id': b.binding_id,
                                 'multiplier': b.source.get_float('multiplier'),
                                 'bias': b.source.get_float('bias')}
                                for b in bindings
                            ],
                        }
                        if phase in (8,9):
                            assert record['builds']==proof['phases'][-1]['builds'], (label, 'parameter edit rebuilt topology')
                        proof['phases'].append(record);images[label]=pixels
                        errors=[e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors,errors[:5]
                        print('PASS',label,record['rgb'],flush=True)
                        if phase==len(plans)-1:
                            np.testing.assert_allclose(images['restored'],images['slot_scale_offset'],atol=.005)
                            assert np.abs(images['slot_scale_offset']-images['slot_offset_scale']).sum()>1.
                            complete=True;native.exit();return
                        phase+=1
                        if phase==1: mount(slots([effects['Affine']]))
                        elif phase==2: mount(slots([effects['Scale'],effects['Offset']]))
                        elif phase==3: mount(slots([effects['Offset'],effects['Scale']]))
                        elif phase==4: mount(slots([outer_result.guid]))
                        elif phase==5:
                            outer['entries'].reverse();publish_live_effect_group_document(str(outer_path),outer)
                            # This isolated callback owns an unattached stack, so it
                            # does not participate in RenderStack.instance() scene
                            # discovery. Deliver its topology invalidation explicitly.
                            host.stack.invalidate_graph()
                        elif phase==6: mount(list(reversed(slots([effects['Scale']],[effects['Offset']]))))
                        elif phase==7:
                            inner['entries'][0]['overrides']={'multiplier':3.};publish_live_effect_group_document(str(inner_path),inner);mount(slots([inner_result.guid]))
                        elif phase==8:
                            inner['entries'][0]['overrides']={'multiplier':4.5};publish_live_effect_group_document(str(inner_path),inner)
                        elif phase==9: source.set_float('multiplier',4.)
                        elif phase==10: mount(slots([effects['Scale'],effects['Offset']]))
                        elif phase==11:
                            inner['entries'][0]['enabled']=False;publish_live_effect_group_document(str(inner_path),inner);mount(slots([inner_result.guid]))
                        elif phase==12:
                            values=slots([inner_result.guid]);values[0].enabled=False;mount(values)
                        elif phase==13:
                            source.set_float('multiplier',2.);mount(slots([effects['Scale'],effects['Offset']]))
                        changed,ticket=frames,None
                    if frames>350:
                        raise AssertionError(('Group order GPU timeout',plans[phase][0]))
                except BaseException as error:
                    failures.append(error);native.exit()
            native.set_post_draw_callback(after_draw)
            native.run()
            if failures: raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None);host.stack.on_destroy();native.cleanup()
    proof['passed']=True
    if args.proof:
        args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('PASS nested effect ordering and parameter-only publication',flush=True)


if __name__=='__main__': main()
