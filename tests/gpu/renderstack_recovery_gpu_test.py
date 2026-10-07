"""Independent GPU evidence for the tutorial's exact effect failure and topology recovery."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.core.asset_ref import RenderEffectRef
from infernux.lib import RenderPipelineCallback,SceneManager
from infernux.renderstack import EffectSlot,RenderEffect,RenderPipeline,RenderStack
from infernux.renderstack.render_effect_asset import RenderEffectAsset

class StackHost(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack=RenderStack()

    def render(self,context,camera):
        self.stack.render(context,camera)


def display(value):
    return np.where(value<=.0031308,value*12.92,1.055*value**(1/2.4)-.055)


COLORS=(np.array([.3,.12,.08]),np.array([.05,.3,.15]))


@inx.renderstack.render_effect_feature('audit.recovery.missing_probe')
class MissingProbe(inx.renderstack.FullScreenEffect):
    name='Missing Probe'
    injection_point='before_post_process'
    requires={'color','missing_probe'}
    def setup_passes(self,graph,bus):
        if bus.get('missing_probe') is None:
            raise ValueError('missing effect-stage resource: missing_probe')


@inx.renderstack.render_effect_feature('audit.recovery.allocate_then_fail')
class AllocateThenFail(inx.renderstack.FullScreenEffect):
    name='Allocation Failure'
    injection_point='before_post_process'
    requires={'color'}
    modifies={'color'}
    def setup_passes(self,graph,bus):
        target=graph.create_texture('orphan_texture')
        graph.create_buffer('orphan_buffer',byte_size=16,storage=True)
        graph.add_pass('OrphanPass').write_color(target).fullscreen_quad('Recovery B')
        bus.set('color',target)
        raise ValueError('deliberate effect-stage failure')


class RecoveryPipeline(RenderPipeline):
    name='Recovery Proof Pipeline'
    duplicate=False
    version=0
    builds=0
    def define_topology(self,graph):
        self.builds+=1
        color=graph.create_texture('color',camera_target=True)
        graph.add_pass('Background').write_color(color).fullscreen_quad('Recovery '+('A' if self.version==0 else 'B'))
        initial=graph.publish_pass_result('initial',{'color':color})
        with graph.pass_result(initial):
            graph.effects('route_probe',scope='composite',inputs={'color'},outputs={'color'})
            graph.effects('scene_probe',scope='composite',inputs={'color'},outputs={'color'})
            if self.duplicate:
                graph.effects('scene_probe',scope='composite',inputs={'color'},outputs={'color'})
        graph.set_output(color)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    proof={'passed':False,'package':inx.__file__,'phases':[],'scope':'Independent installed GPU: exact missing_probe guard, failed effect pass/texture/buffer/bus rollback, last graph retention under duplicate stage, no per-frame retries, repaired topology with a deliberately different constant shader output. No Editor pixels.'}
    with tempfile.TemporaryDirectory(prefix='infernux-recovery-proof-') as folder:
        project=Path(folder)
        for name in ('Assets','Packages','ProjectSettings'):
            (project/name).mkdir()
        frontend=inx.Engine();native=frontend.get_native_engine();host=StackHost()
        failures=[];complete=False
        try:
            frontend.init_renderer(96,64,folder);frontend.resize_game_render_target(96,64)
            native.set_scene_view_visible(False);native.set_editor_fps_cap(240.);native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            for name,color in zip(('A','B'),COLORS):
                path=project/'Assets'/('Recovery'+name+'.frag')
                path.write_text('#version 450\nShaderInfo { Name "Recovery '+name+'" Hidden On Capabilities [Fullscreen] Inputs { Float2 inUV } Outputs { Float4 outColor } }\nvoid main() { outColor=vec4('+','.join(map(str,color))+',1); }\n',encoding='utf-8')
                imported=AssetManager.import_asset(str(path),database=database)
                assert imported,imported.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():scene.destroy_game_object(obj)
            scene.create_game_object('Recovery Camera').add_component(inx.Camera)
            pipeline=RecoveryPipeline();host.stack._pipeline=pipeline;frontend.set_render_pipeline(host);native.set_game_camera_enabled(True)
            effects={key:RenderEffect(RenderEffectAsset('audit.recovery.'+key)) for key in ('missing_probe','allocate_then_fail')}
            phases=('baseline','missing_probe_guard','allocation_failure_rolled_back','duplicate_topology_retains_accepted_output','repaired_topology_changes_output','clean_restored')
            phase=frames=changed=0;ticket=None;last_builds=None
            def after_draw():
                nonlocal phase,frames,changed,ticket,complete,last_builds
                try:
                    frames+=1
                    if ticket is None and frames>=changed+12:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        expected=display(COLORS[1 if phase>=4 else 0])
                        np.testing.assert_allclose(pixels[...,:3],np.broadcast_to(expected,pixels[...,:3].shape),atol=.005,err_msg=phases[phase])
                        state=host.stack._graph_state
                        errors=host.stack.effect_compile_errors
                        if phase==1:assert len(errors)==1 and 'route_probe/recovery-slot' in errors[0] and 'missing_probe' in errors[0],errors
                        if phase in (2,4):assert any('deliberate effect-stage failure' in e for e in errors),errors
                        if phase in (0,5):assert not errors,errors
                        assert state.build_failed==(phase==3),(phases[phase],errors)
                        description=state.description
                        assert not any('OrphanPass' in p.name for p in description.passes)
                        assert not any('orphan_texture' in t.name for t in description.textures)
                        assert not any('orphan_buffer' in b.name for b in description.buffers)
                        if phase==3:
                            assert 'scene_probe' in str(errors) and 'unique' in str(errors).lower(),errors
                            assert pipeline.builds==last_builds,'Rejected topology retried during steady frames'
                        proof['phases'].append({'phase':phases[phase],'rgb':pixels[32,48,:3].tolist(),'revision':description.source_revision,'errors':list(errors),'builds':pipeline.builds})
                        print('PASS',phases[phase],flush=True)
                        if phase==5:complete=True;native.exit();return
                        phase+=1
                        if phase in (1,2):
                            key='missing_probe' if phase==1 else 'allocate_then_fail'
                            host.stack.effect_slots=[EffectSlot(stage_id='route_probe',slot_id='recovery-slot',effect=RenderEffectRef(effect=effects[key]))]
                        elif phase==3:
                            pipeline.duplicate=True;pipeline.version=1
                        elif phase==4:pipeline.duplicate=False
                        elif phase==5:host.stack.effect_slots=[]
                        host.stack.invalidate_graph()
                        changed=frames;ticket=None;last_builds=None
                    if phase==3 and frames==changed+6:last_builds=pipeline.builds
                    if frames>220:raise AssertionError('Recovery proof timed out')
                except BaseException as error:
                    failures.append(error);native.exit()
            native.set_post_draw_callback(after_draw);native.run()
            if failures:raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None);host.stack.on_destroy();native.cleanup()
    proof['passed']=True
    if args.proof:
        args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('PASS effect rollback and topology recovery',flush=True)


if __name__=='__main__':main()
