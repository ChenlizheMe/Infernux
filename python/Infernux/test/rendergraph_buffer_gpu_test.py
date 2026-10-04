"""Render graph buffer publication, repeated uploads and deterministic teardown."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import infernux as inx
from Infernux import Engine
from Infernux.core.assets import AssetManager
from Infernux.lib import ConsolePanel, SceneManager, Vector3
from Infernux.renderstack import RenderPipeline, RenderStack
from Infernux.renderstack.render_stack_pipeline import RenderStackPipeline

SHADER = '''#version 450
ShaderInfo {
    Name "Graph Buffer Color"
    Hidden On
    Capabilities [Fullscreen]
    Resources { BufferUInt _Values }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    outColor = vec4(vec3(_Values.data[0], _Values.data[1], _Values.data[2]) / 255.0,
                    float(_Values.data[3]) / 255.0);
}
'''

class GraphBufferAudit(RenderPipeline):
    name = "Graph Buffer Audit"

    def define_topology(self, graph):
        self.build_count += 1
        source = graph.import_buffer("source_values", self.values)
        if self.copy_values:
            output = graph.create_buffer("copied_values", self.values.nbytes)
            with graph.add_copy_pass("CopyValues") as render_pass:
                render_pass.copy_buffer(source, output)
        else:
            output = source
        result = self.publish_result("values", {"values": output})
        color = graph.create_texture("color", camera_target=True)
        with graph.add_pass("BufferColor") as render_pass:
            render_pass.set_buffer("_Values", result.sample("values"))
            render_pass.write_color(color)
            render_pass.fullscreen_quad("Graph Buffer Color")
        with graph.add_present_pass("Present") as render_pass:
            render_pass.present(color)

def expected_color(values):
    linear = values[:3].astype(np.float32) / 255
    return np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', choices=('copy', 'direct'), required=True)
    parser.add_argument('--completion', choices=('pending', 'completed'), required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="infernux-graph-buffer-") as root:
        project = Path(root)
        for directory in ("Assets", "Packages", "ProjectSettings"):
            (project / directory).mkdir()
        shader = project / 'Assets/BufferColor.frag'
        shader.write_text(SHADER, encoding='ascii')
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        values = None
        stack = None
        failures = []
        completed = False
        try:
            frontend.init_renderer(64, 48, str(project))
            frontend.resize_game_render_target(64, 48)
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            imported = AssetManager.import_asset(str(shader), database=frontend.get_asset_database())
            assert imported, imported.error
            camera = scene.create_game_object('Buffer Camera')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component('Camera')
            engine.set_game_camera_enabled(True)
            initial = np.array([51,102,153,255], dtype=np.uint32)
            changed = np.array([153,51,102,255], dtype=np.uint32)
            values = inx.buffer(shape=4, dtype=np.uint32, device='gpu', data=initial)
            if args.completion == 'completed':
                np.testing.assert_array_equal(values.get_data().numpy(), initial)
            stack = RenderStack()
            scene.create_game_object('Buffer RenderStack').add_py_component(stack)
            stack.set_pipeline('Graph Buffer Audit')
            pipeline = stack.pipeline
            pipeline.values, pipeline.build_count = values, 0
            pipeline.copy_values = args.check == 'copy'
            frontend.set_render_pipeline(RenderStackPipeline())
            frame, ticket, update_frame, phase = 0, None, 0, 0
            builds = None
            def after_draw():
                nonlocal frame, ticket, update_frame, phase, completed, builds
                try:
                    frame += 1
                    if ticket is None and frame >= update_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        expected = expected_color(initial if phase % 2 == 0 else changed)
                        print('Graph Buffer',phase,'center',pixels[24,32].tolist(),'builds',pipeline.build_count,flush=True)
                        errors = [entry for entry in console._get_visible_log_snapshot(1000) if entry['level'] in ('ERROR','FATAL')]
                        assert not errors, errors
                        np.testing.assert_allclose(pixels[..., :3], np.broadcast_to(expected, pixels[..., :3].shape), atol=.003)
                        if builds is None:
                            builds = pipeline.build_count
                        else:
                            assert pipeline.build_count == builds, 'Buffer update rebuilt graph topology'
                        if phase < 5:
                            next_values = changed if phase % 2 == 0 else initial
                            values.set_data(next_values)
                            if args.completion == 'completed':
                                # A collected upload still needs GPU visibility
                                # when another queue later consumes the buffer.
                                np.testing.assert_array_equal(values.get_data().numpy(), next_values)
                            ticket, update_frame, phase = None, frame, phase + 1
                        else:
                            completed = True
                            engine.exit()
                    if frame > 120:
                        raise AssertionError('Graph buffer readback timed out')
                except BaseException as error:
                    failures.append(error)
                    engine.exit()
            engine.set_post_draw_callback(after_draw)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            if stack is not None:
                stack.on_destroy()
            frontend.set_render_pipeline(None)
            if values is not None:
                values.close()
            inx.compute._release_engine_resources()
            engine.cleanup()

if __name__=='__main__':
    main()
