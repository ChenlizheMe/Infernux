"""GPU model/bounds publication across geometry passes, COW and view history."""

from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.lib import ConsolePanel, PrimitiveType, SceneManager, Vector3
from slang_surface_buffers_gpu_test import BuffersPipeline, CONSUMER, SURFACE


@inx.compute.kernel
def write_poses(domain, models, bounds, x, bounds_x):
    i = inx.compute.index(domain)
    position = x if i == 2 else 100.0
    box = bounds_x if i == 2 else 100.0
    for column in range(4):
        for row in range(4):
            models[i, column * 4 + row] = 1.0 if row == column else 0.0
    models[i, 12] = position
    bounds[i, 0] = box - 1.0
    bounds[i, 1] = -1.0
    bounds[i, 2] = -0.01
    bounds[i, 3] = 0.0
    bounds[i, 4] = box + 1.0
    bounds[i, 5] = 1.0
    bounds[i, 6] = 0.01
    bounds[i, 7] = 0.0


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-compute-pose-") as directory:
        root = Path(directory)
        for name in ("Assets", "ProjectSettings"):
            (root / name).mkdir()
        source = root / "Assets/cutout.frag.py"
        consumer = root / "Assets/compare.frag.py"
        source.write_text(SURFACE, encoding="utf8")
        # With 60-degree vertical FOV, a 4:3 view and z=5, dx=0.01
        # produces dx/(2*z*tan(FOV/2)*aspect) = 0.001299038 UV motion.
        consumer_source = CONSUMER.replace('        if not covered:\n',
            '        if covered and shader.length(motion.xy) > 0.00001:\n'
            '            if shader.abs(motion.x - 0.001299038) > 0.00002 or shader.abs(motion.y) > 0.00002:\n'
            '                return shader.vec4(1.0, 0.5, 1.0, 1.0)\n'
            '        if not covered:\n')
        consumer.write_text(consumer_source, encoding="utf8")
        frontend, console, pipeline = inx.Engine(), ConsolePanel(), BuffersPipeline()
        native = frontend.get_native_engine()
        pipeline.builds, pipeline.consumer = 0, str(consumer)
        failures, complete, buffers = [], False, []
        try:
            frontend.init_renderer(160, 120, directory)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera = scene.create_game_object("相机")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera).field_of_view = 60
            owner = scene.create_primitive(PrimitiveType.Quad, "设备姿态")
            owner.transform.position = Vector3(100, 100, 100)
            renderer = owner.get_component(inx.MeshRenderer)
            material = inx.Material.from_shader(str(source))
            assert material.vert_shader_name == "Standard"
            material.cull_mode = 0  # The shared built-in quad faces +Z; view it from -Z.
            material.alpha_clip_enabled, material.alpha_clip_threshold = True, 0.5
            renderer.set_material(material)
            cpu_owner = scene.create_primitive(PrimitiveType.Quad, "同材质CPU姿态")
            cpu_owner.transform.position = Vector3(-2.5, 0, 0)
            cpu_owner.get_component(inx.MeshRenderer).set_material(material)
            # Shared material, vertex/index buffers and submesh must still
            # split on pose source, without leaking one source's descriptor.
            consumer_source = consumer_source.replace('picking.x == self.object_id and picking.y == 0',
                f'(picking.x == self.object_id or picking.x == {cpu_owner.id}) and picking.y == 0')
            consumer.write_text(consumer_source, encoding="utf8")
            domain = inx.buffer(shape=3, dtype="int32", device="gpu")
            models = inx.buffer(shape=(3, 16), dtype="float32", device="gpu", data=np.zeros((3, 16), np.float32))
            bounds = inx.buffer(shape=(3, 8), dtype="float32", device="gpu", data=np.zeros((3, 8), np.float32))
            buffers.extend((domain, models, bounds))
            inx.compute.launch(write_poses, params=(domain, models, bounds, 0.0, 0.0))
            poses = inx.compute.PoseBuffer(models, bounds)
            model_input, bounds_input = models, bounds
            renderer.set_pose_buffer(poses, 2)
            readback_bytes = inx.compute.statistics().readback_bytes
            pipeline.object_id = owner.id
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            frame = changed = phase = 0
            ticket = baseline = baseline_center = None
            labels = ("gpu_baseline", "gpu_offset", "moving", "stopped", "gpu_bounds_cull", "bounds_restore",
                      "unpublished_write", "published_write", "cpu_restore")

            def update(x, box=None, *, publish=True):
                inx.compute.launch(write_poses, params=(domain, models, bounds, x, x if box is None else box))
                if publish:
                    poses.publish(model_input, bounds_input)

            def after_draw():
                nonlocal frame, changed, phase, ticket, baseline, baseline_center, complete
                nonlocal poses, model_input, bounds_input
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        red = (pixels[..., 0] > 0.9) & (pixels[..., 1] < 0.05) & (pixels[..., 2] < 0.05)
                        green = (pixels[..., 1] > 0.9) & (pixels[..., 0] < 0.05) & (pixels[..., 2] < 0.05)
                        covered = red | green
                        device_covered = covered & (np.indices(covered.shape)[1] >= 60)
                        bad = (pixels[..., 0] > 0.9) & (pixels[..., 2] > 0.9)
                        assert not bad.any(), (labels[phase], np.unique(pixels[bad, 1]))
                        assert inx.compute.statistics().readback_bytes == readback_bytes
                        assert not [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                        assert pipeline.builds == 1
                        center = float(np.nonzero(device_covered)[1].mean()) if device_covered.any() else 0.0
                        assert covered[:, :60].sum() > 100, "The shared CPU instance lost its own pose"
                        if baseline is not None:
                            np.testing.assert_array_equal(pixels[:, :60], baseline[:, :60])
                        if phase == 4:
                            assert not device_covered.any(), "Device bounds must cull a visible model on every pass"
                        else:
                            assert device_covered.sum() > 100, (labels[phase], device_covered.sum())
                        if phase == 0:
                            baseline, baseline_center = pixels, center
                        elif phase in (3, 5, 6, 8):
                            np.testing.assert_array_equal(pixels, baseline)
                        elif phase in (1, 7):
                            assert center > baseline_center + 6
                        if phase == 2:
                            assert green.sum() > device_covered.sum() * 0.9
                        else:
                            assert not green.any(), (labels[phase], green.sum())
                        print("POSE", labels[phase], "covered", int(covered.sum()), "moving", int(green.sum()), "center", center, flush=True)
                        phase += 1
                        if phase == len(labels):
                            complete = True
                            native.exit()
                            return
                        if phase == 1:
                            update(0.8)
                        elif phase in (3, 5):
                            update(0.0)
                        elif phase == 4:
                            update(0.0, 100.0)
                        elif phase == 6:
                            model_input = models.view(offset=32, shape=(1, 16))
                            bounds_input = bounds.view(offset=16, shape=(1, 8))
                            buffers.extend((model_input, bounds_input))
                            poses = inx.compute.PoseBuffer(model_input, bounds_input)
                            renderer.set_pose_buffer(poses)
                            update(0.8, publish=False)
                        elif phase == 7:
                            poses.publish(model_input, bounds_input)
                        elif phase == 8:
                            renderer.clear_pose_buffer()
                            owner.transform.position = Vector3(0, 0, 0)
                        changed, ticket = frame, None
                    if phase == 2:
                        # Two complete producer generations between rendering
                        # samples must not replace the view's previous value.
                        update(0.01 * (frame - changed) + 0.03)
                        update(0.01 * (frame - changed))
                    if frame > 500:
                        raise AssertionError("Device pose rendering timed out")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
            print("INFERNUX_COMPUTE_POSE_OK", flush=True)
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            for value in buffers:
                value.close()
            inx.compute._release_engine_resources()
            native.cleanup()


if __name__ == "__main__":
    main()
