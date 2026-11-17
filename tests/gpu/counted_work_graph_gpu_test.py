"""A device-produced prefix feeds dense storage and an authored fullscreen pass."""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from infernux import Engine, buffer, compute
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager
from infernux.renderstack import RenderPipeline


CAPACITY = 67
SENTINEL = 191
REQUESTS = (67, 1, 0, 70)
HEIGHT = 32

SHADER = '''import infernux.shader as shader

class CountedColor(shader.fullscreen):
    shader_id = "tests/counted_work"
    hidden = True
    values = shader.input("Values", shader.Buffer[shader.uint])

    def main(self, uv: shader.vec2) -> shader.vec4:
        i = int(uv.x * float(len(self.values)))
        value = self.values[i]
        return shader.vec4(float(value & 31) / 31.0, float(value >> 5) / 31.0, 0.0, 1.0)
'''


@compute.kernel
def build_count(request, count):
    i = compute.index(count)
    count[i] = request[i]


@compute.kernel
def reset_result(result):
    i = compute.index(result)
    result[i] = SENTINEL


@compute.kernel
def write_prefix(count, payload):
    i = compute.index(payload, count=count)
    payload[i] = 3 * i + 7


@compute.kernel
def read_prefix(count, payload, result):
    i = compute.index(payload, count=count)
    result[i] = 2 * payload[i]


class CountedWorkPipeline(RenderPipeline):
    name = "Device-counted work"

    def __init__(self, work, request):
        super().__init__()
        self.work, self.request = work, request

    def define_topology(self, graph):
        source = graph.import_buffer("request", self.request)
        binding = graph.bind_work("counted", self.work, inputs={"request": source})
        graph.add_compute_pass("CountedRows").compute_work(binding)
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("CountedColor").set_buffer("values", binding.output("result")).write_color(color).fullscreen_quad(
            "tests/counted_work",
        )
        graph.set_output(color)


def expect_rejected(operations, storage, inputs, outputs, message):
    try:
        work = compute.prepare_work(operations, private_buffers=storage, outputs=outputs, inputs=inputs)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        work.close()
        raise AssertionError(f"Expected an invalid recipe to be rejected with {message!r}")


def prepare(request, count, payload, result):
    make_count = (build_count, (request, count))
    reset = (reset_result, (result,))
    write = (write_prefix, (count, payload))
    read = (read_prefix, (count, payload, result.view(count=CAPACITY)))
    storage = request, count, payload, result
    inputs = {"request": request}
    dense = {"result": result}
    expect_rejected((reset, write, read), storage[1:], {}, dense, "without a prior complete writer")
    expect_rejected((make_count, reset, write, make_count, read), storage, inputs, dense,
                    "matching current count and payload writer")
    expect_rejected((make_count, reset, write, (reset_result, (payload,)), read), storage, inputs, dense,
                    "matching current count and payload writer")
    expect_rejected((make_count, reset, write, read), storage, inputs, {"result": payload},
                    "complete writer for its entire port")
    return compute.prepare_work((make_count, reset, write, read), private_buffers=storage, inputs=inputs,
                                outputs=dense)


def expected_image(request):
    expected = np.full(CAPACITY + 1, SENTINEL, dtype=np.uint32)
    active = min(CAPACITY, request)
    expected[:active] = 6 * np.arange(active, dtype=np.uint32) + 14
    linear = np.stack((expected % 32, expected // 32, np.zeros_like(expected)), axis=-1) / 31.0
    return np.broadcast_to(linear, (HEIGHT, CAPACITY + 1, 3))


def main():
    with TemporaryDirectory(prefix="infernux-counted-work-") as directory:
        project = Path(directory)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (project / name).mkdir()
        shader = project / "Assets/counted_color.frag.py"
        shader.write_text(SHADER, encoding="ascii")
        frontend = Engine()
        native = frontend.get_native_engine()
        work = None
        try:
            frontend.init_renderer(CAPACITY + 1, HEIGHT, directory)
            frontend.resize_game_render_target(CAPACITY + 1, HEIGHT)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            native.set_game_camera_enabled(True)
            imported = AssetManager.import_asset(str(shader), database=frontend.get_asset_database())
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            for node in scene.get_root_objects():
                scene.destroy_game_object(node)
            scene.create_game_object("Counted work camera").add_component("Camera")

            request = buffer(shape=1, dtype=np.uint32, device="gpu", data=[REQUESTS[0]])
            count = buffer(shape=3, dtype=np.uint32, device="gpu").view(offset=1, count=1)
            payload = buffer(shape=CAPACITY + 7, dtype=np.uint32, device="gpu").view(offset=4, count=CAPACITY)
            result = buffer(shape=CAPACITY + 1, dtype=np.uint32, device="gpu")
            work = prepare(request, count, payload, result)
            frontend.set_render_pipeline(CountedWorkPipeline(work, request))
            console = ConsolePanel()
            phase, frame, request_frame, ticket = 0, 0, 0, None
            failure = None
            deadline = monotonic() + 20

            def after_draw():
                nonlocal phase, frame, request_frame, ticket, failure
                try:
                    frame += 1
                    errors = [entry for entry in console._get_visible_log_snapshot(1000)
                              if entry["level"] in ("ERROR", "FATAL")]
                    assert not errors, errors
                    if ticket is None and native.game_presentation_ready and frame >= request_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy()[..., :3]
                        np.testing.assert_allclose(pixels, expected_image(REQUESTS[phase]), atol=.0005, rtol=0)
                        print(f"Request {REQUESTS[phase]} matched all {CAPACITY + 1} work rows in the render target", flush=True)
                        phase += 1
                        if phase == len(REQUESTS):
                            native.exit()
                            return
                        request.set_data(np.array([REQUESTS[phase]], dtype=np.uint32))
                        ticket, request_frame = None, frame
                    if monotonic() >= deadline:
                        raise AssertionError(f"Counted work graph timed out in phase {phase}")
                except BaseException as error:
                    failure = error
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failure is not None:
                raise failure
            assert phase == len(REQUESTS)
        finally:
            frontend.set_render_pipeline(None)
            if work is not None:
                work.close()
            compute._release_engine_resources()
            native.cleanup()


if __name__ == "__main__":
    main()
