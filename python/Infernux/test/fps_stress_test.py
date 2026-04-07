"""fps_stress_test.py
───────────────────
Infernux version of a simple CPU-bound FPS stress benchmark.

Attach ``FpsStressTest`` to any GameObject. The component runs a heavy
trigonometric workload every frame, uses ``Infernux.jit.njit`` when Numba
is available, and updates a ``UIText`` label with rolling FPS / frame-time
stats.

If no ``fps_text`` reference is assigned, the component will try to create
``Canvas -> FPS Stress Text`` automatically in the active scene.
"""

from __future__ import annotations

from math import cos, pi, sin

from Infernux.components import InxComponent, add_component_menu, int_field, serialized_field
from Infernux.debug import Debug
from Infernux.jit import JIT_AVAILABLE, njit, warmup
from Infernux.lib import SceneManager
from Infernux.timing import Time
from Infernux.ui import UICanvas, UIText
from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache


@njit(cache=True, fastmath=True, auto_parallel=True)
def _run_stress_kernel(iterations: int) -> float:
    count = iterations if iterations > 0 else 1
    step = pi / count
    acc = 0.0

    for index in range(count):
        angle = index * step
        acc += sin(angle) * cos(angle * 0.5)

    return acc


def _find_or_create_canvas():
    scene = SceneManager.instance().get_active_scene()
    if scene is None:
        return None, None

    for game_object in scene.get_all_objects():
        canvas = game_object.get_component(UICanvas)
        if canvas is not None:
            return scene, game_object

    canvas_go = scene.create_game_object("Canvas")
    if canvas_go is None:
        return scene, None

    canvas_go.add_component(UICanvas)
    invalidate_canvas_cache()
    return scene, canvas_go


@add_component_menu("Benchmark/FPS Stress Test")
class FpsStressTest(InxComponent):
    iterations: int = int_field(
        200000,
        range=(1, 100000000),
        slider=False,
        tooltip="Loop count executed every frame by the benchmark kernel.",
    )
    update_interval: float = serialized_field(
        default=0.5,
        range=(0.05, 5.0),
        slider=False,
        drag_speed=0.05,
        tooltip="Seconds between FPS label refreshes.",
    )
    auto_create_text: bool = serialized_field(
        default=True,
        tooltip="Create a Canvas and UIText automatically when fps_text is not assigned.",
    )
    fps_text: UIText = None

    def awake(self):
        self._kernel = _run_stress_kernel
        self._jit_mode = "ON" if JIT_AVAILABLE else "OFF"
        self._iterations = 1
        self._update_interval_seconds = 0.5
        self._result = 0.0
        self._accum_frames = 0
        self._accum_total_dt = 0.0
        self._next_update_time = 0.0

    def start(self):
        self._ensure_text_target()
        self._refresh_config()

        warmup(self._kernel, self._iterations)
        self._result = self._kernel(self._iterations)

        self._accum_frames = 0
        self._accum_total_dt = 0.0
        self._next_update_time = Time.realtime_since_startup + self._update_interval_seconds

        if self.fps_text is None:
            Debug.log_warning("FpsStressTest: fps_text is not assigned and could not be auto-created.")
        else:
            self._write_stats(force=True)

    def update(self, delta_time: float):
        self._refresh_config()
        self._result = self._kernel(self._iterations)

        frame_dt = max(float(Time.unscaled_delta_time or delta_time), 0.0)
        self._accum_total_dt += frame_dt
        self._accum_frames += 1

        if Time.realtime_since_startup < self._next_update_time:
            return

        self._write_stats(force=False)
        self._accum_frames = 0
        self._accum_total_dt = 0.0
        self._next_update_time = Time.realtime_since_startup + self._update_interval_seconds

    def on_validate(self):
        self._refresh_config()
        self.iterations = self._iterations
        self.update_interval = self._update_interval_seconds

    def _refresh_config(self) -> None:
        self._iterations = max(1, int(getattr(self, "iterations", 200000)))
        self._update_interval_seconds = max(0.05, float(getattr(self, "update_interval", 0.5)))

    def _ensure_text_target(self) -> None:
        if self.fps_text is not None:
            return
        if not bool(getattr(self, "auto_create_text", True)):
            return

        scene, canvas_go = _find_or_create_canvas()
        if scene is None or canvas_go is None:
            return

        text_go = scene.create_game_object("FPS Stress Text")
        if text_go is None:
            return

        text_go.set_parent(canvas_go, world_position_stays=False)
        text = text_go.add_component(UIText)
        if text is None:
            return

        text.x = 32.0
        text.y = 32.0
        text.width = 420.0
        text.height = 140.0
        text.font_size = 28.0
        text.text = "FPS: --\nFrame Time: --\nIterations: --"
        text.color = [1.0, 1.0, 1.0, 1.0]

        self.fps_text = text
        invalidate_canvas_cache()

    def _write_stats(self, *, force: bool) -> None:
        if self.fps_text is None:
            return

        if force or self._accum_frames <= 0 or self._accum_total_dt <= 1e-8:
            fps = 0.0
            ms = 0.0
        else:
            fps = self._accum_frames / self._accum_total_dt
            ms = (self._accum_total_dt / self._accum_frames) * 1000.0

        self.fps_text.text = (
            f"FPS: {fps:.1f}\n"
            f"Frame Time: {ms:.2f} ms\n"
            f"Iterations: {self._iterations:,}\n"
            f"JIT: {self._jit_mode}  Mode: {getattr(self._kernel, 'selected_mode', 'n/a')}\n"
            f"Result: {self._result:.3f}"
        )