"""A wait describes a duration; progress belongs to each yield activation."""
import pytest

from infernux.coroutine import (
    CoroutineScheduler, WaitForSeconds, WaitForSecondsRealtime,
    WaitForFrames, WaitForEndOfFrame,
)
from infernux import coroutine


@pytest.fixture(params=['scaled', 'realtime', 'frames', 'late'])
def wait_case(request, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(coroutine._time, 'monotonic', lambda: clock[0])
    instruction = dict(
        scaled=lambda: WaitForSeconds(1),
        realtime=lambda: WaitForSecondsRealtime(1),
        frames=lambda: WaitForFrames(4),
        late=lambda: WaitForEndOfFrame(4),
    )[request.param]()

    def tick(*schedulers):
        clock[0] += .25
        for scheduler in schedulers:
            if request.param == 'late':
                scheduler.tick_late_update(.25)
            else:
                scheduler.tick_update(.25)

    return instruction, tick, clock


def test_repeated_yield_waits_the_full_duration_every_time(wait_case):
    instruction, tick, clock = wait_case
    scheduler = CoroutineScheduler()
    completed = []

    def sequence():
        for _ in range(3):
            yield instruction
            completed.append(clock[0])

    handle = scheduler.start(sequence())
    for _ in range(12):
        tick(scheduler)
    assert completed == [1.0, 2.0, 3.0]
    assert handle.is_finished


@pytest.mark.parametrize('separate_schedulers', [False, True])
@pytest.mark.parametrize('reverse_tick_order', [False, True])
def test_shared_wait_has_independent_staggered_start_times(
    wait_case, separate_schedulers, reverse_tick_order,
):
    instruction, tick, clock = wait_case
    first = CoroutineScheduler()
    second = CoroutineScheduler() if separate_schedulers else first
    completed = {}

    def sequence(name):
        yield instruction
        completed[name] = clock[0]

    first.start(sequence('first'))
    tick(first)
    tick(first)
    second.start(sequence('second'))
    schedulers = [first, second] if separate_schedulers else [first]
    if reverse_tick_order:
        schedulers.reverse()
    for _ in range(4):
        tick(*schedulers)
    assert completed == {'first': 1.0, 'second': 1.5}


def test_stopping_one_waiter_does_not_change_another(wait_case):
    instruction, tick, clock = wait_case
    scheduler = CoroutineScheduler()
    completed = []

    def sequence():
        yield instruction
        completed.append(clock[0])

    stopped = scheduler.start(sequence())
    scheduler.start(sequence())
    tick(scheduler)
    scheduler.stop(stopped)
    for _ in range(3):
        tick(scheduler)
    assert completed == [1.0]


def test_reconfiguring_description_only_affects_future_yields(wait_case):
    instruction, tick, clock = wait_case
    scheduler = CoroutineScheduler()
    completed = {}

    def sequence(name):
        yield instruction
        completed[name] = clock[0]

    scheduler.start(sequence('original'))
    tick(scheduler)
    if isinstance(instruction, (WaitForFrames, WaitForEndOfFrame)):
        instruction.frames = 2
    else:
        instruction.duration = .5
    scheduler.start(sequence('modified'))
    for _ in range(3):
        tick(scheduler)
    assert completed == {'modified': .75, 'original': 1.0}


def test_realtime_clock_starts_at_yield_instead_of_instruction_construction(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(coroutine._time, 'monotonic', lambda: clock[0])
    instruction = WaitForSecondsRealtime(1)
    clock[0] = 100.0
    scheduler = CoroutineScheduler()
    completed = []

    def sequence():
        yield instruction
        completed.append(clock[0])

    scheduler.start(sequence())
    scheduler.tick_update(10)
    assert completed == []
    clock[0] = 101.0
    scheduler.tick_update(0)
    assert completed == [101.0]


@pytest.mark.parametrize('phase', ['update', 'late'])
def test_components_share_and_repeat_wait_in_actual_native_frames(tmp_path, phase):
    from pathlib import Path
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), phase],
        env=dict(os.environ), capture_output=True, text=True, encoding='utf-8',
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SHARED_WAIT_NATIVE_FRAMES_OK' in result.stdout


def _native_frame_probe(project, phase):
    from infernux.components import InxComponent
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    frame = 0
    completed = {'first': [], 'second': []}
    instruction = WaitForFrames(4) if phase == 'update' else WaitForEndOfFrame(4)

    class Waiter(InxComponent):
        def start(self):
            self.start_coroutine(self.sequence())

        def sequence(self):
            for _ in range(3):
                yield instruction
                completed[self.game_object.name].append(frame)

    (project / 'Assets').mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        scene = manager.create_scene('SharedWait')
        manager.set_active_scene(scene)
        for name in completed:
            scene.create_game_object(name).add_component(Waiter)
        manager.play()
        for frame in range(1, 13):
            engine.tick(1 / 60)
        assert completed == {'first': [4, 8, 12], 'second': [4, 8, 12]}, completed
    finally:
        engine.exit()


if __name__ == '__main__':
    from pathlib import Path
    import sys

    _native_frame_probe(Path(sys.argv[1]), sys.argv[2])
    print('SHARED_WAIT_NATIVE_FRAMES_OK')
