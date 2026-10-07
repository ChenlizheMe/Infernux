"""A failed wait retires its coroutine without interrupting native frame dispatch."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("wait_name", ["WaitUntil", "WaitWhile"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_wait_failure_does_not_starve_other_components(tmp_path, wait_name, cleanup_fails):
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), wait_name, str(int(cleanup_fails))],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "COROUTINE_FRAME_ISOLATION_OK" in child.stdout


def exercise(project, wait_name, cleanup_fails):
    from infernux.components import InxComponent
    from infernux import coroutine
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    calls = []
    wait_type = getattr(coroutine, wait_name)

    class Failing(InxComponent):
        def start(self):
            self.predicate_calls = 0
            self.close_calls = 0
            self.handle = self.start_coroutine(self.wait())

        def predicate(self):
            self.predicate_calls += 1
            raise ValueError("intentional wait predicate failure")

        def wait(self):
            try:
                yield wait_type(self.predicate)
            finally:
                self.close_calls += 1
                if cleanup_fails:
                    raise RuntimeError("intentional wait cleanup failure")

    class Healthy(InxComponent):
        def update(self, dt):
            calls.append("update")

        def late_update(self, dt):
            calls.append("late")

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        scene = manager.create_scene("CoroutineIsolation")
        manager.set_active_scene(scene)
        bad = scene.create_game_object("Failing").add_component(Failing)
        scene.create_game_object("Healthy").add_component(Healthy)
        manager.play()
        for _ in range(4):
            engine.tick(1 / 60)
        assert calls == ["update", "late"] * 4, calls
        assert bad.predicate_calls == bad.close_calls == 1
        assert bad.handle.is_finished
        assert bad._coroutine_scheduler.count == 0
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
    print("COROUTINE_FRAME_ISOLATION_OK")
