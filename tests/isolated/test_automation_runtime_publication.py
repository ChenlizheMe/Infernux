"""Exercise real CPU/headless native frames in an isolated process."""
from pathlib import Path
import subprocess
import sys


def test_real_transform_publication_and_distinct_step_receipts(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "publication and step receipts passed" in result.stdout


def probe(project):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic
    from infernux.engine import Engine
    from infernux.engine.play_mode import PlayModeState
    from infernux.lib import RuntimeMode, SceneManager, Vector3, LogLevel
    from infernux.host.bridge import AutomationBridge
    from infernux.host.editor_operations import install_editor_operations
    from infernux.host.operation_support import operation, on_editor
    from infernux.host.operations import OperationKind, OperationRegistry
    from infernux_automation import AutomationClient

    project = Path(project).resolve()
    (project / "ProjectSettings").mkdir(parents=True, exist_ok=True)
    engine = Engine(LogLevel.Error, mode=RuntimeMode.Headless)
    engine.init_headless(str(project), gpu=False)
    registry = OperationRegistry.instance()
    install_editor_operations(str(project), registry)
    bridge = AutomationBridge(project, registry=registry)
    native = SceneManager.instance()
    play = engine._play_mode_manager
    registration = bridge._attach_runtime_observations(engine.get_runtime_change_journal(), native, play)
    client = AutomationClient(project)
    scene = native.create_scene("Automation publication probe")
    native.set_active_scene(scene)
    actor = scene.create_game_object("Observed actor")
    read_count = []
    def read():
        def owner_read():
            read_count.append(1)
            return {"x": float(actor.transform.position.x)}
        return on_editor("test.publication.read", owner_read)
    registry.register(operation("test.publication.read", OperationKind.QUERY, "Read actual native transform",
                                read, capability="test.read", owner="publication-probe"))

    def pump(futures, timeout=8):
        deadline = monotonic() + timeout
        while not all(future.done() for future in futures):
            assert monotonic() < deadline, "Actual native publication did not complete"
            engine.tick(1.0 / 60.0)
            Event().wait(0.001)

    try:
        engine.tick(1.0 / 60.0)
        assert not registration.enabled
        with ThreadPoolExecutor(max_workers=3) as workers:
            watch = client.observe("test.publication.read")
            initial = workers.submit(client.observation, watch["id"], after=0)
            pump([initial])
            assert initial.result()["value"] == {"x": 0.0}
            assert registration.enabled
            # This is the native Transform write used by game scripts/physics,
            # outside ActionJournal and outside any Host mutation operation.
            actor.transform.position = Vector3(5, 0, 0)
            changed = workers.submit(client.observation, watch["id"], after=initial.result()["revision"], timeout=5)
            pump([changed])
            assert changed.result()["value"] == {"x": 5.0}, {"snapshot": changed.result(),
                "journal_revision": engine.get_runtime_change_journal().revision,
                "transform_serial": native.get_global_transform_serial(), "reads": len(read_count)}
            client.stop_observing(watch["id"])
            engine.tick(1.0 / 60.0)
            assert not registration.enabled
            count = len(read_count)
            actor.transform.position = Vector3(9, 0, 0)
            for _ in range(4):
                engine.tick(1.0 / 60.0)
            assert len(read_count) == count

            native.play()
            native.pause()
            play._state = PlayModeState.PAUSED
            start_sequence = int(play.step_sequence)
            start_frames = int(native.runtime_frame_count)
            requests = [workers.submit(client.call, "infernux.runtime.step", {"timeout_seconds": 5.0}) for _ in range(2)]
            pump(requests)
            receipts = [request.result() for request in requests]
            assert sorted(receipt["target_step_sequence"] for receipt in receipts) == [start_sequence + 1, start_sequence + 2]
            for receipt in receipts:
                assert receipt["transition_complete"]
                assert receipt["runtime"]["step_sequence"] >= receipt["target_step_sequence"]
            assert int(native.runtime_frame_count) >= start_frames + 2
            assert int(native.published_fixed_step_index) == int(native.fixed_step_index)
            assert not native.fixed_step_pending
        print("publication and step receipts passed")
    finally:
        client.close()
        bridge.close()
        if native.is_playing():
            native.stop()
        engine.exit()


if __name__ == "__main__":
    probe(sys.argv[1])
