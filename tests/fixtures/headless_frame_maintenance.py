"""Isolated real-engine worker for run/tick frame-maintenance contracts."""
from pathlib import Path
import json
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))


def drive_until(engine, drive, probe):
    from infernux.host import MainThreadCommandQueue

    if drive == 'tick':
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            engine.tick(1.0 / 60.0)
            if probe():
                return
            time.sleep(0.001)
        raise AssertionError('manual frames did not finish accepted document work')

    results = []
    queue = MainThreadCommandQueue.instance()

    def check_on_owner():
        ready = probe()
        if ready:
            results.append(True)
            engine.request_exit()
        return ready

    def observer():
        try:
            while not queue.run_sync('test.frame.progress', check_on_owner, timeout_ms=10000):
                time.sleep(0.001)
        except BaseException as error:
            results.append(str(error))
            engine.request_exit()

    watcher = threading.Thread(target=observer)
    timer = threading.Timer(12.0, engine.request_exit)
    watcher.start()
    timer.start()
    try:
        engine.run()
    finally:
        timer.cancel()
        watcher.join(timeout=11.0)
    assert not watcher.is_alive() and results == [True], results


def check(case, drive, root):
    from infernux.components import InxComponent
    from infernux.core.anim_state_machine import AnimStateMachine
    from infernux.core.assets import AssetManager
    from infernux.engine.deferred_task import DeferredTaskRunner
    from infernux.engine.engine import Engine
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    (root / 'Assets').mkdir(parents=True)
    (root / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(root / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    runner = DeferredTaskRunner.instance()
    runner.cancel()
    details = {}
    try:
        engine.init_headless(str(root))
        native = SceneManager.instance()
        core = EditorInteractionCore.instance()
        if case == 'steps':
            steps, updates, late, complete = [], [], [], []

            class FrameObserver(InxComponent):
                def update(self, delta_time):
                    updates.append(list(steps))

                def late_update(self, delta_time):
                    late.append(list(steps))
                    if len(late) == 4:
                        engine.request_exit()

            native.get_active_scene().create_game_object('FrameObserver').add_component(FrameObserver)
            assert runner.submit('Four steps', [
                (str(i), i / 4.0, lambda i=i: steps.append(i)) for i in range(1, 5)
            ], on_done=lambda ok: complete.append(ok))
            for invalid_delta in (-1.0, float('nan'), float('inf')):
                try:
                    engine.tick(invalid_delta)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid frame delta was accepted')
            assert not steps and runner.is_busy
            native.play()
            if drive == 'tick':
                for _ in range(4):
                    engine.tick(1.0 / 60.0)
            else:
                timer = threading.Timer(10.0, engine.request_exit)
                timer.start()
                try:
                    engine.run()
                finally:
                    timer.cancel()
            expected = [list(range(1, n + 1)) for n in range(1, 5)]
            assert updates == late == expected, (updates, late)
            assert complete == [True] and not runner.is_busy
            details.update(updates=updates, late_updates=late, completions=complete)
        elif case == 'asset_save':
            path = root / 'Assets/Queued.animfsm'
            fsm = AnimStateMachine(name='Queued')
            assert fsm.save(str(path))
            fsm = AnimStateMachine.load(str(path))
            fsm.add_state('SavedState')
            AssetManager.schedule_asset_save('animfsm', str(path), fsm, debounce_sec=0.0)
            def ready():
                return [state.name for state in AnimStateMachine.load(str(path)).states] == ['SavedState']
            drive_until(engine, drive, ready)
            assert ready()
            details['saved_states'] = ['SavedState']
        else:
            files = SceneFileManager.instance()
            scene = native.get_active_scene()
            witness = scene.create_game_object('OriginalWitness')
            document_id = files.register_loaded_scene(scene, '', dirty=True)
            assert files.activate_loaded_scene(scene)
            path = root / 'Assets/Saved.scene'
            initial = core.documents.request_save_to_resource(document_id, str(path))
            assert initial.accepted, initial
            assert path.is_file() and not core.documents.require(document_id).is_dirty
            if case == 'scene_save':
                witness.name = 'SavedWitness'
                core.documents.mark_changed(document_id)
                assert core.documents.defer_save(document_id).accepted
                def ready():
                    return not core.documents.require(document_id).is_dirty and 'SavedWitness' in path.read_text(encoding='utf-8')
                drive_until(engine, drive, ready)
                assert 'SavedWitness' in path.read_text(encoding='utf-8')
                details['saved_scene'] = True
            elif case == 'scene_open':
                target = native.create_scene('Opened')
                target.create_game_object('LoadedWitness')
                target_path = root / 'Assets/Opened.scene'
                target_path.write_text(target.serialize_asset(), encoding='utf-8')
                native.unload_scene(target)
                imported = AssetManager.import_asset(str(target_path), database=engine.get_asset_database())
                assert imported, imported.error
                assert files.open_scene(str(target_path))
                def ready():
                    current = native.get_active_scene()
                    if not files.is_loading and current.name == 'Opened':
                        assert 'LoadedWitness' in current.serialize_asset()
                        assert Path(files.current_scene_path) == target_path
                        return True
                    return False
                drive_until(engine, drive, ready)
                details['loaded_scene'] = True
            else:
                raise ValueError(case)
    finally:
        runner.cancel()
        engine.exit()
    result = dict(case=case, drive=drive, passed=True, **details)
    (root / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    check(sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve())
