"""Play Mode script reload must publish to every World, not only the active one.

The editor prepares one transaction per World (authoring plus each runtime
World) and commits them under that World's scope. This test pins the coverage
without a native engine by substituting the scope object and the scene manager.
"""

from __future__ import annotations

import infernux.lib
from infernux.engine import play_mode


class _Scope:
    def __init__(self, identity, entered):
        self.identity = int(identity)
        self._entered = entered

    def __enter__(self):
        self._entered.append(('enter', self.identity))
        return self

    def __exit__(self, *exc_info):
        self._entered.append(('exit', self.identity))
        return False


class _Transaction:
    def __init__(self, identity, events):
        self.identity = int(identity)
        self.events = events
        self.had_live_targets = True
        self.recovered_count = 0

    def commit(self):
        self.events.append(('commit', self.identity))
        return {}

    def rollback(self):
        self.events.append(('rollback', self.identity))

    def finalize(self):
        self.events.append(('finalize', self.identity))


class _Batch:
    def __init__(self, identity, events):
        self.transaction = _Transaction(identity, events)
        self.members = ()


class _SceneManager:
    _authoring_runtime_world_id = 11
    _runtime_world_ids = [21, 31]


def test_reload_prepares_and_commits_each_world(monkeypatch):
    events = []
    entered = []

    class _FakeNative:
        @staticmethod
        def _RuntimeWorldScope(identity):
            return _Scope(identity, entered)

    monkeypatch.setattr(infernux.lib, '_Infernux', _FakeNative, raising=False)
    monkeypatch.setattr(play_mode, '_get_scene_manager', lambda: _SceneManager())

    manager = play_mode.PlayModeManager()
    manager._state = play_mode.PlayModeState.EDIT

    def scoped_prepare(revisions, *, replacing_scene=None, rebuild_helpers=False):
        assert entered and entered[-1][0] == 'enter'
        return _Batch(entered[-1][1], events)

    monkeypatch.setattr(manager, '_prepare_script_reload_current_world', scoped_prepare)

    batch = manager.prepare_script_reload_batch((object(),))
    prepared = [event[1] for event in entered if event[0] == 'enter']
    assert prepared == [11, 21, 31], prepared

    result = manager.commit_script_reload_batch(batch)
    assert result.success
    committed = [event[1] for event in events if event[0] == 'commit']
    assert committed == [11, 21, 31], committed
