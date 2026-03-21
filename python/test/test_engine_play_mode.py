"""Tests for InfEngine.engine.play_mode — PlayModeState, PlayModeEvent, PlayModeManager."""

from InfEngine.engine.play_mode import PlayModeState, PlayModeEvent, PlayModeManager


# ══════════════════════════════════════════════════════════════════════
# PlayModeState enum
# ══════════════════════════════════════════════════════════════════════

class TestPlayModeState:
    def test_members_exist(self):
        assert PlayModeState.EDIT is not None
        assert PlayModeState.PLAYING is not None
        assert PlayModeState.PAUSED is not None

    def test_distinct_values(self):
        values = {PlayModeState.EDIT, PlayModeState.PLAYING, PlayModeState.PAUSED}
        assert len(values) == 3


# ══════════════════════════════════════════════════════════════════════
# PlayModeEvent
# ══════════════════════════════════════════════════════════════════════

class TestPlayModeEvent:
    def test_fields(self):
        evt = PlayModeEvent(
            old_state=PlayModeState.EDIT,
            new_state=PlayModeState.PLAYING,
            timestamp=1.0,
        )
        assert evt.old_state is PlayModeState.EDIT
        assert evt.new_state is PlayModeState.PLAYING
        assert evt.timestamp == 1.0


# ══════════════════════════════════════════════════════════════════════
# PlayModeManager
# ══════════════════════════════════════════════════════════════════════

class TestPlayModeManager:
    def test_initial_state_is_edit(self):
        mgr = PlayModeManager()
        assert mgr._state is PlayModeState.EDIT

    def test_singleton_instance(self):
        mgr = PlayModeManager()
        assert PlayModeManager.instance() is mgr

    def test_timing_defaults(self):
        mgr = PlayModeManager()
        assert mgr._delta_time == 0.0
        assert mgr._time_scale == 1.0
        assert mgr._total_play_time == 0.0

    def test_scene_backup_none_initially(self):
        mgr = PlayModeManager()
        assert mgr._scene_backup is None
        assert mgr._scene_path_backup is None

    def test_listener_list_empty(self):
        mgr = PlayModeManager()
        assert mgr._state_change_listeners == []

    def test_set_asset_database(self):
        mgr = PlayModeManager()
        mgr.set_asset_database("fake_db")
        assert mgr._asset_database == "fake_db"
