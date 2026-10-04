from infernux.engine.ui.engine_status import EngineStatus


def setup_function():
    EngineStatus.clear()


def test_persistent_progress_has_progress_kind():
    EngineStatus.set("Importing", 0.4)
    assert EngineStatus.get() == ("Importing", 0.4, "progress")


def test_completion_and_failure_flashes_do_not_remain_progress_bars():
    EngineStatus.flash("Done", 1.0, duration=5.0)
    assert EngineStatus.get() == ("Done", 1.0, "success")

    EngineStatus.flash("Failed", 0.0, duration=5.0)
    assert EngineStatus.get() == ("Failed", 0.0, "error")


def test_expired_flash_clears_all_state(monkeypatch):
    EngineStatus.flash("Done", 1.0, duration=1.0)
    monkeypatch.setattr("infernux.engine.ui.engine_status.time.monotonic", lambda: float("inf"))
    assert EngineStatus.get() == ("", -1.0, "idle")


def test_concurrent_sources_do_not_hide_higher_priority_work():
    EngineStatus.set("Importing", 0.25, source="import", priority=10)
    EngineStatus.flash("Saved", 1.0, duration=5.0, source="save")
    assert EngineStatus.get() == ("Importing", 0.25, "progress")

    EngineStatus.clear(source="import")
    assert EngineStatus.get() == ("Saved", 1.0, "success")


def test_clearing_one_source_preserves_other_sources():
    EngineStatus.set("One", 0.1, source="one")
    EngineStatus.set("Two", 0.2, source="two")
    EngineStatus.clear(source="two")
    assert EngineStatus.get() == ("One", 0.1, "progress")
