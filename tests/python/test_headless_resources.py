from __future__ import annotations

from pathlib import Path
import pytest

from infernux import resources
from infernux.engine import headless
from infernux.engine import library_sync
from infernux.plugins import PluginManager


def test_headless_prepares_resources_before_plugin_startup(tmp_path, monkeypatch):
    events: list[object] = []

    class _Native:
        exit_requested = False

    class _Engine:
        def __init__(self, *_args):
            events.append("engine.create")

        def init_headless(self, project):
            events.append(("engine.init", project))

        def get_native_engine(self):
            return _Native()

        def exit(self):
            events.append("engine.exit")

    class _Plugins:
        @staticmethod
        def shutdown():
            events.append("plugins.shutdown")

    monkeypatch.setattr(headless, "Engine", _Engine)
    monkeypatch.setattr(
        library_sync,
        "sync_resources",
        lambda project: events.append(("resources.sync", project)),
    )
    monkeypatch.setattr(
        resources,
        "activate_library",
        lambda project: events.append(("resources.activate", project)),
    )
    monkeypatch.setattr(
        "infernux.engine._acquire_project_lock",
        lambda project, owner: events.append(("lock.acquire", project, owner))
        or (str(tmp_path / "project.lock"), "token"),
    )
    monkeypatch.setattr(
        "infernux.engine._remove_project_lock",
        lambda path, token: events.append(("lock.remove", path, token)),
    )
    monkeypatch.setattr(
        PluginManager,
        "startup",
        lambda project, engine=None, runtime=False: events.append(
            ("plugins.startup", project, runtime)
        )
        or _Plugins(),
    )

    from infernux.version import ENGINE_RELEASE

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".infernux-version").write_text(ENGINE_RELEASE + "\n", encoding="utf-8")
    project = str(project_dir)
    assert headless.run_headless(project, lambda *_: True, max_frames=0) == 0

    assert events.index(("lock.acquire", project, "headless")) < events.index(("resources.sync", project))

    assert events.index(("resources.sync", project)) < events.index(
        ("plugins.startup", project, False)
    )
    assert events.index(("resources.activate", project)) < events.index(
        ("plugins.startup", project, False)
    )


@pytest.mark.parametrize("mode", ["editor", "headless"])
@pytest.mark.parametrize("failure", ["claim", "sync"])
def test_host_owns_project_before_resource_writes_and_releases_failed_startup(tmp_path, monkeypatch, mode, failure):
    from infernux import engine as entry
    from infernux.version import ENGINE_RELEASE
    project = tmp_path / "project"
    project.mkdir()
    (project / ".infernux-version").write_text(ENGINE_RELEASE + "\n", encoding="utf-8")
    monkeypatch.delenv("_INFERNUX_PROJECT_LOCK_PATH", raising=False)
    monkeypatch.delenv("_INFERNUX_PROJECT_LOCK_TOKEN", raising=False)
    calls = []

    def sync(path):
        assert Path(entry._default_lock_path(path)).is_file()
        calls.append("sync")
        raise RuntimeError("sync failed")

    if failure == "claim":
        monkeypatch.setattr(entry, "_acquire_project_lock",
                            lambda *_: (_ for _ in ()).throw(RuntimeError("claim failed")))
    monkeypatch.setattr(library_sync, "sync_resources", sync)
    with pytest.raises(RuntimeError, match=failure + " failed"):
        if mode == "editor":
            entry.release_engine(str(project))
        else:
            headless.run_headless(str(project), lambda *_: True, max_frames=0)
    assert calls == (["sync"] if failure == "sync" else [])
    assert not Path(entry._default_lock_path(str(project))).exists()
