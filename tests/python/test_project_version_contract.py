from __future__ import annotations

import pytest

from infernux.engine import release_engine, run_headless
@pytest.mark.parametrize("required", [None, "0.0.0"])
@pytest.mark.parametrize("entry", [release_engine, run_headless])
def test_runtime_engine_starts_without_a_matching_project_version(tmp_path, monkeypatch, entry, required):
    pin = tmp_path / ".infernux-version"
    if required is not None:
        pin.write_text(required + "\n", encoding="utf-8")

    class ResourceSyncReached(Exception):
        pass

    def sync_resources(_project):
        raise ResourceSyncReached

    monkeypatch.setattr("infernux.engine.library_sync.sync_resources", sync_resources)
    with pytest.raises(ResourceSyncReached):
        if entry is run_headless:
            entry(str(tmp_path), lambda *_: False)
        else:
            entry(str(tmp_path))
    if required is None:
        assert not pin.exists()
    else:
        assert pin.read_text(encoding="utf-8") == required + "\n"
