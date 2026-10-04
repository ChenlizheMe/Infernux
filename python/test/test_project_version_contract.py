from __future__ import annotations

import pytest

from Infernux.engine import release_engine, run_headless
from Infernux.engine.project_version import validate_project_engine_version
from Infernux.version import ENGINE_VERSION


def test_exact_engine_version_is_required(tmp_path):
    pin = tmp_path / ".infernux-version"
    with pytest.raises(RuntimeError, match="must declare"):
        validate_project_engine_version(str(tmp_path))
    pin.write_text(f"# pinned\n{ENGINE_VERSION}\n", encoding="utf-8")
    assert validate_project_engine_version(str(tmp_path)) == ENGINE_VERSION
    pin.write_text(f"{ENGINE_VERSION}\n9.9.9\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one"):
        validate_project_engine_version(str(tmp_path))


@pytest.mark.parametrize("entry", [release_engine, run_headless])
def test_wrong_engine_version_rejects_before_resource_sync(tmp_path, monkeypatch, entry):
    pin = tmp_path / ".infernux-version"
    pin.write_text("0.0.0\n", encoding="utf-8")
    before = pin.read_bytes()
    monkeypatch.setattr("Infernux.engine.library_sync.sync_resources", lambda *_: pytest.fail("must not touch project"))
    with pytest.raises(RuntimeError, match="exact required version"):
        if entry is run_headless:
            entry(str(tmp_path), lambda *_: False)
        else:
            entry(str(tmp_path))
    assert pin.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [".infernux-version"]
