"""Retained DataAsset values save by GUID after real project mutations."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["rename", "reuse", "folder", "undo", "delete", "delete_reuse"])
def test_retained_data_asset_save_targets_bound_identity(tmp_path, case):
    import infernux
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), case],
        env=dict(os.environ, PYTHONPATH=str(Path(infernux.__file__).resolve().parent.parent)),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DATA_ASSET_RELOCATION_OK" in result.stdout


def exercise(project, case):
    from infernux.components import serialized_field
    from infernux.core.assets import AssetManager
    from infernux.core.data_asset import DataAsset
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.interaction import EditorInteractionCore, ProjectAssetCommandService, SelectionDomain
    from infernux.engine.undo import UndoManager
    from infernux.lib import LogLevel, RuntimeMode

    class PuzzleData(DataAsset):
        __serialized_type_id__ = "test.relocation.puzzle"
        count: int = serialized_field(default=3)

    folder = project / "Assets" / "Puzzle"
    folder.mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = AssetManager.require_asset_database()
        EditorInteractionCore.instance().panels.register_selection_authority("project", (SelectionDomain.ASSET,))
        service = ProjectAssetCommandService.instance()
        service.configure(str(project), database)
        path = folder / "Puzzle.inxdata"
        original = PuzzleData(count=3)
        original.save_to(str(path))
        guid = original.guid
        if case.startswith("delete"):
            path.unlink()
            assert AssetManager.delete_asset(str(path))
            if case == "delete_reuse":
                other = PuzzleData(count=200)
                other.save_to(str(path))
                baseline = path.read_bytes()
            original.count = 9
            with pytest.raises(RuntimeError, match="registered"):
                original.save()
            assert original.guid == guid
            if case == "delete_reuse":
                assert path.read_bytes() == baseline
                assert other.guid != guid
            else:
                assert not path.exists()
        else:
            if case == "folder":
                moved = Path(service.rename(str(folder), "Moved")) / path.name
            else:
                moved = Path(service.rename(str(path), "Renamed"))
            assert moved.is_file() and not path.exists()
            assert database.get_guid_from_path(str(moved)) == guid
            if case == "reuse":
                other = PuzzleData(count=200)
                other.save_to(str(path))
                baseline = path.read_bytes()
            if case == "undo":
                UndoManager.instance().undo()
                assert path.is_file() and not moved.exists()
                moved = path
            original.count = 9
            original.save()
            assert original.guid == guid
            assert Path(original.file_path) == moved
            assert DataAsset.load(str(moved)).count == 9
            if case == "reuse":
                assert path.read_bytes() == baseline and other.guid != guid
            elif case != "undo":
                assert not path.exists()
        print("DATA_ASSET_RELOCATION_OK")
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
