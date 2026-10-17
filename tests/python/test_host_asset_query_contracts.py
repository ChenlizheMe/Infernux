"""Deterministic Host JSON Pointer edits and bounded asset catalog queries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from infernux.host import OperationError

BAD_INDICES = ["-1", "01", "+1", "1 ", " 1", "１", "١", "1_0", "-", "3", "00", "", "1.0", "1e0"]


@pytest.mark.parametrize("case", ["invalid_edits", "valid_undo", "catalog_assets", "catalog_packages", "catalog_all"])
def test_real_headless_host_asset_contract(tmp_path, case):
    import infernux

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), case],
        env=dict(os.environ, PYTHONPATH=str(Path(infernux.__file__).resolve().parent.parent)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HOST_ASSET_CONTRACT_OK" in result.stdout


def _exercise(project, case):
    from infernux.components import serialized_field
    from infernux.core.assets import AssetManager
    from infernux.core.data_asset import DataAsset
    from infernux.engine.engine import Engine
    from infernux.engine.interaction import DocumentKey, DocumentKind, EditorInteractionCore, SelectionDomain
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.undo import UndoManager
    from infernux.host import EditorAutomationHost, MainThreadCommandQueue, OperationRegistry
    from infernux.host.asset_operations import build_asset_operations
    from infernux.host.data_asset_operations import build_data_asset_operations
    from infernux.lib import LogLevel, RuntimeMode

    class Steps(DataAsset):
        __serialized_type_id__ = "test.host.canonical_pointer.steps"
        values: list[int] = serialized_field(default=[10, 20, 30])

    for folder in ("Assets", "Packages", "ProjectSettings"):
        (project / folder).mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = AssetManager.require_asset_database()
        core = EditorInteractionCore.instance()
        core.project_assets.configure(str(project), database)
        core.panels.register_selection_authority("project", (SelectionDomain.ASSET,))
        undo = UndoManager(core.action_journal)
        EditorAutomationHost.set_provider(EditorAutomationHost())
        MainThreadCommandQueue.instance().drain()
        registry = OperationRegistry()
        for operation in (*build_asset_operations(str(project)), *build_data_asset_operations()):
            registry.register(operation)

        def execute(name, **arguments):
            return registry.execute(name, arguments, capabilities=("asset.*",))

        if case.startswith("catalog_"):
            for root in ("Assets", "Packages"):
                for index in range(4):
                    path = project / root / f"Entry{index}.txt"
                    path.write_text(str(index), encoding="utf-8")
                    assert database.import_asset(str(path)) is not None
            root = case.removeprefix("catalog_")
            expected = 8 if root == "all" else 4
            full = execute("infernux.asset.list", root=root, limit=2000)
            assert full["catalog_count"] == expected
            for limit in (1, 2, 2000):
                for query, extension in (("", ""), ("Entry0", ".txt"), ("NoMatch", ""), ("", ".nope")):
                    result = execute("infernux.asset.list", root=root, limit=limit, query=query, extension=extension)
                    assert result["catalog_count"] == expected, result
                    assert result["global_catalog_count"] == full["global_catalog_count"]
                    matches = [
                        item
                        for item in full["assets"]
                        if query.casefold() in item["path"].casefold() and item["path"].casefold().endswith(extension)
                    ]
                    assert result["assets"] == matches[:limit]
                    assert result["returned"] == min(limit, len(matches))
        else:
            path = project / "Assets" / "Steps.inxdata"
            asset = Steps()
            asset.save_to(str(path), database=database)
            guid = asset.guid
            before = path.read_bytes()
            meta_before = path.with_suffix(".inxdata.meta").read_bytes()
            history_before = len(core.action_journal.applied_entries())
            if case == "invalid_edits":
                for token in BAD_INDICES:
                    with pytest.raises(OperationError) as raised:
                        execute(
                            "infernux.data_asset.property.set",
                            asset_guid=guid,
                            pointer="/fields/values/" + token,
                            value=99,
                        )
                    assert raised.value.code == "operation.invalid_arguments"
                    assert path.read_bytes() == before
                    assert path.with_suffix(".inxdata.meta").read_bytes() == meta_before
                    assert database.get_guid_from_path(str(path)) == guid
                    assert len(core.action_journal.applied_entries()) == history_before
            else:
                result = execute(
                    "infernux.data_asset.property.set", asset_guid=guid, pointer="/fields/values/1", value=99
                )
                assert result["document"]["fields"]["values"] == [10, 99, 30]
                assert json.loads(path.read_text(encoding="utf-8"))["fields"]["values"] == [10, 99, 30]
                undo.undo()
                controller = core.documents.get_by_key(DocumentKey.asset(DocumentKind.DATA_ASSET, guid)).controller
                controller.flush_autosave(force=True)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    AssetManager.poll_pending_asset_writes()
                    if json.loads(path.read_text(encoding="utf-8"))["fields"]["values"] == [10, 20, 30]:
                        break
                    time.sleep(0.01)
                assert json.loads(path.read_text(encoding="utf-8"))["fields"]["values"] == [10, 20, 30]
                assert database.get_guid_from_path(str(path)) == guid
        print("HOST_ASSET_CONTRACT_OK", case)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), sys.argv[2])
