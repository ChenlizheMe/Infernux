"""Script retirement follows actual native catalog moves in isolated projects."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("origin,suffix", [
    (origin, suffix) for origin in ("watcher", "internal", "batch") for suffix in (".txt", ".py")
] + [("watcher-pending", ".txt"), ("watcher", ".mat")])
def test_script_domain_relocation(tmp_path, origin, suffix):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), origin, suffix],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, origin, suffix):
    import time
    from watchdog.events import FileMovedEvent
    from infernux.components.registry import get_component_registrations
    from infernux.components.script_loader import ScriptLoadError
    from infernux.core.assets import AssetManager
    from infernux.engine.component_restore import create_component_instance
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.project_context import get_script_module_name
    from infernux.engine.resources_manager import ResourcesManager, _AssetImportNotReady
    from infernux.engine.runtime_dispatch import current_runtime_epoch
    from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    manager = None
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        manager = ResourcesManager(str(project), engine.get_native_engine())
        handler = manager._ensure_event_handler()
        source = assets / "RelocationSource.py"
        target = assets / f"Relocated{suffix}"
        source.write_text('from infernux.components import InxComponent\n'
                          'class RelocationProbe(InxComponent):\n'
                          '    _uses_component_data_store = False\n'
                          '    ticks = 0\n'
                          '    def update(self, delta_time): self.ticks += 1\n'
                          '    def marker(self): return "preserved"\n', encoding="utf-8")
        imported = database.import_asset(str(source))
        assert imported, imported.error
        guid = imported.guid

        def drain():
            deadline = time.monotonic() + 5.0
            while handler.pending_count and time.monotonic() < deadline:
                manager.process_pending_reloads(force=True)
            assert not handler.pending_count

        handler._check_script(str(source), force=True)
        drain()
        module_name = get_script_module_name(str(source))
        component_type = getattr(sys.modules[module_name], "RelocationProbe")
        type_guid = component_type._get_type_guid()
        scene = SceneManager.instance().create_scene("Relocation")
        instance = scene.create_game_object("Witness").add_py_component(component_type())
        instance._script_guid = guid
        instance._script_path = str(source)
        snapshot = scene.serialize_document()
        assert handler.dependency_graph.module_for_path(str(source)) is not None
        assert any(row.type_name == "RelocationProbe" for row in get_component_registrations(project_root=str(project)))
        assert current_runtime_epoch().has_phase(component_type, "update")
        if origin == "watcher-pending":
            handler._check_script(str(source), force=True)
            handler.process_script_worker()

        source.rename(target)
        if suffix == ".mat":
            with pytest.raises(_AssetImportNotReady):
                handler._commit_moved(str(source), str(target))
        elif origin.startswith("watcher"):
            handler.on_moved(FileMovedEvent(str(source), str(target)))
            drain()
        elif origin == "internal":
            moved = AssetManager.move_asset(str(source), str(target), database=database)
            assert moved, moved.error
            drain()
        else:
            moved = AssetManager.move_assets_batch([(str(source), str(target))], database=database)
            assert len(moved) == 1 and moved[0]
            drain()
        assert database.get_guid_from_path(str(target)) == guid
        assert not source.exists() and target.is_file()
        assert module_name not in sys.modules
        assert handler.dependency_graph.module_for_path(str(source)) is None
        records = [row for row in get_component_registrations(project_root=str(project))
                   if row.type_name == "RelocationProbe"]
        if suffix != ".py":
            assert not records, records
            assert component_type in current_runtime_epoch().retired_types
            assert not current_runtime_epoch().has_phase(component_type, "update")
            for prefer_loaded in (False, True):
                with pytest.raises(ScriptLoadError, match="Not a Python file"):
                    create_component_instance(guid, type_guid, "RelocationProbe", database,
                                              prefer_loaded_type=prefer_loaded)
            reopened = SceneManager.instance().create_scene("Reopened")
            transaction = SceneDocumentTransaction(reopened, document=snapshot, asset_database=database,
                                                    clear_registries=False, prefer_loaded_types=True)
            assert transaction.run_to_completion(), transaction.error
            missing = reopened.find("Witness").get_py_components()[0]
            assert missing._is_broken and missing._script_guid == guid
            assert missing._get_type_guid() == type_guid
            if suffix == ".txt":
                target.rename(source)
                handler.on_moved(FileMovedEvent(str(target), str(source)))
                drain()
                assert database.get_guid_from_path(str(source)) == guid
                restored = reopened.find("Witness").get_py_components()[0]
                assert not restored._is_broken
                assert restored.marker() == "preserved"
                assert restored._script_guid == guid and restored._get_type_guid() == type_guid
                assert instance.marker() == "preserved"
                assert component_type not in current_runtime_epoch().retired_types
        else:
            assert len(records) == 1 and Path(records[0].script_path) == target
            assert component_type not in current_runtime_epoch().retired_types
            assert instance.marker() == "preserved"
    finally:
        if manager is not None:
            manager.stop()
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
