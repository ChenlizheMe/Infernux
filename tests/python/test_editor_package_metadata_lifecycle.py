"""Import-sidecar edits must not retire live editor plugin services."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_editor_script_metadata_publication_keeps_service_source_generation(tmp_path):
    import infernux
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(Path(infernux.__file__).resolve().parent.parent)),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EDITOR_METADATA_LIFECYCLE_OK" in result.stdout


def exercise(project):
    from infernux.core.assets import AssetManager
    from infernux.engine.engine import Engine
    from infernux.engine.project_context import package_script_role
    from infernux.engine.resources_manager import ResourceChangeHandler, ResourcesManager
    from infernux.lib import LogLevel, RuntimeMode

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    package = project / "Packages" / "example" / "service"
    path = package / "editor" / "Service.py"
    path.parent.mkdir(parents=True)
    (package / "inx_package.json").write_text(json.dumps({"reference": "example/service"}), encoding="utf-8")
    path.write_text("class Service:\n    pass\n", encoding="utf-8")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    patch = pytest.MonkeyPatch()
    try:
        engine.init_headless(str(project))
        assert package_script_role(str(path), str(project)) == "editor"
        imported = AssetManager.import_asset(str(path))
        assert imported and imported.guid
        database = engine.get_asset_database()
        handler = ResourceChangeHandler(engine.get_native_engine(), project_path=str(project))
        notifications = []
        manager = SimpleNamespace(notify_script_catalog_changed=lambda *args: notifications.append(args))
        patch.setattr(ResourcesManager, "instance", classmethod(lambda cls: manager))
        source_before = path.read_bytes()
        sidecar = Path(str(path) + ".meta")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["metadata"]["audit_marker"] = {"type": "int", "value": 41}
        sidecar.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        handler._commit_metadata_modified(str(path))
        assert database.get_meta_by_guid(imported.guid).get_int("audit_marker") == 41
        assert database.get_guid_from_path(str(path)) == imported.guid
        assert path.read_bytes() == source_before
        assert notifications == [], notifications

        # Actual source saves still publish the lifecycle event. The sidecar
        # distinction must not disable reload of an edited plugin module.
        path.write_text("class Service:\n    changed = True\n", encoding="utf-8")
        handler._commit_modified(str(path))
        assert notifications == [(str(path), "modified")], notifications
        print("EDITOR_METADATA_LIFECYCLE_OK")
    finally:
        patch.undo()
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]))
