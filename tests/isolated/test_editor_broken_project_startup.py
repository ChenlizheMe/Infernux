"""Broken project content must leave a real editor available for repair."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ != "__main__":
    from tests.isolated import engine_python_path


def test_editor_opens_broken_content_and_can_load_a_repaired_scene(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": engine_python_path(), "INFERNUX_DATA_ROOT": str(tmp_path / "User")},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BROKEN_CONTENT_EDITOR_READY" in result.stdout


def exercise(project):
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors
    from infernux.components.script_loader import get_script_error_by_path
    from infernux.lib import LogLevel, SceneManager

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    bad_script = assets / "BrokenStartup.py"
    bad_script.write_text("def invalid(:\n", encoding="utf-8")
    bad_scene = assets / "Broken.scene"
    bad_scene.write_text("{ damaged scene", encoding="utf-8")
    (assets / "Broken.png").write_bytes(b"not an image")
    bootstrap = EditorBootstrap(str(project), LogLevel.Warn)
    try:
        bootstrap.run()
        assert bootstrap.window_manager is not None
        assert bootstrap.interaction_core is not None
        assert get_script_error_by_path(str(bad_script))
        errors = AssetLoadErrors.instance()
        errors.poll()
        assert not bootstrap.scene_file_manager._do_open_scene(str(bad_scene))
        assert bootstrap.scene_file_manager.last_scene_load["status"] == "failed"
        assert errors.error(str(bad_scene))
        native = SceneManager.instance()
        repaired = native.create_scene("修复后的场景")
        repaired.create_game_object("可操作的物体")
        good_path = assets / "Repaired.scene"
        good_path.write_text(repaired.serialize_asset(), encoding="utf-8")
        native.unload_scene(repaired)
        database = bootstrap.engine.get_asset_database()
        assert database.import_asset(str(good_path)).succeeded
        assert bootstrap.scene_file_manager._do_open_scene(str(good_path))
        assert native.get_active_scene().find("可操作的物体") is not None
        print("BROKEN_CONTENT_EDITOR_READY", flush=True)
    finally:
        if bootstrap.engine is not None:
            bootstrap.engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]))
