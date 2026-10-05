"""External scene reloads keep the addressed World and other resident authors intact."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("target_active", [True, False])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("valid", [True, False])
def test_external_reload_uses_resident_document_owner(tmp_path, target_active, dirty, valid):
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path),
         str(int(target_active)), str(int(dirty)), str(int(valid))],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "RESIDENT_SCENE_RELOAD_OK" in child.stdout


def _exercise(project: Path, target_active: bool, dirty: bool, valid: bool) -> None:
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.engine.resources_manager import ResourceChangeHandler
    from infernux.engine.interaction import (
        DocumentRegistry, DocumentState, ExternalDocumentConflictService,
    )
    from infernux.lib import LogLevel, RuntimeMode, SceneManager
    from infernux.ui import UIImage
    from infernux.renderstack import RenderStack

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        native = SceneManager.instance()
        database = engine.get_asset_database()
        manager = SceneFileManager.instance()
        registry = DocumentRegistry.instance()
        paths = {}
        for label in ("A", "B"):
            source = native.create_scene("Source" + label)
            source.create_game_object("Original" + label)
            source.create_game_object("Stack" + label).add_component(RenderStack)
            path = assets / (label + ".scene")
            path.write_text(json.dumps(source.serialize_document()), encoding="utf-8")
            native.unload_scene(source)
            assert database.import_asset(str(path)).succeeded
            paths[label] = path
        assert manager.load_scene_additive_immediate(str(paths["A"]), record_history=False)
        scene_a = native.get_active_scene()
        doc_a = registry.require(manager.document_id)
        assert manager.load_scene_additive_immediate(str(paths["B"]), record_history=False)
        scene_b = native.get_active_scene()
        doc_b = registry.require(manager.document_id)
        other_object = scene_b.find("OriginalB")
        other_stack = RenderStack.instance(scene_b)
        assert other_stack is not None
        other_component = UIImage()
        other_object.add_py_component(other_component)
        other_component.width = 137.0
        registry.mark_changed(doc_b.document_id)
        assert manager.activate_loaded_scene(scene_a if target_active else scene_b)
        if dirty:
            scene_a.find("OriginalA").name = "LocalDraftA"
            registry.mark_changed(doc_a.document_id)
        target_before = scene_a.serialize_document()
        other_before = scene_b.serialize_document()
        other_revision = (doc_b.revision, doc_b.saved_revision)
        world_a, world_b = int(scene_a.world_id), int(scene_b.world_id)
        active_before = native.get_active_scene()
        active_document_before = manager.document_id
        path_before = manager.current_scene_path
        doc_a_id, doc_b_id = doc_a.document_id, doc_b.document_id

        external = json.loads(paths["A"].read_text(encoding="utf-8"))
        external["objects"][0]["name"] = "ExternalA"
        disk_text = json.dumps(external) if valid else '{"objects": [invalid'
        paths["A"].write_text(disk_text, encoding="utf-8")
        handler = ResourceChangeHandler(engine.get_native_engine(), project_path=str(project))
        handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(paths["A"])))
        assert handler.process_pending_reloads(force=True) == 1
        assert handler.pending_count == 0
        if dirty:
            assert doc_a.state is DocumentState.CONFLICT
            conflicts = ExternalDocumentConflictService(registry)
            conflicts.poll()
            assert conflicts.active is not None and conflicts.active.document_id == doc_a_id
            result = conflicts.reload(conflicts.active.conflict_id)
            assert result.accepted is valid, result.message
        if valid:
            assert scene_a.find("ExternalA") is not None
            assert doc_a.state is DocumentState.READY and not doc_a.is_dirty
            assert doc_a.durable_file_state is not None
        else:
            assert doc_a.state is DocumentState.CONFLICT
            assert scene_a.serialize_document() == target_before
        assert (doc_a.document_id, doc_b.document_id) == (doc_a_id, doc_b_id)
        assert (int(scene_a.world_id), int(scene_b.world_id)) == (world_a, world_b)
        assert native.get_active_scene() is active_before
        assert manager.document_id == active_document_before
        assert manager.current_scene_path == path_before
        assert scene_b.serialize_document() == other_before
        assert (doc_b.revision, doc_b.saved_revision) == other_revision
        assert scene_b.find("OriginalB") is other_object
        assert other_component.game_object is other_object and other_component.width == 137.0
        assert other_component in other_object.get_py_components()
        assert RenderStack.instance(scene_a) is not None
        assert RenderStack.instance(scene_b) is other_stack
        assert other_stack.game_object.scene is scene_b
        assert manager.scene_for_document(doc_a_id) is scene_a
        assert manager.scene_for_document(doc_b_id) is scene_b
        assert paths["A"].read_text(encoding="utf-8") == disk_text
        print("RESIDENT_SCENE_RELOAD_OK", flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), *(bool(int(arg)) for arg in sys.argv[2:]))
