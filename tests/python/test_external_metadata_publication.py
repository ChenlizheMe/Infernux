"""External sidecar publication through the real importer and document controller."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("event_kind", [
    "modified", "created", "atomic_move", "delete_create", "stale_delete", "guid_replaced", "malformed",
])
@pytest.mark.parametrize("draft", [False, True])
def test_external_metadata_refreshes_import_settings(tmp_path, event_kind, draft):
    import infernux

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), event_kind, str(int(draft))],
        env=dict(os.environ, PYTHONPATH=str(Path(infernux.__file__).resolve().parent.parent)),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METADATA_PUBLICATION_OK" in result.stdout


def exercise(project, event_kind, draft):
    from PIL import Image
    from infernux.core.assets import AssetManager
    from infernux.core.asset_types import read_texture_import_settings
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.resources_manager import ResourceChangeHandler
    from infernux.engine.interaction import (
        DocumentRegistry, DocumentKind, DocumentKey, DocumentCapability,
        AssetMutationService, SelectionService,
    )
    from infernux.engine.ui.asset_details_renderer import _ImportSettingsController
    from infernux.lib import LogLevel, RuntimeMode

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        path = project / "Assets" / "Shared.png"
        Image.new("RGBA", (8, 8), (210, 32, 42, 255)).save(path)
        imported = AssetManager.import_asset(str(path))
        assert imported and imported.guid
        database = engine._engine.get_asset_database()
        handler = ResourceChangeHandler(engine._engine)
        documents = DocumentRegistry.instance()
        service = AssetMutationService(documents, SelectionService.instance())
        changes = []
        service.add_observer(changes.append)
        settings = read_texture_import_settings(str(path))
        controller = _ImportSettingsController("texture", str(path), settings)
        document = documents.create(
            DocumentKind.IMPORT_SETTINGS, "Shared import", key=DocumentKey.asset(
                DocumentKind.IMPORT_SETTINGS, imported.guid), resource_path=str(path),
            capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
            controller=controller,
        )
        controller.document_id = document.document_id
        if draft:
            controller.settings.max_size = 32
            documents.mark_changed(document.document_id)

        meta = Path(str(path) + ".meta")
        before_source = path.read_bytes()
        data = json.loads(meta.read_text(encoding="utf-8"))
        portable = database.get_meta_by_guid(imported.guid).serialize_document_portable(str(project))
        assert portable == data
        assert portable["metadata"]["file_path"]["value"] == "Assets/Shared.png"
        assert "content_hash" not in portable["metadata"]
        if event_kind in {"guid_replaced", "malformed"}:
            generation = database.query_generation
            if event_kind == "guid_replaced":
                data["metadata"]["guid"]["value"] = "1" * 32
                candidate = json.dumps(data)
                error = RuntimeError
            else:
                candidate = "{ invalid document"
                error = json.JSONDecodeError
            meta.write_text(candidate, encoding="utf-8")
            with pytest.raises(error):
                handler._commit_metadata_modified(str(path))
            assert database.query_generation == generation
            assert database.get_guid_from_path(str(path)) == imported.guid
            assert controller.settings.max_size == (32 if draft else settings.max_size)
            assert changes == []
            assert meta.read_text(encoding="utf-8") == candidate
            assert path.read_bytes() == before_source
            print("METADATA_PUBLICATION_OK")
            return
        key = "max_size"
        assert key in data["metadata"], data
        old_size = database.get_meta_by_guid(imported.guid).get_int(key)
        assert old_size != 64
        data["metadata"][key]["value"] = 64
        payload = json.dumps(data, ensure_ascii=False, indent=4) + "\n"
        event = SimpleNamespace(is_directory=False, src_path=str(meta))
        if event_kind == "atomic_move":
            temp = Path(str(meta) + ".tmp.123.456")
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, meta)
            handler.on_moved(SimpleNamespace(is_directory=False, src_path=str(temp), dest_path=str(meta)))
        else:
            if event_kind == "delete_create":
                meta.unlink()
                handler.on_deleted(event)
            meta.write_text(payload, encoding="utf-8")
            (handler.on_created if event_kind in {"created", "delete_create"} else handler.on_modified)(event)
            if event_kind == "stale_delete":
                handler.on_deleted(event)

        assert handler.pending_count == 1, "metadata change was dropped at watcher ingress"
        handler.process_pending_reloads(force=True)
        assert database.get_meta_by_guid(imported.guid).get_int(key) == 64
        assert controller.settings.max_size == 64
        assert controller.disk_settings.max_size == 64
        assert not document.is_dirty
        assert len(changes) == 1
        assert path.read_bytes() == before_source
        assert database.get_guid_from_path(str(path)) == imported.guid
        generation = database.query_generation
        committed_meta = meta.read_bytes()
        # Repeated watcher notifications, including our own atomic publication,
        # must not reimport or cancel a newer Inspector draft.
        controller.settings.max_size = 128
        documents.mark_changed(document.document_id)
        for _ in range(3):
            handler.on_modified(event)
            handler.process_pending_reloads(force=True)
        assert controller.settings.max_size == 128 and document.is_dirty
        assert database.query_generation == generation and len(changes) == 1
        assert meta.read_bytes() == committed_meta
        # A second external edit inside any self-write suppression window is
        # nevertheless a real publication, not an echo.
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["metadata"][key]["value"] = 256
        meta.write_text(json.dumps(data), encoding="utf-8")
        handler.on_modified(event)
        handler.process_pending_reloads(force=True)
        assert database.get_meta_by_guid(imported.guid).get_int(key) == 256
        assert controller.settings.max_size == 256 and not document.is_dirty
        assert len(changes) == 2
        print("METADATA_PUBLICATION_OK")
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
