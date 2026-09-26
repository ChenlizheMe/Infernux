"""Explicit Apply must not parse on the editor thread or publish stale drafts."""
import time
from pathlib import Path

import pytest

from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.core.assets import AssetManager
from Infernux.engine.interaction import AuthoringMutationService, DocumentRegistry
from Infernux.engine.undo import UndoManager
from Infernux.engine.ui import asset_details_renderer as details
from Infernux.engine.ui.asset_execution_layer import AssetAccessMode, AssetExecutionLayer
from Infernux.lib import AssetRegistry


@pytest.fixture
def model(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Async.gltf"
    source.parent.mkdir()
    source.write_bytes((Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_hierarchy.gltf").read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    yield database, source, imported.guid
    AssetRegistry.instance().invalidate_asset(imported.guid)
    database.delete_asset(str(source))
    source.unlink(missing_ok=True)
    Path(str(source) + ".meta").unlink(missing_ok=True)


def poll(database):
    deadline = time.monotonic() + 30
    while True:
        result = database.try_commit_model_reimport()
        if result is not None:
            return result
        assert time.monotonic() < deadline
        time.sleep(.002)


def test_apply_snapshot_is_unpublished_until_owner_commit(model):
    database, source, guid = model
    original = Path(str(source) + ".meta").read_bytes()
    settings = read_mesh_import_settings(str(source))
    settings.scale_factor = 2
    snapshot = settings.to_dict()
    database.begin_model_reimport(str(source), snapshot)
    snapshot["scale_factor"] = 9
    assert Path(str(source) + ".meta").read_bytes() == original
    assert database.get_guid_from_path(str(source)) == guid
    with pytest.raises(RuntimeError, match="already in progress"):
        database.begin_model_reimport(str(source), settings.to_dict())
    result = poll(database)
    assert result and result.database_committed, result.error
    assert read_mesh_import_settings(str(source)).scale_factor == 2
    with pytest.raises(RuntimeError, match="No model Apply"):
        database.try_commit_model_reimport()


@pytest.mark.parametrize("change", ["source", "metadata", "deleted", "discard"])
def test_apply_rejects_stale_or_discarded_input_without_writing(model, change):
    database, source, guid = model
    sidecar = Path(str(source) + ".meta")
    original = sidecar.read_bytes()
    settings = read_mesh_import_settings(str(source))
    settings.scale_factor = 3
    database.begin_model_reimport(str(source), settings.to_dict())
    if change == "source":
        source.write_bytes(source.read_bytes() + b"\n ")
    elif change == "metadata":
        sidecar.write_bytes(original + b"\n ")
    elif change == "deleted":
        database.delete_asset(str(source))
    else:
        database.discard_model_reimport()
    expected = sidecar.read_bytes() if sidecar.exists() else None
    result = poll(database)
    assert not result and not result.database_committed and result.error
    assert (sidecar.read_bytes() if sidecar.exists() else None) == expected
    if change != "deleted":
        assert read_mesh_import_settings(str(source)).scale_factor == 1
        # A rejected transaction releases its slot, without automatic retry.
        database.begin_model_reimport(str(source), settings.to_dict())
        assert poll(database)


@pytest.mark.parametrize("close", ["none", "unregister", "clear"])
def test_document_save_keeps_newer_draft_dirty_and_close_prevents_publication(model, monkeypatch, close):
    database, source, guid = model
    registry = DocumentRegistry.instance()
    monkeypatch.setattr(UndoManager, "_instance", None)
    monkeypatch.setattr(AuthoringMutationService, "_instance", None)
    UndoManager()
    details._ensure_categories()
    state = details._State()
    state.file_path, state.category = str(source), "mesh"
    state.meta = {"guid": guid}
    state.settings = read_mesh_import_settings(str(source))
    state.disk_settings = state.settings.copy()
    state.exec_layer = AssetExecutionLayer("mesh", str(source), AssetAccessMode.READ_ONLY_RESOURCE)
    details._bind_import_settings_document(state, details._categories["mesh"])
    controller = state.import_controller
    assert details._edit_import_settings(state, "scale_factor", lambda s: setattr(s, "scale_factor", 2), "Set scale")
    outcome = registry.request_save(state.document_id)
    assert outcome.accepted and registry.is_save_pending(state.document_id)
    if close != "none":
        if close == "clear":
            registry.clear()
        else:
            registry.unregister(state.document_id)
        assert controller._pending_model_save is None
        result = poll(database)
        assert not result and "discarded" in result.error
        assert read_mesh_import_settings(str(source)).scale_factor == 1
        return
    assert details._edit_import_settings(state, "scale_factor", lambda s: setattr(s, "scale_factor", 3), "Set scale")
    registry.detach_view("inspector")
    deadline = time.monotonic() + 30
    while registry.is_save_pending(state.document_id):
        registry.process_pending_saves()
        assert time.monotonic() < deadline
        time.sleep(.002)
    assert read_mesh_import_settings(str(source)).scale_factor == 2
    assert state.settings.scale_factor == 3
    assert state.disk_settings.scale_factor == 2
    assert registry.require(state.document_id).is_dirty
    assert registry.request_discard(state.document_id).accepted
    assert state.settings.scale_factor == 2
    assert not registry.require(state.document_id).is_dirty
