from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import time

import pytest

from Infernux.core.asset_types import MeshImportSettings
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.core.assets import AssetManager
from Infernux.lib import AssetRegistry
from Infernux.engine.interaction import (
    AuthoringMutationService,
    DocumentCapability,
    DocumentRegistry,
)
from Infernux.engine.undo import UndoManager
from Infernux.engine.ui import asset_details_renderer as details


class _ExecutionLayer:
    def __init__(self) -> None:
        self.file_path = ""

    def refresh_binding(self, _category, file_path) -> None:
        self.file_path = file_path


@pytest.fixture
def authoring_services():
    previous_registry = DocumentRegistry._instance
    previous_manager = UndoManager._instance
    previous_mutations = AuthoringMutationService._instance
    registry = DocumentRegistry()
    manager = UndoManager()
    try:
        yield registry, manager
    finally:
        details._ImportSettingsBatch._active_apply = None
        AuthoringMutationService._instance = previous_mutations
        UndoManager._instance = previous_manager
        DocumentRegistry._instance = previous_registry


def _model_state(path: str, guid: str, scale: float, *, attach_view: bool):
    details._ensure_categories()
    state = details._State()
    state.file_path = path
    state.category = "mesh"
    state.meta = {"guid": guid}
    state.settings = MeshImportSettings(scale_factor=scale)
    state.disk_settings = state.settings.copy()
    state.exec_layer = _ExecutionLayer()
    details._bind_import_settings_document(
        state,
        details._categories["mesh"],
        attach_view=attach_view,
    )
    return state


def test_multi_model_field_edit_is_one_undoable_document_transaction(authoring_services):
    registry, manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))

    assert batch.is_mixed("scale_factor")
    assert batch.apply_mutation(
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 3.0),
        "Set scale on selected models",
    )
    assert first.settings.scale_factor == 3.0
    assert second.settings.scale_factor == 3.0
    assert registry.require(first.document_id).revision == 1
    assert registry.require(second.document_id).revision == 1
    assert len(manager.action_journal.applied_entries()) == 1

    manager.undo()
    assert first.settings.scale_factor == 1.0
    assert second.settings.scale_factor == 2.0
    assert registry.require(first.document_id).revision == 0
    assert registry.require(second.document_id).revision == 0

    manager.redo()
    assert first.settings.scale_factor == 3.0
    assert second.settings.scale_factor == 3.0


def test_multi_model_renderer_marks_mixed_and_writes_every_draft(authoring_services):
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))

    class _Context:
        descriptors = None

        @staticmethod
        def calc_text_width(value):
            return float(len(value) * 8)

        def render_property_batch(self, descriptors, _label_width):
            self.descriptors = descriptors
            return {0: 4.0}

    context = _Context()
    field = details.FieldDef(
        "scale_factor",
        "asset.scale_factor",
        details.WidgetType.FLOAT,
        float_speed=0.01,
        float_range=(0.0001, 100.0),
    )
    details._render_import_field_batch(context, first, batch, [field])

    assert context.descriptors[0]["mix"] is True
    assert first.settings.scale_factor == 4.0
    assert second.settings.scale_factor == 4.0


def test_batch_apply_preflights_every_document_before_starting_work(authoring_services, monkeypatch):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))
    assert batch.apply_mutation(
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 3.0),
        "Set scale on selected models",
    )

    started = []
    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.begin_model_reimport",
        lambda path, _settings: started.append(path) or object(),
    )
    registry.update_metadata(
        second.document_id,
        capabilities=DocumentCapability.DISCARD,
    )

    with pytest.raises(RuntimeError, match="cannot be applied"):
        batch.request_apply()
    assert started == []
    assert registry.active_save_ticket(first.document_id) is None
    assert registry.active_save_ticket(second.document_id) is None


def test_batch_apply_starts_next_model_only_after_first_publication(authoring_services, monkeypatch):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))
    assert batch.apply_mutation(
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 3.0),
        "Set scale on selected models",
    )

    started = []
    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.begin_model_reimport",
        lambda path, _settings: started.append(path) or object(),
    )
    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.poll_model_reimport",
        lambda _database: SimpleNamespace(error=""),
    )

    assert batch.request_apply()
    assert started == [first.file_path]
    assert registry.active_save_ticket(first.document_id) is not None
    assert registry.active_save_ticket(second.document_id) is None
    registry.process_pending_saves()
    assert started == [first.file_path, second.file_path]
    assert registry.active_save_ticket(first.document_id) is None
    # The document pump may poll the newly started second controller in the
    # same pass, but it cannot start it before the first ticket completes.
    assert registry.active_save_ticket(second.document_id) is None
    assert not batch.is_applying()
    assert not batch.dirty_document_ids()


def test_batch_apply_leaves_later_draft_dirty_when_its_start_fails(authoring_services, monkeypatch):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))
    assert batch.apply_mutation(
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 3.0),
        "Set scale on selected models",
    )

    def begin(path, _settings):
        if path == second.file_path:
            raise RuntimeError("second import rejected")
        return object()

    monkeypatch.setattr("Infernux.core.assets.AssetManager.begin_model_reimport", begin)
    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.poll_model_reimport",
        lambda _database: SimpleNamespace(error=""),
    )

    assert batch.request_apply()
    registry.process_pending_saves()
    assert first.import_controller._pending_model_save is None
    assert registry.active_save_ticket(first.document_id) is None
    assert registry.active_save_ticket(second.document_id) is None
    assert not registry.require(first.document_id).is_dirty
    assert registry.require(second.document_id).is_dirty
    assert not batch.is_applying()


def test_batch_apply_stops_after_failed_owner_commit(authoring_services, monkeypatch):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))
    assert batch.apply_mutation(
        "scale_factor", lambda settings: setattr(settings, "scale_factor", 3.0), "Set scale"
    )
    started = []
    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.begin_model_reimport",
        lambda path, _settings: started.append(path) or object(),
    )

    class _FailedCommit:
        error = "source changed during import"

        def __bool__(self):
            return False

    monkeypatch.setattr(
        "Infernux.core.assets.AssetManager.poll_model_reimport",
        lambda _database: _FailedCommit(),
    )
    assert batch.request_apply()
    registry.process_pending_saves()
    assert started == [first.file_path]
    assert not batch.is_applying()
    assert registry.require(first.document_id).is_dirty
    assert registry.require(second.document_id).is_dirty
    assert registry.active_save_ticket(second.document_id) is None


def test_batch_apply_serializes_real_database_owner_commits(
    authoring_services, engine, tmp_path, monkeypatch
):
    registry, _manager = authoring_services
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source_fixture = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_hierarchy.gltf"
    model_dir = Path(database.assets_root) / tmp_path.name
    model_dir.mkdir()
    sources = (model_dir / "First.gltf", model_dir / "Second.gltf")
    states = []
    try:
        for source in sources:
            source.write_bytes(source_fixture.read_bytes())
            imported = AssetManager.import_asset(str(source), database=database)
            assert imported, imported.error
            states.append(_model_state(str(source), imported.guid, 1.0, attach_view=len(states) == 0))
        batch = details._ImportSettingsBatch(tuple(states))
        assert batch.apply_mutation(
            "scale_factor", lambda settings: setattr(settings, "scale_factor", 3.0), "Set scale"
        )
        assert batch.request_apply()
        assert registry.active_save_ticket(states[0].document_id) is not None
        assert registry.active_save_ticket(states[1].document_id) is None
        assert read_mesh_import_settings(str(sources[1])).scale_factor == 1.0

        deadline = time.monotonic() + 30
        while registry.active_save_ticket(states[0].document_id) is not None:
            registry.process_pending_saves()
            assert time.monotonic() < deadline
            time.sleep(.002)
        assert read_mesh_import_settings(str(sources[0])).scale_factor == 3.0
        assert registry.active_save_ticket(states[1].document_id) is not None
        assert read_mesh_import_settings(str(sources[1])).scale_factor == 1.0
        while batch.is_applying():
            registry.process_pending_saves()
            assert time.monotonic() < deadline
            time.sleep(.002)
        assert read_mesh_import_settings(str(sources[1])).scale_factor == 3.0
        assert not batch.dirty_document_ids()
    finally:
        for state, source in zip(states, sources):
            AssetRegistry.instance().invalidate_asset(state.meta["guid"])
            database.delete_asset(str(source))
            source.unlink(missing_ok=True)
            Path(str(source) + ".meta").unlink(missing_ok=True)


def test_switching_assets_retains_unapplied_model_draft(authoring_services):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    assert details._edit_import_settings(
        first,
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 2.5),
        "Set scale",
    )

    _model_state("C:/Project/Assets/second.fbx", "second-guid", 1.0, attach_view=True)
    reopened = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)

    assert reopened.settings.scale_factor == 2.5
    assert registry.require(reopened.document_id).is_dirty
    reopened.extra["model_active_page"] = "materials"
    assert reopened.settings.scale_factor == 2.5


def test_batch_revert_preflights_every_document_before_discarding(authoring_services):
    registry, _manager = authoring_services
    first = _model_state("C:/Project/Assets/first.fbx", "first-guid", 1.0, attach_view=True)
    second = _model_state("C:/Project/Assets/second.fbx", "second-guid", 2.0, attach_view=False)
    batch = details._ImportSettingsBatch((first, second))
    assert batch.apply_mutation(
        "scale_factor",
        lambda settings: setattr(settings, "scale_factor", 3.0),
        "Set scale on selected models",
    )

    registry.update_metadata(second.document_id, capabilities=DocumentCapability.SAVE)
    with pytest.raises(RuntimeError, match="cannot be reverted"):
        batch.request_revert()

    assert first.settings.scale_factor == 3.0
    assert second.settings.scale_factor == 3.0
    assert registry.require(first.document_id).is_dirty
    assert registry.require(second.document_id).is_dirty

    registry.update_metadata(
        second.document_id,
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
    )
    assert batch.request_revert()
    assert first.settings.scale_factor == 1.0
    assert second.settings.scale_factor == 2.0
    assert not registry.require(first.document_id).is_dirty
    assert not registry.require(second.document_id).is_dirty
