"""Rejected Inspector edits cannot poison the published RenderEffect view."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.core.document_store import DocumentStore
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.ui.inspector_renderstack import (
    _resolve_effect_document_controller,
    _resolve_effect_group_document_controller,
)
from infernux.engine.ui.render_effect_inspector import (
    _inspector_parameter_instance,
    apply_render_effect_parameter_edit,
)
from infernux.engine.undo import UndoManager
from infernux.renderstack.render_effect_asset import (
    EffectAssetReference,
    RenderEffectAsset,
    RenderEffectGroupAsset,
    RenderEffectGroupEntry,
    dump_render_effect_document,
)
from infernux.renderstack.render_effect_compiler import (
    RenderEffectArtifactRegistry,
    expand_render_effect_reference,
    get_render_effect_feature,
)


@pytest.fixture(params=["effect", "group"])
def effect_document(engine, tmp_path, monkeypatch, request):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(UndoManager, "_instance", None)
    core = EditorInteractionCore()
    manager = UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    RenderEffectArtifactRegistry.clear()
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source_path = folder / "Bloom.effect"
    source_path.write_text(dump_render_effect_document(RenderEffectAsset(
        feature_type="infernux.post.bloom",
        parameters={"intensity": 0.5, "max_iterations": 3},
    )), encoding="utf-8")
    result = AssetManager.import_asset(str(source_path), database=database)
    assert result.succeeded, result.error
    source_bytes = source_path.read_bytes()
    is_group = request.param == "group"
    path = source_path
    if is_group:
        path = folder / "Post.effectgroup"
        path.write_text(dump_render_effect_document(RenderEffectGroupAsset(entries=(
            RenderEffectGroupEntry("bloom", EffectAssetReference(guid=result.guid),
                                   overrides={"intensity": 0.75}),
        ))), encoding="utf-8")
        result = AssetManager.import_asset(str(path), database=database)
        assert result.succeeded, result.error
    reference = RenderEffectRef(guid=result.guid)
    effect = expand_render_effect_reference(reference)[0]
    if is_group:
        controller = _resolve_effect_group_document_controller(effect)
        effect.bind_group_document_controller(controller)
    else:
        controller = _resolve_effect_document_controller(effect)
    assert controller is not None
    try:
        yield SimpleNamespace(
            core=core, manager=manager, effect=effect, controller=controller,
            path=path, reference=reference, is_group=is_group,
            source_path=source_path, source_bytes=source_bytes,
            initial=0.75 if is_group else 0.5,
        )
    finally:
        DocumentStore.flush(str(path))
        AssetManager.poll_pending_asset_writes()
        core.shutdown()
        RenderEffectArtifactRegistry.clear()


def _view(effect):
    return _inspector_parameter_instance(effect, get_render_effect_feature(effect.feature_type))


def _accept_undo_save_reopen(document, old_view, *, requested=1.75, expected=1.75):
    effect, controller = document.effect, document.controller
    assert apply_render_effect_parameter_edit(
        effect, "intensity", requested, resource_controller=controller,
    )
    assert old_view.intensity == pytest.approx(document.initial)
    assert effect.get_float("intensity") == pytest.approx(expected)
    assert _view(effect).intensity == pytest.approx(expected)
    assert document.core.documents.require(controller.document_id).is_dirty
    document.manager.undo()
    assert effect.get_float("intensity") == pytest.approx(document.initial)
    assert _view(effect).intensity == pytest.approx(document.initial)
    document.manager.redo()
    assert effect.get_float("intensity") == pytest.approx(expected)
    assert _view(effect).intensity == pytest.approx(expected)
    assert controller.flush_autosave(force=True)
    DocumentStore.flush(str(document.path))
    AssetManager.poll_pending_asset_writes()
    controller.poll_pending_writes()
    assert not document.core.documents.require(controller.document_id).is_dirty
    saved = json.loads(document.path.read_text(encoding="utf-8"))
    if document.is_group:
        assert saved["entries"][0]["overrides"]["intensity"] == pytest.approx(expected)
        assert document.source_path.read_bytes() == document.source_bytes
    else:
        assert saved["parameters"]["intensity"] == pytest.approx(expected)
    RenderEffectArtifactRegistry.clear()
    reopened = expand_render_effect_reference(document.reference)[0]
    assert reopened.get_float("intensity") == pytest.approx(expected)
    assert _view(reopened).intensity == pytest.approx(expected)


@pytest.mark.parametrize("rejection", ["no_document", "journal_disabled", "journal_executing", "exception"])
def test_rejected_edit_preserves_view_and_accepts_identical_retry(effect_document, monkeypatch, rejection):
    document = effect_document
    effect, controller, manager = document.effect, document.controller, document.manager
    view = _view(effect)
    original = effect.to_dict()
    revision = effect.revision
    disk = document.path.read_bytes()
    active_controller = controller
    if rejection == "no_document":
        active_controller = None
        if document.is_group:
            effect.bind_group_document_controller(None)
    if rejection == "journal_disabled":
        manager.enabled = False
    scope = manager.suppress() if rejection == "journal_executing" else nullcontext()
    with monkeypatch.context() as local, scope:
        if rejection == "exception":
            def reject(*_args, **_kwargs):
                raise RuntimeError("injected document submission rejection")
            local.setattr(controller, "apply_document", reject)
            with pytest.raises(RuntimeError, match="injected document submission rejection"):
                apply_render_effect_parameter_edit(effect, "intensity", 1.75,
                                                   resource_controller=active_controller)
        else:
            assert not apply_render_effect_parameter_edit(effect, "intensity", 1.75,
                                                          resource_controller=active_controller)
    manager.enabled = True
    if document.is_group:
        effect.bind_group_document_controller(controller)
    assert effect.to_dict() == original and effect.revision == revision
    assert document.path.read_bytes() == disk
    assert not document.core.action_journal.can_undo
    assert not document.core.documents.require(controller.document_id).is_dirty
    assert _view(effect) is view
    assert view.intensity == pytest.approx(document.initial)
    _accept_undo_save_reopen(document, view)


def test_invalid_candidate_preserves_view_and_next_valid_edit(effect_document):
    document = effect_document
    effect = document.effect
    view = _view(effect)
    original = effect.to_dict()
    revision = effect.revision
    with pytest.raises(ValueError, match="could not convert string to float"):
        apply_render_effect_parameter_edit(effect, "intensity", "not-a-number",
                                           resource_controller=document.controller)
    assert effect.to_dict() == original and effect.revision == revision
    assert _view(effect) is view and view.intensity == pytest.approx(document.initial)
    assert not document.core.action_journal.can_undo
    _accept_undo_save_reopen(document, view)


def test_candidate_keeps_typed_range_normalization(effect_document):
    document = effect_document
    _accept_undo_save_reopen(document, _view(document.effect), requested=7.0, expected=5.0)
