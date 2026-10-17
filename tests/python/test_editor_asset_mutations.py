from __future__ import annotations

import json

import pytest

from infernux.core.asset_ref import ParticleGraphRef
from infernux.engine.interaction import (
    ActionOrigin,
    AssetContentChange,
    AssetMutationKind,
    AssetRenameContentRegistry,
    AssetMutationService,
    DocumentKey,
    DocumentKind,
    DocumentRegistry,
    DocumentState,
    SelectionService,
    SelectionTarget,
)
from infernux.engine.path_utils import path_key
from infernux.engine.project_context import get_project_root, set_project_root
from infernux.engine.ui import project_file_ops
from infernux.engine.undo import LambdaCommand, UndoManager
from infernux.particle.artifact import (
    PARTICLE_RUNTIME_INDEX_FILENAME,
    PARTICLE_RUNTIME_INDEX_SCHEMA,
    ParticleArtifactRegistry,
)


@pytest.fixture(autouse=True)
def _isolate_asset_mutation_service():
    yield
    service = AssetMutationService.instance()
    if service is not None:
        service.shutdown()
    AssetRenameContentRegistry.instance().shutdown()


class _Controller:
    def __init__(self) -> None:
        self.moves = []
        self.reloaded = False

    def resource_moved(self, **payload) -> None:
        self.moves.append(payload)

    def reload_from_resource(self, **_payload) -> bool:
        self.reloaded = True
        return True


def test_content_mutations_are_typed_ordered_and_never_create_undo(tmp_path):
    documents = DocumentRegistry()
    selection = SelectionService()
    service = AssetMutationService(documents, selection)
    published = []
    service.add_observer(published.append)
    path = tmp_path / "Smoke.png"

    created = service.publish_content_change(
        str(path),
        AssetMutationKind.CREATED,
        guid="texture-guid",
        origin=ActionOrigin.USER,
    )
    modified = service.publish_content_change(
        str(path),
        AssetMutationKind.MODIFIED,
        guid="texture-guid",
        origin=ActionOrigin.EXTERNAL,
    )
    deleted = service.publish_content_change(
        str(path),
        AssetMutationKind.DELETED,
        guid="texture-guid",
        origin=ActionOrigin.USER,
    )

    assert published == [created, modified, deleted]
    assert all(isinstance(item, AssetContentChange) for item in published)
    assert [item.mutation.kind for item in published] == [
        AssetMutationKind.CREATED,
        AssetMutationKind.MODIFIED,
        AssetMutationKind.DELETED,
    ]
    assert [item.revision for item in published] == [1, 2, 3]
    assert service.resolve_path_hint("texture-guid") == ""


def test_only_external_content_mutation_advances_document_external_revision(tmp_path):
    documents = DocumentRegistry()
    service = AssetMutationService(documents, SelectionService())
    path = tmp_path / "Smoke.particlegraph"
    path.write_text("baseline", encoding="utf-8")
    controller = _Controller()
    document = documents.create(
        DocumentKind.PARTICLE_GRAPH,
        "Smoke",
        key=DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid"),
        resource_path=str(path),
        revision=1,
        saved_revision=0,
        controller=controller,
    )

    service.publish_content_change(
        str(path),
        AssetMutationKind.MODIFIED,
        guid="particle-guid",
        origin=ActionOrigin.USER,
    )
    assert document.external_revision == 0
    assert document.state is DocumentState.READY

    path.write_text("external-change", encoding="utf-8")
    service.publish_content_change(
        str(path),
        AssetMutationKind.MODIFIED,
        guid="particle-guid",
        origin=ActionOrigin.EXTERNAL,
    )
    assert document.external_revision == 1
    assert controller.reloaded
    assert document.state is DocumentState.READY
    assert not document.is_dirty


def test_asset_move_remaps_documents_selection_and_reference_display(tmp_path):
    old_path = tmp_path / "Old.particlegraph"
    new_path = tmp_path / "New.particlegraph"
    controller = _Controller()
    documents = DocumentRegistry()
    selection = SelectionService()
    service = AssetMutationService(documents, selection)
    document = documents.create(
        DocumentKind.PARTICLE_GRAPH,
        "Old",
        key=DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "graph-guid"),
        resource_path=str(old_path),
        controller=controller,
    )
    selection.select(
        SelectionTarget.asset("graph-guid"),
        owner_id="project",
        record_history=False,
    )
    reference = ParticleGraphRef(guid="graph-guid", path_hint="Assets/Old.particlegraph")

    change = service.publish_move(
        str(old_path),
        str(new_path),
        guid="graph-guid",
        origin=ActionOrigin.USER,
        operation_id="rename-operation",
    )

    assert change.remapped_document_ids == (document.document_id,)
    assert change.selection_changed is False
    assert document.resource_path == str(new_path)
    assert document.title == "New"
    assert document.key == DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "graph-guid")
    assert selection.snapshot.primary == SelectionTarget.asset("graph-guid")
    assert reference.display_name == "New.particlegraph"
    assert reference.to_dict() == {"guid": "graph-guid"}
    assert controller.moves == [
        {
            "document_id": document.document_id,
            "source_path": str(old_path),
            "destination_path": str(new_path),
            "guid": "graph-guid",
        }
    ]


def test_asset_move_rekeys_path_documents_and_subresource_selection(tmp_path):
    old_path = tmp_path / "Old.fbx"
    new_path = tmp_path / "New.fbx"
    documents = DocumentRegistry()
    selection = SelectionService()
    service = AssetMutationService(documents, selection)
    document = documents.create(
        DocumentKind.GENERIC,
        "Old",
        key=DocumentKey.resource(DocumentKind.GENERIC, str(old_path)),
        resource_path=str(old_path),
    )
    target = SelectionTarget.asset_subresource(
        "model-guid",
        "mesh-0",
        sub_kind="mesh",
    )
    selection.select(target, owner_id="project", record_history=False)

    service.publish_move(str(old_path), str(new_path), guid="model-guid")

    assert document.key == DocumentKey.resource(DocumentKind.GENERIC, str(new_path))
    assert selection.snapshot.primary == SelectionTarget.asset_subresource(
        "model-guid",
        "mesh-0",
        sub_kind="mesh",
    )


def test_particle_artifact_index_follows_global_asset_move(tmp_path):
    previous_root = get_project_root()
    set_project_root(str(tmp_path))
    try:
        artifact_root = tmp_path / "Library" / "Artifacts" / "Particle"
        artifact_root.mkdir(parents=True)
        index_path = artifact_root / PARTICLE_RUNTIME_INDEX_FILENAME
        index_path.write_text(
            json.dumps(
                {
                    "$schema": PARTICLE_RUNTIME_INDEX_SCHEMA,
                    "entries": [
                        {
                            "guid": "graph-guid",
                            "path_hint": "Assets/Old.particlegraph",
                            "stable_id": "stable-graph",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        old_path = tmp_path / "Assets" / "Old.particlegraph"
        new_path = tmp_path / "Assets" / "New.particlegraph"

        ParticleArtifactRegistry.remap_source(
            str(old_path),
            str(new_path),
            guid="graph-guid",
        )

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert payload["entries"] == [
            {
                "guid": "graph-guid",
                "path_hint": "Assets/New.particlegraph",
                "stable_id": "stable-graph",
            }
        ]
    finally:
        ParticleArtifactRegistry.clear()
        set_project_root(previous_root)


def test_document_move_is_rejected_while_save_is_pending(tmp_path):
    old_path = tmp_path / "Old.scene"
    new_path = tmp_path / "New.scene"
    documents = DocumentRegistry()
    selection = SelectionService()
    service = AssetMutationService(documents, selection)
    document = documents.create(
        DocumentKind.SCENE,
        "Old",
        key=DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=str(old_path),
    )
    documents.begin_save(document.document_id)

    with pytest.raises(RuntimeError, match="while document is saving"):
        service.publish_move(str(old_path), str(new_path), guid="scene-guid")

    assert path_key(document.resource_path) == path_key(old_path)


def test_project_move_preflights_before_touching_the_workspace(tmp_path):
    class _Database:
        @staticmethod
        def get_guid_from_path(_path):
            return "scene-guid"

    old_path = tmp_path / "Old.scene"
    new_path = tmp_path / "New.scene"
    old_path.write_text("{}", encoding="utf-8")
    documents = DocumentRegistry()
    service = AssetMutationService(documents, SelectionService())
    document = documents.create(
        DocumentKind.SCENE,
        "Old",
        key=DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=str(old_path),
    )
    documents.begin_save(document.document_id)

    with pytest.raises(RuntimeError, match="while document is saving"):
        project_file_ops.move_path(str(old_path), str(new_path), _Database())

    assert old_path.exists()
    assert not new_path.exists()


def test_undo_command_and_journal_share_operation_identity():
    manager = UndoManager()
    command = LambdaCommand("Identity", lambda: None, lambda: None)

    assert manager.execute(command)

    assert manager.action_journal.entries[0].operation_id == command.operation_id
