from __future__ import annotations

import json

import pytest

from Infernux.engine.model_instance_sync import (
    _matrix_trs,
    reconcile_scene_document_model_instances,
    reconcile_scene_document_model_source_graphs,
)
from Infernux.engine.runtime_scene_transaction import SceneDocumentTransaction


class _Meta:
    def __init__(self, manifest, *, generate_colliders=False):
        self._manifest = manifest
        self._generate_colliders = generate_colliders

    def has_key(self, key):
        return key == "model_meshes"

    def get_string(self, key):
        assert key == "model_meshes"
        return json.dumps(self._manifest)

    def serialize_document(self):
        return {
            "metadata": {
                "generate_colliders": {
                    "type": "bool",
                    "value": self._generate_colliders,
                }
            }
        }


class _Database:
    def __init__(self, manifest, *, generate_colliders=False):
        self._meta = _Meta(manifest, generate_colliders=generate_colliders)

    def get_meta_by_guid(self, guid):
        assert guid == "model-guid"
        return self._meta


def _renderer(identifier: str, path: list[str]):
    return {
        "type_id": "native:infernux.MeshRenderer",
        "data": {
            "meshAssetGuid": "model-guid",
            "modelSubresourceId": identifier,
            "modelNodePath": path,
        },
    }


def _node(name: str, identifier: str, path: list[str], *, components=(), children=()):
    return {
        "name": name,
        "components": [_renderer(identifier, path), *components],
        "children": list(children),
        "model_source": {"guid": "model-guid", "path": path},
    }


def _transform(component_id: int, position=(0.0, 0.0, 0.0)):
    return {
        "component_id": component_id,
        "type": "Transform",
        "enabled": True,
        "execution_order": 0,
        "position": list(position),
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def _scene_node(name, object_id, path, *, components=(), children=()):
    return {
        "id": object_id,
        "name": name,
        "active": True,
        "is_static": False,
        "tag": "Untagged",
        "layer": 0,
        "transform": _transform(object_id + 100, (7.0, 8.0, 9.0)),
        "components": list(components),
        "children": list(children),
        "model_source": {"guid": "model-guid", "path": list(path)},
    }


def _root(*children):
    return _scene_node("Instance", 1, [], children=children)


def _source(name, parent_index, node_group, *, tx=0.0):
    return {
        "name": name,
        "parent_index": parent_index,
        "node_group": node_group,
        "visible": True,
        "local_matrix": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_source_matrix_decomposition_preserves_translation_rotation_and_scale():
    position, rotation, scale = _matrix_trs(
        [
            [0.0, -3.0, 0.0, 4.0],
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 4.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert position == pytest.approx([4.0, 5.0, 6.0])
    assert scale == pytest.approx([2.0, 3.0, 4.0])
    assert rotation == pytest.approx([0.0, 0.0, 90.0], abs=1.0e-4)


def test_surviving_identity_updates_cached_path_before_native_commit():
    node = _node("AuthorName", "stable", ["OldParent", "OldName"])
    document = {"objects": [node]}

    changed = reconcile_scene_document_model_instances(
        document,
        _Database([{"path": ["NewParent", "NewName"], "subresource_id": "stable"}]),
    )

    assert changed == 1
    assert node["components"][0]["data"]["modelNodePath"] == ["NewParent", "NewName"]
    # The live synchronizer uses the old source edge to distinguish a source
    # reparent from an authored hierarchy edit.
    assert node["model_source"]["path"] == ["OldParent", "OldName"]


def test_deleted_generated_identity_is_removed_even_when_path_is_recreated():
    old = _node("AddedFromBlender", "old-id", ["Root", "AddedFromBlender"])
    document = {"objects": [old]}

    changed = reconcile_scene_document_model_instances(
        document,
        _Database(
            [
                {
                    "path": ["Root", "AddedFromBlender"],
                    "subresource_id": "new-id",
                }
            ]
        ),
    )

    assert changed == 1
    assert document["objects"] == []


def test_deleted_generated_parent_does_not_delete_surviving_child_identity():
    child = _node("MovedChild", "child-id", ["OldParent", "MovedChild"])
    parent = _node(
        "OldParent",
        "deleted-parent-id",
        ["OldParent"],
        children=(child,),
    )
    document = {"objects": [parent]}

    changed = reconcile_scene_document_model_instances(
        document,
        _Database(
            [
                {"path": ["NewParent", "MovedChild"], "subresource_id": "child-id"},
            ]
        ),
    )

    assert changed == 2
    assert document["objects"] == [child]
    assert child["components"][0]["data"]["modelNodePath"] == [
        "NewParent",
        "MovedChild",
    ]
    assert child["model_source"]["path"] == ["OldParent", "MovedChild"]


def test_deleted_identity_keeps_authored_object_but_retires_imported_geometry():
    authored = {"type_id": "python:gameplay.Health", "data": {"value": 10}}
    collider = {"type_id": "native:infernux.MeshCollider", "data": {}}
    old = _node(
        "AuthoredInstance",
        "deleted-id",
        ["Root", "Deleted"],
        components=(collider, authored),
    )
    document = {"objects": [old]}

    changed = reconcile_scene_document_model_instances(
        document,
        _Database([]),
    )

    assert changed == 3
    assert document["objects"] == [old]
    assert old["components"] == [authored]
    assert "model_source" not in old


def test_duplicate_imported_identity_fails_before_native_commit():
    document = {"objects": [_node("Mesh", "duplicate", ["Root", "Mesh"])]}
    database = _Database(
        [
            {"path": ["Root", "A"], "subresource_id": "duplicate"},
            {"path": ["Root", "B"], "subresource_id": "duplicate"},
        ]
    )

    with pytest.raises(ValueError, match="stable identity is duplicated"):
        reconcile_scene_document_model_instances(document, database)


@pytest.mark.parametrize(
    "manifest, diagnostic",
    [
        ([None], "entries must be objects"),
        ([{"path": ["Root", "Mesh"], "subresource_id": ""}], "require identity and path"),
        (
            [
                {"path": ["Root", "Mesh"], "subresource_id": "first"},
                {"path": ["Root", "Mesh"], "subresource_id": "second"},
            ],
            "source path is duplicated",
        ),
    ],
)
def test_malformed_manifest_fails_instead_of_retiring_scene_content(manifest, diagnostic):
    old = _node("Mesh", "old-id", ["Root", "Mesh"])
    document = {"objects": [old]}

    with pytest.raises(ValueError, match=diagnostic):
        reconcile_scene_document_model_instances(document, _Database(manifest))

    assert document["objects"] == [old]


def test_scene_transaction_retires_deleted_identity_before_native_commit(scene):
    existing = scene.create_game_object("ExistingWorldObject")
    stale = scene.create_game_object("DeletedModelNode")
    stale.add_component("MeshRenderer")
    document = scene.serialize_document()
    stale_document = next(node for node in document["objects"] if node["name"] == stale.name)
    renderer = next(
        component
        for component in stale_document["components"]
        if component["type_id"] == "native:infernux.MeshRenderer"
    )
    renderer["data"].update(
        {
            "meshAssetGuid": "model-guid",
            "modelSubresourceId": "deleted-id",
            "modelNodePath": ["Root", "DeletedModelNode"],
            "useInlineMesh": False,
        }
    )
    stale_document["model_source"] = {
        "guid": "model-guid",
        "path": ["Root", "DeletedModelNode"],
    }
    scene.destroy_game_object(stale)
    scene.process_pending_destroys()

    transaction = SceneDocumentTransaction(
        scene,
        document=document,
        asset_database=_Database(
            [{"path": ["Root", "DeletedModelNode"], "subresource_id": "new-id"}]
        ),
    )

    assert transaction.run_to_completion(raise_on_failure=False) is True
    assert transaction.document_reconciliation_count == 1
    assert scene.find("ExistingWorldObject") is not existing
    assert scene.find("DeletedModelNode") is None


def test_source_graph_reparent_and_pivot_rename_are_atomic_and_preserve_authored_state(monkeypatch):
    authored = {"type_id": "python:gameplay.Marker", "data": {"value": 9}}
    renderer = _renderer("stable", ["OldPivot", "Mesh"])
    renderer["data"]["materials"] = ["authored-material"]
    mesh = _scene_node(
        "Artist Mesh Name", 5, ["OldPivot", "Mesh"],
        components=(renderer, authored),
    )
    pivot = _scene_node("OldPivot", 3, ["OldPivot"], children=(mesh,))
    root = _root(pivot)
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (
            _source("NewPivot", -1, -1),
            _source("Mesh", 0, 0),
        ),
    )

    changed = reconcile_scene_document_model_source_graphs(
        document,
        _Database([{"path": ["NewPivot", "Mesh"], "subresource_id": "stable"}]),
    )

    assert changed >= 3
    assert root["children"] == [pivot]
    assert pivot["name"] == "NewPivot"
    assert pivot["model_source"]["path"] == ["NewPivot"]
    assert pivot["children"] == [mesh]
    assert mesh["name"] == "Artist Mesh Name"
    assert mesh["transform"]["position"] == [7.0, 8.0, 9.0]
    assert mesh["components"][-1] is authored
    assert renderer["data"]["materials"] == ["authored-material"]
    assert renderer["data"]["modelNodePath"] == ["NewPivot", "Mesh"]


def test_source_graph_creates_new_nodes_before_commit_with_stable_identity(monkeypatch):
    root = _root()
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (
            _source("Pivot", -1, -1, tx=2.0),
            _source("Added", 0, 4, tx=3.0),
        ),
    )

    changed = reconcile_scene_document_model_source_graphs(
        document,
        _Database([{"path": ["Pivot", "Added"], "subresource_id": "new-stable"}]),
    )

    assert changed >= 2
    pivot = root["children"][0]
    added = pivot["children"][0]
    assert pivot["model_source"]["path"] == ["Pivot"]
    assert pivot["transform"]["position"] == [2.0, 0.0, 0.0]
    assert added["model_source"]["path"] == ["Pivot", "Added"]
    assert added["transform"]["position"] == [3.0, 0.0, 0.0]
    data = added["components"][0]["data"]
    assert data["modelSubresourceId"] == "new-stable"
    assert data["modelNodePath"] == ["Pivot", "Added"]
    assert len({root["id"], pivot["id"], added["id"], pivot["transform"]["component_id"],
                added["transform"]["component_id"], added["components"][0]["component_id"]}) == 6


def test_source_graph_new_geometry_honors_generate_colliders(monkeypatch):
    root = _root()
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (_source("Mesh", -1, 0),),
    )

    reconcile_scene_document_model_source_graphs(
        document,
        _Database(
            [{"path": ["Mesh"], "subresource_id": "stable"}],
            generate_colliders=True,
        ),
    )

    assert [component["type_id"] for component in root["children"][0]["components"]] == [
        "native:infernux.MeshRenderer",
        "native:infernux.MeshCollider",
    ]


def test_same_path_recreated_geometry_is_new_object_not_path_rebound(monkeypatch):
    old = _scene_node("Mesh", 7, ["Mesh"], components=(_renderer("old-id", ["Mesh"]),))
    root = _root(old)
    document = {"objects": [root]}
    reconcile_scene_document_model_instances(
        document,
        _Database([{"path": ["Mesh"], "subresource_id": "new-id"}]),
    )
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (_source("Mesh", -1, 0),),
    )

    reconcile_scene_document_model_source_graphs(
        document,
        _Database([{"path": ["Mesh"], "subresource_id": "new-id"}]),
    )

    assert len(root["children"]) == 1
    current = root["children"][0]
    assert current["id"] != 7
    assert current["components"][0]["data"]["modelSubresourceId"] == "new-id"


def test_authored_scene_reparent_is_preserved_while_source_binding_updates(monkeypatch):
    mesh = _scene_node("Mesh", 7, ["Old", "Mesh"], components=(_renderer("stable", ["Old", "Mesh"]),))
    authored_parent = _scene_node("Authored Parent", 3, ["Old"], components=(
        {"type_id": "python:gameplay.Anchor", "data": {}},
    ), children=(mesh,))
    # The mesh source path says Old/Mesh, but its actual parent deliberately
    # does not own source path Old: this is an authored hierarchy override.
    authored_parent["model_source"]["path"] = ["Different"]
    root = _root(authored_parent)
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (
            _source("New", -1, -1),
            _source("Mesh", 0, 0),
        ),
    )

    reconcile_scene_document_model_source_graphs(
        document,
        _Database([{"path": ["New", "Mesh"], "subresource_id": "stable"}]),
    )

    assert mesh in authored_parent["children"]
    assert mesh["model_source"]["path"] == ["New", "Mesh"]
    assert mesh["components"][0]["data"]["modelNodePath"] == ["New", "Mesh"]


def test_source_reparent_between_existing_pivots_does_not_rename_the_old_pivot(monkeypatch):
    mesh = _scene_node("Mesh", 7, ["Left", "Mesh"], components=(_renderer("stable", ["Left", "Mesh"]),))
    left = _scene_node("Left", 3, ["Left"], children=(mesh,))
    right = _scene_node("Right", 5, ["Right"])
    root = _root(left, right)
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (
            _source("Left", -1, -1),
            _source("Right", -1, -1),
            _source("Mesh", 1, 0),
        ),
    )

    reconcile_scene_document_model_source_graphs(
        document,
        _Database([{"path": ["Right", "Mesh"], "subresource_id": "stable"}]),
    )

    assert left["name"] == "Left"
    assert right["name"] == "Right"
    assert mesh not in left["children"]
    assert mesh in right["children"]


def test_deleted_generated_pivot_hoists_authored_child(monkeypatch):
    authored_child = {
        "id": 9,
        "name": "Gameplay",
        "active": True,
        "is_static": False,
        "tag": "Untagged",
        "layer": 0,
        "transform": _transform(109),
        "components": [{"type_id": "python:gameplay.Marker", "data": {}}],
        "children": [],
    }
    old_pivot = _scene_node("DeletedPivot", 3, ["DeletedPivot"], children=(authored_child,))
    root = _root(old_pivot)
    document = {"objects": [root]}
    monkeypatch.setattr(
        "Infernux.engine.model_instance_sync._load_model_source_nodes",
        lambda guid: (_source("CurrentPivot", -1, -1),),
    )

    reconcile_scene_document_model_source_graphs(document, _Database([]))

    assert authored_child in root["children"]
    assert all(child is not old_pivot for child in root["children"])
    assert any(child.get("name") == "CurrentPivot" for child in root["children"])
