from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from Infernux.host import Operation, OperationRegistry, install_editor_operations


def test_editor_authoring_operations_exist_without_mcp_plugin(tmp_path):
    registry = OperationRegistry()
    operation_ids = install_editor_operations(str(tmp_path), registry)

    assert len(operation_ids) == 34
    assert "infernux.scene.component.schema" in operation_ids
    assert "infernux.scene.component.property.set" in operation_ids
    assert "infernux.asset.inspect" in operation_ids
    assert "infernux.asset.model.inspect" in operation_ids
    assert "infernux.asset.model.material.extract" in operation_ids
    assert "infernux.scene.model.instantiate" in operation_ids
    assert "infernux.asset.text.set" in operation_ids
    assert "infernux.data_asset.inspect" in operation_ids
    assert "infernux.data_asset.schema" in operation_ids
    assert "infernux.data_asset.property.set" in operation_ids
    assert {registry.get(name).owner for name in operation_ids} == {
        "infernux/engine"
    }
    for name in operation_ids:
        schema = registry.get(name).schema
        assert schema.thread == "owner"
        assert schema.availability == ("editor",)
        assert schema.phase in {"editor.read", "editor.authoring"}
        assert schema.output_schema["properties"]
        assert schema.output_schema["required"]


def test_transport_projection_cannot_replace_engine_owner(tmp_path):
    registry = OperationRegistry()
    operation_ids = install_editor_operations(str(tmp_path), registry)
    operation_id = operation_ids[0]
    authoritative = registry.get(operation_id)

    registry.register(
        Operation(
            schema=authoritative.schema,
            handler=lambda: {"transport": True},
            owner="infernux/mcp",
        )
    )

    retained = registry.get(operation_id)
    assert retained is authoritative
    assert retained.owner == "infernux/engine"


def test_model_inspection_rejects_non_model_before_loading(monkeypatch):
    from Infernux.host import asset_operations as operations
    from Infernux.core.mesh import Mesh
    from Infernux.host.operations import OperationError

    monkeypatch.setattr(operations, "on_editor", lambda _name, callback: callback())
    monkeypatch.setattr(operations, "asset_path", lambda _: "Assets/Readme.txt")
    monkeypatch.setattr(Mesh, "load_guid", lambda _: pytest.fail("non-model was loaded"))
    with pytest.raises(OperationError, match="requires a mesh asset"):
        operations._inspect_model("text-guid")


def test_material_extraction_uses_project_command_and_declared_schema(tmp_path, monkeypatch):
    from Infernux.host import asset_operations as operations

    calls = []
    monkeypatch.setattr(operations, "on_editor", lambda _name, callback: callback())
    monkeypatch.setattr(operations, "asset_path", lambda _: "Assets/Source.glb")
    monkeypatch.setattr(operations, "asset_identity", lambda path: {"path": path, "guid": "material-guid"})

    def extract(guid, slot, destination):
        calls.append((guid, slot, destination))
        return destination

    monkeypatch.setattr(operations.EditorAutomationHost, "instance",
                        lambda: SimpleNamespace(extract_model_material=extract))
    operation = next(value for value in operations.build_asset_operations(str(tmp_path))
                     if value.schema.id == "infernux.asset.model.material.extract")
    result = operation.handler(asset_guid="model-guid", slot=1, destination="Assets/Green.mat")
    assert calls == [("model-guid", 1, os.path.join(str(tmp_path), "Assets/Green.mat"))]
    assert result["source_slot"] == 1 and result["source_guid"] == "model-guid"
    assert set(result) == set(operation.schema.output_schema["required"])


def test_scene_open_reports_scheduling_and_schema_matches(monkeypatch):
    from Infernux.host import scene_operations as operations

    monkeypatch.setattr(operations, "on_editor", lambda _name, callback: callback())
    monkeypatch.setattr(operations, "asset_path", lambda *_args, **_kwargs: "Assets/TankBattle.scene")
    monkeypatch.setattr(operations.EditorAutomationHost, "instance", lambda: SimpleNamespace(open_scene=lambda _path: True))
    operation = next(value for value in operations.build_scene_operations()
                     if value.schema.id == "infernux.scene.open")

    result = operation.handler(asset_guid="scene-guid")

    assert result == {"asset_guid": "scene-guid", "path": "Assets/TankBattle.scene", "scheduled": True}
    assert set(operation.schema.output_schema["required"]) == set(result)
    assert "opened" not in operation.schema.output_schema["properties"]


def test_model_instantiation_routes_guid_to_shared_scene_mutation(monkeypatch):
    from Infernux.host import scene_operations as operations

    calls = []
    parent = SimpleNamespace(id=41)
    created = SimpleNamespace(id=73, name="Authored Model", get_parent=lambda: parent)
    host = SimpleNamespace(
        instantiate_scene_model=lambda guid, parent_id, name: calls.append(
            (guid, parent_id, name)
        ) or created
    )
    monkeypatch.setattr(operations, "on_editor", lambda _name, callback: callback())
    monkeypatch.setattr(operations, "asset_path", lambda _guid: "Assets/Models/Assembly.blend")
    monkeypatch.setattr(operations.EditorAutomationHost, "instance", lambda: host)
    operation = next(
        value for value in operations.build_scene_operations()
        if value.schema.id == "infernux.scene.model.instantiate"
    )

    result = operation.handler(
        asset_guid="model-guid", parent_id=41, name="Authored Model"
    )

    assert calls == [("model-guid", 41, "Authored Model")]
    assert result == {
        "id": 73,
        "name": "Authored Model",
        "asset_guid": "model-guid",
        "parent_id": 41,
    }
    assert set(operation.schema.output_schema["required"]) == set(result)


def test_asset_listing_uses_filesystem_identity_for_root_membership(tmp_path, monkeypatch):
    from Infernux.host import asset_operations as operations
    from Infernux.engine.path_utils import resolved_path

    project = tmp_path / "Project"
    paths = [project / root / "item.txt" for root in ("Assets", "Packages", "AssetsOther")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(operations, "on_editor", lambda name, callback: callback())
    monkeypatch.setattr(operations, "asset_database", lambda: SimpleNamespace(
        get_all_asset_paths=lambda: [str(path) for path in paths], asset_count=3))
    monkeypatch.setattr(operations, "asset_identity", lambda path: {"path": path})
    root = str(project).swapcase() if os.name == "nt" else str(project)

    result = operations._list_assets(root, root="all")
    assert result["assets"] == [{"path": resolved_path(path)} for path in paths[:2]]
    assert result["catalog_count"] == 2
    assert result["global_catalog_count"] == 3


def test_asset_creation_rejects_directory_escape_before_calling_authoring(tmp_path, monkeypatch):
    from Infernux.host import asset_operations as operations
    from Infernux.host import OperationError

    project = tmp_path / "Project"
    (project / "Assets").mkdir(parents=True)
    monkeypatch.setattr(operations, "on_editor", lambda name, callback: callback())
    with pytest.raises(OperationError, match="inside the project"):
        operations._create_asset(str(project), "folder", "Assets/../../Outside", "new")
