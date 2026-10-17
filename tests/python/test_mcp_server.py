from __future__ import annotations

import importlib
import json
import inspect
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from infernux.host import OperationRegistry
from infernux_mcp import server
from infernux_mcp import capabilities
from infernux_mcp.adapter import (
    adapter_status,
    register_gateways,
    shutdown_adapter,
)


class _FakeMCP:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self, *args, **kwargs):
        name = kwargs.get("name") or (args[0] if args else "")

        def decorate(fn):
            self.tools[str(name or fn.__name__)] = fn.__wrapped__ if inspect.iscoroutinefunction(fn) else fn
            return fn

        return decorate


def test_loaded_scene_query_reports_every_resident_scene_and_active_owner(scene):
    from infernux.lib import SceneManager

    manager = SceneManager.instance()
    additive = manager.create_scene("McpLoadedAdditive")
    additive.create_game_object("AdditiveMarker")
    try:
        manager.set_active_scene(additive)
        from infernux.host import scene_operations as engine_scene_operations

        snapshot = engine_scene_operations._loaded_scene_snapshot()
        by_world = {item["world_id"]: item for item in snapshot["scenes"]}

        assert int(scene.world_id) in by_world
        assert int(additive.world_id) in by_world
        assert by_world[int(additive.world_id)]["name"] == "McpLoadedAdditive"
        assert by_world[int(additive.world_id)]["root_count"] == 1
        assert by_world[int(additive.world_id)]["active"] is True
        assert snapshot["active_world_id"] == int(additive.world_id)
    finally:
        manager.set_active_scene(scene)
        manager.unload_scene(additive)


def test_default_grants_cover_every_operation_and_diff_is_machine_readable(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    mcp = _FakeMCP()
    try:
        # Omitting granted_capabilities keeps the default: every capability.
        register_gateways(mcp, str(tmp_path), {})
        config = capabilities.current_config()
        assert config["granted_capabilities"] == ["*"]

        snapshot = mcp.tools["host_capabilities"]()
        assert snapshot["ok"] is True
        data = snapshot["data"]
        assert data["blocked_operation_count"] == 0
        assert data["blocked_operations"] == []
        assert data["capabilities"]
        assert all(row["granted"] for row in data["capabilities"])
        assert str(data["config_path"]).endswith("mcp_capabilities.json")
    finally:
        shutdown_adapter()


def test_read_only_grants_block_writes_with_grant_remediation(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": ["*.read"]})

        snapshot = mcp.tools["host_capabilities"]()
        assert snapshot["ok"] is True
        data = snapshot["data"]
        assert data["blocked_operation_count"] > 0
        blocked_ids = {entry["operation"] for entry in data["blocked_operations"]}
        assert "infernux.scene.object.create" in blocked_ids
        rows = {row["capability"]: row for row in data["capabilities"]}
        assert rows["scene.read"]["granted"] is True
        assert rows["scene.write"]["granted"] is False

        # Pattern grants keep read operations available.
        allowed = mcp.tools["operation_query_execute"]("infernux.mcp.checkpoint.list", {})
        assert allowed["ok"] is True

        denied = mcp.tools["operation_command_execute"](
            "infernux.scene.object.create", {"kind": "empty", "name": "Blocked"}
        )
        assert denied["ok"] is False
        assert denied["error"]["code"] == "operation.permission_denied"
        details = denied["error"]["details"]
        assert details["required"] == ["scene.write"]
        assert "*.read" in details["granted"]
        remediation = details["grant_remediation"]
        assert remediation["config_pointer"] == "/granted_capabilities"
        assert str(remediation["config_path"]).endswith("mcp_capabilities.json")
    finally:
        shutdown_adapter()


def test_schema_search_and_execution_use_formal_operation_ids(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)})
        expected = {
            "project": "infernux.project.info",
            "checkpoint": "infernux.mcp.checkpoint.list",
            "attempt": "infernux.mcp.attempt.start",
            "shutdown": "infernux.mcp.supervisor.shutdown",
        }
        for query, operation in expected.items():
            result = mcp.tools["operation_schema_search"](query, 200)
            assert result["ok"] is True
            assert operation in {item["id"] for item in result["data"]["operations"]}

        missing_search = mcp.tools["operation_schema_search"]("missing.operation", 200)
        assert missing_search["ok"] is True
        assert missing_search["data"]["operations"] == []

        result = mcp.tools["operation_query_execute"]("infernux.mcp.checkpoint.list", {})
        assert result["ok"] is True
        assert "checkpoints" in result["data"]["result"]
        missing_execute = mcp.tools["operation_query_execute"]("missing.operation", {})
        assert missing_execute["ok"] is False
        assert missing_execute["error"]["code"] == "operation.not_found"
        mismatch = mcp.tools["operation_command_execute"]("infernux.mcp.checkpoint.list", {})
        assert mismatch["ok"] is False
        assert mismatch["error"]["code"] == "operation.kind_mismatch"
        rejected = mcp.tools["operation_command_execute"](
            "infernux.mcp.supervisor.shutdown",
            {"lease_token": "not-a-live-lease"},
        )
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "mcp.supervisor_lease"
        host_status = mcp.tools["host_session_status"]()
        assert host_status["ok"] is True
        assert host_status["data"]["session"]["project_root"] == str(tmp_path)
        guidance = host_status["data"]["workflow_guidance"]
        assert "os_foreground_control" not in guidance
        assert "render-target capture" in guidance["visual_capture"]
        assert "Scene, Game, or Player" in guidance["visual_capture"]
    finally:
        shutdown_adapter()


def test_schema_gateway_can_create_and_transform_real_scene_object(tmp_path, scene):
    from infernux.engine.hierarchy_creation_service import HierarchyCreationService
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    previous_creation = HierarchyCreationService._instance
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    creation = HierarchyCreationService()
    HierarchyCreationService._instance = creation
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)})
        created = mcp.tools["operation_command_execute"](
            "infernux.scene.object.create",
            {"kind": "empty", "name": "SchemaCreated"},
        )
        assert created["ok"] is True
        object_id = created["data"]["result"]["id"]
        assert scene.find_by_id(object_id).name == "SchemaCreated"

        transformed = mcp.tools["operation_command_execute"](
            "infernux.scene.object.transform.set",
            {
                "object_id": object_id,
                "position": [1.0, 2.0, 3.0],
                "rotation": [10.0, 20.0, 30.0],
                "scale": [2.0, 2.0, 2.0],
            },
        )
        assert transformed["ok"] is True

        hierarchy = mcp.tools["operation_query_execute"](
            "infernux.scene.hierarchy.get",
            {},
        )
        assert hierarchy["ok"] is True
        authored = next(item for item in hierarchy["data"]["result"]["objects"] if item["id"] == object_id)
        assert authored["transform"] == {
            "position": [1.0, 2.0, 3.0],
            "rotation": [10.0, 20.0, 30.0],
            "scale": [2.0, 2.0, 2.0],
        }
        assert len(undo.action_journal.applied_entries()) == 2
    finally:
        shutdown_adapter()
        core.shutdown()
        UndoManager._instance = previous_undo
        HierarchyCreationService._instance = previous_creation
        queue.release_owner("MCP editing operation test finished")


@pytest.mark.parametrize("invalid", [True, False])
def test_schema_gateway_transform_rejection_does_not_masquerade_as_no_change(scene, tmp_path, invalid):
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    owner = scene.create_game_object("Rejected Transform")
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)})
        before = owner.serialize_document()
        if not invalid:
            undo.enabled = False
        rejected = mcp.tools["operation_command_execute"](
            "infernux.scene.object.transform.set",
            {
                "object_id": owner.id,
                "position": [float("nan") if invalid else 5, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
            },
        )
        assert not rejected["ok"], rejected
        assert rejected["error"]["code"] == "scene.edit_rejected", rejected
        assert owner.serialize_document() == before
        assert not undo.action_journal.applied_entries()
    finally:
        shutdown_adapter()
        core.shutdown()
        UndoManager._instance = previous_undo
        queue.release_owner("MCP rejected transform test finished")


@pytest.mark.parametrize(
    "kind,invalid", [("native", True), ("python", True), ("render_stack", True), ("native", False)]
)
def test_schema_gateway_component_field_rejection_is_not_no_change(scene, tmp_path, kind, invalid):
    from infernux.components import InxComponent, serialized_field
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.renderstack import RenderStack

    class McpRejectedFieldProbe(InxComponent):
        speed: float = serialized_field(default=3.0, range=(0.0, 20.0))

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    owner = scene.create_game_object("MCP Rejected Component Field")
    if kind == "native":
        component = owner.add_component("Light")
        field, value = "intensity", "invalid" if invalid else 2.5
    elif kind == "python":
        component = owner.add_py_component(McpRejectedFieldProbe())
        field, value = "speed", "invalid" if invalid else 8.0
    else:
        component = owner.add_py_component(RenderStack())
        field, value = "pipeline_class_name", 123 if invalid else "Default Deferred"
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)})
        before = owner.serialize_document()
        revision = core.action_journal.revision
        if not invalid:
            undo.enabled = False
        rejected = mcp.tools["operation_command_execute"](
            "infernux.scene.component.property.set",
            {"object_id": owner.id, "component_id": component.component_id, "field": field, "value": value},
        )
        assert not rejected["ok"], rejected
        assert owner.serialize_document() == before
        assert core.action_journal.revision == revision
        assert not core.action_journal.applied_entries()
    finally:
        shutdown_adapter()
        core.shutdown()
        UndoManager._instance = previous_undo
        queue.release_owner("MCP rejected component field test finished")


def test_schema_gateway_native_and_python_fields_share_constraints_and_undo(scene, tmp_path):
    from infernux.components import InxComponent, serialized_field
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue

    class McpPythonFieldProbe(InxComponent):
        speed: float = serialized_field(default=3.0, range=(0.0, 20.0))

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    owner = scene.create_game_object("MCP Field Authority")
    native = owner.add_component("Light")
    python_component = owner.add_py_component(McpPythonFieldProbe())
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)})

        native_schema = mcp.tools["operation_query_execute"](
            "infernux.scene.component.schema",
            {"object_id": owner.id, "component_id": native.component_id},
        )
        python_schema = mcp.tools["operation_query_execute"](
            "infernux.scene.component.schema",
            {
                "object_id": owner.id,
                "component_id": python_component.component_id,
            },
        )
        assert native_schema["ok"] is True
        assert python_schema["ok"] is True
        native_fields = {item["name"]: item for item in native_schema["data"]["result"]["fields"]}
        python_fields = {item["name"]: item for item in python_schema["data"]["result"]["fields"]}
        assert native_fields["intensity"]["range"] == [0.0, 10.0]
        assert python_fields["speed"]["range"] == [0.0, 20.0]

        native_change = mcp.tools["operation_command_execute"](
            "infernux.scene.component.property.set",
            {
                "object_id": owner.id,
                "component_id": native.component_id,
                "field": "intensity",
                "value": 2.5,
            },
        )
        python_change = mcp.tools["operation_command_execute"](
            "infernux.scene.component.property.set",
            {
                "object_id": owner.id,
                "component_id": python_component.component_id,
                "field": "speed",
                "value": 8.0,
            },
        )
        assert native_change["ok"] is True
        assert python_change["ok"] is True
        assert native.intensity == pytest.approx(2.5)
        assert python_component.speed == pytest.approx(8.0)
        assert len(undo.action_journal.applied_entries()) == 2

        undo.undo()
        assert python_component.speed == pytest.approx(3.0)
        undo.undo()
        assert native.intensity == pytest.approx(1.0)

        rejected = mcp.tools["operation_command_execute"](
            "infernux.scene.component.property.set",
            {
                "object_id": owner.id,
                "component_id": python_component.component_id,
                "field": "undeclared",
                "value": 1,
            },
        )
        assert rejected["ok"] is False
    finally:
        shutdown_adapter()
        core.shutdown()
        UndoManager._instance = previous_undo
        queue.release_owner("MCP field authority test finished")


def test_schema_gateway_can_edit_real_material_document(engine):
    from infernux.core.assets import AssetManager
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui import project_file_ops
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.plugins import PluginManager

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    name = f"McpMaterial_{uuid.uuid4().hex}"
    path = assets / f"{name}.mat"
    created, error = project_file_ops.create_material(
        str(assets),
        name,
        database,
    )
    assert created, error
    guid = database.get_guid_from_path(str(path))
    assert guid

    previous_plugins = PluginManager._instance
    previous_database = AssetManager._asset_database
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    core.project_assets.configure(str(project_root), database)
    undo = UndoManager(core.action_journal)
    manager = PluginManager(str(project_root), engine=engine)
    PluginManager._instance = manager
    AssetManager._asset_database = database
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(
            mcp, str(project_root), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)}
        )
        changed = mcp.tools["operation_command_execute"](
            "infernux.material.property.set",
            {
                "asset_guid": guid,
                "pointer": "/properties/baseColor/value",
                "value": [0.25, 0.5, 0.75, 1.0],
            },
        )
        assert changed["ok"] is True
        assert changed["data"]["result"]["document"]["properties"]["baseColor"]["value"] == [
            0.25,
            0.5,
            0.75,
            1.0,
        ]
        document = None
        for _ in range(100):
            AssetManager.poll_pending_asset_writes()
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except PermissionError:
                # Windows may briefly deny readers while the asynchronous
                # atomic publication replaces the destination file.
                time.sleep(0.01)
                continue
            if document["properties"]["baseColor"]["value"] == pytest.approx([0.25, 0.5, 0.75, 1.0]):
                break
            time.sleep(0.01)
        assert document is not None
        assert document["properties"]["baseColor"]["value"] == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert len(undo.action_journal.applied_entries()) == 1
        repeated = mcp.tools["operation_command_execute"](
            "infernux.material.property.set",
            {
                "asset_guid": guid,
                "pointer": "/shaders",
                "value": changed["data"]["result"]["document"]["shaders"],
            },
        )
        assert repeated["ok"] is True, repeated
        assert len(undo.action_journal.applied_entries()) == 1
        assert repeated["data"]["result"]["document"] == changed["data"]["result"]["document"]
        undo.enabled = False
        rejected = mcp.tools["operation_command_execute"](
            "infernux.material.property.set",
            {
                "asset_guid": guid,
                "pointer": "/properties/baseColor/value",
                "value": [0.8, 0.4, 0.6, 1.0],
            },
        )
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "material.edit_rejected"
        assert len(undo.action_journal.applied_entries()) == 1
    finally:
        shutdown_adapter()
        manager.shutdown()
        core.shutdown()
        UndoManager._instance = previous_undo
        PluginManager._instance = previous_plugins
        AssetManager._asset_database = previous_database
        queue.release_owner("MCP material editing operation test finished")
        database.delete_asset(str(path))
        path.unlink(missing_ok=True)
        Path(str(path) + ".meta").unlink(missing_ok=True)


def test_schema_gateway_can_edit_real_data_asset_document(engine):
    from infernux.components import serialized_field
    from infernux.core import AssetManager, DataAsset
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.plugins import PluginManager

    class McpDataAsset(DataAsset):
        __serialized_type_id__ = "tests.mcp.data_asset"

        speed: float = serialized_field(default=3.0, range=(0.0, 20.0))

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    path = assets / f"McpData_{uuid.uuid4().hex}.inxdata"
    source = McpDataAsset(speed=3.0)

    previous_plugins = PluginManager._instance
    previous_database = AssetManager._asset_database
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    core.project_assets.configure(str(project_root), database)
    undo = UndoManager(core.action_journal)
    manager = PluginManager(str(project_root), engine=engine)
    PluginManager._instance = manager
    AssetManager._asset_database = database
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        source.save_to(str(path), database=database)
        guid = source.guid
        register_gateways(
            mcp, str(project_root), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)}
        )

        schema = mcp.tools["operation_query_execute"]("infernux.data_asset.schema", {"asset_guid": guid})
        assert schema["ok"] is True
        speed_schema = next(
            field for field in schema["data"]["result"]["fields"] if field["property_path"] == "McpDataAsset.speed"
        )
        assert speed_schema["attributes"]["range"] == [0.0, 20.0]

        inspected = mcp.tools["operation_query_execute"]("infernux.data_asset.inspect", {"asset_guid": guid})
        assert inspected["ok"] is True
        assert inspected["data"]["result"]["document"]["fields"]["speed"] == 3.0

        changed = mcp.tools["operation_command_execute"](
            "infernux.data_asset.property.set",
            {
                "asset_guid": guid,
                "pointer": "/fields/speed",
                "value": 100.0,
            },
        )
        assert changed["ok"] is True, changed
        assert changed["data"]["result"]["document"]["fields"]["speed"] == 20.0
        for _ in range(100):
            AssetManager.poll_pending_asset_writes()
            document = json.loads(path.read_text(encoding="utf-8"))
            if document["fields"]["speed"] == pytest.approx(20.0):
                break
            time.sleep(0.01)
        assert document["fields"]["speed"] == pytest.approx(20.0)
        assert len(undo.action_journal.applied_entries()) == 1

        undo.undo()
        restored = mcp.tools["operation_query_execute"]("infernux.data_asset.inspect", {"asset_guid": guid})
        assert restored["ok"] is True
        assert restored["data"]["result"]["document"]["fields"]["speed"] == 3.0

        rejected = mcp.tools["operation_command_execute"](
            "infernux.data_asset.property.set",
            {
                "asset_guid": guid,
                "pointer": "/type_id",
                "value": "tests.replaced",
            },
        )
        assert rejected["ok"] is False
    finally:
        shutdown_adapter()
        manager.shutdown()
        core.shutdown()
        UndoManager._instance = previous_undo
        PluginManager._instance = previous_plugins
        AssetManager._asset_database = previous_database
        queue.release_owner("MCP DataAsset editing operation test finished")
        if database.contains_path(str(path)):
            database.delete_asset(str(path))
        path.unlink(missing_ok=True)
        Path(str(path) + ".meta").unlink(missing_ok=True)


def test_schema_gateway_can_validate_and_edit_real_particle_graph(engine):
    from infernux.core.assets import AssetManager
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui import project_file_ops
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.particle.asset import ParticleGraphAsset
    from infernux.plugins import PluginManager

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    name = f"McpParticle_{uuid.uuid4().hex}"
    path = assets / f"{name}.particlegraph"
    created, error = project_file_ops.create_particlegraph(
        str(assets),
        name,
        database,
    )
    assert created, error
    guid = database.get_guid_from_path(str(path))
    assert guid

    previous_plugins = PluginManager._instance
    previous_database = AssetManager._asset_database
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    core.project_assets.configure(str(project_root), database)
    undo = UndoManager(core.action_journal)
    manager = PluginManager(str(project_root), engine=engine)
    PluginManager._instance = manager
    AssetManager._asset_database = database
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(
            mcp, str(project_root), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)}
        )
        changed = mcp.tools["operation_command_execute"](
            "infernux.particle.graph.property.set",
            {
                "asset_guid": guid,
                "pointer": "/name",
                "value": "Schema Particle Graph",
            },
        )
        assert changed["ok"] is True
        result = changed["data"]["result"]
        assert result["document"]["name"] == "Schema Particle Graph"
        assert len(result["semantic_hash"]) == 64
        assert ParticleGraphAsset.load(str(path)).name == "Schema Particle Graph"
        assert len(undo.action_journal.applied_entries()) == 1
    finally:
        shutdown_adapter()
        manager.shutdown()
        core.shutdown()
        UndoManager._instance = previous_undo
        PluginManager._instance = previous_plugins
        AssetManager._asset_database = previous_database
        queue.release_owner("MCP Particle Graph editing operation test finished")
        database.delete_asset(str(path))
        path.unlink(missing_ok=True)
        Path(str(path) + ".meta").unlink(missing_ok=True)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_schema_gateway_reads_and_replaces_guid_text_asset_with_undo(engine, newline):
    from infernux.core.assets import AssetManager
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.plugins import PluginManager

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    path = assets / f"McpText_{uuid.uuid4().hex}.txt"
    original = ("before" + newline).encode("utf-8")
    path.write_bytes(original)
    guid = AssetManager.import_asset(str(path), database=database).guid

    previous_plugins = PluginManager._instance
    previous_database = AssetManager._asset_database
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    core.project_assets.configure(str(project_root), database)
    undo = UndoManager(core.action_journal)
    manager = PluginManager(str(project_root), engine=engine)
    PluginManager._instance = manager
    AssetManager._asset_database = database
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(
            mcp, str(project_root), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)}
        )
        read = mcp.tools["operation_query_execute"]("infernux.asset.text.read", {"asset_guid": guid})
        assert read["ok"] is True
        assert read["data"]["result"]["content"] == original.decode("utf-8")

        changed = mcp.tools["operation_command_execute"](
            "infernux.asset.text.set",
            {"asset_guid": guid, "content": "after\n"},
        )
        assert changed["ok"] is True, changed
        assert path.read_bytes() == b"after\n"
        assert len(undo.action_journal.applied_entries()) == 1

        undo.undo()
        assert path.read_bytes() == original
    finally:
        shutdown_adapter()
        manager.shutdown()
        core.shutdown()
        UndoManager._instance = previous_undo
        PluginManager._instance = previous_plugins
        AssetManager._asset_database = previous_database
        queue.release_owner("MCP text asset editing operation test finished")
        if database.contains_path(str(path)):
            database.delete_asset(str(path))
        path.unlink(missing_ok=True)
        Path(str(path) + ".meta").unlink(missing_ok=True)


def test_schema_gateway_can_create_move_and_delete_guid_asset(engine):
    from infernux.core.assets import AssetManager
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui import project_file_ops
    from infernux.engine.undo import UndoManager
    from infernux.host import MainThreadCommandQueue
    from infernux.plugins import PluginManager

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    name = f"McpAsset_{uuid.uuid4().hex}"
    original = assets / f"{name}.mat"
    moved = assets / f"{name}_Moved.mat"

    previous_plugins = PluginManager._instance
    previous_database = AssetManager._asset_database
    previous_undo = UndoManager._instance
    core = EditorInteractionCore()
    core.project_assets.configure(str(project_root), database)

    def create_asset(kind, directory, asset_name, _variant):
        assert kind == "material"
        return project_file_ops.create_material(directory, asset_name, database)

    core.project_asset_interactions.configure(
        unique_name=project_file_ops.get_unique_name,
        create=create_asset,
        open_asset=lambda *_args: True,
        reveal=lambda *_args: True,
        read_external_clipboard=lambda: (),
        request_delete=lambda paths, callback: callback(list(paths)),
    )
    undo = UndoManager(core.action_journal)
    manager = PluginManager(str(project_root), engine=engine)
    PluginManager._instance = manager
    AssetManager._asset_database = database
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    mcp = _FakeMCP()
    try:
        register_gateways(
            mcp, str(project_root), {"granted_capabilities": list(capabilities.DEFAULT_GRANTED_CAPABILITIES)}
        )
        kinds = mcp.tools["operation_query_execute"]("infernux.asset.create.kinds", {})
        assert kinds["ok"] is True
        assert {item["kind"] for item in kinds["data"]["result"]["kinds"]} >= {
            "data_asset",
            "material",
            "scene",
        }
        created = mcp.tools["operation_command_execute"](
            "infernux.asset.create",
            {
                "kind": "material",
                "directory": str(assets),
                "name": name,
            },
        )
        assert created["ok"] is True
        guid = created["data"]["result"]["asset"]["guid"]
        assert guid and original.is_file()

        moved_result = mcp.tools["operation_command_execute"](
            "infernux.asset.move",
            {"asset_guid": guid, "destination": str(moved)},
        )
        assert moved_result["ok"] is True
        assert moved_result["data"]["result"]["asset"]["guid"] == guid
        assert moved.is_file() and not original.exists()

        deleted = mcp.tools["operation_command_execute"](
            "infernux.asset.delete",
            {"asset_guids": [guid]},
        )
        assert deleted["ok"] is True
        assert not moved.exists()
        assert len(undo.action_journal.applied_entries()) == 3
    finally:
        shutdown_adapter()
        manager.shutdown()
        core.shutdown()
        UndoManager._instance = previous_undo
        PluginManager._instance = previous_plugins
        AssetManager._asset_database = previous_database
        queue.release_owner("MCP GUID asset operation test finished")
        for candidate in (original, moved):
            database.delete_asset(str(candidate))
            candidate.unlink(missing_ok=True)
            Path(str(candidate) + ".meta").unlink(missing_ok=True)


def test_server_start_rejects_occupied_port_without_false_loaded_state(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        reservation.listen(1)
        port = reservation.getsockname()[1]
        with pytest.raises(RuntimeError, match="failed before becoming ready"):
            server.start_server(str(tmp_path), port=port)
    assert server.is_running() is False
    assert adapter_status()["active"] is False


def test_direct_http_clients_connect_and_survive_editor_restart(tmp_path):
    import asyncio
    import httpx
    from fastmcp import Client
    from infernux.engine.project_context import get_project_root, using_project_root
    from infernux.host import Operation, OperationKind, OperationSchema
    from infernux_mcp.client import create_loopback_client

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = server.endpoint_url(port=port)
    health = server.health_url(port=port)
    blocked, release = threading.Event(), threading.Event()
    registry = OperationRegistry.instance()

    def slow_query():
        blocked.set()
        assert release.wait(10), "test did not release the gateway"
        return {"root": get_project_root()}

    async def verify():
        async with httpx.AsyncClient(trust_env=False) as http:
            # DNS-rebinding guard for /mcp: foreign Host and foreign browser origins.
            initialize = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
            accept = {"Accept": "application/json, text/event-stream"}
            rebinding = {**accept, "Host": f"attacker.example:{port}"}
            assert (await http.post(endpoint, headers=rebinding, json=initialize)).status_code == 421
            foreign = {**accept, "Origin": "https://attacker.example"}
            assert (await http.post(endpoint, headers=foreign, json=initialize)).status_code == 403
            assert (await http.get(health)).json() == {"name": "Infernux Editor", "ready": True}
            async with create_loopback_client(endpoint, timeout_seconds=20) as client:
                assert (await client.call_tool("mcp_ping")).data["ok"]
                query = asyncio.create_task(
                    client.call_tool("operation_query_execute", {"operation": "test.transport.slow"})
                )
                deadline = time.monotonic() + 5
                while not blocked.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert blocked.is_set()
                assert (await http.get(health, timeout=2)).status_code == 200
                assert not release.is_set()
                release.set()
                result = (await query).data
                assert result["ok"] and result["data"]["result"]["root"] == str(tmp_path)
            config = server.connection_info(port=port)["clients"]["generic"]["config"]
            assert config["mcpServers"]["infernux-editor"]["url"] == endpoint
            async with Client(config, timeout=20) as direct:
                assert (await direct.call_tool("mcp_ping")).data["ok"]
            server.stop_server()
            assert server.start_server(str(tmp_path), port=port)
            async with Client(config, timeout=20) as direct:
                assert (await direct.call_tool("mcp_ping")).data["ok"]

    try:
        with using_project_root(str(tmp_path)):
            assert server.start_server(str(tmp_path), port=port)
            registry.register(
                Operation(
                    OperationSchema(
                        id="test.transport.slow",
                        kind=OperationKind.QUERY,
                        summary="Block one gateway while testing health isolation.",
                        input_schema={"type": "object", "properties": {}},
                        output_schema={"type": "object"},
                        errors=(),
                        side_effects=(),
                        reversible=False,
                        cost={"class": "constant"},
                        thread="any",
                        capabilities=("project.read",),
                    ),
                    slow_query,
                    "transport-test",
                )
            )
            asyncio.run(verify())
    finally:
        release.set()
        server.stop_server()
        registry.unregister_owner("transport-test")


def test_server_hot_reload_keeps_one_owner_and_stops_without_error(tmp_path, monkeypatch):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    errors: list[str] = []
    monkeypatch.setattr(
        server.Debug,
        "log_error",
        lambda message, *args, **kwargs: errors.append(str(message)),
    )

    try:
        assert server.start_server(str(tmp_path), port=port) is True
        assert server.is_running() is True
        assert sum(thread.name == "InfernuxMCPHTTP" and thread.is_alive() for thread in threading.enumerate()) == 1

        sys.modules.pop(server.__name__, None)
        replacement = importlib.import_module(server.__name__)
        assert replacement is not server
        assert replacement.is_running() is True
        assert replacement.start_server(str(tmp_path), port=port) is True
        assert sum(thread.name == "InfernuxMCPHTTP" and thread.is_alive() for thread in threading.enumerate()) == 1

        replacement.stop_server()
        assert replacement.is_running() is False
        assert errors == []

        # Shutdown is synchronous: a following generation may bind the exact
        # endpoint immediately instead of colliding with the old transport.
        assert replacement.start_server(str(tmp_path), port=port) is True
        assert replacement.is_running() is True
    finally:
        server.stop_server()

    assert server.is_running() is False
    assert errors == []


def test_server_async_retirement_releases_owner_before_next_generation(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    try:
        assert server.start_server(str(tmp_path), port=port) is True
        server.request_stop_server()

        deadline = time.monotonic() + 10.0
        while server.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.is_running() is False

        assert server.start_server(str(tmp_path), port=port) is True
        assert server.is_running() is True
    finally:
        server.stop_server()


def test_server_shutdown_closes_connected_streamable_http_session(tmp_path):
    import asyncio
    from infernux_mcp.client import create_loopback_client

    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async def connected_shutdown():
        async with create_loopback_client(server.endpoint_url(port=port), timeout_seconds=20) as client:
            state = server._active_state()
            ping = await client.call_tool("mcp_ping")
            assert ping.data["ok"]
            assert ping.data["data"]["message"] == "pong"
            await asyncio.to_thread(server.stop_server)
            assert not server.is_running()
            assert state.error is None
            assert not (tmp_path / "mcp.json").exists()

    try:
        assert server.start_server(str(tmp_path), port=port)
        asyncio.run(connected_shutdown())
        assert server.start_server(str(tmp_path), port=port)
        assert server.is_running()
    finally:
        server.stop_server()


@pytest.mark.parametrize("cancelled_reply_delay", [0.0, 0.3])
def test_server_shutdown_does_not_wait_for_queued_editor_query(tmp_path, monkeypatch, cancelled_reply_delay):
    import asyncio
    from infernux.host import MainThreadCommandQueue
    from infernux.host.commands import CommandFuture
    from infernux_mcp.client import create_loopback_client

    original_result = CommandFuture.result

    def delayed_cancelled_reply(future, timeout=None):
        try:
            return original_result(future, timeout)
        except TimeoutError:
            # Model a worker that has observed cancellation but has not yet
            # returned its error response to the HTTP transport.
            time.sleep(cancelled_reply_delay)
            raise

    monkeypatch.setattr(CommandFuture, "result", delayed_cancelled_reply)
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    command_queue = MainThreadCommandQueue()
    command_queue.drain()
    monkeypatch.setattr(MainThreadCommandQueue, "_instance", command_queue)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async def queued_shutdown():
        async with create_loopback_client(server.endpoint_url(port=port), timeout_seconds=20) as client:
            # Tool-result validation lazily fetches schemas. Discover them while
            # the server is alive so shutdown cannot trigger a new tools/list.
            await client.list_tools()
            query = asyncio.create_task(
                client.call_tool("operation_query_execute", {"operation": "infernux.project.info"})
            )
            try:
                for _ in range(100):
                    if command_queue._queue.qsize():
                        break
                    await asyncio.sleep(0.01)
                assert command_queue._queue.qsize() == 1
                state = server._active_state()
                await asyncio.to_thread(server.stop_server)
                assert not server.is_running()
                assert state.error is None
                pending = list(command_queue._queue.queue)
                assert not any(future.can_execute() for _name, _fn, future in pending)
                result = await query
                assert not result.data["ok"]
            finally:
                command_queue.cancel_pending("test cleanup")
                (result,) = await asyncio.gather(query, return_exceptions=True)
                assert isinstance(result, BaseException) or not result.data["ok"]

    try:
        assert server.start_server(str(tmp_path), port=port)
        asyncio.run(queued_shutdown())
    finally:
        command_queue.cancel_pending("test cleanup")
        server.stop_server()


def test_failed_transport_retirement_does_not_publish_successful_unload(monkeypatch, tmp_path):
    from infernux.runtime_services import get_runtime_service, install_runtime_service, remove_runtime_service

    failure = RuntimeError("active build job did not stop")

    def reject_shutdown():
        raise failure

    state = server._ServerState(
        project_path=str(tmp_path),
        host="127.0.0.1",
        port=9713,
        mcp=object(),
        transport=object(),
        adapter_shutdown=reject_shutdown,
    )
    messages = []
    monkeypatch.setattr(server.Debug, "log_error", staticmethod(messages.append))
    install_runtime_service(server._RUNTIME_SERVICE_NAME, state)
    try:
        server._reap_state(state)
        assert state.error is failure
        assert get_runtime_service(server._RUNTIME_SERVICE_NAME) is state
        assert not state.stopped.is_set(), "Failed cleanup cannot acknowledge unload to PluginManager"
        assert len(messages) == 1
    finally:
        remove_runtime_service(server._RUNTIME_SERVICE_NAME, state)
