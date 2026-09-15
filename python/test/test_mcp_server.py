from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path

import pytest

from Infernux.host import OperationRegistry
from infernux_mcp import server
from infernux_mcp import capabilities
from infernux_mcp import scene_operations
from infernux_mcp.adapter import (
    MAX_GATEWAY_TOOLS,
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
            self.tools[str(name or fn.__name__)] = fn
            return fn

        return decorate


def test_capability_config_discards_unknown_schema_fields(tmp_path):
    settings = tmp_path / "ProjectSettings"
    settings.mkdir()
    path = settings / "mcp_capabilities.json"
    path.write_text(
        json.dumps(
            {
                "profile": "developer_assist",
                "unknown_section": {"unused": True},
                "limits": {"batch_max_steps": 12, "unknown_limit": 99},
            }
        ),
        encoding="utf-8",
    )

    loaded = capabilities.configure(str(tmp_path), write_default=True)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert "unknown_section" not in loaded
    assert "unknown_section" not in persisted
    assert "unknown_limit" not in persisted["limits"]
    assert persisted["limits"]["batch_max_steps"] == 12


def test_default_mcp_surface_is_schema_gateway_not_flat_tools(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    mcp = _FakeMCP()
    try:
        state = register_gateways(mcp, str(tmp_path), {})

        assert state["operation_count"] == 98
        assert state["owned_operation_count"] == 67
        assert state["gateway_count"] == 14
        assert 0.0 < state["registration_ms"] < 5000.0
        assert 0 < state["compact_schema_bytes"] < 128 * 1024
        assert state["compact_schema_bytes"] < state["full_schema_bytes"] < 512 * 1024
        assert len(mcp.tools) <= MAX_GATEWAY_TOOLS
        assert set(mcp.tools) == {
            "mcp_ping",
            "operation_schema_list",
            "operation_schema_get",
            "operation_schema_search",
            "operation_query_execute",
            "operation_command_execute",
            "operation_workflow_invoke",
            "operation_execute",
            "operation_batch_execute",
            "operation_job_submit",
            "operation_job_status",
            "operation_job_cancel",
            "host_capabilities",
            "host_session_status",
        }
        documents = OperationRegistry.instance().list()
        required = {
            "$schema",
            "id",
            "kind",
            "input_schema",
            "output_schema",
            "errors",
            "thread",
            "side_effects",
            "reversible",
            "capabilities",
            "cost",
            "availability",
            "phase",
        }
        assert len(documents) == 98
        assert all(required <= set(document) for document in documents)
        operation_ids = {document["id"] for document in documents}
        assert {
            "infernux.project.info",
            "infernux.mcp.checkpoint.list",
            "infernux.mcp.checkpoint.status",
            "infernux.mcp.supervisor.shutdown",
            "infernux.mcp.attempt.start",
            "infernux.mcp.attempt.stop",
            "infernux.mcp.blocker.report",
            "infernux.scene.object.create",
            "infernux.scene.loaded.get",
            "infernux.scene.additive.load",
            "infernux.scene.object.transform.set",
            "infernux.scene.component.property.set",
            "infernux.asset.create",
            "infernux.asset.mesh.save-copy",
            "infernux.scene.mesh.assign",
            "infernux.asset.move",
            "infernux.material.property.set",
            "infernux.data_asset.inspect",
            "infernux.data_asset.schema",
            "infernux.data_asset.property.set",
            "infernux.material.slot.assign",
            "infernux.renderer.parameter.get",
            "infernux.renderer.parameter.set",
            "infernux.renderer.parameter.remove",
            "infernux.renderer.parameter.clear",
            "infernux.particle.graph.document.replace",
            "infernux.camera.editor.state.set",
            "infernux.runtime.play",
            "infernux.runtime.performance.begin",
            "infernux.runtime.performance.get",
            "infernux.input.key",
            "infernux.ui.semantic.snapshot",
            "infernux.capture.request",
            "infernux.player.validation.launch",
            "infernux.player.targets",
            "infernux.docs.search",
            "infernux.console.read",
        } <= operation_ids
        assert not any(
            operation_id.startswith(
                (
                    "authoring_",
                    "assets_",
                    "camera_",
                    "particle_",
                    "runtime_",
                    "mcp_session_",
                )
            )
            for operation_id in operation_ids
        )
    finally:
        shutdown_adapter()
    remaining = OperationRegistry.instance().list()
    assert len(remaining) == 31
    assert {
        OperationRegistry.instance().get(document["id"]).owner
        for document in remaining
    } == {"infernux/engine"}
    assert adapter_status()["active"] is False


def test_loaded_scene_query_reports_every_resident_scene_and_active_owner(scene):
    from Infernux.lib import SceneManager

    manager = SceneManager.instance()
    additive = manager.create_scene("McpLoadedAdditive")
    additive.create_game_object("AdditiveMarker")
    try:
        manager.set_active_scene(additive)
        from Infernux.host import scene_operations as engine_scene_operations

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


def test_capability_grants_are_explicit_and_diff_is_machine_readable(tmp_path):
    (tmp_path / "Assets").mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    mcp = _FakeMCP()
    try:
        register_gateways(mcp, str(tmp_path), {})

        # The default grant is an explicit whitelist, never a wildcard.
        config = capabilities.current_config()
        assert "*" not in config["granted_capabilities"]
        assert config["granted_capabilities"] == list(
            capabilities.DEFAULT_GRANTED_CAPABILITIES
        )

        # Drift guard: every capability referenced by a registered operation
        # must be covered by the default enumeration.
        documents = OperationRegistry.instance().list()
        used = {
            str(name)
            for document in documents
            for name in document["capabilities"]
        }
        assert used <= set(capabilities.DEFAULT_GRANTED_CAPABILITIES)

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
        allowed = mcp.tools["operation_query_execute"](
            "infernux.mcp.checkpoint.list", {}
        )
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
        register_gateways(mcp, str(tmp_path), {})
        expected = {
            "project": "infernux.project.info",
            "checkpoint": "infernux.mcp.checkpoint.list",
            "attempt": "infernux.mcp.attempt.start",
            "shutdown": "infernux.mcp.supervisor.shutdown",
        }
        for query, operation in expected.items():
            result = mcp.tools["operation_schema_search"](query, 200)
            assert result["ok"] is True
            assert operation in {
                item["id"] for item in result["data"]["operations"]
            }

        missing_search = mcp.tools["operation_schema_search"]("missing.operation", 200)
        assert missing_search["ok"] is True
        assert missing_search["data"]["operations"] == []

        result = mcp.tools["operation_query_execute"](
            "infernux.mcp.checkpoint.list", {}
        )
        assert result["ok"] is True
        assert "checkpoints" in result["data"]["result"]
        missing_execute = mcp.tools["operation_query_execute"]("missing.operation", {})
        assert missing_execute["ok"] is False
        assert missing_execute["error"]["code"] == "operation.not_found"
        mismatch = mcp.tools["operation_command_execute"](
            "infernux.mcp.checkpoint.list", {}
        )
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
    from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue

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
        register_gateways(mcp, str(tmp_path), {})
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
        authored = next(
            item
            for item in hierarchy["data"]["result"]["objects"]
            if item["id"] == object_id
        )
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


def test_schema_gateway_native_and_python_fields_share_constraints_and_undo(
    scene, tmp_path
):
    from Infernux.components import InxComponent, serialized_field
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue

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
        register_gateways(mcp, str(tmp_path), {})

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
        native_fields = {
            item["name"]: item
            for item in native_schema["data"]["result"]["fields"]
        }
        python_fields = {
            item["name"]: item
            for item in python_schema["data"]["result"]["fields"]
        }
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
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.ui import project_file_ops
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue
    from Infernux.plugins import PluginManager

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
        register_gateways(mcp, str(project_root), {})
        changed = mcp.tools["operation_command_execute"](
            "infernux.material.property.set",
            {
                "asset_guid": guid,
                "pointer": "/properties/baseColor/value",
                "value": [0.2, 0.4, 0.6, 1.0],
            },
        )
        assert changed["ok"] is True
        assert changed["data"]["result"]["document"]["properties"]["baseColor"]["value"] == [
            0.2,
            0.4,
            0.6,
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
            if document["properties"]["baseColor"]["value"] == pytest.approx(
                [0.2, 0.4, 0.6, 1.0]
            ):
                break
            time.sleep(0.01)
        assert document is not None
        assert document["properties"]["baseColor"]["value"] == pytest.approx(
            [0.2, 0.4, 0.6, 1.0]
        )
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
    from Infernux.components import serialized_field
    from Infernux.core import AssetManager, DataAsset
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue
    from Infernux.plugins import PluginManager

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
        register_gateways(mcp, str(project_root), {})

        schema = mcp.tools["operation_query_execute"](
            "infernux.data_asset.schema", {"asset_guid": guid}
        )
        assert schema["ok"] is True
        speed_schema = next(
            field
            for field in schema["data"]["result"]["fields"]
            if field["property_path"] == "McpDataAsset.speed"
        )
        assert speed_schema["attributes"]["range"] == [0.0, 20.0]

        inspected = mcp.tools["operation_query_execute"](
            "infernux.data_asset.inspect", {"asset_guid": guid}
        )
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
        restored = mcp.tools["operation_query_execute"](
            "infernux.data_asset.inspect", {"asset_guid": guid}
        )
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
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.ui import project_file_ops
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue
    from Infernux.particle.asset import ParticleGraphAsset
    from Infernux.plugins import PluginManager

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
        register_gateways(mcp, str(project_root), {})
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


def test_schema_gateway_reads_and_replaces_guid_text_asset_with_undo(engine):
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue
    from Infernux.plugins import PluginManager

    database = engine.get_asset_database()
    project_root = Path(database.project_root)
    assets = project_root / "Assets"
    assets.mkdir(exist_ok=True)
    path = assets / f"McpText_{uuid.uuid4().hex}.txt"
    path.write_text("before\n", encoding="utf-8")
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
        register_gateways(mcp, str(project_root), {})
        read = mcp.tools["operation_query_execute"](
            "infernux.asset.text.read", {"asset_guid": guid}
        )
        assert read["ok"] is True
        assert read["data"]["result"]["content"] == "before\n"

        changed = mcp.tools["operation_command_execute"](
            "infernux.asset.text.set",
            {"asset_guid": guid, "content": "after\n"},
        )
        assert changed["ok"] is True, changed
        assert path.read_text(encoding="utf-8") == "after\n"
        assert len(undo.action_journal.applied_entries()) == 1

        undo.undo()
        assert path.read_text(encoding="utf-8") == "before\n"
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
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import EditorInteractionCore
    from Infernux.engine.ui import project_file_ops
    from Infernux.engine.undo import UndoManager
    from Infernux.host import MainThreadCommandQueue
    from Infernux.plugins import PluginManager

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
        register_gateways(mcp, str(project_root), {})
        kinds = mcp.tools["operation_query_execute"](
            "infernux.asset.create.kinds", {}
        )
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
