from __future__ import annotations

from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.lib import AssetMutationErrorCode, AssetMutationResult, ResourceType


def _result(operation, path, *, succeeded=True, guid="guid", previous_path=""):
    result = AssetMutationResult()
    result.operation = operation
    result.path = path
    result.previous_path = previous_path
    result.guid = guid
    result.succeeded = succeeded
    result.database_committed = succeeded
    result.changed = succeeded
    if not succeeded:
        result.error_code = AssetMutationErrorCode.NOT_FOUND
        result.error = "not found"
    return result


class _Database:
    def __init__(self, order):
        self.order = order
        self.paths = {"old.txt": "guid"}
        self.import_errors = {}

    def get_guid_from_path(self, path):
        return self.paths.get(path, "")

    def get_resource_type(self, path):
        return ResourceType.DefaultText

    def import_asset(self, path, *, write_scope=None):
        self.order.append("db-import")
        self.paths[path] = "new-guid"
        return _result("import", path, guid="new-guid")

    def reimport_asset(self, path):
        self.order.append("db-reimport")
        self.import_errors.pop(self.get_guid_from_path(path), None)
        return _result("reimport", path, succeeded=path in self.paths)

    def report_runtime_load_failure(self, guid, error):
        self.import_errors[guid] = error

    def get_meta_by_guid(self, guid):
        return _Metadata() if guid == "guid" else None

    def delete_asset(self, path, *, write_scope=None):
        self.order.append("db-delete")
        self.paths.pop(path, None)
        return _result("delete", path)

    def move_asset(self, old_path, new_path, *, write_scope=None):
        self.order.append("db-move")
        guid = self.paths.pop(old_path, "")
        if not guid:
            return _result("move", new_path, succeeded=False, previous_path=old_path)
        self.paths[new_path] = guid
        return _result("move", new_path, guid=guid, previous_path=old_path)


class _Registry:
    def __init__(self, order):
        self.order = order

    def is_loaded(self, guid):
        return guid == "guid"

    def get_asset_residency(self, guid):
        return SimpleNamespace(runtime_version=1)

    def reload_asset(self, guid):
        self.order.append("registry-reload")
        return True

    def remove_asset(self, guid):
        self.order.append("registry-remove")

    def update_loaded_asset_path(self, guid, new_path):
        self.order.append(("registry-move", guid, new_path))


class _Metadata:
    def get_resource_type(self):
        return ResourceType.Shader

    def has_key(self, key):
        return key == "shader_id"

    def get_string(self, key):
        assert key == "shader_id"
        return "previous-shader"


class _NativeEngine:
    has_renderer = True
    _shader_reload_sources = ("old.vert.py",)
    _shader_reload_diagnostics = ()

    def __init__(self, order):
        self.order = order

    def reload_shader_runtime(self, path, previous_shader_id):
        assert path == "old.vert.py"
        assert previous_shader_id == "previous-shader"
        self.order.append("shader-runtime")
        return ""


class _FailingNativeEngine(_NativeEngine):
    def reload_shader_runtime(self, path, previous_shader_id):
        super().reload_shader_runtime(path, previous_shader_id)
        return "shader compile failed"


def _isolate_side_effects(monkeypatch, order):
    from infernux.engine.ui.asset_load_errors import clear_asset_load_error

    def publish(_cls, path, event="modified", **_kwargs):
        clear_asset_load_error(path)
        order.append(f"editor-{event}")

    AssetManager._watcher_echo_suppression.clear()
    registry = _Registry(order)
    monkeypatch.setattr(AssetManager, "_get_registry", classmethod(lambda _cls: registry))
    monkeypatch.setattr(AssetManager, "invalidate", classmethod(lambda _cls, _guid: order.append("py-evict")))
    monkeypatch.setattr(AssetManager, "_publish_asset_content_change", classmethod(publish))
    monkeypatch.setattr(AssetManager, "_invalidate_project_panel_cache", classmethod(lambda _cls: None))


def test_delete_commits_database_before_evicting_live_registry(monkeypatch):
    order = []
    database = _Database(order)
    _isolate_side_effects(monkeypatch, order)

    result = AssetManager.delete_asset("old.txt", database=database)
    assert result and result.database_committed
    # Native catalog publication owns resident retirement. The facade must
    # project editor consequences once, without a second Registry removal.
    assert order == ["db-delete", "py-evict", "editor-deleted"]


def test_move_commits_mapping_before_patching_loaded_path(monkeypatch):
    order = []
    database = _Database(order)
    _isolate_side_effects(monkeypatch, order)

    result = AssetManager.move_asset("old.txt", "new.txt", database=database)
    assert result and result.previous_path == "old.txt"
    assert order == ["db-move", "py-evict"]


def test_shader_runtime_failure_reports_committed_database_state(monkeypatch):
    from infernux.debug import Debug, DebugConsole, LogType

    monkeypatch.setattr(DebugConsole, "_instance", None)
    console = DebugConsole.instance()
    order = []
    database = _Database(order)
    database.paths = {"old.vert.py": "guid"}
    _isolate_side_effects(monkeypatch, order)
    monkeypatch.setattr(
        AssetManager,
        "_native_engine",
        classmethod(lambda _cls: _FailingNativeEngine(order)),
    )

    result = AssetManager.reimport_asset("old.vert.py", database=database)

    assert not result
    assert result.database_committed is True
    assert result.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    assert result.error == "shader compile failed"
    assert order == ["db-reimport", "shader-runtime"]
    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].log_type is LogType.ERROR
    assert entries[0].source_file == "old.vert.py"
    assert entries[0].message == "shader compile failed"

    # Corrected source retires its diagnostic only after runtime publication.
    Debug.log_error("unrelated error")
    monkeypatch.setattr(
        AssetManager,
        "_native_engine",
        classmethod(lambda _cls: _NativeEngine(order)),
    )
    assert AssetManager.reimport_asset("old.vert.py", database=database)
    assert [entry.message for entry in console.get_entries()] == ["unrelated error"]


@pytest.mark.parametrize("extension", [".effect", ".particlegraph"])
def test_compiled_asset_failure_has_source_diagnostic_until_corrected(monkeypatch, extension):
    from infernux.debug import Debug, DebugConsole, LogType
    from infernux.renderstack.render_effect_compiler import RenderEffectArtifactRegistry
    from infernux.particle.artifact import ParticleArtifactRegistry

    monkeypatch.setattr(DebugConsole, "_instance", None)
    console = DebugConsole.instance()
    order = []
    database = _Database(order)
    path = "New" + extension
    database.get_runtime_artifact_path = lambda *_args: ""
    _isolate_side_effects(monkeypatch, order)
    monkeypatch.setattr(AssetManager, "_get_cached", classmethod(lambda _cls, _guid: None))
    monkeypatch.setattr(AssetManager, "_prime_material_preview", classmethod(lambda _cls, _path: None))
    registry, method = (
        (RenderEffectArtifactRegistry, "compile_and_publish")
        if extension in (".effect", ".effectgroup")
        else (ParticleArtifactRegistry, "compile_path")
    )

    def fail(_cls, *_args, **_kwargs):
        raise ValueError("missing effect-stage resource: missing_probe")

    monkeypatch.setattr(registry, method, classmethod(fail))
    result = AssetManager.import_asset(path, database=database)
    assert not result and result.database_committed
    assert result.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    entries = console.get_entries()
    assert len(entries) == 1
    assert entries[0].log_type is LogType.ERROR
    assert entries[0].source_file == path
    assert entries[0].stack_trace == result.error
    assert "missing_probe" in result.error
    assert database.import_errors[database.paths[path]] == result.error
    assert "editor-created" not in order

    # Another rejected save replaces the source diagnostic; unrelated errors
    # survive the later successful publication.
    Debug.log_error("unrelated error")
    assert not AssetManager.reimport_asset(path, database=database)
    assert len([entry for entry in console.get_entries() if entry.source_file == path]) == 1
    monkeypatch.setattr(
        registry, method, classmethod(lambda _cls, *_args, **_kwargs: (SimpleNamespace(revision=1), None))
    )
    assert AssetManager.reimport_asset(path, database=database)
    assert [entry.message for entry in console.get_entries()] == ["unrelated error"]
    assert "editor-modified" in order


def test_shader_diagnostic_is_retired_only_when_every_rejected_consumer_recovers(monkeypatch):
    from infernux.debug import Debug, DebugConsole

    monkeypatch.setattr(DebugConsole, "_instance", None)
    monkeypatch.setattr(AssetManager, "_shader_diagnostic_roots", {})
    receipt = SimpleNamespace(_shader_reload_sources=["One.frag", "Two.frag"], _shader_reload_diagnostics=[])
    AssetManager._publish_shader_compile_result("Deleted.glsl", "both consumers rejected", receipt)
    Debug.log_error("unrelated error")

    receipt._shader_reload_sources = ["One.frag"]
    AssetManager._publish_shader_compile_result("One.frag", "", receipt)
    assert {entry.message for entry in DebugConsole.instance().get_entries()} == {
        "both consumers rejected",
        "unrelated error",
    }
    # A failure for the remaining root is not a successful receipt.
    receipt._shader_reload_sources = ["Two.frag"]
    AssetManager._publish_shader_compile_result("Two.frag", "still rejected", receipt)
    assert len(DebugConsole.instance().get_entries()) == 3
    AssetManager._publish_shader_compile_result("Two.frag", "", receipt)
    assert [entry.message for entry in DebugConsole.instance().get_entries()] == ["unrelated error"]
    assert not AssetManager._shader_diagnostic_roots


def test_shader_receipt_keeps_errors_from_unrelated_consumer_sets(monkeypatch):
    from infernux.debug import DebugConsole

    monkeypatch.setattr(DebugConsole, "_instance", None)
    monkeypatch.setattr(AssetManager, "_shader_diagnostic_roots", {})
    receipt = SimpleNamespace(_shader_reload_sources=["Other.frag"], _shader_reload_diagnostics=[])
    AssetManager._publish_shader_compile_result("Other.glsl", "unrelated rejection", receipt)
    receipt._shader_reload_sources = ["Repaired.frag"]
    AssetManager._publish_shader_compile_result("Replacement.glsl", "", receipt)
    assert [entry.message for entry in DebugConsole.instance().get_entries()] == ["unrelated rejection"]
