from __future__ import annotations
import pytest
from infernux.engine.interaction import (
    ModalService,
)
from infernux.engine.ui import project_file_ops
from infernux.engine.ui.project_delete_confirmation import ProjectDeleteConfirmationCoordinator


def _project_delete_confirmation() -> ProjectDeleteConfirmationCoordinator:
    return ProjectDeleteConfirmationCoordinator(ModalService())


def test_dependency_restart_uses_shared_modal_and_cancelling_save_prevents_restart(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator
    from infernux.engine.ui.plugin_install_confirmation import PluginInstallConfirmationCoordinator
    import infernux_editor

    core = EditorInteractionCore()
    monkeypatch.setattr(EditorInteractionCore, "_instance", core)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    scene = SceneFileManager()
    confirmations, native_events, restarts = [], [], []
    monkeypatch.setattr(scene, "_native_engine_for_close", lambda: SimpleNamespace(
        confirm_close=lambda: native_events.append("close"),
        cancel_close=lambda: native_events.append("cancel"),
    ))
    monkeypatch.setattr(scene, "_is_play_mode", lambda: False)
    dirty = DirtyPanelConfirmationCoordinator.instance()
    monkeypatch.setattr(dirty, "request_exit", lambda complete, cancel: confirmations.append((complete, cancel)))
    monkeypatch.setattr(infernux_editor, "restart_after_exit", lambda project: restarts.append(project))
    manager = SimpleNamespace(project_root=str(tmp_path), pending_install={"loaded": ["anyio"]})
    coordinator = PluginInstallConfirmationCoordinator(core.modals)
    try:
        assert coordinator.request_restart(manager)
        ctx = _ProjectDeleteSemanticContext()
        assert coordinator.render(ctx)
        assert coordinator.MODAL_ID in ctx.semantics
        next(callback for label, callback in ctx.buttons.items() if label.endswith("##cancel"))()
        assert not confirmations and manager.pending_install is not None

        for cancel_save in (True, False):
            assert coordinator.request_restart(manager)
            ctx = _ProjectDeleteSemanticContext()
            coordinator.render(ctx)
            next(callback for label, callback in ctx.buttons.items() if label.endswith("##confirm"))()
            assert not restarts
            complete, cancel = confirmations[-1]
            if cancel_save:
                cancel()
                assert native_events == ["cancel"] and not restarts
            else:
                complete()
                assert restarts == [str(tmp_path)] and native_events[-1] == "close"
    finally:
        core.shutdown()


class _ProjectDeleteSemanticContext:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed = False
        self.semantics: list[str] = []
        self.buttons: dict[str, object] = {}

    def open_popup(self, popup_id: str) -> None:
        self.opened.append(popup_id)

    @staticmethod
    def get_dpi_scale() -> float:
        return 1.0

    @staticmethod
    def get_main_viewport_bounds():
        return 0.0, 0.0, 1280.0, 720.0

    @staticmethod
    def set_next_window_pos(_x, _y, _condition, _pivot_x, _pivot_y) -> None:
        pass

    @staticmethod
    def begin_popup_modal(_popup_id: str, _flags: int) -> bool:
        return True

    def record_semantic_window(self, _kind, _label, semantic_id) -> None:
        self.semantics.append(semantic_id)

    def record_semantic_item(self, _kind, _label, _enabled, semantic_id) -> None:
        self.semantics.append(semantic_id)

    @staticmethod
    def text_wrapped(_value: str) -> None:
        pass

    @staticmethod
    def spacing() -> None:
        pass

    @staticmethod
    def separator() -> None:
        pass

    @staticmethod
    def same_line() -> None:
        pass

    @staticmethod
    def end_popup() -> None:
        pass

    def button(self, label: str, callback, width: float = 0.0, height: float = 0.0) -> None:
        self.buttons[label] = callback

    def close_current_popup(self) -> None:
        self.closed = True


@pytest.mark.parametrize("load_error", ["", "Assets/Checkpoint.prefab: invalid document"])
def test_project_delete_modal_publishes_semantics_and_cancel_preserves_asset(tmp_path, load_error):
    asset = tmp_path / "Checkpoint.prefab"
    asset.write_text("prefab", encoding="utf-8")
    deleted: list[list[str]] = []
    coordinator = _project_delete_confirmation()

    assert coordinator.request([str(asset)], lambda paths: deleted.append(paths) or True, load_error=load_error)
    ctx = _ProjectDeleteSemanticContext()
    coordinator.render(ctx)

    assert len(ctx.opened) == 1
    assert ctx.opened[0].endswith("###project_delete_confirm")
    assert {
        "project.delete.dialog",
        "project.delete.confirm",
        "project.delete.cancel",
    }.issubset(ctx.semantics)
    next(callback for label, callback in ctx.buttons.items() if label.endswith("##cancel"))()
    assert coordinator.is_active is False
    assert asset.exists()
    assert deleted == []


def test_project_delete_modal_confirms_deduplicated_existing_paths(tmp_path):
    first = tmp_path / "First.prefab"
    second = tmp_path / "Second.prefab"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    received: list[list[str]] = []
    coordinator = _project_delete_confirmation()

    assert coordinator.request(
        [str(first), str(first), str(tmp_path / "missing.prefab"), str(second)],
        lambda paths: received.append(paths) or True,
    )
    ctx = _ProjectDeleteSemanticContext()
    coordinator.render(ctx)
    next(callback for label, callback in ctx.buttons.items() if label.endswith("##confirm"))()

    assert received == [[str(first.resolve()), str(second.resolve())]]
    assert coordinator.is_active is False
    assert ctx.closed is True


def test_project_delete_confirmation_rejects_reentrant_double_submit(tmp_path):
    asset = tmp_path / "DoubleClick.prefab"
    asset.write_text("prefab", encoding="utf-8")
    calls = 0
    coordinator = _project_delete_confirmation()
    ctx = _ProjectDeleteSemanticContext()

    def delete(paths):
        nonlocal calls
        calls += 1
        assert paths == [str(asset.resolve())]
        coordinator._confirm(ctx)
        return True

    assert coordinator.request([str(asset)], delete)
    coordinator._confirm(ctx)

    assert calls == 1
    assert coordinator.is_active is False
    assert ctx.closed is True


def test_asset_error_cancel_does_not_repeat_until_user_reopens(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors
    from infernux.engine.path_utils import lexical_path_key

    asset = tmp_path / "Damaged.mat"
    damaged = b"{ invalid material"
    asset.write_bytes(damaged)
    failure = {"guid": "damaged", "path": str(asset), "error": "Invalid material document",
               "modified_ns": asset.stat().st_mtime_ns, "size": len(damaged)}
    failures = [failure]
    database = SimpleNamespace(project_root=str(tmp_path), query_generation=1,
                               get_import_failures=lambda: failures,
                               get_import_error=lambda guid: failure["error"] if failures else "",
                               get_guid_from_path=lambda path: "damaged",
                               report_asset_load_failure=lambda guid, error: None)
    coordinator = _project_delete_confirmation()
    monkeypatch.setattr(ProjectDeleteConfirmationCoordinator, "instance", classmethod(lambda cls: coordinator))
    monkeypatch.setattr(AssetLoadErrors, "_instance", None)
    deleted = []
    errors = AssetLoadErrors(database, SimpleNamespace(delete=lambda paths: deleted.extend(paths) or True))
    errors.poll()
    assert coordinator.is_active
    assert coordinator._load_error == "Damaged.mat\nInvalid material document"
    ctx = _ProjectDeleteSemanticContext()
    coordinator.render(ctx)
    next(callback for label, callback in ctx.buttons.items() if label.endswith("##cancel"))()
    errors.poll()
    assert not coordinator.is_active
    assert asset.read_bytes() == damaged
    assert deleted == []
    with monkeypatch.context() as lookup:
        lookup.setattr(__import__("os").path, "realpath", lambda *_: pytest.fail("Inspector lookup touched disk"))
        assert errors.error(str(asset)) == failure["error"]
    assert errors.errors[lexical_path_key(str(asset))]["error"] == failure["error"]

    errors.report(str(asset), failure["error"])
    errors.poll()
    assert coordinator.is_active
    failures.clear()
    database.query_generation += 1
    errors.poll()
    ctx = _ProjectDeleteSemanticContext()
    coordinator.render(ctx)
    assert not coordinator.is_active and ctx.closed
    assert asset.read_bytes() == damaged and not deleted

    failures.append(failure)
    database.query_generation += 1
    errors.poll()
    ctx = _ProjectDeleteSemanticContext()
    coordinator.render(ctx)
    next(callback for label, callback in ctx.buttons.items() if label.endswith("##confirm"))()
    assert deleted == [str(asset.resolve())]
    assert not coordinator.is_active


def test_project_script_delete_uses_meta_guid_when_database_path_lookup_misses(monkeypatch, tmp_path):
    script = tmp_path / "Attached.py"
    script.write_text("class Attached:\n    pass\n", encoding="utf-8")
    script.with_suffix(".py.meta").write_text(
        '{"metadata":{"guid":{"type":"string","value":"attached-guid"}}}',
        encoding="utf-8",
    )

    class _Database:
        @staticmethod
        def get_guid_from_path(_path):
            return ""

    from infernux.core.assets import AssetManager

    delete_calls = []

    monkeypatch.setattr(
        AssetManager,
        "delete_asset",
        classmethod(lambda _cls, path, **kwargs: delete_calls.append((path, kwargs)) or True),
    )

    assert project_file_ops.delete_item(str(script), _Database()) is True
    assert not script.exists()
    assert len(delete_calls) == 1
    deleted_path, delete_kwargs = delete_calls[0]
    assert deleted_path == str(script)
    assert isinstance(delete_kwargs["database"], _Database)
    assert delete_kwargs["guid_hint"] == "attached-guid"
