from __future__ import annotations
from infernux.engine.interaction import (
    ModalService,
)
from infernux.engine.ui import project_file_ops
from infernux.engine.ui.project_delete_confirmation import ProjectDeleteConfirmationCoordinator


def _project_delete_confirmation() -> ProjectDeleteConfirmationCoordinator:
    return ProjectDeleteConfirmationCoordinator(ModalService())


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


def test_project_delete_modal_publishes_semantics_and_cancel_preserves_asset(tmp_path):
    asset = tmp_path / "Checkpoint.prefab"
    asset.write_text("prefab", encoding="utf-8")
    deleted: list[list[str]] = []
    coordinator = _project_delete_confirmation()

    assert coordinator.request([str(asset)], lambda paths: deleted.append(paths) or True)
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
