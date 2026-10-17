from __future__ import annotations

import sys
import types

from infernux.engine.project_context import get_project_root, set_project_root
from infernux.scene import SceneManager


def test_runtime_scene_load_prepares_persistent_group_before_commit(monkeypatch):
    import infernux.lib as native_lib
    import infernux.scene as scene_api

    calls: list[str] = []
    monkeypatch.setattr(SceneManager, "_runtime_scene_service", None)
    monkeypatch.setattr(SceneManager, "_unload_other_scenes", staticmethod(lambda _scene: None))

    class NativeManager:
        def get_active_scene(self):
            return object()

        def prepare_active_scene_replacement(self):
            calls.append("prepare")

    native_manager = NativeManager()
    monkeypatch.setattr(
        scene_api,
        "_NativeSceneManager",
        types.SimpleNamespace(instance=lambda: native_manager),
    )

    class AssetRegistry:
        @staticmethod
        def instance():
            return types.SimpleNamespace(get_asset_database=lambda: None)

    monkeypatch.setattr(native_lib, "AssetRegistry", AssetRegistry)

    scene_manager_module = types.ModuleType("infernux.engine.scene_manager")
    scene_manager_module.SceneFileManager = types.SimpleNamespace(instance=staticmethod(lambda: None))
    monkeypatch.setitem(sys.modules, "infernux.engine.scene_manager", scene_manager_module)

    transaction_module = types.ModuleType("infernux.engine.scene_document_transaction")

    class Transaction:
        error = ""

        def __init__(self, _scene, **kwargs):
            self._before_commit = kwargs.get("before_commit")

        def run_to_completion(self, *, raise_on_failure=False):
            assert raise_on_failure is False
            assert callable(self._before_commit)
            self._before_commit()
            return True

    transaction_module.SceneDocumentTransaction = Transaction
    monkeypatch.setitem(
        sys.modules,
        "infernux.engine.scene_document_transaction",
        transaction_module,
    )

    assert SceneManager._do_load("Assets/Scenes/Runtime.scene") is True
    assert calls == ["prepare"]


def test_path_reference_does_not_fall_back_to_an_unrelated_basename(tmp_path):
    project = tmp_path / "Project"
    gallery = project / "Assets" / "Scenes" / "VFX Gallery.scene"
    gallery.parent.mkdir(parents=True)
    gallery.write_text("{}", encoding="utf-8")
    previous_root = get_project_root()

    try:
        set_project_root(str(project))
        assert (
            SceneManager._resolve_build_scene(
                "Assets/Other/VFX Gallery.scene",
                [str(gallery)],
            )
            is None
        )
    finally:
        set_project_root(previous_root)


def test_scene_reload_requires_explicit_discard_and_uses_deferred_path(tmp_path, monkeypatch):
    from infernux.engine.scene_manager import SceneFileManager

    scene_path = tmp_path / "Assets" / "Scenes" / "Start.scene"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}", encoding="utf-8")
    previous = SceneFileManager._instance
    manager = SceneFileManager()
    calls = []
    try:
        manager._current_scene_path = str(scene_path)
        monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
        monkeypatch.setattr(manager, "_stage_scene_navigation", lambda: calls.append("stage"))
        monkeypatch.setattr(manager, "_save_camera_state", lambda path: calls.append(("camera", path)))
        monkeypatch.setattr(manager, "_begin_deferred_open", lambda path: calls.append(("open", path)))

        manager.mark_dirty()
        assert manager.reload_current_scene() is False
        assert calls == []
        assert manager.reload_current_scene(discard_changes=True) is True
        assert calls[0] == "stage"
        assert calls[-1] == ("open", str(scene_path))
    finally:
        SceneFileManager._instance = previous
