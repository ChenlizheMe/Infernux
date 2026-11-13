"""Preview ownership restores live authoring values without persisting samples."""

import numpy as np
import pytest

from infernux.core.material import Material
from infernux.engine.interaction import DocumentKey, DocumentKind, DocumentRegistry
from infernux.engine.interaction.previews import PreviewService
from infernux.lib import SceneManager, Vector3
from infernux.ui import UIText
from test_plugin_preload_failures import package_project


@pytest.fixture
def previews(engine, monkeypatch):
    import infernux.application as application

    monkeypatch.setattr(application, "_runtime_kind", "editor")
    previous = PreviewService._instance
    service = PreviewService()
    from infernux.lib import RenderPipelineCallback

    class Pipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    pipeline = Pipeline()
    engine.set_render_pipeline(pipeline)
    try:
        yield service
    finally:
        service.close()
        PreviewService._instance = previous
        engine.set_render_pipeline(None)


def test_preview_restores_transform_ui_and_existing_character_state(scene, previews, engine):
    from infernux.resources import engine_font_cjk_path, engine_font_path, engine_mono_font_path

    engine.set_gui_fonts(engine_font_path, engine_font_cjk_path, engine_mono_font_path, 18)
    owner = scene.create_game_object("Preview target")
    text = owner.add_component(UIText)
    text.text = "A中"
    original_geometry = text._resolve_text_geometry()
    original = text.create_character_buffer()
    original.offsets[:] = 4
    text.set_character_data(original)
    scope = previews.begin("text-panel")
    assert not scope.closed
    scope.set(owner.transform, "position", Vector3(5, 6, 7))
    scope.set(text, "color", [1, 0, 0, 0.5])
    scope.capture_text(text)
    changed = text.create_character_buffer()
    changed.rotations[:] = 90
    text.set_character_data(changed)
    text.max_visible_characters = 1
    assert text._resolve_text_geometry() is not original_geometry
    with pytest.raises(RuntimeError, match="already owns"):
        previews.begin("other").capture(text, "color")
    scope.close()
    scope.close()
    assert scope.closed
    assert (owner.transform.position.x, owner.transform.position.y, owner.transform.position.z) == (0, 0, 0)
    assert list(text.color) == [1, 1, 1, 1]
    assert text._resolve_text_geometry() is original_geometry
    assert text.max_visible_characters == -1 and np.all(original.offsets == 4)


def test_preview_material_keeps_other_material_autosaves_working(tmp_path, previews):
    first, second = Material.create_unlit("Preview"), Material.create_unlit("Authoring")
    first_path, second_path = tmp_path / "first.mat", tmp_path / "second.mat"
    assert first.save(str(first_path)) and second.save(str(second_path))
    before = first_path.read_bytes()
    scope = previews.begin("material-panel")
    scope.capture_material(first)
    scope.set(first, "alpha_clip_threshold", 0.6)
    first.set_color("base_color", 1, 0, 0, 1)
    second.set_color("base_color", 0, 1, 0, 1)
    second.flush()
    Material.flush_all_pending()
    assert first_path.read_bytes() == before
    assert second.get_color("base_color") == (0, 1, 0, 1)
    scope.close()
    assert first.get_color("base_color") == (1, 1, 1, 1)
    assert first_path.read_bytes() == before and not first._preview_active


@pytest.mark.parametrize("clean_request", [False, True])
def test_save_cancels_preview_before_capturing_document(scene, previews, clean_request):
    from infernux.engine.interaction import DocumentCapability

    owner = scene.create_game_object("Authoring snapshot")
    registry = DocumentRegistry.instance()
    document, _ = registry.open_or_create(
        DocumentKey.asset(DocumentKind.SCENE, "preview-test-scene"), "Preview scene",
        capabilities=DocumentCapability.SAVE,
    )
    scope = previews.begin("scene-preview")
    scope.set(owner.transform, "position", Vector3(4, 0, 0))
    revision = document.revision
    if clean_request:
        from infernux.engine.interaction import DocumentActionStatus

        assert registry.request_save(document.document_id).status is DocumentActionStatus.NO_OP
    else:
        ticket = registry.begin_save(document.document_id)
    assert owner.transform.position.x == 0 and document.revision == revision
    assert scope.closed
    if not clean_request:
        registry.complete_save(ticket.ticket_id, success=True)


def test_deleted_target_retires_preview_update(scene, previews, engine):
    owner = scene.create_game_object("Transient preview owner")
    scope = previews.begin("temporary-panel")
    scope.set(owner.transform, "position", Vector3(3, 0, 0))
    calls = []
    update = scope.add_update(calls.append)
    engine.tick(0.02)
    assert calls == [pytest.approx(0.02)]
    scene.destroy_game_object(owner)
    engine.tick(0.02)
    assert not owner
    calls_at_retirement = len(calls)
    engine.tick(0.02)
    engine.tick(0.02)
    assert len(calls) == calls_at_retirement and not update.enabled
    assert not previews._claims


def test_runtime_cannot_open_an_editor_preview(scene, previews):
    manager = SceneManager.instance()
    manager.play()
    try:
        with pytest.raises(RuntimeError, match="Edit mode"):
            previews.begin("game")
    finally:
        manager.stop()


def test_plugin_reload_and_disable_restore_owned_preview(scene, previews, package_project):
    from infernux.engine.interaction import EditorInteractionCore

    core = EditorInteractionCore()
    owner = scene.create_game_object("Owned preview")
    _, manager, install = package_project
    source = '''from infernux.lifecycle import InxPreload
from infernux.lib import SceneManager, Vector3
class Startup(InxPreload):
    def preload(self, context):
        scope = context.create_preview()
        scene = SceneManager.instance().get_active_scene()
        owner = next(obj for obj in scene.get_root_objects() if obj.name == "Owned preview")
        scope.set(owner.transform, "position", Vector3(5, 0, 0))
'''
    assert install("test/preview-owner", {"editor/startup.py": source}).loaded
    assert owner.transform.position.x == 5
    assert manager.reload("test/preview-owner").loaded
    assert owner.transform.position.x == 5 and len(core.previews._scopes) == 1
    manager.set_enabled("test/preview-owner", False)
    assert owner.transform.position.x == 0 and not core.previews._scopes
