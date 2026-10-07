"""Animation events survive the editor's real authoring and save transactions."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("with_events", [True, False])
def test_clip_edit_undo_save_reopen_preserves_events(tmp_path, with_events):
    # A separate real Headless engine owns the project and editor services;
    # the suite's Graphical engine retains its own document/asset singletons.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(int(with_events))],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ANIMATION_EVENT_AUTHORING_OK" in result.stdout


def _exercise(project: Path, with_events: bool) -> None:
    from PIL import Image
    from infernux.core.animation_clip import AnimationClip, AnimationFrame
    from infernux.core.animation_event import AnimationEvent
    from infernux.core.asset_types import (
        SpriteFrame, TextureType, TextureImportSettings,
        TextureCompression, write_texture_import_settings,
    )
    from infernux.core.document_store import DocumentStore
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.engine import Engine
    from infernux.engine.interaction import DocumentRegistry, EditorInteractionCore
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.ui.animclip2d_editor_panel import AnimClip2DEditorPanel
    from infernux.engine.ui.editor_services import EditorServices
    from infernux.engine.undo import UndoManager
    from infernux.lib import LogLevel, RuntimeMode

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        bootstrap = EditorBootstrap(str(project))
        bootstrap.engine = engine
        services = EditorServices()
        services._engine = engine
        services._asset_database = engine.get_asset_database()
        services._project_path = str(project)
        services._interaction_core = EditorInteractionCore.instance()
        database = engine.get_asset_database()
        texture = project / "Assets" / "Sprite.png"
        Image.new("RGBA", (4, 4), (255, 127, 0, 255)).save(texture)
        assert database.import_asset(str(texture)).succeeded
        frame = SpriteFrame(name="Puzzle", x=0, y=0, w=4, h=4)
        settings = TextureImportSettings(
            texture_type=TextureType.SPRITE, compression=TextureCompression.NONE,
            generate_mipmaps=False, sprite_frames=[frame],
        )
        assert write_texture_import_settings(str(texture), settings)
        assert database.reimport_asset(str(texture)).succeeded
        events = [AnimationEvent(0.5, "Unlock", "DoorA", 7.0),
                  AnimationEvent(1.0, "Finish", "DoorB", 9.0)] if with_events else []
        clip = AnimationClip(
            name="Puzzle", authoring_texture_guid=database.get_guid_from_path(str(texture)),
            frames=[AnimationFrame(sprite_frame_id=frame.stable_id)], events=events,
        )
        path = project / "Assets" / "Puzzle.animclip2d"
        assert clip.save(str(path))
        assert database.import_asset(str(path)).succeeded
        panel = AnimClip2DEditorPanel()
        assert panel.open_document_resource_immediate(str(path))
        assert panel._active_clip.events == events
        assert panel._apply_authoring_mutation(
            "Change frame rate", lambda: setattr(panel._active_clip, "fps", 24.0),
        )
        manager = UndoManager.instance()
        manager.undo()
        assert panel._active_clip.fps == 12.0 and panel._active_clip.events == events
        manager.redo()
        assert panel._active_clip.fps == 24.0 and panel._active_clip.events == events
        # Draft restoration and deferred save snapshots must own their events.
        draft = panel.capture_document_restore_state(panel.document_id)
        if with_events:
            panel._active_clip.events[0].function = "Unsaved"
        panel.restore_document_restore_state(draft)
        assert panel._active_clip.events == events
        captured = panel._capture_animation_clip()
        if with_events:
            captured.events[0].function = "Detached"
            assert panel._active_clip.events == events
        registry = DocumentRegistry.instance()
        assert registry.request_save(panel.document_id).accepted
        DocumentStore.flush(str(path))
        registry.process_pending_saves()
        assert registry.require(panel.document_id).state.value == "ready"
        assert not registry.require(panel.document_id).is_dirty
        reopened = AnimationClip.load(str(path))
        assert reopened is not None
        assert reopened.fps == 24.0 and reopened.frames == clip.frames
        assert reopened.events == events
        assert panel.open_document_resource_immediate(str(path))
        assert panel._active_clip.events == events
        print("ANIMATION_EVENT_AUTHORING_OK", flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), bool(int(sys.argv[2])))
