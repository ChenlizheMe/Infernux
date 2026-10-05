"""Real imported assets, graph transactions and Undo for animation canvas drops."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["2d", "3d", "timeline"])
def test_canvas_drop_validates_before_creating_or_replacing_state(tmp_path, mode):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), mode],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ANIMATION_CANVAS_ASSETS_OK" in result.stdout


def _exercise(project: Path, mode: str) -> None:
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.engine import Engine
    from infernux.engine.interaction import (
        EditorInteractionCore, SelectionDomain, SelectionService,
    )
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.ui import project_file_ops as ops
    from infernux.engine.ui.animfsm_editor_panel import AnimFSMEditorPanel
    from infernux.engine.ui.editor_services import EditorServices
    from infernux.engine.undo import UndoManager
    from infernux.lib import LogLevel, RuntimeMode

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        core = EditorInteractionCore.instance()
        core.panels.register_selection_authority("animfsm_editor", (SelectionDomain.GRAPH_ELEMENT,))
        bootstrap = EditorBootstrap(str(project))
        bootstrap.engine = engine
        services = EditorServices()
        services._engine = engine
        services._asset_database = engine.get_asset_database()
        services._project_path = str(project)
        services._interaction_core = core
        database = engine.get_asset_database()
        creator, extension, payload = {
            "2d": (ops.create_animclip, ".animclip2d", "ANIMCLIP_FILE"),
            "3d": (ops.create_animclip3d, ".animclip3d", "ANIMCLIP3D_FILE"),
            "timeline": (ops.create_animtimeline, ".animtimeline", "ANIMTIMELINE_FILE"),
        }[mode]
        paths = [project / "Assets" / (name + extension) for name in ("First", "Second")]
        for path in paths:
            ok, message = creator(str(path.parent), path.stem, database)
            assert ok, message
        guids = [database.get_guid_from_path(str(path)) for path in paths]
        assert all(guids) and guids[0] != guids[1]
        panel = AnimFSMEditorPanel()
        panel._graph_selection.bind(SelectionService.instance())
        panel._new_fsm_immediate(mode=mode, dirty=False)
        drop = panel._view.on_canvas_drop
        field = "timeline_guid" if mode == "timeline" else "clip_guid"
        manager = UndoManager.instance()
        before = panel._fsm.to_dict()
        # Missing GUID and incompatible asset types must not create empty states.
        missing = str(project / "Assets" / ("Missing" + extension))
        drop(payload, missing, 500.0, 500.0)
        assert panel._fsm.to_dict() == before and not manager.can_undo
        other_creator, other_ext, other_payload = (
            (ops.create_animclip3d, ".animclip3d", "ANIMCLIP3D_FILE") if mode != "3d"
            else (ops.create_animclip, ".animclip2d", "ANIMCLIP_FILE")
        )
        assert other_creator(str(project / "Assets"), "Incompatible", database)[0]
        incompatible = str(project / "Assets" / ("Incompatible" + other_ext))
        drop(other_payload, incompatible, 500.0, 500.0)
        assert panel._fsm.to_dict() == before and not manager.can_undo

        drop(payload, str(paths[0]), 500.0, 500.0)
        assert panel._fsm.state_count == 1
        assert getattr(panel._fsm.states[0], field) == guids[0]
        first = panel._fsm.to_dict()
        manager.undo()
        assert panel._fsm.to_dict() == before
        manager.redo()
        assert panel._fsm.to_dict() == first

        drop(payload, missing, 500.0, 500.0)
        drop(other_payload, incompatible, 500.0, 500.0)
        assert panel._fsm.to_dict() == first
        drop(payload, str(paths[1]), 500.0, 500.0)
        assert panel._fsm.state_count == 1
        assert getattr(panel._fsm.states[0], field) == guids[1]
        second = panel._fsm.to_dict()
        manager.undo()
        assert panel._fsm.to_dict() == first
        manager.redo()
        assert panel._fsm.to_dict() == second
        print("ANIMATION_CANVAS_ASSETS_OK", flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), sys.argv[2])
