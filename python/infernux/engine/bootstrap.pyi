"""EditorBootstrap — structured editor initialization.

Orchestrates the full editor startup sequence (JIT pre-compilation,
renderer init, manager creation, panel wiring, layout persistence, etc.).

Typically invoked via :func:`infernux.engine.release_engine`.
"""

from __future__ import annotations

from typing import Optional

from infernux.engine.engine import Engine, LogLevel
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.ui.window_manager import WindowManager
from infernux.engine.ui.editor_services import EditorServices
from infernux.lib import HierarchyPanel, ConsolePanel, InspectorPanel, ProjectPanel
from infernux.engine.ui.scene_view_panel import SceneViewPanel
from infernux.engine.ui.game_view_panel import GameViewPanel
from infernux.engine.ui.ui_editor_panel import UIEditorPanel


class EditorBootstrap:
    """Orchestrates the full editor startup sequence.

    Example::

        bootstrap = EditorBootstrap("/path/to/project", LogLevel.Info)
        bootstrap.run()
        bootstrap.engine.show()
        bootstrap.engine.run()
    """

    project_path: str
    engine_log_level: LogLevel

    engine: Optional[Engine]
    undo_manager: object
    scene_file_manager: Optional[SceneFileManager]
    window_manager: Optional[WindowManager]
    services: Optional[EditorServices]

    menu_bar: object
    toolbar: object
    status_bar: object

    hierarchy: Optional[HierarchyPanel]
    inspector_panel: Optional[InspectorPanel]
    project_panel: Optional[ProjectPanel]
    console: Optional[ConsolePanel]
    scene_view: Optional[SceneViewPanel]
    game_view: Optional[GameViewPanel]
    ui_editor: Optional[UIEditorPanel]

    def __init__(self, project_path: str, engine_log_level: LogLevel = ...) -> None: ...

    def run(self) -> None:
        """Execute all bootstrap phases and prepare the main loop."""
        ...
