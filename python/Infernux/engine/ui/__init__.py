import os as _os

# ── Editor-only imports ─────────────────────────────────────────────
# In standalone player builds (_INFERNUX_PLAYER_MODE=1) the heavy
# editor panels are never needed.  Skipping them avoids pulling in
# dozens of editor-only modules and keeps startup fast.
if not _os.environ.get("_INFERNUX_PLAYER_MODE"):
    from Infernux.lib import MenuBarPanel
    from .closable_panel import ClosablePanel
    from Infernux.lib import HierarchyPanel
    from Infernux.lib import InspectorPanel
    from Infernux.lib import ConsolePanel
    from .scene_view_panel import SceneViewPanel
    from .game_view_panel import GameViewPanel
    from Infernux.lib import ProjectPanel
    from .window_manager import WindowManager, WindowInfo
    from Infernux.lib import ToolbarPanel
    from .frame_scheduler_panel import FrameSchedulerPanel
    from .tag_layer_settings import TagLayerSettingsPanel
    from Infernux.lib import StatusBarPanel
    from .engine_status import EngineStatus
    from .build_settings_panel import BuildSettingsPanel
    from .viewport_utils import ViewportInfo, capture_viewport_info
    from .ui_editor_panel import UIEditorPanel
    from .selection_manager import SelectionManager
    from .animclip2d_editor_panel import AnimClip2DEditorPanel
    from .animfsm_editor_panel import AnimFSMEditorPanel

    # New panel framework
    from .editor_panel import EditorPanel
    from .editor_services import EditorServices
    from .event_bus import EditorEventBus, EditorEvent
    from .panel_registry import PanelRegistry, editor_panel
    from .editor_window import EditorWindow, editor_window

    __all__ = [
        "MenuBarPanel",
        "ToolbarPanel",
        "FrameSchedulerPanel",
        "HierarchyPanel",
        "InspectorPanel",
        "ConsolePanel",
        "SceneViewPanel",
        "GameViewPanel",
        "ProjectPanel",
        "ClosablePanel",
        "WindowManager",
        "WindowInfo",
        "TagLayerSettingsPanel",
        "StatusBarPanel",
        "EngineStatus",
        "BuildSettingsPanel",
        "ViewportInfo",
        "capture_viewport_info",
        "UIEditorPanel",
        "SelectionManager",
        "AnimClip2DEditorPanel",
        # New panel framework
        "EditorPanel",
        "EditorServices",
        "EditorEventBus",
        "EditorEvent",
        "PanelRegistry",
        "editor_panel",
        "EditorWindow",
        "editor_window",
    ]
else:
    __all__ = []