"""Infernux Editor UI — panels, managers, and framework.

Re-exports all editor panel classes and the panel framework.
Skipped entirely in standalone player builds.
"""

from __future__ import annotations

from infernux.lib import MenuBarPanel as MenuBarPanel
from infernux.engine.ui.closable_panel import ClosablePanel as ClosablePanel
from infernux.lib import HierarchyPanel as HierarchyPanel
from infernux.lib import InspectorPanel as InspectorPanel
from infernux.lib import ConsolePanel as ConsolePanel
from infernux.engine.ui.scene_view_panel import SceneViewPanel as SceneViewPanel
from infernux.engine.ui.game_view_panel import GameViewPanel as GameViewPanel
from infernux.lib import ProjectPanel as ProjectPanel
from infernux.engine.ui.window_manager import WindowManager as WindowManager, WindowInfo as WindowInfo
from infernux.lib import ToolbarPanel as ToolbarPanel
from infernux.engine.ui.tag_layer_settings import TagLayerSettingsPanel as TagLayerSettingsPanel, PhysicsLayerMatrixPanel as PhysicsLayerMatrixPanel
from infernux.lib import StatusBarPanel as StatusBarPanel
from infernux.engine.ui.engine_status import EngineStatus as EngineStatus
from infernux.engine.ui.build_settings_panel import BuildSettingsPanel as BuildSettingsPanel
from infernux.engine.ui.preferences_panel import PreferencesPanel as PreferencesPanel
from infernux.engine.ui.history_panel import HistoryPanel as HistoryPanel
from infernux.engine.ui.environment_settings_panel import EnvironmentSettingsPanel as EnvironmentSettingsPanel
from infernux.engine.ui.viewport_utils import ViewportInfo as ViewportInfo, capture_viewport_info as capture_viewport_info
from infernux.engine.ui.ui_editor_panel import UIEditorPanel as UIEditorPanel
from infernux.engine.ui.particle_graph_editor_panel import ParticleGraphEditorPanel as ParticleGraphEditorPanel
from infernux.engine.ui.editor_panel import EditorPanel as EditorPanel, FloatingEditorPanel as FloatingEditorPanel
from infernux.engine.ui.editor_window import EditorWindow as EditorWindow, editor_window as editor_window
from infernux.engine.ui.editor_services import EditorServices as EditorServices
from infernux.engine.ui.panel_registry import PanelRegistry as PanelRegistry, editor_panel as editor_panel
from infernux.engine.ui.modal_portal import ModalPortal as ModalPortal
from infernux.engine.ui import panel_state as panel_state

__all__ = [
    "MenuBarPanel",
    "ToolbarPanel",
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
    "PhysicsLayerMatrixPanel",
    "StatusBarPanel",
    "EngineStatus",
    "BuildSettingsPanel",
    "PreferencesPanel",
    "HistoryPanel",
    "EnvironmentSettingsPanel",
    "ViewportInfo",
    "capture_viewport_info",
    "UIEditorPanel",
    "ParticleGraphEditorPanel",
    "EditorPanel",
    "FloatingEditorPanel",
    "EditorWindow",
    "EditorServices",
    "PanelRegistry",
    "editor_panel",
    "editor_window",
    "ModalPortal",
]
