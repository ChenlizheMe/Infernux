from InfEngine.lib import InfGUIRenderable, InfGUIContext
from typing import TYPE_CHECKING
from .build_settings_panel import BuildSettingsPanel
from .preferences_panel import PreferencesPanel
from .tag_layer_settings import PhysicsLayerMatrixPanel
from InfEngine.engine.project_context import get_project_root
from InfEngine.engine.i18n import t
from .theme import Theme, ImGuiCol, ImGuiStyleVar

if TYPE_CHECKING:
    from .window_manager import WindowManager
    from InfEngine.engine.scene_manager import SceneFileManager

# ImGuiKey constants
KEY_S = 564
KEY_N = 559
KEY_Z = 571
KEY_Y = 570
KEY_LEFT_CTRL = 527
KEY_RIGHT_CTRL = 531

class MenuBarPanel(InfGUIRenderable):
    def __init__(self, app):
        super().__init__()
        self.__app = app
        self.__native_engine = None
        if hasattr(app, 'get_native_engine'):
            self.__native_engine = app.get_native_engine()
        elif hasattr(app, 'is_close_requested'):
            self.__native_engine = app
        self.__window_manager = None
        self._dark_mode = True  # default to dark
        self._scene_file_manager = None
        self._build_settings = BuildSettingsPanel()
        self._preferences = PreferencesPanel()
        self._physics_layer_matrix = PhysicsLayerMatrixPanel()
        self._physics_layer_matrix.set_project_path(get_project_root() or "")

    def set_window_manager(self, window_manager: 'WindowManager'):
        """Set the window manager for the Window menu."""
        self.__window_manager = window_manager

    def set_scene_file_manager(self, sfm: 'SceneFileManager'):
        """Set the SceneFileManager for File menu operations."""
        self._scene_file_manager = sfm

    def _toggle_physics_layer_matrix(self):
        """Toggle the standalone physics layer matrix floating window."""
        if self._physics_layer_matrix.is_open:
            self._physics_layer_matrix.close()
        else:
            self._physics_layer_matrix.open()

    def on_render(self, ctx: InfGUIContext):
        # Handle global shortcuts (before any menu logic)
        self._handle_shortcuts(ctx)

        # Poll pending save dialog results every frame
        if self._scene_file_manager:
            self._scene_file_manager.poll_pending_save()
            self._scene_file_manager.poll_deferred_load()

        # Check for window close request (SDL_EVENT_QUIT intercepted by C++)
        if self._scene_file_manager and self.__native_engine:
            if self.__native_engine.is_close_requested():
                self._scene_file_manager.request_close()

        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.TOOLBAR_FRAME_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.TOOLBAR_ITEM_SPC)
        ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, *Theme.TOOLBAR_WIN_PAD)
        ctx.push_style_color(ImGuiCol.MenuBarBg, *Theme.WINDOW_BG)
        ctx.push_style_color(ImGuiCol.PopupBg, *Theme.POPUP_BG)
        ctx.push_style_color(ImGuiCol.HeaderHovered, *Theme.HEADER_HOVERED)
        ctx.push_style_color(ImGuiCol.HeaderActive, *Theme.HEADER_ACTIVE)

        if ctx.begin_main_menu_bar():
            # Project menu - build configuration (leftmost)
            if ctx.begin_menu(t("menu.project"), True):
                if ctx.menu_item(t("menu.build_settings"), "", self._build_settings.is_open, True):
                    if self._build_settings.is_open:
                        self._build_settings.close()
                    else:
                        self._build_settings.open()

                if self.__window_manager:
                    if ctx.menu_item(t("menu.physics_layer_matrix"), "", self._physics_layer_matrix.is_open, True):
                        self._toggle_physics_layer_matrix()

                ctx.separator()
                if ctx.menu_item(t("menu.preferences"), "", self._preferences.is_open, True):
                    if self._preferences.is_open:
                        self._preferences.close()
                    else:
                        self._preferences.open()

                ctx.end_menu()

            # Window menu - show all registered window types
            if ctx.begin_menu(t("menu.window"), True):
                if self.__window_manager:
                    registered_types = self.__window_manager.get_registered_types()
                    open_windows = self.__window_manager.get_open_windows()
                    
                    if registered_types:
                        for type_id, info in registered_types.items():
                            # Check if window is already open
                            is_open = open_windows.get(type_id, False)
                            # Grayed out if already open (for singletons)
                            can_create = not (info.singleton and is_open)
                            
                            # Show checkmark if window is open
                            label = info.display_name
                            if ctx.menu_item(label, "", is_open, can_create):
                                if is_open:
                                    # Close the window
                                    self.__window_manager.close_window(type_id)
                                else:
                                    # Open the window
                                    self.__window_manager.open_window(type_id)
                    else:
                        ctx.menu_item(t("menu.no_windows"), "", False, False)
                else:
                    ctx.menu_item(t("menu.no_wm"), "", False, False)
                
                ctx.separator()
                if ctx.menu_item(t("menu.reset_layout"), "", False, True):
                    if self.__window_manager:
                        self.__window_manager.reset_layout()
                
                ctx.end_menu()

            ctx.end_main_menu_bar()

        ctx.pop_style_color(4)
        ctx.pop_style_var(3)

        # Render floating windows (not docked)
        self._build_settings.render(ctx)
        self._preferences.render(ctx)
        self._physics_layer_matrix.render(ctx)

        # Render save-confirmation modal (if pending)
        if self._scene_file_manager:
            self._scene_file_manager.render_confirmation_popup(ctx)

    def _handle_shortcuts(self, ctx: InfGUIContext):
        """Process global keyboard shortcuts."""
        ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)
        if not ctrl:
            return

        if ctx.is_key_pressed(KEY_S):
            if self._scene_file_manager:
                self._scene_file_manager.save_current_scene()

        if ctx.is_key_pressed(KEY_N):
            if self._scene_file_manager:
                self._scene_file_manager.new_scene()

        if ctx.is_key_pressed(KEY_Z):
            undo_mgr = self._get_undo_manager()
            if undo_mgr and undo_mgr.can_undo:
                undo_mgr.undo()

        if ctx.is_key_pressed(KEY_Y):
            undo_mgr = self._get_undo_manager()
            if undo_mgr and undo_mgr.can_redo:
                undo_mgr.redo()

    @staticmethod
    def _get_undo_manager():
        """Lazily fetch the UndoManager singleton."""
        from InfEngine.engine.undo import UndoManager
        return UndoManager.instance()
