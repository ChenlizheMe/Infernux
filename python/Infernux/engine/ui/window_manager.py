"""
Window Manager for Infernux Editor.
Manages window visibility, registration, and provides Window menu functionality.
"""
from collections import deque
from typing import Deque, Dict, Type, Callable, Optional, Any
from Infernux.lib import InxGUIRenderable


class WindowInfo:
    """Information about a registered window type."""
    def __init__(self, 
                 window_class: Type[InxGUIRenderable],
                 display_name: str,
                 factory: Optional[Callable[[], InxGUIRenderable]] = None,
                 singleton: bool = True,
                 title_key: Optional[str] = None,
                 menu_path: str = "Window"):
        self.window_class = window_class
        self._display_name = display_name
        self.title_key = title_key
        self.factory = factory or (lambda: window_class())
        self.singleton = singleton  # If True, only one instance allowed
        self.menu_path = menu_path  # Slash-separated menu path, e.g. "Animation/2D Animation"

    @property
    def display_name(self) -> str:
        if self.title_key:
            from Infernux.engine.i18n import t
            return t(self.title_key)
        return self._display_name


class WindowManager:
    """
    Centralized window manager for the editor.
    
    Features:
    - Register window types for the Window menu
    - Track open/closed windows
    - Create new window instances
    - Provide window state to panels
    """
    
    _instance: Optional['WindowManager'] = None
    
    def __init__(self, engine):
        self._engine = engine
        self._registered_types: Dict[str, WindowInfo] = {}  # type_id -> WindowInfo
        self._open_windows: Dict[str, bool] = {}  # window_id -> is_open
        self._window_instances: Dict[str, InxGUIRenderable] = {}  # window_id -> instance
        self._default_instances: Dict[str, InxGUIRenderable] = {}  # window_id -> original instance
        self._builtin_defaults: set = set()  # window_ids that should reopen on reset
        self._project_console_front_id = "project"
        self._on_state_changed: Optional[Callable[[], None]] = None
        self._imgui_ini_path: Optional[str] = None
        self._pending_actions: Deque[Callable[[], None]] = deque()
        self._is_processing_actions = False
        WindowManager._instance = self
    
    @classmethod
    def instance(cls) -> Optional['WindowManager']:
        """Get the singleton instance."""
        return cls._instance

    def set_on_state_changed(self, callback: Optional[Callable[[], None]]):
        self._on_state_changed = callback

    def _notify_state_changed(self):
        if self._on_state_changed is not None:
            self._on_state_changed()

    def note_panel_focus(self, panel_id: str):
        if panel_id in {"project", "console"}:
            self._project_console_front_id = panel_id
        self._notify_state_changed()
    
    def register_window_type(self, 
                             type_id: str,
                             window_class: Type[InxGUIRenderable],
                             display_name: str,
                             factory: Optional[Callable[[], InxGUIRenderable]] = None,
                             singleton: bool = True,
                             title_key: Optional[str] = None,
                             menu_path: str = "Window"):
        """
        Register a window type that can be created from the Window menu.
        
        Args:
            type_id: Unique identifier for this window type
            window_class: The class of the window
            display_name: Display name shown in menus
            factory: Optional factory function to create instances
            singleton: If True, only one instance of this window is allowed
            title_key: Optional i18n key for dynamic title resolution
            menu_path: Slash-separated menu path (e.g. "Window", "Animation/2D Animation")
        """
        self._registered_types[type_id] = WindowInfo(
            window_class=window_class,
            display_name=display_name,
            factory=factory,
            singleton=singleton,
            title_key=title_key,
            menu_path=menu_path,
        )
    
    def open_window(self, type_id: str, instance_id: Optional[str] = None) -> Optional[InxGUIRenderable]:
        """
        Open a window of the specified type.
        
        Args:
            type_id: The registered type ID
            instance_id: Optional specific instance ID (for non-singleton windows)
            
        Returns:
            The window instance, or None if cannot be created
        """
        if type_id not in self._registered_types:
            print(f"[WindowManager] Unknown window type: {type_id}")
            return None
        
        info = self._registered_types[type_id]
        window_id = instance_id or type_id
        
        # Check if already open (for singletons)
        if info.singleton and window_id in self._open_windows and self._open_windows[window_id]:
            print(f"[WindowManager] Window already open: {window_id}")
            return self._window_instances.get(window_id)
        
        pending_instance = self._window_instances.get(window_id)
        if pending_instance is not None and self._open_windows.get(window_id, False):
            return pending_instance

        # Reuse the original default instance when reopening a closed built-in
        # singleton so panel state survives hide/show and restart restore.
        instance = self._window_instances.get(window_id)
        if instance is None and info.singleton:
            instance = self._default_instances.get(window_id)
        if instance is None:
            instance = info.factory()

        if hasattr(instance, 'set_window_manager'):
            instance.set_window_manager(self)
        if hasattr(instance, 'open'):
            instance.open()
        self._window_instances[window_id] = instance
        self._open_windows[window_id] = True
        # Ensure singleton panels participate in save/load persistence
        # so their open state survives engine restarts.
        if info.singleton and window_id not in self._default_instances:
            self._default_instances[window_id] = instance
        self._notify_state_changed()

        def _register_instance(target_id=window_id, target_instance=instance):
            if not self._open_windows.get(target_id, False):
                return
            if self._window_instances.get(target_id) is not target_instance:
                return
            self._engine.register_gui(target_id, target_instance)

        self._enqueue_action(_register_instance)
        return instance
    
    def close_window(self, window_id: str):
        """Close a window by its ID."""
        if window_id in self._open_windows:
            self._open_windows[window_id] = False
            self._notify_state_changed()
            if window_id in self._window_instances:
                instance = self._window_instances[window_id]
                if hasattr(instance, '_is_open'):
                    instance._is_open = False

                def _unregister_instance(target_id=window_id, target_instance=instance):
                    if self._open_windows.get(target_id, False):
                        return
                    if self._window_instances.get(target_id) is not target_instance:
                        return
                    self._engine.unregister_gui(target_id)
                    self._window_instances.pop(target_id, None)

                self._enqueue_action(_unregister_instance)
    
    def is_window_open(self, window_id: str) -> bool:
        """Check if a window is currently open."""
        return self._open_windows.get(window_id, False)
    
    def set_window_open(self, window_id: str, is_open: bool):
        """Set window open state (called by window when close button is clicked)."""
        if not is_open and window_id in self._open_windows:
            self.close_window(window_id)
    
    def get_registered_types(self) -> Dict[str, WindowInfo]:
        """Get all registered window types."""
        return self._registered_types.copy()
    
    def get_open_windows(self) -> Dict[str, bool]:
        """Get all window open states."""
        return self._open_windows.copy()

    def save_state(self) -> Dict[str, Any]:
        from .closable_panel import ClosablePanel

        return {
            "open_windows": {
                window_id: bool(self._open_windows.get(window_id, False))
                for window_id in self._default_instances.keys()
            },
            "active_panel_id": ClosablePanel.get_active_panel_id() or "",
            "project_console_front_id": self._project_console_front_id,
        }

    def load_state(self, data: Dict[str, Any]):
        if not data:
            return

        open_windows = data.get('open_windows', {}) or {}
        for window_id, is_open in open_windows.items():
            # Lazily create registered-type windows not yet in _default_instances.
            # If the user opened this panel in a prior session and it was saved
            # as open, restore it now.
            if window_id not in self._default_instances:
                if is_open and window_id in self._registered_types:
                    self.open_window(window_id)
                continue

            instance = self._default_instances[window_id]
            if hasattr(instance, 'set_window_manager'):
                instance.set_window_manager(self)

            if is_open:
                self._open_windows[window_id] = True
                if hasattr(instance, '_is_open'):
                    instance._is_open = True
                if self._window_instances.get(window_id) is None:
                    self._window_instances[window_id] = instance
            else:
                self._open_windows[window_id] = False
                if hasattr(instance, '_is_open'):
                    instance._is_open = False
                if self._window_instances.get(window_id) is instance:
                    self._engine.unregister_gui(window_id)
                    self._window_instances.pop(window_id, None)

        active_panel_id = str(data.get('active_panel_id', '') or '')
        project_console_front_id = str(data.get('project_console_front_id', '') or '')
        if project_console_front_id in {"project", "console"}:
            self._project_console_front_id = project_console_front_id

        focus_panel_id = ""
        if self._project_console_front_id and self._open_windows.get(self._project_console_front_id, False):
            focus_panel_id = self._project_console_front_id
        elif active_panel_id and self._open_windows.get(active_panel_id, False):
            focus_panel_id = active_panel_id

        if focus_panel_id:
            self._engine.select_docked_window(focus_panel_id)
    
    def register_existing_window(self, window_id: str, instance: InxGUIRenderable, type_id: Optional[str] = None):
        """
        Register an already-created window instance.
        Used when windows are created directly (e.g., at startup).
        """
        self._window_instances[window_id] = instance
        self._open_windows[window_id] = True
        self._default_instances[window_id] = instance
        self._builtin_defaults.add(window_id)
        if window_id in {"project", "console"} and self._project_console_front_id not in {"project", "console"}:
            self._project_console_front_id = window_id
        
        # Store type_id association if provided
        if type_id:
            instance._window_type_id = type_id

    def set_imgui_ini_path(self, path: str):
        """Set the imgui.ini path used for docking layout persistence."""
        self._imgui_ini_path = path

    def reset_layout(self):
        """Reset to default layout: re-open default panels, clear ImGui docking state."""
        self._enqueue_action(self._apply_reset_layout)

    def process_pending_actions(self):
        """Run queued GUI mutations before ImGui starts building the next frame."""
        if self._is_processing_actions:
            return

        self._is_processing_actions = True
        try:
            while self._pending_actions:
                action = self._pending_actions.popleft()
                action()
        finally:
            self._is_processing_actions = False

    def _enqueue_action(self, action: Callable[[], None]):
        self._pending_actions.append(action)

    def _apply_reset_layout(self):
        # 1. Close any dynamically-opened windows (not part of builtin default set)
        dynamic_ids = [wid for wid in list(self._open_windows) if wid not in self._builtin_defaults]
        for wid in dynamic_ids:
            self._open_windows[wid] = False
            self._engine.unregister_gui(wid)
            self._window_instances.pop(wid, None)

        # 2. Force ALL builtin default panels to be open and registered
        for window_id in self._builtin_defaults:
            instance = self._default_instances.get(window_id)
            if instance is None:
                continue
            if hasattr(instance, '_is_open'):
                instance._is_open = True

            if hasattr(instance, 'set_window_manager'):
                instance.set_window_manager(self)

            if self._window_instances.get(window_id) is not instance:
                self._window_instances[window_id] = instance

            if not self._open_windows.get(window_id, False):
                self._open_windows[window_id] = True
                self._engine.register_gui(window_id, instance)
            else:
                self._open_windows[window_id] = True

        self._project_console_front_id = "project"
        self._notify_state_changed()

        # 3. Clear ImGui in-memory docking layout + delete ini file (C++ side)
        self._engine.reset_imgui_layout()
