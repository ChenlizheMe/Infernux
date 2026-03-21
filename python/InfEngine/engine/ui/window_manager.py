"""
Window Manager for InfEngine Editor.
Manages window visibility, registration, and provides Window menu functionality.
"""
from typing import Dict, Type, Callable, Optional, Any
from InfEngine.lib import InfGUIRenderable


class WindowInfo:
    """Information about a registered window type."""
    def __init__(self, 
                 window_class: Type[InfGUIRenderable],
                 display_name: str,
                 factory: Optional[Callable[[], InfGUIRenderable]] = None,
                 singleton: bool = True,
                 title_key: Optional[str] = None):
        self.window_class = window_class
        self._display_name = display_name
        self.title_key = title_key
        self.factory = factory or (lambda: window_class())
        self.singleton = singleton  # If True, only one instance allowed

    @property
    def display_name(self) -> str:
        if self.title_key:
            from InfEngine.engine.i18n import t
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
        self._window_instances: Dict[str, InfGUIRenderable] = {}  # window_id -> instance
        self._default_instances: Dict[str, InfGUIRenderable] = {}  # window_id -> original instance
        self._imgui_ini_path: Optional[str] = None
        WindowManager._instance = self
    
    @classmethod
    def instance(cls) -> Optional['WindowManager']:
        """Get the singleton instance."""
        return cls._instance
    
    def register_window_type(self, 
                             type_id: str,
                             window_class: Type[InfGUIRenderable],
                             display_name: str,
                             factory: Optional[Callable[[], InfGUIRenderable]] = None,
                             singleton: bool = True,
                             title_key: Optional[str] = None):
        """
        Register a window type that can be created from the Window menu.
        
        Args:
            type_id: Unique identifier for this window type
            window_class: The class of the window
            display_name: Display name shown in the Window menu (e.g., "层级 Hierarchy")
            factory: Optional factory function to create instances
            singleton: If True, only one instance of this window is allowed
            title_key: Optional i18n key for dynamic title resolution
        """
        self._registered_types[type_id] = WindowInfo(
            window_class=window_class,
            display_name=display_name,
            factory=factory,
            singleton=singleton,
            title_key=title_key,
        )
    
    def open_window(self, type_id: str, instance_id: Optional[str] = None) -> Optional[InfGUIRenderable]:
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
        
        # Create new instance
        instance = info.factory()
        # Wire up window manager reference for ClosablePanel subclasses
        if hasattr(instance, 'set_window_manager'):
            instance.set_window_manager(self)
        self._window_instances[window_id] = instance
        self._open_windows[window_id] = True
        self._engine.register_gui(window_id, instance)
        return instance
    
    def close_window(self, window_id: str):
        """Close a window by its ID."""
        if window_id in self._open_windows:
            self._open_windows[window_id] = False
            if window_id in self._window_instances:
                self._engine.unregister_gui(window_id)
                del self._window_instances[window_id]
    
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
    
    def register_existing_window(self, window_id: str, instance: InfGUIRenderable, type_id: Optional[str] = None):
        """
        Register an already-created window instance.
        Used when windows are created directly (e.g., at startup).
        """
        self._window_instances[window_id] = instance
        self._open_windows[window_id] = True
        self._default_instances[window_id] = instance
        
        # Store type_id association if provided
        if type_id:
            instance._window_type_id = type_id

    def set_imgui_ini_path(self, path: str):
        """Set the imgui.ini path used for docking layout persistence."""
        self._imgui_ini_path = path

    def reset_layout(self):
        """Reset to default layout: re-open all default panels, clear ImGui docking state."""
        import os
        # 1. Close any dynamically-opened windows (not part of default set)
        dynamic_ids = [wid for wid in list(self._open_windows) if wid not in self._default_instances]
        for wid in dynamic_ids:
            self.close_window(wid)

        # 2. Force ALL default panels to be open and registered
        for window_id, instance in self._default_instances.items():
            # Ensure the panel considers itself open
            if hasattr(instance, '_is_open'):
                instance._is_open = True
            # If the window was closed (unregistered), re-register it
            if not self._open_windows.get(window_id, False):
                self._window_instances[window_id] = instance
                self._open_windows[window_id] = True
                if hasattr(instance, 'set_window_manager'):
                    instance.set_window_manager(self)
                self._engine.register_gui(window_id, instance)

        # 3. Clear ImGui in-memory docking layout + delete ini file (C++ side)
        self._engine.reset_imgui_layout()
