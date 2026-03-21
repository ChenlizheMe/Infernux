"""
Scene file management for InfEngine.

Handles:
- Tracking the current scene file path (.scene)
- Saving / loading scene files (delegates to C++ Scene::SaveToFile / LoadFromFile)
- Python component serialization during save, recreation during load
- Remembering last opened scene per project (EditorSettings.json)
- Default scene fallback when a scene file is missing
- File-dialog for "Save As" when the scene has no file yet
- Enforcing that scenes must be saved under Assets/

The C++ layer already provides ``Scene.serialize / deserialize / save_to_file /
load_from_file`` and ``PendingPyComponent`` for Python component recreation.
This module orchestrates those primitives into a complete workflow.
"""

import os
import json
import threading
from typing import Optional, Callable

from InfEngine.debug import Debug
from InfEngine.engine.project_context import get_project_root
from InfEngine.engine.path_utils import safe_path as _safe_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_EXTENSION = ".scene"
EDITOR_SETTINGS_FILE = "EditorSettings.json"
DEFAULT_SCENE_NAME = "Untitled Scene"

# ImGuiKey enum values (matches imgui.h)
KEY_S = 564          # S
KEY_LEFT_CTRL = 527  # Left Ctrl
KEY_RIGHT_CTRL = 531 # Right Ctrl


# ---------------------------------------------------------------------------
# Editor settings helpers (ProjectSettings/EditorSettings.json)
# ---------------------------------------------------------------------------

def _settings_path() -> Optional[str]:
    root = get_project_root()
    if not root:
        return None
    return os.path.join(root, "ProjectSettings", EDITOR_SETTINGS_FILE)


def _load_editor_settings() -> dict:
    path = _settings_path()
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def _save_editor_settings(settings: dict):
    path = _settings_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File dialog — Win32 native (fast), with tkinter fallback
# ---------------------------------------------------------------------------

def _show_save_dialog(initial_dir: str, callback: Callable[[Optional[str]], None]):
    """Show a native save-file dialog. *callback* receives the chosen path or None."""
    def _run():
        result = _win32_save_dialog(initial_dir)
        callback(result)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _win32_save_dialog(initial_dir: str) -> Optional[str]:
    """Use the Win32 GetSaveFileNameW API directly via ctypes.
    Much faster than tkinter which has to load the entire Tcl/Tk runtime."""
    import ctypes
    import ctypes.wintypes as wt

    OFN_OVERWRITEPROMPT = 0x00000002
    OFN_NOCHANGEDIR     = 0x00000008
    OFN_EXPLORER        = 0x00080000
    MAX_PATH = 1024

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize",       wt.DWORD),
            ("hwndOwner",         wt.HWND),
            ("hInstance",         wt.HINSTANCE),
            ("lpstrFilter",       wt.LPCWSTR),
            ("lpstrCustomFilter", wt.LPWSTR),
            ("nMaxCustFilter",    wt.DWORD),
            ("nFilterIndex",      wt.DWORD),
            ("lpstrFile",         wt.LPWSTR),
            ("nMaxFile",          wt.DWORD),
            ("lpstrFileTitle",    wt.LPWSTR),
            ("nMaxFileTitle",     wt.DWORD),
            ("lpstrInitialDir",   wt.LPCWSTR),
            ("lpstrTitle",        wt.LPCWSTR),
            ("Flags",             wt.DWORD),
            ("nFileOffset",       wt.WORD),
            ("nFileExtension",    wt.WORD),
            ("lpstrDefExt",       wt.LPCWSTR),
            ("lCustData",         ctypes.POINTER(ctypes.c_long)),
            ("lpfnHook",          ctypes.c_void_p),
            ("lpTemplateName",    wt.LPCWSTR),
            ("pvReserved",        ctypes.c_void_p),
            ("dwReserved",        wt.DWORD),
            ("FlagsEx",           wt.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    ofn = OPENFILENAMEW()
    ofn.lStructSize    = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter    = "Scene files (*.scene)\0*.scene\0All files (*.*)\0*.*\0\0"
    ofn.lpstrFile      = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile       = MAX_PATH
    ofn.lpstrInitialDir = initial_dir
    ofn.lpstrTitle     = "保存场景 Save Scene"
    ofn.Flags          = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR | OFN_EXPLORER
    ofn.lpstrDefExt    = "scene"

    if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
        return buf.value
    return None


# ---------------------------------------------------------------------------
# SceneFileManager  — the main public API
# ---------------------------------------------------------------------------

class SceneFileManager:
    """Manages the mapping between the active C++ Scene and its file on disk.

    Typical usage (wired in ``release_engine``):

        sfm = SceneFileManager()
        # at startup:
        sfm.load_last_scene_or_default()
        # on Ctrl+S:
        sfm.save_current_scene()
        # on double-click a .scene in Project panel:
        sfm.open_scene(path)
    """

    _instance: Optional["SceneFileManager"] = None

    def __init__(self):
        SceneFileManager._instance = self
        self._current_scene_path: Optional[str] = None
        self._dirty: bool = False
        self._on_scene_changed: Optional[Callable[[], None]] = None
        self._pending_save_path: Optional[str] = None  # set by file dialog
        self._asset_database = None  # Set via set_asset_database()
        self._engine = None  # set via set_engine()

        # Confirmation-dialog state
        self._pending_action: Optional[str] = None   # 'new' | 'open' | 'close'
        self._pending_open_path: Optional[str] = None
        self._show_confirm: bool = False
        self._post_save_callback: Optional[Callable[[], None]] = None

        # Deferred scene loading — actual load runs on the NEXT frame so
        # the scene view has one frame to stop rendering old 3D content,
        # preventing in-flight GPU resources from being destroyed mid-use.
        self._deferred_load_path: Optional[str] = None   # non-None → load pending
        self._deferred_new_scene: bool = False            # True → new scene pending

        # Guard against repeated request_close() calls while a close
        # confirmation dialog is already visible.  Without this,
        # is_close_requested() being True every frame would re-trigger
        # _request_save_confirmation() and re-open the ImGui popup every
        # frame, making the buttons unclickable.
        self._close_in_progress: bool = False

    @classmethod
    def instance(cls) -> Optional["SceneFileManager"]:
        return cls._instance

    def set_asset_database(self, asset_db):
        """Set the AssetDatabase for GUID→path resolution during scene load."""
        self._asset_database = asset_db

    def set_engine(self, engine):
        """Set the native InfEngine reference (for close-request handling)."""
        self._engine = engine

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_scene_path(self) -> Optional[str]:
        return self._current_scene_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def is_loading(self) -> bool:
        """True while a deferred scene load is pending."""
        return self._deferred_load_path is not None or self._deferred_new_scene

    def mark_dirty(self):
        if self._is_play_mode():
            return
        self._dirty = True

    def clear_dirty(self):
        """Clear the dirty flag (e.g. when undo returns to save point)."""
        self._dirty = False

    def set_on_scene_changed(self, cb: Callable[[], None]):
        """Register callback invoked after a scene is opened/created."""
        self._on_scene_changed = cb

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _is_play_mode(self) -> bool:
        """Return True if the engine is in Play or Pause mode."""
        from InfEngine.engine.play_mode import PlayModeManager, PlayModeState
        pm = PlayModeManager.instance()
        if pm and pm.state != PlayModeState.EDIT:
            return True
        return False

    def save_current_scene(self) -> bool:
        """Save the current scene.  If no file is associated, show a Save-As dialog.

        Returns True if the save happened synchronously, False if a dialog was
        opened (the actual save happens asynchronously via the dialog callback).
        """
        if self._is_play_mode():
            Debug.log_warning("Cannot save scene while in Play mode.")
            return False

        if self._current_scene_path:
            return self._do_save(self._current_scene_path)

        # No file yet — show a Save As dialog
        self._show_save_as_dialog()
        return False

    def save_scene_as(self):
        """Force a Save-As dialog regardless of whether a path exists."""
        if self._is_play_mode():
            Debug.log_warning("Cannot save scene while in Play mode.")
            return
        self._show_save_as_dialog()

    def open_scene(self, path: str) -> bool:
        """Load a .scene file, replacing the current scene.

        If the current scene is dirty, shows a save-confirmation popup first.
        The actual load is deferred to the next frame so the scene view can
        stop rendering old 3D content first.
        """
        # Save current camera state before switching
        if self._current_scene_path:
            self._save_camera_state(self._current_scene_path)
        if self._dirty:
            self._request_save_confirmation('open', path)
            return False
        self._begin_deferred_open(path)
        return True

    def new_scene(self):
        """Replace the current scene with a fresh default scene (no file).

        If the current scene is dirty, shows a save-confirmation popup first.
        The actual creation is deferred to the next frame.
        """
        # Persist camera state before switching away
        if self._current_scene_path:
            self._save_camera_state(self._current_scene_path)
        if self._dirty:
            self._request_save_confirmation('new')
            return
        self._begin_deferred_new()

    def request_close(self):
        """Called when the window close button is pressed.

        If the scene is dirty, shows a save-confirmation popup.
        Otherwise, confirms the close immediately.

        During play mode the close is confirmed without a save dialog
        because the live scene is a temporary simulation snapshot — saving
        it would persist play-mode state, not the user's edit-mode work.
        ``engine.exit()`` will restore and clean up play-mode state before
        the C++ teardown begins.
        """
        # Guard: the menu bar polls is_close_requested() every frame.
        # Without this, a dirty scene would re-open the save dialog
        # each frame, preventing the user from clicking any button.
        if self._close_in_progress:
            return
        self._close_in_progress = True

        # Always persist camera state before closing
        if self._current_scene_path:
            self._save_camera_state(self._current_scene_path)

        # During play mode, close immediately — the save dialog's "Save"
        # button cannot save the pre-play state properly.
        if self._is_play_mode():
            if self._engine:
                self._engine.confirm_close()
            return

        if self._dirty:
            self._request_save_confirmation('close')
        elif self._engine:
            self._engine.confirm_close()

    def load_last_scene_or_default(self):
        """Called at startup — load the last opened scene, or create a default.

        Uses immediate (non-deferred) loading since no rendering occurs yet.
        """
        settings = _load_editor_settings()
        last_scene = settings.get("lastOpenedScene")
        if last_scene and os.path.isfile(last_scene):
            if self._do_open_scene(last_scene):
                return
            Debug.log_warning(f"Last scene file missing or invalid: {last_scene}")

        # Fallback to default (immediate — no rendering loop yet)
        self._do_new_scene()

    # ------------------------------------------------------------------
    # Ctrl+S handler  (called from menu bar / toolbar every frame)
    # ------------------------------------------------------------------

    def handle_shortcut(self, ctx) -> bool:
        """Check for Ctrl+S and save. Returns True if a save was triggered."""
        ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)
        if ctrl and ctx.is_key_pressed(KEY_S):
            self.save_current_scene()
            return True
        return False

    def poll_pending_save(self):
        """Check if the file dialog has produced a result and perform the save."""
        if self._pending_save_path is not None:
            path = self._pending_save_path
            self._pending_save_path = None  # consume
            if path:
                success = self._do_save(path)
                if success and self._post_save_callback:
                    cb = self._post_save_callback
                    self._post_save_callback = None
                    cb()
                elif not success:
                    self._post_save_callback = None
            else:
                # User cancelled the dialog — clear chained action
                self._post_save_callback = None

    # ------------------------------------------------------------------
    # Deferred scene loading (called from menu_bar every frame)
    # ------------------------------------------------------------------

    def _begin_deferred_open(self, path: str):
        """Schedule a scene open for the next frame."""
        self._deferred_load_path = path
        self._deferred_new_scene = False

    def _begin_deferred_new(self):
        """Schedule a new-scene creation for the next frame."""
        self._deferred_load_path = None
        self._deferred_new_scene = True

    def poll_deferred_load(self):
        """Execute a pending deferred scene load/new.

        Must be called every frame (from menu_bar).  The one-frame delay
        between _begin_deferred_open/new and this method gives the
        current frame's GPU submission a chance to complete before
        _prepare_native_scene_swap() calls WaitForGpuIdle(), which
        performs a full vkDeviceWaitIdle + FlushDeletionQueue.

        The old scene's texture naturally remains in the render target
        until the new scene's first Execute() overwrites it, so no
        placeholder or extra-frame delay is needed.
        """
        if self._deferred_load_path is not None:
            path = self._deferred_load_path
            self._deferred_load_path = None
            try:
                self._do_open_scene(path)
            except Exception as exc:
                Debug.log_error(f"Scene load failed: {exc}")
        elif self._deferred_new_scene:
            self._deferred_new_scene = False
            try:
                self._do_new_scene()
            except Exception as exc:
                Debug.log_error(f"New scene failed: {exc}")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def get_display_name(self) -> str:
        """Return a short display string for the current scene (for title bars)."""
        if self._current_scene_path:
            name = os.path.splitext(os.path.basename(self._current_scene_path))[0]
        else:
            name = DEFAULT_SCENE_NAME
        if self._dirty:
            name += " *"
        return name

    # ------------------------------------------------------------------
    # Save-confirmation popup (rendered from menu_bar every frame)
    # ------------------------------------------------------------------

    def _request_save_confirmation(self, action: str, open_path: Optional[str] = None):
        """Set up the confirmation popup state."""
        self._pending_action = action
        self._pending_open_path = open_path
        self._show_confirm = True

    def _execute_pending_action(self) -> bool:
        """Run the action that was deferred by the confirmation dialog."""
        action = self._pending_action
        path = self._pending_open_path
        self._pending_action = None
        self._pending_open_path = None

        if action == 'new':
            self._begin_deferred_new()
            return True
        elif action == 'open' and path:
            self._begin_deferred_open(path)
            return True
        elif action == 'close' and self._engine:
            self._engine.confirm_close()
            return True
        return False

    def _clear_pending_action(self):
        self._pending_action = None
        self._pending_open_path = None

    def render_confirmation_popup(self, ctx):
        """Must be called every frame (by menu_bar).

        Draws the modal "Save before …?" dialog when ``_show_confirm`` is set.
        """
        POPUP_ID = "Save Scene?##save_confirm"

        if not self._show_confirm and self._pending_action is None:
            return

        if self._show_confirm:
            ctx.open_popup(POPUP_ID)
            self._show_confirm = False

        # ImGuiWindowFlags_AlwaysAutoResize = 1 << 6 = 64
        if ctx.begin_popup_modal(POPUP_ID, 64):
            ctx.label("当前场景有未保存的修改。")
            ctx.label("The current scene has unsaved changes.")
            ctx.label("")
            ctx.separator()
            ctx.label("")

            def _on_save():
                if self._current_scene_path:
                    if self._do_save(self._current_scene_path):
                        self._execute_pending_action()
                else:
                    self._post_save_callback = self._execute_pending_action
                    self._show_save_as_dialog()
                ctx.close_current_popup()

            def _on_dont_save():
                action = self._pending_action
                if action == 'close':
                    self._dirty = False
                    self._execute_pending_action()
                else:
                    self._execute_pending_action()
                ctx.close_current_popup()

            def _on_cancel():
                if self._pending_action == 'close' and self._engine:
                    self._engine.cancel_close()
                self._close_in_progress = False
                self._clear_pending_action()
                ctx.close_current_popup()

            ctx.button("  保存  Save  ", _on_save)
            ctx.same_line()
            ctx.button("  不保存  Don't Save  ", _on_dont_save)
            ctx.same_line()
            ctx.button("  取消  Cancel  ", _on_cancel)

            ctx.end_popup()

    # ------------------------------------------------------------------
    # Internal — actual scene operations (no dirty check)
    # ------------------------------------------------------------------

    def _prepare_native_scene_swap(self):
        """Clear native editor state and drain GPU work before scene replacement."""
        if not self._engine:
            return

        # Clear editor-only native state first so the replacement frame cannot
        # reference stale scene objects through outline/gizmo paths.
        try:
            self._engine.clear_selection_outline()
        except Exception:
            pass

        try:
            self._engine.clear_component_gizmos()
        except Exception:
            pass

        try:
            self._engine.clear_component_gizmo_icons()
        except Exception:
            pass

        try:
            self._engine.wait_for_gpu_idle()
        except Exception as exc:
            Debug.log_warning(f"Failed to drain GPU before scene switch: {exc}")

    def _do_open_scene(self, path: str) -> bool:
        """Load a .scene file, replacing the current scene (no dirty guard)."""
        if not path or not os.path.isfile(path):
            Debug.log_warning(f"Scene file not found: {path}")
            return False

        if not self._is_under_assets(path):
            Debug.log_warning("Scene file must be under the project's Assets/ directory.")
            return False

        # Clear the RenderStack singleton before load — it's just a Python
        # class attribute and safe to nil out.  Registry / cache clearing must
        # wait until AFTER load_from_file() finishes, because C++ destroys old
        # GameObjects during Deserialize() and their PyComponentProxy::OnDestroy
        # callbacks still need to reach live Python objects.
        from InfEngine.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        from InfEngine.lib import SceneManager
        sm = SceneManager.instance()
        scene = sm.get_active_scene()

        if not scene:
            scene = sm.create_scene(DEFAULT_SCENE_NAME)

        self._prepare_native_scene_swap()

        if not scene.load_from_file(_safe_path(path)):
            Debug.log_error(f"Failed to load scene from: {path}")
            return False

        self._current_scene_path = os.path.abspath(path)
        self._dirty = False
        self._reset_undo_history(scene_is_dirty=False)

        # Now that C++ has finished destroying old objects, clear stale Python
        # registries and restore the new scene's Python components.
        try:
            self._restore_py_components(scene)
        except Exception as exc:
            Debug.log_error(f"Error restoring Python components: {exc}")

        self._restore_camera_state(self._current_scene_path)
        self._remember_last_scene(self._current_scene_path)

        Debug.log_internal(f"Scene loaded: {os.path.basename(path)}")
        if self._on_scene_changed:
            self._on_scene_changed()
        return True


    def _do_new_scene(self):
        """Create a blank scene with default Camera and Light (no dirty guard)."""
        from InfEngine.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        from InfEngine.lib import SceneManager
        sm = SceneManager.instance()

        scene = sm.get_active_scene()
        if not scene:
            scene = sm.create_scene(DEFAULT_SCENE_NAME)
        else:
            self._prepare_native_scene_swap()
            empty_json = json.dumps({
                "schema_version": 1,
                "name": DEFAULT_SCENE_NAME,
                "isPlaying": False,
                "objects": []
            })
            scene.deserialize(empty_json)

        # C++ has finished destroying old objects — now safe to clear Python
        # registries so stale entries don't accumulate.
        from InfEngine.components.component import InfComponent
        InfComponent._clear_all_instances()
        from InfEngine.components.builtin_component import BuiltinComponent
        BuiltinComponent._clear_cache()

        try:
            self._populate_default_objects(scene)
        except Exception as exc:
            Debug.log_error(f"Error populating default objects: {exc}")

        self._current_scene_path = None
        self._dirty = False
        self._reset_undo_history(scene_is_dirty=False)

        # Invalidate gizmos icon cache (scene objects are new)
        from InfEngine.gizmos.collector import notify_scene_changed
        notify_scene_changed()

        Debug.log_internal("New scene created")
        if self._on_scene_changed:
            self._on_scene_changed()

    @staticmethod
    def _populate_default_objects(scene) -> None:
        """Add a default Main Camera and Directional Light to *scene*.

        Called when creating a brand-new scene so the user doesn't start
        with a completely empty viewport.  Mirrors the Unity convention of
        providing a usable camera and a sun-like directional light by default.
        """
        from InfEngine.lib import LightType, LightShadows
        from InfEngine.math import Vector3

        # ---- Main Camera ----
        cam_obj = scene.create_game_object("Main Camera")
        cam_obj.tag = "MainCamera"
        cam_obj.add_component("Camera")
        cam_obj.transform.position = Vector3(0.0, 1.0, -10.0)

        # ---- Directional Light ----
        light_obj = scene.create_game_object("Directional Light")
        light_obj.transform.euler_angles = Vector3(50.0, -30.0, 0.0)
        light = light_obj.add_component("Light")
        if light is not None:
            light.light_type = LightType.Directional
            light.color = Vector3(1.0, 0.95, 0.9)
            light.intensity = 1.0
            light.shadows = LightShadows.Soft


    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_save(self, path: str) -> bool:
        """Actually write the scene to *path*."""
        from InfEngine.engine.ui.engine_status import EngineStatus
        ok = self._do_save_inner(path)
        if ok:
            EngineStatus.flash("保存完成 Saved ✓", 1.0, duration=1.5)
        else:
            EngineStatus.flash("保存失败 Save Failed", 0.0, duration=2.0)
        return ok

    def _do_save_inner(self, path: str) -> bool:
        """Internal save implementation.

        Serializes the scene on the main thread (fast, touches C++ scene graph),
        then writes the JSON to disk on a background thread so the editor stays
        responsive for large scenes.
        """
        if not self._is_under_assets(path):
            Debug.log_warning("Cannot save scene outside of Assets/ directory.")
            return False

        # Ensure .scene extension
        if not path.lower().endswith(SCENE_EXTENSION):
            path += SCENE_EXTENSION

        from InfEngine.lib import SceneManager
        sm = SceneManager.instance()
        scene = sm.get_active_scene()
        if not scene:
            Debug.log_warning("No active scene to save.")
            return False

        # Step 1 (main thread): serialize scene graph → JSON string
        try:
            json_str = scene.serialize()
        except Exception as exc:
            Debug.log_error(f"Failed to serialize scene: {exc}")
            return False

        if not json_str:
            Debug.log_error("Scene serialization returned empty data.")
            return False

        # Step 2 (background thread): write JSON to file
        import threading

        abs_path = os.path.abspath(path)
        write_error: list = []  # mutable container for thread result

        def _write():
            try:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            except Exception as exc:
                write_error.append(exc)

        t = threading.Thread(target=_write, daemon=True)
        t.start()
        t.join(timeout=10.0)  # generous timeout for large files

        if t.is_alive():
            Debug.log_error(f"Scene file write timed out: {abs_path}")
            return False

        if write_error:
            Debug.log_error(f"Failed to write scene file: {write_error[0]}")
            return False

        self._current_scene_path = abs_path
        self._dirty = False

        # Notify undo system of clean state
        from InfEngine.engine.undo import UndoManager
        mgr = UndoManager.instance()
        if mgr:
            mgr.mark_save_point()

        # Update scene name to match file
        scene.name = os.path.splitext(os.path.basename(path))[0]

        # Persist editor camera state for this scene
        self._save_camera_state(self._current_scene_path)

        self._remember_last_scene(self._current_scene_path)
        Debug.log_internal(f"Scene saved: {path}")
        return True

    def _reset_undo_history(self, scene_is_dirty: bool = False):
        """Reset undo/redo history to match the newly active scene state."""
        from InfEngine.engine.undo import UndoManager
        mgr = UndoManager.instance()
        if not mgr:
            return
        mgr.clear(scene_is_dirty=scene_is_dirty)
        mgr.sync_dirty_state()


    def _show_save_as_dialog(self):
        """Open a file dialog (on a background thread)."""
        root = get_project_root()
        if not root:
            Debug.log_warning("No project root set — cannot save scene.")
            return

        assets_dir = os.path.join(root, "Assets")
        os.makedirs(assets_dir, exist_ok=True)

        def _on_result(chosen_path: Optional[str]):
            if chosen_path:
                # Validate under Assets/
                if self._is_under_assets(chosen_path):
                    self._pending_save_path = chosen_path
                else:
                    Debug.log_warning("Scene must be saved under Assets/ directory.")
                    self._pending_save_path = None
            else:
                self._pending_save_path = None

        _show_save_dialog(assets_dir, _on_result)

    def _is_under_assets(self, path: str) -> bool:
        """Check if *path* is within the project's Assets/ directory."""
        root = get_project_root()
        if not root:
            return False
        assets = os.path.normcase(os.path.abspath(os.path.join(root, "Assets")))
        target = os.path.normcase(os.path.abspath(path))
        return target.startswith(assets + os.sep) or target == assets

    def _remember_last_scene(self, path: str):
        settings = _load_editor_settings()
        settings["lastOpenedScene"] = path
        _save_editor_settings(settings)

    def _save_camera_state(self, scene_path: str):
        """Save current editor camera state for the given scene path."""
        if not self._engine or not scene_path:
            return
        cam = self._engine.editor_camera
        if not cam:
            return
        pos = cam.position
        rot = cam.rotation
        fp = cam.focus_point
        fd = cam.focus_distance
        state = {
            "position": [pos.x, pos.y, pos.z],
            "focusPoint": [fp.x, fp.y, fp.z],
            "focusDistance": fd,
            "yaw": rot[0],
            "pitch": rot[1],
        }
        settings = _load_editor_settings()
        if "sceneCameraStates" not in settings:
            settings["sceneCameraStates"] = {}
        key = os.path.normcase(os.path.abspath(scene_path))
        settings["sceneCameraStates"][key] = state
        _save_editor_settings(settings)

    def _restore_camera_state(self, scene_path: str):
        """Restore editor camera state for the given scene path."""
        if not self._engine or not scene_path:
            return
        cam = self._engine.editor_camera
        if not cam:
            return
        settings = _load_editor_settings()
        states = settings.get("sceneCameraStates", {})
        key = os.path.normcase(os.path.abspath(scene_path))
        state = states.get(key)
        if not state:
            return
        p = state["position"]
        f = state["focusPoint"]
        cam.restore_state(
            p[0], p[1], p[2],
            f[0], f[1], f[2],
            state["focusDistance"],
            state["yaw"],
            state["pitch"],
        )

    # ------------------------------------------------------------------
    # Python component serialization helpers
    # ------------------------------------------------------------------

    def _restore_py_components(self, scene):
        """After loading, recreate Python component instances from pending data."""

        # Always clear registries — the old scene's objects are gone regardless
        # of whether the new scene has Python components.
        from InfEngine.components.component import InfComponent
        InfComponent._clear_all_instances()
        from InfEngine.components.builtin_component import BuiltinComponent
        BuiltinComponent._clear_cache()

        # Notify GizmosCollector that the scene has changed (icon cache stale).
        from InfEngine.gizmos.collector import notify_scene_changed
        notify_scene_changed()

        has_pending = scene.has_pending_py_components()
        if not has_pending:
            return

        pending = scene.take_pending_py_components()
        if not pending:
            return

        restored = 0
        for pc in pending:
            try:
                self._restore_single_py_component(scene, pc)
                restored += 1
            except Exception as exc:
                Debug.log_error(
                    f"Failed to restore component '{pc.type_name}' on "
                    f"GameObject {pc.game_object_id}: {exc}"
                )
                # Create BrokenComponent so the data is preserved on save
                try:
                    go = scene.find_by_id(pc.game_object_id)
                    if go:
                        from InfEngine.components.component import BrokenComponent
                        broken = BrokenComponent()
                        broken._broken_type_name = pc.type_name
                        broken._script_guid = pc.script_guid
                        broken._broken_fields_json = pc.fields_json or "{}"
                        broken._broken_error = f"Restore failed: {exc}"
                        broken.enabled = pc.enabled
                        go.add_py_component(broken)
                except Exception:
                    pass

        Debug.log_internal(f"Restored {restored}/{len(pending)} Python component(s) from scene file")

    def _restore_single_py_component(self, scene, pc):
        """Restore one pending Python component. May raise on failure."""
        go = scene.find_by_id(pc.game_object_id)
        if not go:
            Debug.log_warning(
                f"Cannot restore component '{pc.type_name}': "
                f"GameObject {pc.game_object_id} not found"
            )
            return

        # Resolve script path from GUID
        script_path = None
        if pc.script_guid and self._asset_database:
            script_path = self._asset_database.get_path_from_guid(pc.script_guid)

        # In packaged builds .py sources are compiled to .pyc and removed.
        if script_path and not os.path.exists(script_path) and script_path.endswith('.py'):
            pyc_path = script_path + 'c'
            if os.path.exists(pyc_path):
                script_path = pyc_path

        # Packaged-build fallback: use build-time GUID manifest
        if not script_path and pc.script_guid:
            from InfEngine.engine.project_context import resolve_guid_to_path
            script_path = resolve_guid_to_path(pc.script_guid)

        instance = None
        if not script_path:
            # Fallback: try to find by type name in registry
            from InfEngine.components.registry import get_type
            comp_class = get_type(pc.type_name)
            if comp_class:
                instance = comp_class()
            # else: instance stays None → will become BrokenComponent below
        else:
            from InfEngine.components.script_loader import load_and_create_component
            instance = load_and_create_component(
                script_path, asset_database=self._asset_database
            )

        if instance is None:
            # Script failed to load — create a BrokenComponent placeholder
            # that preserves the serialized data.
            from InfEngine.components.component import BrokenComponent
            from InfEngine.components.script_loader import get_script_error_by_path
            broken = BrokenComponent()
            broken._broken_type_name = pc.type_name
            broken._script_guid = pc.script_guid
            broken._broken_fields_json = pc.fields_json or "{}"
            error_msg = None
            if script_path:
                error_msg = get_script_error_by_path(script_path)
            if not error_msg:
                error_msg = (
                    f"Cannot find script for component '{pc.type_name}' "
                    f"(guid={pc.script_guid})"
                    if not script_path
                    else f"Script '{script_path}' contains no InfComponent subclass "
                         f"for '{pc.type_name}'"
                )
            broken._broken_error = error_msg
            broken.enabled = pc.enabled
            go.add_py_component(broken)
            Debug.log_warning(
                f"Component '{pc.type_name}' on '{go.name}' loaded as broken "
                f"placeholder — fix the script to restore functionality."
            )
            return

        # Apply serialized fields
        if pc.fields_json:
            instance._deserialize_fields(pc.fields_json)

        # Set enabled state
        instance.enabled = pc.enabled

        # Attach to GameObject
        go.add_py_component(instance)

        # Call lifecycle hook
        instance._call_on_after_deserialize()

