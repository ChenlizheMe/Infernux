"""
Scene file management for Infernux.

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

from Infernux.debug import Debug
from Infernux.engine.project_context import get_project_root
from Infernux.engine.path_utils import safe_path as _safe_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_EXTENSION = ".scene"
EDITOR_SETTINGS_FILE = "EditorSettings.json"
DEFAULT_SCENE_NAME = "Untitled Scene"
DEFAULT_SCENE_FILE_BASE = "UntitledScene"
PREFAB_MODE_SCENE_NAME = "__PrefabMode__"
PREFAB_RESTORE_SCENE_NAME = "__PrefabRestore__"

# ImGuiKey enum values (matches imgui.h)
KEY_S = 564          # S
KEY_LEFT_CTRL = 527  # Left Ctrl
KEY_RIGHT_CTRL = 531 # Right Ctrl


def _empty_scene_json(name: str) -> str:
    return json.dumps({
        "schema_version": 1,
        "name": name,
        "isPlaying": False,
        "objects": [],
    })


def _get_scene_root_objects(scene):
    if scene is None:
        return []
    if hasattr(scene, "get_root_objects"):
        roots = scene.get_root_objects()
        return roots if roots is not None else []
    if hasattr(scene, "get_root_game_objects"):
        roots = scene.get_root_game_objects()
        return roots if roots is not None else []
    return []


# ---------------------------------------------------------------------------
# Editor settings helpers (ProjectSettings/EditorSettings.json)
# ---------------------------------------------------------------------------

def _settings_path() -> Optional[str]:
    root = _effective_project_root()
    if not root:
        return None
    return os.path.join(root, "ProjectSettings", EDITOR_SETTINGS_FILE)


def _effective_project_root() -> Optional[str]:
    """Best-effort project-root resolution for editor/runtime edge cases."""
    root = get_project_root()
    if root and os.path.isdir(root):
        return root

    try:
        from Infernux.engine.ui.editor_services import EditorServices
        services = EditorServices.instance()
        if services and services.project_path and os.path.isdir(services.project_path):
            return os.path.abspath(services.project_path)
    except Exception:
        pass

    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "Assets")):
        return cwd
    return None


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

def _show_save_dialog(initial_dir: str, callback: Callable[[Optional[str]], None],
                      default_filename: str = "Untitled Scene.scene"):
    """Show a native save-file dialog. *callback* receives the chosen path or None."""
    def _run():
        result: Optional[str] = None
        try:
            if os.name == "nt":
                result = _win32_save_dialog(initial_dir, default_filename)
        except Exception as exc:
            Debug.log_warning(f"Save dialog unavailable on this platform: {exc}")
        callback(result)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _win32_save_dialog(initial_dir: str, default_filename: str = "Untitled Scene.scene") -> Optional[str]:
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

    default_filename = (default_filename or "Untitled Scene.scene").strip() or "Untitled Scene.scene"
    for ch in '<>:"/\\|?*':
        default_filename = default_filename.replace(ch, '_')

    default_target = os.path.join(initial_dir, default_filename)

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    buf.value = default_target
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
        self._deferred_exit_prefab: bool = False           # True → exit prefab mode task pending

        # True while _do_open_scene / _do_new_scene is running.
        # Prevents stacking deferred loads from rapid user clicks.
        self._load_in_progress: bool = False

        # Guard against repeated request_close() calls while a close
        # confirmation dialog is already visible.  Without this,
        # is_close_requested() being True every frame would re-trigger
        # _request_save_confirmation() and re-open the ImGui popup every
        # frame, making the buttons unclickable.
        self._close_in_progress: bool = False

        # Prefab Mode state
        self.is_prefab_mode = False
        self.prefab_mode_path = None
        self.prefab_envelope = {}
        self._previous_scene_path = None
        self._previous_scene_dirty = False
        self._previous_scene_json = ""

    @classmethod
    def instance(cls) -> Optional["SceneFileManager"]:
        return cls._instance

    def set_asset_database(self, asset_db):
        """Set the AssetDatabase for GUID→path resolution during scene load."""
        self._asset_database = asset_db

    def set_engine(self, engine):
        """Set the native Infernux reference (for close-request handling)."""
        self._engine = engine

    def _native_engine_for_close(self):
        """Return native engine for close confirmation, with service fallback."""
        if self._engine is not None:
            return self._engine
        try:
            from Infernux.engine.ui.editor_services import EditorServices
            services = EditorServices.instance()
            native = services.native_engine if services else None
            if native is not None:
                self._engine = native
            return native
        except Exception:
            return None

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
        return self._deferred_load_path is not None or self._deferred_new_scene or self._deferred_exit_prefab

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
        from Infernux.engine.play_mode import PlayModeManager, PlayModeState
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

        # In Prefab Mode, Ctrl+S is ignored — prefab auto-saves on exit.
        if self.is_prefab_mode:
            return False

        if self._current_scene_path:
            return self._do_save(self._current_scene_path)

        # No file yet — show a Save As dialog
        self._show_save_as_dialog()
        return False

    def _save_prefab(self) -> bool:
        """Save the currently-edited prefab in Prefab Mode."""
        if not self.prefab_mode_path:
            Debug.log_warning("No prefab path in Prefab Mode.")
            return False

        from Infernux.lib import SceneManager
        from Infernux.engine.prefab_manager import _strip_prefab_fields, _strip_prefab_runtime_fields

        scene = SceneManager.instance().get_active_scene()
        roots = _get_scene_root_objects(scene)
        if not roots:
            Debug.log_warning("No root objects in Prefab Mode scene.")
            return False

        prefab_root = roots[0]
        try:
            root_data = json.loads(prefab_root.serialize())
        except Exception as exc:
            Debug.log_error(f"Failed to serialize prefab root: {exc}")
            return False

        _strip_prefab_fields(root_data)
        _strip_prefab_runtime_fields(root_data)

        envelope = dict(self.prefab_envelope) if isinstance(self.prefab_envelope, dict) else {}
        envelope["root_object"] = root_data

        try:
            with open(self.prefab_mode_path, 'w', encoding='utf-8') as f:
                json.dump(envelope, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            Debug.log_error(f"Failed to save prefab: {exc}")
            return False

        self._dirty = False
        Debug.log_internal(f"Prefab saved: {self.prefab_mode_path}")
        return True

    def save_scene_as(self):
        """Force a Save-As dialog regardless of whether a path exists."""
        if self._is_play_mode():
            Debug.log_warning("Cannot save scene while in Play mode.")
            return
        self._show_save_as_dialog()


    def open_prefab_mode(self, prefab_path: str):
        """Enter Prefab Mode, pushing the current scene to memory."""
        if self.is_prefab_mode or not prefab_path or not os.path.isfile(prefab_path):
            return False

        if self._is_play_mode():
            Debug.log_warning("Cannot enter Prefab Mode while in Play mode.")
            return False

        from Infernux.lib import SceneManager
        from Infernux.engine.prefab_manager import _restore_pending_py_components, _strip_prefab_runtime_fields
        from Infernux.engine.ui.selection_manager import SelectionManager

        active_scene = SceneManager.instance().get_active_scene()
        if active_scene is None:
            Debug.log_warning("No active scene available for Prefab Mode.")
            return False

        try:
            with open(prefab_path, "r", encoding="utf-8") as f:
                prefab_data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            Debug.log_error(f"Failed to open prefab for Prefab Mode: {exc}")
            return False

        root_obj_data = prefab_data.get("root_object")
        if root_obj_data is None:
            Debug.log_error("Invalid prefab file: missing 'root_object'.")
            return False

        # Validate prefab version
        from Infernux.engine.prefab_manager import PREFAB_VERSION
        file_version = prefab_data.get("prefab_version", 0)
        if file_version > PREFAB_VERSION:
            Debug.log_error(
                f"Prefab '{prefab_path}' uses version {file_version} but this "
                f"engine only supports up to version {PREFAB_VERSION}."
            )
            return False

        root_obj_data = json.loads(json.dumps(root_obj_data))
        _strip_prefab_runtime_fields(root_obj_data)

        self._previous_scene_json = active_scene.serialize()
        self._previous_scene_path = self._current_scene_path
        self._previous_scene_dirty = self._dirty
        self.prefab_envelope = prefab_data

        # Clear the RenderStack singleton before the swap — matches the
        # pattern in _do_open_scene / _do_new_scene to avoid stale refs.
        from Infernux.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        self._prepare_native_scene_swap()

        # Destroy ALL objects in the original scene so their physics bodies
        # are removed from the global PhysicsWorld.  Without this, invisible
        # colliders from the main scene interfere with the prefab scene.
        active_scene.deserialize(_empty_scene_json(active_scene.name))

        sm = SceneManager.instance()
        new_scene = sm.get_scene(PREFAB_MODE_SCENE_NAME)
        if new_scene is None:
            new_scene = sm.create_scene(PREFAB_MODE_SCENE_NAME)
        # Always deserialize empty JSON to clear old objects and component
        # registries (MeshRenderer, physics, etc.) — prevents the previous
        # scene's renderers from leaking into Prefab Mode.
        new_scene.deserialize(_empty_scene_json(PREFAB_MODE_SCENE_NAME))
        sm.set_active_scene(new_scene)

        root_json = json.dumps(root_obj_data)
        root_obj = new_scene.instantiate_from_json(root_json)
        if root_obj is None:
            Debug.log_error("Failed to instantiate prefab in Prefab Mode.")
            return False

        if new_scene.has_pending_py_components():
            try:
                _restore_pending_py_components(new_scene, self._asset_database)
            except Exception as exc:
                Debug.log_error(f"Failed to restore prefab Python components: {exc}")

        roots = _get_scene_root_objects(new_scene)
        if roots:
            SelectionManager.instance().select(roots[0].id)

        self.is_prefab_mode = True
        self.prefab_mode_path = os.path.abspath(prefab_path)
        self._current_scene_path = prefab_path
        self._dirty = False
        self._reset_undo_history(scene_is_dirty=False)

        if self._on_scene_changed:
            self._on_scene_changed()
        return True

    def exit_prefab_mode(self):
        """Schedule exit from Prefab Mode on a later frame.

        Uses ``DeferredTaskRunner`` instead of ``poll_deferred_load`` so the
        exit cannot be consumed again later in the same GUI frame. This avoids
        tearing down the prefab scene while its resources may still be in use
        by the just-submitted frame.
        """
        if not self.is_prefab_mode:
            return False
        if self._deferred_exit_prefab:
            return True

        from Infernux.engine.deferred_task import DeferredTaskRunner

        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            Debug.log_warning("Cannot exit Prefab Mode: a deferred task is already running")
            return False

        self._deferred_exit_prefab = True
        runner.submit(
            "Exit Prefab Mode",
            [("Exiting Prefab Mode...", 0.6, self._run_deferred_exit_prefab_task)],
        )
        return True

    def _run_deferred_exit_prefab_task(self):
        """DeferredTaskRunner step wrapper for prefab-mode exit."""
        self._deferred_exit_prefab = False
        return self._do_exit_prefab_mode()

    def _do_exit_prefab_mode(self):
        """Internal: perform the actual Prefab Mode exit (called by poll_deferred_load)."""
        if not self.is_prefab_mode:
            return False

        from Infernux.lib import SceneManager
        from Infernux.engine.prefab_manager import _restore_pending_py_components

        # Always save the prefab on exit
        if self.prefab_mode_path:
            self._save_prefab()

        # Always resolve the prefab GUID so instances can be refreshed,
        # regardless of whether the save happened now or earlier via Ctrl+S.
        saved_prefab_guid = None
        if self.prefab_mode_path and self._asset_database:
            try:
                saved_prefab_guid = self._asset_database.get_guid_from_path(
                    self.prefab_mode_path
                ) or None
            except Exception:
                pass

        # Clear the RenderStack singleton before the swap — matches the
        # pattern in _do_open_scene / _do_new_scene to avoid stale refs.
        from Infernux.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        self._prepare_native_scene_swap()

        sm = SceneManager.instance()

        # Destroy all objects in the prefab scene FIRST so their physics
        # bodies (Colliders, Rigidbodies) are removed from the global
        # PhysicsWorld before we restore the main scene.
        prefab_scene = sm.get_scene(PREFAB_MODE_SCENE_NAME)
        if prefab_scene is not None:
            prefab_scene.deserialize(_empty_scene_json(PREFAB_MODE_SCENE_NAME))

        scene = sm.get_scene(PREFAB_RESTORE_SCENE_NAME)
        if scene is None:
            scene = sm.create_scene(PREFAB_RESTORE_SCENE_NAME)
        # Always deserialize empty first so ClearComponentRegistries runs,
        # preventing prefab scene renderers from leaking into the restored scene.
        scene.deserialize(_empty_scene_json(PREFAB_RESTORE_SCENE_NAME))
        sm.set_active_scene(scene)

        if self._previous_scene_json:
            scene.deserialize(self._previous_scene_json)
            if scene.has_pending_py_components():
                try:
                    _restore_pending_py_components(scene, self._asset_database)
                except Exception as exc:
                    Debug.log_error(f"Failed to restore scene Python components: {exc}")

        # Refresh instances of the edited prefab so changes propagate
        if saved_prefab_guid:
            self._refresh_prefab_instances(
                scene, saved_prefab_guid, self.prefab_mode_path,
                self._asset_database
            )

        self.is_prefab_mode = False
        self.prefab_mode_path = None
        self._current_scene_path = self._previous_scene_path
        self._dirty = self._previous_scene_dirty
        self.prefab_envelope = {}
        self._previous_scene_json = ""
        self._previous_scene_dirty = False
        self._previous_scene_path = None
        self._reset_undo_history(scene_is_dirty=self._dirty)

        if self._on_scene_changed:
            self._on_scene_changed()
        return True

    # ------------------------------------------------------------------
    # Prefab instance refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_prefab_instances(scene, prefab_guid: str, prefab_path: str,
                                  asset_database=None):
        """Re-instantiate all instances of a prefab to pick up updated data.

        Iterates root objects, finds those whose *prefab_guid* matches,
        then replaces them in-place (preserving only the root's local position).
        """
        from Infernux.engine.prefab_manager import instantiate_prefab

        if not prefab_guid or not prefab_path:
            return

        roots = _get_scene_root_objects(scene)
        if not roots:
            return

        def _collect_instances(objects):
            found = []
            for obj in objects:
                guid = getattr(obj, 'prefab_guid', '')
                is_root = getattr(obj, 'prefab_root', False)
                if guid == prefab_guid and is_root:
                    found.append(obj)
                else:
                    children = list(obj.get_children()) if hasattr(obj, 'get_children') else []
                    found.extend(_collect_instances(children))
            return found

        instances = _collect_instances(roots)
        if not instances:
            return

        Debug.log_internal(
            f"Refreshing {len(instances)} prefab instance(s) for GUID={prefab_guid}"
        )

        for old_obj in instances:
            try:
                parent = old_obj.get_parent() if hasattr(old_obj, 'get_parent') else None
                # Preserve the instance's full local transform (position,
                # rotation, scale) — each instance keeps its own placement.
                tf = old_obj.transform
                local_pos = tf.local_position if tf else None
                local_rot = tf.local_rotation if tf else None
                local_scl = tf.local_scale if tf else None

                new_obj = instantiate_prefab(
                    file_path=prefab_path,
                    scene=scene,
                    parent=parent,
                    asset_database=asset_database,
                )
                if new_obj:
                    new_tf = new_obj.transform
                    if new_tf:
                        if local_pos is not None:
                            new_tf.local_position = local_pos
                        if local_rot is not None:
                            new_tf.local_rotation = local_rot
                        if local_scl is not None:
                            new_tf.local_scale = local_scl

                scene.destroy_game_object(old_obj)
            except Exception as exc:
                Debug.log_warning(f"Failed to refresh prefab instance: {exc}")

    def sync_all_prefab_instances(self, scene=None):
        """Sync every prefab instance in *scene* to its latest on-disk data.

        Called after scene load and after exiting Prefab Mode so that all
        prefab instances reflect the most recent prefab files.
        """
        if scene is None:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
        if scene is None or not self._asset_database:
            return

        roots = _get_scene_root_objects(scene)
        if not roots:
            return

        # Collect unique (prefab_guid → prefab_path) pairs
        guid_to_path: dict[str, str] = {}

        def _walk(objects):
            for obj in objects:
                guid = getattr(obj, 'prefab_guid', '')
                is_root = getattr(obj, 'prefab_root', False)
                if guid and is_root and guid not in guid_to_path:
                    try:
                        p = self._asset_database.get_path_from_guid(guid)
                        if p and os.path.isfile(p):
                            guid_to_path[guid] = p
                    except Exception:
                        pass
                children = list(obj.get_children()) if hasattr(obj, 'get_children') else []
                _walk(children)

        _walk(roots)

        for guid, path in guid_to_path.items():
            self._refresh_prefab_instances(
                scene, guid, path, self._asset_database
            )

    def open_scene(self, path: str) -> bool:
        """Load a .scene file, replacing the current scene.

        If the current scene is dirty, shows a save-confirmation popup first.
        The actual load is deferred to the next frame so the scene view can
        stop rendering old 3D content first.
        """
        if self._load_in_progress:
            Debug.log_warning("Scene load already in progress — ignoring open_scene()")
            return False
        if self.is_prefab_mode:
            # Auto-save prefab and schedule exit.  The deferred exit runs
            # before the deferred open in poll_deferred_load, so the
            # original scene is restored first.
            if self.prefab_mode_path:
                self._save_prefab()
            self.exit_prefab_mode()
            if self._previous_scene_path:
                self._save_camera_state(self._previous_scene_path)
            if self._previous_scene_dirty:
                self._request_save_confirmation('open', path)
                return False
            self._begin_deferred_open(path)
            return True

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
        if self.is_prefab_mode:
            if self.prefab_mode_path:
                self._save_prefab()
            self.exit_prefab_mode()
            if self._previous_scene_path:
                self._save_camera_state(self._previous_scene_path)
            if self._previous_scene_dirty:
                self._request_save_confirmation('new')
                return
            self._begin_deferred_new()
            return

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

        # During play mode, close immediately — the save dialog's "Save"
        # button cannot save the pre-play state properly.
        if self._is_play_mode():
            native = self._native_engine_for_close()
            if native:
                native.confirm_close()
            return

        # In Prefab Mode: auto-save the prefab, schedule the deferred exit
        # back to the original scene, then decide based on the *original*
        # scene's dirty state.  By the time the user interacts with the
        # save dialog (next frame), _do_exit_prefab_mode has already
        # restored the original scene so _do_save operates on it.
        if self.is_prefab_mode:
            if self.prefab_mode_path:
                self._save_prefab()
            if self._previous_scene_path:
                self._save_camera_state(self._previous_scene_path)
            self.exit_prefab_mode()  # deferred
            if self._previous_scene_dirty:
                self._request_save_confirmation('close')
            else:
                native = self._native_engine_for_close()
                if native:
                    native.confirm_close()
            return

        # Always persist camera state before closing
        if self._current_scene_path:
            self._save_camera_state(self._current_scene_path)

        if self._dirty:
            self._request_save_confirmation('close')
        else:
            native = self._native_engine_for_close()
            if native:
                native.confirm_close()

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
                    # Save failed — cancel pending close/open/new chain so user can retry.
                    if self._post_save_callback is not None:
                        if self._pending_action == 'close' and self._engine:
                            self._engine.cancel_close()
                        self._close_in_progress = False
                        self._clear_pending_action()
                    self._post_save_callback = None
            else:
                # User cancelled the Save As dialog — cancel pending close/open/new chain.
                if self._post_save_callback is not None:
                    if self._pending_action == 'close' and self._engine:
                        self._engine.cancel_close()
                    self._close_in_progress = False
                    self._clear_pending_action()
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
        """Execute a pending deferred scene load/new/prefab-exit.

        Must be called every frame (from menu_bar).  The one-frame delay
        between _begin_deferred_open/new and this method gives the
        current frame's GPU submission a chance to complete before
        _prepare_native_scene_swap() calls WaitForGpuIdle(), which
        performs a full vkDeviceWaitIdle + FlushDeletionQueue.

        The old scene's texture naturally remains in the render target
        until the new scene's first Execute() overwrites it, so no
        placeholder or extra-frame delay is needed.
        """
        if self._load_in_progress:
            return
        if self._deferred_load_path is not None:
            path = self._deferred_load_path
            self._deferred_load_path = None
            self._load_in_progress = True
            try:
                self._do_open_scene(path)
            except Exception as exc:
                Debug.log_error(f"Scene load failed: {exc}")
            finally:
                self._load_in_progress = False
        elif self._deferred_new_scene:
            self._deferred_new_scene = False
            self._load_in_progress = True
            try:
                self._do_new_scene()
            except Exception as exc:
                Debug.log_error(f"New scene failed: {exc}")
            finally:
                self._load_in_progress = False

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
        elif action == 'close':
            native = self._native_engine_for_close()
            if native:
                native.confirm_close()
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
                    action = self._pending_action
                    if self._do_save(self._current_scene_path):
                        if not self._execute_pending_action():
                            native = self._native_engine_for_close()
                            if native and action == 'close':
                                native.confirm_close()
                    else:
                        native = self._native_engine_for_close()
                        if self._pending_action == 'close' and native:
                            native.cancel_close()
                        self._close_in_progress = False
                        self._clear_pending_action()
                else:
                    # On close with an untitled scene, auto-save into Assets/
                    # to avoid Save-As dialog platform differences.
                    if self._pending_action == 'close':
                        default_path = self._default_scene_save_path()
                        if default_path and self._do_save(default_path):
                            if not self._execute_pending_action():
                                native = self._native_engine_for_close()
                                if native:
                                    native.confirm_close()
                        else:
                            native = self._native_engine_for_close()
                            if native:
                                native.cancel_close()
                            self._close_in_progress = False
                            self._clear_pending_action()
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
                native = self._native_engine_for_close()
                if self._pending_action == 'close' and native:
                    native.cancel_close()
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

    def _pump_events_safe(self):
        """Pump the OS message queue to prevent Windows 'Not Responding'."""
        if self._engine:
            try:
                self._engine.pump_events()
            except Exception:
                pass

    def _prepare_native_scene_swap(self):
        """Clear native editor state and drain GPU work before scene replacement."""
        if not self._engine:
            return

        # Clear editor-only native state first so the replacement frame cannot
        # reference stale scene objects through outline/gizmo paths.
        try:
            self._engine.clear_selection_outline()
        except Exception as exc:
            Debug.log_warning(f"Failed to clear selection outline: {exc}")

        try:
            self._engine.clear_component_gizmos()
        except Exception as exc:
            Debug.log_warning(f"Failed to clear component gizmos: {exc}")

        try:
            self._engine.clear_component_gizmo_icons()
        except Exception as exc:
            Debug.log_warning(f"Failed to clear gizmo icons: {exc}")

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
        from Infernux.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        from Infernux.lib import SceneManager
        sm = SceneManager.instance()
        scene = sm.get_active_scene()

        if not scene:
            scene = sm.create_scene(DEFAULT_SCENE_NAME)

        self._prepare_native_scene_swap()

        if not scene.load_from_file(_safe_path(path)):
            Debug.log_error(f"Failed to load scene from: {path}")
            return False

        # Pump OS message queue after the heavy C++ deserialization + GPU
        # uploads so Windows doesn't flag the app as Not Responding.
        self._pump_events_safe()

        self._current_scene_path = os.path.abspath(path)
        self._dirty = False
        self._reset_undo_history(scene_is_dirty=False)

        # Now that C++ has finished destroying old objects, clear stale Python
        # registries and restore the new scene's Python components.
        try:
            self._restore_py_components(scene)
        except Exception as exc:
            Debug.log_error(f"Error restoring Python components: {exc}")

        self._pump_events_safe()

        self._restore_camera_state(self._current_scene_path)
        self._remember_last_scene(self._current_scene_path)

        # Sync all prefab instances to the latest on-disk prefab data
        self.sync_all_prefab_instances(scene)

        Debug.log_internal(f"Scene loaded: {os.path.basename(path)}")
        if self._on_scene_changed:
            self._on_scene_changed()
        return True


    def _do_new_scene(self):
        """Create a blank scene with default Camera and Light (no dirty guard)."""
        from Infernux.renderstack.render_stack import RenderStack
        RenderStack._active_instance = None

        from Infernux.lib import SceneManager
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
        from Infernux.components.component import InxComponent
        InxComponent._clear_all_instances()
        from Infernux.components.builtin_component import BuiltinComponent
        BuiltinComponent._clear_cache()

        try:
            self._populate_default_objects(scene)
        except Exception as exc:
            Debug.log_error(f"Error populating default objects: {exc}")

        self._current_scene_path = None
        self._dirty = False
        self._reset_undo_history(scene_is_dirty=False)

        # Invalidate gizmos icon cache (scene objects are new)
        from Infernux.gizmos.collector import notify_scene_changed
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
        from Infernux.lib import LightType, LightShadows
        from Infernux.math import Vector3

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
            light.shadow_bias = 0.0
  

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_save(self, path: str) -> bool:
        """Actually write the scene to *path*."""
        from Infernux.engine.ui.engine_status import EngineStatus
        ok = self._do_save_inner(path)
        if ok:
            EngineStatus.flash("保存完成 Saved", 1.0, duration=1.5)
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

        from Infernux.lib import SceneManager
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
        from Infernux.engine.undo import UndoManager
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
        from Infernux.engine.undo import UndoManager
        mgr = UndoManager.instance()
        if not mgr:
            return
        mgr.clear(scene_is_dirty=scene_is_dirty)
        mgr.sync_dirty_state()


    def _default_scene_save_path(self) -> Optional[str]:
        """Return a unique default scene path under Assets/ for untitled saves."""
        root = _effective_project_root()
        if not root:
            return None

        assets_dir = os.path.join(root, "Assets")
        os.makedirs(assets_dir, exist_ok=True)

        base_name = DEFAULT_SCENE_FILE_BASE
        candidate = os.path.join(assets_dir, f"{base_name}{SCENE_EXTENSION}")
        if not os.path.exists(candidate):
            return candidate

        index = 1
        while True:
            candidate = os.path.join(assets_dir, f"{base_name} {index}{SCENE_EXTENSION}")
            if not os.path.exists(candidate):
                return candidate
            index += 1


    def _show_save_as_dialog(self):
        """Open a file dialog (on a background thread)."""
        root = _effective_project_root()
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
                    self._pending_save_path = ""  # cancel sentinel
            else:
                self._pending_save_path = ""  # cancel sentinel

        if self._current_scene_path:
            default_filename = os.path.basename(self._current_scene_path)
        else:
            default_filename = f"{DEFAULT_SCENE_FILE_BASE}{SCENE_EXTENSION}"
        _show_save_dialog(assets_dir, _on_result, default_filename)

    def _is_under_assets(self, path: str) -> bool:
        """Check if *path* is within the project's Assets/ directory."""
        root = _effective_project_root()
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
        from Infernux.engine.component_restore import restore_pending_py_components
        restore_pending_py_components(
            scene,
            asset_database=self._asset_database,
            clear_registries=True,
        )

