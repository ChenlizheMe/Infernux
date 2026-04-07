"""
PlayMode - Runtime/Editor mode manager for Infernux.

Manages the play mode state machine:
- Edit Mode: Normal editor state, scene changes are persistent
- Play Mode: Runtime simulation, scene changes are temporary
- Pause Mode: Runtime paused, can step frame by frame

Handles:
- Scene state save/restore for play mode isolation (Unity-style)
- Delta time management
- Python component recreation after scene restore
"""

import time
import os
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass
from Infernux.debug import Debug, LogType
from Infernux.engine.project_context import resolve_script_path

if TYPE_CHECKING:
    from Infernux.lib import SceneManager, Scene, GameObject
    from Infernux.components.component import InxComponent


class PlayModeState(Enum):
    """Play mode states."""
    EDIT = auto()      # Normal editor mode
    PLAYING = auto()   # Runtime playing
    PAUSED = auto()    # Runtime paused


@dataclass
class PlayModeEvent:
    """Event data for play mode state changes."""
    old_state: PlayModeState
    new_state: PlayModeState
    timestamp: float


def _get_scene_manager():
    """Get the SceneManager singleton from C++ bindings."""
    from Infernux.lib import SceneManager
    return SceneManager.instance()

from ._play_mode_serialization import PlayModeSerializationMixin


class PlayModeManager(PlayModeSerializationMixin):
    """
    Manages the runtime/editor play mode.
    
    Implements Unity-style scene isolation:
    - On Play: Serialize entire scene state (C++ objects + Python components)
    - During Play: All changes are runtime-only
    - On Stop: Deserialize to restore original scene state
    
    Handles:
    - State transitions (Edit ↔ Play ↔ Pause)
    - Scene state save/restore via C++ serialization
    - Python component recreation after restore
    - Timing for UI display
    - (Lifecycle is driven by C++)
    
    Usage:
        play_mode = PlayModeManager()
        
        # Start play mode
        play_mode.enter_play_mode()
        
        # In game loop
        play_mode.tick(delta_time)
        
        # Stop and restore
        play_mode.exit_play_mode()
    """
    
    _instance: Optional['PlayModeManager'] = None
    
    def __init__(self):
        self._state = PlayModeState.EDIT
        
        # Timing
        self._last_frame_time: float = 0.0
        self._delta_time: float = 0.0
        self._time_scale: float = 1.0
        self._total_play_time: float = 0.0
        
        # Scene state backup (JSON string from C++ Scene::Serialize)
        self._scene_backup: Optional[str] = None
        # Original scene file path (to restore correct scene on Stop)
        self._scene_path_backup: Optional[str] = None
        self._scene_dirty_backup: bool = False
        
        # Event listeners
        self._state_change_listeners: List[Callable[[PlayModeEvent], None]] = []
        
        # Store singleton reference
        PlayModeManager._instance = self

        # Asset database for GUID-based script lookup
        self._asset_database = None
        self._runtime_hidden_object_ids: set[int] = set()

        # C++ engine handle for renderer-level play mode signalling
        self._native_engine = None
    
    @classmethod
    def instance(cls) -> Optional['PlayModeManager']:
        """Get the singleton instance if it exists."""
        return cls._instance
    
    def _get_scene_manager(self):
        """Get the SceneManager singleton."""
        return _get_scene_manager()

    def set_asset_database(self, asset_database):
        """Set AssetDatabase for GUID-based script resolution."""
        self._asset_database = asset_database

    def clear_runtime_hidden_object_ids(self):
        self._runtime_hidden_object_ids.clear()

    def register_runtime_hidden_object(self, game_object) -> None:
        if game_object is None:
            return
        try:
            object_id = int(game_object.id)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return
        if object_id > 0:
            self._runtime_hidden_object_ids.add(object_id)

    def get_runtime_hidden_object_ids(self) -> set[int]:
        return set(self._runtime_hidden_object_ids)

    def is_runtime_hidden_object_id(self, object_id: int) -> bool:
        try:
            return int(object_id) in self._runtime_hidden_object_ids
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return False
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def state(self) -> PlayModeState:
        """Current play mode state."""
        return self._state
    
    @property
    def is_playing(self) -> bool:
        """True if in play or paused mode."""
        return self._state in (PlayModeState.PLAYING, PlayModeState.PAUSED)
    
    @property
    def is_paused(self) -> bool:
        """True if currently paused."""
        return self._state == PlayModeState.PAUSED
    
    @property
    def is_edit_mode(self) -> bool:
        """True if in edit mode."""
        return self._state == PlayModeState.EDIT
    
    @property
    def delta_time(self) -> float:
        """Time since last frame in seconds."""
        return self._delta_time
    
    @property
    def time_scale(self) -> float:
        """Time scale factor (1.0 = normal speed)."""
        return self._time_scale
    
    @time_scale.setter
    def time_scale(self, value: float):
        """Set time scale (clamped to >= 0)."""
        self._time_scale = max(0.0, value)
        # Keep static Time class in sync
        try:
            from Infernux.timing import Time
            Time._time_scale = self._time_scale
        except ImportError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass  # Time module not available yet during early init
        except Exception as exc:
            Debug.log_warning(f"Failed to sync time_scale to Time class: {exc}")
    
    @property
    def total_play_time(self) -> float:
        """Total time elapsed since entering play mode."""
        return self._total_play_time
    
    # ========================================================================
    # State Transitions
    # ========================================================================
    
    def enter_play_mode(self) -> bool:
        """
        Enter play mode from edit mode.
        Saves scene state and initializes components.
        
        Returns:
            True if successfully entered play mode
        """
        if self._state != PlayModeState.EDIT:
            Debug.log_warning("Cannot enter play mode: not in edit mode")
            return False

        # Block play mode while editing a prefab
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm and sfm.is_prefab_mode:
            Debug.log_warning("Cannot enter Play mode while in Prefab Mode. Exit Prefab Mode first.")
            return False

        # Pre-flight check: block play if any script has load errors
        from Infernux.components.script_loader import has_script_errors, get_script_errors
        if has_script_errors():
            errors = get_script_errors()
            for path, tb in errors.items():
                Debug.log_error(
                    f"Cannot enter Play Mode — script error in "
                    f"{os.path.basename(path)}:\n{tb.splitlines()[-1]}",
                    source_file=path,
                )
            Debug.log_error(
                f"Play Mode blocked: {len(errors)} script(s) have errors. "
                "Fix all script errors before playing."
            )
            return False

        # Pre-flight check: block play if any BrokenComponent placeholders exist
        from Infernux.components.component import BrokenComponent, InxComponent
        broken = [
            c for comps in InxComponent._active_instances.values()
            for c in comps if isinstance(c, BrokenComponent)
        ]
        if broken:
            names = sorted({c.type_name for c in broken})
            if len(broken) == 1:
                comp = broken[0]
                owner = comp.game_object
                owner_name = getattr(owner, "name", "<Missing GameObject>")
                Debug.log_error(
                    f"Play Mode blocked by broken component '{comp.type_name}' on '{owner_name}'. "
                    "Fix or remove it before playing.",
                    context=owner if owner is not None else comp,
                )
            else:
                for comp in broken:
                    owner = comp.game_object
                    owner_name = getattr(owner, "name", "<Missing GameObject>")
                    Debug.log_error(
                        f"Play Mode blocked by broken component '{comp.type_name}' on '{owner_name}'. "
                        "Fix or remove it before playing.",
                        context=owner if owner is not None else comp,
                    )
                Debug.log_error(
                    f"Play Mode blocked: {len(broken)} broken component(s) "
                    f"({', '.join(names)}). Fix or remove them before playing."
                )
            return False
        
        Debug.log_internal("▶ Entering Play Mode...")

        from Infernux.engine.deferred_task import DeferredTaskRunner
        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            Debug.log_warning("Cannot enter play mode: a deferred task is already running")
            return False

        # ── Step functions (closures capture self) ───────────────────
        def step_enter():
            """Save scene, rebuild from snapshot, and activate play — all in one frame."""
            # 1. Serialize scene + clear undo + init timing
            self._save_scene_state()
            from Infernux.engine.undo import UndoManager
            _undo = UndoManager.instance()
            if _undo:
                _undo.clear()
            self._last_frame_time = time.time()
            self._total_play_time = 0.0
            self._delta_time = 0.0
            try:
                from Infernux.timing import Time
                Time._reset()
            except (ImportError, Exception) as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
            from Infernux.components.builtin_component import BuiltinComponent
            BuiltinComponent._clear_cache()

            # 2. Rebuild scene from snapshot
            if not self._rebuild_active_scene(self._scene_backup, for_play=True):
                Debug.log_error("Failed to rebuild runtime scene for Play Mode")
                self._state = PlayModeState.EDIT
                try:
                    self._rebuild_active_scene(self._scene_backup, for_play=False, restore_scene_path=True)
                except Exception as exc:
                    Debug.log_error(f"Failed to restore scene after play-mode build failure: {exc}")
                self._notify_state_change(PlayModeState.EDIT, PlayModeState.EDIT)
                return False

            # 3. Transition state and enter C++ play mode
            old_state = self._state
            self._state = PlayModeState.PLAYING
            try:
                from Infernux.core.material import Material
                Material._suppress_auto_save = True
            except ImportError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
            self._notify_state_change(old_state, self._state)
            scene_manager = self._get_scene_manager()
            if scene_manager:
                scene_manager.play()
            Debug.log_internal("[OK] Play Mode started (C++ lifecycle update path)")

        def on_done(ok):
            from Infernux.engine.ui.engine_status import EngineStatus
            if ok:
                EngineStatus.flash("已启动 Playing", 1.0, duration=1.5)
            else:
                EngineStatus.flash("启动失败 Play Failed", 0.0, duration=2.0)

        runner.submit("Enter Play Mode", [
            ("启动运行模式 Entering play mode...", 0.5, step_enter),
        ], on_done=on_done)
        return True
    
    def exit_play_mode(self, on_complete: Optional[Callable[[bool], None]] = None) -> bool:
        """
        Exit play mode and return to edit mode.
        Restores scene state to before play mode.
        
        Returns:
            True if successfully exited play mode
        """
        if self._state == PlayModeState.EDIT:
            Debug.log_warning("Cannot exit play mode: already in edit mode")
            return False
        
        Debug.log_internal("■ Exiting Play Mode...")

        from Infernux.engine.deferred_task import DeferredTaskRunner
        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            Debug.log_warning("Cannot exit play mode: a deferred task is already running")
            return False

        # ── Immediate actions (same frame as button click) ───────────
        # 1. Stop C++ gameplay loop immediately so no further Update /
        #    FixedUpdate / LateUpdate runs on the play-mode scene.
        #    This prevents an extra simulation frame between the Stop
        #    click and the deferred restore, eliminating a class of bugs
        #    where user scripts modify state after the user expected
        #    simulation to end.
        old_state = self._state
        scene_manager = self._get_scene_manager()
        if scene_manager:
            scene_manager.stop()

        # 2. Transition Python state to EDIT immediately so:
        #    - PlayModeManager.tick() becomes a no-op (no timing / scene loads)
        #    - Toolbar shows "Play" right away
        #    - No deferred scene loads from user scripts are processed
        self._state = PlayModeState.EDIT

        # Re-enable material auto-save now that play mode is over.
        try:
            from Infernux.core.material import Material
            Material._suppress_auto_save = False
        except ImportError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        # 3. Discard any pending runtime scene load queued by user scripts
        #    during the last play frame — we're about to restore the backup.
        try:
            from Infernux.scene import SceneManager as _SceneMgr
            _SceneMgr._pending_scene_load = None
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        from Infernux.components.builtin_component import BuiltinComponent
        BuiltinComponent._clear_cache()

        # ── Deferred step (single frame to avoid flicker) ─────────

        def step_exit():
            """Restore scene from backup and finalize — all in one frame."""
            # 1. Deserialize backup snapshot and recreate Python components
            restore_ok = self._rebuild_active_scene(
                self._scene_backup, for_play=False, restore_scene_path=True
            )
            if not restore_ok:
                Debug.log_error(
                    "Failed to restore scene after exiting Play Mode "
                    "— editor may be in a degraded state"
                )

            # 2. Clear undo and notify listeners
            from Infernux.engine.undo import UndoManager
            _undo = UndoManager.instance()
            if _undo:
                _undo.clear(scene_is_dirty=self._scene_dirty_backup)
                _undo.sync_dirty_state()
            else:
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    if self._scene_dirty_backup:
                        sfm.mark_dirty()
                    else:
                        sfm.clear_dirty()
            self._notify_state_change(old_state, PlayModeState.EDIT)

        def on_done(ok):
            from Infernux.engine.ui.engine_status import EngineStatus
            if ok:
                EngineStatus.flash("已停止 Stopped ■", 1.0, duration=1.5)
            else:
                EngineStatus.flash("停止失败 Stop Failed", 0.0, duration=2.0)
            if on_complete:
                try:
                    on_complete(ok)
                except Exception as exc:
                    Debug.log_error(f"exit_play_mode on_complete callback failed: {exc}")

        runner.submit("Exit Play Mode", [
            ("恢复编辑模式 Restoring edit mode...", 0.5, step_exit),
        ], on_done=on_done)
        return True
    
    def pause(self) -> bool:
        """
        Pause play mode.
        
        Returns:
            True if successfully paused
        """
        if self._state != PlayModeState.PLAYING:
            Debug.log_warning("Cannot pause: not currently playing")
            return False
        
        scene_manager = self._get_scene_manager()
        if scene_manager:
            scene_manager.pause()

        old_state = self._state
        self._state = PlayModeState.PAUSED
        
        Debug.log_internal("⏸ Play Mode Paused")
        self._notify_state_change(old_state, self._state)
        return True
    
    def resume(self) -> bool:
        """
        Resume from pause.
        
        Returns:
            True if successfully resumed
        """
        if self._state != PlayModeState.PAUSED:
            Debug.log_warning("Cannot resume: not currently paused")
            return False
        
        # Reset timing to avoid large delta_time after unpause
        self._last_frame_time = time.time()
        
        scene_manager = self._get_scene_manager()
        if scene_manager:
            scene_manager.play()

        old_state = self._state
        self._state = PlayModeState.PLAYING
        
        Debug.log_internal("▶ Play Mode Resumed")
        self._notify_state_change(old_state, self._state)
        return True
    
    def toggle_pause(self) -> bool:
        """Toggle between playing and paused states."""
        if self._state == PlayModeState.PLAYING:
            return self.pause()
        elif self._state == PlayModeState.PAUSED:
            return self.resume()
        return False
    
    def step_frame(self):
        """
        Execute a single frame while paused.
        Useful for debugging frame-by-frame.
        """
        if self._state != PlayModeState.PAUSED:
            Debug.log_warning("Step only works when paused")
            return
        
        scene_manager = self._get_scene_manager()
        if scene_manager:
            dt = self._delta_time if self._delta_time > 0 else (1.0 / 60.0)
            scene_manager.step(dt)
            Debug.log_internal(f"[Step] Stepped one frame (dt={dt:.4f}s)")
    
    # ========================================================================
    # Game Loop Integration
    # ========================================================================
    
    def tick(self, external_delta_time: float = None):
        """
        Called every frame by the engine.
        Updates timing and processes deferred scene loads.
        
        Args:
            external_delta_time: Optional externally provided delta time.
                                If None, calculates from wall clock.
        """
        if self._state == PlayModeState.EDIT:
            return

        # --- Process deferred scene loads (must run outside C++ iteration) ---
        from Infernux.scene import SceneManager as _SceneMgr
        _SceneMgr.process_pending_load()
        
        if self._state == PlayModeState.PAUSED:
            # Don't update timing when paused
            return
        
        # Calculate delta time
        current_time = time.time()
        if external_delta_time is not None:
            raw_dt = external_delta_time
        else:
            raw_dt = current_time - self._last_frame_time
        
        self._last_frame_time = current_time

        # Sync time_scale from the static Time class (user may set Time.time_scale)
        try:
            from Infernux.timing import Time
            self._time_scale = Time._time_scale
            Time._tick(raw_dt)
            # Read back computed values so PlayModeManager stays in sync
            self._delta_time = Time.delta_time
            self._total_play_time = Time.time
            # Read game-only frame cost from C++ (previous frame's measurement)
            if self._native_engine is not None:
                try:
                    Time._game_delta_time = self._native_engine.get_game_only_frame_ms() / 1000.0
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    pass
        except ImportError:
            self._delta_time = min(raw_dt * self._time_scale, 0.1)
            self._total_play_time += self._delta_time
        except Exception as exc:
            Debug.log_warning(f"Time sync failed: {exc}")
            self._delta_time = min(raw_dt * self._time_scale, 0.1)
            self._total_play_time += self._delta_time
        
        # NOTE: Lifecycle update is driven by C++ only.

    def _rebuild_active_scene(
        self,
        snapshot: Optional[str],
        *,
        for_play: bool,
        restore_scene_path: bool = False,
    ) -> bool:
        """Deserialize *snapshot* into the active scene and recreate Python components.

        This is the core of the unified component mode: play/edit transitions no
        longer try to reset lifecycle flags on existing objects. Instead, the
        active scene is rebuilt from serialized data, producing a fresh native
        component graph and fresh Python component instances.
        """
        self.clear_runtime_hidden_object_ids()

        if not snapshot:
            Debug.log_warning("Cannot rebuild scene: empty snapshot")
            return False

        scene_manager = self._get_scene_manager()
        if not scene_manager:
            Debug.log_warning("Cannot rebuild scene: no SceneManager")
            return False

        scene = scene_manager.get_active_scene()
        if not scene:
            Debug.log_warning("Cannot rebuild scene: no active scene")
            return False

        from Infernux.components.component import InxComponent
        from Infernux.components.builtin_component import BuiltinComponent
        from Infernux.renderstack.render_stack import RenderStack

        # Only nil out the RenderStack singleton before deserialize (safe).
        # Registry/cache clearing must wait until AFTER deserialize() because
        # C++ destroys old GameObjects and their PyComponentProxy::OnDestroy
        # callbacks still need live Python objects.
        RenderStack._active_instance = None

        if not scene.deserialize(snapshot):
            return False

        # Old GameObjects are gone — now safe to clear Python registries.
        InxComponent._clear_all_instances()
        BuiltinComponent._clear_cache()

        # When entering play mode, mark the scene as playing BEFORE restoring
        # Python components so newly attached PyComponentProxy instances use the
        # runtime lifecycle path instead of edit-mode lifecycle.
        if for_play and hasattr(scene, "set_playing"):
            scene.set_playing(True)

        self._restore_pending_py_components()

        if restore_scene_path:
            self._restore_scene_file_path()

        return True

    # ========================================================================
    # Python component helpers (serialization / reload)
    # ========================================================================

    def reload_components_from_script(self, file_path: str):
        """
        Reload all Python components that originate from the given script file.

        This is intended for Edit mode live-updates when a script changes.
        """
        if self._state != PlayModeState.EDIT:
            return
        if not self._asset_database:
            return

        script_path_abs = resolve_script_path(file_path)
        if not script_path_abs or not os.path.exists(script_path_abs):
            return
        scene_manager = self._get_scene_manager()
        if not scene_manager:
            return
        scene = scene_manager.get_active_scene()
        if not scene:
            return

        from Infernux.components.script_loader import (
            create_component_instance,
            load_all_components_from_file,
        )

        reloaded_count = 0
        target_guid = None
        if self._asset_database:
            target_guid = self._asset_database.get_guid_from_path(script_path_abs)
        if not target_guid:
            return

        pending_reload: list[tuple[int, Any, Dict[str, Any]]] = []
        for obj in scene.get_all_objects():
            if not hasattr(obj, "get_py_components"):
                continue
            py_components = list(obj.get_py_components())

            for comp in py_components:
                comp_guid = getattr(comp, "_script_guid", None)
                if comp_guid != target_guid:
                    continue
                try:
                    state = self._serialize_py_component(comp)
                except Exception as exc:
                    Debug.log_error(
                        f"Failed to snapshot component '{getattr(comp, 'type_name', type(comp).__name__)}' "
                        f"before reloading {os.path.basename(script_path_abs)}: {exc}"
                    )
                    continue
                pending_reload.append((obj.id, comp, state))

        if not pending_reload:
            return

        try:
            reloaded_classes = load_all_components_from_file(script_path_abs)
        except Exception as exc:
            Debug.log_error(
                f"Failed to reload component classes from {os.path.basename(script_path_abs)}: {exc}"
            )
            return

        if not reloaded_classes:
            return

        reloaded_by_name = {cls.__name__: cls for cls in reloaded_classes}

        for object_id, old_comp, state in pending_reload:
            obj = scene.find_by_id(object_id)
            if obj is None:
                continue

            target_type_name = state.get("type_name") or getattr(old_comp, "type_name", type(old_comp).__name__)
            component_class = reloaded_by_name.get(target_type_name)

            if component_class is None:
                Debug.log_error(
                    f"Failed to reload component '{target_type_name}' from {os.path.basename(script_path_abs)}: "
                    f"type not found after reload"
                )
                continue

            try:
                new_comp = create_component_instance(component_class)
            except Exception as exc:
                Debug.log_error(
                    f"Failed to recreate component '{target_type_name}' from {os.path.basename(script_path_abs)}: {exc}"
                )
                new_comp = None

            if new_comp is None:
                continue

            new_comp._script_guid = target_guid

            try:
                self._apply_py_component_state(new_comp, state)
            except Exception as exc:
                Debug.log_error(
                    f"Failed to apply state to reloaded component '{target_type_name}': {exc}"
                )

            if hasattr(obj, "remove_py_component"):
                obj.remove_py_component(old_comp)
            obj.add_py_component(new_comp)
            reloaded_count += 1

        if reloaded_count > 0:
            try:
                from Infernux.engine.undo import _bump_inspector_structure
                _bump_inspector_structure()
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
            Debug.log_internal(f"Reloaded {reloaded_count} component(s) from {os.path.basename(script_path_abs)}")

    # ========================================================================
    # Scene Snapshot (for runtime isolation)
    # ========================================================================

    # ========================================================================
    # Python Component Restoration (after C++ scene deserialize)
    # ========================================================================

    # ========================================================================
    # Scene State Management  
    # ========================================================================
    
    def _save_scene_state(self):
        """
        Save scene state before entering play mode.
        Uses C++ Scene::Serialize() which includes:
        - All GameObjects with their hierarchy
        - Transform data
        - C++ components (MeshRenderer, etc.)
        - Python component metadata (script GUID, fields)
        Also saves the current scene file path so we can return to
        the correct scene if the user switches scenes during play.
        """
        scene_manager = self._get_scene_manager()
        if not scene_manager:
            Debug.log_warning("Cannot save scene state: no SceneManager")
            return
        
        scene = scene_manager.get_active_scene()
        if scene:
            self._scene_backup = scene.serialize()
            # Remember which scene file was open
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm:
                self._scene_path_backup = sfm.current_scene_path
                self._scene_dirty_backup = sfm.is_dirty
            else:
                self._scene_dirty_backup = False
            Debug.log_internal("Scene state saved (C++ serialization)")
        else:
            Debug.log_warning("No active scene to save")

    def _restore_scene_file_path(self):
        """Restore SceneFileManager's current path and camera to the pre-play scene."""
        if self._scene_path_backup is None:
            return
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm and sfm.current_scene_path != self._scene_path_backup:
            Debug.log_internal(
                f"Restoring editor scene path: "
                f"{os.path.basename(self._scene_path_backup)}"
            )
            sfm._current_scene_path = self._scene_path_backup
            sfm._dirty = self._scene_dirty_backup
            # Restore the editor camera to the position saved for this scene
            sfm._restore_camera_state(self._scene_path_backup)
            if sfm._on_scene_changed:
                sfm._on_scene_changed()
    
    # ========================================================================
    # Event System
    # ========================================================================
    
    def add_state_change_listener(self, callback: Callable[[PlayModeEvent], None]):
        """Add a listener for play mode state changes."""
        if callback not in self._state_change_listeners:
            self._state_change_listeners.append(callback)
    
    def remove_state_change_listener(self, callback: Callable[[PlayModeEvent], None]):
        """Remove a state change listener."""
        if callback in self._state_change_listeners:
            self._state_change_listeners.remove(callback)
    
    def _notify_state_change(self, old_state: PlayModeState, new_state: PlayModeState):
        """Notify all listeners of state change."""
        # Tell the C++ renderer whether we're in play mode so it can
        # bypass the editor FPS cap and idle sleep.
        is_playing = new_state != PlayModeState.EDIT
        if self._native_engine is not None:
            try:
                self._native_engine.set_play_mode_rendering(is_playing)
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

        event = PlayModeEvent(
            old_state=old_state,
            new_state=new_state,
            timestamp=time.time()
        )
        
        for listener in self._state_change_listeners:
            listener(event)
    

