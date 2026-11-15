"""
PlayMode - Runtime/Editor mode manager for infernux.

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
import types
from enum import Enum, auto
from typing import Optional, List, Any, Callable, Iterable
from dataclasses import dataclass
from infernux.debug import Debug
from infernux.lib import InputManager as _NativeInputManager
from infernux.engine.project_context import resolve_script_path


class PlayModeState(Enum):
    """Play mode states."""
    EDIT = auto()      # Normal editor mode
    PLAYING = auto()   # Runtime playing
    PAUSED = auto()    # Runtime paused
    RESTORING = auto()  # Simulation stopped; authored scenes are not published yet
    RECOVERY_REQUIRED = auto()  # Restore failed; retain the snapshot and block authoring saves
    ENTERING = auto()  # Snapshot retained; runtime preparation is not published yet


@dataclass
class PlayModeEvent:
    """Event data for play mode state changes."""
    old_state: PlayModeState
    new_state: PlayModeState
    timestamp: float


@dataclass(frozen=True)
class _PlaySceneBackup:
    """One resident Scene and its editor-document state at the Play boundary."""

    world_id: int
    name: str
    scene: Any
    snapshot: Any
    document_id: str = ""
    resource_path: Optional[str] = None
    revision: int = 0
    saved_revision: int = 0
    document_state: Any = None
    was_active: bool = False


def _get_scene_manager():
    """Get the SceneManager singleton from C++ bindings."""
    from infernux.lib import SceneManager
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

        # These modules are stable process-level services.  Cache their
        # classes once at manager construction so the frame callback does not
        # repeat import resolution before every timing/load check.
        from infernux.scene import SceneManager as _SceneManagerAPI
        from infernux.timing import Time as _TimeAPI
        self._scene_manager_api = _SceneManagerAPI
        self._time_api = _TimeAPI
        
        # Timing
        self._last_frame_time: float = 0.0
        self._delta_time: float = 0.0
        self._time_scale: float = 1.0
        self._total_play_time: float = 0.0
        # Monotonic within one Play session.  A paused editor frame does not
        # advance Time.frame_count and may not consume a fixed physics step,
        # so automation needs an explicit completion sequence for Step.
        self._step_sequence: int = 0
        self._step_completion = None
        self._pending_step_requests: int = 0
        
        # Typed scene document captured before entering play mode.
        self._scene_backup: Optional[Any] = None
        self._scene_backups: tuple[_PlaySceneBackup, ...] = ()
        self._prefab_sources_during_play: dict[str, dict] = {}
        # Original scene file path (to restore correct scene on Stop)
        self._scene_path_backup: Optional[str] = None
        self._scene_document_id_backup: str = ""
        self._scene_revision_backup: int = 0
        self._scene_saved_revision_backup: int = 0
        self._scene_document_state_backup = None
        
        # Event listeners
        self._state_change_listeners: List[Callable[[PlayModeEvent], None]] = []
        
        # Store singleton reference
        PlayModeManager._instance = self

        # Asset database for GUID-based script lookup
        self._asset_database = None
        self._runtime_hidden_object_ids: set[int] = set()
        self._runtime_hidden_listeners: list[Callable[[], None]] = []
        self._last_rebuild_timings_ms: dict[str, float] = {}
        self._last_transition_timings_ms: dict[str, Any] = {}

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
        if not self._runtime_hidden_object_ids:
            return
        self._runtime_hidden_object_ids.clear()
        self._notify_runtime_hidden_changed()

    def register_runtime_hidden_object(self, game_object) -> None:
        if game_object is None:
            return
        object_id = int(game_object.id)
        if object_id > 0:
            previous_count = len(self._runtime_hidden_object_ids)
            self._runtime_hidden_object_ids.add(object_id)
            if len(self._runtime_hidden_object_ids) != previous_count:
                self._notify_runtime_hidden_changed()

    def add_runtime_hidden_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._runtime_hidden_listeners:
            self._runtime_hidden_listeners.append(callback)
            from infernux.engine.reload.owners import own_cleanup
            own_cleanup(callback, lambda: self.remove_runtime_hidden_listener(callback))

    def remove_runtime_hidden_listener(self, callback: Callable[[], None]) -> None:
        try:
            self._runtime_hidden_listeners.remove(callback)
        except ValueError:
            pass

    def _notify_runtime_hidden_changed(self) -> None:
        for callback in tuple(self._runtime_hidden_listeners):
            callback()

    def get_runtime_hidden_object_ids(self) -> set[int]:
        return set(self._runtime_hidden_object_ids)

    def is_runtime_hidden_object_id(self, object_id: int) -> bool:
        return int(object_id) in self._runtime_hidden_object_ids
    
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
    def is_restoring(self) -> bool:
        """True while the deferred Stop transaction owns the scene graph."""
        return self._state == PlayModeState.RESTORING
    
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
        """Set the native gameplay time scale."""
        from infernux.timing import Time
        Time.time_scale = value
        self._time_scale = Time.time_scale
    
    @property
    def total_play_time(self) -> float:
        """Total time elapsed since entering play mode."""
        return self._total_play_time

    @property
    def step_sequence(self) -> int:
        """Number of completed paused Step commands in this Play session."""
        return self._step_sequence

    @property
    def pending_step_requests(self) -> int:
        return int(self._pending_step_requests)

    @property
    def last_transition_timings_ms(self) -> dict[str, Any]:
        """Timing breakdown for the most recently completed Play Mode transition."""
        timings = dict(self._last_transition_timings_ms)
        phases = timings.get("phases")
        if isinstance(phases, dict):
            timings["phases"] = dict(phases)
        return timings
    
    # ========================================================================
    # State Transitions
    # ========================================================================
    
    def enter_play_mode(self) -> bool:
        """
        Enter play mode from edit mode.
        Saves scene state and initializes components.
        
        Returns:
            True if the deferred Play request was accepted.
        """
        if self._state != PlayModeState.EDIT:
            Debug.log_warning("Cannot enter play mode: not in edit mode")
            return False

        # Block play mode while editing a prefab
        from infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm and sfm.is_prefab_mode:
            Debug.log_warning("Cannot enter Play mode while in Prefab Mode. Exit Prefab Mode first.")
            return False
        if sfm and sfm.is_loading:
            Debug.log_warning("Cannot enter Play mode while an editor scene transaction is pending.")
            return False

        from infernux.engine.deferred_task import DeferredTaskRunner
        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            if runner.active_task_name == "Enter Play Mode":
                return True
            Debug.log_warning(
                "Cannot enter play mode while deferred task "
                f"'{runner.active_task_name or runner.active_step_label or 'unknown'}' is running"
            )
            return False

        # Play must consume saved script revisions before rebuilding the scene.
        from infernux.engine.resources_manager import ResourcesManager

        resources = ResourcesManager.instance()
        if resources is not None:
            try:
                resources.prepare_play_scripts()
            except (OSError, RuntimeError, ValueError) as exc:
                Debug.log_error(f"Play Mode blocked: saved scripts could not be published. {exc}")
                return False

        progress = None

        def report_progress(value, message):
            if progress is not None:
                progress.update(value, message)

        def step_enter():
            """Save scene, rebuild from snapshot, and activate play — all in one frame."""
            from infernux.engine.interaction.previews import close_editor_previews

            close_editor_previews()
            transition_started = time.perf_counter()
            report_progress(0.05, "Saving the authored scene")
            # Capture authoring before any preparation can modify the graph.
            # Capture failure leaves Edit untouched and must not reuse an old backup.
            snapshot_started = time.perf_counter()
            self._save_scene_state()
            snapshot_ms = (time.perf_counter() - snapshot_started) * 1000.0
            from infernux.core.assets import AssetManager
            from infernux.core.material import Material
            from infernux.renderstack.render_effect import RenderEffect
            from infernux.timing import Time

            self._state = PlayModeState.ENTERING
            Material._suppress_auto_save = True
            RenderEffect._suppress_auto_save = True
            try:
                self._last_frame_time = time.perf_counter()
                self._total_play_time = 0.0
                self._delta_time = 0.0
                self._step_sequence = 0
                self._pending_step_requests = 0
                self._step_completion = None
                Time._reset()
                AssetManager._begin_play_data_asset_isolation()

                # Clear-on-Play is a preparation event, before lifecycle logs.
                notify_started = time.perf_counter()
                self._notify_state_change(PlayModeState.EDIT, PlayModeState.ENTERING)
                notify_ms = (time.perf_counter() - notify_started) * 1000.0
                sprite_init_started = time.perf_counter()
                from infernux.components.builtin.sprite_renderer import SpriteRenderer
                report_progress(0.2, "Preparing sprite resources")
                for scene in self._loaded_scenes():
                    SpriteRenderer.init_all_in_scene(scene)
                sprite_init_ms = (time.perf_counter() - sprite_init_started) * 1000.0

                # Retain native objects on the success path, replacing only the
                # scripting domain. The same full snapshot owns every failure.
                rebuild_started = time.perf_counter()
                report_progress(0.4, "Preparing script components")
                prepared = (
                    self._prepare_loaded_scenes_for_play()
                    if self._scene_backups
                    else self._prepare_active_scene_for_play(self._scene_backup)
                )
                if not prepared:
                    raise RuntimeError("Play scripting-domain preparation was rejected")
                rebuild_ms = (time.perf_counter() - rebuild_started) * 1000.0

                # Retired edit particle graphs must drain before Start creates
                # runtime graphs that can reuse their native handles.
                scene_manager = self._get_scene_manager()
                if scene_manager is None:
                    raise RuntimeError("Cannot enter Play Mode without SceneManager")
                native_start_started = time.perf_counter()
                self._invalidate_native_gpu_view_state()
                from infernux.engine.startup_warmup import run_project_script_warmups, run_component_warmups
                from infernux.engine.project_context import get_project_root
                report_progress(0.6, "Warming project scripts")
                run_project_script_warmups(
                    project_path=get_project_root(), scope="editor-startup"
                )
                report_progress(0.75, "Starting scene components")
                scene_manager.play()
                initialization_errors = [
                    error
                    for scene in self._loaded_scenes()
                    for obj in scene.get_all_objects()
                    for component in obj.get_py_components()
                    if (error := component.__dict__.get("_initialization_error")) is not None
                ]
                if initialization_errors:
                    raise ExceptionGroup("Current component initialization failed", initialization_errors)
                from infernux.components.component import InxComponent
                report_progress(0.9, "Warming component resources")
                run_component_warmups(
                    (
                        component
                        for values in InxComponent._active_instances.values()
                        for component in values
                    ),
                    scope="editor-play",
                    project_path=get_project_root(),
                )
                self._mark_native_scene_temporal_discontinuity()
                native_start_ms = (time.perf_counter() - native_start_started) * 1000.0
                notify_started = time.perf_counter()
                self._state = PlayModeState.PLAYING
                self._notify_state_change(PlayModeState.ENTERING, PlayModeState.PLAYING)
                notify_ms += (time.perf_counter() - notify_started) * 1000.0
            except Exception as enter_error:
                self._state = PlayModeState.RESTORING
                try:
                    scene_manager = self._get_scene_manager()
                    if scene_manager is not None:
                        scene_manager.stop()
                    from infernux.input import Input
                    Input.set_game_focused(False)
                    Input.set_cursor_locked(False)
                    _NativeInputManager.instance().set_editor_cursor_capture_allowed(False)
                    self._cancel_runtime_scene_loads()
                    AssetManager._end_play_data_asset_isolation()
                    restored = (
                        self._restore_loaded_scenes_after_play()
                        if self._scene_backups
                        else self._rebuild_active_scene(
                            self._scene_backup, for_play=False, restore_scene_path=True
                        )
                    )
                    self._invalidate_native_gpu_view_state()
                    if not restored:
                        raise RuntimeError("Authored scene restoration was rejected")
                    self._state = PlayModeState.EDIT
                    self._notify_state_change(PlayModeState.ENTERING, PlayModeState.EDIT)
                    Material._suppress_auto_save = False
                    RenderEffect._suppress_auto_save = False
                    self._pending_step_requests = 0
                    self._step_completion = None
                    self._total_play_time = self._delta_time = 0.0
                    Time._reset()
                except Exception as restore_error:
                    self._state = PlayModeState.RECOVERY_REQUIRED
                    if self._native_engine is not None:
                        self._native_engine.set_play_mode_rendering(False)
                    raise RuntimeError(
                        f"Play initialization failed ({enter_error}); authored scene recovery "
                        f"also failed ({restore_error}). Snapshot retained; repair and use Stop."
                    ) from restore_error
                raise
            total_ms = (time.perf_counter() - transition_started) * 1000.0
            self._last_transition_timings_ms = {
                "transition": "enter",
                "total": total_ms,
                "snapshot": snapshot_ms,
                "rebuild": rebuild_ms,
                "native_start": native_start_ms,
                "sprite_init": sprite_init_ms,
                "notify": notify_ms,
            }

        def enter_with_progress():
            nonlocal progress
            from infernux.application import Application
            from infernux.lib import RuntimeMode
            if (
                Application.is_editor()
                and self._native_engine is not None
                and self._native_engine.runtime_mode == RuntimeMode.Graphical
            ):
                from infernux.engine.ui.busy_progress import BusyProgress
                try:
                    progress = BusyProgress()
                except OSError as exc:
                    # The progress window is presentation-only. A missing or
                    # blocked helper executable must never block Enter Play.
                    from infernux.debug import Debug
                    Debug.report_recovered(exc, "PlayMode.progress_window")
                    progress = None
            try:
                step_enter()
            finally:
                if progress is not None:
                    progress.close()
                    progress = None

        def on_done(ok):
            from infernux.engine.ui.engine_status import EngineStatus
            if ok:
                EngineStatus.flash("已启动 Playing", 1.0, duration=1.5)
            else:
                EngineStatus.flash("启动失败 Play Failed", 0.0, duration=2.0)

        runner.submit("Enter Play Mode", [
            ("启动运行模式 Entering play mode...", 0.5, enter_with_progress),
        ], on_done=on_done)
        return True
    
    def exit_play_mode(self, on_complete: Optional[Callable[[bool], None]] = None) -> bool:
        """
        Exit play mode and return to edit mode.
        Restores scene state to before play mode.
        
        Returns:
            True if the Stop request was accepted. ``on_complete`` reports restoration success.
        """
        if self._state == PlayModeState.EDIT:
            Debug.log_warning("Cannot exit play mode: already in edit mode")
            return False
        
        from infernux.engine.deferred_task import DeferredTaskRunner
        runner = DeferredTaskRunner.instance()
        if runner.is_busy:
            if runner.active_task_name == "Exit Play Mode":
                return True
            Debug.log_warning(
                "Cannot exit play mode while deferred task "
                f"'{runner.active_task_name or runner.active_step_label or 'unknown'}' is running"
            )
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
        if scene_manager and old_state in (
            PlayModeState.PLAYING, PlayModeState.PAUSED, PlayModeState.RECOVERY_REQUIRED,
        ):
            scene_manager.stop()

        # Stop owns the editor cursor boundary even when the Game View is
        # hidden and therefore cannot run its normal per-frame input route.
        from infernux.input import Input
        Input.set_game_focused(False)
        Input.set_cursor_locked(False)
        _NativeInputManager.instance().set_editor_cursor_capture_allowed(False)

        # Stopping simulation does not publish an authored scene. Keep saves
        # disabled until the retained snapshot has been restored successfully.
        self._state = PlayModeState.RESTORING
        self._pending_step_requests = 0
        self._step_completion = None
        from infernux.core.material import Material
        from infernux.renderstack.render_effect import RenderEffect
        Material._suppress_auto_save = True
        RenderEffect._suppress_auto_save = True

        # 3. Discard any pending runtime scene load queued by user scripts
        #    during the last play frame — we're about to restore the backup.
        self._cancel_runtime_scene_loads()

        # ── Deferred step (single frame to avoid flicker) ─────────

        def step_exit():
            """Restore scene from backup and finalize — all in one frame."""
            transition_started = time.perf_counter()
            from infernux.core.assets import AssetManager
            AssetManager._end_play_data_asset_isolation()
            # 1. Deserialize backup snapshot and recreate Python components
            rebuild_started = time.perf_counter()
            restore_ok = (
                self._restore_loaded_scenes_after_play()
                if self._scene_backups
                else self._rebuild_active_scene(
                    self._scene_backup, for_play=False, restore_scene_path=True
                )
            )
            self._invalidate_native_gpu_view_state()
            rebuild_ms = (time.perf_counter() - rebuild_started) * 1000.0
            if not restore_ok:
                raise RuntimeError(
                    "Cannot restore authored scenes after Play Mode; the snapshot is retained. "
                    "Resolve the error and use Stop to restore again."
                )

            # The rebuild above restores the authored document identity and
            # its exact revision/saved-revision pair. A second dirty-baseline
            # write here would duplicate ownership and discard revision data.
            notify_started = time.perf_counter()
            self._state = PlayModeState.EDIT
            self._notify_state_change(old_state, PlayModeState.EDIT)
            Material._suppress_auto_save = False
            RenderEffect._suppress_auto_save = False
            notify_ms = (time.perf_counter() - notify_started) * 1000.0
            total_ms = (time.perf_counter() - transition_started) * 1000.0
            self._last_transition_timings_ms = {
                "transition": "exit",
                "total": total_ms,
                "rebuild": rebuild_ms,
                "notify": notify_ms,
                "phases": dict(self._last_rebuild_timings_ms),
            }

        def on_done(ok):
            from infernux.engine.ui.engine_status import EngineStatus
            if ok:
                EngineStatus.flash("已停止 Stopped ■", 1.0, duration=1.5)
            else:
                self._state = PlayModeState.RECOVERY_REQUIRED
                Material._suppress_auto_save = True
                RenderEffect._suppress_auto_save = True
                if self._native_engine is not None:
                    self._native_engine.set_play_mode_rendering(False)
                EngineStatus.flash("恢复失败，请修正错误后再次停止 Restore Failed", 0.0, duration=4.0)
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

        from infernux.input import Input
        Input.set_game_focused(False)
        _NativeInputManager.instance().set_editor_cursor_capture_allowed(False)

        old_state = self._state
        self._state = PlayModeState.PAUSED
        
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
        
        self._pending_step_requests = 0
        self._step_completion = None
        # Reset timing to avoid large delta_time after unpause
        self._last_frame_time = time.perf_counter()
        
        scene_manager = self._get_scene_manager()
        if scene_manager:
            scene_manager.play()

        old_state = self._state
        self._state = PlayModeState.PLAYING
        
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
        
        self._pending_step_requests += 1
        return True

    def process_pending_step(self) -> bool:
        """Issue one logical step and count it only after its publication completes."""
        if self._state != PlayModeState.PAUSED:
            self._step_completion = None
            return False
        if self._pending_step_requests <= 0 and self._step_completion is None:
            return False
        scene_manager = self._get_scene_manager()
        if scene_manager is None:
            return False
        if scene_manager.is_stepping():
            return False
        if self._step_completion is not None:
            epoch, step, frame = self._step_completion
            if (int(scene_manager.simulation_epoch) == epoch and scene_manager.published_fixed_step_index >= step
                    and scene_manager.runtime_frame_count >= frame):
                self._step_sequence += 1
            self._step_completion = None
        if self._pending_step_requests <= 0:
            return False
        self._pending_step_requests -= 1
        dt = self._delta_time if self._delta_time > 0 else (1.0 / 60.0)
        epoch = int(scene_manager.simulation_epoch)
        step = int(scene_manager.fixed_step_index) + int(not scene_manager.fixed_step_pending)
        frame = int(scene_manager.runtime_frame_count) + 1
        scene_manager.step(dt)
        if scene_manager.is_stepping():
            self._step_completion = epoch, step, frame
        elif (int(scene_manager.simulation_epoch) == epoch and scene_manager.published_fixed_step_index >= step
                and scene_manager.runtime_frame_count >= frame):
            self._step_sequence += 1
        return True

    def _prepare_active_scene_for_play(self, snapshot: Optional[Any]) -> bool:
        """Refresh Python component instances while preserving native objects."""
        if not snapshot:
            raise ValueError("Cannot prepare scene for Play Mode: empty snapshot")
        scene_manager = self._get_scene_manager()
        scene = scene_manager.get_active_scene() if scene_manager else None
        if scene is None:
            raise RuntimeError("Cannot prepare scene for Play Mode: no active scene")

        return self._prepare_scene_for_play(scene, snapshot)

    def _prepare_scene_for_play(self, scene, snapshot: Optional[Any]) -> bool:
        if scene is None or not snapshot:
            raise ValueError("Play Mode preparation requires a Scene snapshot")
        from infernux.engine.component_restore import replace_scene_python_components_for_play
        from infernux.renderstack.render_stack import RenderStack

        replace_scene_python_components_for_play(
            scene,
            snapshot,
            asset_database=self._asset_database,
        )
        RenderStack.clear_active_instance(scene)
        scene.set_playing(True)
        return True

    def _prepare_loaded_scenes_for_play(self) -> bool:
        self.clear_runtime_hidden_object_ids()
        for backup in self._scene_backups:
            if not self._prepare_scene_for_play(backup.scene, backup.snapshot):
                raise RuntimeError(f"Play preparation rejected Scene '{backup.name}'")
        return True

    @staticmethod
    def _cancel_runtime_scene_loads() -> None:
        from infernux.scene import SceneManager as RuntimeSceneManager

        RuntimeSceneManager._scene_load_generation += 1
        transaction = RuntimeSceneManager._active_scene_transaction
        if transaction is not None and not transaction.is_complete:
            transaction.cancel()
        RuntimeSceneManager._clear_runtime_load_state()
    
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
        if not self.is_playing:
            return

        # --- Process deferred scene loads (must run outside C++ iteration) ---
        # The common path has no pending request.  Avoid crossing into the
        # scene-load service until a script actually queued a load or an
        # existing transaction needs polling.
        scene_manager_api = self._scene_manager_api
        if (
            scene_manager_api._pending_scene_load is not None
            or scene_manager_api._active_scene_transaction is not None
        ):
            scene_manager_api.process_pending_load()
        
        if self._state == PlayModeState.PAUSED:
            # Don't update timing when paused
            return
        
        # Calculate delta time
        current_time = time.perf_counter()
        if external_delta_time is not None:
            raw_dt = external_delta_time
        else:
            raw_dt = current_time - self._last_frame_time
        
        self._last_frame_time = current_time

        # Sync time_scale from the static Time class (user may set Time.time_scale)
        Time = self._time_api
        self._time_scale = Time.time_scale
        Time._tick(raw_dt)
        # Read back computed values so PlayModeManager stays in sync
        self._delta_time = Time.delta_time
        self._total_play_time = Time.time
        # Read game-only frame cost from C++ (previous frame's measurement)
        if self._native_engine is not None:
            Time._game_delta_time = self._native_engine.get_game_only_frame_ms() / 1000.0
        
        # NOTE: Lifecycle update is driven by C++ only.

    def _rebuild_active_scene(
        self,
        snapshot: Optional[Any],
        *,
        for_play: bool,
        restore_scene_path: bool = False,
    ) -> bool:
        """Restore *snapshot* into the active scene and recreate Python components.

        This is the core of the unified component mode: play/edit transitions no
        longer try to reset lifecycle flags on existing objects. Instead, the
        active scene is rebuilt from serialized data, producing a fresh native
        component graph and fresh Python component instances.
        """
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

        return self._rebuild_scene(scene, snapshot, for_play=for_play,
                                   restore_scene_path=restore_scene_path)

    def _rebuild_scene(
        self,
        scene,
        snapshot: Optional[Any],
        *,
        for_play: bool,
        restore_scene_path: bool = False,
    ) -> bool:
        if scene is None or not snapshot:
            Debug.log_warning("Cannot rebuild scene: missing Scene snapshot")
            return False

        from infernux.renderstack.render_stack import RenderStack

        def before_commit():
            # The incoming scene owns a fresh RenderStack instance. Clear the
            # previous scene's singleton before component deserialization so
            # on_after_deserialize() can promote the replacement naturally.
            RenderStack.clear_active_instance(scene)

        def after_publish():
            self.clear_runtime_hidden_object_ids()
            RenderStack.refresh_active_instance(scene)
            if for_play:
                scene.set_playing(True)
            from infernux.components.builtin.sprite_renderer import SpriteRenderer
            SpriteRenderer.init_all_in_scene(scene)

        from infernux.engine.scene_document_transaction import SceneDocumentTransaction
        transaction = SceneDocumentTransaction(
            scene,
            document=snapshot,
            asset_database=self._asset_database,
            clear_registries=True,
            borrow_document=True,
            prefer_loaded_types=True,
            before_commit=before_commit,
            after_publish=after_publish,
        )
        if not transaction.run_to_completion(raise_on_failure=False):
            self._last_rebuild_timings_ms = transaction.phase_timings_ms
            Debug.log_error(f"Cannot rebuild scene: document transaction failed: {transaction.error}")
            return False
        self._last_rebuild_timings_ms = transaction.phase_timings_ms

        # A play/edit transition replaces native objects while preserving
        # their authored IDs. Cached editor projections therefore cannot use
        # identity alone to detect the replacement. Publish the rebuild once
        # so Inspector, render extraction, and other revision consumers all
        # observe the fresh graph on the next frame.
        from infernux.engine.runtime_change_journal import (
            RuntimeChangeDomain,
            runtime_change_journal,
        )

        change_journal = runtime_change_journal()
        change_journal.publish(RuntimeChangeDomain.SCENE_TOPOLOGY, broad=True)
        change_journal.publish(RuntimeChangeDomain.COMPONENT_STRUCTURE, broad=True)
        change_journal.publish(RuntimeChangeDomain.TRANSFORM_LOCAL, broad=True)
        change_journal.publish(RuntimeChangeDomain.TRANSFORM_WORLD, broad=True)
        if restore_scene_path:
            self._restore_scene_file_path()

        return True

    def _restore_loaded_scenes_after_play(self) -> bool:
        scene_manager = self._get_scene_manager()
        if scene_manager is None:
            raise RuntimeError("Cannot restore Play Mode scenes without SceneManager")

        authored_world_ids = {backup.world_id for backup in self._scene_backups}
        for scene in reversed(self._loaded_scenes()):
            world_id = int(getattr(scene, "world_id", 0) or 0)
            if world_id not in authored_world_ids:
                scene_manager.unload_scene(scene)

        restored_scenes: dict[int, Any] = {}
        phase_totals: dict[str, float] = {}
        for backup in self._scene_backups:
            scene = scene_manager.get_scene_by_world_id(backup.world_id)
            if scene is None:
                scene = scene_manager.create_scene(backup.name)
            if not self._rebuild_scene(
                scene,
                backup.snapshot,
                for_play=False,
            ):
                return False
            restored_scenes[backup.world_id] = scene
            for phase, duration in self._last_rebuild_timings_ms.items():
                phase_totals[phase] = phase_totals.get(phase, 0.0) + float(duration)
        self._last_rebuild_timings_ms = phase_totals
        self._restore_scene_documents_after_play(restored_scenes)
        self._restore_reloaded_prefabs(restored_scenes.values())
        return True

    def _restore_reloaded_prefabs(self, scenes) -> None:
        if not self._prefab_sources_during_play:
            return
        from infernux.engine.scene_manager import SceneFileManager

        # Rebase the restored authored scene, keeping its pre-Play overrides.
        # Retain only documents that successfully reached the live world.
        sources = self._prefab_sources_during_play
        SceneFileManager.instance()._sync_prefab_scenes(
            scenes, affected_guids=sources.keys(), source_documents=sources,
        )
        sources.clear()

    # ========================================================================
    # Python component helpers (serialization / reload)
    # ========================================================================


    def _scenes_for_script_reload(self) -> tuple[Any, ...]:
        scenes = list(self._loaded_scenes())
        scene_manager = self._get_scene_manager()
        get_persistent = (
            getattr(scene_manager, "get_runtime_persistent_scene", None)
            if scene_manager is not None
            else None
        )
        persistent = get_persistent() if callable(get_persistent) else None
        if persistent is not None and persistent not in scenes:
            scenes.append(persistent)
        return tuple(scenes)


    def mark_components_missing_for_script(self, script_guid, file_path):
        from infernux.engine.resources_manager import ResourcesManager
        resources = ResourcesManager.instance()
        if resources is not None:
            resources.retire_script_paths((file_path,))
        else:
            from infernux.engine.reload.components import ComponentAdapter
            ComponentAdapter(self._scenes_for_script_reload).retire((file_path,))


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
        Uses the typed C++ Scene document which includes:
        - All GameObjects with their hierarchy
        - Transform data
        - C++ components (MeshRenderer, etc.)
        - Python component metadata (script GUID, fields)
        Also saves the current scene file path so we can return to
        the correct scene if the user switches scenes during play.
        """
        scene_manager = self._get_scene_manager()
        if scene_manager is None:
            raise RuntimeError("Cannot enter Play Mode without SceneManager")
        active_scene = scene_manager.get_active_scene()
        if active_scene is None:
            raise RuntimeError("Cannot enter Play Mode without an active scene")

        from infernux.engine.scene_manager import SceneFileManager

        sfm = SceneFileManager.instance()
        if sfm is None:
            raise RuntimeError("Cannot enter Play Mode without SceneFileManager")
        from infernux.engine.interaction import DocumentRegistry

        registry = DocumentRegistry.instance()
        backups = []
        for scene in self._loaded_scenes():
            snapshot = scene._capture_play_mode_snapshot()
            document_id = sfm.document_id_for_scene(scene)
            if not document_id and scene is active_scene:
                document_id = sfm.document_id
            document = registry.get(document_id) if document_id else None
            backups.append(
                _PlaySceneBackup(
                    world_id=int(getattr(scene, "world_id", 0) or 0),
                    name=str(getattr(scene, "name", "Scene") or "Scene"),
                    scene=scene,
                    snapshot=snapshot,
                    document_id=document_id,
                    resource_path=(
                        sfm._binding_for_document(document_id).resource_path
                        if document_id and sfm._binding_for_document(document_id) is not None
                        else sfm.current_scene_path if scene is active_scene else None
                    ),
                    revision=document.revision if document is not None else 0,
                    saved_revision=document.saved_revision if document is not None else 0,
                    document_state=document.state if document is not None else None,
                    was_active=scene is active_scene,
                )
            )
        self._scene_backups = tuple(backups)
        self._prefab_sources_during_play.clear()
        active_backup = next(backup for backup in backups if backup.was_active)
        self._scene_backup = active_backup.snapshot
        self._scene_path_backup = active_backup.resource_path
        self._scene_document_id_backup = active_backup.document_id
        self._scene_revision_backup = active_backup.revision
        self._scene_saved_revision_backup = active_backup.saved_revision
        self._scene_document_state_backup = active_backup.document_state

    def _loaded_scenes(self) -> tuple[Any, ...]:
        scene_manager = self._get_scene_manager()
        if scene_manager is None:
            return ()
        get_scene_at = getattr(scene_manager, "get_scene_at", None)
        count = int(getattr(scene_manager, "scene_count", 0) or 0)
        if callable(get_scene_at) and count > 0:
            return tuple(
                scene
                for index in range(count)
                if (scene := get_scene_at(index)) is not None
            )
        get_active_scene = getattr(scene_manager, "get_active_scene", None)
        if not callable(get_active_scene):
            return ()
        active = get_active_scene()
        return (active,) if active is not None else ()

    def _restore_scene_documents_after_play(
        self,
        restored_scenes: dict[int, Any],
    ) -> None:
        from infernux.engine.scene_manager import SceneFileManager
        from infernux.engine.interaction import DocumentRegistry, DocumentState

        sfm = SceneFileManager.instance()
        if sfm is None:
            raise RuntimeError("Cannot restore Play Mode scenes without SceneFileManager")
        registry = DocumentRegistry.instance()
        sfm.restore_loaded_scene_bindings(
            (
                restored_scenes[backup.world_id],
                backup.resource_path or "",
                backup.document_id,
            )
            for backup in self._scene_backups
            if backup.document_id
        )
        active_backup = None
        for backup in self._scene_backups:
            if backup.document_id:
                document = registry.require(backup.document_id)
                registry.restore_revision_state(
                    backup.document_id,
                    revision=backup.revision,
                    saved_revision=backup.saved_revision,
                    state=(document.state if DocumentState.CONFLICT in (document.state, backup.document_state)
                           else backup.document_state or DocumentState.READY),
                )
            if backup.was_active:
                active_backup = backup
        if active_backup is None:
            raise RuntimeError("Play Mode snapshot has no active Scene")
        active_scene = restored_scenes[active_backup.world_id]
        if not sfm.activate_loaded_scene(active_scene):
            raise RuntimeError("Cannot restore the active Scene document after Play Mode")
        if sfm.current_scene_path:
            sfm._restore_camera_state(sfm.current_scene_path)
            sfm._remember_last_scene(sfm.current_scene_path)

    def _restore_scene_file_path(self):
        """Restore the exact editor Scene document identity after Play Mode."""
        if not self._scene_document_id_backup:
            return
        from infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm is None:
            raise RuntimeError("Cannot restore Play Mode scene without SceneFileManager")
        from infernux.engine.interaction import (
            DocumentIdentityKind,
            DocumentRegistry,
            DocumentState,
        )

        registry = DocumentRegistry.instance()
        document = registry.require(self._scene_document_id_backup)
        current_path = (
            sfm._scene_path_for_guid(document.key.identity)
            if document.key.identity_kind is DocumentIdentityKind.ASSET_GUID
            else self._scene_path_backup
        )
        path_changed = sfm.current_scene_path != current_path
        sfm._current_scene_path = current_path
        sfm._scene_document_id = document.document_id
        registry.restore_revision_state(
            document.document_id,
            revision=self._scene_revision_backup,
            saved_revision=self._scene_saved_revision_backup,
            state=(document.state if DocumentState.CONFLICT in (document.state, self._scene_document_state_backup)
                   else self._scene_document_state_backup or DocumentState.READY),
        )
        if path_changed:
            if sfm.current_scene_path:
                sfm._restore_camera_state(sfm.current_scene_path)
            if sfm._on_scene_changed:
                sfm._on_scene_changed()
        # Reassert the authored scene at the Stop boundary so the next Editor
        # launch returns to the pre-play document.
        if sfm.current_scene_path:
            sfm._remember_last_scene(sfm.current_scene_path)
    
    # ========================================================================
    # Event System
    # ========================================================================
    
    def add_state_change_listener(self, callback: Callable[[PlayModeEvent], None]):
        """Add a listener for play mode state changes."""
        if callback not in self._state_change_listeners:
            self._state_change_listeners.append(callback)
            from infernux.engine.reload.owners import own_cleanup
            own_cleanup(callback, lambda: self.remove_state_change_listener(callback))
    
    def remove_state_change_listener(self, callback: Callable[[PlayModeEvent], None]):
        """Remove a state change listener."""
        if callback in self._state_change_listeners:
            self._state_change_listeners.remove(callback)
    
    def _mark_native_scene_temporal_discontinuity(self) -> None:
        """Ask the renderer to drop Game/Scene caches on the next frame."""
        from infernux.lib import SceneManager as NativeSceneManager

        native = NativeSceneManager.instance()
        if native is not None:
            native.mark_temporal_discontinuity()

    def _invalidate_native_gpu_view_state(self) -> None:
        """Drain GPU work after Play/Stop retires native particle graphs.

        Game RenderGraphs keep camera IDs across a same-scene rebuild. The
        next frame must not reuse compiled particle cullers or cached Game
        submissions that still import the retired buffers. Call this after
        the old graphs are retired and before new ones are created.
        """
        self._mark_native_scene_temporal_discontinuity()
        engine = self._native_engine
        if engine is None:
            return
        wait = getattr(engine, "wait_for_gpu_idle", None)
        if callable(wait):
            wait()

    def _notify_state_change(self, old_state: PlayModeState, new_state: PlayModeState):
        """Notify all listeners of state change."""
        # Tell the C++ renderer whether we're in play mode so it can
        # bypass the editor FPS cap and idle sleep.
        is_playing = new_state in (PlayModeState.PLAYING, PlayModeState.PAUSED)
        if self._native_engine is not None:
            self._native_engine.set_play_mode_rendering(is_playing)

        event = PlayModeEvent(
            old_state=old_state,
            new_state=new_state,
            timestamp=time.time()
        )
        
        for listener in self._state_change_listeners:
            listener(event)
    
