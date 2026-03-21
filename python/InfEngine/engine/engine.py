import os
import time

from InfEngine.lib import InfEngine, InfGUIRenderable, LogLevel, lib_dir
from InfEngine.engine.play_mode import PlayModeManager
from InfEngine.engine.project_context import set_project_root
from InfEngine.engine.path_utils import safe_path as _safe_path
from InfEngine.debug import Debug

_PLAYER_MODE = os.environ.get("_INFENGINE_PLAYER_MODE")
if not _PLAYER_MODE:
    from InfEngine.engine.resources_manager import ResourcesManager


class Engine():
    def __init__(self, engine_log_level=LogLevel.Info):
        self._engine = InfEngine(_safe_path(lib_dir))
        self.set_log_level(engine_log_level)
        self._gui_objects = {}
        self._play_mode_manager = None
        self._render_pipeline = None  # prevents GC of pybind11 trampoline
        self._last_frame_time = time.time()
        self._gizmos_collector = None  # lazy-init GizmosCollector
        self._scene_view_visible = not _PLAYER_MODE  # no Scene View in player
        self._next_reload_poll_time = 0.0
        self._next_gizmo_collect_time = 0.0
        self._reload_poll_interval = 0.1   # 10 Hz is enough for watcher events
        self._gizmo_collect_interval_play = 0.0
        self._gizmo_collect_interval_edit = 1.0 / 60.0
        self._gizmos_uploaded = False
        self._resources_manager = None  # Set in init_renderer (editor only)

    @staticmethod
    def _parse_present_mode(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value if 0 <= value <= 3 else None

        text = str(value).strip().lower()
        if not text:
            return None

        if text.isdigit():
            mode = int(text)
            return mode if 0 <= mode <= 3 else None

        aliases = {
            "immediate": 0,
            "mailbox": 1,
            "fifo": 2,
            "fifo_relaxed": 3,
            "vsync_off": 0,
            "off": 0,
            "vsync_on": 2,
            "on": 2,
        }
        return aliases.get(text)

    def _apply_startup_present_mode(self):
        raw = (os.environ.get("INFENGINE_PRESENT_MODE")
               or os.environ.get("_INFENGINE_PRESENT_MODE"))
        mode = self._parse_present_mode(raw)
        if mode is None:
            return
        try:
            self.set_present_mode(mode)
            Debug.log_internal(f"Startup present mode override applied: {mode}")
        except Exception as exc:
            Debug.log_warning(f"Failed to apply startup present mode override '{raw}': {exc}")

    def init_renderer(self, width, height, project_path):
        from InfEngine.resources import resources_path
        self._engine.init_renderer(
            width, height,
            _safe_path(project_path),
            _safe_path(resources_path),
        )
        self._apply_startup_present_mode()
        set_project_root(project_path)
        if not _PLAYER_MODE:
            self._resources_manager = ResourcesManager(
                project_path=project_path, engine=self._engine
            )
        
        # Load project materials (default material from project's .mat file)
        self._load_project_materials(project_path)
        
        # Initialize AssetManager singleton (GUID ↔ path resolution for refs)
        from InfEngine.core.assets import AssetManager
        AssetManager.initialize(self)
        Debug.log_internal("AssetManager initialized")

        # Initialize PlayModeManager (SceneManager will be set later via binding)
        self._play_mode_manager = PlayModeManager()
        self._play_mode_manager.set_asset_database(self.get_asset_database())
        Debug.log_internal("PlayModeManager initialized")

        # Auto-activate Python SRP rendering path
        # All rendering passes (opaque, skybox, transparent) are driven by Python
        from InfEngine.renderstack import RenderStackPipeline
        self.set_render_pipeline(RenderStackPipeline())
        Debug.log_internal("RenderStackPipeline activated (Python SRP path)")
    
    def _load_project_materials(self, project_path):
        """Load all .mat files from the project via AssetRegistry.

        Scans the project's ``materials/`` directory for ``.mat`` files.
        The first file named ``default_lit.mat`` is promoted to the
        engine-wide default material; every other ``.mat`` file is
        loaded via AssetRegistry so that scene deserialization can find it.
        """
        from InfEngine.lib import AssetRegistry

        registry = AssetRegistry.instance()

        # Collect candidate directories
        search_dirs = []
        materials_dir = os.path.join(project_path, "materials")
        if os.path.isdir(materials_dir):
            search_dirs.append(materials_dir)

        default_loaded = False
        extra_count = 0

        for mat_dir in search_dirs:
            for fname in os.listdir(mat_dir):
                if not fname.endswith(".mat"):
                    continue
                mat_path = os.path.join(mat_dir, fname)

                # Load the default material via MaterialManager (updates builtin cache)
                if fname == "default_lit.mat" and not default_loaded:
                    from InfEngine.lib import MaterialManager
                    if MaterialManager.instance().load_default_material_from_file(mat_path):
                        Debug.log_internal(f"Loaded default material from: {mat_path}")
                        default_loaded = True
                    else:
                        Debug.log_warning(f"Failed to load default material from: {mat_path}")
                else:
                    # Load via AssetRegistry (unified cache)
                    native = registry.load_material(mat_path)
                    if native:
                        extra_count += 1

        if not default_loaded:
            Debug.log_internal("No project default material found, using engine default")
        if extra_count:
            Debug.log_internal(f"Loaded {extra_count} additional project material(s)")

    def run(self):
        if self._resources_manager:
            self._resources_manager.start()

        # Install a pre-GUI callback so that DeferredTaskRunner steps
        # (which may mutate the scene via deserialize) execute BEFORE
        # any ImGui panel renders.  This prevents panels from accessing
        # destroyed pybind11 objects during play-mode Stop.
        def _pre_gui_tick():
            from InfEngine.engine.deferred_task import DeferredTaskRunner
            try:
                DeferredTaskRunner.instance().tick()
            except Exception:
                pass
        self._engine.set_pre_gui_callback(_pre_gui_tick)

        Debug.log_internal("Engine started")
        self._engine.run()
        # C++ Run() returned (main loop ended, but Cleanup not yet called).
        # Optimised shutdown order:
        #  1. Signal ResourcesManager to stop (non-blocking).
        #  2. Run C++ Cleanup (Vulkan teardown — the heavy part).
        #  3. Join the ResourcesManager thread (should already have exited
        #     during step 2, so the join returns instantly).
        self.exit()
    
    def tick_play_mode(self):
        """
        Called each frame to update play mode timing only.
        Lifecycle updates are driven by C++.
        """
        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time
        
        # Process pending script reloads on the main thread, but throttle polling.
        rm = self._resources_manager
        if rm and current_time >= self._next_reload_poll_time:
            try:
                rm.process_pending_reloads()
            except Exception as exc:
                from InfEngine.debug import Debug
                Debug.log_error(f"Script reload error: {exc}")
            self._next_reload_poll_time = current_time + self._reload_poll_interval
        
        pmm = self._play_mode_manager
        is_playing = pmm is not None and pmm.is_playing

        # Tick play mode manager (timing only)
        if pmm:
            pmm.tick(delta_time)

        # Flush throttled material saves — skip during play mode
        if not is_playing:
            self._flush_pending_material_saves()

        # Collect/upload gizmos only when Scene View is visible.
        if self._scene_view_visible:
            interval = self._gizmo_collect_interval_play if is_playing else self._gizmo_collect_interval_edit
            if interval <= 0.0 or current_time >= self._next_gizmo_collect_time:
                self._tick_gizmos()
                self._next_gizmo_collect_time = current_time + interval if interval > 0.0 else current_time
        else:
            self._clear_uploaded_gizmos()
        
        return delta_time

    def _tick_gizmos(self):
        """Collect component gizmos and upload to C++ each frame."""
        if self._gizmos_collector is None:
            from InfEngine.gizmos.collector import GizmosCollector
            self._gizmos_collector = GizmosCollector()
        self._gizmos_collector.collect_and_upload(self)
        self._gizmos_uploaded = True

    @staticmethod
    def _flush_pending_material_saves():
        """Flush all Material wrappers that have throttled pending saves."""
        try:
            from InfEngine.core.material import Material
            Material.flush_all_pending()
        except Exception:
            pass

    def _clear_uploaded_gizmos(self):
        """Clear uploaded gizmo buffers once when Scene View is hidden."""
        if not self._gizmos_uploaded:
            return
        native = self.get_native_engine()
        if not native:
            self._gizmos_uploaded = False
            return
        native.clear_component_gizmos()
        native.clear_component_gizmo_icons()
        self._gizmos_uploaded = False

    def set_scene_view_visible(self, visible: bool):
        """Called by SceneView panel to gate expensive gizmo updates and skip scene rendering."""
        visible = bool(visible)
        if self._scene_view_visible == visible:
            return
        self._scene_view_visible = visible
        # Gate C++ scene graph execution
        if self._engine is not None:
            self._engine.set_scene_view_visible(visible)
        if visible:
            self._next_gizmo_collect_time = 0.0
        else:
            self._clear_uploaded_gizmos()
    
    def get_play_mode_manager(self) -> PlayModeManager:
        """Get the play mode manager for controlling play/pause/stop."""
        return self._play_mode_manager
    
    def exit(self):
        """
        Clean up and exit the engine completely.

        Shutdown order (optimised to run ResourcesManager stop in parallel
        with C++ Vulkan teardown):
          0. Force-stop play mode (destroy Python components cleanly)
          1. Signal ResourcesManager stop (non-blocking — just sets _stop_event)
          2. C++ Cleanup (the heavy part — GPU drain + resource destruction)
          3. Join ResourcesManager thread (should already have exited by now)
        """
        # 0. If still in play mode, tear down Python components before C++
        #    objects are destroyed.  Without this, the C++ renderer
        #    destruction can trigger PyComponentProxy::OnDestroy callbacks
        #    on already-invalid Python state, and physics/audio may block.
        self._shutdown_play_mode()

        # 1. Signal the file-watcher / scanning thread to stop (non-blocking).
        #    The thread will wake within 0.25 s and begin its own teardown
        #    while C++ cleanup runs in parallel on this thread.
        if self._resources_manager:
            self._resources_manager._stop_event.set()

        # 2. C++ Cleanup — destroys renderer, Vulkan device, etc.
        if self._engine:
            self._engine.cleanup()
        
        # 3. Join the ResourcesManager thread.  It had the entire C++ cleanup
        #    duration to shut itself down, so the join should be near-instant.
        if self._resources_manager:
            self._resources_manager.stop()
        
        # Clear all references
        self._gui_objects.clear()
        self._engine = None
        self._resources_manager = None

    def _shutdown_play_mode(self):
        """Immediately tear down play-mode components for a clean shutdown.

        Unlike ``exit_play_mode()`` (which uses deferred tasks to restore the
        saved scene across several frames), this performs only the minimal
        cleanup needed so that the subsequent C++ ``Cleanup()`` does not
        encounter live Python component state.
        """
        from InfEngine.engine.play_mode import PlayModeState

        pmm = self._play_mode_manager
        if not pmm or pmm.state == PlayModeState.EDIT:
            return

        Debug.log_internal("Shutting down Play Mode for engine exit…")

        # 1. Stop the C++ simulation loop (no more Update/FixedUpdate calls)
        try:
            from InfEngine.lib import SceneManager as _NativeSceneMgr
            sm = _NativeSceneMgr.instance()
            if sm:
                sm.stop()
        except Exception:
            pass

        # 2. Destroy all live Python components (on_destroy + GC helpers)
        try:
            from InfEngine.components.component import InfComponent
            for comp_list in list(InfComponent._active_instances.values()):
                for comp in list(comp_list):
                    try:
                        comp._call_on_destroy()
                    except Exception:
                        pass
            InfComponent._clear_all_instances()
        except Exception:
            pass

        # 3. Flip state to EDIT so nothing else treats us as playing
        pmm._state = PlayModeState.EDIT

    def set_gui_font(self, font_path, font_size=18):
        self._engine.set_gui_font(_safe_path(font_path), font_size)

    def get_display_scale(self) -> float:
        """Return the OS display scale factor (e.g. 2.0 for 200% Windows scaling)."""
        return self._engine.get_display_scale()

    def set_log_level(self, engine_log_level):
        self._engine.set_log_level(engine_log_level)

    def register_gui(self, name: str, gui_object: InfGUIRenderable):
        self._engine.register_gui_renderable(name, gui_object)
        self._gui_objects[name] = gui_object

    def unregister_gui(self, name: str):
        self._engine.unregister_gui_renderable(name)
        self._gui_objects.pop(name, None)

    def reset_imgui_layout(self):
        """Clear ImGui docking layout (in-memory + on disk)."""
        self._engine.reset_imgui_layout()
    
    def show(self):
        self._engine.show()

    def hide(self):
        self._engine.hide()

    def set_window_icon(self, icon_path):
        """Set the editor window icon from a PNG file."""
        self._engine.set_window_icon(_safe_path(icon_path))

    def set_fullscreen(self, fullscreen: bool):
        """Set the window to fullscreen or windowed mode."""
        if self._engine:
            self._engine.set_fullscreen(fullscreen)

    def set_window_title(self, title: str):
        """Set the window title bar text."""
        if self._engine:
            self._engine.set_window_title(title)

    def set_present_mode(self, mode: int):
        """Set swapchain present mode: 0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED."""
        if self._engine:
            self._engine.set_present_mode(int(mode))

    def get_present_mode(self) -> int:
        """Get current swapchain present mode: 0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED."""
        if self._engine:
            return int(self._engine.get_present_mode())
        return 1
    
    def get_native_engine(self):
        """Get the underlying native InfEngine instance for direct API access."""
        return self._engine
    
    def get_resource_preview_manager(self):
        """Get the resource preview manager for file previews in Inspector."""
        if self._engine:
            return self._engine.get_resource_preview_manager()
        return None

    def get_asset_database(self):
        """Get the asset database instance for project asset operations."""
        if self._engine:
            return self._engine.get_asset_database()
        return None

    # ========================================================================
    # Editor Camera — property-based access (EditorCamera object)
    # ========================================================================

    @property
    def editor_camera(self):
        """Get the editor camera controller (EditorCamera object with
        properties: position, rotation, fov, near_clip, far_clip,
        focus_point, focus_distance; methods: reset(), focus_on(),
        restore_state(), world_to_screen_point())."""
        if self._engine:
            return self._engine.editor_camera
        return None

    def process_scene_view_input(self, delta_time: float, right_mouse_down: bool, middle_mouse_down: bool,
                                  mouse_delta_x: float, mouse_delta_y: float, scroll_delta: float,
                                  key_w: bool, key_a: bool, key_s: bool, key_d: bool,
                                  key_q: bool, key_e: bool, key_shift: bool):
        """Process scene view input for editor camera control."""
        if self._engine:
            self._engine.process_scene_view_input(
                delta_time, right_mouse_down, middle_mouse_down,
                mouse_delta_x, mouse_delta_y, scroll_delta,
                key_w, key_a, key_s, key_d, key_q, key_e, key_shift
            )

    # ========================================================================
    # Scene Render Target API - for offscreen scene rendering
    # ========================================================================

    def get_scene_texture_id(self) -> int:
        """Get scene render target texture ID for ImGui display."""
        if self._engine:
            return self._engine.get_scene_texture_id()
        return 0

    def resize_scene_render_target(self, width: int, height: int):
        """Resize the scene render target to match viewport size."""
        if self._engine:
            self._engine.resize_scene_render_target(width, height)

    # ========================================================================
    # Game Render Target API - for game camera rendering
    # ========================================================================

    def get_game_texture_id(self) -> int:
        """Get game render target texture ID for ImGui display."""
        if self._engine:
            return self._engine.get_game_texture_id()
        return 0

    def resize_game_render_target(self, width: int, height: int):
        """Resize the game render target to match game viewport size."""
        if self._engine:
            self._engine.resize_game_render_target(width, height)

    def set_game_camera_enabled(self, enabled: bool):
        """Enable or disable game camera rendering."""
        if self._engine:
            self._engine.set_game_camera_enabled(enabled)

    def get_screen_ui_renderer(self):
        """Get the GPU screen-space UI renderer (None before game RT init)."""
        if self._engine:
            return self._engine.get_screen_ui_renderer()
        return None

    # ========================================================================
    # Scene Picking API - for editor selection
    # ========================================================================

    def pick_scene_object_id(self, screen_x: float, screen_y: float, viewport_width: float, viewport_height: float) -> int:
        """Pick a scene object by screen-space coordinates; returns object ID or 0."""
        if self._engine:
            return self._engine.pick_scene_object_id(screen_x, screen_y, viewport_width, viewport_height)
        return 0

    # ========================================================================
    # Editor Tools API — highlight + ray for Python-side gizmo interaction
    # ========================================================================

    def pick_gizmo_axis(self, screen_x: float, screen_y: float,
                        viewport_width: float, viewport_height: float) -> int:
        """Lightweight gizmo axis proximity test for hover highlighting."""
        if self._engine:
            return self._engine.pick_gizmo_axis(screen_x, screen_y, viewport_width, viewport_height)
        return 0

    def set_editor_tool_highlight(self, axis: int):
        """Set the highlighted (hovered) gizmo axis. 0=None, 1=X, 2=Y, 3=Z."""
        if self._engine:
            self._engine.set_editor_tool_highlight(axis)

    def set_editor_tool_mode(self, mode: int):
        """Set the active editor tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale."""
        if self._engine:
            self._engine.set_editor_tool_mode(mode)

    def get_editor_tool_mode(self) -> int:
        """Get the active editor tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale."""
        if self._engine:
            return self._engine.get_editor_tool_mode()
        return 0

    def set_editor_tool_local_mode(self, local: bool):
        """Enable/disable local coordinate mode for editor tool gizmos."""
        if self._engine:
            self._engine.set_editor_tool_local_mode(local)

    def screen_to_world_ray(self, screen_x: float, screen_y: float,
                            viewport_width: float, viewport_height: float):
        """Build a world-space ray from screen coordinates.

        Returns (origin_x, origin_y, origin_z, dir_x, dir_y, dir_z).
        """
        if self._engine:
            return self._engine.screen_to_world_ray(screen_x, screen_y,
                                                     viewport_width, viewport_height)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # ========================================================================
    # Editor Gizmos API - for toggling visual aids in scene view
    # ========================================================================

    def get_selected_object_id(self) -> int:
        """Get the currently selected object ID (0 if none)."""
        if self._engine:
            return self._engine.get_selected_object_id()
        return 0

    def set_show_grid(self, show: bool):
        """Set visibility of ground grid."""
        if self._engine:
            self._engine.set_show_grid(show)

    def is_show_grid(self) -> bool:
        """Get visibility of ground grid."""
        if self._engine:
            return self._engine.is_show_grid()
        return False

    def pick_scene_object_ids(self, screen_x: float, screen_y: float, viewport_width: float, viewport_height: float):
        """Pick ordered scene object candidate IDs at screen coordinates (for editor cycling selection)."""
        if self._engine is None:
            return []
        return list(self._engine.pick_scene_object_ids(screen_x, screen_y, viewport_width, viewport_height))

    # ========================================================================
    # Render Pipeline API (SRP)
    # ========================================================================

    def set_render_pipeline(self, asset_or_pipeline=None):
        """
        Set a custom render pipeline.

        Args:
            asset_or_pipeline: A RenderPipelineAsset (calls create_pipeline()),
                               a RenderPipeline instance, or None to revert to
                               the default C++ rendering path.
        """
        if self._engine is None:
            return

        if asset_or_pipeline is None:
            self._render_pipeline = None
            self._engine.set_render_pipeline(None)
            return

        # If it's an asset, create the pipeline from it
        if hasattr(asset_or_pipeline, "create_pipeline"):
            pipeline = asset_or_pipeline.create_pipeline()
        else:
            pipeline = asset_or_pipeline

        # MUST keep a Python-side reference! Without this, the Python wrapper
        # gets GC'd (ref count → 0), pybind11 removes the C++ → Python mapping
        # from registered_instances, and get_override() can't find the Python
        # object → "pure virtual function" error.
        self._render_pipeline = pipeline
        self._engine.set_render_pipeline(pipeline)
