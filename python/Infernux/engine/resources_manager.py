import os
import threading
import time

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
except ImportError:
    # Standalone player builds exclude watchdog; ResourcesManager
    # still importable but .start() becomes a no-op.
    Observer = None
    FileSystemEventHandler = object
    _HAS_WATCHDOG = False

from Infernux.lib import Infernux
from Infernux.engine.script_compiler import get_script_compiler
from Infernux.debug import Debug


class ResourceChangeHandler(FileSystemEventHandler):

    def __init__(self, engine: Infernux):
        self._engine = engine
        self._script_compiler = get_script_compiler()
        # Queue for pending script reloads (processed in main thread)
        self._pending_reload_queue = []
        self._queue_lock = threading.Lock()
        # Pending deletes for move-detection (path -> timestamp)
        self._pending_deletes = {}
        self._pending_lock = threading.Lock()
        self._move_grace_seconds = 1.0
        # Callbacks for shader cache invalidation
        self._shader_cache_invalidation_callbacks = []
        # Get AssetDatabase for resource registration
        self._asset_database = engine.get_asset_database()

    def _should_ignore(self, file_path: str) -> bool:
        """Ignore meta/temp/cache files to avoid GUID churn and noisy events."""
        lower = file_path.replace("\\", "/").lower()
        if lower.endswith(".meta") or lower.endswith(".meta.tmp") or lower.endswith(".tmp"):
            return True
        if "/__pycache__/" in lower or lower.endswith(".pyc"):
            return True
        basename = lower.rsplit("/", 1)[-1]
        if basename == "imgui.ini":
            return True
        return False

    def _try_match_move(self, new_path: str) -> str:
        """Try to match a recent delete as a move; returns old_path if matched."""
        new_base = os.path.basename(new_path)
        now = time.time()
        with self._pending_lock:
            for old_path, ts in list(self._pending_deletes.items()):
                if now - ts > self._move_grace_seconds:
                    continue
                if os.path.basename(old_path) == new_base:
                    del self._pending_deletes[old_path]
                    return old_path
        return ""

    def _process_pending_deletes(self):
        """Finalize deletes that were not matched as moves."""
        now = time.time()
        expired = []
        with self._pending_lock:
            for path, ts in self._pending_deletes.items():
                if now - ts >= self._move_grace_seconds:
                    expired.append(path)
            for path in expired:
                del self._pending_deletes[path]

        for path in expired:
            # Cascade: evict from GPU caches, C++ registries, Python caches
            from Infernux.core.assets import AssetManager
            AssetManager.on_asset_deleted(path)

            if self._asset_database:
                self._asset_database.delete_asset(path)
            else:
                self._engine.delete_resources(path)
            if path.endswith('.py'):
                rm = ResourcesManager.instance()
                if rm is not None:
                    rm.notify_script_catalog_changed(path, "deleted")

    def _import_with_retry(self, file_path: str) -> str:
        """Try to import an asset, retrying on failure (handles file-still-being-written race)."""
        max_retries = 3
        for attempt in range(max_retries):
            if self._asset_database:
                guid = self._asset_database.import_asset(file_path)
                if guid:
                    return guid
            if attempt < max_retries - 1:
                time.sleep(0.15)
        return ""

    def on_created(self, event):
        if not event.is_directory:
            if self._should_ignore(event.src_path):
                return
            # Debug.log_internal(f"[Added] {event.src_path}")
            # Try to match a move (delete+add pattern)
            old_path = self._try_match_move(event.src_path)
            if old_path:
                if self._asset_database:
                    self._asset_database.on_asset_moved(old_path, event.src_path)
                    # Debug.log_internal(f"[Moved] {old_path} -> {event.src_path}")
                else:
                    self._engine.move_resources(old_path, event.src_path)
                # Notify AssetManager/AssetRegistry after DB mapping is updated
                from Infernux.core.assets import AssetManager
                AssetManager.on_asset_moved(old_path, event.src_path)
                # For moved scripts, queue reload
                if event.src_path.endswith('.py'):
                    self._queue_script_reload(event.src_path)
                return
            # Check Python scripts on creation
            if event.src_path.endswith('.py'):
                self._queue_script_reload(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            if self._should_ignore(event.src_path):
                return
            Debug.log_internal(f"[Deleted] {event.src_path}")
            # Defer delete to detect move (delete+add)
            with self._pending_lock:
                self._pending_deletes[event.src_path] = time.time()

    def on_modified(self, event):
        if not event.is_directory:
            src = event.src_path
            lower = src.replace("\\", "/").lower()

            # .meta file changed → propagate as modification of the owning asset
            if lower.endswith(".meta") and not lower.endswith(".meta.tmp"):
                owner_path = src[:-5]  # strip ".meta"
                if os.path.isfile(owner_path):
                    Debug.log_internal(f"[Meta Modified] {src} -> owner {owner_path}")
                    # Refresh in-memory meta from disk (re-register the resource)
                    # so that subsequent reloads read the updated import settings.
                    if self._asset_database:
                        self._asset_database.on_asset_modified(owner_path)
                    from Infernux.core.assets import AssetManager
                    AssetManager.on_asset_modified(owner_path)
                return

            if self._should_ignore(src):
                return
            # Debug.log_internal(f"[Modified] {src}")
            # Invalidate stale asset caches (material / texture)
            from Infernux.core.assets import AssetManager
            AssetManager.on_asset_modified(src)
            # Use AssetDatabase to update meta (preserves GUID, updates content_hash)
            if self._asset_database:
                self._asset_database.on_asset_modified(event.src_path)
            else:
                self._engine.modify_resources(event.src_path)
            # Check Python scripts on modification
            if event.src_path.endswith('.py'):
                self._queue_script_reload(event.src_path)
            # Queue shader hot-reload
            elif event.src_path.endswith('.vert') or event.src_path.endswith('.frag'):
                self._queue_shader_reload(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            if self._should_ignore(event.src_path) or self._should_ignore(event.dest_path):
                return
            Debug.log_internal(f"[Moved] {event.src_path} -> {event.dest_path}")
            with self._pending_lock:
                if event.src_path in self._pending_deletes:
                    del self._pending_deletes[event.src_path]
            # IMPORTANT: Update AssetDatabase mapping FIRST so that
            # AssetRegistry.OnAssetMoved can resolve GUID from the new path.
            if self._asset_database:
                self._asset_database.on_asset_moved(event.src_path, event.dest_path)
            else:
                self._engine.move_resources(event.src_path, event.dest_path)
            # Now notify AssetManager/AssetRegistry (GUID→path map is up-to-date)
            from Infernux.core.assets import AssetManager
            AssetManager.on_asset_moved(event.src_path, event.dest_path)
            # Check Python scripts after move
            if event.dest_path.endswith('.py'):
                self._queue_script_reload(event.dest_path)
                rm = ResourcesManager.instance()
                if rm is not None:
                    rm.notify_script_catalog_changed(event.dest_path, "moved")
    
    def _queue_script_reload(self, file_path: str):
        """Queue a script for reload (will be processed in main thread)."""
        with self._queue_lock:
            # Avoid duplicates
            if file_path not in self._pending_reload_queue:
                self._pending_reload_queue.append(file_path)
    
    def _queue_shader_reload(self, file_path: str):
        """Queue a shader for hot-reload (will be processed in main thread)."""
        with self._queue_lock:
            key = ("shader", file_path)
            if key not in self._pending_reload_queue:
                self._pending_reload_queue.append(key)
    
    def process_pending_reloads(self):
        """Process pending script reloads. Call this from main thread (e.g., in update loop)."""
        self._process_pending_deletes()
        with self._queue_lock:
            pending = list(self._pending_reload_queue)
            self._pending_reload_queue.clear()
        
        for item in pending:
            try:
                if isinstance(item, tuple) and item[0] == "shader":
                    self._reload_shader(item[1])
                elif isinstance(item, str):
                    self._check_script(item)
            except Exception as exc:
                Debug.log_error(f"Reload failed for {item}: {exc}")
    
    def _check_script(self, file_path: str):
        """Check a Python script for syntax errors and hot-reload components."""
        # Verify file exists and is readable (ensures write is complete)
        if not os.path.exists(file_path):
            return

        errors = self._script_compiler.check_file(file_path)
        if errors:
            from Infernux.components.script_loader import set_script_error
            combined = "\n".join(
                f"{os.path.basename(e.file_path)}:{e.line_number}  {e.message}"
                for e in errors
            )
            set_script_error(file_path, combined)
            for error in errors:
                Debug.log_error(
                    f"Script Error in {os.path.basename(error.file_path)}:{error.line_number}\n{error.message}",
                    source_file=error.file_path,
                    source_line=error.line_number)
        else:
            from Infernux.components.script_loader import _clear_script_error
            _clear_script_error(file_path)
            Debug.log_internal(f"[OK] Script OK: {os.path.basename(file_path)}")
            rm = ResourcesManager.instance()
            if rm is not None:
                rm.notify_script_catalog_changed(file_path, "modified")
            # Notify registered per-file callbacks (e.g. RenderStack pipeline reload)
            # Callbacks are stored on ResourcesManager to avoid handler-init races
            abs_path = os.path.abspath(file_path)
            if rm is not None:
                for cb in list(rm._script_reload_callbacks.get(abs_path, [])):
                    cb(file_path)
            # Hot-reload InxComponents from this script
            from Infernux.engine.play_mode import PlayModeManager
            play_mode = PlayModeManager.instance()
            if play_mode:
                play_mode.reload_components_from_script(file_path)

    def _reload_shader(self, file_path: str):
        """Reload a shader file and update the rendering pipeline."""
        if not os.path.exists(file_path):
            return

        Debug.log_internal(f"[Shader] Reloading: {os.path.basename(file_path)}")

        # Invalidate shader caches in UI
        for callback in self._shader_cache_invalidation_callbacks:
            callback()

        # Use the engine to reload shader
        if hasattr(self._engine, 'reload_shader'):
            result = self._engine.reload_shader(file_path)
            if not result:  # empty string = success
                Debug.log_internal(f"[OK] Shader reloaded: {os.path.basename(file_path)}")
                # Bump property generation so inspectors re-sync @property annotations
                from Infernux.engine.ui import inspector_shader_utils as _su
                _su.bump_shader_property_generation()
            else:
                Debug.log_error(f"[ERROR] Shader compile error ({os.path.basename(file_path)}):\n{result}")
        else:
            Debug.log_warning("Engine does not support shader hot-reload yet")
    
    def register_shader_cache_callback(self, callback):
        """Register a callback to be called when shader cache should be invalidated."""
        if callback not in self._shader_cache_invalidation_callbacks:
            self._shader_cache_invalidation_callbacks.append(callback)

class ResourcesManager:
    _instance: 'ResourcesManager | None' = None

    @classmethod
    def instance(cls) -> 'ResourcesManager | None':
        """Return the active ResourcesManager singleton, or None."""
        return cls._instance

    def __init__(self, project_path: str, engine: Infernux):
        ResourcesManager._instance = self
        self._engine = engine
        self._assets_path = os.path.join(project_path, "Assets")
        self._observer = None
        self._thread = None
        self._stop_event = threading.Event()
        self._event_handler = None
        self._script_reload_callbacks = {}  # file_path -> [callbacks]
        self._script_catalog_callbacks = []  # [callback(file_path, event_type)]

    def _shutdown_observer(self, *, join_timeout: float = 5.0) -> None:
        """Stop watchdog threads aggressively so the Python process can exit."""
        observer = self._observer
        self._observer = None
        if observer is None:
            return

        try:
            observer.stop()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        try:
            observer.unschedule_all()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        try:
            observer.join(timeout=join_timeout)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        if getattr(observer, "is_alive", lambda: False)():
            Debug.log_warning("ResourcesManager observer did not stop cleanly before timeout")

    def start(self):
        """
        Start to scan the project directory for resources in a sub-thread.
        """
        if not _HAS_WATCHDOG:
            return  # watchdog not available (standalone player build)
        if self._thread and self._thread.is_alive():
            Debug.log_warning("ResourcesManager is already running")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_resources, daemon=True)
        self._thread.start()

    def _scan_resources(self):
        """
        Use watchdog to monitor file changes in _assets_path.
        """
        if not os.path.exists(self._assets_path):
            Debug.log_warning(f"Assets path not found: {self._assets_path}")
            return

        self._event_handler = ResourceChangeHandler(self._engine)
        self._observer = Observer()
        try:
            # If watchdog fails to shut down for any reason, do not let its
            # worker thread keep the entire engine process alive forever.
            self._observer.daemon = True
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        try:
            self._observer.schedule(self._event_handler, self._assets_path, recursive=True)
            self._observer.start()

            # Initial full scan: check every .py file in Assets/ so that
            # pre-existing script errors are detected on engine startup.
            self._initial_script_scan()

            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.25)  # wake quickly on shutdown
        finally:
            self._shutdown_observer(join_timeout=5.0)

    def _initial_script_scan(self):
        """Walk Assets/ and syntax-check every .py file.

        Called once from the watchdog thread right after the observer
        starts so that errors present *before* the engine was opened
        are detected immediately.
        """
        from Infernux.engine.script_compiler import get_script_compiler
        from Infernux.components.script_loader import set_script_error, _clear_script_error

        compiler = get_script_compiler()
        error_count = 0
        for root, dirs, files in os.walk(self._assets_path):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(root, fname)
                errors = compiler.check_file(fpath)
                if errors:
                    combined = "\n".join(
                        f"{os.path.basename(e.file_path)}:{e.line_number}  {e.message}"
                        for e in errors
                    )
                    set_script_error(fpath, combined)
                    error_count += 1
                    for e in errors:
                        Debug.log_error(
                            f"Script Error in {os.path.basename(e.file_path)}:{e.line_number}\n{e.message}",
                            source_file=e.file_path,
                            source_line=e.line_number,
                        )
        if error_count:
            Debug.log_warning(f"Startup scan: {error_count} script(s) with errors")
        else:
            Debug.log_internal("\u2713 All scripts passed startup validation")

    def process_pending_reloads(self):
        """Process pending script reloads in main thread. Call this from update loop."""
        if self._event_handler:
            self._event_handler.process_pending_reloads()

    def register_script_reload_callback(self, file_path: str, callback) -> None:
        """Subscribe *callback(file_path)* to be called when *file_path* is saved.

        Called on the main thread after a successful syntax check.
        Safe to call multiple times (duplicates are ignored).
        """
        import os as _os
        abs_path = _os.path.abspath(file_path)
        cbs = self._script_reload_callbacks.setdefault(abs_path, [])
        if callback not in cbs:
            cbs.append(callback)

    def unregister_script_reload_callback(self, callback) -> None:
        """Remove *callback* from all file-path subscriptions."""
        for cbs in self._script_reload_callbacks.values():
            if callback in cbs:
                cbs.remove(callback)

    def register_script_catalog_callback(self, callback) -> None:
        """Subscribe to global Python script catalog changes.

        Callback signature: ``callback(file_path, event_type)`` where
        ``event_type`` is one of ``modified``, ``deleted``, ``moved``.
        """
        if callback not in self._script_catalog_callbacks:
            self._script_catalog_callbacks.append(callback)

    def unregister_script_catalog_callback(self, callback) -> None:
        """Unsubscribe from global Python script catalog changes."""
        if callback in self._script_catalog_callbacks:
            self._script_catalog_callbacks.remove(callback)

    def notify_script_catalog_changed(self, file_path: str, event_type: str) -> None:
        """Notify listeners that Python script catalog may have changed."""
        for cb in list(self._script_catalog_callbacks):
            try:
                cb(file_path, event_type)
            except Exception as e:
                Debug.log_error(f"Script catalog callback failed: {e}")

    def register_shader_cache_callback(self, callback):
        """Register a callback to be called when shader cache should be invalidated."""
        if self._event_handler:
            self._event_handler.register_shader_cache_callback(callback)

    def stop(self):
        """
        Stop the resource monitoring and clean up resources.
        """
        self._stop_event.set()

        # Stop watchdog immediately from the calling thread as well. This makes
        # shutdown robust even if the worker thread is blocked or delayed.
        self._shutdown_observer(join_timeout=5.0)

        # Join the scanning thread (its finally block handles observer teardown).
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                Debug.log_warning("ResourcesManager thread did not stop cleanly before timeout")

    def is_running(self):
        """
        Check if the ResourcesManager is currently running.
        
        Returns:
            bool: True if the manager is running, False otherwise.
        """
        return (self._thread is not None and 
                self._thread.is_alive() and 
                not self._stop_event.is_set())

    def cleanup(self):
        """
        Clean up all resources and stop monitoring.
        This method ensures complete cleanup of the ResourcesManager.
        """
        self.stop()
            
        # Reset internal state
        self._observer = None
        self._thread = None
        self._stop_event.clear()
        self._engine = None
        self._event_handler = None
        self._script_reload_callbacks.clear()
        self._script_catalog_callbacks.clear()
        if ResourcesManager._instance is self:
            ResourcesManager._instance = None
        
        Debug.log_internal("ResourcesManager cleanup completed")
