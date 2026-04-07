"""
Asset Manager — Python-side unified asset loading & caching.

Provides a singleton interface for loading assets by path or GUID,
with WeakRef-based caching to avoid duplicate loads.

Usage::

    from Infernux.core.assets import AssetManager

    # Load by path
    mat = AssetManager.load("Assets/Materials/gold.mat")

    # Load by GUID
    mat = AssetManager.load_by_guid("a1b2c3d4-e5f6-...")

    # Search
    mats = AssetManager.find_assets("*.mat")
"""

from __future__ import annotations

import fnmatch
import os
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Type

from Infernux.core.material import Material
from Infernux.core.texture import Texture
from Infernux.core.shader import Shader
from Infernux.core.audio_clip import AudioClip
from Infernux.core.asset_types import (
    IMAGE_EXTENSIONS, SHADER_EXTENSIONS, MATERIAL_EXTENSIONS, AUDIO_EXTENSIONS,
    asset_category_from_extension,
)

# ── Constants ──
_META_SUPPRESSION_TIMEOUT: float = 2.0  # seconds
_DEFAULT_DEBOUNCE_SEC: float = 0.35  # seconds


class AssetManager:
    """Python-side asset loading & caching manager (singleton pattern).

    Integrates with the C++ AssetDatabase for GUID ↔ path resolution
    and caches loaded assets via weak references.
    """

    # Weak-ref cache: guid → weakref to loaded Python wrapper
    _cache: Dict[str, weakref.ref] = {}

    # Strong-ref cache for textures: guid → Texture
    # Textures are expensive to reload from disk, so keep them alive.
    _texture_cache: Dict[str, Any] = {}

    # Reference to the C++ AssetDatabase (set during engine init)
    _asset_database = None

    # Reference to engine for resource pipeline
    _engine = None

    # Debounced save scheduler: key -> {deadline: float, save_fn: callable}
    _scheduled_saves: Dict[str, Dict[str, Any]] = {}

    # Category -> strategy callables
    _import_apply_handlers: Dict[str, Callable[[str, object], bool]] = {}
    _save_handlers: Dict[str, Callable[[object], object]] = {}
    _execution_strategies_initialized: bool = False

    # Cached reference to C++ AssetRegistry singleton
    _registry = None

    # Paths for which .meta-watcher notifications should be suppressed.
    # Maps normalized path → expiry time (monotonic).  The Apply flow already
    # handles the reload synchronously, so all watcher events that arrive
    # within the window are redundant (Windows may fire >1 event per write).
    _meta_write_suppression: Dict[str, float] = {}

    @classmethod
    def initialize(cls, engine) -> None:
        """Initialize the AssetManager with the engine.

        Called once during engine startup. Sets up the C++ AssetDatabase
        reference and AssetRegistry for unified asset management.
        """
        cls._engine = engine
        native = cls._native_engine()
        if native is not None and hasattr(native, "get_asset_database"):
            cls._asset_database = native.get_asset_database()
        # Cache the AssetRegistry singleton
        cls._registry = cls._resolve_registry()

    @classmethod
    def load(cls, path: str, asset_type: Optional[Type] = None) -> Optional[Any]:
        """Load an asset by file path.

        Supports: .mat (Material)
        More types will be added as wrappers are implemented.

        Args:
            path: File path to the asset (relative or absolute).
            asset_type: Optional type hint. If None, inferred from extension.

        Returns:
            The loaded asset wrapper, or None if loading failed.
        """
        # Try GUID-based cache first
        guid = cls._get_guid_from_path(path)
        if guid:
            cached = cls._get_cached(guid)
            if cached is not None:
                return cached

        # Infer type from extension if not specified
        ext = os.path.splitext(path)[1].lower()
        resolved_type = asset_type or cls._type_from_extension(ext)

        asset = cls._load_by_type(path, resolved_type)
        if asset is not None and guid:
            cls._put_cache(guid, asset)
        return asset

    @classmethod
    def load_by_guid(cls, guid: str, asset_type: Optional[Type] = None) -> Optional[Any]:
        """Load an asset by its GUID.

        Args:
            guid: The asset GUID string.
            asset_type: Optional type hint.

        Returns:
            The loaded asset wrapper, or None.
        """
        # Check cache
        cached = cls._get_cached(guid)
        if cached is not None:
            return cached

        # Resolve path from GUID
        path = cls._get_path_from_guid(guid)
        if not path:
            return None

        ext = os.path.splitext(path)[1].lower()
        resolved_type = asset_type or cls._type_from_extension(ext)

        asset = cls._load_by_type(path, resolved_type)
        if asset is not None:
            if hasattr(asset, "_guid"):
                try:
                    asset._guid = guid
                except (AttributeError, TypeError) as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    pass
            cls._put_cache(guid, asset)
        return asset

    @classmethod
    def find_assets(cls, pattern: str, asset_type: Optional[Type] = None) -> List[str]:
        """Search for asset paths matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g. "*.mat", "Assets/Textures/*.png").
            asset_type: If specified, filter by type.

        Returns:
            List of matching asset paths.
        """
        if not cls._asset_database:
            return []

        results = []
        try:
            guids = cls._asset_database.get_all_guids()
            for guid in guids:
                path = cls._asset_database.get_path_from_guid(guid)
                if path and fnmatch.fnmatch(os.path.basename(path), pattern):
                    if asset_type is not None:
                        ext = os.path.splitext(path)[1].lower()
                        if cls._type_from_extension(ext) != asset_type:
                            continue
                    results.append(path)
        except Exception as e:
            from Infernux.debug import Debug
            Debug.log_warning(f"find_assets error: {e}")
        return results

    @classmethod
    def invalidate(cls, guid: str) -> None:
        """Invalidate a cached asset (e.g. on file change).

        Args:
            guid: GUID of the asset to invalidate.
        """
        cls._cache.pop(guid, None)
        cls._texture_cache.pop(guid, None)

    @classmethod
    def invalidate_path(cls, path: str) -> None:
        """Invalidate a cached asset by path."""
        guid = cls._get_guid_from_path(path)
        if guid:
            cls.invalidate(guid)

    @classmethod
    def flush(cls) -> None:
        """Clear all cached assets."""
        cls._cache.clear()
        cls._texture_cache.clear()

    # ======================================================================
    # Unified execution APIs (Inspector-facing)
    # ======================================================================

    @classmethod
    def register_import_strategy(cls, asset_category: str, apply_fn: Callable[[str, object], bool]):
        """Register import-settings apply function for an asset category."""
        cls._import_apply_handlers[asset_category] = apply_fn

    @classmethod
    def register_save_strategy(cls, asset_category: str, save_fn: Callable[[object], object]):
        """Register save function for an editable asset category."""
        cls._save_handlers[asset_category] = save_fn

    @classmethod
    def _ensure_execution_strategies(cls):
        if cls._execution_strategies_initialized:
            return

        from Infernux.core.asset_types import write_texture_import_settings, write_audio_import_settings, write_mesh_import_settings

        cls.register_import_strategy("texture", write_texture_import_settings)
        cls.register_import_strategy("audio", write_audio_import_settings)
        cls.register_import_strategy("mesh", write_mesh_import_settings)
        cls.register_save_strategy("material", cls._save_material_resource)

        cls._execution_strategies_initialized = True

    @classmethod
    def apply_import_settings(cls, asset_category: str, path: str, settings_obj) -> bool:
        """Apply import settings by category and trigger reimport in one unified step."""
        cls._ensure_execution_strategies()

        apply_fn = cls._import_apply_handlers.get(asset_category)
        if apply_fn is None:
            return False

        # Suppress the file-watcher echo for this .meta write
        normalized = cls._normalize_asset_path(path)
        if normalized:
            cls._meta_write_suppression[normalized] = time.monotonic() + _META_SUPPRESSION_TIMEOUT

        ok = apply_fn(path, settings_obj)
        if not ok:
            cls._meta_write_suppression.pop(normalized, None)
            return False
        cls.reimport_asset(path)

        # Invalidate GPU texture cache so materials pick up the new format
        if asset_category == "texture":
            cls._invalidate_texture_ui_cache(path)
            cls._reload_gpu_texture(path)

        # Reload mesh in AssetRegistry so the new import settings take effect
        if asset_category == "mesh":
            cls._reload_mesh_asset(path)

        return True

    @classmethod
    def reimport_asset(cls, path: str) -> bool:
        """Reimport one asset through AssetDatabase."""
        adb = cls._asset_database
        if not adb or not hasattr(adb, "import_asset"):
            return False
        try:
            guid = adb.import_asset(path)
            return bool(guid)
        except (RuntimeError, OSError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return False

    @classmethod
    def move_asset(cls, old_path: str, new_path: str) -> bool:
        """Move asset path in AssetDatabase while preserving mapping/GUID."""
        adb = cls._asset_database
        if not adb or not hasattr(adb, "move_asset"):
            return False
        try:
            return bool(adb.move_asset(old_path, new_path))
        except (RuntimeError, OSError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return False

    @classmethod
    def schedule_save(cls, key: str, save_fn: Callable[[], object], debounce_sec: float = _DEFAULT_DEBOUNCE_SEC):
        """Schedule a debounced save callback for a resource key (usually file path)."""
        cls._scheduled_saves[key] = {
            "deadline": time.perf_counter() + max(0.0, float(debounce_sec)),
            "save_fn": save_fn,
        }

    @classmethod
    def schedule_asset_save(cls, asset_category: str, key: str, resource_obj, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC):
        """Schedule a debounced save by category strategy, without exposing save callback to caller."""
        cls._ensure_execution_strategies()

        save_handler = cls._save_handlers.get(asset_category)
        if save_handler is None:
            return

        cls.schedule_save(key, lambda: save_handler(resource_obj), debounce_sec=debounce_sec)

    @classmethod
    def _save_material_resource(cls, resource_obj):
        """Save a material resource and invalidate editor preview caches."""
        save = getattr(resource_obj, "save", None)
        if not callable(save):
            return False

        result = save()
        save_ok = bool(result) if result is not None else True
        if save_ok:
            file_path = getattr(resource_obj, "file_path", "") or ""
            if file_path:
                cls.on_material_saved(file_path)
        return result

    @classmethod
    def on_material_saved(cls, path: str) -> None:
        """Invalidate caches that depend on a material asset's file contents."""
        if not path:
            return
        cls.invalidate_path(path)
        cls._invalidate_material_ui_cache(path)

    @classmethod
    def flush_scheduled_saves(cls, key: Optional[str] = None):
        """Execute due scheduled saves. If key is given, only flush that key."""
        now = time.perf_counter()

        if key is not None:
            record = cls._scheduled_saves.get(key)
            if not record:
                return
            if now < float(record.get("deadline", 0.0)):
                return
            try:
                save_fn = record.get("save_fn")
                if callable(save_fn):
                    save_fn()
            finally:
                cls._scheduled_saves.pop(key, None)
            return

        due_keys = [k for k, v in cls._scheduled_saves.items() if now >= float(v.get("deadline", 0.0))]
        for k in due_keys:
            record = cls._scheduled_saves.get(k)
            try:
                if record:
                    save_fn = record.get("save_fn")
                    if callable(save_fn):
                        save_fn()
            finally:
                cls._scheduled_saves.pop(k, None)

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    @classmethod
    def _native_engine(cls):
        """Return the underlying C++ engine handle (unwrap Python wrapper if needed)."""
        engine = cls._engine
        if engine is None:
            return None
        return getattr(engine, '_engine', engine)

    @classmethod
    def _resolve_registry(cls):
        """Resolve the C++ AssetRegistry singleton (lazy, cached)."""
        try:
            from Infernux.lib import AssetRegistry
            return AssetRegistry.instance()
        except (ImportError, RuntimeError, AttributeError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return None

    @classmethod
    def _get_registry(cls):
        """Return the cached AssetRegistry, resolving lazily if needed."""
        if cls._registry is None:
            cls._registry = cls._resolve_registry()
        return cls._registry

    @classmethod
    def _get_guid_from_path(cls, path: str) -> Optional[str]:
        if not cls._asset_database:
            return None
        try:
            guid = cls._asset_database.get_guid_from_path(path)
            return guid if guid else None
        except Exception as e:
            from Infernux.debug import Debug
            Debug.log_warning(f"_get_guid_from_path failed for '{path}': {e}")
            return None

    @classmethod
    def _get_path_from_guid(cls, guid: str) -> Optional[str]:
        if not cls._asset_database:
            return None
        try:
            path = cls._asset_database.get_path_from_guid(guid)
            return path if path else None
        except Exception as e:
            from Infernux.debug import Debug
            Debug.log_warning(f"_get_path_from_guid failed for '{guid}': {e}")
            return None

    @classmethod
    def _get_cached(cls, guid: str) -> Optional[Any]:
        # Strong texture cache (never GC'd until explicit invalidation)
        tex = cls._texture_cache.get(guid)
        if tex is not None:
            return tex
        ref = cls._cache.get(guid)
        if ref is not None:
            obj = ref()
            if obj is not None:
                return obj
            # Dead reference — clean up
            del cls._cache[guid]
        return None

    @classmethod
    def _put_cache(cls, guid: str, asset) -> None:
        if isinstance(asset, Texture):
            cls._texture_cache[guid] = asset
        try:
            cls._cache[guid] = weakref.ref(asset)
        except TypeError as _exc:
            # Object doesn't support weakref — skip caching
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    @classmethod
    def _type_from_extension(cls, ext: str) -> Optional[Type]:
        """Map file extension to Python asset type."""
        ext = ext.lower()
        if ext in MATERIAL_EXTENSIONS:
            return Material
        if ext in IMAGE_EXTENSIONS:
            return Texture
        if ext in SHADER_EXTENSIONS:
            return Shader
        if ext in AUDIO_EXTENSIONS:
            return AudioClip
        return None

    @classmethod
    def _load_by_type(cls, path: str, asset_type: Optional[Type]) -> Optional[Any]:
        """Load an asset given its path and resolved type."""
        if asset_type is Material or (asset_type is None and path.endswith(".mat")):
            return Material.load(path)
        if asset_type is Texture:
            return Texture.load(path)
        # Shader is a static utility — return a ShaderAssetInfo descriptor instead
        if asset_type is Shader:
            from Infernux.core.asset_types import ShaderAssetInfo
            guid = cls._get_guid_from_path(path) or ""
            return ShaderAssetInfo.from_path(path, guid=guid)
        if asset_type is AudioClip:
            return AudioClip.load(path)
        return None

    @classmethod
    def _reload_gpu_texture(cls, path: str) -> None:
        """Invalidate the C++ GPU texture cache so materials re-resolve it.

        Phase 3+ uses GUID-based cache keys, so we resolve path → GUID first.
        Falls back to path-based invalidation for textures not yet in AssetDatabase.
        """
        guid = cls._get_guid_from_path(path)
        native = cls._native_engine()
        if native is not None and hasattr(native, 'reload_texture'):
            # Always pass the file path — C++ ReloadTexture() calls
            # GetGuidFromPath() internally, which fails if given a GUID string.
            native.reload_texture(path)
        # Evict from the Python-side strong cache
        if guid:
            cls._texture_cache.pop(guid, None)
            cls._cache.pop(guid, None)

    @classmethod
    def _reload_mesh_asset(cls, path: str) -> None:
        """Reload a mesh asset in AssetRegistry so updated import settings take effect."""
        guid = cls._get_guid_from_path(path)
        native = cls._native_engine()
        if native is not None and hasattr(native, 'reload_mesh'):
            native.reload_mesh(path)
        if guid:
            cls._cache.pop(guid, None)

    @staticmethod
    def _normalize_asset_path(path: str) -> str:
        if not path:
            return ""
        result = os.path.normpath(path).replace("\\", "/")
        if os.name == "nt":
            result = result.lower()
        return result

    @classmethod
    def _invalidate_texture_ui_cache(cls, path: str) -> None:
        """Invalidate editor-side UI texture previews for a texture asset path."""
        # Resolve GUID — the UI cache is keyed by GUID when possible
        guid = cls._get_guid_from_path(path)
        normalized = cls._normalize_asset_path(path)
        # Collect all identifiers to invalidate (GUID + path variants)
        identifiers = {path, normalized, normalized.replace("/", "\\")}
        if guid:
            identifiers.add(guid)

        try:
            from Infernux.ui import get_shared_cache
            cache = get_shared_cache()
            for ident in identifiers:
                if ident:
                    cache.invalidate(ident)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

        native = cls._native_engine()
        if native is None or not hasattr(native, 'remove_imgui_texture'):
            return

        for ident in identifiers:
            if not ident:
                continue
            try:
                native.remove_imgui_texture(f"__ui_img__{ident}")
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

    @classmethod
    def _invalidate_material_ui_cache(cls, path: str) -> None:
        """Invalidate editor-side cached material thumbnails for a material path."""
        if not path:
            return

        try:
            from Infernux.engine.ui.window_manager import WindowManager
            wm = WindowManager.instance()
            if wm is not None:
                for panel in list(getattr(wm, "_window_instances", {}).values()):
                    invalidate = getattr(panel, "invalidate_material_thumbnail", None)
                    if callable(invalidate):
                        invalidate(path)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    @classmethod
    def _remove_material_pipeline(cls, material_key: str) -> None:
        """Remove MaterialPipelineManager render data by material key."""
        native = cls._native_engine()
        if native is not None and hasattr(native, 'remove_material_pipeline') and material_key:
            native.remove_material_pipeline(material_key)

    @classmethod
    def _remove_material_pipeline_by_path(cls, path: str) -> None:
        """Remove MaterialPipelineManager render data for a material by file path.

        The pipeline manager keys by material name (stem of filename), so we
        derive the key from the path and call engine.remove_material_pipeline().
        """
        import os
        native = cls._native_engine()
        if native is None or not hasattr(native, 'remove_material_pipeline'):
            return
        mat_name = os.path.splitext(os.path.basename(path))[0]
        if mat_name:
            native.remove_material_pipeline(mat_name)

    @classmethod
    def _clear_deleted_texture_from_active_ui(cls, path: str) -> bool:
        """Clear stale texture_path fields from active UI Python components."""
        normalized = cls._normalize_asset_path(path)
        if not normalized:
            return False

        changed = False

        try:
            from Infernux.lib import SceneManager

            scene = SceneManager.instance().get_active_scene()
            if scene is None:
                return False

            for game_object in scene.get_all_objects():
                if game_object is None:
                    continue
                for py_comp in game_object.get_py_components():
                    tex_path = getattr(py_comp, "texture_path", None)
                    if not isinstance(tex_path, str) or not tex_path:
                        continue
                    if cls._normalize_asset_path(tex_path) != normalized:
                        continue
                    setattr(py_comp, "texture_path", "")
                    changed = True
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return False

        if changed:
            try:
                from Infernux.engine.scene_manager import SceneFileManager

                sfm = SceneFileManager.instance()
                if sfm is not None:
                    sfm.mark_dirty()
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

        return changed

    @classmethod
    def on_asset_deleted(cls, path: str) -> None:
        """Handle deletion — delegate to AssetRegistry, then Python-side cleanup."""
        import os
        from Infernux.core.asset_types import IMAGE_EXTENSIONS, MATERIAL_EXTENSIONS

        # Resolve GUID before AssetRegistry clears the mapping
        guid = cls._get_guid_from_path(path)

        # Unified delegation — AssetRegistry evicts from C++ cache
        registry = cls._get_registry()
        if registry:
            registry.on_asset_deleted(path)

        # Invalidate Python-side weak-ref cache
        cls.invalidate_path(path)

        ext = os.path.splitext(path)[1].lower()

        # GPU pipeline cleanup for deleted materials
        if ext in MATERIAL_EXTENSIONS:
            if guid:
                cls._remove_material_pipeline(guid)
            else:
                cls._remove_material_pipeline_by_path(path)

        # GPU texture cache cleanup
        if ext in IMAGE_EXTENSIONS:
            cls._invalidate_texture_ui_cache(path)
            cls._clear_deleted_texture_from_active_ui(path)
            cls._reload_gpu_texture(path)

    @classmethod
    def on_asset_moved(cls, old_path: str, new_path: str) -> None:
        """Handle rename/move — delegate to AssetRegistry, then Python-side cleanup."""
        import os
        from Infernux.core.asset_types import IMAGE_EXTENSIONS

        # Unified delegation — AssetRegistry updates GUID↔path mapping
        registry = cls._get_registry()
        if registry:
            registry.on_asset_moved(old_path, new_path)

        # Invalidate Python-side caches for old path
        cls.invalidate_path(old_path)

        # GPU texture cache needs explicit invalidation for old path
        ext = os.path.splitext(old_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            cls._invalidate_texture_ui_cache(old_path)

    @classmethod
    def on_asset_modified(cls, path: str) -> None:
        """Handle file modification — delegate to AssetRegistry, then Python-side cleanup."""
        import os
        from Infernux.core.asset_types import IMAGE_EXTENSIONS
        from Infernux.debug import Debug

        # Check suppression (set by apply_import_settings to avoid echo reloads).
        normalized = cls._normalize_asset_path(path)
        expiry = cls._meta_write_suppression.get(normalized)
        if expiry is not None:
            if time.monotonic() < expiry:
                Debug.log_internal(f"[AssetManager] suppressed watcher event for '{path}'")
                return
            cls._meta_write_suppression.pop(normalized, None)
            Debug.log_internal(f"[AssetManager] suppression expired for '{path}', processing normally")

        # Unified delegation: AssetRegistry handles reload for ALL cached asset types
        registry = cls._get_registry()
        if registry:
            registry.on_asset_modified(path)

        # Invalidate Python-side weak-ref cache
        cls.invalidate_path(path)

        # GPU texture cache requires explicit invalidation (not managed by AssetRegistry)
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            cls._invalidate_texture_ui_cache(path)
            cls._reload_gpu_texture(path)
