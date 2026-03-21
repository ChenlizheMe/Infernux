"""
Unity-style Project panel for browsing assets.

File operations and templates are in ``project_file_ops``.
Pure utilities are in ``project_utils``.
"""

import os
import shutil
import threading
from InfEngine.lib import InfGUIContext, TextureLoader, InputManager
from InfEngine.engine.i18n import t
from InfEngine.resources import file_type_icons_dir
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from . import project_file_ops as file_ops
from . import project_utils
from .imgui_keys import (KEY_F2, KEY_DELETE, KEY_ENTER, KEY_ESCAPE,
                         KEY_C, KEY_V, KEY_X,
                         KEY_LEFT_CTRL, KEY_RIGHT_CTRL,
                         KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT)


# ---------------------------------------------------------------------------
# Module-level thumbnail pixel cache (survives panel re-creation / hot-reload)
# ---------------------------------------------------------------------------

def _downsample_texture(file_path: str, max_px: int):
    """Load *file_path* via stb and return ``(pixels, w, h)``.

    If the source image is larger than *max_px* in either dimension it is
    nearest-neighbour downsampled so the longest edge equals *max_px*.
    Returns ``list[int]`` suitable for ``upload_texture_for_imgui``.
    """
    tex_data = TextureLoader.load_from_file(file_path)
    if not tex_data or not tex_data.is_valid():
        raise ValueError("load failed")
    src_w, src_h = tex_data.width, tex_data.height
    if src_w > max_px or src_h > max_px:
        raw = tex_data.get_pixels()  # bytes – fast C++ transfer
        scale = max_px / max(src_w, src_h)
        w = max(1, int(src_w * scale))
        h = max(1, int(src_h * scale))
        pixels = []
        row_stride = src_w * 4
        for dy in range(h):
            sy = min(int((dy + 0.5) * src_h / h), src_h - 1)
            row_off = sy * row_stride
            for dx in range(w):
                sx = min(int((dx + 0.5) * src_w / w), src_w - 1)
                idx = row_off + sx * 4
                pixels.append(raw[idx])
                pixels.append(raw[idx + 1])
                pixels.append(raw[idx + 2])
                pixels.append(raw[idx + 3])
    else:
        pixels = tex_data.get_pixels_list()
        w, h = src_w, src_h
    return pixels, w, h


class _ThumbnailPixelCache:
    """Background pre-loader that holds downsampled RGBA pixels in memory.

    On ``start_preload`` a daemon thread walks the project tree, loads every
    image file via *stb*, downsamples to *max_px* and stores the pixel list.
    ``ProjectPanel._get_thumbnail`` queries this cache; if a hit is found the
    only remaining work is a fast GPU upload.
    """

    def __init__(self, max_px: int = 128):
        self._max_px = max_px
        self._lock = threading.Lock()
        # path -> (pixels: list[int], width, height, mtime_ns)
        self._data: dict = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- public API ---------------------------------------------------------

    def start_preload(self, root_path: str, extensions: set):
        """Begin (or restart) background scanning of *root_path*."""
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(root_path, frozenset(extensions)),
            daemon=True,
            name="ThumbnailPreloader",
        )
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=3.0)
        self._thread = None

    def get(self, path: str, mtime_ns: int):
        """Return ``(pixels, w, h)`` if cached and fresh, else *None*."""
        with self._lock:
            entry = self._data.get(path)
        if entry is not None and entry[3] == mtime_ns:
            return entry[0], entry[1], entry[2]
        return None

    def put(self, path: str, pixels, w: int, h: int, mtime_ns: int):
        """Store pixel data produced by a foreground load."""
        with self._lock:
            self._data[path] = (pixels, w, h, mtime_ns)

    # -- background worker --------------------------------------------------

    def _worker(self, root_path: str, extensions: frozenset):
        import time
        for dirpath, _dirnames, filenames in os.walk(root_path):
            if self._stop.is_set():
                return
            for fname in filenames:
                if self._stop.is_set():
                    return
                ext = os.path.splitext(fname)[1].lower()
                if ext not in extensions:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    mtime = os.stat(fpath).st_mtime_ns
                except OSError:
                    continue
                # Already cached and fresh?
                with self._lock:
                    cached = self._data.get(fpath)
                if cached is not None and cached[3] == mtime:
                    continue
                try:
                    pixels, w, h = _downsample_texture(fpath, self._max_px)
                except Exception:
                    continue
                with self._lock:
                    self._data[fpath] = (pixels, w, h, mtime)
                # Yield GIL so the render thread stays responsive
                time.sleep(0.005)


_thumb_pixel_cache = _ThumbnailPixelCache(max_px=128)


@editor_panel("Project", type_id="project", title_key="panel.project")
class ProjectPanel(EditorPanel):
    """
    Unity-style Project panel for browsing assets.
    Left: Folder tree view
    Right: File grid/list view
    Supports: Create folders, create scripts, double-click to open
    """
    
    WINDOW_TYPE_ID = "project"
    WINDOW_DISPLAY_NAME = "Project"
    
    # File extensions to hide
    HIDDEN_EXTENSIONS = {'.meta', '.pyc', '.pyo'}
    HIDDEN_PREFIXES = {'.', '__'}
    
    # Image extensions that support thumbnails
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.gif'}
    
    # Extension -> icon filename (without .png) mapping
    # Icons are 80x80 PNGs located in resources/icons/
    ICON_MAP = {
        # Directories
        '__dir__':  'folder',
        # Scripts
        '.py':      'script_py',
        '.lua':     'script_lua',
        '.cs':      'script_cs',
        '.cpp':     'script_cpp',
        '.c':       'script_cpp',
        '.h':       'script_cpp',
        # Shaders — each type has its own icon
        '.vert':    'shader_vert',
        '.frag':    'shader_frag',
        '.glsl':    'shader_glsl',
        '.hlsl':    'shader_hlsl',
        # Materials
        '.mat':     'material',
        # Images
        '.png':     'image',
        '.jpg':     'image',
        '.jpeg':    'image',
        '.bmp':     'image',
        '.tga':     'image',
        '.gif':     'image',
        # 3D models
        '.fbx':     'model_3d',
        '.obj':     'model_3d',
        '.gltf':    'model_3d',
        '.glb':     'model_3d',
        # Audio
        '.wav':     'audio',
        # Fonts
        '.ttf':     'font',
        '.otf':     'font',
        # Text / docs
        '.txt':     'text',
        '.md':      'readme',
        # Config
        '.json':    'config',
        '.yaml':    'config',
        '.yml':     'config',
        '.xml':     'config',
        # Scene
        '.scene':   'scene',
        # Prefab
        '.prefab':  'prefab',
    }
    
    SCRIPT_TEMPLATE = file_ops.SCRIPT_TEMPLATE
    VERTEX_SHADER_TEMPLATE = file_ops.VERTEX_SHADER_TEMPLATE
    FRAGMENT_SHADER_TEMPLATE = file_ops.FRAGMENT_SHADER_TEMPLATE
    MATERIAL_TEMPLATE = file_ops.MATERIAL_TEMPLATE

    # Thumbnail size for image previews (downsampled from original)
    THUMBNAIL_MAX_PX = 128

    # 3D model extensions that support sub-asset expansion
    MODEL_EXTENSIONS = {'.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.ply', '.stl'}
    
    # Drag-drop: extension -> (payload_type, label_prefix)
    _DRAG_DROP_MAP = {
        '.py':    ("SCRIPT_FILE",  "Script"),
        '.mat':   ("MATERIAL_FILE", "Material"),
        '.vert':  ("SHADER_FILE",  "Shader"),
        '.frag':  ("SHADER_FILE",  "Shader"),
        '.glsl':  ("SHADER_FILE",  "Shader"),
        '.hlsl':  ("SHADER_FILE",  "Shader"),
        '.png':   ("TEXTURE_FILE", "Texture"),
        '.jpg':   ("TEXTURE_FILE", "Texture"),
        '.jpeg':  ("TEXTURE_FILE", "Texture"),
        '.bmp':   ("TEXTURE_FILE", "Texture"),
        '.tga':   ("TEXTURE_FILE", "Texture"),
        '.gif':   ("TEXTURE_FILE", "Texture"),
        '.psd':   ("TEXTURE_FILE", "Texture"),
        '.hdr':   ("TEXTURE_FILE", "Texture"),
        '.pic':   ("TEXTURE_FILE", "Texture"),
        '.wav':   ("AUDIO_FILE",   "Audio"),
        '.ttf':   ("FONT_FILE",    "Font"),
        '.otf':   ("FONT_FILE",    "Font"),
        '.scene': ("SCENE_FILE",   "Scene"),
    }
    # Extensions that need GUID resolution for drag-drop
    _DRAG_DROP_GUID_MAP = {
        '.prefab': ("PREFAB_GUID", "PREFAB_FILE", "Prefab"),
        '.fbx':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.obj':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.gltf':   ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.glb':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.dae':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.3ds':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.ply':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
        '.stl':    ("MODEL_GUID",  "MODEL_FILE",  "Model"),
    }

    def __init__(self, root_path: str = "", title: str = "Project", engine=None):
        super().__init__(title, window_id="project")
        self.__root_path = root_path
        # Default to Assets folder if it exists
        assets_path = os.path.join(root_path, "Assets") if root_path else ""
        if assets_path and os.path.exists(assets_path):
            self.__current_path = assets_path
        else:
            self.__current_path = root_path
        self.__selected_file = None
        self.__selected_files: list[str] = []  # Multi-select: ordered list of paths
        self.__on_file_selected = None
        self.__on_file_double_click = None

        # Cached breadcrumb text (avoids os.path.relpath every frame)
        self.__breadcrumb_path = ""
        self.__breadcrumb_text = ""
        
        # Engine and asset database reference
        self.__engine = engine
        self.__native_engine = None
        self.__asset_database = None
        if engine:
            self.set_engine(engine)
        
        # Directory snapshot cache: path -> snapshot dict keyed by dir mtime
        self.__dir_cache = {}

        # Thumbnail cache: path -> (texture_id, last_modified_time)
        self.__thumbnail_cache = {}
        
        # File-type icon cache: icon_key (str) -> imgui texture id (int)
        self.__type_icon_cache = {}
        self.__type_icons_loaded = False
        
        # State for create dialogs
        self.__show_create_folder_popup = False
        self.__show_create_script_popup = False
        self.__new_item_name = ""
        self.__create_error = ""
        
        # Double-click detection state
        self.__last_clicked_file = None
        self.__last_click_time = 0.0

        # Rename state
        self.__renaming_path = None
        self.__renaming_name = ""
        self.__rename_focus_requested = False

        # Clipboard state (copy/cut)
        self.__clipboard_path = None   # Path of copied/cut file
        self.__clipboard_is_cut = False

        # Model asset expansion state: path -> expanded bool
        self.__expanded_models: set = set()
        # Model sub-asset cache: path -> {'slot_names': [...], 'submeshes': [...]} or None
        self.__model_sub_cache: dict = {}

        # Kick off background thumbnail pre-load
        if root_path:
            _thumb_pixel_cache.start_preload(root_path, self.IMAGE_EXTENSIONS)

    # ---- State persistence ----
    def save_state(self) -> dict:
        return {
            "current_path": self.__current_path,
        }

    def load_state(self, data: dict) -> None:
        path = data.get("current_path", "")
        if path and os.path.isdir(path):
            self.__current_path = path
    
    def set_root_path(self, path: str):
        self.__root_path = path
        self._invalidate_dir_cache()
        # Default to Assets folder
        assets_path = os.path.join(path, "Assets")
        if os.path.exists(assets_path):
            self.__current_path = assets_path
        else:
            self.__current_path = path
        # Restart background thumbnail pre-load for the new root
        if path:
            _thumb_pixel_cache.start_preload(path, self.IMAGE_EXTENSIONS)
    
    def set_on_file_selected(self, callback):
        self.__on_file_selected = callback

    def clear_selection(self):
        """Clear current file selection and notify listeners."""
        if self.__selected_file is not None or self.__selected_files:
            self.__selected_file = None
            self.__selected_files.clear()
            if self.__on_file_selected:
                self.__on_file_selected(None)
    
    def set_on_file_double_click(self, callback):
        self.__on_file_double_click = callback
    
    def set_engine(self, engine):
        """Set the engine instance for resource management."""
        self.__engine = engine
        print(f"[ProjectPanel] set_engine called with: {type(engine)}")
        self.__native_engine = None
        if engine:
            # Try to get FileManager from engine
            if hasattr(engine, 'has_imgui_texture'):
                self.__native_engine = engine
            elif hasattr(engine, 'get_native_engine'):
                self.__native_engine = engine.get_native_engine()
            # Try to get AssetDatabase from engine
            if hasattr(engine, 'get_asset_database'):
                self.__asset_database = engine.get_asset_database()
            elif hasattr(engine, 'get_native_engine'):
                native = engine.get_native_engine()
                if native and hasattr(native, 'get_asset_database'):
                    self.__asset_database = native.get_asset_database()
    
    def _invalidate_dir_cache(self):
        self.__dir_cache.clear()
        self.__model_sub_cache.clear()

    @staticmethod
    def _get_mtime_ns(path: str):
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return None

    def _get_dir_snapshot(self, path: str):
        """Return a cached directory snapshot for *path*."""
        if not path or not os.path.isdir(path):
            return None

        mtime_ns = self._get_mtime_ns(path)
        cached = self.__dir_cache.get(path)
        if cached is not None and cached.get('mtime_ns') == mtime_ns:
            return cached

        dirs = []
        files = []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    name = entry.name
                    if not self._should_show(name):
                        continue

                    full_path = entry.path
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        continue

                    if is_dir:
                        dirs.append({
                            'type': 'dir',
                            'name': name,
                            'path': full_path,
                            'ext': '',
                            'mtime_ns': None,
                        })
                        continue

                    ext = os.path.splitext(name)[1].lower()
                    file_mtime_ns = None
                    if ext in self.IMAGE_EXTENSIONS:
                        try:
                            file_mtime_ns = entry.stat().st_mtime_ns
                        except OSError:
                            file_mtime_ns = None
                    files.append({
                        'type': 'file',
                        'name': name,
                        'path': full_path,
                        'ext': ext,
                        'mtime_ns': file_mtime_ns,
                    })
        except OSError:
            return None

        dirs.sort(key=lambda item: item['name'].lower())
        files.sort(key=lambda item: item['name'].lower())
        snapshot = {
            'mtime_ns': mtime_ns,
            'dirs': dirs,
            'files': files,
            'items': dirs + files,
        }
        self.__dir_cache[path] = snapshot
        return snapshot

    def _get_thumbnail(self, file_path: str, size: float, cached_mtime_ns=None) -> int:
        """Return an ImGui texture id for a low-res thumbnail, or 0.

        Pixel data is served from the module-level ``_thumb_pixel_cache``
        (populated by the background pre-loader at startup).  Only the fast
        GPU upload happens on the render thread.
        """
        if not file_path:
            return 0

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.IMAGE_EXTENSIONS:
            return 0

        thumbnail_name = f"__thumb__{file_path}"

        native_engine = self._get_native_engine()
        if not native_engine:
            return 0

        if cached_mtime_ns is None:
            if not os.path.exists(file_path):
                return 0
            cached_mtime_ns = self._get_mtime_ns(file_path)

        # 1. Python-side GPU-id cache (fastest)
        if file_path in self.__thumbnail_cache:
            cached_id, cached_mtime = self.__thumbnail_cache[file_path]
            if cached_mtime == cached_mtime_ns and cached_id != 0:
                return cached_id

        # 2. Engine-side GPU texture cache
        if native_engine.has_imgui_texture(thumbnail_name):
            tex_id = native_engine.get_imgui_texture_id(thumbnail_name)
            if tex_id != 0:
                self.__thumbnail_cache[file_path] = (tex_id, cached_mtime_ns)
                return tex_id

        # 3. Pixel cache (from background pre-loader) — only a GPU upload needed
        cached_pixels = _thumb_pixel_cache.get(file_path, cached_mtime_ns)
        if cached_pixels is not None:
            pixels, w, h = cached_pixels
            tex_id = native_engine.upload_texture_for_imgui(
                thumbnail_name, pixels, w, h)
            if tex_id != 0:
                self.__thumbnail_cache[file_path] = (tex_id, cached_mtime_ns)
            return tex_id

        # 4. Foreground load (pixel cache miss) — throttled to 2 per frame
        if not hasattr(self, '_thumbnails_loaded_this_frame'):
            self._thumbnails_loaded_this_frame = 0
        if self._thumbnails_loaded_this_frame >= 2:
            return 0

        try:
            pixels, w, h = _downsample_texture(file_path, self.THUMBNAIL_MAX_PX)
        except Exception:
            return 0

        # Store in pixel cache so future visits are instant
        _thumb_pixel_cache.put(file_path, pixels, w, h, cached_mtime_ns)

        tex_id = native_engine.upload_texture_for_imgui(
            thumbnail_name, pixels, w, h)
        if tex_id != 0:
            self.__thumbnail_cache[file_path] = (tex_id, cached_mtime_ns)
            self._thumbnails_loaded_this_frame += 1
            return tex_id

        return 0
    
    def _reset_frame_counters(self):
        """Reset per-frame counters. Call at start of on_render."""
        self._thumbnails_loaded_this_frame = 0
    
    # ------------------------------------------------------------------ icons
    def _get_native_engine(self):
        """Resolve to the raw C++ InfEngine object."""
        return self.__native_engine

    def _ensure_type_icons_loaded(self):
        """Lazily upload all file-type icons to GPU (once)."""
        if self.__type_icons_loaded:
            return
        native = self._get_native_engine()
        if native is None:
            return

        # Collect unique icon filenames we need
        needed = set(self.ICON_MAP.values())
        needed.add('file')  # generic fallback

        for icon_key in needed:
            tex_name = f"__typeicon__{icon_key}"
            # Already uploaded in a previous session / hot-reload?
            if native.has_imgui_texture(tex_name):
                self.__type_icon_cache[icon_key] = native.get_imgui_texture_id(tex_name)
                continue

            icon_path = os.path.join(file_type_icons_dir, f"{icon_key}.png")
            if not os.path.isfile(icon_path):
                continue  # user hasn't added this icon yet, will fall back to text

            tex_data = TextureLoader.load_from_file(icon_path)
            if tex_data and tex_data.is_valid():
                pixels = tex_data.get_pixels_list()
                tid = native.upload_texture_for_imgui(
                    tex_name, pixels, tex_data.width, tex_data.height)
                if tid != 0:
                    self.__type_icon_cache[icon_key] = tid

        self.__type_icons_loaded = True

    def _get_type_icon_id(self, item_type: str, ext: str) -> int:
        """Return the ImGui texture id for a file type, or 0 if not available."""
        if item_type == 'dir':
            key = self.ICON_MAP.get('__dir__')
        else:
            key = self.ICON_MAP.get(ext)
        if key is None:
            key = 'file'  # generic fallback
        return self.__type_icon_cache.get(key, 0)

    def _get_model_sub_assets(self, file_path: str):
        """Return cached sub-asset info for a model file, or None.

        Returns dict with keys ``slot_names`` (list[str]) and
        ``submeshes`` (list[dict]) on success, or *None* if the mesh
        cannot be loaded (e.g. not yet imported).
        """
        cached = self.__model_sub_cache.get(file_path)
        if cached is not None:
            return cached
        if not self.__asset_database:
            return None
        try:
            guid = self.__asset_database.get_guid_from_path(file_path)
        except Exception:
            return None
        if not guid:
            return None
        try:
            from InfEngine.lib import AssetRegistry
            registry = AssetRegistry.instance()
            mesh = registry.load_mesh_by_guid(guid)
        except Exception:
            return None
        if mesh is None:
            return None
        slot_names = list(mesh.material_slot_names) if mesh.material_slot_names else []
        submeshes = []
        for i in range(mesh.submesh_count):
            try:
                submeshes.append(mesh.get_submesh_info(i))
            except Exception:
                break
        result = {'slot_names': slot_names, 'submeshes': submeshes,
                  'vertex_count': mesh.vertex_count, 'index_count': mesh.index_count}
        self.__model_sub_cache[file_path] = result
        return result

    def _delete_item(self, item_path: str):
        file_ops.delete_item(item_path, self.__asset_database)
        self._invalidate_dir_cache()
        if self.__selected_file == item_path:
            self.__selected_file = None
            self.__selected_files.clear()
            if self.__on_file_selected:
                self.__on_file_selected(None)
        if item_path in self.__thumbnail_cache:
            del self.__thumbnail_cache[item_path]
    
    def _should_show(self, name: str) -> bool:
        return project_utils.should_show(name)
    
    def _open_file_with_system(self, file_path: str):
        project_utils.open_file_with_system(file_path, project_root=self.__root_path)
    
    def _create_folder(self, folder_name: str):
        ok, err = file_ops.create_folder(self.__current_path, folder_name)
        if not ok:
            self.__create_error = err
        else:
            self._invalidate_dir_cache()
        return ok, err
    
    def _create_script(self, script_name: str):
        ok, err = file_ops.create_script(self.__current_path, script_name,
                                         self.__asset_database)
        if not ok:
            self.__create_error = err
        else:
            self._invalidate_dir_cache()
        return ok, err
    
    def _create_shader(self, shader_name: str, shader_type: str):
        ok, err = file_ops.create_shader(self.__current_path, shader_name, shader_type,
                                         self.__asset_database)
        if not ok:
            self.__create_error = err
        else:
            self._invalidate_dir_cache()
        return ok, err
    
    def _create_material(self, material_name: str):
        ok, err = file_ops.create_material(self.__current_path, material_name,
                                           self.__asset_database)
        if not ok:
            self.__create_error = err
        else:
            self._invalidate_dir_cache()
        return ok, err

    def _create_scene(self, scene_name: str):
        """Create a new .scene file and return ``(True, path)`` or ``(False, error)``."""
        ok, result = file_ops.create_scene(self.__current_path, scene_name,
                                           self.__asset_database)
        if not ok:
            self.__create_error = result
            return False, result
        self._invalidate_dir_cache()
        return True, result

    def _create_prefab_from_hierarchy(self, obj_id: int):
        """Create a .prefab file from a hierarchy GameObject dragged onto the Project panel."""
        from InfEngine.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        game_object = scene.find_by_id(obj_id)
        if not game_object:
            return
        ok, result = file_ops.create_prefab_from_gameobject(
            game_object, self.__current_path, self.__asset_database
        )
        if ok:
            self._invalidate_dir_cache()
        else:
            from InfEngine.debug import Debug
            Debug.log_warning(f"Failed to create prefab: {result}")

    def _open_scene_file(self, file_path: str):
        """Open a .scene file via the SceneFileManager."""
        from InfEngine.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.open_scene(file_path)
        else:
            from InfEngine.debug import Debug
            Debug.log_warning("SceneFileManager not initialized")

    def _get_unique_name(self, base_name: str, extension: str = "") -> str:
        return file_ops.get_unique_name(self.__current_path, base_name, extension)
    
    def _do_rename(self):
        if not self.__renaming_path or not self.__renaming_name:
            self.__renaming_path = None
            return
        old_path = self.__renaming_path
        new_path = file_ops.do_rename(old_path, self.__renaming_name,
                                      self.__asset_database)
        if new_path:
            self._invalidate_dir_cache()
        if new_path and self.__selected_file == old_path:
            self.__selected_file = new_path
            self.__selected_files = [new_path]
        if new_path and self.__current_path == old_path:
            self.__current_path = new_path
        self.__renaming_path = None
    
    def _update_material_name_in_file(self, mat_path: str, new_name: str):
        project_utils.update_material_name_in_file(mat_path, new_name)

    def _create_and_rename(self, create_fn, base_name: str, extension: str = ""):
        """Create an item then immediately enter rename mode for it.

        *create_fn* is called with the generated name and must return ``(ok, ...)``.
        """
        name = self._get_unique_name(base_name, extension)
        ok, *_ = create_fn(name)
        if ok:
            file_name = name + extension if extension and not name.endswith(extension) else name
            new_path = os.path.join(self.__current_path, file_name)
            self.__selected_file = new_path
            self.__selected_files = [new_path]
            if self.__on_file_selected:
                self.__on_file_selected(new_path)
            # Enter rename mode
            self.__renaming_path = new_path
            disp = os.path.basename(new_path)
            if os.path.isfile(new_path):
                disp = os.path.splitext(disp)[0]
            self.__renaming_name = disp
            self.__rename_focus_requested = True

    def _reveal_in_file_explorer(self, path: str):
        """Open the OS file explorer and highlight *path*."""
        project_utils.reveal_in_file_explorer(path)

    # ── clipboard helpers ────────────────────────────────────────────────

    def _clipboard_copy(self, path: str):
        self.__clipboard_path = path
        self.__clipboard_is_cut = False

    def _clipboard_cut(self, path: str):
        self.__clipboard_path = path
        self.__clipboard_is_cut = True

    def _clipboard_paste(self):
        """Paste the clipboard file/folder into the current directory."""
        src = self.__clipboard_path
        if not src or not os.path.exists(src):
            self.__clipboard_path = None
            return
        name = os.path.basename(src)
        dst = os.path.join(self.__current_path, name)
        # Avoid overwriting — generate unique name
        if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
            base, ext = os.path.splitext(name)
            if os.path.isdir(src):
                ext = ""
                base = name
            dst_name = file_ops.get_unique_name(self.__current_path, base, ext)
            dst = os.path.join(self.__current_path, dst_name + ext)
        if self.__clipboard_is_cut:
            try:
                shutil.move(src, dst)
            except OSError:
                return
            self.__clipboard_path = None
        else:
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            except OSError:
                return
        self._invalidate_dir_cache()
        self.__selected_file = dst
        self.__selected_files = [dst]
        if self.__on_file_selected:
            self.__on_file_selected(dst)

    # ── external file drop (from Windows Explorer) ───────────────────────

    def _handle_external_file_drops(self):
        """Check InputManager for files dropped from the OS and copy them in."""
        try:
            mgr = InputManager.instance()
        except Exception:
            return
        if not mgr.has_dropped_files():
            return
        dropped = mgr.get_dropped_files()
        if not dropped or not self.__current_path:
            return
        last_path = None
        for src in dropped:
            if not os.path.exists(src):
                continue
            name = os.path.basename(src)
            dst = os.path.join(self.__current_path, name)
            if os.path.exists(dst):
                base, ext = os.path.splitext(name)
                if os.path.isdir(src):
                    ext = ""
                    base = name
                dst_name = file_ops.get_unique_name(self.__current_path, base, ext)
                dst = os.path.join(self.__current_path, dst_name + ext)
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                last_path = dst
            except OSError:
                continue
        if last_path:
            self._invalidate_dir_cache()
            self.__selected_file = last_path
            self.__selected_files = [last_path]
            if self.__on_file_selected:
                self.__on_file_selected(last_path)
    
    def _handle_item_click(self, item: dict, ctx=None):
        """Handle click on an item (file or folder) with ctrl/shift multi-select."""
        import time
        current_time = time.time()
        double_clicked = (self.__last_clicked_file == item['path'] and 
                          current_time - self.__last_click_time < 0.4)
        
        self.__last_clicked_file = item['path']
        self.__last_click_time = current_time

        path = item['path']

        # Detect modifier keys
        ctrl = False
        shift = False
        if ctx is not None:
            ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)
            shift = ctx.is_key_down(KEY_LEFT_SHIFT) or ctx.is_key_down(KEY_RIGHT_SHIFT)

        if ctrl and not double_clicked:
            # Ctrl+click: toggle this item in the selection
            if path in self.__selected_files:
                self.__selected_files.remove(path)
                self.__selected_file = self.__selected_files[-1] if self.__selected_files else None
            else:
                self.__selected_files.append(path)
                self.__selected_file = path
            if self.__on_file_selected:
                self.__on_file_selected(self.__selected_file)
            return
        elif shift and not double_clicked and self.__selected_file and hasattr(self, '_visible_items'):
            # Shift+click: range select from last selected to current
            try:
                paths = [it['path'] for it in self._visible_items]
                anchor_idx = paths.index(self.__selected_file)
                target_idx = paths.index(path)
                lo, hi = min(anchor_idx, target_idx), max(anchor_idx, target_idx)
                self.__selected_files = paths[lo:hi + 1]
                # Keep __selected_file as the anchor (don't change it)
            except ValueError:
                self.__selected_files = [path]
                self.__selected_file = path
            if self.__on_file_selected:
                self.__on_file_selected(self.__selected_file)
            return

        # Normal click (no modifier): single-select
        self.__selected_files = [path]
        self.__selected_file = path
        if self.__on_file_selected:
            self.__on_file_selected(self.__selected_file)
        
        if item['type'] == 'dir':
            if double_clicked:
                self.__current_path = item['path']
                self.__last_clicked_file = None
        elif item['type'] in ('sub_mesh', 'sub_material'):
            pass  # Sub-asset items: select only
        else:
            if double_clicked:
                ext = os.path.splitext(item['path'])[1].lower()
                if ext in self.MODEL_EXTENSIONS:
                    if item['path'] in self.__expanded_models:
                        self.__expanded_models.discard(item['path'])
                    else:
                        self.__expanded_models.add(item['path'])
                elif ext == '.scene':
                    self._open_scene_file(item['path'])
                else:
                    self._open_file_with_system(item['path'])

    def _render_context_menu(self, ctx: InfGUIContext):
        """Render right-click context menu for creating items."""
        if ctx.begin_popup_context_window("ProjectContextMenu", 1):
            if ctx.begin_menu(t("project.create_menu")):
                if ctx.selectable(t("project.create_folder"), False, 0, 0, 0):
                    self._create_and_rename(self._create_folder, "NewFolder")
                ctx.separator()
                if ctx.selectable(t("project.create_script"), False, 0, 0, 0):
                    self._create_and_rename(self._create_script, "NewComponent", ".py")
                ctx.separator()
                if ctx.selectable(t("project.create_vert_shader"), False, 0, 0, 0):
                    self._create_and_rename(
                        lambda n: self._create_shader(n, "vert"), "NewShader", ".vert")
                if ctx.selectable(t("project.create_frag_shader"), False, 0, 0, 0):
                    self._create_and_rename(
                        lambda n: self._create_shader(n, "frag"), "NewShader", ".frag")
                ctx.separator()
                if ctx.selectable(t("project.create_material"), False, 0, 0, 0):
                    self._create_and_rename(self._create_material, "NewMaterial", ".mat")
                ctx.separator()
                if ctx.selectable(t("project.create_scene"), False, 0, 0, 0):
                    self._create_and_rename(self._create_scene, "NewScene", ".scene")
                ctx.end_menu()
            
            if self.__selected_file and os.path.exists(self.__selected_file):
                ctx.separator()
                if ctx.selectable(t("project.reveal_in_explorer"), False, 0, 0, 0):
                    self._reveal_in_file_explorer(self.__selected_file)
                ctx.separator()
                if ctx.selectable(t("project.copy"), False, 0, 0, 0):
                    self._clipboard_copy(self.__selected_file)
                if ctx.selectable(t("project.cut"), False, 0, 0, 0):
                    self._clipboard_cut(self.__selected_file)
                if self.__clipboard_path and os.path.exists(self.__clipboard_path):
                    if ctx.selectable(t("project.paste"), False, 0, 0, 0):
                        self._clipboard_paste()
                ctx.separator()
                if ctx.selectable(t("project.rename"), False, 0, 0, 0):
                    self.__renaming_path = self.__selected_file
                    name = os.path.basename(self.__selected_file)
                    if os.path.isfile(self.__renaming_path):
                        name = os.path.splitext(name)[0]
                    self.__renaming_name = name
                    self.__rename_focus_requested = True
                if ctx.selectable(t("project.delete"), False, 0, 0, 0):
                    self._delete_item(self.__selected_file)
            else:
                # No item selected — still offer paste & reveal on folder
                ctx.separator()
                if ctx.selectable(t("project.reveal_in_explorer"), False, 0, 0, 0):
                    self._reveal_in_file_explorer(self.__current_path)
                if self.__clipboard_path and os.path.exists(self.__clipboard_path):
                    if ctx.selectable(t("project.paste"), False, 0, 0, 0):
                        self._clipboard_paste()
            
            ctx.end_popup()
    
    def _render_folder_tree(self, ctx: InfGUIContext, path: str, depth: int = 0):
        """Recursively render folder tree."""
        snapshot = self._get_dir_snapshot(path)
        if snapshot is None:
            return
        dirs = snapshot['dirs']

        for d in dirs:
            full_path = d['path']
            is_selected = self.__current_path == full_path

            # Check if folder has subfolders
            sub_snapshot = self._get_dir_snapshot(full_path)
            sub_dirs = sub_snapshot['dirs'] if sub_snapshot else []

            if sub_dirs:
                # Has subfolders - use tree node (arrow indicator shows expandability)
                # ImGuiTreeNodeFlags_OpenOnArrow = 32, click arrow to expand, click label to select
                node_label = f"{d['name']}##{full_path}"
                node_open = ctx.tree_node(node_label)
                # Check if item was clicked (not just arrow) - select this folder
                if ctx.is_item_clicked():
                    self.__current_path = full_path
                if node_open:
                    self._render_folder_tree(ctx, full_path, depth + 1)
                    ctx.tree_pop()
            else:
                # No subfolders - use selectable (leaf node)
                if ctx.selectable(f"{d['name']}##{full_path}", is_selected):
                    self.__current_path = full_path
    
    def _get_file_type(self, filename: str) -> str:
        return project_utils.get_file_type(filename)
    
    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (800, 250)

    def _pre_render(self, ctx):
        self._reset_frame_counters()
        self._ensure_type_icons_loaded()

    def on_render_content(self, ctx: InfGUIContext):
            # Top toolbar with cached breadcrumb
            cur = self.__current_path
            if cur != self.__breadcrumb_path:
                self.__breadcrumb_path = cur
                rel = os.path.relpath(cur, self.__root_path) if self.__root_path else cur
                if rel == '.':
                    rel = os.path.basename(self.__root_path) if self.__root_path else 'Project'
                self.__breadcrumb_text = f"Path: {rel}"
            ctx.label(self.__breadcrumb_text)
            ctx.separator()
            
            # Left panel: Folder tree showing entire project (about 25% width)
            tree_width = 200
            if ctx.begin_child("FolderTree", tree_width, 0, False):
                # Show entire project root in tree
                if self.__root_path and os.path.exists(self.__root_path):
                    project_name = os.path.basename(self.__root_path)
                    is_root_selected = self.__current_path == self.__root_path
                    # Root node - click to select, arrow to expand
                    ctx.set_next_item_open(True, Theme.COND_FIRST_USE_EVER)
                    node_open = ctx.tree_node(f"{project_name}##{self.__root_path}")
                    if ctx.is_item_clicked():
                        self.__current_path = self.__root_path
                    if node_open:
                        self._render_folder_tree(ctx, self.__root_path)
                        ctx.tree_pop()
                else:
                    ctx.label(t("project.no_project_path"))
            ctx.end_child()
            
            ctx.same_line()
            
            # Right panel: File grid/list (use border=True so WindowPadding applies, hide border color)
            ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, *Theme.PROJECT_PANEL_PAD)
            Theme.push_transparent_border(ctx)  # 1 colour
            if ctx.begin_child("FileGrid", 0, 0, True):
                # Right-click context menu for creating items
                self._render_context_menu(ctx)
                
                if self.__current_path and os.path.exists(self.__current_path):
                    snapshot = self._get_dir_snapshot(self.__current_path)
                    items = snapshot['items'] if snapshot else []

                    # Inject virtual sub-asset items for expanded model files
                    if self.__expanded_models:
                        _augmented = []
                        for _it in items:
                            _augmented.append(_it)
                            if (_it['type'] == 'file'
                                    and _it.get('ext', '') in self.MODEL_EXTENSIONS
                                    and _it['path'] in self.__expanded_models):
                                _sub = self._get_model_sub_assets(_it['path'])
                                if _sub:
                                    for _si, _sm in enumerate(_sub['submeshes']):
                                        _tri = _sm.get('index_count', 0) // 3
                                        _vtx = _sm.get('vertex_count', 0)
                                        _augmented.append({
                                            'type': 'sub_mesh',
                                            'name': (_sm.get('name', '') or f'SubMesh {_si}')
                                                   + f'  ({_tri} tris, {_vtx} verts)',
                                            'path': f"{_it['path']}##sub_mesh_{_si}",
                                            'parent_path': _it['path'],
                                            'ext': '', 'mtime_ns': None,
                                        })
                                    for _mi, _mn in enumerate(_sub['slot_names']):
                                        _augmented.append({
                                            'type': 'sub_material',
                                            'name': _mn or f'Material {_mi}',
                                            'path': f"{_it['path']}##sub_mat_{_mi}",
                                            'parent_path': _it['path'],
                                            'ext': '', 'mtime_ns': None,
                                            'slot_index': _mi,
                                        })
                        items = _augmented

                    # Back button
                    if self.__current_path != self.__root_path:
                        if ctx.selectable("[..]", False):
                            self.__current_path = os.path.dirname(self.__current_path)

                    # Grid config
                    icon_size = 64
                    padding = 10
                    cell_width = icon_size + padding
                    avail_w = ctx.get_content_region_avail_width()
                    cols = int(avail_w / cell_width)
                    if cols < 1: cols = 1

                    if not items and self.__current_path == self.__root_path:
                        ctx.label(t("project.empty_folder"))
                        ctx.label(t("project.right_click_hint"))

                    # Handle F2 for rename, Delete, Ctrl+C/X/V
                    if self.__selected_file and not self.__renaming_path:
                        ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)
                        if ctx.is_key_pressed(KEY_F2):
                            self.__renaming_path = self.__selected_file
                            name = os.path.basename(self.__selected_file)
                            if os.path.isfile(self.__renaming_path):
                                name = os.path.splitext(name)[0]
                            self.__renaming_name = name
                            self.__rename_focus_requested = True
                        elif ctx.is_key_pressed(KEY_DELETE):
                            self._delete_item(self.__selected_file)
                        elif ctrl and ctx.is_key_pressed(KEY_C):
                            self._clipboard_copy(self.__selected_file)
                        elif ctrl and ctx.is_key_pressed(KEY_X):
                            self._clipboard_cut(self.__selected_file)
                        elif ctrl and ctx.is_key_pressed(KEY_V):
                            self._clipboard_paste()
                    elif not self.__selected_file and not self.__renaming_path:
                        ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)
                        if ctrl and ctx.is_key_pressed(KEY_V):
                            self._clipboard_paste()

                    # Handle external file drops from Windows Explorer
                    self._handle_external_file_drops()

                    if ctx.begin_table("FileGrid", cols, 0, 0.0):
                        self._visible_items = items  # Store for shift-range
                        _sel_set = set(self.__selected_files)
                        for item in items:
                            ctx.table_next_column()
                            ctx.begin_group()

                            item_path = item['path']
                            item_type = item['type']
                            item_name = item['name']
                            ext = item.get('ext', '')
                            is_selected = item_path in _sel_set

                            # Check if this is an image file that can have a thumbnail
                            thumbnail_id = 0
                            if item_type == 'file' and ext in self.IMAGE_EXTENSIONS:
                                thumbnail_id = self._get_thumbnail(
                                    item_path, icon_size, item.get('mtime_ns'))

                            # Determine which texture to show for the cell
                            if item_type == 'sub_mesh':
                                display_tex_id = self.__type_icon_cache.get('model_3d', 0)
                            elif item_type == 'sub_material':
                                display_tex_id = self.__type_icon_cache.get('material', 0)
                            else:
                                display_tex_id = thumbnail_id
                                if display_tex_id == 0:
                                    display_tex_id = self._get_type_icon_id(item_type, ext)

                            if display_tex_id != 0:
                                # Render clickable icon image
                                # Zero out FramePadding so the image is centered in the cell
                                ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.ICON_BTN_NO_PAD)
                                # Remove border on unselected items, highlight selected
                                if is_selected:
                                    Theme.push_selected_icon_style(ctx)  # 2 colours
                                else:
                                    Theme.push_unselected_icon_style(ctx)  # 2 colours + 1 var
                                if ctx.image_button(f"##icon_{item_path}", display_tex_id, icon_size, icon_size):
                                    self._handle_item_click(item, ctx)
                                # Right-click on item → select it (so context menu acts on it)
                                if ctx.is_item_clicked(1):
                                    self.__selected_file = item_path
                                    if item_path not in self.__selected_files:
                                        self.__selected_files = [item_path]
                                    if self.__on_file_selected:
                                        self.__on_file_selected(self.__selected_file)
                                if is_selected:
                                    ctx.pop_style_color(2)
                                else:
                                    ctx.pop_style_color(2)
                                    ctx.pop_style_var(1)  # FrameBorderSize
                                ctx.pop_style_var(1)  # FramePadding
                            else:
                                # Ultimate fallback — text label (only when icon PNGs are missing)
                                label_icon = self._get_file_type(item_name) if item_type != 'dir' else '[DIR]'
                                if ctx.selectable(f"{label_icon}\n##{item_path}", is_selected, 0, icon_size, icon_size):
                                    self._handle_item_click(item, ctx)
                                # Right-click on item → select it
                                if ctx.is_item_clicked(1):
                                    self.__selected_file = item_path
                                    if item_path not in self.__selected_files:
                                        self.__selected_files = [item_path]
                                    if self.__on_file_selected:
                                        self.__on_file_selected(self.__selected_file)

                            # Drag-drop sources (map-driven)
                            if item_type == 'file':
                                _dd = self._DRAG_DROP_MAP.get(ext)
                                if _dd is not None:
                                    if ctx.begin_drag_drop_source(0):
                                        ctx.set_drag_drop_payload_str(_dd[0], item_path)
                                        ctx.label(f"{_dd[1]}: {item_name}")
                                        ctx.end_drag_drop_source()
                                else:
                                    _ddg = self._DRAG_DROP_GUID_MAP.get(ext)
                                    if _ddg is not None:
                                        if ctx.begin_drag_drop_source(0):
                                            guid = ""
                                            if self.__asset_database:
                                                try:
                                                    guid = self.__asset_database.get_guid_from_path(item_path)
                                                except Exception:
                                                    guid = ""
                                            if guid:
                                                ctx.set_drag_drop_payload_str(_ddg[0], guid)
                                            else:
                                                ctx.set_drag_drop_payload_str(_ddg[1], item_path)
                                            ctx.label(f"{_ddg[2]}: {item_name}")
                                            ctx.end_drag_drop_source()

                            # Name or Rename Input
                            if self.__renaming_path == item_path:
                                if self.__rename_focus_requested:
                                    ctx.set_keyboard_focus_here()
                                    self.__rename_focus_requested = False

                                ctx.set_next_item_width(cell_width - 8)
                                new_name = ctx.text_input(f"##rename_{item_path}", self.__renaming_name, 256)
                                self.__renaming_name = new_name

                                if ctx.is_key_pressed(KEY_ENTER):
                                    self._do_rename()
                                elif ctx.is_key_pressed(KEY_ESCAPE):
                                    self.__renaming_path = None
                                elif ctx.is_item_deactivated():
                                    self._do_rename()

                            else:
                                name_display = item_name
                                if item_type == 'file':
                                    name_display = os.path.splitext(item_name)[0]
                                    # Show expand arrow for model files
                                    if ext in self.MODEL_EXTENSIONS:
                                        _is_exp = item_path in self.__expanded_models
                                        name_display += '  \u25BC' if _is_exp else '  \u25B6'
                                elif item_type in ('sub_mesh', 'sub_material'):
                                    name_display = f'\u21B3 {item_name}'
                                ctx.label(name_display)

                            ctx.end_group()
                        ctx.end_table()

                else:
                    ctx.label(t("project.invalid_path"))

                # Drop target: accept hierarchy GameObjects to create prefab files
                remaining_h = ctx.get_content_region_avail_height()
                if remaining_h > 10:
                    ctx.invisible_button("##drop_prefab_area",
                                         ctx.get_content_region_avail_width(), remaining_h)
                    from .igui import IGUI
                    IGUI.drop_target(ctx, "HIERARCHY_GAMEOBJECT",
                                     self._create_prefab_from_hierarchy)

            ctx.end_child()
            ctx.pop_style_color(1)   # Border (from push_transparent_border)
            ctx.pop_style_var(1)     # WindowPadding
