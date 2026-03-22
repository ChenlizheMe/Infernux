"""
Unity-style Inspector panel with properties and raw data preview.

Component rendering helpers live in ``inspector_components``, and shared
layout helpers in ``inspector_utils``.

All asset inspectors (texture, audio, shader, material) are driven by the
unified ``asset_inspector`` module.  Material body rendering is delegated
to ``inspector_material``.
"""

import json
import os
from enum import Enum, auto
from InfEngine.lib import InfGUIContext, TextureLoader
from InfEngine.engine.i18n import t
from InfEngine.components.component import InfComponent
from InfEngine.resources import component_icons_dir
from InfEngine.core.asset_types import asset_category_from_extension
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from .inspector_utils import (
    max_label_w, field_label, render_compact_section_header, render_info_text,
    render_inspector_checkbox,
    render_material_property, render_component_header,
    DRAG_SPEED_DEFAULT, DRAG_SPEED_FINE,
)
from . import inspector_components as comp_ui
from . import inspector_shader_utils as shader_utils
from .inspector_components import _notify_scene_modified, _record_property, _record_add_component
from .asset_inspector import render_asset_inspector, invalidate as invalidate_asset_inspector
from .object_execution_layer import ObjectExecutionLayer
from .igui import IGUI


class InspectorMode(Enum):
    """Inspector display mode — mutually exclusive."""
    OBJECT = auto()    # GameObject selected from Hierarchy
    ASSET = auto()     # Asset file selected from Project panel
    PREVIEW = auto()   # Non-editable file preview


@editor_panel("Inspector", type_id="inspector", title_key="panel.inspector")
class InspectorPanel(EditorPanel):
    """
    Unity-style Inspector panel with two modules:
    1. Properties module (top) - displays object properties (Transform, etc.)
    2. Raw Data module (bottom) - displays file content preview using backend ResourcePreviewManager
    
    The backend ResourcePreviewManager handles all file type detection and preview rendering.
    When new previewers (e.g., Material, Model) are added to the backend, they automatically
    work here without any frontend changes.
    
    A splitter bar controls the ratio between the two modules.
    """
    
    WINDOW_TYPE_ID = "inspector"
    WINDOW_DISPLAY_NAME = "Inspector"
    
    # Minimum heights for splitter
    MIN_PROPERTIES_HEIGHT = Theme.INSPECTOR_MIN_PROPS_H
    MIN_RAW_DATA_HEIGHT = Theme.INSPECTOR_MIN_RAWDATA_H
    SPLITTER_HEIGHT = Theme.INSPECTOR_SPLITTER_H
    
    def __init__(self, title: str = "Inspector", engine=None):
        super().__init__(title, window_id="inspector")
        self.__engine = None
        self.__preview_manager = None  # ResourcePreviewManager from backend
        self.__asset_database = None
        self.__selected_object = None
        self.__selected_object_id = 0
        self.__selected_file = None
        self.__current_loaded_file = None
        self.__right_click_remove_enabled = True
        # Ratio of properties height to total available height (properties on top)
        self.__properties_ratio = Theme.INSPECTOR_DEFAULT_RATIO

        # Inspector mode state
        self.__inspector_mode = InspectorMode.OBJECT
        self.__asset_category: str = ""  # "material" | "texture" | "shader" | ""
        
        self.__add_component_search = ""  # Search text for Add Component popup
        self.__add_component_scripts = []  # Cached list of (display_name, path)
        self.__add_component_native_types = []  # Cached list of native type names
        self.__object_exec = ObjectExecutionLayer()

        # Component icon cache
        self.__comp_icon_cache: dict[str, int] = {}  # type_name_lower -> imgui tex id
        self.__comp_icons_loaded = False

        # Inline material cache (for MeshRenderer inline rendering)
        self.__inline_mat_ref = None           # strong ref — prevent pybind11 wrapper GC
        self.__inline_mat_id: int = 0          # id() of the native material currently cached
        self.__inline_mat_data: dict = {}      # deserialized JSON
        self._inline_mat_version: int = -1     # C++ version counter for cheap cache invalidation
        self.__inline_shader_cache: dict = {".vert": None, ".frag": None}
        self.__inline_mat_frag_shader_id: str = ""
        self._inline_mat_sync_key: str = ""

        # Unified undo tracking
        from InfEngine.engine.undo import InspectorUndoTracker
        self._undo_tracker = InspectorUndoTracker()

        # Multi-edit component cache (invalidated on selection change)
        self._multi_cache_ids: tuple = ()
        self._multi_cache_cpp: set = set()
        self._multi_cache_py: set = set()
        self._multi_cache_per_cpp: dict = {}
        self._multi_cache_per_py: dict = {}
        self._multi_cache_has_transform: bool = True

        # Idle-skip state: tracks how many consecutive frames the
        # inspector child was neither hovered nor had an active item.
        self._inspector_idle_frames: int = 0

        # Register MeshRenderer into the component renderer registry so
        # dispatch is fully unified (uses bound method for panel-level state).
        comp_ui.register_component_renderer("MeshRenderer", self._render_mesh_renderer)

        # Initialize engine if provided
        if engine:
            self.set_engine(engine)
    
    def set_engine(self, engine):
        """Set the engine instance for resource preview."""
        self.__engine = engine
        if engine:
            # Check if it's a Python Engine wrapper or native InfEngine
            if hasattr(engine, 'get_resource_preview_manager'):
                self.__preview_manager = engine.get_resource_preview_manager()
                if hasattr(engine, 'get_asset_database'):
                    self.__asset_database = engine.get_asset_database()
            elif hasattr(engine, 'get_native_engine'):
                # It's the Python Engine class, get native and then preview manager
                native = engine.get_native_engine()
                if native and hasattr(native, 'get_resource_preview_manager'):
                    self.__preview_manager = native.get_resource_preview_manager()
                if native and hasattr(native, 'get_asset_database'):
                    self.__asset_database = native.get_asset_database()
    
    # ---- Component icon helpers ----

    def _load_component_icons(self, native):
        """Lazily load component icon PNGs from resources/icons/components/."""
        if self.__comp_icons_loaded:
            return
        if not os.path.isdir(component_icons_dir):
            self.__comp_icons_loaded = True
            return
        for fname in os.listdir(component_icons_dir):
            if not fname.startswith("component_") or not fname.endswith(".png"):
                continue
            key = fname[len("component_"):-len(".png")]  # e.g. "camera"
            tex_name = f"__compicon__{key}"
            if native.has_imgui_texture(tex_name):
                self.__comp_icon_cache[key] = native.get_imgui_texture_id(tex_name)
                continue
            icon_path = os.path.join(component_icons_dir, fname)
            tex_data = TextureLoader.load_from_file(icon_path)
            if tex_data and tex_data.is_valid():
                pixels = tex_data.get_pixels_list()
                tid = native.upload_texture_for_imgui(
                    tex_name, pixels, tex_data.width, tex_data.height)
                if tid != 0:
                    self.__comp_icon_cache[key] = tid
        self.__comp_icons_loaded = True

    def _load_custom_icon(self, icon_path: str, type_name: str) -> int:
        """Load a custom icon specified by the ``@icon`` decorator and return
        its ImGui texture id, or 0 on failure."""
        key = type_name.lower()
        if key in self.__comp_icon_cache:
            return self.__comp_icon_cache[key]
        native = self._get_native_engine()
        if not native:
            return 0
        # Resolve project-relative paths
        if not os.path.isabs(icon_path):
            from InfEngine.engine.project_context import get_project_root
            root = get_project_root()
            if root:
                icon_path = os.path.join(root, icon_path)
        if not os.path.isfile(icon_path):
            self.__comp_icon_cache[key] = 0  # cache miss
            return 0
        tex_name = f"__compicon__{key}"
        if native.has_imgui_texture(tex_name):
            tid = native.get_imgui_texture_id(tex_name)
            self.__comp_icon_cache[key] = tid
            return tid
        tex_data = TextureLoader.load_from_file(icon_path)
        if tex_data and tex_data.is_valid():
            pixels = tex_data.get_pixels_list()
            tid = native.upload_texture_for_imgui(
                tex_name, pixels, tex_data.width, tex_data.height)
            if tid != 0:
                self.__comp_icon_cache[key] = tid
                return tid
        self.__comp_icon_cache[key] = 0
        return 0

    def _get_component_icon_id(self, type_name: str, is_script: bool = False) -> int:
        """Return ImGui texture id for a component icon, or 0 if unavailable.

        For script components, falls back to the generic ``component_script.png``
        icon when no component-specific icon is found.
        """
        tid = self.__comp_icon_cache.get(type_name.lower(), 0)
        if tid == 0 and is_script:
            tid = self.__comp_icon_cache.get("script", 0)
        return tid

    def _render_component_header_icon(self, ctx, type_name: str,
                                       is_script: bool = False, py_comp=None):
        """Draw the component icon inline before a collapsing header.

        Call this *before* ``collapsing_header`` — it renders a 16×16 image
        and uses ``same_line`` so the header appears right after the icon.

        For Python script components, the ``@icon("path")`` decorator is
        checked first.  If no decorator icon is found, falls back to the
        bundled ``component_<name>.png``, then to ``component_script.png``.
        """
        icon_id = 0
        # 1) Check @icon decorator on the Python component class
        if py_comp is not None:
            custom_path = getattr(py_comp.__class__, '_component_icon_', None)
            if custom_path:
                icon_id = self._load_custom_icon(custom_path, type_name)
        # 2) Fall back to bundled icon cache
        if icon_id == 0:
            icon_id = self._get_component_icon_id(type_name, is_script)
        if icon_id == 0:
            return
        ctx.image(icon_id, Theme.COMPONENT_ICON_SIZE, Theme.COMPONENT_ICON_SIZE)
        ctx.same_line()

    def set_selected_object(self, obj):
        """Set the selected scene object for properties display."""
        old_object_id = self.__selected_object_id
        self.__selected_object = obj
        self.__selected_object_id = obj.id if obj is not None else 0
        if self.__selected_object_id != old_object_id:
            self._reset_inline_material_cache()
        # Clear file selection when object is selected
        if obj is not None:
            self.__selected_file = None
            self.__current_loaded_file = None
            self.__inspector_mode = InspectorMode.OBJECT
            self.__asset_category = ""
        else:
            # Selection cleared — invalidate inline material cache so we
            # don't hold a strong pybind11 ref to a potentially destroyed material.
            self._reset_inline_material_cache()

    def _get_selected_object(self):
        """Resolve selected object by ID to avoid stale pointers after scene reload."""
        return self.__object_exec.resolve_selected_object(self.__selected_object_id)

    def _get_all_selected_objects(self):
        """Resolve all selected objects from SelectionManager."""
        from .selection_manager import SelectionManager
        sel = SelectionManager.instance()
        ids = sel.get_ids()
        if not ids:
            return []
        from InfEngine.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return []
        result = []
        for oid in ids:
            obj = scene.find_by_id(oid)
            if obj is not None:
                result.append(obj)
        return result
    
    def set_selected_file(self, file_path: str):
        """Set the selected file for raw data display."""
        if file_path != self.__selected_file:
            self.__selected_file = file_path
            self.__current_loaded_file = None
            self._reset_inline_material_cache()
            # Invalidate unified asset inspector state on file change
            invalidate_asset_inspector()
        # Determine inspector mode & asset category
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            cat = asset_category_from_extension(ext)
            if cat:
                self.__inspector_mode = InspectorMode.ASSET
                self.__asset_category = cat
            else:
                self.__inspector_mode = InspectorMode.PREVIEW
                self.__asset_category = ""
            # Clear object selection when file is selected
            self.__selected_object = None
            self.__selected_object_id = 0
        else:
            self.__inspector_mode = InspectorMode.OBJECT
            self.__asset_category = ""
    
    def set_detail_file(self, file_path: str):
        """Open an asset file in the detail (raw-data) module while keeping
        the current object selection, triggering a split view.

        Unlike ``set_selected_file`` (used by the Project panel), this does
        **not** clear the hierarchy selection — the properties module keeps
        showing the current object while the bottom half shows the asset
        editor.
        """
        if file_path != self.__selected_file:
            self.__selected_file = file_path
            self.__current_loaded_file = None
            self._reset_inline_material_cache()
            invalidate_asset_inspector()
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            cat = asset_category_from_extension(ext)
            if cat:
                self.__inspector_mode = InspectorMode.ASSET
                self.__asset_category = cat
            else:
                self.__inspector_mode = InspectorMode.PREVIEW
                self.__asset_category = ""
        else:
            self.__inspector_mode = InspectorMode.OBJECT
            self.__asset_category = ""

    def _get_file_extension(self, file_path: str) -> str:
        """Get lowercase file extension with dot."""
        if not file_path:
            return ""
        _, ext = os.path.splitext(file_path)
        return ext.lower()
    
    def _can_preview_file(self, file_path: str) -> bool:
        """Check if the backend has a previewer for this file type."""
        if not self.__preview_manager or not file_path:
            return False
        ext = self._get_file_extension(file_path)
        return self.__preview_manager.has_previewer(ext)

    def _reset_inline_material_cache(self):
        """Clear inline MeshRenderer material cache and shader sync state."""
        self.__inline_mat_ref = None
        self.__inline_mat_id = 0
        self.__inline_mat_data = {}
        self._inline_mat_version = -1
        self.__inline_shader_cache = {".vert": None, ".frag": None}
        self.__inline_mat_frag_shader_id = ""
        self._inline_mat_sync_key = ""
    
    def _load_preview(self, file_path: str) -> bool:
        """Load file for preview using backend ResourcePreviewManager."""
        if not self.__preview_manager or not file_path:
            return False
        
        # Skip if already loaded
        if self.__current_loaded_file == file_path and self.__preview_manager.is_preview_loaded():
            return True
        
        # Load the file
        if self.__preview_manager.load_preview(file_path):
            self.__current_loaded_file = file_path
            return True
        return False
    
    def _render_raw_data_module(self, ctx: InfGUIContext, height: float):
        """Render the Raw Data module - shows asset editor, file preview, or material details."""
        child_visible = ctx.begin_child("RawDataModule", 0, height, True)
        if child_visible:
            # Asset mode — route to dedicated asset inspectors
            if self.__selected_file and self.__inspector_mode == InspectorMode.ASSET:
                self._render_asset_inspector(ctx)
            # Priority 3: Preview mode — generic file preview
            elif self.__selected_file:
                self._render_file_preview(ctx)
            else:
                ctx.label(t("inspector.no_selection"))
                ctx.label(t("inspector.select_file_hint"))
        ctx.end_child()

    def _render_asset_inspector(self, ctx: InfGUIContext):
        """Delegate to the unified asset inspector."""
        fp = self.__selected_file
        cat = self.__asset_category
        if cat:
            render_asset_inspector(ctx, self, fp, cat)
        else:
            self._render_file_preview(ctx)
    
    def _render_file_preview(self, ctx: InfGUIContext):
        """Render file preview using backend ResourcePreviewManager."""
        if os.path.isdir(self.__selected_file):
            folder_name = os.path.basename(self.__selected_file)
            ctx.label(t("inspector.folder_label").format(name=folder_name))
            ctx.separator()
            ctx.label(t("inspector.path_label").format(path=self.__selected_file))
        elif not self.__preview_manager:
            filename = os.path.basename(self.__selected_file)
            ctx.label(t("inspector.file_label").format(name=filename))
            ctx.separator()
            ctx.label(t("inspector.preview_not_init"))
        elif not self._can_preview_file(self.__selected_file):
            filename = os.path.basename(self.__selected_file)
            ctx.label(t("inspector.file_label").format(name=filename))
            ctx.separator()
            ctx.label(t("inspector.no_previewer"))
            ext = self._get_file_extension(self.__selected_file)
            ctx.label(t("inspector.extension_label").format(ext=ext))
        elif not self._load_preview(self.__selected_file):
            filename = os.path.basename(self.__selected_file)
            ctx.label(t("inspector.file_label").format(name=filename))
            ctx.separator()
            ctx.label(t("inspector.preview_failed"))
        else:
            # Render metadata from backend
            self.__preview_manager.render_metadata(ctx)
            ctx.separator()
            
            # Get remaining space for preview content
            avail_width = ctx.get_content_region_avail_width()
            avail_height = ctx.get_content_region_avail_height()
            
            # Render the actual preview (image, text, model, etc.)
            if avail_width > 0 and avail_height > 0:
                self.__preview_manager.render_preview(ctx, avail_width, avail_height)
    
    def _get_native_engine(self):
        """Get the native C++ InfEngine instance."""
        if self.__engine:
            if hasattr(self.__engine, 'get_native_engine'):
                return self.__engine.get_native_engine()
            elif hasattr(self.__engine, 'refresh_material_pipeline'):
                return self.__engine
        return None

    @staticmethod
    def _ensure_material_file_path(material) -> str:
        """Ensure *material* has a ``file_path``; assign a default one if needed.

        Returns the resolved file path, or ``""`` on failure.
        """
        if getattr(material, 'file_path', ''):
            return material.file_path
        # Try to resolve via GUID → path from AssetDatabase
        guid = getattr(material, 'guid', '') or ''
        if guid:
            try:
                from InfEngine.lib import AssetRegistry
                adb = AssetRegistry.instance().get_asset_database()
                if adb:
                    resolved = adb.get_path_from_guid(guid)
                    if resolved:
                        material.file_path = resolved
                        return resolved
            except (ImportError, RuntimeError, AttributeError):
                pass
        from InfEngine.engine.project_context import get_project_root
        project_root = get_project_root()
        if not project_root:
            return ""
        materials_dir = os.path.join(project_root, "materials")
        os.makedirs(materials_dir, exist_ok=True)
        mat_name = getattr(material, 'name', 'DefaultUnlit')
        if mat_name == "DefaultLit":
            mat_file = os.path.join(materials_dir, "default_lit.mat")
        elif mat_name == "DefaultUnlit":
            mat_file = os.path.join(materials_dir, "default_unlit.mat")
        else:
            import re as _re
            file_name = _re.sub(r'([A-Z])', r'_\1', mat_name).lower().strip('_') + ".mat"
            mat_file = os.path.join(materials_dir, file_name)
        material.file_path = mat_file
        return mat_file

    # ------------------------------------------------------------------
    # Layout helpers — delegates to inspector_utils
    # ------------------------------------------------------------------
    _max_label_w = staticmethod(max_label_w)
    _field_label = staticmethod(field_label)

    # ------------------------------------------------------------------
    # Tag & Layer rendering
    # ------------------------------------------------------------------
    def _render_tag_layer_row(self, ctx: InfGUIContext, obj):
        """Render tag and layer dropdowns for a GameObject, on a single row."""
        from InfEngine.lib import TagLayerManager
        mgr = TagLayerManager.instance()

        all_tags = list(mgr.get_all_tags())
        current_tag = obj.tag if hasattr(obj, 'tag') else "Untagged"
        tag_idx = all_tags.index(current_tag) if current_tag in all_tags else 0

        all_layers = list(mgr.get_all_layers())
        current_layer = obj.layer if hasattr(obj, 'layer') else 0
        # Build display labels: "0: Default", "3: (empty)", etc.
        layer_labels = []
        for i, name in enumerate(all_layers):
            if name:
                layer_labels.append(f"{i}: {name}")
            else:
                layer_labels.append(f"{i}: ---")

        # Append "Add Tag..." / "Add Layer..." at the end of each combo
        tag_items = all_tags + ["Add Tag..."]
        layer_items = layer_labels + ["Add Layer..."]

        # Layout: Tag [combo▼]   Layer [combo▼]
        # Each label is rendered, then the combo is placed right after it with a small gap.
        # A fixed mid-point keeps the two columns aligned.
        avail_w = ctx.get_content_region_avail_width()
        half_w = avail_w * 0.5 - 4

        # --- Tag (left column) ---
        ctx.label(t("inspector.tag"))
        ctx.same_line(0, 4)
        new_tag_idx = IGUI.searchable_combo(ctx, "Tag", tag_idx, tag_items, width=half_w - 30)
        if new_tag_idx != tag_idx:
            if new_tag_idx == len(all_tags):
                # "Add Tag..." selected — open Tag & Layer settings panel
                if self._window_manager:
                    self._window_manager.open_window("tag_layer_settings")
            elif 0 <= new_tag_idx < len(all_tags):
                _record_property(obj, "tag", all_tags[tag_idx], all_tags[new_tag_idx], "Set Tag")

        # --- Layer (right column) — start at fixed half-width mark ---
        ctx.same_line(half_w + 8)
        ctx.label(t("inspector.layer"))
        ctx.same_line(0, 4)
        layer_combo_w = ctx.get_content_region_avail_width()
        new_layer = IGUI.searchable_combo(ctx, "Layer", current_layer, layer_items, width=layer_combo_w)
        if new_layer != current_layer:
            if new_layer == len(layer_labels):
                # "Add Layer..." selected — open Tag & Layer settings panel
                if self._window_manager:
                    self._window_manager.open_window("tag_layer_settings")
            else:
                _record_property(obj, "layer", current_layer, new_layer, "Set Layer")

    def _render_prefab_header(self, ctx: InfGUIContext, obj):
        """Render the prefab instance header bar with Select/Open/Apply/Revert buttons."""
        ctx.dummy(0, 4)

        # Blue-tinted background bar
        ctx.push_style_color(ImGuiCol.ChildBg, 0.10, 0.16, 0.24, 1.0)
        ctx.begin_child("##prefab_header_bar", 0, 28, True)

        ctx.push_style_color(ImGuiCol.Text, *Theme.PREFAB_TEXT)
        ctx.label(t("inspector.prefab_label"))
        ctx.pop_style_color(1)

        ctx.same_line(0, 8)
        if ctx.small_button(t("inspector.prefab_select") + "##prefab_select"):
            self._prefab_select_asset(obj)

        ctx.same_line(0, 4)
        if ctx.small_button(t("inspector.prefab_open") + "##prefab_open"):
            self._prefab_open_asset(obj)

        ctx.same_line(0, 12)

        # Override count badge
        overrides = self._get_prefab_overrides(obj)
        override_count = len(overrides) if overrides else 0

        if override_count > 0:
            ctx.push_style_color(ImGuiCol.Text, *Theme.WARNING_TEXT)
            ctx.label(f"{override_count} " + t("inspector.overrides"))
            ctx.pop_style_color(1)
            ctx.same_line(0, 8)
            if ctx.small_button(t("inspector.prefab_apply") + "##prefab_apply"):
                self._prefab_apply(obj)
            ctx.same_line(0, 4)
            if ctx.small_button(t("inspector.prefab_revert") + "##prefab_revert"):
                self._prefab_revert(obj)
        else:
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
            ctx.label(t("inspector.no_overrides"))
            ctx.pop_style_color(1)

        ctx.end_child()
        ctx.pop_style_color(1)

    def _get_prefab_overrides(self, obj):
        """Get override list for a prefab instance (cached per frame-ish)."""
        guid = getattr(obj, 'prefab_guid', '')
        if not guid:
            return []
        prefab_path = self._resolve_prefab_path(guid)
        if not prefab_path:
            return []
        try:
            from InfEngine.engine.prefab_overrides import compute_overrides
            return compute_overrides(obj, prefab_path)
        except Exception:
            return []

    def _resolve_prefab_path(self, guid: str):
        """Resolve a prefab GUID to a file path via AssetDatabase."""
        if not guid:
            return None
        try:
            from InfEngine.lib import AssetRegistry
            registry = AssetRegistry.instance()
            if registry:
                adb = registry.get_asset_database()
                if adb:
                    return adb.get_path_from_guid(guid)
        except Exception:
            pass
        return None

    def _prefab_select_asset(self, obj):
        """Select the .prefab asset in the Project panel."""
        guid = getattr(obj, 'prefab_guid', '')
        path = self._resolve_prefab_path(guid)
        if path:
            from InfEngine.engine.ui.editor_event_bus import EditorEventBus
            EditorEventBus.publish("select_asset", path)

    def _prefab_open_asset(self, obj):
        """Open the .prefab file in the asset inspector."""
        guid = getattr(obj, 'prefab_guid', '')
        path = self._resolve_prefab_path(guid)
        if path:
            from InfEngine.engine.ui.editor_event_bus import EditorEventBus
            EditorEventBus.publish("open_asset", path)

    def _prefab_apply(self, obj):
        """Apply all overrides back to the .prefab file."""
        guid = getattr(obj, 'prefab_guid', '')
        path = self._resolve_prefab_path(guid)
        if path:
            from InfEngine.engine.prefab_overrides import apply_overrides_to_prefab
            apply_overrides_to_prefab(obj, path)

    def _prefab_revert(self, obj):
        """Revert the instance to match the source .prefab file."""
        guid = getattr(obj, 'prefab_guid', '')
        path = self._resolve_prefab_path(guid)
        if path:
            from InfEngine.engine.prefab_overrides import revert_overrides
            revert_overrides(obj, path)

    # ------------------------------------------------------------------
    # Component rendering — delegates to inspector_components
    # ------------------------------------------------------------------
    def _render_transform_component(self, ctx: InfGUIContext, trans):
        comp_ui.render_transform_component(ctx, trans)

    def _render_mesh_renderer(self, ctx: InfGUIContext, renderer):
        """Render MeshRenderer component fields.

        Material and mesh fields need custom rendering (not CppProperty).
        CppProperty fields (shadows, etc.) are rendered by the unified
        ``render_py_component`` path automatically.
        """
        from InfEngine.components.builtin_component import BuiltinComponent

        # Ensure we have the Python wrapper
        if not isinstance(renderer, BuiltinComponent):
            wrapper_cls = BuiltinComponent._builtin_registry.get("MeshRenderer")
            go = getattr(renderer, 'game_object', None)
            if wrapper_cls and go is not None:
                renderer = wrapper_cls._get_or_create_wrapper(renderer, go)

        # Fetch mesh asset metadata once for slot names / submesh info
        _slot_names = []
        _submesh_infos = []
        if hasattr(renderer, 'get_material_slot_names'):
            try:
                _slot_names = renderer.get_material_slot_names()
            except Exception:
                _slot_names = []
        if hasattr(renderer, 'get_submesh_infos'):
            try:
                _submesh_infos = renderer.get_submesh_infos()
            except Exception:
                _submesh_infos = []

        lw = self._max_label_w(ctx, [t("inspector.mesh"), t("inspector.materials"), "Element 0"])

        # ── Mesh field ─────────────────────────────────────────────────
        self._field_label(ctx, t("inspector.mesh"), lw)
        if renderer.has_inline_mesh():
            inline_name = getattr(renderer, 'inline_mesh_name', '') or ''
            mesh_display = inline_name if inline_name else "(Primitive)"
        elif getattr(renderer, 'has_mesh_asset', False):
            mesh_guid = getattr(renderer, 'mesh_asset_guid', '') or ''
            mesh_display = getattr(renderer, 'mesh_name', '') or mesh_guid[:8] or 'Mesh'
        else:
            mesh_display = "None"
        self._render_object_field(ctx, "mesh_field", mesh_display, "Mesh", clickable=False)

        # ── Submesh summary (triangle counts per submesh) ──────────────
        if _submesh_infos:
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
            for si, info in enumerate(_submesh_infos):
                sub_name = info.get('name', '') or f"SubMesh {si}"
                ctx.label(f"  {sub_name}")
            ctx.pop_style_color(1)

        ctx.separator()

        # ── Material slots (multi-material) ────────────────────────────
        mat_count = getattr(renderer, 'material_count', 0) or 1
        self._field_label(ctx, "Materials", lw)
        ctx.label(f"Size: {mat_count}")

        from .inspector_components import _picker_assets

        for slot_idx in range(mat_count):
            mat = renderer.get_effective_material(slot_idx)
            mat_name = getattr(mat, 'name', 'None') if mat else 'None'

            # Check if slot has an explicitly assigned material
            guids = renderer.get_material_guids() if hasattr(renderer, 'get_material_guids') else []
            is_default = (slot_idx >= len(guids)) or (not guids[slot_idx])
            display_name = f"{mat_name}" + (" (Default)" if is_default else "")

            mat_file = getattr(mat, 'file_path', '') if mat else ''
            is_selected = bool(mat_file) and self.__selected_file == mat_file

            _slot = slot_idx  # capture for closures

            def _make_on_drop(s):
                def on_material_drop(mat_path):
                    self._apply_dropped_material(renderer, mat_path, slot=s)
                return on_material_drop

            def _make_on_pick(s):
                def _on_mat_pick(picked_path):
                    self._apply_dropped_material(renderer, str(picked_path), slot=s)
                return _on_mat_pick

            def _make_on_clear(s):
                def _on_mat_clear():
                    old_mat = renderer.get_material(s) if hasattr(renderer, 'get_material') else renderer.render_material
                    renderer.set_material(s, "")
                    _record_property(renderer, f"material_slot_{s}", old_mat, None, f"Clear Material Slot {s}")
                return _on_mat_clear

            # Use model-file slot name if available, otherwise generic label
            if slot_idx < len(_slot_names) and _slot_names[slot_idx]:
                slot_label = f"{_slot_names[slot_idx]} (Slot {slot_idx})"
            else:
                slot_label = f"Element {slot_idx}"

            self._field_label(ctx, slot_label, lw)
            self._render_object_field(ctx, f"mat_{slot_idx}", display_name, "Material",
                                      selected=is_selected, clickable=False,
                                      accept_drag_type="MATERIAL_FILE",
                                      on_drop_callback=_make_on_drop(_slot),
                                      picker_asset_items=lambda filt: _picker_assets(filt, "*.mat"),
                                      on_pick=_make_on_pick(_slot),
                                      on_clear=_make_on_clear(_slot))

        ctx.separator()

        # ── CppProperty fields (shadows, etc.) via unified path ────────
        comp_ui.render_py_component(ctx, renderer)

    def _render_object_material_sections(self, ctx: InfGUIContext, renderers):
        """Render effective MeshRenderer materials after all components."""
        valid_entries = []
        for renderer in renderers:
            try:
                mat = renderer.get_effective_material()
            except RuntimeError:
                mat = None
            if mat is None:
                continue
            valid_entries.append((renderer, mat))

        if not valid_entries:
            return

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP * 1.5)
        ctx.separator()
        ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT)
        ctx.label(t("inspector.material_overrides"))
        ctx.pop_style_color(1)
        ctx.separator()
        ctx.set_next_item_open(True, Theme.COND_FIRST_USE_EVER)
        if not render_compact_section_header(ctx, "Materials##object_materials", level="primary"):
            return

        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)
        multiple = len(valid_entries) > 1
        for index, (renderer, mat) in enumerate(valid_entries):
            if multiple:
                title = f"Element {index}"
                if not render_compact_section_header(ctx, f"{title}##material_element_{index}", level="secondary"):
                    continue

            is_default = False
            try:
                is_default = not renderer.has_render_material()
            except RuntimeError:
                pass

            if is_default:
                render_info_text(ctx, "Using the renderer's effective default material")
            self._render_inline_material(ctx, mat)

            if index != len(valid_entries) - 1:
                ctx.separator()
        ctx.pop_style_var(2)

    def _render_inline_material(self, ctx: InfGUIContext, native_mat):
        """Render material properties inline inside the object inspector.

        Used by the bottom-of-inspector material section for MeshRenderer
        objects, after all components have been rendered.
        """
        # ── Cache management — re-serialize only when the material changes ──
        mat_id = id(native_mat)
        # Use lightweight version counter to detect changes (avoids
        # full Serialize() + string compare every frame).
        try:
            mat_version = native_mat.get_version()
        except (AttributeError, RuntimeError):
            mat_version = -1  # fallback: always refresh

        cache_hit = (mat_id == self.__inline_mat_id
                     and mat_version == getattr(self, '_inline_mat_version', -1)
                     and mat_version != -1)

        if not cache_hit:
            try:
                current_json = native_mat.serialize()
            except RuntimeError:
                current_json = ""
            self.__inline_mat_ref = native_mat  # prevent GC → stabilize id()
            try:
                fresh = json.loads(current_json) if current_json else {}
            except (ValueError, json.JSONDecodeError):
                fresh = {}
            # Preserve Python-only metadata (_shader_property_order, hdr, …)
            # that the C++ serialiser does not round-trip.
            old_data = self.__inline_mat_data
            if isinstance(old_data, dict) and fresh:
                # Keep _shader_property_order from previous cache
                if "_shader_property_order" in old_data and "_shader_property_order" not in fresh:
                    fresh["_shader_property_order"] = old_data["_shader_property_order"]
                # Merge per-property metadata (hdr, etc.)
                old_props = old_data.get("properties") if isinstance(old_data.get("properties"), dict) else {}
                new_props = fresh.get("properties") if isinstance(fresh.get("properties"), dict) else {}
                for pname, fprop in new_props.items():
                    if isinstance(fprop, dict) and pname in old_props and isinstance(old_props[pname], dict):
                        for mk, mv in old_props[pname].items():
                            if mk not in ("value", "guid"):
                                fprop[mk] = mv
            self.__inline_mat_data = fresh
            if mat_id != self.__inline_mat_id:
                self.__inline_shader_cache = {".vert": None, ".frag": None}
                self.__inline_mat_frag_shader_id = ""
                self._inline_mat_sync_key = ""
            self.__inline_mat_id = mat_id
            self._inline_mat_version = mat_version

        mat_data = self.__inline_mat_data
        if not mat_data:
            return

        is_builtin = mat_data.get("builtin", False)
        changed = False
        requires_deserialize = False
        requires_pipeline_refresh = False

        # --- Register inline material for undo tracking ---
        _nmat = native_mat  # capture
        _mat_guid = getattr(native_mat, "guid", "") or str(mat_id)
        def _mat_snapshot(_m=_nmat):
            try:
                return _m.serialize()
            except RuntimeError:
                return ""
        def _mat_restore(s, _m=_nmat, _self=self):
            try:
                _m.deserialize(s)
                engine = _self._get_native_engine()
                if engine and hasattr(engine, 'refresh_material_pipeline'):
                    engine.refresh_material_pipeline(_m)
                if hasattr(_m, 'save'):
                    _m.save()
                _self._inline_mat_version = -1  # force cache refresh
            except (RuntimeError, ValueError):
                pass
        self._undo_tracker.track(
            f"material:{_mat_guid}",
            _mat_snapshot, _mat_restore, "Edit Material",
        )

        # Sync shader annotations
        frag_shader_id = mat_data.get("shaders", {}).get("fragment", "")
        if frag_shader_id and not mat_data.get("_shader_property_order"):
            self._inline_mat_sync_key = ""
        prop_gen = shader_utils.get_shader_property_generation()
        inline_sync_key = f"{frag_shader_id}:{prop_gen}"
        if inline_sync_key != getattr(self, '_inline_mat_sync_key', ''):
            old_id = getattr(self, '_inline_mat_sync_key', '').rsplit(":", 1)[0] if getattr(self, '_inline_mat_sync_key', '') else ''
            remove = (frag_shader_id == old_id) and bool(old_id)
            self._inline_mat_sync_key = inline_sync_key
            self.__inline_mat_frag_shader_id = frag_shader_id
            if frag_shader_id:
                shader_utils.sync_properties_from_shader(
                    mat_data, frag_shader_id, ".frag", remove_unknown=remove,
                )

        ctx.separator()

        # ── Shader ─────────────────────────────────────────────────────
        if is_builtin:
            ctx.begin_disabled(True)
        if render_compact_section_header(ctx, "Shader##inline_mat", level="secondary"):
            shaders = mat_data.setdefault("shaders", {})
            vert_path = shaders.get("vertex", "")
            frag_path = shaders.get("fragment", "")
            s_lw = max_label_w(ctx, ["Vertex", "Fragment"])
            from .inspector_components import _picker_assets

            # Vertex
            field_label(ctx, "Vertex", s_lw)
            vert_items = shader_utils.get_shader_candidates(".vert", self.__inline_shader_cache)
            vert_display = shader_utils.shader_display_from_value(vert_path, vert_items)

            def _on_ivert_pick(picked):
                nonlocal changed, requires_deserialize, requires_pipeline_refresh
                shaders["vertex"] = picked
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

            if self._render_object_field(
                ctx, "imat_vert", vert_display, "Vert",
                clickable=True,
                accept_drag_type="SHADER_FILE",
                on_drop_callback=lambda p: self._on_inline_shader_drop(p, ".vert", shaders),
                picker_asset_items=lambda filt: _picker_assets(filt, "*.vert"),
                on_pick=_on_ivert_pick,
            ):
                ctx.open_popup("imat_vert_popup")
            if ctx.begin_popup("imat_vert_popup"):
                for display, value in vert_items:
                    if ctx.selectable(display, value == vert_path):
                        shaders["vertex"] = value
                        changed = True
                        requires_deserialize = True
                        requires_pipeline_refresh = True
                ctx.end_popup()

            # Fragment
            field_label(ctx, "Fragment", s_lw)
            frag_items = shader_utils.get_shader_candidates(".frag", self.__inline_shader_cache)
            frag_display = shader_utils.shader_display_from_value(frag_path, frag_items)

            def _on_ifrag_pick(picked):
                nonlocal changed, requires_deserialize, requires_pipeline_refresh
                old_frag = shaders.get("fragment", "")
                shaders["fragment"] = picked
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True
                if picked != old_frag:
                    shader_utils.sync_properties_from_shader(
                        mat_data, picked, ".frag", remove_unknown=True,
                    )
                    self.__inline_mat_frag_shader_id = picked
                    self._inline_mat_sync_key = f"{picked}:{shader_utils.get_shader_property_generation()}"

            if self._render_object_field(
                ctx, "imat_frag", frag_display, "Frag",
                clickable=True,
                accept_drag_type="SHADER_FILE",
                on_drop_callback=lambda p: self._on_inline_shader_drop(p, ".frag", shaders),
                picker_asset_items=lambda filt: _picker_assets(filt, "*.frag"),
                on_pick=_on_ifrag_pick,
            ):
                ctx.open_popup("imat_frag_popup")
            if ctx.begin_popup("imat_frag_popup"):
                for display, value in frag_items:
                    if ctx.selectable(display, value == frag_path):
                        old_frag = shaders.get("fragment", "")
                        shaders["fragment"] = value
                        changed = True
                        requires_deserialize = True
                        requires_pipeline_refresh = True
                        if value != old_frag:
                            shader_utils.sync_properties_from_shader(
                                mat_data, value, ".frag", remove_unknown=True,
                            )
                            self.__inline_mat_frag_shader_id = value
                            self._inline_mat_sync_key = f"{value}:{shader_utils.get_shader_property_generation()}"
                ctx.end_popup()
        if is_builtin:
            ctx.end_disabled()

        ctx.separator()

        # ── Surface Options (Render Settings) ──────────────────────────
        if is_builtin:
            ctx.begin_disabled(True)
        if render_compact_section_header(ctx, "Surface Options##inline_mat", level="secondary"):
            rs = mat_data.setdefault("renderState", {})
            overrides = int(mat_data.get("renderStateOverrides", 0))

            so_labels = ["Surface Type", "Cull Mode", "Depth Write",
                         "Depth Test", "Blend Mode", "Alpha Clip",
                         "Render Queue"]
            so_lw = max_label_w(ctx, so_labels)

            # --- Surface Type ---
            surface_items = ["Opaque", "Transparent"]
            cur_surface = 1 if rs.get("blendEnable", False) else 0
            field_label(ctx, "Surface Type", so_lw)
            new_surface = ctx.combo("##imat_surface_type", cur_surface, surface_items)
            if new_surface != cur_surface:
                if new_surface == 1:
                    rs["blendEnable"] = True
                    rs["srcColorBlendFactor"] = 6
                    rs["dstColorBlendFactor"] = 7
                    rs["colorBlendOp"] = 0
                    rs["srcAlphaBlendFactor"] = 0   # ZERO  (preserve dst alpha)
                    rs["dstAlphaBlendFactor"] = 1   # ONE
                    rs["alphaBlendOp"] = 0          # ADD
                    rs["depthWriteEnable"] = False
                    rs["renderQueue"] = 3000
                    overrides |= 0x80 | 0x10 | 0x20 | 0x02 | 0x40
                else:
                    rs["blendEnable"] = False
                    rs["depthWriteEnable"] = True
                    rs["renderQueue"] = 2000
                    overrides |= 0x80 | 0x10 | 0x02 | 0x40
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

            # --- Cull Mode ---
            cull_items = ["None", "Front", "Back"]
            cull_val = int(rs.get("cullMode", 2))
            cull_idx = {0: 0, 1: 1, 2: 2}.get(cull_val, 2)
            field_label(ctx, "Cull Mode", so_lw)
            new_cull_idx = ctx.combo("##imat_cull_mode", cull_idx, cull_items)
            if new_cull_idx != cull_idx:
                rs["cullMode"] = {0: 0, 1: 1, 2: 2}[new_cull_idx]
                overrides |= 0x01
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

            # --- Depth Write ---
            dw_val = rs.get("depthWriteEnable", True)
            field_label(ctx, "Depth Write", so_lw)
            new_dw = ctx.checkbox("##imat_depth_write", dw_val)
            if new_dw != dw_val:
                rs["depthWriteEnable"] = new_dw
                overrides |= 0x02
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

            # --- Depth Test ---
            compare_items = ["Never", "Less", "Equal", "Less or Equal",
                             "Greater", "Not Equal", "Greater or Equal", "Always"]
            dt_op = int(rs.get("depthCompareOp", 1))
            field_label(ctx, "Depth Test", so_lw)
            new_op = ctx.combo("##imat_depth_test", dt_op, compare_items)
            if new_op != dt_op:
                rs["depthCompareOp"] = new_op
                overrides |= 0x08
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

            # --- Blend Mode (when transparent) ---
            if rs.get("blendEnable", False):
                blend_items = ["Alpha", "Additive", "Premultiply"]
                src = int(rs.get("srcColorBlendFactor", 6))
                dst = int(rs.get("dstColorBlendFactor", 7))
                if src == 1 and dst == 1:
                    cur_blend_idx = 1
                elif src == 1 and dst == 7:
                    cur_blend_idx = 2
                else:
                    cur_blend_idx = 0
                field_label(ctx, "Blend Mode", so_lw)
                new_blend_idx = ctx.combo("##imat_blend_mode", cur_blend_idx, blend_items)
                if new_blend_idx != cur_blend_idx:
                    if new_blend_idx == 0:
                        rs["srcColorBlendFactor"] = 6
                        rs["dstColorBlendFactor"] = 7
                    elif new_blend_idx == 1:
                        rs["srcColorBlendFactor"] = 1
                        rs["dstColorBlendFactor"] = 1
                    elif new_blend_idx == 2:
                        rs["srcColorBlendFactor"] = 1
                        rs["dstColorBlendFactor"] = 7
                    rs["colorBlendOp"] = 0
                    overrides |= 0x20
                    mat_data["renderStateOverrides"] = overrides
                    changed = True
                    requires_deserialize = True
                    requires_pipeline_refresh = True

            # --- Alpha Clip ---
            ac_enabled = rs.get("alphaClipEnabled", False)
            ac_threshold = float(rs.get("alphaClipThreshold", 0.5))
            field_label(ctx, "Alpha Clip", so_lw)
            new_ac = ctx.checkbox("##imat_alpha_clip", ac_enabled)
            if new_ac != ac_enabled:
                rs["alphaClipEnabled"] = new_ac
                if new_ac and "alphaClipThreshold" not in rs:
                    rs["alphaClipThreshold"] = 0.5
                overrides |= 0x100  # AlphaClip
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True
            if rs.get("alphaClipEnabled", False):
                field_label(ctx, "Threshold", so_lw)
                new_threshold = ctx.float_slider("##imat_alpha_threshold", ac_threshold, 0.0, 1.0)
                if abs(new_threshold - ac_threshold) > 1e-5:
                    rs["alphaClipThreshold"] = new_threshold
                    overrides |= 0x100  # AlphaClip
                    mat_data["renderStateOverrides"] = overrides
                    changed = True
                    requires_deserialize = True
                    requires_pipeline_refresh = True

            # --- Render Queue (clamped by surface type) ---
            is_transparent = rs.get("blendEnable", False)
            rq_min, rq_max = (2501, 5000) if is_transparent else (0, 2500)
            rq = int(rs.get("renderQueue", 2000))
            rq = max(rq_min, min(rq, rq_max))
            field_label(ctx, "Render Queue", so_lw)
            new_rq = int(ctx.drag_int("##imat_render_queue", rq, 1.0, rq_min, rq_max))
            if new_rq != rq:
                rs["renderQueue"] = new_rq
                overrides |= 0x40
                mat_data["renderStateOverrides"] = overrides
                changed = True
                requires_deserialize = True
                requires_pipeline_refresh = True

        if is_builtin:
            ctx.end_disabled()

        ctx.separator()

        # ── Properties ─────────────────────────────────────────────────
        if render_compact_section_header(ctx, "Properties##inline_mat", level="secondary"):
            props = mat_data.get("properties", {})
            if not props:
                ctx.label(t("inspector.no_properties"))
            else:
                prop_names = shader_utils.get_material_property_display_order(mat_data)
                plw = max_label_w(ctx, prop_names)
                for prop_name in prop_names:
                    prop = props[prop_name]
                    ptype = int(prop.get("type", 0))
                    value = prop.get("value")
                    if render_material_property(
                        ctx, prop_name, prop, ptype, value, plw,
                        wid_prefix="imat",
                    ):
                        if ptype == 6:
                            self._apply_inline_native_prop(native_mat, prop_name, prop.get("guid", ""), ptype)
                        else:
                            self._apply_inline_native_prop(native_mat, prop_name, prop["value"], ptype)
                        changed = True

        # ── Apply changes ──────────────────────────────────────────────
        if changed:
            # Refuse to save a material whose backing .mat file was deleted
            if hasattr(native_mat, 'is_deleted') and native_mat.is_deleted():
                return
            try:
                if requires_deserialize:
                    native_mat.deserialize(json.dumps(mat_data))
                if requires_pipeline_refresh:
                    engine = self._get_native_engine()
                    if engine and hasattr(engine, 'refresh_material_pipeline'):
                        engine.refresh_material_pipeline(native_mat)
                self._ensure_material_file_path(native_mat)
                if hasattr(native_mat, 'save'):
                    native_mat.save()
                self._inline_mat_version = -1  # force cache refresh next frame
            except (RuntimeError, ValueError):
                pass

    def _on_inline_shader_drop(self, path: str, required_ext: str, shaders_dict: dict):
        """Handle shader drag-drop on the inline material editor."""
        if path.lower().endswith(required_ext):
            key = "vertex" if required_ext == ".vert" else "fragment"
            old = shaders_dict.get(key, "")
            shaders_dict[key] = path
            if key == "fragment" and path != old and self.__inline_mat_data:
                shader_utils.sync_properties_from_shader(
                    self.__inline_mat_data, path, ".frag", remove_unknown=True,
                )
                self.__inline_mat_frag_shader_id = path
                self._inline_mat_sync_key = f"{path}:{shader_utils.get_shader_property_generation()}"

    @staticmethod
    def _apply_inline_native_prop(native_mat, prop_name: str, value, ptype: int):
        """Forward a property change to the native C++ material."""
        if not native_mat:
            return
        if ptype == 0:
            native_mat.set_float(prop_name, float(value))
        elif ptype == 1:
            native_mat.set_vector2(prop_name, (float(value[0]), float(value[1])))
        elif ptype == 2:
            native_mat.set_vector3(prop_name, (float(value[0]), float(value[1]), float(value[2])))
        elif ptype == 3:
            native_mat.set_vector4(prop_name, (float(value[0]), float(value[1]), float(value[2]), float(value[3])))
        elif ptype == 4:
            native_mat.set_int(prop_name, int(value))
        elif ptype == 5:
            native_mat.set_matrix(prop_name, [float(v) for v in value])
        elif ptype == 6:
            native_mat.set_texture_guid(prop_name, str(value))
        elif ptype == 7:
            native_mat.set_color(prop_name, (float(value[0]), float(value[1]), float(value[2]), float(value[3])))

    
    def _render_object_field(self, ctx: InfGUIContext, field_id: str, display_text: str,
                             type_hint: str, selected: bool = False, clickable: bool = True,
                             accept_drag_type: str = None, on_drop_callback=None,
                             picker_scene_items=None, picker_asset_items=None,
                             on_pick=None, on_clear=None) -> bool:
        return comp_ui.render_object_field(ctx, field_id, display_text, type_hint, selected,
                                           clickable, accept_drag_type, on_drop_callback,
                                           picker_scene_items, picker_asset_items,
                                           on_pick, on_clear)

    def _apply_dropped_material(self, renderer, mat_path: str, slot: int = 0):
        """Apply a dropped material file to the MeshRenderer via GUID resolution.

        Resolves path → GUID via AssetDatabase, loads via GUID through
        AssetRegistry.

        Args:
            renderer: MeshRenderer Python wrapper or C++ component.
            mat_path: Absolute path to the .mat file.
            slot: Material slot index (default 0 for backward compat).
        """
        from InfEngine.lib import AssetRegistry
        from InfEngine.debug import Debug

        registry = AssetRegistry.instance()
        adb = registry.get_asset_database()
        if not adb:
            Debug.log_warning("_apply_dropped_material: no AssetDatabase available")
            return

        guid = adb.get_guid_from_path(mat_path)
        if not guid:
            Debug.log_warning(f"Cannot resolve material path to GUID: {mat_path}")
            return

        # Load material by GUID through AssetRegistry (GUID-first)
        material = registry.load_material_by_guid(guid)
        if material:
            if hasattr(renderer, 'set_material') and slot > 0:
                old_mat = renderer.get_material(slot) if hasattr(renderer, 'get_material') else None
                renderer.set_material(slot, guid)
                _record_property(renderer, f"material_slot_{slot}", old_mat, material, f"Set Material Slot {slot}")
            else:
                old_mat = renderer.render_material
                renderer.material_guid = guid
                _record_property(renderer, "render_material", old_mat, material, "Set Material")
            Debug.log_internal(f"Applied material (GUID={guid[:12]}...) to slot {slot} from: {mat_path}")
        else:
            Debug.log_warning(f"Failed to load material GUID={guid} from: {mat_path}")

    def _render_cpp_component_generic(self, ctx: InfGUIContext, comp):
        comp_ui.render_cpp_component_generic(ctx, comp)
    
    def _open_add_component_popup(self, ctx: InfGUIContext):
        """Open the Add Component popup and refresh script list."""
        self.__add_component_search = ""
        self.__add_component_scripts = self._scan_project_scripts()
        self.__add_component_native_types = self._get_native_component_types()
        # Pre-cache menu paths once (avoids exec_module per frame)
        self.__script_menu_paths: dict[str, str | None] = {}
        for _, path in self.__add_component_scripts:
            self.__script_menu_paths[path] = self._get_script_menu_path(path)
        ctx.open_popup("##add_component_popup")

    def _get_native_component_types(self):
        """Get list of available native (C++) component type names."""
        from InfEngine.lib import get_registered_component_types
        types = get_registered_component_types()
        # Filter out Transform (always present)
        return [t for t in sorted(types) if t != "Transform"]

    def _scan_project_scripts(self):
        """Scan project root for .py files containing InfComponent subclasses."""
        results = []
        from InfEngine.engine.project_context import get_project_root
        project_root = get_project_root()
        if not project_root or not os.path.isdir(project_root):
            return results

        for dirpath, _dirnames, filenames in os.walk(project_root):
            # Skip hidden dirs, __pycache__, etc.
            rel = os.path.relpath(dirpath, project_root)
            if any(part.startswith('.') or part == '__pycache__' for part in rel.split(os.sep)):
                continue
            for fn in filenames:
                if not fn.endswith('.py') or fn.startswith('_'):
                    continue
                full = os.path.join(dirpath, fn)
                # Quick check: file must reference InfComponent
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4096)
                if 'InfComponent' in content:
                    display = fn[:-3]  # strip .py
                    results.append((display, full))
        results.sort(key=lambda x: x[0].lower())
        return results

    def _get_script_menu_path(self, script_path: str) -> str | None:
        """Return the ``@add_component_menu`` path for a script, or None.

        Loads the script module to inspect the class attribute.  Results are
        cached implicitly because ``_scan_project_scripts`` only runs on popup
        open.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location("_tmp_scan", script_path)
        if not spec or not spec.loader:
            return None
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            # Script has syntax errors or import failures — skip gracefully.
            return None
        from InfEngine.components.component import InfComponent
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if (isinstance(obj, type) and issubclass(obj, InfComponent)
                    and obj is not InfComponent):
                return getattr(obj, '_component_menu_path_', None)
        return None

    def _render_add_component_popup(self, ctx: InfGUIContext):
        """Render the searchable Add Component popup content."""
        # Styled padding
        ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, *Theme.POPUP_ADD_COMP_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.POPUP_ADD_COMP_SPC)
        
        # Search field
        ctx.set_next_item_width(Theme.ADD_COMP_SEARCH_W)
        new_text = ctx.input_text_with_hint("##comp_search", t("inspector.search_components"),
                                            self.__add_component_search)
        if isinstance(new_text, str):
            self.__add_component_search = new_text
        
        ctx.separator()
        
        search = self.__add_component_search.lower().strip()
        found_any = False
        
        # --- Native (C++) components grouped by category ---
        # Dynamically read _component_category_ from each wrapper class
        from InfEngine.components.builtin_component import BuiltinComponent
        native_types = getattr(self, '_InspectorPanel__add_component_native_types', [])
        native_matched = [t for t in native_types if not search or search in t.lower()]
        if native_matched:
            # Bucket matched types into categories via _component_category_
            cat_items: dict[str, list[str]] = {}
            uncategorized_native: list[str] = []
            for t in native_matched:
                wrapper_cls = BuiltinComponent._builtin_registry.get(t)
                cat = getattr(wrapper_cls, '_component_category_', '') if wrapper_cls else ''
                if cat:
                    cat_items.setdefault(cat, []).append(t)
                else:
                    uncategorized_native.append(t)

            # Render each category in stable sorted order
            for cat in sorted(cat_items.keys()):
                items = cat_items[cat]
                ctx.label(cat)
                ctx.separator()
                for type_name in items:
                    found_any = True
                    if ctx.selectable(f"  {type_name}"):
                        self._add_native_component(type_name)
                        ctx.close_current_popup()
                ctx.dummy(0, 4)

            # Any remaining native types not in a known category
            if uncategorized_native:
                ctx.label(t("inspector.miscellaneous"))
                ctx.separator()
                for type_name in uncategorized_native:
                    found_any = True
                    if ctx.selectable(f"  {type_name}"):
                        self._add_native_component(type_name)
                        ctx.close_current_popup()
                ctx.dummy(0, 4)
        
        # --- Engine Python components (Rendering) ---
        engine_py_components = self._get_engine_py_components()
        engine_matched = [(n, c) for n, c in engine_py_components
                          if not search or search in n.lower()]
        if engine_matched:
            # Group engine Python components by their _component_category_
            engine_cats: dict[str, list] = {}
            for comp_name, comp_cls in engine_matched:
                cat = getattr(comp_cls, '_component_category_', '') or 'Miscellaneous'
                engine_cats.setdefault(cat, []).append((comp_name, comp_cls))
            for cat in sorted(engine_cats.keys()):
                # Merge into the same category label as native components if it
                # hasn't been drawn yet; otherwise draw a new header.
                ctx.label(cat)
                ctx.separator()
                for comp_name, comp_cls in engine_cats[cat]:
                    found_any = True
                    if ctx.selectable(f"  {comp_name}"):
                        self._add_engine_py_component(comp_cls)
                        ctx.close_current_popup()
                ctx.dummy(0, 4)

        # --- Script components (grouped by @add_component_menu categories) ---
        script_matched = [(d, p) for d, p in self.__add_component_scripts
                          if not search or search in d.lower()]
        if script_matched:
            # Build category tree from @add_component_menu paths
            categorized: dict[str, list] = {}   # category -> [(display, path)]
            uncategorized: list = []
            for display_name, script_path in script_matched:
                menu_path = self.__script_menu_paths.get(script_path)
                if menu_path:
                    # Use the first path segment as the category header
                    parts = menu_path.split('/')
                    category = parts[0]
                    leaf_name = parts[-1] if len(parts) > 1 else display_name
                    categorized.setdefault(category, []).append((leaf_name, script_path))
                else:
                    uncategorized.append((display_name, script_path))

            # Render categorized scripts
            for cat in sorted(categorized.keys()):
                ctx.label(cat)
                ctx.separator()
                for leaf_name, spath in categorized[cat]:
                    found_any = True
                    if ctx.selectable(f"  {leaf_name}"):
                        self._handle_script_drop(spath)
                        ctx.close_current_popup()
                ctx.dummy(0, 4)

            # Render uncategorized scripts
            if uncategorized:
                ctx.label(t("inspector.scripts"))
                ctx.separator()
                for display_name, spath in uncategorized:
                    found_any = True
                    if ctx.selectable(f"  {display_name}"):
                        self._handle_script_drop(spath)
                        ctx.close_current_popup()
        
        if not found_any:
            ctx.label(t("inspector.no_components_found"))
        
        ctx.pop_style_var(2)

    def _add_native_component(self, type_name: str):
        """Add a built-in (C++) component to all selected objects."""
        from .selection_manager import SelectionManager
        sel = SelectionManager.instance()
        objects = self._get_all_selected_objects() if sel.is_multi() else []
        if not objects:
            obj = self._get_selected_object()
            objects = [obj] if obj else []
        if not objects:
            return
        from InfEngine.debug import Debug
        for obj in objects:
            result = obj.add_component(type_name)
            if result is not None:
                Debug.log_internal(f"Added component: {type_name}")
                _record_add_component(obj, type_name, result, is_py=False)
            else:
                Debug.log_error(f"Failed to add component: {type_name}")
        self._multi_cache_ids = ()  # invalidate component cache
        from InfEngine.gizmos.collector import notify_scene_changed
        notify_scene_changed()

    @staticmethod
    def _get_engine_py_components():
        """Return a list of (display_name, class) for engine-level Python
        components that should appear in the Add Component popup."""
        result = []
        from InfEngine.renderstack.render_stack import RenderStack
        result.append(("RenderStack", RenderStack))
        return result

    def _add_engine_py_component(self, comp_cls):
        """Instantiate and attach an engine-level Python component."""
        selected_object = self._get_selected_object()
        if not selected_object:
            return
        from InfEngine.debug import Debug
        # Enforce singleton for @disallow_multiple (RenderStack uses class singleton)
        if getattr(comp_cls, '_disallow_multiple_', False):
            for pc in selected_object.get_py_components():
                if isinstance(pc, comp_cls):
                    Debug.log_warning(
                        f"Cannot add another '{comp_cls.__name__}' — "
                        f"only one per scene is allowed")
                    return
        instance = comp_cls()
        # Resolve script GUID for engine Python components so they
        # survive save/load round-trips (same logic as script drops).
        if self.__asset_database:
            import inspect as _inspect
            src_file = _inspect.getfile(comp_cls)
            if src_file:
                guid = self.__asset_database.get_guid_from_path(src_file)
                if not guid:
                    guid = self.__asset_database.import_asset(src_file)
                if guid:
                    instance._script_guid = guid
        selected_object.add_py_component(instance)
        Debug.log_internal(f"Added component: {comp_cls.__name__}")
        _record_add_component(selected_object, comp_cls.__name__, instance, is_py=True)

    def _handle_script_drop(self, script_path: str):
        """Handle script file drop - load and attach component."""
        selected_object = self._get_selected_object()
        if not selected_object:
            return
        
        from InfEngine.components import load_and_create_component
        from InfEngine.debug import Debug

        # Load component from script file
        try:
            component_instance = load_and_create_component(script_path, asset_database=self.__asset_database)
        except Exception as exc:
            Debug.log_error(f"Failed to load script '{script_path}': {exc}")
            return
        if component_instance is None:
            Debug.log_error(f"No InfComponent found in '{script_path}'")
            return

        # --- Enforce @disallow_multiple ---
        comp_cls = component_instance.__class__
        if getattr(comp_cls, '_disallow_multiple_', False):
            existing = selected_object.get_py_components()
            for ec in existing:
                if type(ec).__name__ == comp_cls.__name__:
                    Debug.log_warning(
                        f"Cannot add another '{comp_cls.__name__}' — "
                        f"@disallow_multiple is set")
                    return

        # --- Enforce @require_component ---
        required = getattr(comp_cls, '_require_components_', [])
        for req_type in required:
            req_name = req_type if isinstance(req_type, str) else req_type.__name__
            # Check C++ components
            has_it = False
            if hasattr(selected_object, 'get_components'):
                for c in selected_object.get_components():
                    if c.type_name == req_name:
                        has_it = True
                        break
            # Check existing Python components
            if not has_it and hasattr(selected_object, 'get_py_components'):
                for pc in selected_object.get_py_components():
                    if pc.type_name == req_name:
                        has_it = True
                        break
            if not has_it:
                # Try to auto-add the required component
                if isinstance(req_type, str):
                    selected_object.add_component(req_type)
                    Debug.log_internal(
                        f"Auto-added required component '{req_name}'")
                else:
                    Debug.log_warning(
                        f"'{comp_cls.__name__}' requires '{req_name}' — "
                        f"please add it manually")

        # Track script path for reload
        if self.__asset_database:
            guid = self.__asset_database.get_guid_from_path(script_path)
            if not guid:
                guid = self.__asset_database.import_asset(script_path)
            component_instance._script_guid = guid

        # Attach to selected GameObject
        selected_object.add_py_component(component_instance)
        _record_add_component(selected_object, component_instance.type_name, component_instance, is_py=True)

        Debug.log_internal(f"Added component {component_instance.type_name} from {os.path.basename(script_path)}")


    def _render_py_component(self, ctx: InfGUIContext, py_comp):
        comp_ui.render_py_component(ctx, py_comp)

    # ------------------------------------------------------------------
    # Component clipboard (copy / paste properties)
    # ------------------------------------------------------------------
    # Stored as a module-level dict: {"type_name": str, "data": str (JSON),
    #   "is_native": bool, "script_guid": str|None}
    _component_clipboard: dict | None = None

    def _copy_component_properties(self, comp, type_name: str, *, is_native: bool):
        """Serialize a component's properties into the clipboard."""
        try:
            if is_native:
                from InfEngine.components.builtin_component import BuiltinComponent
                wrapper_cls = BuiltinComponent._builtin_registry.get(type_name)
                if wrapper_cls and hasattr(comp, 'game_object'):
                    wrapper = wrapper_cls._get_or_create_wrapper(comp, comp.game_object)
                    data = wrapper._serialize_fields()
                else:
                    data = comp.serialize()
                InspectorPanel._component_clipboard = {
                    "type_name": type_name,
                    "data": data,
                    "is_native": True,
                    "script_guid": None,
                }
            else:
                data = comp._serialize_fields()
                InspectorPanel._component_clipboard = {
                    "type_name": type_name,
                    "data": data,
                    "is_native": False,
                    "script_guid": getattr(comp, '_script_guid', None),
                }
            from InfEngine.debug import Debug
            Debug.log_internal(f"Copied properties of '{type_name}'")
        except Exception as exc:
            from InfEngine.debug import Debug
            Debug.log_warning(f"Failed to copy properties of '{type_name}': {exc}")

    def _paste_component_as_new(self, selected_object):
        """Add a new component using the copied clipboard properties."""
        cb = InspectorPanel._component_clipboard
        if not cb:
            return
        from InfEngine.debug import Debug
        try:
            type_name = cb["type_name"]
            if cb["is_native"]:
                result = selected_object.add_component(type_name)
                if result is None:
                    Debug.log_error(f"Failed to add native component '{type_name}'")
                    return
                from InfEngine.components.builtin_component import BuiltinComponent
                wrapper_cls = BuiltinComponent._builtin_registry.get(type_name)
                if wrapper_cls and hasattr(result, 'game_object'):
                    wrapper = wrapper_cls._get_or_create_wrapper(result, result.game_object)
                    wrapper._deserialize_fields(cb["data"])
                else:
                    data = json.loads(cb["data"])
                    ignore = {"schema_version", "type", "enabled", "component_id",
                              "__schema_version__", "__type_name__", "__component_id__"}
                    filtered = {k: v for k, v in data.items() if k not in ignore}
                    filtered["type"] = type_name
                    filtered["enabled"] = True
                    result.deserialize(json.dumps(filtered))
                _record_add_component(selected_object, type_name, result, is_py=False)
            else:
                guid = cb.get("script_guid")
                instance = None
                if guid and self.__asset_database:
                    script_path = self.__asset_database.get_path_from_guid(guid)
                    if script_path:
                        from InfEngine.components import load_and_create_component
                        instance = load_and_create_component(
                            script_path, asset_database=self.__asset_database)
                        if instance:
                            instance._script_guid = guid
                if instance is None:
                    Debug.log_error(
                        f"Cannot paste '{type_name}' — script not found")
                    return
                instance._deserialize_fields(cb["data"])
                selected_object.add_py_component(instance)
                _record_add_component(selected_object, type_name, instance, is_py=True)
            Debug.log_internal(f"Pasted '{type_name}' as new component")
            _notify_scene_modified()
            from InfEngine.gizmos.collector import notify_scene_changed
            notify_scene_changed()
        except Exception as exc:
            Debug.log_warning(f"Failed to paste component: {exc}")

    def _paste_properties_onto(self, comp, type_name: str, *, is_native: bool):
        """Overwrite properties of an existing component from clipboard."""
        cb = InspectorPanel._component_clipboard
        if not cb or cb["type_name"] != type_name:
            return
        from InfEngine.debug import Debug
        try:
            if is_native:
                from InfEngine.components.builtin_component import BuiltinComponent
                wrapper_cls = BuiltinComponent._builtin_registry.get(type_name)
                if wrapper_cls and hasattr(comp, 'game_object'):
                    wrapper = wrapper_cls._get_or_create_wrapper(comp, comp.game_object)
                    wrapper._deserialize_fields(cb["data"])
                else:
                    data = json.loads(cb["data"])
                    ignore = {"schema_version", "type", "enabled", "component_id",
                              "__schema_version__", "__type_name__", "__component_id__"}
                    filtered = {k: v for k, v in data.items() if k not in ignore}
                    filtered["type"] = type_name
                    filtered["enabled"] = bool(comp.enabled)
                    comp.deserialize(json.dumps(filtered))
            else:
                comp._deserialize_fields(cb["data"])
            Debug.log_internal(f"Pasted properties onto '{type_name}'")
            _notify_scene_modified()
        except Exception as exc:
            Debug.log_warning(f"Failed to paste properties: {exc}")

    @staticmethod
    def _open_script_in_editor(script_path: str):
        """Open a script file in the user's code editor (VS Code preferred)."""
        import shutil
        import subprocess
        import os

        code_exe = shutil.which("code")

        # Fallback: common VS Code install locations on Windows
        if code_exe is None and os.name == "nt":
            candidates = [
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"),
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
                r"C:\Program Files\Microsoft VS Code\bin\code.cmd",
                r"C:\Program Files\Microsoft VS Code\Code.exe",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    code_exe = c
                    break

        if code_exe is not None:
            try:
                flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
                subprocess.Popen([code_exe, "-g", script_path],
                                 creationflags=flags)
                return
            except OSError:
                pass

        # Last resort: open with OS default application
        try:
            os.startfile(script_path)
        except (OSError, AttributeError):
            from InfEngine.debug import Debug
            Debug.log_warning(
                f"Cannot open script — no suitable editor found. "
                f"Script: {script_path}")

    def _get_clipboard_type_name(self) -> str | None:
        """Return the type_name from the component clipboard, or None."""
        cb = InspectorPanel._component_clipboard
        return cb["type_name"] if cb else None

    # ------------------------------------------------------------------
    # Single-object inspector (extracted from old _render_properties_module)
    # ------------------------------------------------------------------
    def _render_single_object(self, ctx: InfGUIContext, selected_object):
        """Render inspector for a single selected object."""
        from InfEngine.engine.undo import snapshot_renderstack, restore_renderstack

        components = []
        if hasattr(selected_object, 'get_components'):
            try:
                components = list(selected_object.get_components())
            except RuntimeError:
                components = []

        py_components = []
        if hasattr(selected_object, 'get_py_components'):
            try:
                py_components = list(selected_object.get_py_components())
            except RuntimeError:
                py_components = []

        # --- Register undo tracking for all inspectable targets ---
        tracker = self._undo_tracker
        obj_id = selected_object.id

        # Track GameObject name / active
        def _go_snapshot():
            import json as _j
            return _j.dumps({"name": selected_object.name,
                             "active": selected_object.active})
        def _go_restore(s):
            import json as _j
            d = _j.loads(s)
            selected_object.name = d["name"]
            selected_object.active = d["active"]
        tracker.track(f"go:{obj_id}", _go_snapshot, _go_restore, "Edit GameObject")

        # Track Transform
        trans = selected_object.get_transform()
        if trans:
            _trans = trans  # capture
            tracker.track(
                f"transform:{obj_id}",
                lambda: _trans.serialize(),
                lambda s: _trans.deserialize(s),
                "Edit Transform",
            )

        # Track C++ components
        for comp in components:
            try:
                tn = comp.type_name
                if tn == "Transform":
                    continue
                if hasattr(comp, 'get_py_component'):
                    continue
                cid = getattr(comp, "component_id", None) or id(comp)
                _c = comp  # capture
                tracker.track(
                    f"native:{cid}",
                    lambda _cc=_c: _cc.serialize(),
                    lambda s, _cc=_c: _cc.deserialize(s),
                    f"Edit {tn}",
                )
            except (RuntimeError, AttributeError):
                pass

        # Track Python components (includes RenderStack)
        for py_comp in py_components:
            pc_id = getattr(py_comp, "component_id", None) or id(py_comp)
            from InfEngine.renderstack.render_stack import RenderStack
            if isinstance(py_comp, RenderStack):
                _rs = py_comp  # capture
                tracker.track(
                    f"renderstack:{pc_id}",
                    lambda _s=_rs: snapshot_renderstack(_s),
                    lambda s, _s=_rs: restore_renderstack(_s, s),
                    "Edit RenderStack",
                )
            else:
                _pc = py_comp  # capture
                tracker.track(
                    f"pycomp:{pc_id}",
                    lambda _p=_pc: _p._serialize_fields(),
                    lambda s, _p=_pc: _p._deserialize_fields(s),
                    f"Edit {py_comp.type_name}",
                )

        ctx.push_id_str(f"selected_obj_{selected_object.id}")
        # Active checkbox (no label — matches Unity's checkbox-only style)
        is_active = selected_object.active
        new_active = render_inspector_checkbox(ctx, "##obj_active", is_active)
        if new_active != is_active:
            _record_property(selected_object, "active", is_active, new_active, "Set Active")

        ctx.same_line(0, 6)
        # Editable object name
        ctx.set_next_item_width(-1)
        old_name = selected_object.name
        new_name = ctx.text_input("##obj_name", old_name, 256)
        if new_name != old_name:
            _record_property(selected_object, "name", old_name, new_name, "Rename")

        # --- Tag & Layer dropdowns ---
        self._render_tag_layer_row(ctx, selected_object)

        # --- Prefab instance header ---
        if getattr(selected_object, 'is_prefab_instance', False):
            self._render_prefab_header(ctx, selected_object)

        ctx.dummy(0, Theme.INSPECTOR_TITLE_GAP)
        ctx.separator()
        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)

        # Check if any py_component hides Transform
        _hide_transform = False
        for _pc in py_components:
            if getattr(type(_pc), '_hide_transform_', False):
                _hide_transform = True
                break

        # Transform (skip for screen-space UI elements)
        if not _hide_transform:
            trans = selected_object.get_transform()
            transform_icon = self._get_component_icon_id("Transform")
            if render_component_header(ctx, "Transform", icon_id=transform_icon, show_enabled=False):
                self._render_transform_component(ctx, trans)

        # C++ components (MeshRenderer, etc.)
        mesh_renderers = []
        if components:
            for comp in components:
                try:
                    type_name = comp.type_name
                    if type_name == "Transform":
                        continue
                    if hasattr(comp, 'get_py_component'):
                        continue

                    if type_name == "MeshRenderer":
                        mesh_renderers.append(comp)

                    comp_id = getattr(comp, "component_id", None)
                    if not comp_id:
                        comp_id = id(comp)
                    current_enabled = bool(comp.enabled)
                except Exception as _comp_meta_exc:
                    from InfEngine.debug import Debug
                    Debug.log_warning(
                        f"[Inspector] Skipping invalid native component: {_comp_meta_exc}"
                    )
                    continue

                ctx.push_id_str(f"native_{type_name}_{comp_id}")
                icon_id = self._get_component_icon_id(type_name)
                header_open, new_enabled = render_component_header(
                    ctx,
                    type_name,
                    icon_id=icon_id,
                    show_enabled=True,
                    is_enabled=current_enabled,
                )
                # Right-click context menu
                if self.__right_click_remove_enabled and ctx.begin_popup_context_item("comp_ctx"):
                    # -- Copy Properties --
                    if ctx.selectable(t("inspector.copy_properties")):
                        self._copy_component_properties(comp, type_name, is_native=True)
                    # -- Paste as New Component --
                    cb_type = self._get_clipboard_type_name()
                    has_cb = cb_type is not None
                    ctx.begin_disabled(not has_cb)
                    if ctx.selectable(t("inspector.paste_as_new")):
                        self._paste_component_as_new(selected_object)
                    ctx.end_disabled()
                    # -- Paste as Properties --
                    can_paste_props = has_cb and cb_type == type_name
                    ctx.begin_disabled(not can_paste_props)
                    if ctx.selectable(t("inspector.paste_properties")):
                        self._paste_properties_onto(comp, type_name, is_native=True)
                    ctx.end_disabled()
                    ctx.separator()
                    # -- Remove --
                    if ctx.selectable(t("inspector.remove")):
                        if hasattr(selected_object, 'remove_component'):
                            blockers = []
                            if hasattr(selected_object, 'get_remove_component_blockers'):
                                try:
                                    blockers = list(selected_object.get_remove_component_blockers(comp) or [])
                                except RuntimeError:
                                    blockers = []
                            can_remove = not blockers
                            if can_remove and hasattr(selected_object, 'can_remove_component'):
                                can_remove = selected_object.can_remove_component(comp)
                            if can_remove:
                                from InfEngine.engine.undo import UndoManager, RemoveNativeComponentCommand
                                mgr = UndoManager.instance()
                                if mgr:
                                    mgr.execute(RemoveNativeComponentCommand(
                                        selected_object.id, type_name, comp))
                                else:
                                    selected_object.remove_component(comp)
                                    _notify_scene_modified()
                                    from InfEngine.gizmos.collector import notify_scene_changed
                                    notify_scene_changed()
                            else:
                                from InfEngine.debug import Debug
                                suffix = (
                                    f" required by: {', '.join(blockers)}"
                                    if blockers else
                                    "another component depends on it"
                                )
                                Debug.log_warning(
                                    f"Cannot remove '{type_name}' — "
                                    f"{suffix}")
                        ctx.end_popup()
                        continue
                    ctx.end_popup()

                if new_enabled != current_enabled:
                    _record_property(comp, "enabled", current_enabled, new_enabled, f"Toggle {type_name}")

                try:
                    if header_open:
                        comp_ui.render_component(ctx, comp)
                except Exception as _comp_exc:
                    from InfEngine.debug import Debug
                    Debug.log_warning(
                        f"[Inspector] C++ component render error ({type_name}): {_comp_exc}"
                    )
                finally:
                    ctx.pop_id()

        # Python components (InfComponent subclasses)
        if py_components:
            for py_comp in py_components:
                type_name = py_comp.type_name
                comp_id = getattr(py_comp, "component_id", None)
                if not comp_id:
                    comp_id = id(py_comp)
                ctx.push_id_str(f"py_comp_{type_name}_{comp_id}")
                icon_id = 0
                custom_path = getattr(py_comp.__class__, '_component_icon_', None)
                if custom_path:
                    icon_id = self._load_custom_icon(custom_path, type_name)
                if icon_id == 0:
                    icon_id = self._get_component_icon_id(type_name, is_script=True)
                header_open, new_enabled = render_component_header(
                    ctx,
                    type_name,
                    icon_id=icon_id,
                    show_enabled=True,
                    is_enabled=py_comp.enabled,
                    suffix=" (Script)",
                )
                # Right-click context menu
                if self.__right_click_remove_enabled and ctx.begin_popup_context_item("py_comp_ctx"):
                    # -- Copy Properties --
                    if ctx.selectable(t("inspector.copy_properties")):
                        self._copy_component_properties(py_comp, type_name, is_native=False)
                    # -- Paste as New Component --
                    cb_type = self._get_clipboard_type_name()
                    has_cb = cb_type is not None
                    ctx.begin_disabled(not has_cb)
                    if ctx.selectable(t("inspector.paste_as_new")):
                        self._paste_component_as_new(selected_object)
                    ctx.end_disabled()
                    # -- Paste as Properties --
                    can_paste_props = has_cb and cb_type == type_name
                    ctx.begin_disabled(not can_paste_props)
                    if ctx.selectable(t("inspector.paste_properties")):
                        self._paste_properties_onto(py_comp, type_name, is_native=False)
                    ctx.end_disabled()
                    ctx.separator()
                    # -- Show Script (non-built-in only) --
                    _py_guid = getattr(py_comp, '_script_guid', None)
                    _script_path = None
                    if _py_guid and self.__asset_database:
                        _script_path = self.__asset_database.get_path_from_guid(_py_guid)
                    from InfEngine.components.builtin_component import BuiltinComponent
                    _is_builtin = isinstance(py_comp, BuiltinComponent)
                    if not _is_builtin and _script_path:
                        if ctx.selectable(t("inspector.show_script")):
                            self._open_script_in_editor(_script_path)
                    ctx.separator()
                    # -- Remove --
                    if ctx.selectable(t("inspector.remove")):
                        if hasattr(selected_object, 'remove_py_component'):
                            from InfEngine.engine.undo import UndoManager, RemovePyComponentCommand
                            mgr = UndoManager.instance()
                            if mgr:
                                mgr.execute(RemovePyComponentCommand(
                                    selected_object.id, py_comp))
                            elif not selected_object.remove_py_component(py_comp):
                                from InfEngine.debug import Debug
                                Debug.log_warning(
                                    f"Cannot remove '{type_name}' — "
                                    f"another component depends on it")
                            else:
                                _notify_scene_modified()
                        ctx.end_popup()
                        continue
                    ctx.end_popup()

                if new_enabled != py_comp.enabled:
                    _record_property(py_comp, "enabled", py_comp.enabled, new_enabled, f"Toggle {type_name}")

                try:
                    # Check if this component's script has load errors
                    _script_err = None
                    if getattr(py_comp, '_is_broken', False):
                        _script_err = getattr(py_comp, '_broken_error', None) or 'Script failed to load'
                    else:
                        _py_guid = getattr(py_comp, '_script_guid', None)
                        if _py_guid and self.__asset_database:
                            from InfEngine.components.script_loader import get_script_error_by_path
                            _py_path = self.__asset_database.get_path_from_guid(_py_guid)
                            if _py_path:
                                _script_err = get_script_error_by_path(_py_path)

                    if _script_err:
                        if header_open:
                            ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
                            ctx.text_wrapped(_script_err)
                            ctx.pop_style_color(1)
                    elif header_open:
                        self._render_py_component(ctx, py_comp)
                finally:
                    ctx.pop_id()

        # Add Component area
        ctx.separator()
        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)

        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.ADD_COMP_FRAME_PAD)
        ctx.set_cursor_pos_x(Theme.INSPECTOR_ACTION_ALIGN_X)
        ctx.button(t("inspector.add_component"), lambda: self._open_add_component_popup(ctx), -1, 0)
        ctx.pop_style_var(1)

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)

        if ctx.begin_popup("##add_component_popup"):
            self._render_add_component_popup(ctx)
            ctx.end_popup()

        self._render_object_material_sections(ctx, mesh_renderers)
        ctx.pop_id()

    # ------------------------------------------------------------------
    # Multi-object transform
    # ------------------------------------------------------------------
    def _render_multi_transform(self, ctx: InfGUIContext, objects: list):
        """Render Transform for multiple selected objects.

        Shows the primary object's values; edits apply as deltas to ALL
        selected objects (Unity behaviour).  Uses dedicated C++ function.
        """
        from InfEngine.lib import Vector3

        transforms = []
        for o in objects:
            t = o.get_transform()
            if t is not None:
                transforms.append(t)
        if not transforms:
            return

        lw = self._max_label_w(ctx, ["Position", "Rotation", "Scale"])
        primary = transforms[0]

        pos = primary.local_position
        px, py_, pz = pos[0], pos[1], pos[2]
        rot = primary.local_euler_angles
        rx, ry, rz = rot[0], rot[1], rot[2]
        scl = primary.local_scale
        sx, sy, sz = scl[0], scl[1], scl[2]

        r = ctx.render_transform_fields(
            px, py_, pz, rx, ry, rz, sx, sy, sz,
            DRAG_SPEED_DEFAULT, DRAG_SPEED_DEFAULT, DRAG_SPEED_FINE, lw,
        )

        npx, npy, npz = r[0], r[1], r[2]
        dx, dy, dz = npx - px, npy - py_, npz - pz
        if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dz) > 1e-6:
            for t in transforms:
                old = t.local_position
                _record_property(t, "local_position", old,
                                 Vector3(old[0] + dx, old[1] + dy, old[2] + dz),
                                 "Set Position (Multi)")

        nrx, nry, nrz = r[3], r[4], r[5]
        drx, dry, drz = nrx - rx, nry - ry, nrz - rz
        if abs(drx) > 1e-6 or abs(dry) > 1e-6 or abs(drz) > 1e-6:
            for t in transforms:
                old = t.local_euler_angles
                _record_property(t, "local_euler_angles", old,
                                 Vector3(old[0] + drx, old[1] + dry, old[2] + drz),
                                 "Set Rotation (Multi)")

        nsx, nsy, nsz = r[6], r[7], r[8]
        dsx, dsy, dsz = nsx - sx, nsy - sy, nsz - sz
        if abs(dsx) > 1e-6 or abs(dsy) > 1e-6 or abs(dsz) > 1e-6:
            for t in transforms:
                old = t.local_scale
                _record_property(t, "local_scale", old,
                                 Vector3(old[0] + dsx, old[1] + dsy, old[2] + dsz),
                                 "Set Scale (Multi)")

    # ------------------------------------------------------------------
    # Multi-object MeshRenderer display
    # ------------------------------------------------------------------
    def _render_multi_mesh_renderer(self, ctx: InfGUIContext, renderers: list):
        """Render MeshRenderer fields for multiple selected objects.

        Shows "-" for mesh name and materials when values differ across
        the selected renderers.
        """
        from InfEngine.components.builtin_component import BuiltinComponent

        # Wrap all renderers
        wrapped = []
        for r in renderers:
            if not isinstance(r, BuiltinComponent):
                wrapper_cls = BuiltinComponent._builtin_registry.get("MeshRenderer")
                go = getattr(r, 'game_object', None)
                if wrapper_cls and go is not None:
                    r = wrapper_cls._get_or_create_wrapper(r, go)
            wrapped.append(r)

        lw = self._max_label_w(ctx, ["Mesh", "Materials", "Element 0"])

        # ── Mesh field — show "-" if any differ ──
        mesh_displays = []
        for r in wrapped:
            if r.has_inline_mesh():
                name = getattr(r, 'inline_mesh_name', '') or ''
                mesh_displays.append(name if name else "(Primitive)")
            elif getattr(r, 'has_mesh_asset', False):
                mesh_displays.append(getattr(r, 'mesh_name', '') or 'Mesh')
            else:
                mesh_displays.append("None")
        all_same_mesh = len(set(mesh_displays)) <= 1
        mesh_display = mesh_displays[0] if all_same_mesh else "—"

        self._field_label(ctx, "Mesh", lw)
        self._render_object_field(ctx, "mesh_field", mesh_display, "Mesh", clickable=False)
        ctx.separator()

        # ── Material slots — union count, show "-" when names differ ──
        mat_counts = [getattr(r, 'material_count', 0) or 1 for r in wrapped]
        max_mat = max(mat_counts) if mat_counts else 1
        self._field_label(ctx, "Materials", lw)
        if len(set(mat_counts)) == 1:
            ctx.label(f"Size: {mat_counts[0]}")
        else:
            ctx.label(f"Size: —")

        for slot_idx in range(max_mat):
            mat_names = []
            for r in wrapped:
                mc = getattr(r, 'material_count', 0) or 1
                if slot_idx < mc:
                    mat = r.get_effective_material(slot_idx)
                    mat_names.append(getattr(mat, 'name', 'None') if mat else 'None')
                else:
                    mat_names.append("—")
            all_same_mat = len(set(mat_names)) <= 1
            display = mat_names[0] if all_same_mat else "—"
            slot_label = f"Element {slot_idx}"
            self._field_label(ctx, slot_label, lw)
            self._render_object_field(ctx, f"mat_{slot_idx}", display, "Material", clickable=False)

        ctx.separator()

        # ── CppProperty fields via first renderer as representative ──
        comp_ui.render_py_component(ctx, wrapped[0])

    # ------------------------------------------------------------------
    # Multi-object inspector
    # ------------------------------------------------------------------
    def _render_multi_edit(self, ctx: InfGUIContext, objects: list):
        """Render inspector for multiple selected objects (Unity-style multi-edit)."""
        import json as _json
        from InfEngine.engine.undo import snapshot_renderstack, restore_renderstack
        from InfEngine.renderstack.render_stack import RenderStack

        # --- Undo tracking registration ---
        # Only re-iterate objects when the selection set changed;
        # otherwise just mark existing entries as active.
        obj_ids = tuple(o.id for o in objects)
        tracker = self._undo_tracker

        if obj_ids != self._multi_cache_ids:
            # Selection changed — full registration
            for obj in objects:
                oid = obj.id
                _o = obj  # capture
                # Track GameObject name / active
                def _go_snap(_oo=_o):
                    return _json.dumps({"name": _oo.name, "active": _oo.active})
                def _go_rest(s, _oo=_o):
                    d = _json.loads(s)
                    _oo.name = d["name"]
                    _oo.active = d["active"]
                tracker.track(f"go:{oid}", _go_snap, _go_rest, "Edit Objects (Multi)")

                # Track Transform
                trans = obj.get_transform()
                if trans:
                    _t = trans
                    tracker.track(
                        f"transform:{oid}",
                        lambda _tt=_t: _tt.serialize(),
                        lambda s, _tt=_t: _tt.deserialize(s),
                        "Edit Transform (Multi)",
                    )

                # Track C++ components
                try:
                    comps = list(obj.get_components()) if hasattr(obj, 'get_components') else []
                except RuntimeError:
                    comps = []
                for comp in comps:
                    try:
                        tn = comp.type_name
                        if tn == "Transform" or hasattr(comp, 'get_py_component'):
                            continue
                        cid = getattr(comp, "component_id", None) or id(comp)
                        _c = comp
                        tracker.track(
                            f"native:{cid}",
                            lambda _cc=_c: _cc.serialize(),
                            lambda s, _cc=_c: _cc.deserialize(s),
                            f"Edit {tn} (Multi)",
                        )
                    except (RuntimeError, AttributeError):
                        pass

                # Track Python components
                try:
                    pcs = list(obj.get_py_components()) if hasattr(obj, 'get_py_components') else []
                except RuntimeError:
                    pcs = []
                for pc in pcs:
                    pc_id = getattr(pc, "component_id", None) or id(pc)
                    if isinstance(pc, RenderStack):
                        _rs = pc
                        tracker.track(
                            f"renderstack:{pc_id}",
                            lambda _s=_rs: snapshot_renderstack(_s),
                            lambda s, _s=_rs: restore_renderstack(_s, s),
                            "Edit RenderStack (Multi)",
                        )
                    else:
                        _pc = pc
                        tracker.track(
                            f"pycomp:{pc_id}",
                            lambda _p=_pc: _p._serialize_fields(),
                            lambda s, _p=_pc: _p._deserialize_fields(s),
                            f"Edit {pc.type_name} (Multi)",
                        )
        else:
            # Selection unchanged — bulk re-activate, no C++ calls
            tracker.mark_all_active()

        n = len(objects)
        ctx.push_id_str("multi_edit")

        # Header showing count
        ctx.label(f"{n} objects selected")

        # Mixed active toggle — show checkbox with mixed state
        all_active = all(o.active for o in objects)
        any_active = any(o.active for o in objects)
        # Use a checkbox: checked if all active, else unchecked
        check_val = all_active
        new_val = render_inspector_checkbox(ctx, "##multi_active", check_val)
        if new_val != check_val:
            # Toggle all to the new state
            for o in objects:
                if o.active != new_val:
                    _record_property(o, "active", o.active, new_val, "Set Active (Multi)")

        ctx.dummy(0, Theme.INSPECTOR_TITLE_GAP)
        ctx.separator()
        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)

        # Transform — show if at least one object has transform visible
        # (cached along with component discovery below)

        # Find common component types — cached across frames, invalidated
        # when the set of selected object IDs changes.
        if obj_ids != self._multi_cache_ids:
            # --- Transform visibility check ---
            _any_has_transform = False
            common_cpp_types = None
            per_obj_comps: dict[str, list] = {}
            for o in objects:
                try:
                    comps = list(o.get_components()) if hasattr(o, 'get_components') else []
                except RuntimeError:
                    comps = []
                type_set = set()
                for c in comps:
                    tn = c.type_name
                    if tn == "Transform" or hasattr(c, 'get_py_component'):
                        continue
                    type_set.add(tn)
                    per_obj_comps.setdefault(tn, []).append(c)
                if common_cpp_types is None:
                    common_cpp_types = type_set
                else:
                    common_cpp_types &= type_set
            if common_cpp_types is None:
                common_cpp_types = set()

            common_py_types = None
            per_obj_py: dict[str, list] = {}
            for o in objects:
                try:
                    pcs = list(o.get_py_components()) if hasattr(o, 'get_py_components') else []
                except RuntimeError:
                    pcs = []
                type_set = set()
                hide = False
                for pc in pcs:
                    tn = pc.type_name
                    type_set.add(tn)
                    per_obj_py.setdefault(tn, []).append(pc)
                    if getattr(type(pc), '_hide_transform_', False):
                        hide = True
                if not hide:
                    _any_has_transform = True
                if common_py_types is None:
                    common_py_types = type_set
                else:
                    common_py_types &= type_set
            if common_py_types is None:
                common_py_types = set()

            self._multi_cache_ids = obj_ids
            self._multi_cache_cpp = common_cpp_types
            self._multi_cache_py = common_py_types
            self._multi_cache_per_cpp = per_obj_comps
            self._multi_cache_per_py = per_obj_py
            self._multi_cache_has_transform = _any_has_transform
        else:
            common_cpp_types = self._multi_cache_cpp
            common_py_types = self._multi_cache_py
            per_obj_comps = self._multi_cache_per_cpp
            per_obj_py = self._multi_cache_per_py
            _any_has_transform = self._multi_cache_has_transform

        if _any_has_transform:
            transform_icon = self._get_component_icon_id("Transform")
            if render_component_header(ctx, "Transform", icon_id=transform_icon, show_enabled=False, force_open=True)[0]:
                self._render_multi_transform(ctx, objects)

        # Render common C++ components — force open in multi-select,
        # with special handling for MeshRenderer (show "-" for different values).
        for tn in sorted(common_cpp_types):
            ctx.push_id_str(f"multi_cpp_{tn}")
            icon_id = self._get_component_icon_id(tn)
            comps_for_type = per_obj_comps.get(tn, [])

            # Enabled state: mixed if not all the same
            enabled_vals = [c.enabled for c in comps_for_type]
            all_enabled = all(enabled_vals)
            header_open, new_enabled = render_component_header(
                ctx, tn,
                icon_id=icon_id,
                show_enabled=True,
                is_enabled=all_enabled,
                force_open=True,
            )
            if new_enabled != all_enabled:
                for c in comps_for_type:
                    if c.enabled != new_enabled:
                        _record_property(c, "enabled", c.enabled, new_enabled, f"Toggle {tn} (Multi)")

            if header_open:
                if tn == "MeshRenderer" and comps_for_type:
                    self._render_multi_mesh_renderer(ctx, comps_for_type)
                elif comps_for_type:
                    comp_ui.render_component(ctx, comps_for_type[0])
            ctx.pop_id()

        # Render common Python components
        for tn in sorted(common_py_types):
            ctx.push_id_str(f"multi_py_{tn}")
            icon_id = self._get_component_icon_id(tn, is_script=True)
            pcs_for_type = per_obj_py.get(tn, [])

            enabled_vals = [pc.enabled for pc in pcs_for_type]
            all_enabled = all(enabled_vals)
            header_open, new_enabled = render_component_header(
                ctx, tn,
                icon_id=icon_id,
                show_enabled=True,
                is_enabled=all_enabled,
                suffix=" (Script)",
                force_open=True,
            )
            if new_enabled != all_enabled:
                for pc in pcs_for_type:
                    if pc.enabled != new_enabled:
                        _record_property(pc, "enabled", pc.enabled, new_enabled, f"Toggle {tn} (Multi)")

            if header_open and pcs_for_type:
                self._render_py_component(ctx, pcs_for_type[0])
            ctx.pop_id()

        # Add Component button (adds to all selected)
        ctx.separator()
        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.ADD_COMP_FRAME_PAD)
        ctx.set_cursor_pos_x(Theme.INSPECTOR_ACTION_ALIGN_X)
        ctx.button(t("inspector.add_component"), lambda: self._open_add_component_popup(ctx), -1, 0)
        ctx.pop_style_var(1)
        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)

        if ctx.begin_popup("##add_component_popup"):
            self._render_add_component_popup(ctx)
            ctx.end_popup()

        ctx.pop_id()

    def _render_properties_module(self, ctx: InfGUIContext, height: float):
        """Render the Properties module showing object properties (on top)."""
        # Lazily load component icons on first frame
        if not self.__comp_icons_loaded:
            native = self._get_native_engine()
            if native:
                self._load_component_icons(native)

        # Begin undo tracking frame — snapshots are captured before/after
        from InfEngine.engine.undo import UndoManager
        mgr = UndoManager.instance()
        self._undo_tracker.begin_frame()

        child_hovered = False
        child_visible = ctx.begin_child("PropertiesModule", 0, height, True)
        if child_visible:
            # Track hover inside child window for idle detection
            child_hovered = ctx.is_window_hovered(0)

            ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)

            # Multi-selection: delegate to multi-edit renderer
            from .selection_manager import SelectionManager
            sel = SelectionManager.instance()

            # Suppress old per-widget undo recording; the tracker handles it
            if mgr:
                with mgr.suppress_property_recording():
                    if sel.is_multi():
                        all_objects = self._get_all_selected_objects()
                        if len(all_objects) > 1:
                            self._render_multi_edit(ctx, all_objects)
                        else:
                            ctx.label(t("inspector.no_objects_selected"))
                    else:
                        selected_object = self._get_selected_object()
                        if selected_object:
                            self._render_single_object(ctx, selected_object)
                        else:
                            ctx.label(t("inspector.no_object_selected"))
            else:
                if sel.is_multi():
                    all_objects = self._get_all_selected_objects()
                    if len(all_objects) > 1:
                        self._render_multi_edit(ctx, all_objects)
                    else:
                        ctx.label(t("inspector.no_objects_selected"))
                else:
                    selected_object = self._get_selected_object()
                    if selected_object:
                        self._render_single_object(ctx, selected_object)
                    else:
                        ctx.label(t("inspector.no_object_selected"))

            ctx.pop_style_var(2)
        ctx.end_child()

        # End undo tracking frame — compare and record changes
        any_active = ctx.is_any_item_active()
        self._undo_tracker.end_frame(any_active)

        # Update idle-skip counter (available for future optimisation)
        if child_hovered or any_active:
            self._inspector_idle_frames = 0
        else:
            self._inspector_idle_frames += 1

        # Drag-drop target on the entire PropertiesModule child window.
        # Must be called AFTER end_child() — EndChild() submits the child as an item,
        # so BeginDragDropTarget() here applies to the whole child area.
        selected_object = self._get_selected_object()
        if selected_object is not None:
            from .igui import IGUI
            IGUI.drop_target(ctx, "SCRIPT_FILE", lambda p: self._handle_script_drop(p))
    
    def _render_splitter(self, ctx: InfGUIContext, total_height: float) -> float:
        """Render a horizontal splitter bar. Returns the new properties height ratio."""
        # Draw a visible separator line first
        ctx.separator()
        
        # Splitter bar - use full width invisible button
        avail_width = ctx.get_content_region_avail_width()
        
        # Create an invisible button that spans the splitter area
        # The button ID must be unique
        ctx.invisible_button("##InspectorSplitter", avail_width, self.SPLITTER_HEIGHT)
        
        is_hovered = ctx.is_item_hovered()
        is_active = ctx.is_item_active()
        
        # Change mouse cursor to resize style when hovered or active
        # ImGuiMouseCursor_ResizeNS = 3 (vertical resize)
        if is_hovered or is_active:
            ctx.set_mouse_cursor(3)  # ResizeNS cursor
        
        # Handle drag - check if button is being dragged
        if is_active:
            delta_y = ctx.get_mouse_drag_delta_y(0)
            if abs(delta_y) > 1.0:  # Small threshold to avoid jitter
                # Calculate new ratio
                usable_height = total_height - self.SPLITTER_HEIGHT
                if usable_height > 0:
                    # Delta is how much to move the splitter down (increase properties height)
                    new_properties_height = self.__properties_ratio * usable_height + delta_y
                    new_ratio = new_properties_height / usable_height
                    
                    # Clamp to valid range
                    min_ratio = self.MIN_PROPERTIES_HEIGHT / usable_height
                    max_ratio = 1.0 - (self.MIN_RAW_DATA_HEIGHT / usable_height)
                    self.__properties_ratio = max(min_ratio, min(max_ratio, new_ratio))
                    
                ctx.reset_mouse_drag_delta(0)
        
        # Draw another separator for visual feedback
        ctx.separator()
        
        return self.__properties_ratio
    
    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return Theme.INSPECTOR_INIT_SIZE

    def on_render_content(self, ctx: InfGUIContext):
        # Get total available height
        total_height = ctx.get_content_region_avail_height()
        
        # Show split view if file is selected alongside an object
        has_detail_content = bool(self.__selected_file)

        # When a file is selected and no object is active, give full
        # height to the file view (asset editor or generic preview).
        file_only = self.__selected_file and not self._get_selected_object()

        if file_only:
            # Full-height file view (asset inspector or generic preview)
            self._render_raw_data_module(ctx, 0)
        elif has_detail_content and total_height > (self.MIN_PROPERTIES_HEIGHT + self.MIN_RAW_DATA_HEIGHT + self.SPLITTER_HEIGHT):
            # Calculate heights based on ratio
            usable_height = total_height - self.SPLITTER_HEIGHT
            properties_height = usable_height * self.__properties_ratio
            raw_data_height = usable_height - properties_height
            
            # Clamp to minimums
            if properties_height < self.MIN_PROPERTIES_HEIGHT:
                properties_height = self.MIN_PROPERTIES_HEIGHT
                raw_data_height = usable_height - properties_height
            if raw_data_height < self.MIN_RAW_DATA_HEIGHT:
                raw_data_height = self.MIN_RAW_DATA_HEIGHT
                properties_height = usable_height - raw_data_height
            
            # 1. Properties module (TOP)
            self._render_properties_module(ctx, properties_height)
            
            # 2. Splitter bar
            self._render_splitter(ctx, total_height)
            
            # 3. Raw Data module (BOTTOM) - shows file preview or material detail
            self._render_raw_data_module(ctx, raw_data_height)
        else:
            # No file/material selected or not enough space - just show properties
            self._render_properties_module(ctx, 0)  # 0 = fill all
