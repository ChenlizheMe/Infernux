"""
SpriteRenderer — renders a single frame from a sprite-sheet texture.

Wraps the C++ ``SpriteRenderer`` component (which inherits from
``MeshRenderer`` for rendering pipeline compatibility) and manages the
``sprite_unlit`` material, UV rect, and texture binding from Python.

This component is completely independent of the Python ``MeshRenderer``
wrapper — the two are parallel, same-level renderer types.
"""

from __future__ import annotations

from typing import List, Optional

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
from Infernux.components.serialized_field import FieldType
from Infernux.debug import Debug


def _to_native_material(value):
    """Unwrap a Python Material wrapper to native InxMaterial."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    native = getattr(value, "_native", None) or getattr(value, "native", None)
    return native if native is not None else value


def _sprite_color_to_list(t):
    """C++ sprite_color tuple → [r,g,b,a] list for COLOR field."""
    return [t[0], t[1], t[2], t[3]]


def _list_to_sprite_color(lst):
    """[r,g,b,a] list → tuple for C++ sprite_color."""
    c = lst if lst and len(lst) >= 4 else [1, 1, 1, 1]
    return (c[0], c[1], c[2], c[3])


def _get_asset_database():
    """Match the Material Inspector's asset database lookup order."""
    try:
        from Infernux.engine.ui.editor_services import EditorServices
        adb = EditorServices.instance()._asset_database
        if adb:
            return adb
    except Exception:
        pass
    try:
        from Infernux.lib import AssetRegistry
        return AssetRegistry.instance().get_asset_database()
    except Exception:
        return None


class SpriteRenderer(BuiltinComponent):
    """Renders one frame of a sprite-sheet texture on a Quad mesh.

    Wraps the C++ ``SpriteRenderer`` component.  Properties delegate to C++
    for serialization; material management (shader, texture, UV) is handled
    in this Python wrapper.
    """

    _cpp_type_name = "SpriteRenderer"
    _component_category_ = "Rendering"
    _component_menu_path_ = "Rendering/Sprite Renderer"

    # ── CppProperty descriptors (scalar fields → C++) ───────────────
    #
    # IMPORTANT: Python descriptor names MUST match ``cpp_attr`` so that
    # ``_record_builtin_property(comp, cpp_attr, ...)`` → ``setattr``
    # goes through the CppProperty descriptor and reaches C++.

    sprite_guid = CppProperty(
        "sprite_guid",
        FieldType.STRING,
        default="",
    )

    frame_index = CppProperty(
        "frame_index",
        FieldType.INT,
        default=0,
        range=(0, 9999),
        tooltip="Index of the frame to display",
        visible_when=lambda comp: not comp._is_driven_by_animator(),
    )

    sprite_color = CppProperty(
        "sprite_color",
        FieldType.COLOR,
        default=None,
        tooltip="Tint color (RGBA)",
        get_converter=_sprite_color_to_list,
        set_converter=_list_to_sprite_color,
    )

    flip_x = CppProperty(
        "flip_x",
        FieldType.BOOL,
        default=False,
        tooltip="Flip sprite horizontally",
    )

    flip_y = CppProperty(
        "flip_y",
        FieldType.BOOL,
        default=False,
        tooltip="Flip sprite vertically",
    )

    # ── Private runtime state ───────────────────────────────────────

    _sprite_material = None
    _sprite_frames: list = []
    _tex_w: int = 0
    _tex_h: int = 0
    _last_frame_index: int = -1
    _last_flip_x: bool = False
    _last_flip_y: bool = False
    _last_color: tuple = None
    _last_sprite: str = ""
    _material_ready: bool = False
    _instance_counter: int = 0  # class-level counter for unique material names

    # ── Binding hook ────────────────────────────────────────────────

    def _bind_cpp(self, cpp_component, game_object):
        super()._bind_cpp(cpp_component, game_object)
        # Reset per-instance state (class-level defaults are shared).
        self._sprite_frames = []
        self._tex_w = 0
        self._tex_h = 0
        self._last_frame_index = -1
        self._last_flip_x = False
        self._last_flip_y = False
        self._last_color = None
        self._last_sprite = ""
        self._material_ready = False
        self._sprite_material = None
        self._ensure_material()
        self._subscribe_asset_events()

    # ── Asset-change notification ───────────────────────────────────

    def _subscribe_asset_events(self):
        """Subscribe to ASSET_CHANGED so texture reimport refreshes this renderer."""
        try:
            from Infernux.engine.ui.event_bus import EditorEventBus, EditorEvent
            bus = EditorEventBus.instance()
            # Avoid duplicate subscriptions
            bus.unsubscribe(EditorEvent.ASSET_CHANGED, self._on_asset_changed)
            bus.subscribe(EditorEvent.ASSET_CHANGED, self._on_asset_changed)
        except Exception:
            pass

    def _on_asset_changed(self, file_path: str, event_type: str = "modified"):
        """Called when any asset file is modified/deleted on disk."""
        guid = self.sprite
        if not guid:
            return
        try:
            adb = _get_asset_database()
            if not adb:
                return
            asset_path = adb.get_path_from_guid(guid)
            if not asset_path:
                return
            # Normalize both paths for comparison
            import os
            norm_asset = os.path.normpath(asset_path).lower()
            norm_changed = os.path.normpath(file_path).lower()
            # Also check if the .meta was modified
            if norm_changed == norm_asset or norm_changed == norm_asset + ".meta":
                Debug.log_internal(f"SpriteRenderer: asset changed, refreshing texture")
                self._load_sprite_data()
                self._apply_uv_rect()
                self._apply_color()
        except Exception:
            pass

    # ── Scene-wide initialization ───────────────────────────────────

    @staticmethod
    def init_all_in_scene(scene=None):
        """Force wrapper creation for all SpriteRenderers in the scene.

        This ensures each SpriteRenderer gets its own material with the
        correct texture binding *before* the first render frame, avoiding
        the white-quad-until-clicked problem.
        """
        try:
            if scene is None:
                from Infernux.lib import SceneManager
                scene = SceneManager.instance().get_active_scene()
            if scene is None:
                return
            all_objects = scene.get_all_objects()
            count = 0
            for obj in all_objects:
                try:
                    cpp_comp = obj.get_component("SpriteRenderer")
                    if cpp_comp is not None:
                        SpriteRenderer._get_or_create_wrapper(cpp_comp, obj)
                        count += 1
                except Exception:
                    pass
            if count > 0:
                Debug.log_internal(f"SpriteRenderer: initialized {count} instance(s)")
        except Exception as e:
            Debug.log_warning(f"SpriteRenderer.init_all_in_scene failed: {e}")

    # ── Sprite GUID (wraps C++ string, exposes as TEXTURE for Inspector) ──

    @property
    def sprite(self) -> str:
        """Asset GUID of the sprite texture."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.sprite_guid or ""
        return ""

    @sprite.setter
    def sprite(self, value):
        guid = self._extract_guid(value)
        cpp = self._cpp_component
        if cpp is not None:
            cpp.sprite_guid = guid
        self._load_sprite_data()
        self._apply_uv_rect()

    # ── Material access (direct to C++ SpriteRenderer) ──────────────

    @property
    def material(self):
        """The material on slot 0."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material(0)
        return self._sprite_material

    @material.setter
    def material(self, value):
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material(0, _to_native_material(value))

    @property
    def shared_material(self):
        return self.material

    @shared_material.setter
    def shared_material(self, value):
        self.material = value

    # ── Public API ──────────────────────────────────────────────────

    @property
    def sprite_frames(self) -> list:
        """Currently loaded sprite frames (read-only at runtime)."""
        return list(self._sprite_frames)

    @property
    def frame_count(self) -> int:
        return len(self._sprite_frames)

    # ── Custom Inspector rendering ──────────────────────────────────

    def render_inspector(self, ctx):
        """Custom Inspector: texture picker + material + color bar + CppProperty fields."""
        from Infernux.engine.ui.inspector_components import (
            render_builtin_via_setters, field_label, max_label_w,
            render_object_field, _record_builtin_property,
        )
        from Infernux.engine.ui.inspector_utils import _render_color_bar

        labels = ["Sprite", "Material", "Color", "Frame Index", "Flip X", "Flip Y"]
        lw = max_label_w(ctx, labels)

        # ── Sprite texture picker ──────────────────────────────
        guid = self.sprite
        display = "None (Texture)"
        if guid:
            try:
                adb = _get_asset_database()
                path = adb.get_path_from_guid(guid) if adb else ""
                if path:
                    import os
                    display = os.path.basename(path)
            except Exception:
                display = guid[:8] + "…" if len(guid) > 8 else guid

        def _sprite_asset_items(filt):
            from Infernux.engine.ui.inspector_components import _picker_assets
            return (
                _picker_assets(filt, "*.png", assets_only=True)
                + _picker_assets(filt, "*.jpg", assets_only=True)
                + _picker_assets(filt, "*.jpeg", assets_only=True)
                + _picker_assets(filt, "*.tga", assets_only=True)
                + _picker_assets(filt, "*.bmp", assets_only=True)
                + _picker_assets(filt, "*.gif", assets_only=True)
                + _picker_assets(filt, "*.psd", assets_only=True)
                + _picker_assets(filt, "*.hdr", assets_only=True)
                + _picker_assets(filt, "*.pic", assets_only=True)
            )

        field_label(ctx, "Sprite", lw)
        render_object_field(
            ctx,
            "##sprite_texture",
            display,
            "Texture",
            accept_drag_type="TEXTURE_FILE",
            on_drop_callback=self._on_sprite_drop,
            picker_asset_items=_sprite_asset_items,
            on_pick=self._on_sprite_pick,
            on_clear=self._on_sprite_clear,
        )

        # ── Material slot (supports custom materials) ──────────
        mat = self._get_material()
        mat_display = "Default (sprite_unlit)"
        if mat is not None and not self._is_default_material(mat):
            mat_name = getattr(mat, 'name', None) or getattr(mat, 'path', None)
            if mat_name:
                import os
                mat_display = os.path.basename(str(mat_name))
            else:
                mat_display = "Custom Material"
        field_label(ctx, "Material", lw)
        render_object_field(
            ctx,
            "##sprite_material",
            mat_display,
            "Material",
            accept_drag_type="MATERIAL_FILE",
            on_drop_callback=self._on_material_drop,
            on_clear=self._on_material_clear,
        )

        # ── Color (Unity-style color bar, same as Material Inspector) ──
        c = self.sprite_color
        if c is None or len(c) < 4:
            c = [1.0, 1.0, 1.0, 1.0]
        field_label(ctx, "Color", lw)
        nr, ng, nb, na = _render_color_bar(
            ctx, "##sprite_color", c[0], c[1], c[2], c[3])
        if (nr, ng, nb, na) != (c[0], c[1], c[2], c[3]):
            _record_builtin_property(
                self, "sprite_color", c, [nr, ng, nb, na], "Set color")
            self._apply_color()

        # ── Remaining CppProperty fields (frame_index, flip_x, flip_y) ──
        render_builtin_via_setters(
            ctx, self, type(self),
            skip_fields={'sprite_guid', 'sprite_color'})

        # ── Sync material state after Inspector edits ──────────
        self._sync_material_if_dirty()

    def _on_sprite_drop(self, payload):
        from Infernux.engine.ui.inspector_components import _record_builtin_property
        dropped = str(payload).replace("\\", "/")
        Debug.log(f"SpriteRenderer: drop payload = {dropped}")
        old = self.sprite_guid
        guid = self._resolve_texture_guid(dropped)
        Debug.log(f"SpriteRenderer: resolved GUID = {guid!r}")
        if guid:
            _record_builtin_property(self, "sprite_guid", old, guid, "Set sprite")
            self._load_sprite_data()
            self._apply_uv_rect()
        else:
            Debug.log_warning(f"SpriteRenderer: failed to resolve GUID for: {dropped}")

    def _on_sprite_pick(self, picked_path):
        self._on_sprite_drop(picked_path)

    def _on_sprite_clear(self):
        from Infernux.engine.ui.inspector_components import _record_builtin_property
        old = self.sprite_guid
        _record_builtin_property(self, "sprite_guid", old, "", "Clear sprite")
        self._sprite_frames = []
        self._tex_w = 0
        self._tex_h = 0

    def _on_material_drop(self, payload):
        mat_path = str(payload)
        try:
            from Infernux.core.material import Material
            mat = Material.load(mat_path)
            if mat is not None:
                cpp = self._cpp_component
                if cpp is not None:
                    cpp.set_material(0, mat._native)
                    self._sprite_material = mat._native
                    self._apply_texture_to_material()
                    self._apply_uv_rect()
                    self._apply_color()
        except Exception as e:
            Debug.log_warning(f"SpriteRenderer: failed to set material: {e}")

    def _on_material_clear(self):
        """Reset to default sprite_unlit material."""
        cpp = self._cpp_component
        if cpp is None:
            return
        cpp.set_material(0, None)
        self._sprite_material = None
        self._material_ready = False
        self._ensure_material()

    def _is_default_material(self, mat):
        """Check if a material is the auto-created sprite_unlit default."""
        try:
            frag = getattr(mat, 'frag_shader_name', None)
            # Material wrapper has no 'path' property — check native file_path
            native = getattr(mat, '_native', None) or getattr(mat, 'native', mat)
            path = getattr(native, 'file_path', '') or ''
            # Default material has no saved path and uses sprite_unlit
            return frag == 'sprite_unlit' and not path
        except Exception:
            return False

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _extract_guid(value) -> str:
        """Extract a GUID string from various input types."""
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        guid = getattr(value, "guid", None)
        if guid:
            return guid
        return str(value)

    @staticmethod
    def _resolve_texture_guid(path_str: str) -> str:
        """Resolve a file path to an asset GUID using the editor-aware asset DB."""
        if not path_str:
            return ""
        try:
            adb = _get_asset_database()
            if not adb:
                return ""

            candidates = [path_str]
            normalized = path_str.replace("\\", "/")
            if normalized not in candidates:
                candidates.append(normalized)

            try:
                import os
                normpath = os.path.normpath(path_str)
                if normpath not in candidates:
                    candidates.append(normpath)
                slash_norm = normpath.replace("\\", "/")
                if slash_norm not in candidates:
                    candidates.append(slash_norm)
            except Exception:
                pass

            lowered = []
            for candidate in candidates:
                if isinstance(candidate, str):
                    low = candidate.lower()
                    if low not in candidates and low not in lowered:
                        lowered.append(low)
            candidates.extend(lowered)

            for candidate in candidates:
                guid = adb.get_guid_from_path(candidate)
                if guid:
                    return guid
        except Exception:
            pass
        return ""

    def sync_visual(self):
        """Public API: push the current C++ properties (frame, flip, color)
        to the material.  Called by external drivers like SpiritAnimator
        after they update ``frame_index`` from Python."""
        self._sync_material_if_dirty()

    def _is_driven_by_animator(self) -> bool:
        """Return True if a SpiritAnimator is attached to this GameObject."""
        try:
            go = self.game_object
            if go is None:
                return False
            from Infernux.components.animator2d import SpiritAnimator
            return go.get_component(SpiritAnimator) is not None
        except Exception:
            return False

    def _sync_material_if_dirty(self):
        """Push changed CppProperty values to the material (called per Inspector frame)."""
        cpp = self._cpp_component
        if cpp is None:
            return

        guid = self.sprite
        fi = cpp.frame_index
        fx = cpp.flip_x
        fy = cpp.flip_y
        try:
            c = tuple(cpp.sprite_color)
        except Exception:
            c = (1, 1, 1, 1)

        uv_dirty = (
            fi != self._last_frame_index
            or fx != self._last_flip_x
            or fy != self._last_flip_y
            or guid != self._last_sprite
        )
        color_dirty = c != self._last_color

        if guid != self._last_sprite:
            self._load_sprite_data()

        if uv_dirty:
            self._apply_uv_rect()
        if color_dirty:
            self._apply_color()

    def _get_material(self):
        """Get the Python Material wrapper for slot 0."""
        cpp = self._cpp_component
        if cpp is None:
            return None
        native = cpp.get_material(0)
        if native is None:
            return None
        from Infernux.core.material import Material
        if isinstance(native, Material):
            return native
        return Material.from_native(native)

    def _ensure_material(self):
        """Create the default sprite_unlit material if none is assigned."""
        cpp = self._cpp_component
        if cpp is None:
            return
        existing = cpp.get_material(0)
        if existing is not None:
            self._sprite_material = existing
            self._material_ready = True
            # Reload sprite data in case we're restoring from a scene
            self._load_sprite_data()
            self._apply_uv_rect()
            self._apply_color()
            return
        try:
            from Infernux.core.material import Material
            mat = Material.create_unlit()
            mat.frag_shader_name = "sprite_unlit"
            # Opaque + alpha clipping: sprites are rendered in the opaque
            # queue with hard-edge alpha test (no blending artefacts).
            mat.surface_type = "opaque"
            mat.alpha_clip_enabled = True
            mat.alpha_clip_threshold = 0.5
            # Give each instance a unique name so GetMaterialKey() returns a
            # unique key — the renderer shares UBO/descriptor data per key,
            # so without this all SpriteRenderers would share one set of
            # material properties (color, texture, uvRect).
            SpriteRenderer._instance_counter += 1
            mat._native.name = f"SpriteUnlit_Inst{SpriteRenderer._instance_counter}"
            mat.set_color("baseColor", 1.0, 1.0, 1.0, 1.0)
            mat.set_vector4("uvRect", 0.0, 0.0, 1.0, 1.0)
            self._sprite_material = mat._native
            self._material_ready = True
            # Bind the sprite texture BEFORE assigning the material to the
            # C++ component.  This ensures the descriptor set that C++ creates
            # on set_material() already contains the real texture, avoiding
            # a one-frame flash of the white fallback texture.
            self._load_sprite_data()
            self._apply_uv_rect()
            self._apply_color()
            cpp.set_material(0, mat._native)
        except Exception as e:
            Debug.log_warning(f"SpriteRenderer: failed to create material: {e}")

    def _load_sprite_data(self):
        """Load sprite frame list and texture dimensions from the asset .meta."""
        self._sprite_frames = []
        self._tex_w = 0
        self._tex_h = 0

        guid = self.sprite
        self._last_sprite = guid
        if not guid:
            self._apply_texture_to_material()
            return

        try:
            adb = _get_asset_database()
            if not adb:
                self._apply_texture_to_material()
                return
            asset_path = adb.get_path_from_guid(guid)
            if not asset_path:
                self._apply_texture_to_material()
                return

            from Infernux.core.asset_types import read_meta_file
            meta = read_meta_file(asset_path)
            if meta is None:
                self._apply_texture_to_material()
                return

            self._tex_w = int(meta.get("width", 0))
            self._tex_h = int(meta.get("height", 0))

            # Only load sprite frames if texture_type is "sprite"
            tex_type = meta.get("texture_type", "default")
            if tex_type == "sprite":
                raw_frames = meta.get("sprite_frames", [])
                if isinstance(raw_frames, str):
                    import json
                    raw_frames = json.loads(raw_frames)

                from Infernux.core.asset_types import SpriteFrame
                self._sprite_frames = [
                    SpriteFrame.from_dict(f) if isinstance(f, dict) else f
                    for f in raw_frames
                ]

            # Assign the texture to the material (if it supports texSampler)
            self._apply_texture_to_material()
        except Exception as e:
            Debug.log_warning(f"SpriteRenderer: failed to load sprite data: {e}")

    def _apply_texture_to_material(self):
        """Pass the sprite texture to texSampler (sprite_unlit shader slot)."""
        guid = self.sprite
        mat = self._get_material()
        if mat is None:
            return
        try:
            # sprite_unlit.frag uses "texSampler" — set it directly without
            # has_property() since programmatic materials may not have the
            # property registered in m_properties until first set_texture call.
            if guid:
                mat.set_texture("texSampler", guid)
            else:
                mat.clear_texture("texSampler")
        except Exception as e:
            Debug.log_warning(f"SpriteRenderer: _apply_texture_to_material failed: {e}")

    def _apply_uv_rect(self):
        """Compute and apply UV rect and display scale from the current frame."""
        cpp = self._cpp_component
        if cpp is None:
            return

        fi = cpp.frame_index
        fx = cpp.flip_x
        fy = cpp.flip_y
        self._last_frame_index = fi
        self._last_flip_x = fx
        self._last_flip_y = fy
        self._last_sprite = self.sprite

        mat = self._get_material()
        if mat is None:
            return

        # Default: full texture
        u, v, su, sv = 0.0, 0.0, 1.0, 1.0
        ds_x, ds_y = 1.0, 1.0  # displayScale for aspect-fit centering

        if self._sprite_frames and self._tex_w > 0 and self._tex_h > 0:
            idx = fi % len(self._sprite_frames)
            frame = self._sprite_frames[idx]
            tw, th = float(self._tex_w), float(self._tex_h)
            u = frame.x / tw
            v = frame.y / th
            su = frame.w / tw
            sv = frame.h / th
            fw = float(frame.w) if frame.w > 0 else 1.0
            fh = float(frame.h) if frame.h > 0 else 1.0
            max_dim = max(fw, fh)
            ds_x = fw / max_dim
            ds_y = fh / max_dim

        if fx:
            u = u + su
            su = -su
        if not fy:
            # Default (flip_y=False): invert V to correct Vulkan UV orientation
            v = v + sv
            sv = -sv

        try:
            mat.set_vector4("uvRect", u, v, su, sv)
        except Exception:
            pass

        # displayScale tells the shader what fraction of the quad the sprite
        # occupies.  The shader centers the image and discards outside pixels.
        try:
            mat.set_vector4("displayScale", ds_x, ds_y, 0.0, 0.0)
        except Exception:
            pass

    def _apply_color(self):
        """Apply tint color to the material."""
        cpp = self._cpp_component
        if cpp is None:
            return

        try:
            c = cpp.sprite_color
            c = (c[0], c[1], c[2], c[3])
        except Exception:
            c = (1, 1, 1, 1)

        self._last_color = c
        mat = self._get_material()
        if mat is None:
            return

        try:
            mat.set_color("baseColor", c[0], c[1], c[2], c[3])
        except Exception:
            pass
