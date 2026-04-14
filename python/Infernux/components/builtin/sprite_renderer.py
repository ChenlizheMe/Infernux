"""
SpriteRenderer — renders a single frame from a sprite-sheet texture.

Internally drives the ``MeshRenderer`` on the same ``GameObject`` with a
Quad mesh and a ``sprite_unlit`` material whose ``uvRect`` property is
updated whenever the visible frame changes.

The user can override the material (via the MeshRenderer Inspector or API)
just like any other renderer.
"""

from __future__ import annotations

from typing import List, Optional

from Infernux.components.component import InxComponent
from Infernux.components.serialized_field import serialized_field, FieldType
from Infernux.components.decorators import add_component_menu, execute_in_edit_mode


@add_component_menu("Rendering/Sprite Renderer")
@execute_in_edit_mode
class SpriteRenderer(InxComponent):
    """Renders one frame of a sprite-sheet texture on a Quad mesh."""

    _component_category_ = "Rendering"

    # ── Serialized fields (auto-rendered in Inspector) ──────────────

    sprite: str = serialized_field(
        default="",
        field_type=FieldType.TEXTURE,
        tooltip="Sprite texture (must be Sprite type with sliced frames)",
        group="Sprite",
    )
    frame_index: int = serialized_field(
        default=0,
        range=(0, 9999),
        tooltip="Index of the frame to display",
        group="Sprite",
    )
    color: list = serialized_field(
        default=[1.0, 1.0, 1.0, 1.0],
        field_type=FieldType.COLOR,
        tooltip="Tint color (RGBA)",
        group="Sprite",
    )
    flip_x: bool = serialized_field(default=False, tooltip="Flip sprite horizontally", group="Sprite")
    flip_y: bool = serialized_field(default=False, tooltip="Flip sprite vertically", group="Sprite")

    # ── Private runtime state ───────────────────────────────────────

    _mesh_renderer = None
    _sprite_material = None
    _sprite_frames: list = []
    _tex_w: int = 0
    _tex_h: int = 0
    _last_frame_index: int = -1
    _last_flip_x: bool = False
    _last_flip_y: bool = False
    _last_color: list = None
    _last_sprite: str = ""
    _initialized: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────

    def reset(self):
        """Called when component is first attached (editor only)."""
        self._setup()

    def awake(self):
        self._setup()

    def on_validate(self):
        """Called when Inspector fields change in edit mode."""
        self._setup()

    def start(self):
        self._setup()

    def _setup(self):
        """Ensure mesh, material, and sprite state are up to date."""
        self._ensure_mesh_renderer()
        self._load_sprite_data()
        self._apply_uv_rect()
        self._apply_color()
        self._initialized = True

    def update(self, delta_time: float):
        if not self._initialized:
            self._setup()
            return

        # Reload sprite data if the texture reference changed
        if self.sprite != self._last_sprite:
            self._load_sprite_data()

        needs_uv = (
            self.frame_index != self._last_frame_index
            or self.flip_x != self._last_flip_x
            or self.flip_y != self._last_flip_y
            or self.sprite != self._last_sprite
        )
        if needs_uv:
            self._apply_uv_rect()

        if self.color != self._last_color:
            self._apply_color()

    # ── Material access (delegates to MeshRenderer) ─────────────────

    @property
    def material(self):
        """The material on the underlying MeshRenderer (slot 0)."""
        mr = self._get_mesh_renderer()
        if mr is not None:
            return mr.material
        return self._sprite_material

    @material.setter
    def material(self, value):
        mr = self._get_mesh_renderer()
        if mr is not None:
            mr.material = value

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

    # ── Internals ───────────────────────────────────────────────────

    def _get_mesh_renderer(self):
        if self._mesh_renderer is not None:
            return self._mesh_renderer
        try:
            from Infernux.components.builtin import MeshRenderer
            self._mesh_renderer = self.game_object.get_component(MeshRenderer)
        except Exception:
            pass
        return self._mesh_renderer

    def _ensure_mesh_renderer(self):
        """Make sure the GO has a MeshRenderer with a Quad mesh."""
        mr = self._get_mesh_renderer()
        if mr is None:
            try:
                cpp_mr = self.game_object.get_cpp_component("MeshRenderer")
                if cpp_mr is None:
                    cpp_mr = self.game_object.add_component("MeshRenderer")
                if cpp_mr is not None:
                    from Infernux.lib import PrimitiveType
                    cpp_mr.set_primitive_mesh(PrimitiveType.Quad)
                from Infernux.components.builtin import MeshRenderer
                self._mesh_renderer = self.game_object.get_component(MeshRenderer)
            except Exception:
                pass

        # Ensure Quad mesh is set (may be missing if GO existed before)
        mr = self._get_mesh_renderer()
        if mr is not None and not mr._cpp_component.has_inline_mesh():
            try:
                from Infernux.lib import PrimitiveType
                mr._cpp_component.set_primitive_mesh(PrimitiveType.Quad)
            except Exception:
                pass

        # Set up the default sprite material if none is assigned
        if mr is not None and mr.material is None:
            self._create_default_material()
        elif mr is not None and self._sprite_material is None:
            self._sprite_material = mr.material

    def _create_default_material(self):
        """Create a transparent sprite_unlit material and assign it."""
        try:
            from Infernux.core.material import Material
            mat = Material.create_unlit("Sprite Material")
            mat.set_shader("sprite_unlit")
            mat.surface_type = "transparent"
            mat.set_color("baseColor", 1.0, 1.0, 1.0, 1.0)
            mat.set_vector4("uvRect", 0.0, 0.0, 1.0, 1.0)
            self._sprite_material = mat
            mr = self._get_mesh_renderer()
            if mr is not None:
                mr.material = mat
        except Exception as e:
            from Infernux.debug import Debug
            Debug.log_warning(f"SpriteRenderer: failed to create material: {e}")

    def _load_sprite_data(self):
        """Load sprite frame list and texture dimensions from the asset .meta."""
        self._sprite_frames = []
        self._tex_w = 0
        self._tex_h = 0
        self._last_sprite = self.sprite

        if not self.sprite:
            return

        try:
            from Infernux.lib import AssetRegistry
            adb = AssetRegistry.instance().get_asset_database()
            asset_path = adb.get_path_from_guid(self.sprite)
            if not asset_path:
                return

            from Infernux.core.asset_types import read_meta_file
            meta = read_meta_file(asset_path)
            if meta is None:
                return

            self._tex_w = int(meta.get("width", 0))
            self._tex_h = int(meta.get("height", 0))

            raw_frames = meta.get("sprite_frames", [])
            if isinstance(raw_frames, str):
                import json
                raw_frames = json.loads(raw_frames)

            from Infernux.core.asset_types import SpriteFrame
            self._sprite_frames = [
                SpriteFrame.from_dict(f) if isinstance(f, dict) else f
                for f in raw_frames
            ]

            # Also assign the texture to the material
            mr = self._get_mesh_renderer()
            if mr is not None and mr.material is not None:
                try:
                    from Infernux.core.material import Material
                    mat = Material.from_native(mr.material) if not isinstance(mr.material, Material) else mr.material
                    mat.set_texture_guid("texSampler", self.sprite)
                except Exception:
                    pass
        except Exception as e:
            from Infernux.debug import Debug
            Debug.log_warning(f"SpriteRenderer: failed to load sprite data: {e}")

    def _apply_uv_rect(self):
        """Compute and apply UV rect from the current frame_index."""
        self._last_frame_index = self.frame_index
        self._last_flip_x = self.flip_x
        self._last_flip_y = self.flip_y
        self._last_sprite = self.sprite

        mr = self._get_mesh_renderer()
        if mr is None or mr.material is None:
            return

        # Default: full texture
        u, v, su, sv = 0.0, 0.0, 1.0, 1.0

        if self._sprite_frames and self._tex_w > 0 and self._tex_h > 0:
            idx = max(0, min(self.frame_index, len(self._sprite_frames) - 1))
            frame = self._sprite_frames[idx]
            tw, th = float(self._tex_w), float(self._tex_h)
            u = frame.x / tw
            v = frame.y / th
            su = frame.w / tw
            sv = frame.h / th

        # Handle flipping
        if self.flip_x:
            u = u + su
            su = -su
        if self.flip_y:
            v = v + sv
            sv = -sv

        try:
            from Infernux.core.material import Material
            mat = mr.material
            if not isinstance(mat, Material):
                mat = Material.from_native(mat)
            mat.set_vector4("uvRect", u, v, su, sv)
        except Exception:
            pass

    def _apply_color(self):
        """Apply tint color to the material."""
        self._last_color = list(self.color) if self.color else [1, 1, 1, 1]

        mr = self._get_mesh_renderer()
        if mr is None or mr.material is None:
            return

        c = self.color if self.color and len(self.color) >= 4 else [1, 1, 1, 1]
        try:
            from Infernux.core.material import Material
            mat = mr.material
            if not isinstance(mat, Material):
                mat = Material.from_native(mat)
            mat.set_color("baseColor", c[0], c[1], c[2], c[3])
        except Exception:
            pass
