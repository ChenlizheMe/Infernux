"""Camera and object motion blur backed by the shared motion-vector target."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from infernux.components.fields import serialized_field
from infernux.renderstack._pipeline_common import COLOR_TEXTURE, DEPTH_TEXTURE, MOTION_TEXTURE
from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class MotionBlurEffect(FullScreenEffect):
    """Depth-aware screen-space blur driven by per-pixel UV motion vectors."""

    name = "Motion Blur"
    injection_point = "before_post_process"
    default_order = 700
    menu_path = "Post-processing/Motion Blur"

    requires = {COLOR_TEXTURE, DEPTH_TEXTURE, MOTION_TEXTURE}
    modifies = {COLOR_TEXTURE}

    intensity: float = serialized_field(
        default=1.0,
        range=(0.0, 2.0),
        drag_speed=0.01,
        slider=False,
        tooltip="Shutter strength applied to camera and object motion",
    )
    max_blur_pixels: float = serialized_field(
        default=32.0,
        range=(0.0, 128.0),
        drag_speed=0.5,
        slider=False,
        tooltip="Maximum blur radius in output pixels",
    )
    depth_rejection: float = serialized_field(
        default=1.0,
        range=(0.0, 8.0),
        drag_speed=0.05,
        slider=False,
        tooltip="Suppresses samples crossing depth discontinuities",
    )

    def get_shader_list(self) -> List[str]:
        return ["infernux/post/motion_blur"]

    def setup_passes(self, graph: "RenderGraph", bus: "ResourceBus") -> None:
        from infernux.rendergraph.graph import Format

        color_in = bus.require_texture(COLOR_TEXTURE)
        depth = bus.require_texture(DEPTH_TEXTURE)
        motion = bus.require_texture(MOTION_TEXTURE)

        color_out = self.get_or_create_texture(
            graph,
            "_motion_blur_out",
            format=Format.RGBA16_SFLOAT,
        )
        with graph.add_pass("MotionBlur_Apply") as render_pass:
            render_pass.set_textures(
                {
                    "source_tex": color_in,
                    "motion_tex": motion,
                    "depth_tex": depth,
                }
            )
            render_pass.write_color(color_out)
            render_pass.set_param("intensity", self.intensity)
            render_pass.set_param("max_blur_pixels", self.max_blur_pixels)
            render_pass.set_param("depth_rejection", self.depth_rejection)
            render_pass.fullscreen_quad("infernux/post/motion_blur")

        bus.set(COLOR_TEXTURE, color_out)
