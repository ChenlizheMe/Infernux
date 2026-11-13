"""Temporal anti-aliasing backed by per-view persistent history."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from infernux.components.fields import serialized_field
from infernux.renderstack._pipeline_common import COLOR_TEXTURE, DEPTH_TEXTURE, MOTION_TEXTURE
from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class TemporalAAEffect(FullScreenEffect):
    """Reproject, clamp, and accumulate the previous view-local HDR frame."""

    name = "Temporal Anti-Aliasing"
    injection_point = "before_post_process"
    default_order = 650
    menu_path = "Post-processing/Temporal Anti-Aliasing"

    requires = {COLOR_TEXTURE, DEPTH_TEXTURE, MOTION_TEXTURE}
    modifies = {COLOR_TEXTURE}

    feedback: float = serialized_field(
        default=0.9,
        range=(0.0, 0.98),
        drag_speed=0.005,
        slider=False,
        tooltip="Maximum contribution from the reprojected previous frame",
    )
    motion_rejection: float = serialized_field(
        default=0.08,
        range=(0.0, 1.0),
        drag_speed=0.005,
        slider=False,
        tooltip="Reduces history on fast-moving pixels",
    )
    depth_rejection: float = serialized_field(
        default=96.0,
        range=(0.0, 512.0),
        drag_speed=1.0,
        slider=False,
        tooltip="Reduces history around depth discontinuities",
    )

    def get_shader_list(self) -> List[str]:
        return ["infernux/post/temporal_aa"]

    def setup_passes(self, graph: "RenderGraph", bus: "ResourceBus") -> None:
        from infernux.rendergraph.graph import Format

        color_in = bus.get(COLOR_TEXTURE)
        depth = bus.get(DEPTH_TEXTURE)
        motion = bus.get(MOTION_TEXTURE)
        if color_in is None or depth is None or motion is None:
            missing = [
                name
                for name, handle in (
                    (COLOR_TEXTURE, color_in),
                    (DEPTH_TEXTURE, depth),
                    (MOTION_TEXTURE, motion),
                )
                if handle is None
            ]
            raise ValueError(
                "Temporal Anti-Aliasing requires effect-stage resources: "
                + ", ".join(missing)
            )

        graph.set_temporal_jitter()
        history_read, history_write = graph.create_temporal_history(
            "_taa_history",
            format=Format.RGBA16_SFLOAT,
        )
        color_out = self.get_or_create_texture(
            graph,
            "_taa_out",
            format=Format.RGBA16_SFLOAT,
        )
        with graph.add_pass("TAA_Resolve") as render_pass:
            render_pass.set_textures(
                {
                    "source_tex": color_in,
                    "history_tex": history_read,
                    "motion_tex": motion,
                    "depth_tex": depth,
                }
            )
            render_pass.write_color(color_out)
            render_pass.set_param("feedback", self.feedback)
            render_pass.set_param("motion_rejection", self.motion_rejection)
            render_pass.set_param("depth_rejection", self.depth_rejection)
            render_pass.fullscreen_quad("infernux/post/temporal_aa")

        with graph.add_copy_pass("TAA_CommitHistory") as commit:
            commit.copy_texture(color_out, history_write)
            commit.set_side_effect()

        bus.set(COLOR_TEXTURE, color_out)
