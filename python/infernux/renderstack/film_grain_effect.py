"""Film grain applied to the display exposure with a luminance response."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect
from infernux.components.fields import serialized_field

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class FilmGrainEffect(FullScreenEffect):
    """Apply a centered grain plate after tone mapping."""

    name = "Film Grain"
    injection_point = "after_post_process"
    # After tone mapping (900): grain modulates the final displayed image,
    # matching real film where grain lives in the print, not the scene light.
    default_order = 950
    menu_path = "Post-processing/Film Grain"

    intensity: float = serialized_field(default=0.2, range=(0.0, 1.0), slider=False)
    response: float = serialized_field(default=0.8, range=(0.0, 1.0), slider=False)
    size: float = serialized_field(
        default=1.0,
        range=(0.5, 4.0),
        slider=False,
        tooltip="Grain size in pixels (1 = fine modern stock, 3+ = coarse vintage)",
    )
    colored: bool = serialized_field(
        default=False,
        tooltip="Add a subtle independent grain component to each color channel",
    )

    def get_shader_list(self) -> List[str]:
        return ["Fullscreen Triangle", "Film Grain"]

    def setup_passes(self, graph: "RenderGraph", bus: "ResourceBus") -> None:
        from infernux.rendergraph.graph import Format

        self.apply_single_source_effect(
            graph,
            bus,
            output_name="_filmgrain_out",
            pass_name="FilmGrain_Apply",
            shader_name="Film Grain",
            format=Format.RGBA16_SFLOAT,
            params={
                "intensity": self.intensity,
                "response": self.response,
                "size": self.size,
                "colored": 1.0 if self.colored else 0.0,
            },
        )
