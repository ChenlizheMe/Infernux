from __future__ import annotations

from typing import Any, List, Union

from Infernux.components.builtin_component import BuiltinComponent

class SpriteRenderer(BuiltinComponent):
    """Renders one frame of a sprite-sheet texture on a quad."""

    _cpp_type_name: str
    _component_category_: str
    _component_menu_path_: str

    sprite_guid: Any
    frame_index: Any
    sprite_color: Any
    flip_x: Any
    flip_y: Any

    @staticmethod
    def init_all_in_scene(scene: Any = ...) -> None:
        """Ensure wrappers exist for all C++ SpriteRenderers in the scene."""
        ...

    @property
    def sprite(self) -> str:
        """Asset GUID of the sprite texture."""
        ...
    @sprite.setter
    def sprite(self, value: Union[str, Any, None]) -> None: ...

    @property
    def material(self) -> Any:
        """Material on slot 0."""
        ...
    @material.setter
    def material(self, value: Any) -> None: ...

    @property
    def shared_material(self) -> Any:
        ...
    @shared_material.setter
    def shared_material(self, value: Any) -> None: ...

    @property
    def sprite_frames(self) -> List[Any]:
        ...
    @property
    def frame_count(self) -> int:
        ...

    def sync_visual(self) -> None:
        """Push frame / flip / color to the material after external updates."""
        ...

    def render_inspector(self, ctx: Any) -> None: ...
