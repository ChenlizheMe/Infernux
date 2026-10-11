"""GameBuilder — cook and package an Infernux project as a standalone Player.

Uses the platform's precompiled Player runtime and packages imported assets,
runtime modules, splash data and the ordered build scene list.

Example::

    builder = GameBuilder(
        project_path="/path/to/project",
        output_dir="/path/to/output",
    )
    builder.build(on_progress=lambda msg, pct: print(f"{msg} ({pct*100:.0f}%)"))
"""

from __future__ import annotations

from typing import Callable, List, Optional


class GameBuilder:
    """Compile a project into a standalone player executable."""

    project_path: str
    project_name: str
    output_dir: str
    icon_guid: str
    display_mode: str
    window_width: int
    window_height: int
    window_resizable: bool
    splash_items: list

    def __init__(
        self,
        project_path: str,
        output_dir: str,
        *,
        game_name: str = ...,
        icon_guid: str = ...,
        display_mode: str = ...,
        window_width: int = ...,
        window_height: int = ...,
        window_resizable: bool = ...,
        splash_items: Optional[List[dict]] = ...,
        debug_mode: bool = ...,
        lto: bool = ...,
        include_jit_runtime: bool = ...,
        player_runtime_root: str = ...,
        build_scene_guids: Optional[List[str]] = ...,
    ) -> None: ...

    def build(
        self,
        on_progress: Optional[Callable[[str, float], None]] = ...,
        cancel_event: object = ...,
    ) -> str:
        """Run the full build pipeline.

        Returns:
            Path to the final output directory containing the built executable.

        Raises:
            RuntimeError: If validation, compilation, or packaging fails.
        """
        ...
