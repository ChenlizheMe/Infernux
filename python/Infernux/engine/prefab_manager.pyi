"""Prefab manager — save, instantiate, and manage prefab assets.

Example::

    from Infernux.engine.prefab_manager import save_prefab, instantiate_prefab

    save_prefab(game_object, "/path/to/my.prefab")
    go = instantiate_prefab(guid="registered-prefab-guid", asset_database=database)
"""

from __future__ import annotations

from typing import Any, Optional


def save_prefab(
    game_object: Any,
    file_path: str,
    asset_database: Any = None,
    source_canvas_name: str = "",
    root_document_template: Optional[dict] = None,
) -> bool:
    """Serialize *game_object* and its children as a prefab file.

    Args:
        game_object: The root ``GameObject`` to save.
        file_path: Destination ``.prefab`` path.
        asset_database: Optional C++ ``AssetDatabase``.

    Returns:
        ``True`` on success.
    """
    ...

def instantiate_prefab(
    file_path: Optional[str] = None,
    guid: Optional[str] = None,
    parent: Any = None,
    asset_database: Any = None,
    *,
    instantiate_in_world_space: bool = False,
    configure_created: Any = None,
) -> Any:
    """Instantiate a prefab into the active scene.

    Args:
        file_path: Editor-selected path, immediately resolved through
            ``asset_database`` when one is supplied.
        guid: Registered asset GUID and authoritative prefab identity.
        parent: Optional parent ``GameObject``.
        asset_database: Optional C++ ``AssetDatabase``.

    Returns:
        The root ``GameObject`` of the instantiated prefab.
    """
    ...
