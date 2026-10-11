"""project_file_ops — create / delete / rename project assets.

Usage::

    from infernux.engine.ui.project_file_ops import (
        create_script, create_material, delete_item, do_rename,
    )
"""

from __future__ import annotations

from typing import Optional
from infernux.core.data_asset import DataAsset
from infernux.engine.asset_creation import AssetCreationResult


# ── Template strings ────────────────────────────────────────────────

SCRIPT_TEMPLATE: str
VERTEX_SHADER_TEMPLATE: str
FRAGMENT_SHADER_TEMPLATE: str


# ── Public API ──────────────────────────────────────────────────────

def get_unique_name(
    current_path: str, base_name: str, extension: str = "",
) -> str:
    """Return a filename that doesn't collide with existing entries."""
    ...

def create_folder(current_path: str, folder_name: str) -> AssetCreationResult: ...

def create_script(current_path: str, script_name: str, asset_database=None) -> AssetCreationResult: ...

def create_shader(current_path: str, shader_name: str, shader_type: str, asset_database=None) -> AssetCreationResult: ...

def create_scene(current_path: str, scene_name: str, asset_database=None) -> AssetCreationResult: ...

def create_material(current_path: str, material_name: str, asset_database=None) -> AssetCreationResult: ...

def create_physic_material(current_path: str, material_name: str, asset_database=None) -> AssetCreationResult: ...

def create_render_texture(current_path: str, asset_name: str, asset_database=None) -> AssetCreationResult: ...

def create_data_asset(current_path: str, asset_name: str, type_id: str, asset_database=None, *, value=None) -> AssetCreationResult: ...

def create_prefab_from_gameobject(game_object, current_path: str, asset_database=None, source_canvas_name: str='') -> AssetCreationResult: ...

def create_animclip(current_path: str, clip_name: str, asset_database=None) -> AssetCreationResult: ...

def create_animclip3d(current_path: str, clip_name: str, asset_database=None) -> AssetCreationResult: ...

def create_animfsm(current_path: str, fsm_name: str, asset_database=None) -> AssetCreationResult: ...

def create_particlegraph(current_path: str, graph_name: str, asset_database=None) -> AssetCreationResult: ...

def create_render_effect(current_path: str, effect_name: str, feature_type: str, asset_database=None) -> AssetCreationResult: ...

def create_render_effect_group(current_path: str, group_name: str, asset_database=None) -> AssetCreationResult: ...

def create_animtimeline(current_path: str, timeline_name: str, asset_database=None) -> AssetCreationResult: ...

def create_timelinefsm(current_path: str, fsm_name: str, asset_database=None) -> AssetCreationResult: ...
def delete_item(
    item_path: str, asset_database: Optional[object] = None,
) -> bool: ...
def do_rename(
    old_path: str,
    new_name: str,
    asset_database: Optional[object] = None,
) -> None: ...

def move_path(
    old_path: str,
    new_path: str,
    asset_database: Optional[object] = None,
) -> None: ...

def move_item_to_directory(
    item_path: str,
    dest_dir: str,
    asset_database: Optional[object] = None,
) -> None: ...
