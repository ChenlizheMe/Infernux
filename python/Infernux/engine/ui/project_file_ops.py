"""
File-system CRUD operations for the Project panel.

Functions accept the required state as explicit parameters so they don't
depend on ``ProjectPanel`` internals.
"""

import os
import shutil


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATE = '''
from Infernux import *


class {class_name}(InxComponent):
    # Public fields (automatically serialized and shown in Inspector)
    # speed = 5.0       # float (use .0 for decimals)
    
    def start(self):
        """Called before first update, after all awake() calls."""
    
    def update(self, delta_time: float):
        """Called every frame."""
'''

VERTEX_SHADER_TEMPLATE = '''#version 450

@shader_id: {shader_id}
'''

FRAGMENT_SHADER_TEMPLATE = '''#version 450

@shader_id: {shader_id}
@shading_model: unlit
@property: baseColor, Color, [1.0, 1.0, 1.0, 1.0]
@property: texSampler, Texture2D, white

void surface(out SurfaceData s) {{
    s = InitSurfaceData();
    vec4 texColor = texture(texSampler, v_TexCoord);
    s.albedo = texColor.rgb * v_Color * material.baseColor.rgb;
    s.alpha  = texColor.a * material.baseColor.a;
}}
'''

SCENE_TEMPLATE = '''{{
  "schema_version": 1,
  "name": "{scene_name}",
  "isPlaying": false,
  "objects": []
}}
'''
MATERIAL_TEMPLATE = '''{{
  "name": "{material_name}",
  "guid": "",
  "shaders": {{
        "vertex": "standard",
    "fragment": "unlit"
  }},
  "renderState": {{
    "cullMode": 2,
        "frontFace": 1,
    "polygonMode": 0,
    "depthTestEnable": true,
    "depthWriteEnable": true,
    "depthCompareOp": 1,
    "blendEnable": false,
    "srcColorBlendFactor": 6,
    "dstColorBlendFactor": 7,
    "colorBlendOp": 0,
    "renderQueue": 2000
  }},
  "properties": {{
    "baseColor": {{
      "type": 3,
      "value": [1.0, 1.0, 1.0, 1.0]
    }}
  }}
}}
'''


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_unique_name(current_path: str, base_name: str, extension: str = "") -> str:
    """Return a *base_name* that doesn't clash with existing entries in *current_path*.

    *extension* is considered when checking for conflicts but is NOT appended
    to the returned string.
    """
    name = base_name + extension
    full_path = os.path.join(current_path, name)
    full_path_no_ext = os.path.join(current_path, base_name)

    if not os.path.exists(full_path) and not os.path.exists(full_path_no_ext):
        return base_name

    counter = 1
    while True:
        candidate = f"{base_name}{counter}"
        name_with_ext = candidate + extension
        fp = os.path.join(current_path, name_with_ext)
        fp_ne = os.path.join(current_path, candidate)
        if not os.path.exists(fp) and not os.path.exists(fp_ne):
            return candidate
        counter += 1
        if counter > 999:
            break
    return f"{base_name}{counter}"


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_path_within(path: str, parent_path: str) -> bool:
    try:
        return os.path.commonpath([_normalize_path(path), _normalize_path(parent_path)]) == _normalize_path(parent_path)
    except ValueError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return False


def _iter_asset_move_pairs(old_path: str, new_path: str):
    if os.path.isdir(old_path):
        for dirpath, _dirnames, filenames in os.walk(old_path):
            rel_dir = os.path.relpath(dirpath, old_path)
            mapped_dir = new_path if rel_dir == "." else os.path.join(new_path, rel_dir)
            for filename in filenames:
                yield os.path.join(dirpath, filename), os.path.join(mapped_dir, filename)
    elif os.path.isfile(old_path):
        yield old_path, new_path


def _update_build_settings_scene_path(old_path: str, new_path: str):
    """Replace *old_path* with *new_path* in BuildSettings.json scene list."""
    try:
        from .build_settings_panel import load_build_settings, save_build_settings
        settings = load_build_settings()
        scenes = settings.get("scenes", [])
        old_norm = os.path.normcase(os.path.abspath(old_path))
        changed = False
        for i, s in enumerate(scenes):
            if os.path.normcase(os.path.abspath(s)) == old_norm:
                scenes[i] = os.path.abspath(new_path)
                changed = True
        if changed:
            settings["scenes"] = scenes
            save_build_settings(settings)
    except Exception as _exc:
        Debug.log(f"[BuildSettings] Failed to update scene path: {_exc}")


def _notify_asset_moved(old_path: str, new_path: str, asset_database=None):
    from Infernux.core.assets import AssetManager
    from . import asset_inspector

    if asset_database:
        try:
            asset_database.on_asset_moved(old_path, new_path)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    try:
        AssetManager.on_asset_moved(old_path, new_path)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass

    # Update BuildSettings.json when a .scene file is renamed/moved
    if old_path.lower().endswith(".scene"):
        _update_build_settings_scene_path(old_path, new_path)

    asset_inspector.invalidate_asset(old_path)
    asset_inspector.invalidate_asset(new_path)


def move_path(old_path: str, new_path: str, asset_database=None):
    """Move or rename a file/directory to *new_path* and notify asset systems."""
    if not old_path or not new_path or not os.path.exists(old_path):
        return None

    old_abs = os.path.abspath(old_path)
    new_abs = os.path.abspath(new_path)
    if _normalize_path(old_abs) == _normalize_path(new_abs):
        return new_abs

    if os.path.isdir(old_abs) and _is_path_within(new_abs, old_abs):
        return None

    move_pairs = list(_iter_asset_move_pairs(old_abs, new_abs))
    try:
        shutil.move(old_abs, new_abs)
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None

    for old_file, new_file in move_pairs:
        _notify_asset_moved(old_file, new_file, asset_database)

    return new_abs


def move_item_to_directory(item_path: str, dest_dir: str, asset_database=None):
    """Move *item_path* into *dest_dir*, generating a unique name on conflicts."""
    if not item_path or not dest_dir or not os.path.exists(item_path) or not os.path.isdir(dest_dir):
        return None

    item_abs = os.path.abspath(item_path)
    dest_abs = os.path.abspath(dest_dir)

    name = os.path.basename(item_abs)
    new_path = os.path.join(dest_abs, name)
    if os.path.exists(new_path) and _normalize_path(new_path) != _normalize_path(item_abs):
        base, ext = os.path.splitext(name)
        if os.path.isdir(item_abs):
            base = name
            ext = ""
        unique_name = get_unique_name(dest_abs, base, ext)
        new_path = os.path.join(dest_abs, unique_name + ext)

    return move_path(item_abs, new_path, asset_database)


# ---------------------------------------------------------------------------
# Create operations
# ---------------------------------------------------------------------------

def create_folder(current_path: str, folder_name: str):
    """Create a folder and return ``(True, "")`` or ``(False, error_msg)``."""
    if not folder_name or not current_path:
        return False, "Invalid folder name"

    folder_name = folder_name.strip()
    if not folder_name:
        return False, "Folder name cannot be empty"

    new_path = os.path.join(current_path, folder_name)
    if os.path.exists(new_path):
        return False, f"'{folder_name}' already exists"

    try:
        os.makedirs(new_path)
    except OSError as exc:
        return False, str(exc)
    return True, ""


def create_script(current_path: str, script_name: str, asset_database=None):
    """Create a Python script from template. Returns ``(True, "")`` or ``(False, error_msg)``."""
    if not script_name or not current_path:
        return False, "Invalid script name"

    script_name = script_name.strip()
    if not script_name:
        return False, "Script name cannot be empty"

    class_name = script_name
    if class_name.endswith('.py'):
        class_name = class_name[:-3]

    if not class_name.isidentifier():
        return False, "Invalid script name (must be valid Python identifier)"

    if not script_name.endswith('.py'):
        script_name = script_name + '.py'

    file_path = os.path.join(current_path, script_name)
    if os.path.exists(file_path):
        return False, f"'{script_name}' already exists"

    content = SCRIPT_TEMPLATE.format(class_name=class_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except OSError as exc:
        return False, str(exc)

    if asset_database:
        try:
            guid = asset_database.import_asset(file_path)
            print(f"[ProjectPanel] Registered script: {script_name} -> {guid}")
        except Exception as exc:
            return False, str(exc)

    return True, ""


def create_shader(current_path: str, shader_name: str, shader_type: str,
                  asset_database=None):
    """Create a shader file from template. Returns ``(True, "")`` or ``(False, error_msg)``."""
    if not shader_name or not current_path:
        return False, "Invalid shader name"

    shader_name = shader_name.strip()
    if not shader_name:
        return False, "Shader name cannot be empty"

    for ext in ['.vert', '.frag', '.glsl']:
        if shader_name.endswith(ext):
            shader_name = shader_name[:-len(ext)]
            break

    shader_id = shader_name.lower().replace(' ', '_')
    extension = f'.{shader_type}'
    file_name = shader_name + extension
    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return False, f"'{file_name}' already exists"

    if shader_type == 'vert':
        content = VERTEX_SHADER_TEMPLATE.format(shader_id=shader_id)
    else:
        content = FRAGMENT_SHADER_TEMPLATE.format(shader_id=shader_id)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except OSError as exc:
        return False, str(exc)

    if asset_database:
        try:
            guid = asset_database.import_asset(file_path)
            print(f"[ProjectPanel] Registered shader: {file_name} -> {guid}")
        except Exception as exc:
            return False, str(exc)

    return True, ""


def create_scene(current_path: str, scene_name: str, asset_database=None):
    """Create a ``.scene`` file from template. Returns ``(True, path)`` or ``(False, error_msg)``."""
    if not scene_name or not current_path:
        return False, "Invalid scene name"

    scene_name = scene_name.strip()
    if not scene_name:
        return False, "Scene name cannot be empty"

    if scene_name.endswith('.scene'):
        scene_name = scene_name[:-6]

    file_name = scene_name + '.scene'
    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return False, f"'{file_name}' already exists"

    content = SCENE_TEMPLATE.format(scene_name=scene_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except OSError as exc:
        return False, str(exc)

    if asset_database:
        try:
            guid = asset_database.import_asset(file_path)
            print(f"[ProjectPanel] Registered scene: {file_name} -> {guid}")
        except Exception as exc:
            return False, str(exc)

    return True, file_path


def create_material(current_path: str, material_name: str, asset_database=None):
    """Create a ``.mat`` file from template. Returns ``(True, "")`` or ``(False, error_msg)``."""
    if not material_name or not current_path:
        return False, "Invalid material name"

    material_name = material_name.strip()
    if not material_name:
        return False, "Material name cannot be empty"

    if material_name.endswith('.mat'):
        material_name = material_name[:-4]

    file_name = material_name + '.mat'
    file_path = os.path.join(current_path, file_name)

    if os.path.exists(file_path):
        return False, f"'{file_name}' already exists"

    content = MATERIAL_TEMPLATE.format(material_name=material_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except OSError as exc:
        return False, str(exc)

    if asset_database:
        try:
            guid = asset_database.import_asset(file_path)
            print(f"[ProjectPanel] Registered material: {file_name} -> {guid}")
        except Exception as exc:
            return False, str(exc)

    return True, ""


def create_prefab_from_gameobject(game_object, current_path: str,
                                  asset_database=None,
                                  source_canvas_name: str = ""):
    """Save a GameObject hierarchy as a ``.prefab`` file.

    Returns ``(True, file_path)`` or ``(False, error_msg)``.
    """
    if game_object is None or not current_path:
        return False, "Invalid parameters"

    from Infernux.engine.prefab_manager import save_prefab, PREFAB_EXTENSION

    prefab_name = get_unique_name(current_path, game_object.name, PREFAB_EXTENSION)
    file_path = os.path.join(current_path, prefab_name + PREFAB_EXTENSION)

    if save_prefab(game_object, file_path, asset_database=asset_database,
                   source_canvas_name=source_canvas_name):
        return True, file_path
    return False, "Failed to save prefab"


# ---------------------------------------------------------------------------
# Delete & Rename
# ---------------------------------------------------------------------------

def _detach_prefab_instances(prefab_path: str, asset_database=None):
    """Clear prefab_guid/prefab_root on all scene objects linked to this prefab."""
    guid = ""
    if asset_database:
        try:
            guid = asset_database.get_guid_from_path(prefab_path)
        except Exception:
            pass
    if not guid:
        return

    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if scene is None:
        return

    def _walk(objects):
        for obj in objects:
            try:
                obj_guid = getattr(obj, 'prefab_guid', '')
                if obj_guid == guid:
                    obj.prefab_guid = ""
                    obj.prefab_root = False
                children = list(obj.get_children()) if hasattr(obj, 'get_children') else []
                _walk(children)
            except Exception:
                pass

    try:
        roots = list(scene.get_root_objects())
        _walk(roots)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")


def delete_item(item_path: str, asset_database=None):
    """Delete a file or directory from the filesystem and notify AssetDatabase."""
    if not item_path or not os.path.exists(item_path):
        return

    is_dir = os.path.isdir(item_path)

    # For .prefab files, detach all scene instances BEFORE deleting the asset.
    # This turns prefab instances into regular scene objects instead of leaving
    # them orphaned with a dangling prefab_guid.
    if not is_dir and item_path.lower().endswith('.prefab'):
        _detach_prefab_instances(item_path, asset_database)

    # Notify BEFORE removing the file — GUID is still resolvable at this point
    if not is_dir:
        from Infernux.core.assets import AssetManager
        AssetManager.on_asset_deleted(item_path)

        if asset_database:
            asset_database.on_asset_deleted(item_path)

    try:
        if is_dir:
            import shutil
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return

    # Invalidate inspector cache so a recreated file won't reuse stale data
    from . import asset_inspector
    asset_inspector.invalidate_asset(item_path)


def do_rename(old_path: str, new_name: str, asset_database=None):
    """Rename a file or directory. Returns the new path on success, ``None`` on failure."""
    from .project_utils import update_material_name_in_file

    if not old_path or not new_name:
        return None

    safe_name = "".join([c for c in new_name if c.isalnum() or c in "._- "]).strip()
    if not safe_name:
        return None

    if os.path.isfile(old_path):
        _, ext = os.path.splitext(old_path)
        if ext:
            safe_name += ext

    new_path = os.path.join(os.path.dirname(old_path), safe_name)

    if old_path == new_path:
        return new_path  # Nothing to do

    _, ext = os.path.splitext(old_path)
    if ext.lower() == '.mat' and os.path.isfile(old_path):
        try:
            update_material_name_in_file(old_path, os.path.splitext(safe_name)[0])
        except OSError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return None

    return move_path(old_path, new_path, asset_database)
