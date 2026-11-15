import os

_package_dir = os.environ.get("_INFERNUX_PACKAGED_RESOURCE_ROOT") or os.path.dirname(
    os.path.abspath(__file__)
)

_UI_FONT = "Inter-VF.ttf"
# The VF build's default instance is Thin (wght 100) and stb_truetype cannot
# select a variation, so the CJK merge uses the static Regular face.
_UI_CJK_FONT = "NotoSansSC-Regular.ttf"
_MONO_FONT = "MapleMono-CN-Regular.ttf"

# Default to package directory; overridden by activate_library() at launch.
icon_path = os.path.join(_package_dir, "icons", "icon.png")
engine_font_path = os.path.join(_package_dir, "fonts", _UI_FONT)
engine_font_cjk_path = os.path.join(_package_dir, "fonts", _UI_CJK_FONT)
engine_mono_font_path = os.path.join(_package_dir, "fonts", _MONO_FONT)
engine_lib_path = os.path.join(_package_dir, "..", "lib")
resources_path = _package_dir
file_type_icons_dir = os.path.join(_package_dir, "icons")
component_icons_dir = os.path.join(_package_dir, "icons", "components")


def get_package_resources_path() -> str:
    """Return the original package resources directory (sync source)."""
    return _package_dir


def activate_library(project_path: str) -> None:
    """Redirect all resource paths to ``<project>/Library/Resources``."""
    root = os.path.join(project_path, "Library", "Resources")
    g = globals()
    g["resources_path"] = root
    g["icon_path"] = os.path.join(root, "icons", "icon.png")
    g["engine_font_path"] = os.path.join(root, "fonts", _UI_FONT)
    g["engine_font_cjk_path"] = os.path.join(root, "fonts", _UI_CJK_FONT)
    g["engine_mono_font_path"] = os.path.join(root, "fonts", _MONO_FONT)
    g["file_type_icons_dir"] = os.path.join(root, "icons")
    g["component_icons_dir"] = os.path.join(root, "icons", "components")


__all__ = [
    "icon_path",
    "engine_font_path",
    "engine_font_cjk_path",
    "engine_mono_font_path",
    "engine_lib_path",
    "resources_path",
    "file_type_icons_dir",
    "component_icons_dir",
    "get_package_resources_path",
    "activate_library",
]
