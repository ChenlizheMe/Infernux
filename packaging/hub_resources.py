"""Resource paths for Infernux Hub (launcher).

Resolves to packaging/resources/ whether running from source or
from a Nuitka standalone bundle.
"""

import os
from hub_utils import get_bundle_dir


def _resource_dir() -> str:
    return os.path.join(get_bundle_dir(), "resources")


RESOURCE_DIR = _resource_dir()
ICON_PATH = os.path.join(RESOURCE_DIR, "icon.png")
FONTS_DIR = os.path.join(RESOURCE_DIR, "fonts")
# The Hub's single typeface (shared with the website); CJK uses the system UI font.
FONT_PATHS = tuple(
    os.path.join(FONTS_DIR, f"SpaceGrotesk-{weight}.ttf")
    for weight in ("Regular", "Medium", "Bold")
)