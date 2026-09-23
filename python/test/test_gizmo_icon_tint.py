"""Scene gizmo icons keep a neutral authored colour contract."""

from pathlib import Path

from PIL import Image

from Infernux.components._gizmo_ids import ICON_KIND_CAMERA, ICON_KIND_LIGHT
from Infernux.components.builtin.camera import Camera
from Infernux.components.builtin.light import Light
from Infernux.gizmos.collector import GizmosCollector
from Infernux.gizmos.gizmos import Gizmos


def test_camera_and_light_billboards_keep_distinct_kinds_and_white_tints():
    camera_tint = GizmosCollector._resolve_class_value(Camera, "_gizmo_icon_color", None)
    camera_kind = GizmosCollector._resolve_class_value(Camera, "_gizmo_icon_kind", None)
    light_tint = GizmosCollector._resolve_class_value(Light, "_gizmo_icon_color", None)
    light_kind = GizmosCollector._resolve_class_value(Light, "_gizmo_icon_kind", None)
    Gizmos._begin_frame()
    try:
        Gizmos.draw_icon((0, 2, 0), 11, camera_tint, icon_kind=camera_kind)
        Gizmos.draw_icon((0, 4, 0), 17, light_tint, icon_kind=light_kind)
        packed = Gizmos._get_packed_icon_data()
        assert packed is not None
        _, _, kind_buffer, icon_count = packed
        assert icon_count == 2
        assert list(kind_buffer) == [ICON_KIND_CAMERA, ICON_KIND_LIGHT]
        assert Gizmos._icon_entries[0][2] == (1.0, 1.0, 1.0)
        assert Gizmos._icon_entries[1][2] == (1.0, 1.0, 1.0)
    finally:
        Gizmos._begin_frame()


def test_camera_and_light_gizmo_textures_are_alpha_silhouettes():
    for name in ("gizmo_camera.png", "gizmo_light.png"):
        image = Image.open(Path("python/Infernux/resources/icons") / name).convert("RGBA")
        pixels = tuple(image.get_flattened_data())
        visible = [pixel for pixel in pixels if pixel[3] != 0]
        assert visible, name
        assert len(visible) < len(pixels), name
        assert max(pixel[3] for pixel in pixels) == 255, name
        assert min(pixel[3] for pixel in pixels) == 0, name

    light = Image.open(
        Path("python/Infernux/resources/icons/gizmo_light.png")
    ).convert("RGBA")
    visible_light = [pixel for pixel in light.get_flattened_data() if pixel[3] != 0]
    assert {pixel[:3] for pixel in visible_light} == {(255, 255, 255)}
