from __future__ import annotations

from types import SimpleNamespace

import pytest

from infernux.engine.ui.scene_view_panel import SceneViewPanel


def test_planar_focus_does_not_hide_mesh_read_failure():
    class MeshRenderer:
        @staticmethod
        def get_positions():
            raise RuntimeError("mesh storage unavailable")

    with pytest.raises(RuntimeError, match="mesh storage unavailable"):
        SceneViewPanel._planar_visible_side(SimpleNamespace(), MeshRenderer())


def test_transform_only_object_has_explicit_point_bounds():
    game_object = SimpleNamespace(
        get_cpp_component=lambda _name: None,
        get_children=lambda: [],
        transform=SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0, z=3.0)),
    )

    assert SceneViewPanel._compute_object_bounds(game_object) == (
        (1.0, 2.0, 3.0),
        1.0,
    )
