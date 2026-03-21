"""Tests for InfEngine.gizmos — Gizmos drawing API and GizmosCollector."""

import math

from InfEngine.gizmos.gizmos import Gizmos


# ══════════════════════════════════════════════════════════════════════
# Per-frame state reset
# ══════════════════════════════════════════════════════════════════════

class TestFrameReset:
    def test_begin_frame_resets_color(self):
        Gizmos.color = (1, 0, 0)
        Gizmos._begin_frame()
        assert Gizmos.color == (1.0, 1.0, 1.0)

    def test_begin_frame_resets_matrix(self):
        Gizmos.matrix = [0] * 16
        Gizmos._begin_frame()
        assert Gizmos.matrix is None

    def test_begin_frame_clears_batches(self):
        Gizmos._draw_batches.append(("dummy",))
        Gizmos._begin_frame()
        assert len(Gizmos._draw_batches) == 0

    def test_begin_frame_clears_icons(self):
        Gizmos._icon_entries.append(("dummy",))
        Gizmos._begin_frame()
        assert len(Gizmos._icon_entries) == 0


# ══════════════════════════════════════════════════════════════════════
# draw_line
# ══════════════════════════════════════════════════════════════════════

class TestDrawLine:
    def test_single_line(self):
        Gizmos._begin_frame()
        Gizmos.color = (1, 0, 0)
        Gizmos.draw_line((0, 0, 0), (1, 1, 1))
        assert len(Gizmos._draw_batches) == 1
        verts, indices, matrix = Gizmos._draw_batches[0]
        assert len(verts) == 2
        assert indices == [0, 1]
        # Verify color embedded in vertices
        assert verts[0][3:6] == [1, 0, 0]

    def test_uses_identity_matrix_by_default(self):
        Gizmos._begin_frame()
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        _, _, matrix = Gizmos._draw_batches[0]
        assert matrix == Gizmos._identity_matrix


# ══════════════════════════════════════════════════════════════════════
# draw_ray
# ══════════════════════════════════════════════════════════════════════

class TestDrawRay:
    def test_ray_endpoint(self):
        Gizmos._begin_frame()
        Gizmos.draw_ray((1, 2, 3), (10, 0, 0))
        verts, _, _ = Gizmos._draw_batches[0]
        # End = origin + direction
        assert verts[1][0] == pytest.approx(11.0)
        assert verts[1][1] == pytest.approx(2.0)
        assert verts[1][2] == pytest.approx(3.0)


# ══════════════════════════════════════════════════════════════════════
# draw_icon
# ══════════════════════════════════════════════════════════════════════

class TestDrawIcon:
    def test_icon_entry(self):
        Gizmos._begin_frame()
        Gizmos.color = (0, 1, 0)
        Gizmos.draw_icon((5, 5, 5), 42)
        assert len(Gizmos._icon_entries) == 1
        pos, obj_id, color = Gizmos._icon_entries[0]
        assert pos == (5, 5, 5)
        assert obj_id == 42
        assert color == (0, 1, 0)

    def test_icon_custom_color(self):
        Gizmos._begin_frame()
        Gizmos.draw_icon((0, 0, 0), 1, color=(1, 0, 0))
        _, _, color = Gizmos._icon_entries[0]
        assert color == (1, 0, 0)


# ══════════════════════════════════════════════════════════════════════
# draw_wire_cube
# ══════════════════════════════════════════════════════════════════════

class TestDrawWireCube:
    def test_produces_8_verts_24_indices(self):
        Gizmos._begin_frame()
        Gizmos.draw_wire_cube((0, 0, 0), (2, 2, 2))
        verts, indices, _ = Gizmos._draw_batches[0]
        assert len(verts) == 8
        assert len(indices) == 24  # 12 edges × 2 indices


# ══════════════════════════════════════════════════════════════════════
# draw_wire_sphere (Python fallback)
# ══════════════════════════════════════════════════════════════════════

class TestDrawWireSphere:
    def test_generates_geometry(self):
        Gizmos._begin_frame()
        Gizmos.draw_wire_sphere((0, 0, 0), 1.0, segments=12)
        assert len(Gizmos._draw_batches) >= 1
        verts, indices, _ = Gizmos._draw_batches[0]
        assert len(verts) > 0
        assert len(indices) > 0

    def test_respects_current_color(self):
        Gizmos._begin_frame()
        Gizmos.color = (0.5, 0.5, 0.5)
        Gizmos.draw_wire_sphere((0, 0, 0), 1.0, segments=8)
        verts, _, _ = Gizmos._draw_batches[0]
        # Color channels in vertex data
        assert verts[0][3] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════
# Custom matrix
# ══════════════════════════════════════════════════════════════════════

class TestCustomMatrix:
    def test_custom_matrix_used(self):
        Gizmos._begin_frame()
        custom = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        Gizmos.matrix = custom
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        _, _, matrix = Gizmos._draw_batches[0]
        assert matrix == custom


import pytest  # noqa: E402 — needed for approx
