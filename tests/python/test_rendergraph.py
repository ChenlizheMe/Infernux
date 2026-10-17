"""Tests for infernux.rendergraph.graph — RenderGraph, Format, TextureHandle (real C++ backend)."""

from __future__ import annotations

import pytest
import infernux.lib as native

from infernux.lib import (
    CommandBuffer,
    DrawParameterBlock,
    GraphBufferUsage,
    GraphCommandType,
    GraphTextureRole,
    GraphPassType,
    PixelFormat,
    SampleCount,
)
from infernux.rendergraph.graph import RenderGraph, Format


# ── Helpers ──


def _make_graph():
    graph = RenderGraph("TestGraph")
    graph.create_texture("color", camera_target=True)
    graph.create_texture("depth", format=Format.D32_SFLOAT)
    return graph


# ══════════════════════════════════════════════════════════════════════
# Format enum
# ══════════════════════════════════════════════════════════════════════


class TestFormat:
    def test_native_description_and_command_buffer_use_rhi_types(self):
        graph = _make_graph()
        with graph.add_pass("Opaque") as render_pass:
            render_pass.write_color("color")
            render_pass.write_depth("depth")
            render_pass.draw_renderers()
        graph.set_output("color")
        description = graph.build()
        assert description.textures[0].format == PixelFormat.RGBA8_UNORM
        assert description.textures[1].format == PixelFormat.D32_SFLOAT

        commands = CommandBuffer("RHI contract")
        with pytest.raises(RuntimeError, match="RenderGraph"):
            commands.get_temporary_rt(64, 64, PixelFormat.RGBA16_SFLOAT, SampleCount.COUNT_4)
        with pytest.raises(RuntimeError, match="RenderGraph"):
            commands.release_temporary_rt(native.CAMERA_TARGET)
        assert commands.command_count == 0

    def test_explicit_draw_parameter_block_is_typed_and_recorded(self):
        block = DrawParameterBlock()
        block.set_float("roughness", 0.4)
        block.set_vector2("wind", (1.0, -0.5))
        block.set_vector3("axis", (0.0, 1.0, 0.0))
        block.set_vector4("weights", (1.0, 0.0, 0.0, 0.0))
        block.set_color("tint", (0.2, 0.4, 0.8, 1.0))
        block.set_int("variant", 2)
        block.set_matrix(
            "local",
            tuple(tuple(float(row * 4 + column) for column in range(4)) for row in range(4)),
        )
        block.set_texture("albedo", "white")
        assert block.size == 8
        assert block.remove("variant")
        assert block.size == 7
        block.clear()
        assert block.size == 0


# ══════════════════════════════════════════════════════════════════════
# TextureHandle
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Buffer and typed non-raster IR
# ══════════════════════════════════════════════════════════════════════


class TestTypedResourcePasses:
    @staticmethod
    def _add_camera_output(graph):
        with graph.add_pass("Opaque") as p:
            p.write_color("color")
            p.draw_renderers()
        graph.set_output("color")

    def test_buffer_copy_serializes_and_adds_transfer_usage(self):
        graph = _make_graph()
        graph.create_buffer("source", 1024, indirect=True)
        graph.create_buffer("destination", 2048)
        self._add_camera_output(graph)
        with graph.add_copy_pass("CopyParticles") as p:
            p.copy_buffer("source", "destination", byte_count=512)
            p.set_side_effect()

        description = graph.build()
        copy_pass = next(p for p in description.passes if p.name == "CopyParticles")
        command = copy_pass.commands[0]
        buffers = {buffer.name: buffer for buffer in description.buffers}
        assert copy_pass.type == GraphPassType.COPY
        assert command.type == GraphCommandType.COPY_BUFFER
        assert command.source_resource == "source"
        assert command.destination_resource == "destination"
        assert command.copy_bytes == 512
        assert buffers["source"].usage & int(GraphBufferUsage.TRANSFER_SOURCE)
        assert buffers["source"].usage & int(GraphBufferUsage.STORAGE)
        assert buffers["source"].usage & int(GraphBufferUsage.INDIRECT)
        assert buffers["destination"].usage & int(GraphBufferUsage.TRANSFER_DESTINATION)

    def test_texture_copy_serializes(self):
        graph = _make_graph()
        graph.create_texture("source", format=Format.RGBA16_SFLOAT)
        graph.create_texture("destination", format=Format.RGBA16_SFLOAT)
        with graph.add_pass("Produce") as p:
            p.write_color("source")
            p.draw_renderers()
        with graph.add_copy_pass("CopyColor") as p:
            p.copy_texture("source", "destination")
        graph.set_output("destination")

        description = graph.build()
        copy_pass = next(p for p in description.passes if p.name == "CopyColor")
        command = copy_pass.commands[0]
        assert copy_pass.type == GraphPassType.COPY
        assert command.type == GraphCommandType.COPY_TEXTURE
        assert command.source_resource == "source"
        assert command.destination_resource == "destination"


# ══════════════════════════════════════════════════════════════════════
# RenderPassBuilder
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# RenderGraph — texture management
# ══════════════════════════════════════════════════════════════════════


class TestGraphTextures:
    def test_imported_volume_asset_uses_guid_and_is_sample_only(self, monkeypatch):
        class TextureAsset:
            guid = "0123456789abcdef0123456789abcdef"
            dimension = "3d"
            pixel_width = 8
            pixel_height = 6
            pixel_depth = 4

        monkeypatch.setattr(native, "InxTexture", TextureAsset)
        graph = RenderGraph("VolumeAsset")
        graph.create_texture("color", camera_target=True)
        volume = graph.import_texture("density", TextureAsset())
        alias = graph.import_texture("same_asset", TextureAsset())
        assert alias is not volume
        assert alias.name == "same_asset"
        assert alias.asset_guid == volume.asset_guid
        with graph.add_pass("SampleVolume") as render_pass:
            render_pass.set_texture("density", volume)
            render_pass.write_color("color")
            render_pass.fullscreen_quad("Tests/VolumeInput")
        graph.set_output("color")

        description = graph.build()
        imported = next(texture for texture in description.textures if texture.name == "density")
        assert imported.role == GraphTextureRole.ASSET
        assert imported.asset_guid == TextureAsset.guid
        assert (imported.width, imported.height, imported.depth) == (8, 6, 4)
        assert imported.is_volume
        assert imported.samples == 1
        assert imported.format == Format.UNDEFINED
        assert description.passes[0].commands[0].input_bindings == [("density", "density")]

        with graph.add_pass("InvalidAssetWriter") as writer:
            writer.write_color(volume)
            writer.draw_renderers()
        with pytest.raises(ValueError, match="sample-only Texture asset"):
            graph.build()

    def test_temporal_history_builds_typed_single_sample_pair(self):
        graph = RenderGraph("Temporal")
        history_read, history_write = graph.create_temporal_history("taa")
        with graph.add_copy_pass("Commit") as commit:
            commit.copy_texture(history_read, history_write)
            commit.set_side_effect()
        graph.set_output(history_read)

        description = graph.build()
        textures = {texture.name: texture for texture in description.textures}

        assert history_read.name == "taa/read"
        assert history_write.name == "taa/write"
        assert textures["taa/read"].role == GraphTextureRole.TEMPORAL_READ
        assert textures["taa/write"].role == GraphTextureRole.TEMPORAL_WRITE
        assert textures["taa/read"].temporal_key == "taa"
        assert textures["taa/write"].temporal_key == "taa"
        assert textures["taa/read"].samples == 1
        assert not description.temporal_jitter
        graph.set_temporal_jitter()
        assert graph.build().temporal_jitter
        graph.set_temporal_jitter(False)
        assert not graph.build().temporal_jitter

    @pytest.mark.parametrize("options", [{"size": (37, 23), "size_divisor": 2}])
    def test_invalid_temporal_dimensions_publish_no_partial_pair(self, options):
        graph = RenderGraph("Temporal")
        with pytest.raises(ValueError):
            graph.create_temporal_history("color", **options)
        assert not graph._textures


# ══════════════════════════════════════════════════════════════════════
# RenderGraph — pass management
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Injection point callbacks
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Overlay reordering
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Build & validation
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Validation error cases
# ══════════════════════════════════════════════════════════════════════
