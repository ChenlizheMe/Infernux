"""View-local graph resource publication contracts."""

import pytest

from Infernux.rendergraph.graph import Format, RenderGraph
from Infernux.renderstack.resource_bus import ResourceBus


def test_buffer_published_to_stage_and_pass_result():
    graph = RenderGraph("view-a")
    color = graph.create_texture("color", camera_target=True)
    data = graph.create_buffer("data", 64, transfer_source=True)
    bus = ResourceBus({"color": color}, graph=graph)
    bus.set("data", data)

    assert bus.require_buffer("data") is data
    assert bus.require_texture("color") is color
    assert graph.publish_pass_result("producer", bus.snapshot()).sample("data") is data


def test_foreign_and_missing_resources_fail_at_publication_or_read():
    graph = RenderGraph("view-a")
    graph.create_texture("color", camera_target=True)
    other = RenderGraph("view-b")
    foreign = other.create_buffer("data", 64)
    graph.create_buffer("data", 64)
    bus = ResourceBus(graph=graph)

    with pytest.raises(ValueError, match="does not belong"):
        bus.set("data", foreign)
    with pytest.raises(ValueError, match="not available"):
        bus.require_buffer("data")
    with pytest.raises(ValueError, match="does not belong"):
        graph.publish_pass_result("foreign", {"data": foreign})


def test_resource_kinds_are_explicit():
    graph = RenderGraph("view")
    texture = graph.create_texture("volume_sample", format=Format.RGBA8_UNORM)
    buffer = graph.create_buffer("data", 64)
    bus = ResourceBus({"volume_sample": texture, "data": buffer}, graph=graph)

    with pytest.raises(TypeError, match="not a graph BufferHandle"):
        bus.require_buffer("volume_sample")
    with pytest.raises(TypeError, match="not a graph TextureHandle"):
        bus.require_texture("data")
