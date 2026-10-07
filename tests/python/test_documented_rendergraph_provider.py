"""Build the bilingual tutorial Provider through its documented RenderStack host."""

import re
from pathlib import Path

import pytest

from infernux.renderstack import RenderStack
from infernux.renderstack.render_pipeline import RenderPipeline


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("output_samples", (1, 2, 4, 8))
def test_documented_integer_provider_builds(language, output_samples):
    source = Path(__file__).resolve().parents[2] / "docs/learn/rendergraph-advanced.md"
    english, chinese = source.read_text(encoding="utf-8").split("<!-- language:zh -->")
    blocks = re.findall(r"```python\n(.*?)```", english if language == "en" else chinese, re.S)
    provider = next(block for block in blocks if "class ObjectIndexPipeline(" in block)
    complete = next(block for block in blocks if "class BaseColorPresentPipeline(" in block)
    topology = complete[complete.index("    def define_topology"):].replace("preview_color", "object_index")
    topology = topology.replace('"Fullscreen Blit"', '"Tutorial Object Index Preview"')
    namespace = {"__name__": __name__}
    exec(compile(provider + "\n" + topology, str(source), "exec"), namespace)
    stack = RenderStack()
    # Supply the documented class directly: this test exercises topology, while
    # the real Editor walkthrough separately exercises asset discovery/selection.
    stack._pipeline = namespace["ObjectIndexPipeline"]()
    description = stack.build_graph(output_samples=output_samples)
    picking = next(render_pass for render_pass in description.passes if render_pass.name == "opaque_object_index")
    assert picking.write_depth == "depth"
    assert picking.clear_color
    assert picking.clear_depth
    assert (picking.clear_color_r, picking.clear_color_g, picking.clear_color_b, picking.clear_color_a) == (0.0, 0.0, 0.0, 0.0)
    assert picking.clear_depth_value == 1.0


class _StandaloneContext:
    def __init__(self, samples):
        self.output_samples = samples
        self.descriptions = []
        self.submissions = []

    def setup_camera_properties(self, camera):
        self.camera = camera

    def cull(self, camera):
        assert camera is self.camera
        return camera

    def is_graph_revision_current(self, revision):
        return bool(self.descriptions and self.descriptions[-1].source_revision == revision)

    def apply_graph(self, description):
        self.descriptions.append(description)

    def submit_culling(self, culling):
        self.submissions.append(culling)


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("output_samples", (1, 2, 4, 8))
def test_documented_base_color_pipeline_builds_standalone(language, output_samples):
    source = Path(__file__).resolve().parents[2] / "docs/learn/rendergraph-advanced.md"
    english, chinese = source.read_text(encoding="utf-8").split("<!-- language:zh -->")
    blocks = re.findall(r"```python\n(.*?)```", english if language == "en" else chinese, re.S)
    complete = next(block for block in blocks if "class BaseColorPresentPipeline(" in block)
    namespace = {"__name__": __name__}
    exec(compile(complete, str(source), "exec"), namespace)
    pipeline = namespace["BaseColorPresentPipeline"]()
    context = _StandaloneContext(output_samples)
    camera = object()
    for _ in range(2):
        pipeline.render(context, camera)
    description, = context.descriptions
    assert description.msaa_samples == output_samples
    assert any(p.name == "opaque_preview_color" for p in description.passes)
    assert any(p.name == "Present" for p in description.passes)
    assert context.submissions == [camera, camera]
    assert pipeline._defining_graph is None
    with pytest.raises(RuntimeError, match="only valid while defining"):
        pipeline.require_buffer("preview_color")


@pytest.mark.parametrize("output_samples", (1, 2, 4, 8))
def test_standalone_pipeline_publishes_immutable_result_revisions(output_samples):
    class ResultsPipeline(RenderPipeline):
        def define_topology(self, graph):
            graph.set_msaa_samples(1)
            original = graph.create_texture("original")
            graph.add_pass("Initial").write_color(original).set_clear(color=(0.0, 0.0, 0.0, 1.0))
            self.initial = self.publish_result("initial", {"color": original})
            updated = graph.create_texture("updated", camera_target=True)
            graph.add_pass("Updated").write_color(updated).set_clear(color=(1.0, 0.0, 0.0, 1.0))
            self.updated = self.write_buffer(self.initial, "color", updated, source="updated")
            graph.set_output(self.sample_buffer(self.updated, "color"))

    pipeline = ResultsPipeline()
    context = _StandaloneContext(output_samples)
    pipeline.render(context, object())
    assert pipeline.initial.sample("color").name == "original"
    assert pipeline.updated.sample("color").name == "updated"
    assert pipeline._defining_graph is None
    with pytest.raises(RuntimeError, match="only valid while defining"):
        pipeline.publish_result("outside", {})
    with pytest.raises(RuntimeError, match="only valid while defining"):
        pipeline.write_buffer(pipeline.initial, "color", pipeline.initial.sample("color"), source="outside")


def test_standalone_definition_scope_clears_after_failure():
    class BrokenPipeline(RenderPipeline):
        def define_topology(self, graph):
            color = graph.create_texture("color", camera_target=True)
            self.publish_result("initial", {"color": color})
            raise ValueError("deliberate topology failure")

    pipeline = BrokenPipeline()
    context = _StandaloneContext(1)
    with pytest.raises(ValueError, match="deliberate topology failure"):
        pipeline.render(context, object())
    assert pipeline._defining_graph is None
    assert pipeline._standalone_graphs == {}
    assert context.descriptions == []
    assert context.submissions == []
