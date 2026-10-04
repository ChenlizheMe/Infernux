"""Build the bilingual tutorial Provider through its documented RenderStack host."""

import re
from pathlib import Path

import pytest

from Infernux.renderstack import RenderStack


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
