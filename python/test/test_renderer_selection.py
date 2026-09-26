"""Real native selection ownership, graph projection and author boundaries."""
import gc
from pathlib import Path
import numpy as np
import pytest

from Infernux.core.material import Material
from Infernux.lib import InxMaterial, CommandBuffer
from Infernux.rendergraph import RendererSelection, DrawParameterBlock, RenderGraph, Format


@pytest.fixture
def selected(scene):
    material = Material(InxMaterial("Mask selection", "Unlit"))
    material.set_color("_BaseColor", 1., 1., 1., 1.)
    renderer = scene.create_game_object("Selected").add_component("MeshRenderer")
    return RendererSelection(material), renderer


def test_selection_edits_preserve_graph_owner_and_source_revision(selected):
    selection, renderer = selected
    graph = RenderGraph("Selected mask")
    graph.create_texture("mask", format=Format.RGBA16_SFLOAT)
    graph.create_texture("depth", format=Format.D32_SFLOAT)
    graph.add_pass("Mask").write_color("mask").write_depth("depth").draw_renderers(renderer_selection=selection)
    graph.set_output("mask")
    desc = graph.build()
    revision = desc.source_revision
    owner = desc.passes[0].commands[0].renderer_selection
    assert owner.size == 0
    values = DrawParameterBlock()
    values.set_color("_BaseColor", (1., 0., 0., 1.))
    selection.set(renderer, parameters=values)
    assert owner.size == 1 and owner.revision == selection.revision
    selection.set(renderer, submesh=2)
    assert owner.size == 2 and desc.source_revision == revision
    assert selection.remove(renderer, submesh=2)
    assert not selection.remove(renderer, submesh=2)
    selection.clear()
    assert owner.size == 0
    selection.set(renderer)
    del selection, selected, graph
    gc.collect()
    assert owner.size == 1  # Native graph ownership, not the Python builder.


def test_bad_capture_is_atomic_and_validated_against_selection_material(selected):
    selection, renderer = selected
    values = DrawParameterBlock()
    values.set_color("_BaseColor", (0., 1., 0., 1.))
    selection.set(renderer, parameters=values)
    revision = selection.revision
    values.set_float("_BaseColor", 2.)
    with pytest.raises(ValueError, match="reflected shader type"):
        selection.set(renderer, parameters=values)
    assert selection.size == 1 and selection.revision == revision
    values.clear()
    values.set_float("undeclared_mask_id", 2.)
    with pytest.raises(ValueError, match="no parameter"):
        selection.set(renderer, parameters=values)


def test_documented_selection_example_uses_the_public_namespace(scene):
    """Execute the actual guide, not just its import spelling."""
    text = (Path(__file__).parents[2] / 'docs/learn/rendergraph-advanced.md').read_text(encoding='utf-8')
    section = text.split('{#renderer-selection}', 1)[1]
    source = section.split('```python\n', 1)[1].split('```', 1)[0]
    material = Material(InxMaterial('Documented mask', 'Unlit'))
    material.set_color('baseColor', 1., 1., 1., 1.)
    renderer = scene.create_game_object('Guide selection').add_component('MeshRenderer')
    graph = RenderGraph('Guide selection')
    mask = graph.create_texture('mask', format=Format.RGBA16_SFLOAT)
    depth = graph.create_texture('mask_depth', format=Format.D32_SFLOAT)
    namespace = dict(mask_material=material, target_renderer=renderer, graph=graph, mask=mask, mask_depth=depth)
    exec(compile(source, 'rendergraph-advanced.md#renderer-selection', 'exec'), namespace)
    graph.set_output('mask')
    command = graph.build().passes[0].commands[0]
    assert command.renderer_selection.size == 1
    assert namespace['selection'].revision == 1


@pytest.mark.parametrize("submesh", [-2, -10])
def test_invalid_submesh_is_rejected(selected, submesh):
    selection, renderer = selected
    with pytest.raises(ValueError, match="submesh"):
        selection.set(renderer, submesh=submesh)
    assert selection.size == 0


def test_selection_cannot_supply_two_material_owners(selected):
    selection, _ = selected
    graph = RenderGraph("Invalid selection")
    with pytest.raises(ValueError, match="already supplies"):
        graph.add_pass("Mask").draw_renderers(renderer_selection=selection, override_material="DefaultLit")
    with pytest.raises(TypeError, match="RendererSelection"):
        graph.add_pass("Bad type").draw_renderers(renderer_selection=[])
    with pytest.raises(TypeError, match="Material"):
        RendererSelection("material-name")
    with pytest.raises(TypeError, match="MeshRenderer"):
        selection.set(13)


@pytest.mark.parametrize("matrix", [np.eye(4), np.eye(4).T, np.eye(4)[::-1], np.eye(4).tolist()])
def test_draw_mesh_uses_shared_numpy_matrix_boundary(matrix):
    # Parsing the transform precedes the native resource validation. None is
    # deliberately rejected by DrawMesh, not mistaken for a matrix layout error.
    with pytest.raises(ValueError, match="mesh") as error:
        CommandBuffer().draw_mesh(None, matrix, None)
    assert "matrix" not in str(error.value)
