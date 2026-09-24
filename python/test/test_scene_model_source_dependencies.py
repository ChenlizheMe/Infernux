from __future__ import annotations


def test_empty_model_root_remains_a_mesh_dependency(scene):
    """A model container can outlive every imported renderer.

    The hidden source identity still owns the model reference and must keep it
    in the Scene dependency closure so a later source-node addition can be
    reconciled without relying on a surviving MeshRenderer.
    """
    from Infernux.lib import _Infernux as native

    root = scene.create_game_object("Empty Model Root")
    root._set_model_source("model-source-guid", [])

    dependencies = native._collect_scene_resource_dependencies(
        scene.serialize_document()
    )

    assert dependencies.count(("model-source-guid", "Mesh")) == 1


def test_nested_model_source_is_collected_once_without_renderers(scene):
    """Every object-level source is authoritative and dependency output deduplicates."""
    from Infernux.lib import _Infernux as native

    root = scene.create_game_object("Model Root")
    root._set_model_source("model-source-guid", [])
    child = scene.create_game_object("Imported Pivot")
    child._set_model_source("model-source-guid", ["Imported Pivot"])
    child.set_parent(root, world_position_stays=False)

    dependencies = native._collect_scene_resource_dependencies(
        scene.serialize_document()
    )

    assert dependencies.count(("model-source-guid", "Mesh")) == 1
