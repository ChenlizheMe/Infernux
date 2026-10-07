"""Render ownership and the component lifecycle index have different keys."""
from infernux.components import InxComponent
from infernux.lib import SceneManager
from infernux.renderstack import RenderStack


def test_attached_stacks_remain_in_both_world_and_component_indices(scene):
    peer = SceneManager.instance().create_scene("RenderStackPeer")
    first_owner = scene.create_game_object("FirstStack")
    second_owner = peer.create_game_object("SecondStack")
    first = first_owner.add_component(RenderStack)
    second = second_owner.add_component(RenderStack)
    RenderStack.refresh_active_instance(scene)
    RenderStack.refresh_active_instance(peer)

    assert first in InxComponent._active_instances.get(first_owner.id, ())
    assert second in InxComponent._active_instances.get(second_owner.id, ())
    assert RenderStack.instance(scene) is first
    assert RenderStack.instance(peer) is second
    first.enabled = False
    assert RenderStack.instance(scene) is None
    assert RenderStack.instance(peer) is second
    first.enabled = True
    RenderStack.refresh_active_instance(scene)
    assert RenderStack.instance(scene) is first
    assert first in InxComponent._active_instances[first_owner.id]
