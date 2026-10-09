"""Embedded Sprite materials must not inherit each process's component IDs."""
import copy


def test_default_sprite_material_survives_reallocation_without_renaming(scene):
    first = scene.create_game_object('First sprite').add_component('SpriteRenderer')
    first.sync_visual()
    source = copy.deepcopy(first.serialize_document())
    for index in range(7):
        scene.create_game_object(f'Other allocation {index}').add_component('MeshRenderer')
    second = scene.create_game_object('Reopened sprite').add_component('SpriteRenderer')
    assert second.component_id != first.component_id
    document = copy.deepcopy(source)
    document['component_id'] = second.component_id
    assert second.deserialize_document(document)
    second.sync_visual()
    assert second.serialize_document()['materials'] == source['materials']
    assert second.material is not first.material
    assert second.material.name == first.material.name
