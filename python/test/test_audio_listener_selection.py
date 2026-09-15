from Infernux.lib import AudioEngine, SceneManager


def test_world_audio_listener_keeps_owner_and_promotes_deterministically(scene):
    audio = AudioEngine.instance()
    owners = [scene.create_game_object(name) for name in ("listener-a", "listener-b", "listener-c")]
    listeners = [owner.add_component("AudioListener") for owner in owners]

    assert audio.active_listener_game_object_id == owners[0].id

    listeners[0].enabled = False
    assert audio.active_listener_game_object_id == min(owners[1].id, owners[2].id)

    listeners[1].enabled = False
    assert audio.active_listener_game_object_id == owners[2].id

    listeners[2].enabled = False
    assert audio.active_listener_game_object_id == 0


def test_loaded_scenes_share_one_listener_independent_of_active_scene(scene):
    manager = SceneManager.instance()
    audio = AudioEngine.instance()
    owner_a = scene.create_game_object("listener-scene-a")
    listener_a = owner_a.add_component("AudioListener")
    additive = manager.create_scene("AudioAdditive")
    owner_b = additive.create_game_object("listener-scene-b")
    owner_b.add_component("AudioListener")
    try:
        assert audio.active_listener_game_object_id == owner_a.id

        manager.set_active_scene(additive)
        assert audio.active_listener_game_object_id == owner_a.id

        listener_a.enabled = False
        assert audio.active_listener_game_object_id == owner_b.id

        listener_a.enabled = True
        assert audio.active_listener_game_object_id == owner_b.id

        manager.unload_scene(additive)
        additive = None
        assert audio.active_listener_game_object_id == owner_a.id
    finally:
        manager.set_active_scene(scene)
        if additive is not None:
            manager.unload_scene(additive)
