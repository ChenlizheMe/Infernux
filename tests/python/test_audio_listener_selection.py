import pytest

from infernux.lib import AudioEngine, SceneManager


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


@pytest.mark.parametrize("device_open", [False, True])
@pytest.mark.parametrize("with_standby", [False, True])
def test_disabled_listener_never_registers_during_first_awake(scene, device_open, with_standby):
    audio = AudioEngine.instance()
    if not device_open:
        audio.shutdown()
    try:
        owner = scene.create_game_object("disabled before first Awake")
        owner.active = False
        listener = owner.add_component("AudioListener")
        listener.enabled = False
        standby = scene.create_game_object("standby") if with_standby else None
        if standby is not None:
            standby.add_component("AudioListener")
        expected = standby.id if standby is not None else 0
        assert audio.active_listener_game_object_id == expected
        for _ in range(2):
            owner.active = True
            assert audio.active_listener_game_object_id == expected
            owner.active = False
            assert audio.active_listener_game_object_id == expected
        owner.active = True
        listener.enabled = True
        assert audio.active_listener_game_object_id == (expected or owner.id)
        listener.enabled = False
        assert audio.active_listener_game_object_id == expected
    finally:
        if not audio.is_initialized:
            audio.initialize()


@pytest.mark.parametrize("device_open", [False, True])
@pytest.mark.parametrize("listener_enabled", [False, True])
def test_scene_loading_preserves_listener_enabled_contract(scene, device_open, listener_enabled):
    audio = AudioEngine.instance()
    owner = scene.create_game_object("authored listener")
    owner.active = False
    listener = owner.add_component("AudioListener")
    listener.enabled = listener_enabled
    document = scene.serialize_document()
    document["objects"][0]["active"] = True
    if not device_open:
        audio.shutdown()
    try:
        assert scene._commit_document(document)
        scene.start()
        restored = scene.find("authored listener")
        assert audio.active_listener_game_object_id == (restored.id if listener_enabled else 0)
        restored.active = False
        assert audio.active_listener_game_object_id == 0
        restored.active = True
        assert audio.active_listener_game_object_id == (restored.id if listener_enabled else 0)
    finally:
        if not audio.is_initialized:
            audio.initialize()
