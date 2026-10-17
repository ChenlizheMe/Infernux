import pytest

from infernux.lib import AudioEngine, SceneManager


@pytest.mark.parametrize("device_open,with_standby", [(False, True), (True, False)])
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


@pytest.mark.parametrize("device_open,listener_enabled", [(False, True), (True, False)])
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


def test_world_listener_selection_survives_additive_activation_and_promotes_live_owners(scene):
    manager, audio = SceneManager.instance(), AudioEngine.instance()
    first, standby = scene.create_game_object("first listener"), scene.create_game_object("standby listener")
    a, b = first.add_component("AudioListener"), standby.add_component("AudioListener")
    additive = manager.create_scene("AudioAdditive")
    other = additive.create_game_object("additive listener")
    other.add_component("AudioListener")
    try:
        assert audio.active_listener_game_object_id == first.id
        manager.set_active_scene(additive)
        assert audio.active_listener_game_object_id == first.id
        a.enabled = False
        assert audio.active_listener_game_object_id == min(standby.id, other.id)
        a.enabled = True
        assert audio.active_listener_game_object_id == standby.id
        b.enabled = False
        assert audio.active_listener_game_object_id == first.id
        a.enabled = False
        assert audio.active_listener_game_object_id == other.id
        b.enabled = True
        assert audio.active_listener_game_object_id == other.id
        manager.unload_scene(additive)
        additive = None
        assert audio.active_listener_game_object_id == standby.id
        b.enabled = False
        assert audio.active_listener_game_object_id == 0
    finally:
        manager.set_active_scene(scene)
        if additive is not None:
            manager.unload_scene(additive)
