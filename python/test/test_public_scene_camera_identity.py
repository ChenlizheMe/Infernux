"""Scene camera discovery must share the component lookup's public proxies."""

import pytest

import infernux as inx


@pytest.mark.parametrize("query", ["main_camera", "effective_game_camera", "active_game_cameras"])
def test_scene_camera_queries_return_canonical_public_proxy(scene, query):
    owner = scene.create_game_object("PublicSceneCamera")
    camera = owner.add_component(inx.Camera)
    scene.main_camera = camera
    result = getattr(scene, query)
    received = result[0] if query == "active_game_cameras" else result
    assert received is camera is owner.get_component(inx.Camera)
    assert isinstance(received, inx.Camera)
    assert received.game_object is owner
    received.field_of_view = 61.0
    assert camera.field_of_view == pytest.approx(61.0)


def test_scene_camera_queries_preserve_preference_and_active_depth_order(scene):
    preferred = scene.create_game_object("PreferredPublicCamera").add_component(inx.Camera)
    lower = scene.create_game_object("LowerDepthPublicCamera").add_component(inx.Camera)
    preferred.depth = 10
    lower.depth = -10
    scene.main_camera = preferred
    assert scene.main_camera is preferred
    assert scene.effective_game_camera is preferred
    assert scene.active_game_cameras == [lower, preferred]
    assert scene.active_game_cameras[0] is lower
    assert scene.active_game_cameras[1] is preferred
    preferred.enabled = False
    assert scene.main_camera is preferred
    assert scene.effective_game_camera is lower
    assert scene.active_game_cameras == [lower]
    preferred.enabled = True
    assert scene.effective_game_camera is preferred
    scene.main_camera = None
    assert scene.main_camera is None
    assert scene.effective_game_camera is lower


@pytest.mark.parametrize("destroy_owner", [False, True])
def test_scene_camera_queries_retire_removed_proxy_and_keep_survivor(scene, destroy_owner):
    owner = scene.create_game_object("RemovedPublicCamera")
    removed = owner.add_component(inx.Camera)
    survivor = scene.create_game_object("SurvivingPublicCamera").add_component(inx.Camera)
    scene.main_camera = removed
    assert scene.main_camera is removed
    if destroy_owner:
        scene.destroy_game_object(owner)
        scene.process_pending_destroys()
    else:
        assert owner.remove_component(removed)
    assert scene.main_camera is None
    assert scene.effective_game_camera is survivor
    assert scene.active_game_cameras == [survivor]
    assert scene.active_game_cameras[0] is survivor
    assert not removed.is_valid


def test_empty_scene_camera_queries_have_no_proxy(scene):
    assert scene.main_camera is None
    assert scene.effective_game_camera is None
    assert scene.active_game_cameras == []
