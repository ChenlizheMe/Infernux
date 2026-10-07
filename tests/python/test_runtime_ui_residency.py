"""Real native Scene residency drives the shared UI submission domain."""
from types import SimpleNamespace

import pytest

from infernux.lib import SceneManager
from infernux.ui import UICanvas, UIButton
from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
from test_runtime_screen_ui_submission import _Engine, _Renderer


def canvas_with_button(scene, name, order=0):
    owner = scene.create_game_object(name)
    canvas = UICanvas()
    owner.add_py_component(canvas)
    canvas.sort_order = order
    child = scene.create_game_object(name + "Button")
    child.set_parent(owner)
    button = UIButton()
    child.add_py_component(button)
    return owner, canvas, button


@pytest.fixture
def submission(monkeypatch):
    import infernux.engine.runtime_screen_ui as module

    renderer = _Renderer()
    engine = _Engine(renderer)
    submitted = []
    monkeypatch.setattr(module, "_get_tex_cache", lambda: SimpleNamespace(
        has_pending=False, generation=0, get_bound=lambda _engine: lambda _path: 0,
    ))
    monkeypatch.setattr(module, "_ui_dispatch", lambda element, *_a, **_kw: submitted.append(element))
    return engine, RuntimeScreenUISubmission(engine), submitted


def test_submission_includes_loaded_peers_and_excludes_preview(scene, submission):
    manager = SceneManager.instance()
    peer = manager.create_scene("Other room")
    preview = manager._create_preview_scene("Inspector preview")
    _, first, first_button = canvas_with_button(scene, "Room A")
    _, second, second_button = canvas_with_button(peer, "Room B", 10)
    canvas_with_button(preview, "Not runtime")
    engine, submit, submitted = submission
    try:
        manager.set_active_scene(scene)
        assert submit.submit()
        assert submitted == [first_button, second_button]
        submitted.clear()
        manager.set_active_scene(peer)
        assert not submit.submit(), "Active selection must not rebuild unchanged runtime UI"
        assert submitted == []
        manager.unload_scene(peer)
        assert submit.submit()
        assert submitted == [first_button]
    finally:
        manager._close_preview_scene(preview)


def test_submission_reorders_cached_canvases_after_sort_order_edit(scene, submission):
    _, first, first_button = canvas_with_button(scene, "First", 0)
    _, second, second_button = canvas_with_button(scene, "Second", 10)
    engine, submit, submitted = submission
    assert submit.submit()
    assert submitted == [first_button, second_button]
    submitted.clear()
    second.sort_order = -1
    assert submit.submit()
    assert submitted == [second_button, first_button]


def test_persistent_canvas_remains_visible_without_an_active_scene(scene, submission):
    manager = SceneManager.instance()
    owner, canvas, button = canvas_with_button(scene, "Persistent HUD")
    engine, submit, submitted = submission
    assert submit.submit()
    manager.play()
    try:
        manager.dont_destroy_on_load(owner)
        manager.prepare_active_scene_replacement()
        manager.set_active_scene(None)
        submitted.clear()
        assert submit.submit()
        assert submitted == [button]
    finally:
        manager.stop()


def test_world_ui_and_cached_membership_follow_moves_and_unload(scene, submission, monkeypatch):
    from infernux.ui import UIText
    from infernux.ui.ui_canvas_utils import runtime_ui_scenes
    from infernux.engine.runtime_screen_ui import collect_runtime_ui_input_surfaces, WorldUIElementTarget
    import infernux.ui.ui_canvas_utils as canvas_utils

    manager = SceneManager.instance()
    peer = manager.create_scene("Room with world UI")
    owner, canvas, button = canvas_with_button(peer, "Room HUD")
    world_owner = peer.create_game_object("World label")
    world_text = UIText()
    world_owner.add_py_component(world_text)
    engine, submit, submitted = submission
    assert submit.submit()
    assert submitted == [world_text, button]
    surfaces = collect_runtime_ui_input_surfaces(*runtime_ui_scenes(manager))
    assert isinstance(surfaces[0], WorldUIElementTarget)
    assert surfaces[0].element is world_text
    assert surfaces[1:] == (canvas,)

    # Warming must retain discovery, even if the active selection changes.
    original = canvas_utils._registered_canvas_entries
    scans = []
    monkeypatch.setattr(canvas_utils, "_registered_canvas_entries",
                        lambda *scenes: scans.append(scenes) or original(*scenes))
    for index in range(40):
        manager.set_active_scene(peer if index % 2 else scene)
        assert not submit.submit()
        assert collect_runtime_ui_input_surfaces(*runtime_ui_scenes(manager)) is surfaces
    assert scans == []

    assert canvas_utils.collect_canvases(peer) == [canvas]
    manager.move_game_object_to_scene(owner, scene)
    # Scene-scoped collection must retire the old membership immediately.
    assert canvas_utils.collect_canvases(peer) == []
    assert canvas_utils.collect_runtime_canvases_with_go(peer) == []
    assert canvas_utils.collect_runtime_canvases_with_go(scene) == [(owner, canvas)]
    manager.unload_scene(peer)
    submitted.clear()
    assert submit.submit()
    assert submitted == [button]
    assert collect_runtime_ui_input_surfaces(*runtime_ui_scenes(manager)) == (canvas,)


@pytest.mark.parametrize("host", ["player", "game_view"])
def test_host_input_hits_nonactive_canvas_and_observes_sort_order(scene, monkeypatch, host):
    from infernux.input import Input
    from infernux.ui.ui_event_data import PointerType
    from infernux.engine.runtime_event_queue import drain, clear

    manager = SceneManager.instance()
    peer = manager.create_scene("Peer UI")
    _, first, first_button = canvas_with_button(scene, "Back HUD", 0)
    _, second, second_button = canvas_with_button(peer, "Front HUD", 10)
    manager.set_active_scene(scene)
    x, y, width, height = second_button.get_rect(1920, 1080)
    mouse = [x + width / 2, y + height / 2, 0, 0, False, False, False]
    monkeypatch.setattr(Input, "get_game_mouse_frame_state", lambda _button: tuple(mouse))
    if host == "player":
        from infernux.engine.player_gui import PlayerGUI
        panel = PlayerGUI(object())
    else:
        from infernux.engine.ui.game_view_panel import GameViewPanel
        panel = GameViewPanel(engine=None)
        panel._display_scale = 1.0
        panel._is_playing = lambda: False
    key = (PointerType.Mouse, -1)
    clear()
    try:
        panel._process_ui_events(1920, 1080)
        assert panel._ui_event_processor._pointers[key].hover_target is second_button
        mouse[4:7] = [True, True, False]
        panel._process_ui_events(1920, 1080)
        assert panel._ui_event_processor._pointers[key].press_target is second_button
        mouse[4:7] = [False, False, True]
        panel._process_ui_events(1920, 1080)
        drain()
        second.sort_order = -1
        mouse[4:7] = [False, False, False]
        panel._process_ui_events(1920, 1080)
        assert panel._ui_event_processor._pointers[key].hover_target is first_button
    finally:
        panel._ui_event_processor.discard()
        clear()


def test_web_submission_and_pointer_collection_use_the_same_resident_worlds(scene, submission, monkeypatch):
    import ast
    from pathlib import Path
    from infernux.lib import ScreenUIList
    import infernux.engine.runtime_screen_ui as runtime_ui

    path = Path(__file__).resolve().parents[2] / "external/plugins/infernux_web/native/bootstrap.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_submit_screen_ui", "_process_screen_ui_events")]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    manager = SceneManager.instance()
    peer = manager.create_scene("Web peer")
    _, first, first_button = canvas_with_button(scene, "Web A")
    _, second, second_button = canvas_with_button(peer, "Web B", 10)
    engine, _, submitted = submission
    namespace.update(
        _player_scene_manager=manager, _screen_ui_renderer=engine.renderer,
        _screen_ui_texture_cache=SimpleNamespace(
            generation=0, has_pending=False, poll=lambda: None, get=lambda _path: 0,
        ),
        _screen_width=1920, _screen_height=1080,
        _screen_ui_snapshot_diagnostic=None, _WebScreenUIList=ScreenUIList,
    )
    namespace["_submit_screen_ui"]()
    assert submitted == [first_button, second_button]
    submitted.clear()
    manager.set_active_scene(peer)
    namespace["_submit_screen_ui"]()
    assert submitted == []
    manager.set_active_scene(scene)
    # Run the actual browser bridge until the pointer processor boundary.
    surfaces_seen = []
    namespace.update(
        _input_scene_token=None,
        _screen_ui_event_processor=SimpleNamespace(
            discard=lambda: None, reset=lambda: None,
            process_pointers=lambda surfaces, *_: surfaces_seen.append(surfaces),
        ),
        _mouse_event_dispatcher=SimpleNamespace(discard=lambda: None, process=lambda *_a, **_kw: None),
    )
    namespace["_process_screen_ui_events"](0.01)
    assert surfaces_seen[-1] == (first, second)
