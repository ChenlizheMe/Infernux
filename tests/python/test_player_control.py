from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infernux.engine import player_control
from infernux.engine import player_gui as player_gui_module
from infernux.engine.player_control import PlayerControlChannel
from infernux.engine.player_gui import PlayerGUI
from infernux.input import Input, TouchPhase
from infernux.lib import InputManager


class _Native:
    def __init__(self) -> None:
        self.last_processed_synthetic_input_sequence = 0
        self.pending_synthetic_input_count = 0
        self.keys: list[tuple[int, bool, bool]] = []
        self.mouse_buttons: list[tuple[int, bool, float, float]] = []
        self.capture_status = "pending_gpu"
        self.capture_output_path = ""
        self.cancelled_capture_ids: list[int] = []

    def queue_synthetic_key_input(self, scancode: int, pressed: bool, repeat: bool) -> int:
        self.keys.append((scancode, pressed, repeat))
        self.pending_synthetic_input_count = 1
        return 11 + len(self.keys)

    def queue_synthetic_mouse_button_input(self, button: int, pressed: bool, x: float, y: float) -> int:
        self.mouse_buttons.append((button, pressed, x, y))
        self.pending_synthetic_input_count = 1
        return 21 + len(self.mouse_buttons)

    def request_capture(self, source: str, output_path: str) -> int:
        assert source == "game"
        self.capture_output_path = output_path
        return 31

    def query_capture(self, capture_id: int):
        return {
            "capture_id": capture_id,
            "status": self.capture_status,
            "output_path": self.capture_output_path,
            "error": "",
        }

    def cancel_capture(self, capture_id: int):
        self.cancelled_capture_ids.append(capture_id)
        return True


class _Engine:
    def __init__(self) -> None:
        self.native = _Native()

    def get_native_engine(self):
        return self.native


def _configure(tmp_path, monkeypatch, *, debug=True):
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    monkeypatch.setenv("_INFERNUX_PLAYER_DEBUG_BUILD", "1" if debug else "0")
    monkeypatch.setenv("_INFERNUX_PLAYER_CONTROL_FILE", str(request))
    monkeypatch.setenv("_INFERNUX_PLAYER_RESPONSE_FILE", str(response))
    monkeypatch.setenv("_INFERNUX_PLAYER_CONTROL_TOKEN", "control-token-123456789")
    monkeypatch.setenv("_INFERNUX_PLAYER_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return request, response, PlayerControlChannel.from_environment()


def _write_request(path, command_id: str, action: str, **values):
    path.write_text(
        json.dumps(
            {
                "command_id": command_id,
                "token": "control-token-123456789",
                "action": action,
                **values,
            }
        ),
        encoding="utf-8",
    )


def test_release_player_does_not_enable_validation_control(tmp_path, monkeypatch):
    _request, _response, channel = _configure(tmp_path, monkeypatch, debug=False)

    assert channel.enabled is False


def test_standalone_player_dispatches_runtime_ui_without_desktop_hover(monkeypatch):
    dispatched: list[tuple[int, int]] = []
    viewport_origins: list[tuple[float, float]] = []
    images = []

    class _PlayerEngine:
        game_presentation_ready = False
        readiness_queries = 0

        def get_native_engine(self):
            self.readiness_queries += 1
            return self

        def resize_game_render_target(self, width, height):
            assert (width, height) == (1280, 720)

        def get_game_texture_id(self):
            return 17

    class _Context:
        def image(self, *args):
            assert args == (17, 1280.0, 720.0, 0.0, 0.0, 1.0, 1.0)
            images.append(args)

    monkeypatch.setattr(
        player_gui_module,
        "capture_viewport_info",
        lambda _ctx: type("_Viewport", (), {"image_min_x": 8.0, "image_min_y": 12.0})(),
    )
    monkeypatch.setattr(
        Input,
        "set_game_viewport_origin",
        lambda x, y: viewport_origins.append((x, y)),
    )
    native_input = InputManager.instance()
    native_input.set_cursor_locked(True)
    monkeypatch.setattr(Input, "get_key_down", lambda _key: True)

    def reject_cursor_change(_locked):
        pytest.fail("Standalone cursor ownership belongs to game scripts")

    monkeypatch.setattr(Input, "set_cursor_locked", reject_cursor_change)

    player = PlayerGUI.__new__(PlayerGUI)
    player._engine = _PlayerEngine()
    player._last_w = 0
    player._last_h = 0
    player._render_scale = 1.0
    player._initial_image_pending = True
    mouse_frame = (32.0, 48.0, 0.0, 0.0, True, True, False)
    samples = []
    routed = []
    hit = object()
    monkeypatch.setattr(Input, "get_game_mouse_frame_state", lambda button: samples.append(button) or mouse_frame)

    def ui_events(width, height, *, mouse_frame):
        dispatched.append((width, height))
        routed.append(mouse_frame)
        return hit

    def mouse_events(width, height, *, scene_hit, mouse_frame):
        assert (width, height) == (1280, 720)
        assert scene_hit is hit
        routed.append(mouse_frame)

    player._process_ui_events = ui_events
    player._process_mouse_events = mouse_events

    # A standalone Player owns its whole window. The context intentionally has
    # no desktop hover API because native touchscreen input does not define it.
    try:
        player._render_game(_Context(), 1280.0, 720.0)
        assert not images and not dispatched and not samples
        player._engine.game_presentation_ready = True
        player._render_game(_Context(), 1280.0, 720.0)
        player._render_game(_Context(), 1280.0, 720.0)
        assert native_input.is_cursor_locked
    finally:
        native_input.set_cursor_locked(False)

    assert len(images) == 2
    assert player._engine.readiness_queries == 2
    assert viewport_origins == [(8.0, 12.0)] * 2
    assert dispatched == [(1280, 720)] * 2
    assert samples == [0, 0]
    assert len(routed) == 4 and all(frame is mouse_frame for frame in routed)


def test_player_primary_touch_drives_gameplay_pointer_in_top_left_pixels():
    secondary = SimpleNamespace(
        is_primary=False,
        phase=TouchPhase.BEGAN,
        began_this_frame=True,
        normalized_position=(0.1, 0.2),
        begin_normalized_position=(0.1, 0.2),
    )
    primary = SimpleNamespace(
        is_primary=True,
        phase=TouchPhase.MOVED,
        began_this_frame=False,
        normalized_position=(0.25, 0.75),
        begin_normalized_position=(0.2, 0.7),
    )

    assert PlayerGUI._primary_touch_scene_frame((secondary, primary), 1280, 720) == pytest.approx(
        (320.0, 180.0, True, False, False, False)
    )


def test_player_same_frame_tap_publishes_one_complete_gameplay_click():
    touch = SimpleNamespace(
        is_primary=True,
        phase=TouchPhase.ENDED,
        began_this_frame=True,
        normalized_position=(0.9, 0.1),
        begin_normalized_position=(0.4, 0.6),
    )

    assert PlayerGUI._primary_touch_scene_frame((touch,), 1000, 500) == pytest.approx(
        (400.0, 200.0, False, True, True, False)
    )


def test_scene_publication_discards_retired_player_input_state_once():
    discarded = []
    player = PlayerGUI.__new__(PlayerGUI)
    player._ui_event_processor = type("_UIProcessor", (), {"discard": lambda self: discarded.append("ui")})()
    player._mouse_event_dispatcher = type("_MouseProcessor", (), {"discard": lambda self: discarded.append("mouse")})()
    player._input_scene_token = (7, 2)
    player._has_shared_scene_query = True
    scene = type(
        "_Scene",
        (),
        {
            "world_id": 7,
            "temporal_discontinuity_revision": 3,
        },
    )()

    player._synchronize_input_scene(scene)
    player._synchronize_input_scene(scene)

    assert discarded == ["ui", "mouse"]
    assert player._input_scene_token == (7, 3)
    assert player._has_shared_scene_query is False


def _player_gui_for_play_gate(session):
    player = PlayerGUI.__new__(PlayerGUI)
    player._engine = type(
        "_PlayerEngine",
        (),
        {
            "get_player_runtime": lambda self: session,
        },
    )()
    player._splash = None
    player._activate_play = None
    player._play_started = False
    player._play_start_failed = False
    return player


def test_player_control_observation_is_token_authenticated(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    monkeypatch.setattr(
        player_control,
        "_observe_player",
        lambda _engine, names, probes=None, **kwargs: {
            "scene_name": "MainMenu",
            "names": names,
            "probes": probes or [],
            "discovery": kwargs,
        },
    )
    _write_request(request, "observe-1", "observe", object_names=["PlayerCar"])

    assert channel.poll(engine) is None
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["data"] == {
        "scene_name": "MainMenu",
        "names": ["PlayerCar"],
        "probes": [],
        "discovery": {
            "include_scene_objects": False,
            "discovery_component_types": [],
            "max_discovered_objects": 32,
        },
    }

    _write_request(request, "observe-2", "observe", object_names=[], token="wrong-token")
    assert channel.poll(engine) is None
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["error"] == "control token mismatch"


def test_player_control_waits_until_sdl_key_delivery(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    _write_request(request, "key-1", "key", scancode=26, pressed=True, repeat=False)

    assert channel.poll(engine) is None
    assert engine.native.keys == [(26, True, False)]
    assert not response.exists()

    engine.native.last_processed_synthetic_input_sequence = 12
    engine.native.pending_synthetic_input_count = 0
    input_manager = type(
        "_InputManager",
        (),
        {
            "get_key": lambda self, scancode: scancode == 26,
            "get_key_down": lambda self, scancode: scancode == 26,
            "get_key_up": lambda self, scancode: False,
        },
    )()
    monkeypatch.setattr("infernux.lib.InputManager.instance", lambda: input_manager)
    monkeypatch.setattr(Input, "is_game_focused", lambda: True)
    assert channel.poll(engine) is None
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["data"] == {
        "sequence": 12,
        "delivered": True,
        "scancode": 26,
        "game_focused": True,
        "held": True,
        "down": True,
        "up": False,
        "pending_input_count": 0,
    }


def test_player_control_capture_owns_frame_bounded_input_and_pauses(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    now = [5.0]
    scene_name = ["MainMenu"]
    frame = [0]
    paused = []
    monkeypatch.setattr(player_control.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(player_control, "_active_scene_name", lambda: scene_name[0])
    monkeypatch.setattr(player_control, "_time_frame_count", lambda: frame[0])
    monkeypatch.setattr(player_control, "_pause_player_scene", lambda: paused.append(True) or True)
    monkeypatch.setattr(
        player_control,
        "_observe_motion_state",
        lambda _engine, _names, _probes: {
            "scene_name": scene_name[0],
            "renderer_frame": 100 + frame[0],
            "objects": {"PlayerCar": {"position": [0.0, 0.5, float(frame[0])]}},
        },
    )
    _write_request(
        request,
        "frame-capture-arm",
        "motion_capture_arm",
        object_names=["PlayerCar"],
        seconds=1.0,
        sample_interval=0.02,
        trigger_scene_name="racetrack",
        hold_scancodes=[26, 4],
        hold_mouse_buttons=[1],
        mouse_x=640.0,
        mouse_y=360.0,
        hold_frame_count=2,
        wait_frame_count=2,
        pause_on_complete=True,
    )

    assert channel.poll(engine) is None
    capture_id = json.loads(response.read_text(encoding="utf-8"))["data"]["capture_id"]
    response.unlink()

    scene_name[0] = "racetrack"
    frame[0] = 20
    now[0] = 6.0
    assert channel.poll(engine) is None
    assert engine.native.keys == [(26, True, False), (4, True, False)]
    assert engine.native.mouse_buttons == [(1, True, 640.0, 360.0)]

    frame[0] = 21
    now[0] = 6.03
    assert channel.poll(engine) is None
    frame[0] = 22
    now[0] = 6.06
    assert channel.poll(engine) is None
    assert engine.native.keys == [(26, True, False), (4, True, False), (4, False, False), (26, False, False)]
    assert engine.native.mouse_buttons == [
        (1, True, 640.0, 360.0),
        (1, False, 640.0, 360.0),
    ]

    engine.native.last_processed_synthetic_input_sequence = 23
    frame[0] = 24
    now[0] = 6.12
    assert channel.poll(engine) is None
    assert paused == [True]

    _write_request(request, "frame-capture-status", "motion_capture_status", capture_id=capture_id)
    assert channel.poll(engine) is None
    completed = json.loads(response.read_text(encoding="utf-8"))["data"]
    assert completed["status"] == "completed"
    assert completed["frame_count"] == 4
    assert completed["hold_frame_count"] == 2
    assert completed["wait_frame_count"] == 2
    assert completed["elapsed_frame_count"] == 4
    assert completed["input_released_after_hold_frame"] == 2
    assert completed["paused_on_complete"] is True
    assert [item["scancode"] for item in completed["input_releases"] if "scancode" in item] == [4, 26]
    assert completed["hold_mouse_buttons"] == [1]
    assert completed["input_presses"][2]["button"] == 1
    assert completed["input_releases"][0]["button"] == 1


def test_player_control_capture_stops_on_sampled_public_condition(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    now = [5.0]
    scene_name = ["MainMenu"]
    frame = [0]
    paused = []
    samples = iter(
        (
            {
                "scene_name": "racetrack",
                "renderer_frame": 20,
                "objects": {
                    "PlayerCar": {
                        "position": [0.0, 0.5, 0.0],
                        "component_fields": {"RaceHUDController[0]": {"player_progress": 0.1}},
                    }
                },
            },
            {
                "scene_name": "racetrack",
                "renderer_frame": 21,
                "objects": {
                    "PlayerCar": {
                        "position": [0.0, 0.5, 2.0],
                        "component_fields": {"RaceHUDController[0]": {"player_progress": 0.8}},
                    }
                },
            },
        )
    )
    monkeypatch.setattr(player_control.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(player_control, "_active_scene_name", lambda: scene_name[0])
    monkeypatch.setattr(player_control, "_time_frame_count", lambda: frame[0])
    monkeypatch.setattr(player_control, "_pause_player_scene", lambda: paused.append(True) or True)
    monkeypatch.setattr(player_control, "_observe_motion_state", lambda *_args: next(samples))
    _write_request(
        request,
        "condition-capture-arm",
        "motion_capture_arm",
        object_names=["PlayerCar"],
        seconds=2.0,
        sample_interval=0.02,
        trigger_scene_name="racetrack",
        hold_scancodes=[26],
        hold_frame_count=120,
        wait_frame_count=2,
        wait_seconds=0.1,
        pause_on_complete=False,
        pause_on_condition=True,
        component_probes=[
            {
                "object_name": "PlayerCar",
                "component_type": "RaceHUDController",
                "fields": ["player_progress"],
            }
        ],
        stop_assertions=[
            {
                "kind": "component_field",
                "object_name": "PlayerCar",
                "component_type": "RaceHUDController",
                "field": "player_progress",
                "operator": "greater_or_equal",
                "value": 0.75,
            }
        ],
    )

    assert channel.poll(engine) is None
    capture_id = json.loads(response.read_text(encoding="utf-8"))["data"]["capture_id"]
    response.unlink()
    scene_name[0] = "racetrack"
    frame[0] = 20
    now[0] = 6.0
    assert channel.poll(engine) is None
    assert engine.native.keys == [(26, True, False)]

    frame[0] = 21
    now[0] = 6.03
    assert channel.poll(engine) is None
    assert engine.native.keys == [(26, True, False), (26, False, False)]

    engine.native.last_processed_synthetic_input_sequence = 13
    frame[0] = 23
    now[0] = 6.06
    assert channel.poll(engine) is None
    assert paused == []

    now[0] = 6.14
    assert channel.poll(engine) is None
    _write_request(request, "condition-capture-status", "motion_capture_status", capture_id=capture_id)
    assert channel.poll(engine) is None
    completed = json.loads(response.read_text(encoding="utf-8"))["data"]
    assert completed["status"] == "condition_met"
    assert completed["stop_condition"]["passed"] is True
    assert completed["condition_met_at_frame"] == 1
    assert completed["condition_settle_until_frame"] == 3
    assert completed["condition_settle_until_time"] == 6.13
    assert completed["input_released_after_hold_frame"] == 1
    assert completed["paused_on_complete"] is True
    assert paused == [True]


def test_player_control_capture_timeout_releases_command_channel(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    clock = [10.0]
    monkeypatch.setattr(player_control.time, "monotonic", lambda: clock[0])
    _write_request(
        request,
        "render-capture-timeout",
        "capture",
        file_name="timeout.png",
        timeout_seconds=0.5,
    )

    assert channel.poll(engine) is None
    clock[0] = 10.6
    assert channel.poll(engine) is None
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["data"]["status"] == "cancelled"
    assert engine.native.cancelled_capture_ids == [31]


def test_player_control_shutdown_returns_main_thread_action(tmp_path, monkeypatch):
    request, response, channel = _configure(tmp_path, monkeypatch)
    engine = _Engine()
    _write_request(request, "shutdown-1", "shutdown")

    assert channel.poll(engine) == "shutdown"
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["data"]["close_requested"] is True
