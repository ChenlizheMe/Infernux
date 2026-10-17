from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "acceptance" / "android_player_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("android_player_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_player_apk(apk: Path, entry_point: str) -> None:
    manifest = {
        "product": {
            "entry_points": [entry_point],
            "single_entry_point": True,
        }
    }
    with zipfile.ZipFile(apk, "w") as archive:
        archive.writestr(
            "assets/player/Fixture_Data/Player.inxmanifest",
            json.dumps(manifest),
        )
        archive.writestr("lib/arm64-v8a/libmain.so", b"arm")


def test_player_identity_is_read_from_current_apk_manifest(tmp_path: Path):
    module = _module()
    apk = tmp_path / "player.apk"
    component = "com.infernux.infernux041labv2/com.infernux.bootstrap.InfernuxActivity"
    _write_player_apk(apk, component)

    assert module.resolve_player_identity(apk, package=None, activity=None) == (
        "com.infernux.infernux041labv2",
        component,
    )


def test_unlock_retries_the_complete_sequence_when_systemui_drops_the_first_swipe(
    monkeypatch,
):
    module = _module()
    commands = []
    policy_calls = 0

    class FakeAdb:
        def run(self, *arguments, **options):
            nonlocal policy_calls
            assert options == {"check": False}
            commands.append(arguments)
            if arguments == ("shell", "locksettings", "get-disabled"):
                return "false"
            if arguments == ("shell", "dumpsys", "window", "policy"):
                policy_calls += 1
                return f"KeyguardServiceDelegate\nshowing={'true' if policy_calls <= 5 else 'false'}"
            if arguments == ("shell", "wm", "size"):
                return "Physical size: 1080x1920"
            return ""

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module.unlock_device(FakeAdb())

    assert commands.count(("shell", "input", "keyevent", "KEYCODE_SLEEP")) == 2
    assert commands.count(("shell", "input", "keyevent", "KEYCODE_WAKEUP")) == 2
    assert commands.count(("shell", "input", "keyevent", "KEYCODE_MENU")) == 2
    assert commands.count(("shell", "wm", "dismiss-keyguard")) == 4


def test_required_runtime_logs_wait_for_every_marker(monkeypatch):
    module = _module()
    logs = iter(("LINE_READY", "LINE_READY\nPARTICLE_READY\nANIMATION_READY"))
    pids = iter(("321", "322"))

    class FakeAdb:
        def run(self, *arguments, **options):
            assert options == {"check": False}
            if arguments == ("shell", "pidof", "com.infernux.fixture"):
                return next(pids)
            assert arguments == ("logcat", "-d", "-v", "brief")
            return next(logs)

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    log = module._wait_for_required_logs(
        FakeAdb(),
        "com.infernux.fixture",
        "321",
        ("LINE_READY", "PARTICLE_READY", "ANIMATION_READY"),
        5.0,
    )

    assert "ANIMATION_READY" in log


def test_surface_destroy_wait_is_observed_before_resume(monkeypatch):
    module = _module()
    logs = iter(
        (
            "INFERNUX_ANDROID_SURFACE_DESTROY_WAIT_COMPLETE",
            "INFERNUX_ANDROID_SURFACE_DESTROY_WAIT_COMPLETE\nINFERNUX_ANDROID_SURFACE_DESTROY_WAIT_COMPLETE",
        )
    )

    class FakeAdb:
        def run(self, *arguments, **options):
            assert arguments == ("logcat", "-d", "-v", "brief")
            assert options == {"check": False}
            return next(logs)

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    log = module._wait_for_log_count(
        FakeAdb(),
        "INFERNUX_ANDROID_SURFACE_DESTROY_WAIT_COMPLETE",
        2,
        5.0,
    )

    assert log.count("INFERNUX_ANDROID_SURFACE_DESTROY_WAIT_COMPLETE") == 2


def test_surface_creation_is_observed_before_next_suspend(monkeypatch):
    module = _module()
    logs = iter(
        (
            "INFERNUX_VULKAN_SURFACE extent=2400x1080",
            "INFERNUX_VULKAN_SURFACE extent=2400x1080\nINFERNUX_VULKAN_SURFACE extent=2400x1080",
        )
    )

    class FakeAdb:
        def run(self, *arguments, **options):
            assert arguments == ("logcat", "-d", "-v", "brief")
            assert options == {"check": False}
            return next(logs)

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    log = module._wait_for_surface_creation(FakeAdb(), 2, 5.0)

    assert len(module.vulkan_surface_extents(log)) == 2


def test_touch_gesture_is_injected_as_distinct_frame_phases(monkeypatch):
    module = _module()
    commands = []
    delays = []

    class FakeAdb:
        def run(self, *arguments):
            commands.append(arguments)

    monkeypatch.setattr(module.time, "sleep", delays.append)

    module.inject_touch_gesture(FakeAdb(), 3200, 1440)

    assert commands == [
        (
            "shell",
            "input",
            "touchscreen",
            "motionevent",
            "DOWN",
            "352",
            "1180",
        ),
        (
            "shell",
            "input",
            "touchscreen",
            "motionevent",
            "MOVE",
            "576",
            "1008",
        ),
        (
            "shell",
            "input",
            "touchscreen",
            "motionevent",
            "UP",
            "704",
            "950",
        ),
    ]
    assert delays == [0.25, 0.25]


def test_device_is_unlocked_before_usb_install_approval_is_requested():
    source = SCRIPT.read_text(encoding="utf-8")
    run_smoke = source.split("def run_smoke(", 1)[1].split("def _parser()", 1)[0]

    assert run_smoke.index("unlock_device(adb)") < run_smoke.index("install_apk(")


def test_apk_install_stages_the_complete_package_before_package_manager():
    source = SCRIPT.read_text(encoding="utf-8")
    install_apk = source.split("def install_apk(", 1)[1].split("def parse_devices(", 1)[0]

    assert 'arguments = ["install", "--no-streaming", "-r", "-t", str(apk)]' in install_apk


def test_signature_mismatch_never_uninstalls_the_existing_game(tmp_path, monkeypatch):
    module = _module()
    installs = []

    class FakeAdb:
        def __init__(self, *args):
            pass

        def run(self, *args, **kwargs):
            if args == ("devices", "-l"):
                return "phone device model:Test\n"
            if args == ("shell", "getprop", "ro.product.cpu.abi"):
                return "arm64-v8a"
            pytest.fail(f"Unexpected device mutation after failed update: {args}")

    def reject_update(*args, **kwargs):
        installs.append((args, kwargs))
        raise RuntimeError("INSTALL_FAILED_UPDATE_INCOMPATIBLE")

    monkeypatch.setattr(module, "Adb", FakeAdb)
    monkeypatch.setattr(module, "unlock_device", lambda adb: None)
    monkeypatch.setattr(module, "apk_abis", lambda apk: {"arm64-v8a"})
    monkeypatch.setattr(module, "install_apk", reject_update)
    arguments = module._parser().parse_args([str(tmp_path / "player.apk"), "--package", "com.infernux.fixture"])

    with pytest.raises(RuntimeError, match="INSTALL_FAILED_UPDATE_INCOMPATIBLE"):
        module.run_smoke(arguments)
    assert len(installs) == 1
    assert installs[0][1] == {"approve_oem_prompt": True}
