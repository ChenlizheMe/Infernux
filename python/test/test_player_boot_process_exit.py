"""The compiled Player entry owns process exit even after bootstrap failure."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_generated_boot_exits_without_finalizing_live_runtime(tmp_path, debug, failure):
    from infernux.engine.game_builder import GameBuilder

    # Generate the production entry without invoking asset cooking/toolchains.
    builder = GameBuilder.__new__(GameBuilder)
    builder.output_dir = str(tmp_path / "build")
    boot = Path(builder._generate_boot_script())
    runtime = tmp_path / "Runtime"
    runtime.mkdir()
    data = tmp_path / "Data"
    data.mkdir()
    (data / "BuildManifest.json").write_text("{}", encoding="utf-8")
    finalized = tmp_path / "interpreter-finalized"
    returned = tmp_path / "entry-returned"
    worker = tmp_path / "run_entry.py"
    worker.write_text('''import atexit
import runpy
import sys
import types
from pathlib import Path

atexit.register(lambda: Path(sys.argv[2]).write_text("finalized", encoding="utf-8"))
native = types.ModuleType("_InfernuxBootstrap")
native._inxplayer_show_error = lambda *_args: (_ for _ in ()).throw(AssertionError("Managed startup must not show a dialog"))
sys.modules[native.__name__] = native
public = types.ModuleType("infernux")
public.__path__ = []
engine = types.ModuleType("infernux.engine")
engine.__path__ = []
platform = types.ModuleType("infernux.engine.platform_player_bootstrap")
platform.prepare_platform_player = lambda data_root, _cache: data_root
lib = types.ModuleType("infernux.lib")
lib.LogLevel = types.SimpleNamespace(Debug=0, Info=1)
def run_player(**_kwargs):
    print("PLAYER_BOOT_BUFFERED_OUTPUT")
    if sys.argv[4] == "1":
        raise RuntimeError("PLAYER_BOOT_FAILURE_EXPECTED")
engine.run_player = run_player
sys.modules.update({"infernux": public, "infernux.engine": engine, "infernux.lib": lib, "infernux.engine.platform_player_bootstrap": platform})
runpy.run_path(sys.argv[1], run_name="__main__")
Path(sys.argv[3]).write_text("returned", encoding="utf-8")
''', encoding="utf-8")
    state_root = tmp_path / "state"
    environment = dict(os.environ,
        LOCALAPPDATA=str(state_root), XDG_STATE_HOME=str(state_root),
        _INFERNUX_PLAYER_DATA_ROOT=str(data),
        _INFERNUX_PLAYER_RUNTIME_ROOT=str(runtime),
        _INFERNUX_PLAYER_DEBUG_BUILD="1" if debug else "0",
        _INFERNUX_PLAYER_CONTROL_FILE=str(tmp_path / "managed-control"),
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(worker), str(boot), str(finalized), str(returned), "1" if failure else "0"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == (1 if failure else 0), result.stdout + result.stderr
    assert not returned.exists(), "The product entry must not return to its embedded interpreter"
    assert not finalized.exists(), "Live native runtime objects must not enter CPython finalization"
    logs = state_root / "Infernux" / "Players" / boot.stem / "Logs"
    log = (logs / "player.log").read_text(encoding="utf-8")
    if failure:
        crash = (logs / "crash.log").read_text(encoding="utf-8")
        assert "RuntimeError: PLAYER_BOOT_FAILURE_EXPECTED" in crash
        assert "CRASH: " in log and "PLAYER_BOOT_FAILURE_EXPECTED" in log
    else:
        assert not (logs / "crash.log").exists()
        assert "boot: run_player returned" in log
    if debug:
        diagnostics = (logs / (boot.stem + "_debug.log")).read_text(encoding="utf-8")
        assert "PLAYER_BOOT_BUFFERED_OUTPUT" in diagnostics
        if failure:
            assert "PLAYER_BOOT_FAILURE_EXPECTED" in diagnostics
