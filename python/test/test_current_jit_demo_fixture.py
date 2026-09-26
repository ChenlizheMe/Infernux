"""Acceptance checks for the copyable current CPU JIT demo."""

import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

from Infernux import _jit_kernels
from Infernux.engine.script_candidate_policy import analyze_script_candidate


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tests/fixtures/jit_current_api/Assets/Scripts/CpuJitDemo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("infernux_current_jit_demo", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_desktop_demo_warmup_execution_and_statistics_are_observable():
    demo = _load_demo()
    result = demo.run_current_jit_demo(sample_count=2048)

    assert result["mode"] == "parallel"
    assert result["sample_count"] == 2048
    assert result["specialization_count"] >= 1
    np.testing.assert_allclose(result["first_position"], (0.1875, 0.109375, -0.0625))


def test_fixture_uses_only_the_current_public_authoring_surface():
    source = _SCRIPT.read_text(encoding="utf-8")

    assert "@inx.jit.compile" in source
    assert "inx.jit.warmup" in source
    assert "inx.jit.statistics" in source
    assert "njit" not in source
    assert "resident(" not in source
    assert "prepare_build" not in source
    assert "texture_path" not in source

    candidate = analyze_script_candidate(source.encode("utf-8"), filename=str(_SCRIPT))
    assert candidate.blocked == ()
    assert candidate.runtime_guard_required == ()


def test_play_mode_component_emits_the_acceptance_marker(monkeypatch):
    demo = _load_demo()
    messages = []
    monkeypatch.setattr(
        demo.inx.Debug,
        "log",
        lambda message, context=None: messages.append((message, context)),
    )
    component = demo.CpuJitDemo()
    component.sample_count = 128

    component.start()

    assert len(messages) == 1
    assert messages[0][0].startswith("INFERNUX_CURRENT_JIT_DEMO_READY ")
    assert "mode=parallel" in messages[0][0]
    assert "samples=128" in messages[0][0]
    assert messages[0][1] is component


def test_web_build_cooks_the_same_demo_as_ordinary_python(tmp_path):
    cooked_script = tmp_path / "CpuJitDemo.py"
    cooked_script.write_text(
        _jit_kernels.build_interpreted_cpu_source(
            _SCRIPT.read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    command = (
        "import importlib.util, json, pathlib, sys; "
        "path=pathlib.Path(sys.argv[1]); "
        "spec=importlib.util.spec_from_file_location('web_jit_demo', path); "
        "module=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "print(json.dumps(module.run_current_jit_demo(sample_count=16), sort_keys=True))"
    )
    environment = {
        **os.environ,
        "INFERNUX_WEB_RUNTIME": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(_ROOT / "python"), os.environ.get("PYTHONPATH")))
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-c", command, str(cooked_script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert '"mode": "ordinary-python"' in completed.stdout
    assert '"sample_count": 16' in completed.stdout
    assert '"specialization_count": 0' in completed.stdout
