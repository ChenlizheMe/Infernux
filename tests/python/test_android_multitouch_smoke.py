from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "tests" / "acceptance"
SCRIPT = ACCEPTANCE / "android_multitouch_smoke.py"


def _module():
    sys.path.insert(0, str(ACCEPTANCE))
    try:
        spec = importlib.util.spec_from_file_location("android_multitouch_smoke", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ACCEPTANCE))


def test_instrumentation_result_requires_success_and_extent():
    module = _module()

    assert module.validate_instrumentation_output(
        """INSTRUMENTATION_RESULT: INFERNUX_MULTITOUCH_INJECTION=passed
INSTRUMENTATION_RESULT: INFERNUX_IME_INJECTION=passed
INSTRUMENTATION_RESULT: INFERNUX_ORIENTATION_INJECTION=passed
INSTRUMENTATION_RESULT: INFERNUX_BACK_INJECTION=passed
INSTRUMENTATION_RESULT: committedText=输入测试中文🙂
INSTRUMENTATION_RESULT: height=1440
INSTRUMENTATION_RESULT: imeInset=612
INSTRUMENTATION_RESULT: landscapeRotation=1
INSTRUMENTATION_RESULT: landscapeSafeInsets=161,0,0,0
INSTRUMENTATION_RESULT: reverseLandscapeRotation=3
INSTRUMENTATION_RESULT: reverseLandscapeSafeInsets=0,0,161,0
INSTRUMENTATION_RESULT: width=3200
INSTRUMENTATION_CODE: -1
"""
    ) == (
        3200,
        1440,
        612,
        "输入测试中文🙂",
        1,
        3,
        (161, 0, 0, 0),
        (0, 0, 161, 0),
    )


def test_screen_state_samples_require_consistent_python_safe_area():
    module = _module()
    log = """
I/python.stdout: INFERNUX_PLATFORM_FIXTURE_SCREEN_STATE revision=4 size=3200x1440 framebuffer=3200x1440 safe=161,161,3039,1219 insets=161,161,0,60 pixel_ratio=3.750000
I/python.stdout: INFERNUX_PLATFORM_FIXTURE_SCREEN_STATE revision=8 size=3200x1440 framebuffer=3200x1440 safe=0,161,3039,1219 insets=0,161,161,60 pixel_ratio=3.750000
I/python.stdout: INFERNUX_PLATFORM_FIXTURE_SCREEN_STATE revision=9 size=3200x1440 framebuffer=3200x1440 safe=0,161,3039,1219 insets=0,161,161,60 pixel_ratio=3.750000
"""

    samples = module.screen_state_samples(log)

    assert len(samples) == 2
    assert samples[0]["safe_insets"] == (161, 161, 0, 60)
    assert samples[1]["safe_insets"] == (0, 161, 161, 60)


@pytest.mark.parametrize("cycles", [0, 1, 2, 10])
def test_resume_requires_exact_completed_cycle_count(cycles):
    output = f"INSTRUMENTATION_RESULT: INFERNUX_SURFACE_RESUME=passed\nINSTRUMENTATION_RESULT: resumeCycles={cycles}\n"
    assert _module().validate_resume_output(output, cycles) == cycles
