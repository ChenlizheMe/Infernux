from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "acceptance" / "web_mobile_input_smoke.cjs"


def _source() -> str:
    return HARNESS.read_text(encoding="utf-8")


def test_web_browser_harness_has_a_real_firefox_launch_path() -> None:
    source = _source()

    assert 'const { chromium, firefox } = require("playwright")' in source
    assert 'process.env.INFERNUX_WEB_BROWSER_ENGINE?.trim() || "chromium"' in source
    assert 'if (browserEngine === "firefox")' in source
    assert "browser = await firefox.launch" in source
    assert "browser = await chromium.launch" in source
    assert "browserEngine," in source
    assert "browserVersion: browser.version()" in source


def test_firefox_never_receives_chromium_only_launch_options() -> None:
    source = _source()
    firefox_branch = source[
        source.index('if (browserEngine === "firefox")') :
        source.index("page = await browser.newPage")
    ]

    assert "args: browserArgs" not in firefox_branch.split("} else {", 1)[0]
    assert "channel:" not in firefox_branch.split("} else {", 1)[0]
    assert "args: browserArgs" in firefox_branch.split("} else {", 1)[1]


def test_firefox_rejects_cdp_only_acceptance_modes_before_launch() -> None:
    environment = os.environ.copy()
    environment["INFERNUX_WEB_BROWSER_ENGINE"] = "firefox"
    result = subprocess.run(
        [
            "node",
            str(HARNESS),
            "http://127.0.0.1:1/infernux-player.html",
            "--cdp-endpoint",
            "http://127.0.0.1:9222",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--cdp-endpoint is only supported by the Chromium engine" in result.stderr


def test_unknown_browser_engine_fails_before_launch() -> None:
    environment = os.environ.copy()
    environment["INFERNUX_WEB_BROWSER_ENGINE"] = "webkit"
    result = subprocess.run(
        ["node", str(HARNESS), "http://127.0.0.1:1/infernux-player.html"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "INFERNUX_WEB_BROWSER_ENGINE must be 'chromium' or 'firefox'" in result.stderr
